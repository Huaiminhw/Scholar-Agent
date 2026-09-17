"""Evaluate frozen V2 queries against the local PaSa BM25 corpus.

Local A/B/C ablation protocol::

    cached Original + Generated Queries
        -> PaSaCorpusBM25 (SQLite FTS5)
        -> top 100 for every query
        -> deduplicate by normalized arXiv ID
        -> weighted reciprocal-rank fusion per ablation arm
        -> exact-arXiv-ID Recall@20/50/100 and MRR

The three arms reuse exactly the same frozen retrieval results:

* A_original: Original query only.
* B_deterministic: Original + Exact/Core/Relation queries.
* C_full: Original + every cached generated query, including Alias/Expanded.

The historical unranked union metrics remain in the report as a diagnostic
upper bound; ranked Top-K metrics are the primary competition-style results.

This script never calls an LLM and never accesses OpenAlex, Semantic Scholar,
arXiv, Google Search, or another network source.  Query Understanding and Query
Generation must have been frozen beforehand by ``build_pasa_v2_query_cache.py``.

PowerShell example::

    $env:PASA_FROZEN_V2_START="0"
    $env:PASA_FROZEN_V2_LIMIT="50"
    python scripts/eval_pasa_frozen_v2_queries.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent_v2.retrieval.sources.pasa_corpus import (  # noqa: E402
    PaSaBM25Result,
    PaSaCorpusBM25,
    normalize_arxiv_id,
)


RETRIEVAL_LIMIT_PER_QUERY = 100
TOP_K_VALUES = (20, 50, 100)
RRF_K = 60
ARM_ALLOWED_KINDS: dict[str, set[str] | None] = {
    "A_original": set(),
    "B_deterministic": {"exact", "core", "relation"},
    "C_full": None,
}
QUERY_KIND_WEIGHTS = {
    "original": 1.00,
    "exact": 1.10,
    "core": 1.00,
    "relation": 0.90,
    "alias": 0.80,
    "expanded": 0.70,
}
DEFAULT_DATASET = PROJECT_ROOT / "data" / "RealScholarQuery" / "test.jsonl"
DEFAULT_CACHE = (
    PROJECT_ROOT
    / "results"
    / "pasa_v2_query_cache"
    / "realscholarquery_v2_queries.jsonl"
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "pasa_frozen_v2_queries"


@dataclass(frozen=True)
class GoldPaper:
    title: str
    arxiv_id: str | None


@dataclass(frozen=True)
class FrozenSearchQuery:
    query_id: str
    text: str
    kind: str
    source: str


@dataclass
class CandidateOrigin:
    route_index: int
    query_id: str
    query_kind: str
    query_source: str
    query_text: str
    rank: int
    bm25_score: float


@dataclass
class DeduplicatedCandidate:
    arxiv_id: str
    title: str
    year: int | None
    corpus_key: str
    first_seen_order: int
    origins: list[CandidateOrigin] = field(default_factory=list)


@dataclass(frozen=True)
class RankedCandidate:
    arxiv_id: str
    title: str
    score: float
    best_rank: int
    matched_routes: int


@dataclass
class AblationArmResult:
    arm: str
    query_count: int
    raw_hits: int
    unique_candidates: int
    duplicate_hits_removed: int
    duplicate_ratio: float
    candidate_inflation_vs_original: float
    recall_at: dict[str, float]
    corpus_recall_at: dict[str, float]
    matched_gold_count_at: dict[str, int]
    matched_corpus_gold_count_at: dict[str, int]
    marginal_gold_hits_at: dict[str, int]
    mrr: float
    union_recall: float
    union_corpus_recall: float
    top_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class QueryEvalResult:
    dataset_index: int
    qid: str
    question: str
    cache_created_at: str | None
    cache_protocol_sha256: str
    gold_count: int
    corpus_gold_count: int
    corpus_missing_count: int
    corpus_coverage: float
    search_query_count: int
    generated_query_count: int
    raw_hits: int
    missing_arxiv_id_hits: int
    deduplicated_candidates: int
    duplicate_hits_removed: int
    original_matched_gold_count: int
    union_matched_gold_count: int
    original_matched_corpus_gold_count: int
    union_matched_corpus_gold_count: int
    original_recall_at_100: float
    union_recall: float
    original_corpus_recall_at_100: float
    union_corpus_recall: float
    marginal_gold_hits: int
    marginal_corpus_gold_hits: int
    retrieval_latency_ms: float
    route_results: list[dict[str, Any]] = field(default_factory=list)
    matched_gold_at_union: list[dict[str, Any]] = field(default_factory=list)
    missed_gold_at_union: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    ablation: dict[str, AblationArmResult] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cached V2 queries with only the frozen local PaSa "
            "SQLite FTS5 BM25 corpus."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.getenv("PASA_REALSCHOLARQUERY_PATH", str(DEFAULT_DATASET))),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(os.getenv("PASA_V2_QUERY_CACHE_PATH", str(DEFAULT_CACHE))),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            os.getenv(
                "PASA_BM25_INDEX_PATH",
                str(PROJECT_ROOT / "models" / "pasa_bm25" / "pasa_corpus.sqlite3"),
            )
        ),
    )
    parser.add_argument(
        "--paper-zip",
        type=Path,
        default=Path(
            os.getenv(
                "PASA_PAPER_ZIP_PATH",
                str(PROJECT_ROOT / "data" / "paper_database" / "cs_paper_2nd.zip"),
            )
        ),
    )
    parser.add_argument(
        "--id-map",
        type=Path,
        default=Path(
            os.getenv(
                "PASA_ID2PAPER_PATH",
                str(PROJECT_ROOT / "data" / "paper_database" / "id2paper.json"),
            )
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=int(os.getenv("PASA_FROZEN_V2_START", "0")),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("PASA_FROZEN_V2_LIMIT", "50")),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.getenv("PASA_FROZEN_V2_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional exact JSON output path.",
    )
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def load_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid dataset JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record.get("question"), str) or not record["question"].strip():
                raise ValueError(f"Dataset line {line_number} has no non-empty question")
            if not isinstance(record.get("answer"), list):
                raise ValueError(f"Dataset line {line_number} has no answer list")
            if not isinstance(record.get("answer_arxiv_id"), list):
                raise ValueError(f"Dataset line {line_number} has no answer_arxiv_id list")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def load_cache(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid cache JSONL at line {line_number}: {exc}"
                ) from exc
            index = record.get("dataset_index")
            if not isinstance(index, int):
                raise ValueError(f"Cache line {line_number} has no integer dataset_index")
            if index in records:
                raise ValueError(f"Duplicate dataset_index={index} in cache")
            records[index] = record
    if not records:
        raise ValueError(f"Query cache is empty: {path}")
    return records


def gold_papers(record: Mapping[str, Any]) -> list[GoldPaper]:
    titles = record.get("answer") or []
    arxiv_ids = record.get("answer_arxiv_id") or []
    if len(titles) != len(arxiv_ids):
        raise ValueError(
            f"{record.get('qid', 'unknown')}: answer and answer_arxiv_id lengths differ"
        )
    return [
        GoldPaper(
            title=str(title).strip(),
            arxiv_id=normalize_arxiv_id(arxiv_id),
        )
        for title, arxiv_id in zip(titles, arxiv_ids)
    ]


def cached_search_queries(
    *,
    source_record: Mapping[str, Any],
    cache_record: Mapping[str, Any],
) -> list[FrozenSearchQuery]:
    question = str(source_record["question"]).strip()
    queries = [
        FrozenSearchQuery(
            query_id="QO",
            text=question,
            kind="original",
            source="original",
        )
    ]
    raw_generated = cache_record.get("search_queries")
    if not isinstance(raw_generated, list):
        raise ValueError(
            f"Cache dataset_index={cache_record.get('dataset_index')} has no search_queries list"
        )
    for position, raw in enumerate(raw_generated, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError("Cached search query must be a JSON object")
        text = str(raw.get("text", "")).strip()
        if not text:
            raise ValueError(f"Cached generated query #{position} has empty text")
        queries.append(
            FrozenSearchQuery(
                query_id=str(raw.get("query_id", f"QG{position}")),
                text=text,
                kind=str(raw.get("kind", "generated")),
                source="generated",
            )
        )
    return queries


def validate_selected_cache(
    *,
    dataset_records: Sequence[Mapping[str, Any]],
    cache_records: Mapping[int, Mapping[str, Any]],
    selected_indices: Sequence[int],
    dataset_sha256: str,
) -> str:
    protocol_hashes: set[str] = set()
    problems: list[str] = []
    for index in selected_indices:
        cached = cache_records.get(index)
        if cached is None:
            problems.append(f"dataset_index={index}: missing cache record")
            continue
        if cached.get("status") != "success":
            problems.append(
                f"dataset_index={index}: cache status={cached.get('status')!r}"
            )
        if cached.get("dataset_sha256") != dataset_sha256:
            problems.append(f"dataset_index={index}: dataset SHA256 mismatch")
        question = str(dataset_records[index]["question"]).strip()
        if cached.get("question") != question:
            problems.append(f"dataset_index={index}: question mismatch")
        expected_question_hash = _sha256_bytes(question.encode("utf-8"))
        if cached.get("question_sha256") != expected_question_hash:
            problems.append(f"dataset_index={index}: question SHA256 mismatch")
        protocol_hash = cached.get("protocol_sha256")
        if not isinstance(protocol_hash, str) or not protocol_hash:
            problems.append(f"dataset_index={index}: missing protocol SHA256")
        else:
            protocol_hashes.add(protocol_hash)
        try:
            cached_search_queries(
                source_record=dataset_records[index],
                cache_record=cached,
            )
        except ValueError as exc:
            problems.append(f"dataset_index={index}: {exc}")

    if len(protocol_hashes) > 1:
        problems.append("selected cache records contain mixed LLM/config protocols")
    if problems:
        raise RuntimeError(
            "Frozen V2 query cache is incomplete or inconsistent:\n  - "
            + "\n  - ".join(problems)
            + "\nRe-run scripts/build_pasa_v2_query_cache.py before evaluation."
        )
    return next(iter(protocol_hashes))


def retrieve_and_deduplicate(
    *,
    corpus: PaSaCorpusBM25,
    queries: Sequence[FrozenSearchQuery],
) -> tuple[
    list[DeduplicatedCandidate],
    list[dict[str, Any]],
    int,
    int,
    float,
]:
    candidates_by_id: dict[str, DeduplicatedCandidate] = {}
    ordered_candidates: list[DeduplicatedCandidate] = []
    route_results: list[dict[str, Any]] = []
    raw_hits = 0
    missing_arxiv_id_hits = 0
    started = time.perf_counter()

    for route_index, query in enumerate(queries):
        route_started = time.perf_counter()
        hits = corpus.search(
            query.text,
            limit=RETRIEVAL_LIMIT_PER_QUERY,
            operator="or",
        )
        route_latency_ms = (time.perf_counter() - route_started) * 1000.0
        raw_hits += len(hits)
        route_ids: list[str] = []
        route_missing_ids = 0

        for hit in hits:
            arxiv_id = normalize_arxiv_id(hit.arxiv_id)
            if not arxiv_id:
                missing_arxiv_id_hits += 1
                route_missing_ids += 1
                continue
            route_ids.append(arxiv_id)
            origin = CandidateOrigin(
                route_index=route_index,
                query_id=query.query_id,
                query_kind=query.kind,
                query_source=query.source,
                query_text=query.text,
                rank=hit.rank,
                bm25_score=hit.bm25_score,
            )
            candidate = candidates_by_id.get(arxiv_id)
            if candidate is None:
                candidate = DeduplicatedCandidate(
                    arxiv_id=arxiv_id,
                    title=hit.title,
                    year=hit.year,
                    corpus_key=hit.corpus_key,
                    first_seen_order=len(ordered_candidates) + 1,
                    origins=[origin],
                )
                candidates_by_id[arxiv_id] = candidate
                ordered_candidates.append(candidate)
            else:
                candidate.origins.append(origin)

        route_results.append(
            {
                "route_index": route_index,
                "query_id": query.query_id,
                "query_kind": query.kind,
                "query_source": query.source,
                "query_text": query.text,
                "retrieved_count": len(hits),
                "missing_arxiv_id_hits": route_missing_ids,
                "unique_arxiv_ids_in_route": len(set(route_ids)),
                "latency_ms": route_latency_ms,
            }
        )

    total_latency_ms = (time.perf_counter() - started) * 1000.0
    return (
        ordered_candidates,
        route_results,
        raw_hits,
        missing_arxiv_id_hits,
        total_latency_ms,
    )


def _matched_gold_indices(
    candidate_ids: Iterable[str],
    gold: Sequence[GoldPaper],
) -> set[int]:
    ids = {normalize_arxiv_id(value) for value in candidate_ids}
    ids.discard(None)
    return {
        index
        for index, paper in enumerate(gold)
        if paper.arxiv_id is not None and paper.arxiv_id in ids
    }


def _origin_allowed(origin: CandidateOrigin, arm: str) -> bool:
    if origin.query_source == "original":
        return True
    allowed = ARM_ALLOWED_KINDS[arm]
    return allowed is None or origin.query_kind in allowed


def rank_candidates_for_arm(
    candidates: Sequence[DeduplicatedCandidate],
    *,
    arm: str,
) -> list[RankedCandidate]:
    """Weighted RRF over only the routes enabled for one ablation arm."""
    if arm not in ARM_ALLOWED_KINDS:
        raise ValueError(f"Unknown ablation arm: {arm}")

    ranked: list[tuple[RankedCandidate, int]] = []
    for candidate in candidates:
        origins = [
            origin for origin in candidate.origins if _origin_allowed(origin, arm)
        ]
        if not origins:
            continue
        score = sum(
            QUERY_KIND_WEIGHTS.get(origin.query_kind, 0.70)
            / (RRF_K + origin.rank)
            for origin in origins
        )
        ranked.append(
            (
                RankedCandidate(
                    arxiv_id=candidate.arxiv_id,
                    title=candidate.title,
                    score=score,
                    best_rank=min(origin.rank for origin in origins),
                    matched_routes=len({origin.route_index for origin in origins}),
                ),
                candidate.first_seen_order,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item[0].score,
            item[0].best_rank,
            item[1],
            item[0].arxiv_id,
        )
    )
    return [item[0] for item in ranked]


def evaluate_ablation_arms(
    *,
    candidates: Sequence[DeduplicatedCandidate],
    route_results: Sequence[Mapping[str, Any]],
    gold: Sequence[GoldPaper],
    available_gold_indices: set[int],
) -> dict[str, AblationArmResult]:
    """Evaluate A/B/C with Gold isolated from retrieval and ranking."""
    arm_results: dict[str, AblationArmResult] = {}
    matches_by_arm: dict[str, dict[int, set[int]]] = {}
    gold_count = len(gold)
    corpus_gold_count = len(available_gold_indices)

    for arm in ARM_ALLOWED_KINDS:
        ranked = rank_candidates_for_arm(candidates, arm=arm)
        selected_routes = [
            route
            for route in route_results
            if route.get("query_source") == "original"
            or ARM_ALLOWED_KINDS[arm] is None
            or route.get("query_kind") in (ARM_ALLOWED_KINDS[arm] or set())
        ]
        raw_hits = sum(int(route.get("retrieved_count", 0)) for route in selected_routes)
        missing_ids = sum(
            int(route.get("missing_arxiv_id_hits", 0)) for route in selected_routes
        )
        comparable_hits = max(0, raw_hits - missing_ids)
        duplicate_hits = max(0, comparable_hits - len(ranked))

        recall_at: dict[str, float] = {}
        corpus_recall_at: dict[str, float] = {}
        matched_counts: dict[str, int] = {}
        matched_corpus_counts: dict[str, int] = {}
        arm_matches: dict[int, set[int]] = {}
        for k in TOP_K_VALUES:
            matched = _matched_gold_indices(
                (candidate.arxiv_id for candidate in ranked[:k]),
                gold,
            )
            arm_matches[k] = matched
            matched_corpus = matched & available_gold_indices
            matched_counts[str(k)] = len(matched)
            matched_corpus_counts[str(k)] = len(matched_corpus)
            recall_at[str(k)] = len(matched) / gold_count if gold_count else 0.0
            corpus_recall_at[str(k)] = (
                len(matched_corpus) / corpus_gold_count if corpus_gold_count else 0.0
            )

        first_relevant_rank = next(
            (
                rank
                for rank, candidate in enumerate(ranked, start=1)
                if _matched_gold_indices([candidate.arxiv_id], gold)
            ),
            None,
        )
        union_matches = _matched_gold_indices(
            (candidate.arxiv_id for candidate in ranked),
            gold,
        )
        union_corpus_matches = union_matches & available_gold_indices
        arm_results[arm] = AblationArmResult(
            arm=arm,
            query_count=len(selected_routes),
            raw_hits=raw_hits,
            unique_candidates=len(ranked),
            duplicate_hits_removed=duplicate_hits,
            duplicate_ratio=(duplicate_hits / comparable_hits if comparable_hits else 0.0),
            candidate_inflation_vs_original=0.0,
            recall_at=recall_at,
            corpus_recall_at=corpus_recall_at,
            matched_gold_count_at=matched_counts,
            matched_corpus_gold_count_at=matched_corpus_counts,
            marginal_gold_hits_at={str(k): 0 for k in TOP_K_VALUES},
            mrr=(1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0),
            union_recall=(len(union_matches) / gold_count if gold_count else 0.0),
            union_corpus_recall=(
                len(union_corpus_matches) / corpus_gold_count
                if corpus_gold_count
                else 0.0
            ),
            top_candidates=[asdict(candidate) for candidate in ranked[:100]],
        )
        matches_by_arm[arm] = arm_matches

    baseline = arm_results["A_original"]
    baseline_unique = baseline.unique_candidates
    for arm, result in arm_results.items():
        result.candidate_inflation_vs_original = (
            (result.unique_candidates - baseline_unique) / baseline_unique
            if baseline_unique
            else 0.0
        )
        result.marginal_gold_hits_at = {
            str(k): len(
                matches_by_arm[arm][k] - matches_by_arm["A_original"][k]
            )
            for k in TOP_K_VALUES
        }

    return arm_results


def evaluate_one(
    *,
    index: int,
    source_record: Mapping[str, Any],
    cache_record: Mapping[str, Any],
    corpus: PaSaCorpusBM25,
) -> QueryEvalResult:
    question = str(source_record["question"]).strip()
    gold = gold_papers(source_record)
    available_gold_indices = {
        gold_index
        for gold_index, paper in enumerate(gold)
        if paper.arxiv_id and corpus.get_by_arxiv_id(paper.arxiv_id) is not None
    }
    queries = cached_search_queries(
        source_record=source_record,
        cache_record=cache_record,
    )
    (
        candidates,
        route_results,
        raw_hits,
        missing_arxiv_id_hits,
        retrieval_latency_ms,
    ) = retrieve_and_deduplicate(corpus=corpus, queries=queries)

    original_ids = {
        candidate.arxiv_id
        for candidate in candidates
        if any(origin.query_source == "original" for origin in candidate.origins)
    }
    union_ids = {candidate.arxiv_id for candidate in candidates}
    original_matches = _matched_gold_indices(original_ids, gold)
    union_matches = _matched_gold_indices(union_ids, gold)
    original_corpus_matches = original_matches & available_gold_indices
    union_corpus_matches = union_matches & available_gold_indices
    gold_count = len(gold)
    corpus_gold_count = len(available_gold_indices)
    ablation = evaluate_ablation_arms(
        candidates=candidates,
        route_results=route_results,
        gold=gold,
        available_gold_indices=available_gold_indices,
    )

    # Add route-level exact-ID contribution after all local retrieval finishes.
    cumulative_matches: set[int] = set()
    for route_index, (query, route_result) in enumerate(zip(queries, route_results)):
        route_candidate_ids = {
            candidate.arxiv_id
            for candidate in candidates
            if any(origin.route_index == route_index for origin in candidate.origins)
        }
        route_matches = _matched_gold_indices(route_candidate_ids, gold)
        marginal = route_matches - cumulative_matches
        cumulative_matches |= route_matches
        route_result["matched_gold_count"] = len(route_matches)
        route_result["marginal_gold_count"] = len(marginal)
        route_result["matched_gold_arxiv_ids"] = [
            gold[item].arxiv_id for item in sorted(route_matches)
        ]

    def gold_json(gold_index: int) -> dict[str, Any]:
        paper = gold[gold_index]
        return {"arxiv_id": paper.arxiv_id, "title": paper.title}

    return QueryEvalResult(
        dataset_index=index,
        qid=str(source_record.get("qid", f"RealScholarQuery_{index}")),
        question=question,
        cache_created_at=str(cache_record.get("created_at", "")) or None,
        cache_protocol_sha256=str(cache_record["protocol_sha256"]),
        gold_count=gold_count,
        corpus_gold_count=corpus_gold_count,
        corpus_missing_count=gold_count - corpus_gold_count,
        corpus_coverage=corpus_gold_count / gold_count if gold_count else 0.0,
        search_query_count=len(queries),
        generated_query_count=len(queries) - 1,
        raw_hits=raw_hits,
        missing_arxiv_id_hits=missing_arxiv_id_hits,
        deduplicated_candidates=len(candidates),
        duplicate_hits_removed=raw_hits - missing_arxiv_id_hits - len(candidates),
        original_matched_gold_count=len(original_matches),
        union_matched_gold_count=len(union_matches),
        original_matched_corpus_gold_count=len(original_corpus_matches),
        union_matched_corpus_gold_count=len(union_corpus_matches),
        original_recall_at_100=(
            len(original_matches) / gold_count if gold_count else 0.0
        ),
        union_recall=len(union_matches) / gold_count if gold_count else 0.0,
        original_corpus_recall_at_100=(
            len(original_corpus_matches) / corpus_gold_count
            if corpus_gold_count
            else 0.0
        ),
        union_corpus_recall=(
            len(union_corpus_matches) / corpus_gold_count
            if corpus_gold_count
            else 0.0
        ),
        marginal_gold_hits=len(union_matches - original_matches),
        marginal_corpus_gold_hits=len(
            union_corpus_matches - original_corpus_matches
        ),
        retrieval_latency_ms=retrieval_latency_ms,
        route_results=route_results,
        matched_gold_at_union=[gold_json(item) for item in sorted(union_matches)],
        missed_gold_at_union=[
            gold_json(item)
            for item in range(gold_count)
            if item not in union_matches
        ],
        candidates=[
            {
                "arxiv_id": candidate.arxiv_id,
                "title": candidate.title,
                "year": candidate.year,
                "corpus_key": candidate.corpus_key,
                "first_seen_order": candidate.first_seen_order,
                "origins": [asdict(origin) for origin in candidate.origins],
            }
            for candidate in candidates
        ],
        ablation=ablation,
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(results: Sequence[QueryEvalResult]) -> dict[str, Any]:
    total_gold = sum(result.gold_count for result in results)
    total_corpus_gold = sum(result.corpus_gold_count for result in results)
    original_hits = sum(result.original_matched_gold_count for result in results)
    union_hits = sum(result.union_matched_gold_count for result in results)
    original_corpus_hits = sum(
        result.original_matched_corpus_gold_count for result in results
    )
    union_corpus_hits = sum(
        result.union_matched_corpus_gold_count for result in results
    )
    summary: dict[str, Any] = {
        "query_count": len(results),
        "gold_total": total_gold,
        "corpus_gold_total": total_corpus_gold,
        "corpus_missing_total": total_gold - total_corpus_gold,
        "corpus_coverage": total_corpus_gold / total_gold if total_gold else 0.0,
        "original_exact_id_hits_at_100": original_hits,
        "union_exact_id_hits": union_hits,
        "marginal_exact_id_hits": union_hits - original_hits,
        "micro_original_recall_at_100": (
            original_hits / total_gold if total_gold else 0.0
        ),
        "micro_union_recall": union_hits / total_gold if total_gold else 0.0,
        "micro_original_corpus_recall_at_100": (
            original_corpus_hits / total_corpus_gold if total_corpus_gold else 0.0
        ),
        "micro_union_corpus_recall": (
            union_corpus_hits / total_corpus_gold if total_corpus_gold else 0.0
        ),
        "macro_original_recall_at_100": _mean(
            result.original_recall_at_100 for result in results
        ),
        "macro_union_recall": _mean(result.union_recall for result in results),
        "macro_original_corpus_recall_at_100": _mean(
            result.original_corpus_recall_at_100 for result in results
        ),
        "macro_union_corpus_recall": _mean(
            result.union_corpus_recall for result in results
        ),
        "total_search_queries": sum(result.search_query_count for result in results),
        "total_generated_queries": sum(
            result.generated_query_count for result in results
        ),
        "total_raw_hits": sum(result.raw_hits for result in results),
        "total_deduplicated_candidates": sum(
            result.deduplicated_candidates for result in results
        ),
        "total_duplicate_hits_removed": sum(
            result.duplicate_hits_removed for result in results
        ),
        "total_missing_arxiv_id_hits": sum(
            result.missing_arxiv_id_hits for result in results
        ),
        "mean_retrieval_latency_ms": _mean(
            result.retrieval_latency_ms for result in results
        ),
        "total_retrieval_latency_ms": sum(
            result.retrieval_latency_ms for result in results
        ),
    }

    ablation_summary: dict[str, Any] = {}
    for arm in ARM_ALLOWED_KINDS:
        total_hits_at = {
            str(k): sum(
                result.ablation[arm].matched_gold_count_at[str(k)]
                for result in results
            )
            for k in TOP_K_VALUES
        }
        total_corpus_hits_at = {
            str(k): sum(
                result.ablation[arm].matched_corpus_gold_count_at[str(k)]
                for result in results
            )
            for k in TOP_K_VALUES
        }
        ablation_summary[arm] = {
            "micro_recall_at": {
                str(k): total_hits_at[str(k)] / total_gold if total_gold else 0.0
                for k in TOP_K_VALUES
            },
            "macro_recall_at": {
                str(k): _mean(
                    result.ablation[arm].recall_at[str(k)] for result in results
                )
                for k in TOP_K_VALUES
            },
            "micro_corpus_recall_at": {
                str(k): (
                    total_corpus_hits_at[str(k)] / total_corpus_gold
                    if total_corpus_gold
                    else 0.0
                )
                for k in TOP_K_VALUES
            },
            "macro_corpus_recall_at": {
                str(k): _mean(
                    result.ablation[arm].corpus_recall_at[str(k)]
                    for result in results
                )
                for k in TOP_K_VALUES
            },
            "mean_mrr": _mean(result.ablation[arm].mrr for result in results),
            "mean_query_count": _mean(
                float(result.ablation[arm].query_count) for result in results
            ),
            "mean_unique_candidates": _mean(
                float(result.ablation[arm].unique_candidates) for result in results
            ),
            "mean_duplicate_ratio": _mean(
                result.ablation[arm].duplicate_ratio for result in results
            ),
            "mean_candidate_inflation_vs_original": _mean(
                result.ablation[arm].candidate_inflation_vs_original
                for result in results
            ),
            "marginal_gold_hits_at": {
                str(k): sum(
                    result.ablation[arm].marginal_gold_hits_at[str(k)]
                    for result in results
                )
                for k in TOP_K_VALUES
            },
        }
    summary["ablation"] = ablation_summary
    return summary


def print_query_result(result: QueryEvalResult) -> None:
    print("\n" + "=" * 88)
    print(f"Frozen-1 #{result.dataset_index}: {result.qid}")
    print("=" * 88)
    print(f"question                   = {result.question}")
    print(
        "queries original/generated = "
        f"1/{result.generated_query_count}"
    )
    print(
        "hits raw/dedup/removed      = "
        f"{result.raw_hits}/{result.deduplicated_candidates}/"
        f"{result.duplicate_hits_removed}"
    )
    print(
        "Gold original/union         = "
        f"{result.original_matched_gold_count}/"
        f"{result.union_matched_gold_count} of {result.gold_count}"
    )
    print(
        "Recall original@100/union   = "
        f"{result.original_recall_at_100:.4f}/"
        f"{result.union_recall:.4f}"
    )
    print(f"marginal Gold hits          = {result.marginal_gold_hits}")
    print(f"retrieval_latency_ms        = {result.retrieval_latency_ms:.2f}")
    print("\nA/B/C weighted-RRF ranking:")
    for arm in ARM_ALLOWED_KINDS:
        item = result.ablation[arm]
        print(
            f"  {arm:<16} queries={item.query_count} "
            f"unique={item.unique_candidates} mrr={item.mrr:.4f} "
            f"R@20/50/100="
            f"{item.recall_at['20']:.4f}/"
            f"{item.recall_at['50']:.4f}/"
            f"{item.recall_at['100']:.4f} "
            f"marginal@100={item.marginal_gold_hits_at['100']}"
        )


def main() -> int:
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    dataset_path = require_file(args.dataset, "RealScholarQuery dataset")
    cache_path = require_file(args.cache, "Frozen V2 query cache")
    dataset_sha256 = _sha256_bytes(dataset_path.read_bytes())
    dataset_records = load_dataset(dataset_path)
    cache_records = load_cache(cache_path)
    end = min(len(dataset_records), args.start + args.limit)
    selected_indices = list(range(args.start, end))
    if not selected_indices:
        raise ValueError(
            f"No records selected: start={args.start}, limit={args.limit}, "
            f"dataset_size={len(dataset_records)}"
        )

    protocol_sha256 = validate_selected_cache(
        dataset_records=dataset_records,
        cache_records=cache_records,
        selected_indices=selected_indices,
        dataset_sha256=dataset_sha256,
    )
    corpus = PaSaCorpusBM25(
        paper_zip_path=args.paper_zip,
        id_map_path=args.id_map,
        index_path=args.index,
    )
    if not corpus.is_index_ready():
        raise RuntimeError(
            f"PaSa BM25 index is not ready: {corpus.index_path}\n"
            "Build it first with:\n"
            "  python -m scholar_agent_v2.retrieval.sources.pasa_corpus build"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.results_dir.expanduser().resolve()
        / f"frozen_v2_queries_{args.start}_{end - 1}_{stamp}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("PaSa Local A/B/C: Frozen V2 Queries -> SQLite FTS5 BM25 -> RRF")
    print("=" * 88)
    print(f"dataset          = {dataset_path}")
    print(f"cache            = {cache_path}")
    print(f"cache_protocol   = {protocol_sha256}")
    print(f"index            = {corpus.index_path}")
    print(f"range            = [{args.start}, {end})")
    print("per_query_top_k  = 100")
    print("deduplication    = exact normalized arXiv ID")
    print(f"ranking          = weighted RRF (k={RRF_K})")
    print("ranked metrics   = Recall@20/50/100, corpus Recall, MRR")
    print("Gold matching    = exact normalized arXiv ID")
    print("LLM/network      = disabled")

    results: list[QueryEvalResult] = []
    for index in selected_indices:
        result = evaluate_one(
            index=index,
            source_record=dataset_records[index],
            cache_record=cache_records[index],
            corpus=corpus,
        )
        results.append(result)
        print_query_result(result)

    summary = aggregate(results)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "name": "pasa_local_ranked_ablation_v1",
            "dataset": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "query_cache": str(cache_path),
            "query_cache_protocol_sha256": protocol_sha256,
            "query_inputs": "original_plus_cached_generated_queries",
            "retriever": "local_sqlite_fts5_bm25",
            "per_query_limit": RETRIEVAL_LIMIT_PER_QUERY,
            "query_operator": "or",
            "ablation_arms": {
                "A_original": ["original"],
                "B_deterministic": ["original", "exact", "core", "relation"],
                "C_full": [
                    "original",
                    "exact",
                    "core",
                    "relation",
                    "alias",
                    "expanded",
                ],
            },
            "candidate_ranker": "weighted_reciprocal_rank_fusion",
            "rrf_k": RRF_K,
            "query_kind_weights": QUERY_KIND_WEIGHTS,
            "ranked_top_k": list(TOP_K_VALUES),
            "candidate_merge_legacy": "union_preserve_first_seen",
            "candidate_dedup": "exact_normalized_arxiv_id",
            "gold_match": "exact_normalized_arxiv_id",
            "title_fallback": False,
            "primary_metrics": [
                "exact_id_recall_at_20",
                "exact_id_recall_at_50",
                "exact_id_recall_at_100",
                "corpus_normalized_recall_at_20_50_100",
                "mrr",
                "marginal_gold_hits_vs_A",
                "candidate_inflation_vs_A",
                "duplicate_ratio",
            ],
            "legacy_metric": "unranked_deduplicated_multi_query_union_recall",
            "llm": False,
            "query_understanding_runtime": False,
            "query_generation_runtime": False,
            "openalex": False,
            "semantic_scholar": False,
            "arxiv_api": False,
            "google_search": False,
            "network": False,
        },
        "range": {"start": args.start, "end_exclusive": end},
        "index_metadata": corpus.index_metadata(),
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("FINAL LOCAL A/B/C SUMMARY")
    print("=" * 88)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
