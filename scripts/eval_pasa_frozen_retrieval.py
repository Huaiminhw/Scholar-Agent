"""Evaluate the simplest frozen PaSa retrieval baseline.

Pipeline:

    RealScholarQuery/test.jsonl
        -> original query
        -> local PaSa corpus
        -> BM25(title + abstract)
        -> top 100
        -> Gold matching

This script intentionally does NOT use Query Understanding, Query Generation,
BGE-M3, RRF, citation expansion, Adaptive Router, Selector, OpenAlex,
Semantic Scholar, arXiv API, Google Search, an LLM, or any network service.

Build the local index before running this evaluation:

    python -m scholar_agent_v2.retrieval.sources.pasa_corpus build

Then run:

    python scripts/eval_pasa_frozen_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent_v2.retrieval.sources.pasa_corpus import (  # noqa: E402
    PaSaBM25Result,
    PaSaCorpusBM25,
    normalize_arxiv_id,
)


TOP_K_VALUES = (20, 50, 100)
RETRIEVAL_LIMIT = 100


@dataclass(frozen=True)
class GoldPaper:
    title: str
    arxiv_id: str | None


@dataclass
class FrozenQueryResult:
    query_index: int
    qid: str
    question: str
    source_published_time: str | None
    gold_count: int
    corpus_gold_count: int
    corpus_missing_count: int
    corpus_coverage: float
    retrieved_count: int = 0
    latency_ms: float = 0.0
    recall_at: dict[str, float] = field(default_factory=dict)
    corpus_recall_at: dict[str, float] = field(default_factory=dict)
    matched_gold_count_at: dict[str, int] = field(default_factory=dict)
    matched_corpus_gold_count_at: dict[str, int] = field(default_factory=dict)
    matched_gold_titles_at_100: list[str] = field(default_factory=list)
    missed_gold_titles_at_100: list[str] = field(default_factory=list)
    top_results: list[dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate original-query BM25 Recall@20/50/100 on the frozen PaSa corpus."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            os.getenv(
                "PASA_REALSCHOLARQUERY_PATH",
                PROJECT_ROOT / "data" / "RealScholarQuery" / "test.jsonl",
            )
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            os.getenv(
                "PASA_BM25_INDEX_PATH",
                PROJECT_ROOT / "models" / "pasa_bm25" / "pasa_corpus.sqlite3",
            )
        ),
    )
    parser.add_argument(
        "--paper-zip",
        type=Path,
        default=Path(
            os.getenv(
                "PASA_PAPER_ZIP_PATH",
                PROJECT_ROOT / "data" / "paper_database" / "cs_paper_2nd.zip",
            )
        ),
    )
    parser.add_argument(
        "--id-map",
        type=Path,
        default=Path(
            os.getenv(
                "PASA_ID2PAPER_PATH",
                PROJECT_ROOT / "data" / "paper_database" / "id2paper.json",
            )
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=int(os.getenv("PASA_FROZEN_EVAL_START", "0")),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("PASA_FROZEN_EVAL_LIMIT", "50")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(record.get("question"), str):
                raise ValueError(f"Dataset line {line_number} has no string question")
            if not isinstance(record.get("answer"), list):
                raise ValueError(f"Dataset line {line_number} has no answer list")
            if not isinstance(record.get("answer_arxiv_id"), list):
                raise ValueError(f"Dataset line {line_number} has no answer_arxiv_id list")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def gold_papers(record: dict[str, Any]) -> list[GoldPaper]:
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


def result_matches_gold(result: PaSaBM25Result, gold: GoldPaper) -> bool:
    """Match retrieval results and Gold with the same exact-ID criterion.

    Title fallback is intentionally excluded.  It can conflate distinct papers
    such as ``Panacea`` and ``Panacea+`` after punctuation normalization, and it
    would make retrieval-hit counting inconsistent with corpus availability.
    """

    result_arxiv_id = normalize_arxiv_id(result.arxiv_id)
    return bool(gold.arxiv_id and result_arxiv_id == gold.arxiv_id)


def matched_gold_indices(
    results: Sequence[PaSaBM25Result],
    gold: Sequence[GoldPaper],
    *,
    limit: int,
) -> set[int]:
    selected = results[:limit]
    return {
        gold_index
        for gold_index, gold_paper in enumerate(gold)
        if any(result_matches_gold(result, gold_paper) for result in selected)
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_one(
    *,
    query_index: int,
    record: dict[str, Any],
    corpus: PaSaCorpusBM25,
) -> FrozenQueryResult:
    question = record["question"].strip()
    gold = gold_papers(record)

    # Corpus-restricted Gold is resolved only by arXiv ID. This deliberately
    # avoids using BM25 search results to decide the evaluation denominator.
    available_gold_indices = {
        index
        for index, paper in enumerate(gold)
        if paper.arxiv_id and corpus.get_by_arxiv_id(paper.arxiv_id) is not None
    }
    corpus_gold_count = len(available_gold_indices)
    gold_count = len(gold)

    started = time.perf_counter()
    retrieved = corpus.search(question, limit=RETRIEVAL_LIMIT, operator="or")
    latency_ms = (time.perf_counter() - started) * 1000.0

    recall_at: dict[str, float] = {}
    corpus_recall_at: dict[str, float] = {}
    matched_counts: dict[str, int] = {}
    matched_corpus_counts: dict[str, int] = {}
    matches_by_k: dict[int, set[int]] = {}
    for k in TOP_K_VALUES:
        matched = matched_gold_indices(retrieved, gold, limit=k)
        matches_by_k[k] = matched
        matched_counts[str(k)] = len(matched)
        recall_at[str(k)] = len(matched) / gold_count if gold_count else 0.0
        matched_available = len(matched & available_gold_indices)
        matched_corpus_counts[str(k)] = matched_available
        corpus_recall_at[str(k)] = (
            matched_available / corpus_gold_count if corpus_gold_count else 0.0
        )

    matched_at_100 = matches_by_k[100]
    return FrozenQueryResult(
        query_index=query_index,
        qid=str(record.get("qid", f"RealScholarQuery_{query_index}")),
        question=question,
        source_published_time=str(
            (record.get("source_meta") or {}).get("published_time", "")
        )
        or None,
        gold_count=gold_count,
        corpus_gold_count=corpus_gold_count,
        corpus_missing_count=gold_count - corpus_gold_count,
        corpus_coverage=corpus_gold_count / gold_count if gold_count else 0.0,
        retrieved_count=len(retrieved),
        latency_ms=latency_ms,
        recall_at=recall_at,
        corpus_recall_at=corpus_recall_at,
        matched_gold_count_at=matched_counts,
        matched_corpus_gold_count_at=matched_corpus_counts,
        matched_gold_titles_at_100=[
            paper.title for index, paper in enumerate(gold) if index in matched_at_100
        ],
        missed_gold_titles_at_100=[
            paper.title for index, paper in enumerate(gold) if index not in matched_at_100
        ],
        top_results=[
            {
                "rank": result.rank,
                "bm25_score": result.bm25_score,
                "arxiv_id": result.arxiv_id,
                "year": result.year,
                "title": result.title,
                "corpus_key": result.corpus_key,
            }
            for result in retrieved
        ],
    )


def aggregate(results: Sequence[FrozenQueryResult]) -> dict[str, Any]:
    total_gold = sum(result.gold_count for result in results)
    total_corpus_gold = sum(result.corpus_gold_count for result in results)
    summary: dict[str, Any] = {
        "query_count": len(results),
        "gold_total": total_gold,
        "corpus_gold_total": total_corpus_gold,
        "corpus_missing_total": total_gold - total_corpus_gold,
        "corpus_coverage": total_corpus_gold / total_gold if total_gold else 0.0,
        "macro_recall_at": {},
        "micro_recall_at": {},
        "macro_corpus_recall_at": {},
        "micro_corpus_recall_at": {},
        "mean_latency_ms": mean(result.latency_ms for result in results),
        "total_latency_ms": sum(result.latency_ms for result in results),
        "mean_retrieved_count": mean(result.retrieved_count for result in results),
    }
    for k in TOP_K_VALUES:
        key = str(k)
        total_matched = sum(result.matched_gold_count_at.get(key, 0) for result in results)
        total_corpus_matched = sum(
            result.matched_corpus_gold_count_at.get(key, 0) for result in results
        )
        summary["macro_recall_at"][key] = mean(
            result.recall_at.get(key, 0.0) for result in results
        )
        summary["micro_recall_at"][key] = (
            total_matched / total_gold if total_gold else 0.0
        )
        summary["macro_corpus_recall_at"][key] = mean(
            result.corpus_recall_at.get(key, 0.0) for result in results
        )
        summary["micro_corpus_recall_at"][key] = (
            total_corpus_matched / total_corpus_gold if total_corpus_gold else 0.0
        )
    return summary


def print_query_result(result: FrozenQueryResult) -> None:
    print("\n" + "=" * 88)
    print(f"Frozen BM25 #{result.query_index}: {result.qid}")
    print("=" * 88)
    print(f"question             = {result.question}")
    print(
        "gold/corpus/missing  = "
        f"{result.gold_count}/{result.corpus_gold_count}/{result.corpus_missing_count}"
    )
    print(f"corpus coverage      = {result.corpus_coverage:.4f}")
    print(f"retrieved            = {result.retrieved_count}")
    print(
        "Recall@20/50/100     = "
        f"{result.recall_at['20']:.4f}/"
        f"{result.recall_at['50']:.4f}/"
        f"{result.recall_at['100']:.4f}"
    )
    print(
        "CorpusR@20/50/100    = "
        f"{result.corpus_recall_at['20']:.4f}/"
        f"{result.corpus_recall_at['50']:.4f}/"
        f"{result.corpus_recall_at['100']:.4f}"
    )
    print(f"latency_ms           = {result.latency_ms:.2f}")


def main() -> int:
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    dataset_path = require_file(args.dataset, "RealScholarQuery dataset")
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

    records = load_records(dataset_path)
    end = min(len(records), args.start + args.limit)
    selected = records[args.start:end]
    if not selected:
        raise ValueError(
            f"No records selected: start={args.start}, limit={args.limit}, "
            f"dataset_size={len(records)}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else PROJECT_ROOT
        / "results"
        / "pasa_frozen_retrieval"
        / f"frozen_bm25_{args.start}_{end - 1}_{stamp}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("PaSa Frozen Retrieval Baseline: Original Query -> BM25 -> Top 100 -> Gold")
    print("=" * 88)
    print(f"dataset = {dataset_path}")
    print(f"index   = {corpus.index_path}")
    print(f"range   = [{args.start}, {end})")
    print("LLM/API = disabled")

    results: list[FrozenQueryResult] = []
    for query_index, record in enumerate(selected, start=args.start):
        result = evaluate_one(query_index=query_index, record=record, corpus=corpus)
        results.append(result)
        print_query_result(result)

    summary = aggregate(results)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "dataset": str(dataset_path),
            "index": str(corpus.index_path),
            "query": "original_query_only",
            "retriever": "sqlite_fts5_bm25",
            "gold_match": "exact_normalized_arxiv_id",
            "title_fallback": False,
            "retrieval_limit": RETRIEVAL_LIMIT,
            "query_operator": "or",
            "query_understanding": False,
            "query_generation": False,
            "dense_retrieval": False,
            "rrf": False,
            "citation_expansion": False,
            "adaptive_router": False,
            "selector": False,
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
    print("FINAL FROZEN BM25 SUMMARY")
    print("=" * 88)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
