"""Evaluate Scholar Agent V2 retrieval recall on PaSa RealScholarQuery.

This benchmark intentionally stops after Candidate Pool hard filtering and
cheap reranking.  It does not call enrichment, Selector, PaSa local models, or
compute precision/F1.

PowerShell example::

    $env:PASA_RETRIEVAL_EVAL_START="0"
    $env:PASA_RETRIEVAL_EVAL_LIMIT="10"
    python scripts/eval_pasa_retrieval.py

The legacy PASA_EVAL_START/PASA_EVAL_LIMIT variables are also accepted.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from scholar_agent_v2.candidate_pool.models import CandidatePaper
from scholar_agent_v2.candidate_pool.pool import CandidatePool
from scholar_agent_v2.query_generation.generator import GeneratorConfig, QueryGenerator
from scholar_agent_v2.query_understanding.analyzer import AnalyzerConfig, QueryAnalyzer
from scholar_agent_v2.retrieval.config import get_enabled_sources
from scholar_agent_v2.retrieval.models import RetrievalPlan, SearchSource
from scholar_agent_v2.retrieval.normalizer import normalize_arxiv_id
from scholar_agent_v2.retrieval.orchestrator import RetrievalOrchestrator
from scholar_agent_v2.retrieval.route_builder import RetrievalRouteBuilder
from scholar_agent_v2.retrieval.sources.arxiv import ArxivAdapter
from scholar_agent_v2.retrieval.sources.google_search import GoogleSearchAdapter
from scholar_agent_v2.retrieval.sources.openalex import OpenAlexAdapter
from scholar_agent_v2.retrieval.sources.semantic_scholar import SemanticScholarAdapter
from tests.real_llm_client import build_client


def _env_int(name: str, default: int, *, fallback: str | None = None) -> int:
    raw = os.getenv(name)
    if raw is None and fallback:
        raw = os.getenv(fallback)
    return int(raw) if raw not in {None, ""} else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in {None, ""} else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


@dataclass(frozen=True)
class EvalConfig:
    start: int
    limit: int
    max_routes: int
    per_route_limit: int
    max_total_hits: int
    route_timeout_seconds: float
    max_attempts: int
    retry_backoff_seconds: float
    max_concurrency: int
    sleep_seconds: float
    use_llm_expansion: bool
    results_dir: Path
    dataset_path: Path

    @classmethod
    def from_env(cls) -> "EvalConfig":
        dataset_env = os.getenv("PASA_REALSCHOLARQUERY_PATH", "").strip()
        candidates = [
            Path(dataset_env).expanduser() if dataset_env else None,
            PROJECT_ROOT / "data" / "RealScholarQuery" / "test.jsonl",
            PROJECT_ROOT / "pasa" / "data" / "RealScholarQuery" / "test.jsonl",
            PROJECT_ROOT.parent / "pasa" / "data" / "RealScholarQuery" / "test.jsonl",
        ]
        dataset_path = next(
            (path.resolve() for path in candidates if path is not None and path.is_file()),
            None,
        )
        if dataset_path is None:
            checked = "\n".join(f"  - {path}" for path in candidates if path is not None)
            raise FileNotFoundError(
                "PaSa RealScholarQuery/test.jsonl was not found. Set "
                f"PASA_REALSCHOLARQUERY_PATH. Checked:\n{checked}"
            )

        config = cls(
            start=_env_int("PASA_RETRIEVAL_EVAL_START", 0, fallback="PASA_EVAL_START"),
            limit=_env_int("PASA_RETRIEVAL_EVAL_LIMIT", 10, fallback="PASA_EVAL_LIMIT"),
            max_routes=_env_int("PASA_RETRIEVAL_MAX_ROUTES", 10),
            per_route_limit=_env_int("PASA_RETRIEVAL_PER_ROUTE_LIMIT", 28),
            max_total_hits=_env_int("PASA_RETRIEVAL_MAX_TOTAL_HITS", 280),
            route_timeout_seconds=_env_float("PASA_RETRIEVAL_ROUTE_TIMEOUT_SECONDS", 35.0),
            max_attempts=_env_int("PASA_RETRIEVAL_MAX_ATTEMPTS", 2),
            retry_backoff_seconds=_env_float("PASA_RETRIEVAL_RETRY_BACKOFF_SECONDS", 1.0),
            max_concurrency=_env_int("PASA_RETRIEVAL_MAX_CONCURRENCY", 6),
            sleep_seconds=_env_float("PASA_RETRIEVAL_SLEEP_SECONDS", 1.0),
            use_llm_expansion=_env_bool("PASA_RETRIEVAL_USE_LLM_EXPANSION", True),
            results_dir=Path(
                os.getenv(
                    "PASA_RETRIEVAL_RESULTS_DIR",
                    str(PROJECT_ROOT / "results" / "pasa_retrieval"),
                )
            ).expanduser(),
            dataset_path=dataset_path,
        )
        if config.start < 0:
            raise ValueError("PASA_RETRIEVAL_EVAL_START must be >= 0")
        if config.limit <= 0:
            raise ValueError("PASA_RETRIEVAL_EVAL_LIMIT must be > 0")
        for name in ("max_routes", "per_route_limit", "max_total_hits", "max_attempts", "max_concurrency"):
            if getattr(config, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        return config


@dataclass(frozen=True)
class GoldPaper:
    title: str
    arxiv_id: str | None = None


@dataclass
class QueryRecallResult:
    index: int
    qid: str
    question: str
    gold_count: int
    source_published_time: str | None
    cutoff_year: int | None

    status: str = "success"
    failed_stage: str | None = None
    error: str | None = None

    query_analysis_api_calls: int = 0
    query_analysis_tokens: int = 0
    query_analysis_ms: float = 0.0
    query_generation_api_calls: int = 0
    query_generation_tokens: int = 0
    query_generation_ms: float = 0.0

    native_routes: int = 0
    executed_routes: int = 0
    retrieval_api_calls: int = 0
    retrieval_retries: int = 0
    retrieval_ms: float = 0.0
    raw_hits: int = 0
    unique_candidates_before_filter: int = 0
    candidates_after_hard_filter: int = 0
    unknown_year_candidates: int = 0

    pre_filter_pool_recall: float = 0.0
    post_filter_pool_recall: float = 0.0
    recall_at: dict[str, float] = field(default_factory=dict)
    cutoff_year_recall_at: dict[str, float] = field(default_factory=dict)

    matched_gold_titles: list[str] = field(default_factory=list)
    missed_gold_titles: list[str] = field(default_factory=list)
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_wall_ms: float = 0.0


def keep_letters(text: str) -> str:
    """PaSa-compatible title normalization."""

    return "".join(ch for ch in str(text) if ch.isalpha()).lower()


def _normalized_arxiv_id(value: str | None) -> str | None:
    return normalize_arxiv_id(value, drop_version=True) if value else None


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
                raise ValueError(f"Record {line_number} has no string question")
            if not isinstance(record.get("answer"), list):
                raise ValueError(f"Record {line_number} has no answer list")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def gold_papers(record: dict[str, Any]) -> list[GoldPaper]:
    titles = [str(value).strip() for value in record.get("answer", [])]
    raw_ids = record.get("answer_arxiv_id", [])
    ids = list(raw_ids) if isinstance(raw_ids, list) else []
    papers: list[GoldPaper] = []
    for index, title in enumerate(titles):
        if not title:
            continue
        arxiv_id = _normalized_arxiv_id(str(ids[index])) if index < len(ids) and ids[index] else None
        papers.append(GoldPaper(title=title, arxiv_id=arxiv_id))
    return papers


def _candidate_matches_gold(candidate: CandidatePaper, gold: GoldPaper) -> bool:
    candidate_arxiv = _normalized_arxiv_id(candidate.identifiers.arxiv_id)
    if gold.arxiv_id and candidate_arxiv and gold.arxiv_id == candidate_arxiv:
        return True
    return bool(candidate.title and keep_letters(candidate.title) == keep_letters(gold.title))


def matched_gold_indices(
    candidates: Sequence[CandidatePaper],
    gold: Sequence[GoldPaper],
    *,
    limit: int | None = None,
) -> set[int]:
    selected = candidates if limit is None else candidates[:limit]
    return {
        index
        for index, paper in enumerate(gold)
        if any(_candidate_matches_gold(candidate, paper) for candidate in selected)
    }


def recall(candidates: Sequence[CandidatePaper], gold: Sequence[GoldPaper], limit: int | None = None) -> float:
    if not gold:
        return 0.0
    return len(matched_gold_indices(candidates, gold, limit=limit)) / len(gold)


def _cutoff_year(record: dict[str, Any]) -> int | None:
    value = str((record.get("source_meta") or {}).get("published_time", ""))
    return int(value[:4]) if len(value) >= 4 and value[:4].isdigit() else None


def build_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer(
        build_client(),
        config=AnalyzerConfig(
            temperature=0.0,
            use_structured_output=False,
            prompt_schema_mode="summary",
            include_few_shot_examples=False,
            enable_repair=True,
            repair_schema_mode="full",
            max_transport_retries=0,
            retain_raw_output=False,
        ),
    )


def build_generator(config: EvalConfig) -> QueryGenerator:
    return QueryGenerator(
        client=build_client() if config.use_llm_expansion else None,
        config=GeneratorConfig(
            temperature=0.0,
            use_structured_output=False,
            enable_llm_expansion=config.use_llm_expansion,
            max_transport_retries=0,
            max_total_queries=5,
            near_duplicate_threshold=0.88,
        ),
    )


def build_adapters(enabled_sources: Sequence[SearchSource], config: EvalConfig) -> dict:
    adapters: dict = {}
    if SearchSource.OPENALEX in enabled_sources:
        key = os.getenv("OPENALEX_API_KEY", "").strip()
        if not key:
            raise RuntimeError("USE_OPENALEX=true but OPENALEX_API_KEY is missing")
        adapters[SearchSource.OPENALEX] = OpenAlexAdapter(
            api_key=key,
            timeout_seconds=config.route_timeout_seconds,
        )
    if SearchSource.SEMANTIC_SCHOLAR in enabled_sources:
        adapters[SearchSource.SEMANTIC_SCHOLAR] = SemanticScholarAdapter(
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None,
            timeout_seconds=config.route_timeout_seconds,
        )
    if SearchSource.ARXIV in enabled_sources:
        adapters[SearchSource.ARXIV] = ArxivAdapter(
            timeout_seconds=config.route_timeout_seconds,
        )
    if SearchSource.GOOGLE_SEARCH in enabled_sources:
        key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
        engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "").strip()
        if not key or not engine_id:
            raise RuntimeError(
                "USE_GOOGLE_SEARCH=true but GOOGLE_SEARCH_API_KEY or "
                "GOOGLE_SEARCH_ENGINE_ID is missing"
            )
        adapters[SearchSource.GOOGLE_SEARCH] = GoogleSearchAdapter(
            api_key=key,
            engine_id=engine_id,
            timeout_seconds=config.route_timeout_seconds,
        )
    return adapters


def execution_plan(native_plan: RetrievalPlan, config: EvalConfig) -> RetrievalPlan:
    """Apply benchmark safety caps without increasing native production budgets."""

    routes = [route for route in native_plan.routes if route.enabled][: config.max_routes]
    routes = [
        route.model_copy(update={"limit": min(route.limit, config.per_route_limit)})
        for route in routes
    ]
    if not routes:
        return native_plan
    max_hits = min(
        native_plan.max_total_hits,
        config.max_total_hits,
        sum(route.limit for route in routes),
    )
    return RetrievalPlan(
        routes=routes,
        max_routes=len(routes),
        max_total_hits=max(1, max_hits),
        warnings=list(native_plan.warnings),
    )


async def evaluate_one(
    *,
    index: int,
    record: dict[str, Any],
    config: EvalConfig,
    analyzer: QueryAnalyzer,
    generator: QueryGenerator,
    orchestrator: RetrievalOrchestrator,
    enabled_sources: Sequence[SearchSource],
) -> QueryRecallResult:
    question = record["question"].strip()
    gold = gold_papers(record)
    published_time = str((record.get("source_meta") or {}).get("published_time", "")) or None
    result = QueryRecallResult(
        index=index,
        qid=str(record.get("qid", f"RealScholarQuery_{index}")),
        question=question,
        gold_count=len(gold),
        source_published_time=published_time,
        cutoff_year=_cutoff_year(record),
    )
    started = time.perf_counter()
    stage = "query_understanding"

    try:
        qa = await analyzer.analyze(question)
        result.query_analysis_api_calls = qa.metrics.api_calls
        result.query_analysis_tokens = qa.metrics.total_tokens
        result.query_analysis_ms = qa.metrics.latency_ms

        stage = "query_generation"
        generated = await generator.generate(qa.result)
        result.query_generation_api_calls = generated.metrics.api_calls
        result.query_generation_tokens = generated.metrics.total_tokens
        result.query_generation_ms = generated.metrics.latency_ms
        if generated.validation.errors:
            errors = " | ".join(f"{issue.code}: {issue.message}" for issue in generated.validation.errors)
            raise RuntimeError(f"Query Generation validation failed: {errors}")

        stage = "route_builder"
        native_plan = RetrievalRouteBuilder(enabled_sources=enabled_sources).build(qa.result, generated)
        plan = execution_plan(native_plan, config)
        result.native_routes = len([route for route in native_plan.routes if route.enabled])

        stage = "retrieval"
        retrieval = await orchestrator.run(plan)
        result.executed_routes = retrieval.metrics.executed_routes
        result.retrieval_api_calls = retrieval.metrics.api_calls
        result.retrieval_retries = retrieval.metrics.retries
        result.retrieval_ms = retrieval.metrics.total_latency_ms
        result.raw_hits = retrieval.metrics.raw_hits
        result.warnings.extend(retrieval.warnings)
        if not retrieval.hits:
            raise RuntimeError("Retrieval returned zero hits")

        stage = "candidate_pool"
        pool = CandidatePool.from_hits(retrieval.hits)
        pre_filter = list(pool.candidates)
        result.unique_candidates_before_filter = len(pre_filter)
        result.pre_filter_pool_recall = recall(pre_filter, gold)

        pool.apply_hard_filters(qa.result)
        pool.prerank(
            qa.result,
            generated,
            num_enabled_sources=len(enabled_sources),
            num_planned_routes=len(plan.routes),
        )
        ranked = pool.top(len(pool.candidates))
        result.candidates_after_hard_filter = len(ranked)
        result.post_filter_pool_recall = recall(ranked, gold)
        result.recall_at = {str(k): recall(ranked, gold, k) for k in (20, 50, 100)}

        result.unknown_year_candidates = sum(candidate.year is None for candidate in ranked)
        if result.cutoff_year is not None:
            # This is an explicitly labelled approximation. The current normalized
            # CandidatePaper contract has year, not a reliable day-level date.
            cutoff_ranked = [
                candidate
                for candidate in ranked
                if candidate.year is not None and candidate.year <= result.cutoff_year
            ]
            result.cutoff_year_recall_at = {
                str(k): recall(cutoff_ranked, gold, k) for k in (20, 50, 100)
            }

        matched = matched_gold_indices(ranked, gold)
        result.matched_gold_titles = [paper.title for i, paper in enumerate(gold) if i in matched]
        result.missed_gold_titles = [paper.title for i, paper in enumerate(gold) if i not in matched]
        result.top_candidates = [
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "title": candidate.title,
                "arxiv_id": candidate.identifiers.arxiv_id,
                "year": candidate.year,
                "score": candidate.retrieval_score,
                "source_count": candidate.source_count,
                "route_count": candidate.route_count,
            }
            for rank, candidate in enumerate(ranked[:100], start=1)
        ]
    except Exception as exc:
        result.status = "failed"
        result.failed_stage = stage
        result.error = f"{type(exc).__name__}: {exc}"
        print(f"\n[ERROR] query #{index} failed at {stage}: {result.error}")
        traceback.print_exc()
    finally:
        result.total_wall_ms = (time.perf_counter() - started) * 1000.0

    return result


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(results: Sequence[QueryRecallResult]) -> dict[str, Any]:
    successful = [result for result in results if result.status == "success"]
    failed = [result for result in results if result.status != "success"]
    total_gold = sum(result.gold_count for result in successful)
    total_matched = sum(len(result.matched_gold_titles) for result in successful)
    return {
        "requested_queries": len(results),
        "successful_queries": len(successful),
        "failed_queries": len(failed),
        "failed_indices": [result.index for result in failed],
        "macro_recall": {
            "pre_filter_pool": _mean(result.pre_filter_pool_recall for result in successful),
            "post_filter_pool": _mean(result.post_filter_pool_recall for result in successful),
            "at_20": _mean(result.recall_at.get("20", 0.0) for result in successful),
            "at_50": _mean(result.recall_at.get("50", 0.0) for result in successful),
            "at_100": _mean(result.recall_at.get("100", 0.0) for result in successful),
        },
        "macro_approximate_cutoff_year_recall": {
            "at_20": _mean(result.cutoff_year_recall_at.get("20", 0.0) for result in successful),
            "at_50": _mean(result.cutoff_year_recall_at.get("50", 0.0) for result in successful),
            "at_100": _mean(result.cutoff_year_recall_at.get("100", 0.0) for result in successful),
        },
        "micro_full_pool_recall": total_matched / total_gold if total_gold else 0.0,
        "gold": {"total": total_gold, "matched": total_matched, "missed": total_gold - total_matched},
        "efficiency": {
            "query_analysis_api_calls": sum(result.query_analysis_api_calls for result in results),
            "query_generation_api_calls": sum(result.query_generation_api_calls for result in results),
            "retrieval_api_calls": sum(result.retrieval_api_calls for result in results),
            "retrieval_retries": sum(result.retrieval_retries for result in results),
            "raw_hits": sum(result.raw_hits for result in results),
            "unique_candidates_before_filter": sum(
                result.unique_candidates_before_filter for result in results
            ),
            "mean_wall_seconds_per_query": _mean(
                result.total_wall_ms / 1000.0 for result in results
            ),
        },
        "notes": {
            "selector_called": False,
            "enrichment_called": False,
            "precision_or_f1_computed": False,
            "live_index_warning": (
                "OpenAlex/Semantic Scholar/arXiv are current indexes. These scores are not "
                "strictly comparable to the frozen PaSa paper results."
            ),
            "cutoff_warning": (
                "cutoff_year_recall is a post-retrieval year-only approximation; it is not a "
                "historical index snapshot or an exact published_time-minus-seven-days replay."
            ),
        },
    }


def print_query_result(result: QueryRecallResult) -> None:
    print("\n" + "=" * 88)
    print(f"PaSa retrieval-only #{result.index}: {result.qid}")
    print("=" * 88)
    print(f"question               = {result.question}")
    print(f"status                 = {result.status}")
    if result.status != "success":
        print(f"failed_stage           = {result.failed_stage}")
        print(f"error                  = {result.error}")
        return
    print(f"gold_count             = {result.gold_count}")
    print(f"routes native/executed = {result.native_routes}/{result.executed_routes}")
    print(f"raw/unique/filtered    = {result.raw_hits}/{result.unique_candidates_before_filter}/{result.candidates_after_hard_filter}")
    print(f"pre_filter_pool_recall = {result.pre_filter_pool_recall:.4f}")
    print(f"post_filter_pool_recall= {result.post_filter_pool_recall:.4f}")
    print(f"Recall@20/50/100       = {result.recall_at.get('20', 0.0):.4f}/{result.recall_at.get('50', 0.0):.4f}/{result.recall_at.get('100', 0.0):.4f}")
    if result.cutoff_year_recall_at:
        print(
            f"CutoffYear@20/50/100   = {result.cutoff_year_recall_at.get('20', 0.0):.4f}/"
            f"{result.cutoff_year_recall_at.get('50', 0.0):.4f}/"
            f"{result.cutoff_year_recall_at.get('100', 0.0):.4f} (approximate)"
        )
    print(f"gold matched/missed    = {len(result.matched_gold_titles)}/{len(result.missed_gold_titles)}")
    print(f"QA/QGen/Retrieval calls= {result.query_analysis_api_calls}/{result.query_generation_api_calls}/{result.retrieval_api_calls}")
    print(f"wall_seconds           = {result.total_wall_ms / 1000.0:.2f}")


def save_checkpoint(
    *,
    config: EvalConfig,
    enabled_sources: Sequence[SearchSource],
    results: Sequence[QueryRecallResult],
    output_path: Path,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "dataset_path": str(config.dataset_path),
        "range": {"start": config.start, "limit": config.limit},
        "routing_caps": {
            "max_routes": config.max_routes,
            "per_route_limit": config.per_route_limit,
            "max_total_hits": config.max_total_hits,
        },
        "enabled_sources": [source.value for source in enabled_sources],
        "results": [asdict(result) for result in results],
        "aggregate": aggregate(results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> int:
    config = EvalConfig.from_env()
    records = load_records(config.dataset_path)
    end = min(len(records), config.start + config.limit)
    selected = records[config.start:end]
    if not selected:
        raise ValueError(
            f"No records selected: start={config.start}, limit={config.limit}, dataset_size={len(records)}"
        )

    enabled_sources = get_enabled_sources()
    if not enabled_sources:
        raise RuntimeError("No retrieval source is enabled")
    adapters = build_adapters(enabled_sources, config)
    orchestrator = RetrievalOrchestrator(
        adapters,
        max_concurrency=config.max_concurrency,
        route_timeout_seconds=config.route_timeout_seconds,
        max_attempts=config.max_attempts,
        retry_backoff_seconds=config.retry_backoff_seconds,
        raise_if_all_failed=True,
    )
    analyzer = build_analyzer()
    generator = build_generator(config)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.results_dir / (
        f"realscholarquery_retrieval_{config.start}_{end - 1}_{stamp}.json"
    )

    print("=" * 88)
    print("PaSa RealScholarQuery -> Scholar Agent V2 Retrieval-Only Benchmark")
    print("=" * 88)
    print(f"dataset       = {config.dataset_path}")
    print(f"range         = [{config.start}, {end})")
    print(f"sources       = {[source.value for source in enabled_sources]}")
    print(
        "routing caps  = "
        f"routes:{config.max_routes} per_route:{config.per_route_limit} "
        f"total_hits:{config.max_total_hits}"
    )
    print(f"LLM expansion = {config.use_llm_expansion}")
    print("selector      = disabled")
    print("enrichment    = disabled")
    print("temporal mode = live index + year-only diagnostic")
    print(f"checkpoint    = {output_path}")

    results: list[QueryRecallResult] = []
    for offset, record in enumerate(selected, start=config.start):
        result = await evaluate_one(
            index=offset,
            record=record,
            config=config,
            analyzer=analyzer,
            generator=generator,
            orchestrator=orchestrator,
            enabled_sources=enabled_sources,
        )
        results.append(result)
        print_query_result(result)
        save_checkpoint(
            config=config,
            enabled_sources=enabled_sources,
            results=results,
            output_path=output_path,
        )
        if config.sleep_seconds > 0 and len(results) < len(selected):
            await asyncio.sleep(config.sleep_seconds)

    summary = aggregate(results)
    print("\n" + "=" * 88)
    print("FINAL RETRIEVAL-ONLY SUMMARY")
    print("=" * 88)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_path}")
    return 0 if summary["failed_queries"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
