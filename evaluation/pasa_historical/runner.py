"""Run the isolated PaSa historical open-source candidate-pool evaluation."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from scholar_agent_v2.retrieval.models import SearchSource  # noqa: E402
from scholar_agent_v2.retrieval.orchestrator import RetrievalOrchestrator  # noqa: E402

from evaluation.pasa_historical.candidate_merge import merge_hits  # noqa: E402
from evaluation.pasa_historical.config import HistoricalEvalConfig  # noqa: E402
from evaluation.pasa_historical.historical_retriever import (  # noqa: E402
    build_historical_adapters,
    collect_route_diagnostics,
)
from evaluation.pasa_historical.metrics import (  # noqa: E402
    HistoricalRecallResult,
    aggregate,
    comparison_row,
    evaluate_candidate_pool,
)
from evaluation.pasa_historical.protocol import build_plan, load_cases  # noqa: E402


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _route_json(execution, diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "route_id": execution.route_id,
        "source": execution.source.value,
        "status": execution.status.value,
        "hits": len(execution.hits),
        "api_calls": execution.api_calls,
        "attempts": execution.attempts,
        "latency_ms": execution.latency_ms,
        "warnings": execution.warnings,
        "error_type": execution.error_type,
        "error_message": execution.error_message,
        "historical_filter": diagnostics,
    }


def _load_optional_summary(path_value: str) -> dict[str, Any] | None:
    if not path_value.strip():
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Comparison report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Comparison report has no summary object: {path}")
    return summary


def _comparison(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [comparison_row("historical_open_sources", summary)]
    live = _load_optional_summary(os.getenv("PASA_HISTORICAL_LIVE_REPORT", ""))
    frozen = _load_optional_summary(os.getenv("PASA_HISTORICAL_FROZEN_REPORT", ""))
    if live is not None:
        rows.append(comparison_row("live_open_sources", live))
    if frozen is not None:
        rows.append(comparison_row("frozen_local_corpus", frozen))
    return rows


def _checkpoint_report(
    *,
    config: HistoricalEvalConfig,
    dataset_sha256: str,
    cache_protocol_sha256: str,
    sources: list[SearchSource],
    results: list[HistoricalRecallResult],
    complete: bool,
) -> dict[str, Any]:
    summary = aggregate(results)
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "complete": complete,
        "protocol": {
            "name": "pasa_historical_open_source_candidate_pool_v1",
            "dataset": str(config.dataset_path),
            "dataset_sha256": dataset_sha256,
            "query_cache": str(config.query_cache_path),
            "query_cache_protocol_sha256": cache_protocol_sha256,
            "cutoff_date": config.cutoff_date.isoformat(),
            "date_rule": "paper_publication_date < cutoff_date",
            "effective_last_date": "2024-09-30",
            "query_inputs": "original_plus_frozen_v2_generated_queries",
            "sources": [source.value for source in sources],
            "per_route_limit": config.per_route_limit,
            "openalex_filter": "to_publication_date:2024-09-30",
            "semantic_scholar_filter": (
                "year<=2024 then publicationDate<2024-10-01; paginate to refill"
            ),
            "arxiv_filter": "submittedDate <= 2024-09-30T23:59",
            "candidate_operation": "union_then_conservative_dedup",
            "reranking": False,
            "rrf": False,
            "selector": False,
            "gold_match": "exact_normalized_arxiv_id",
            "title_gold_fallback": False,
            "llm": False,
            "query_understanding_runtime": False,
            "query_generation_runtime": False,
        },
        "range": {
            "start": config.start,
            "requested_limit": config.limit,
            "completed_queries": len(results),
        },
        "summary": summary,
        "comparison": _comparison(summary),
        "results": [result.to_dict() for result in results],
    }


async def async_main() -> int:
    config = HistoricalEvalConfig.from_env()
    cases, dataset_sha256, cache_protocol_sha256 = load_cases(
        dataset_path=config.dataset_path,
        cache_path=config.query_cache_path,
        start=config.start,
        limit=config.limit,
    )
    adapters = build_historical_adapters(config)
    sources = list(adapters)
    orchestrator = RetrievalOrchestrator(
        adapters,
        max_concurrency=min(config.max_concurrency, max(1, len(adapters))),
        route_timeout_seconds=config.route_timeout_seconds,
        max_attempts=config.max_attempts,
        retry_backoff_seconds=config.retry_backoff_seconds,
        raise_if_all_failed=False,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    end_index = cases[-1].dataset_index
    output_path = config.output_path or (
        config.results_dir
        / f"historical_{config.start}_{end_index}_{stamp}.json"
    )

    print("=" * 92)
    print("PaSa Historical Evaluation: Frozen V2 Queries -> Historical Open Sources")
    print("=" * 92)
    print(f"dataset        = {config.dataset_path}")
    print(f"query_cache    = {config.query_cache_path}")
    print(f"cache_protocol = {cache_protocol_sha256}")
    print(f"cutoff         = paper_date < {config.cutoff_date.isoformat()}")
    print(f"sources        = {[source.value for source in sources]}")
    print(f"per_route_top  = {config.per_route_limit}")
    print("LLM/RRF/Selector = disabled")

    results: list[HistoricalRecallResult] = []
    for case in cases:
        plan = build_plan(
            case,
            sources=sources,
            per_route_limit=config.per_route_limit,
        )
        retrieval = await orchestrator.run(plan)
        executions = retrieval.executions
        hits = [hit for execution in executions for hit in execution.hits]
        candidates = merge_hits(hits)
        routes = [
            _route_json(
                execution,
                collect_route_diagnostics(adapters, execution.route_id),
            )
            for execution in executions
        ]
        result = evaluate_candidate_pool(
            case=case,
            candidates=candidates,
            raw_hits=len(hits),
            api_calls=sum(execution.api_calls for execution in executions),
            retrieval_latency_ms=retrieval.metrics.total_latency_ms,
            routes=routes,
            sources=[source.value for source in sources],
        )
        results.append(result)

        print("\n" + "-" * 92)
        print(f"[{case.dataset_index:02d}] {case.qid}: {case.question}")
        print(
            f"routes={len(plan.routes)} raw={result.raw_hits} "
            f"dedup={result.candidate_count} api_calls={result.api_calls}"
        )
        print(
            f"Gold={result.matched_gold_count}/{result.gold_count} "
            f"Recall={result.recall:.4f} latency_ms={result.retrieval_latency_ms:.2f}"
        )

        _atomic_write_json(
            output_path,
            _checkpoint_report(
                config=config,
                dataset_sha256=dataset_sha256,
                cache_protocol_sha256=cache_protocol_sha256,
                sources=sources,
                results=results,
                complete=False,
            ),
        )

    report = _checkpoint_report(
        config=config,
        dataset_sha256=dataset_sha256,
        cache_protocol_sha256=cache_protocol_sha256,
        sources=sources,
        results=results,
        complete=True,
    )
    _atomic_write_json(output_path, report)

    print("\n" + "=" * 92)
    print("FINAL HISTORICAL CANDIDATE-POOL SUMMARY")
    print("=" * 92)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_path}")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
