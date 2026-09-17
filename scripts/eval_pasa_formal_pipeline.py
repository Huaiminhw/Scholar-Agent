"""Run frozen PaSa A/B/C through the production retrieval pipeline.

This evaluator reads cached Query Understanding and Query Generation results,
then uses the same RouteBuilder, RetrievalOrchestrator, CandidatePool, hard
filters, and preranker used by the online pipeline.  Only the source adapter is
switched to PASA_LOCAL.  It never calls an LLM or a network source.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env", override=False)

from scripts.eval_pasa_frozen_v2_queries import (  # noqa: E402
    GoldPaper,
    gold_papers,
    load_cache,
    load_dataset,
    require_file,
    validate_selected_cache,
)
from scholar_agent_v2.query_generation.models import (  # noqa: E402
    ExpansionBudget,
    GeneratedQuery,
    QueryGenerationPlan,
    QueryGenerationResult,
    QueryKind,
)
from scholar_agent_v2.query_understanding.schema import QueryAnalysisResult  # noqa: E402
from scholar_agent_v2.query_understanding.schema import RetrievalMode  # noqa: E402
from scholar_agent_v2.candidate_pool.pool import CandidatePool  # noqa: E402
from scholar_agent_v2.retrieval.config import RetrievalProfile  # noqa: E402
from scholar_agent_v2.retrieval.citation_expansion import (  # noqa: E402
    CitationExpansionConfig,
)
from scholar_agent_v2.retrieval.factory import build_adapters  # noqa: E402
from scholar_agent_v2.retrieval.models import SearchSource  # noqa: E402
from scholar_agent_v2.retrieval.pipeline import ScholarRetrievalPipeline  # noqa: E402
from scholar_agent_v2.retrieval.route_builder import (  # noqa: E402
    MODE_BUDGETS,
    RetrievalModeBudget,
)
from scholar_agent_v2.retrieval.sources.pasa_local import PaSaLocalAdapter  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "data" / "RealScholarQuery" / "test.jsonl"
DEFAULT_CACHE = (
    PROJECT_ROOT / "results" / "pasa_v2_query_cache" / "smoke_v3_0_4.jsonl"
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "pasa_formal_pipeline"
TOP_K_VALUES = (20, 50, 100)
ROUND_TOP_K_VALUES = (40, 100)
ARM_KINDS: dict[str, set[QueryKind] | None] = {
    "A_original": set(),
    "B_deterministic": {QueryKind.EXACT, QueryKind.CORE, QueryKind.RELATION},
    "C_full": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen queries through the formal local PaSa pipeline."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--route-timeout", type=float, default=180.0)
    comprehensive = MODE_BUDGETS[RetrievalMode.COMPREHENSIVE]
    parser.add_argument(
        "--first-round-limit",
        type=int,
        default=comprehensive.first_round_limit,
        help="Per-route depth used by adaptive round 1.",
    )
    parser.add_argument(
        "--max-route-limit",
        type=int,
        default=comprehensive.max_route_limit,
        help="Maximum per-route depth for productive routes in round 2.",
    )
    parser.add_argument(
        "--min-new-candidate-rate",
        type=float,
        default=comprehensive.min_new_candidate_rate,
        help="Minimum de-duplicated round-1 yield required to deepen a route.",
    )
    parser.add_argument(
        "--citation-snowball",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable one-hop relevance-seeded citation expansion.",
    )
    parser.add_argument(
        "--citation-channel-weight",
        type=float,
        default=float(os.getenv("CITATION_CHANNEL_WEIGHT", "0.70")),
        help="Multiplier for citation-origin candidates during prerank.",
    )
    parser.add_argument("--citation-seed-top-k", type=int, default=3)
    parser.add_argument("--citation-forward-top-k", type=int, default=30)
    parser.add_argument("--citation-backward-top-k", type=int, default=30)
    parser.add_argument("--citation-fetch-limit", type=int, default=50)
    parser.add_argument("--citation-min-seed-score", type=float, default=0.20)
    parser.add_argument("--citation-min-new-rate", type=float, default=0.10)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ir(cache_record: Mapping[str, Any]) -> QueryAnalysisResult:
    raw = cache_record.get("academic_query_ir")
    if not isinstance(raw, Mapping):
        raw_understanding = cache_record.get("query_understanding") or {}
        raw = raw_understanding.get("result")
    return QueryAnalysisResult.model_validate(raw)


def _load_generation(cache_record: Mapping[str, Any]) -> QueryGenerationResult:
    raw_generation = cache_record.get("query_generation") or {}
    raw = raw_generation.get("result")
    return QueryGenerationResult.model_validate(raw)


def _arm_generation(
    base: QueryGenerationResult,
    analysis: QueryAnalysisResult,
    arm: str,
) -> QueryGenerationResult:
    allowed = ARM_KINDS[arm]
    queries = [
        GeneratedQuery(
            query_id="QO",
            text=analysis.original_query,
            kind=QueryKind.CORE,
            priority=1.0,
            reason="Original user query for the A/B/C retrieval ablation.",
        )
    ]
    if allowed is None:
        queries.extend(query.model_copy(deep=True) for query in base.queries)
    elif allowed:
        queries.extend(
            query.model_copy(deep=True)
            for query in base.queries
            if query.kind in allowed
        )

    counts = {kind: sum(query.kind == kind for query in queries) for kind in QueryKind}
    budget = ExpansionBudget(
        exact=counts[QueryKind.EXACT],
        core=counts[QueryKind.CORE],
        relation=counts[QueryKind.RELATION],
        alias=counts[QueryKind.ALIAS],
        expanded=counts[QueryKind.EXPANDED],
    )
    plan = QueryGenerationPlan(
        mode=base.plan.mode,
        budget=budget,
        use_llm=counts[QueryKind.EXPANDED] > 0 and base.llm_used,
        max_total_queries=max(1, len(queries)),
        include_original_content=True,
        reason=f"Formal local PaSa ablation arm {arm}.",
    )
    return QueryGenerationResult(
        original_query=analysis.original_query,
        core_content=(
            analysis.original_query if arm == "A_original" else base.core_content
        ),
        plan=plan,
        queries=queries,
        # Cached generation may have used an LLM, but this evaluation does not.
        llm_used=base.llm_used if arm == "C_full" else False,
        degraded=base.degraded,
        warnings=list(base.warnings),
    )


def _gold_matches(candidate_ids: Iterable[str | None], gold: Sequence[GoldPaper]) -> set[int]:
    ids = {value for value in candidate_ids if value}
    return {
        index
        for index, paper in enumerate(gold)
        if paper.arxiv_id is not None and paper.arxiv_id in ids
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


async def evaluate_arm(
    *,
    arm: str,
    analysis: QueryAnalysisResult,
    base_generation: QueryGenerationResult,
    pipeline: ScholarRetrievalPipeline,
    gold: Sequence[GoldPaper],
    available_gold_indices: set[int],
) -> dict[str, Any]:
    generation = _arm_generation(base_generation, analysis, arm)
    output = await pipeline.run(analysis, generation)
    pool = output.candidate_pool
    ranked = pool.top(len(pool.candidates))
    pool_matched = _gold_matches(
        (candidate.identifiers.arxiv_id for candidate in ranked),
        gold,
    )
    pre_citation_pool = CandidatePool.from_hits(output.retrieval.hits)
    pre_citation_pool.apply_hard_filters(analysis)
    pre_citation_pool.prerank(
        analysis,
        generation,
        num_enabled_sources=len(pipeline.enabled_sources),
        num_planned_routes=len(output.retrieval.plan.routes),
    )
    pre_citation_ranked = pre_citation_pool.top(
        len(pre_citation_pool.candidates)
    )
    pre_citation_matched = _gold_matches(
        (
            candidate.identifiers.arxiv_id
            for candidate in pre_citation_ranked
        ),
        gold,
    )

    matched_at: dict[str, set[int]] = {}
    for k in TOP_K_VALUES:
        matched_at[str(k)] = _gold_matches(
            (candidate.identifiers.arxiv_id for candidate in ranked[:k]),
            gold,
        )
    first_relevant_rank = next(
        (
            rank
            for rank, candidate in enumerate(ranked, start=1)
            if _gold_matches([candidate.identifiers.arxiv_id], gold)
        ),
        None,
    )

    round_reports: list[dict[str, Any]] = []
    for round_result in output.retrieval.rounds:
        round_pool = CandidatePool.from_hits(round_result.hits)
        round_pool.apply_hard_filters(analysis)
        round_pool.prerank(
            analysis,
            generation,
            num_enabled_sources=len(pipeline.enabled_sources),
            num_planned_routes=len(output.retrieval.plan.routes),
        )
        round_ranked = round_pool.top(len(round_pool.candidates))
        round_pool_matched = _gold_matches(
            (candidate.identifiers.arxiv_id for candidate in round_ranked),
            gold,
        )
        round_at = {
            str(k): _gold_matches(
                (
                    candidate.identifiers.arxiv_id
                    for candidate in round_ranked[:k]
                ),
                gold,
            )
            for k in ROUND_TOP_K_VALUES
        }
        round_reports.append(
            {
                **round_result.metrics.model_dump(mode="json"),
                "candidates_after_filter": len(round_ranked),
                "pool_matched_gold_count": len(round_pool_matched),
                "pool_matched_corpus_gold_count": len(
                    round_pool_matched & available_gold_indices
                ),
                "pool_recall": (
                    len(round_pool_matched) / len(gold) if gold else 0.0
                ),
                "pool_corpus_recall": (
                    len(round_pool_matched & available_gold_indices)
                    / len(available_gold_indices)
                    if available_gold_indices
                    else 0.0
                ),
                "matched_gold_count_at": {
                    key: len(value) for key, value in round_at.items()
                },
                "matched_corpus_gold_count_at": {
                    key: len(value & available_gold_indices)
                    for key, value in round_at.items()
                },
                "recall_at": {
                    key: len(value) / len(gold) if gold else 0.0
                    for key, value in round_at.items()
                },
                "corpus_recall_at": {
                    key: (
                        len(value & available_gold_indices)
                        / len(available_gold_indices)
                        if available_gold_indices
                        else 0.0
                    )
                    for key, value in round_at.items()
                },
            }
        )

    return {
        "arm": arm,
        "query_count": len(generation.queries),
        "queries": [query.model_dump(mode="json") for query in generation.queries],
        "planned_routes": len(output.retrieval.plan.routes),
        "executed_routes": output.retrieval.metrics.executed_routes,
        "retrieval_api_calls": output.retrieval.metrics.api_calls,
        "retrieval_latency_ms": output.retrieval.metrics.total_latency_ms,
        "retrieval_round_count": output.retrieval.metrics.retrieval_rounds,
        "retrieval_stop_reason": output.retrieval.metrics.stop_reason,
        "rounds": round_reports,
        "raw_hits": output.retrieval.metrics.raw_hits,
        "candidates_before_filter": pool.stats.candidates_before_filter,
        "candidates_after_filter": pool.stats.candidates_after_filter,
        "rejected_by_hard_filter": pool.stats.rejected_by_hard_filter,
        "unknown_after_filter": pool.stats.unknown_after_filter,
        "pool_matched_gold_count": len(pool_matched),
        "pool_matched_corpus_gold_count": len(pool_matched & available_gold_indices),
        "pool_recall": len(pool_matched) / len(gold) if gold else 0.0,
        "pool_corpus_recall": (
            len(pool_matched & available_gold_indices) / len(available_gold_indices)
            if available_gold_indices
            else 0.0
        ),
        "pre_citation_pool_matched_gold_count": len(pre_citation_matched),
        "pre_citation_pool_recall": (
            len(pre_citation_matched) / len(gold) if gold else 0.0
        ),
        "citation_added_gold_count": len(pool_matched - pre_citation_matched),
        "citation_pool_recall_gain": (
            (len(pool_matched) - len(pre_citation_matched)) / len(gold)
            if gold
            else 0.0
        ),
        "citation_expansion": output.citation_expansion.diagnostics.model_dump(
            mode="json"
        ),
        "matched_gold_count_at": {
            key: len(value) for key, value in matched_at.items()
        },
        "matched_corpus_gold_count_at": {
            key: len(value & available_gold_indices)
            for key, value in matched_at.items()
        },
        "recall_at": {
            key: len(value) / len(gold) if gold else 0.0
            for key, value in matched_at.items()
        },
        "corpus_recall_at": {
            key: (
                len(value & available_gold_indices) / len(available_gold_indices)
                if available_gold_indices
                else 0.0
            )
            for key, value in matched_at.items()
        },
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "score_component_means": {
            component: _mean(
                candidate.score_components.get(component, 0.0)
                for candidate in ranked[:100]
            )
            for component in (
                "rrf",
                "lexical",
                "source_coverage",
                "route_coverage",
                "soft_match",
                "preference",
            )
        },
        "top_candidates": [
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "arxiv_id": candidate.identifiers.arxiv_id,
                "title": candidate.title,
                "year": candidate.year,
                "retrieval_score": candidate.retrieval_score,
                "score_components": candidate.score_components,
                "route_count": candidate.route_count,
                "source_count": candidate.source_count,
            }
            for rank, candidate in enumerate(ranked[:100], start=1)
        ],
        "matched_indices_at": {
            key: sorted(value) for key, value in matched_at.items()
        },
        "warnings": list(output.retrieval.warnings) + list(pool.warnings),
    }


async def evaluate_query(
    *,
    index: int,
    source_record: Mapping[str, Any],
    cache_record: Mapping[str, Any],
    pipeline: ScholarRetrievalPipeline,
    local_adapter: PaSaLocalAdapter,
) -> dict[str, Any]:
    analysis = _load_ir(cache_record)
    generation = _load_generation(cache_record)
    gold = gold_papers(source_record)
    available = {
        position
        for position, paper in enumerate(gold)
        if paper.arxiv_id
        and local_adapter.corpus.get_by_arxiv_id(paper.arxiv_id) is not None
    }
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARM_KINDS:
        arms[arm] = await evaluate_arm(
            arm=arm,
            analysis=analysis,
            base_generation=generation,
            pipeline=pipeline,
            gold=gold,
            available_gold_indices=available,
        )

    baseline = arms["A_original"]
    for arm, result in arms.items():
        result["marginal_gold_hits_at"] = {
            str(k): len(
                set(result["matched_indices_at"][str(k)])
                - set(baseline["matched_indices_at"][str(k)])
            )
            for k in TOP_K_VALUES
        }
        result["candidate_inflation_vs_original"] = (
            (result["candidates_after_filter"] - baseline["candidates_after_filter"])
            / baseline["candidates_after_filter"]
            if baseline["candidates_after_filter"]
            else 0.0
        )

    return {
        "dataset_index": index,
        "qid": str(source_record.get("qid", f"RealScholarQuery_{index}")),
        "question": str(source_record["question"]),
        "gold_count": len(gold),
        "corpus_gold_count": len(available),
        "corpus_coverage": len(available) / len(gold) if gold else 0.0,
        "cache_created_at": cache_record.get("created_at"),
        "arms": arms,
    }


def aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_gold = sum(int(result["gold_count"]) for result in results)
    total_corpus_gold = sum(int(result["corpus_gold_count"]) for result in results)
    arms: dict[str, Any] = {}
    for arm in ARM_KINDS:
        total_at = {
            str(k): sum(
                int(result["arms"][arm]["matched_gold_count_at"][str(k)])
                for result in results
            )
            for k in TOP_K_VALUES
        }
        corpus_total_at = {
            str(k): sum(
                int(result["arms"][arm]["matched_corpus_gold_count_at"][str(k)])
                for result in results
            )
            for k in TOP_K_VALUES
        }
        arms[arm] = {
            "micro_recall_at": {
                str(k): total_at[str(k)] / total_gold if total_gold else 0.0
                for k in TOP_K_VALUES
            },
            "micro_corpus_recall_at": {
                str(k): (
                    corpus_total_at[str(k)] / total_corpus_gold
                    if total_corpus_gold
                    else 0.0
                )
                for k in TOP_K_VALUES
            },
            "mean_mrr": _mean(float(result["arms"][arm]["mrr"]) for result in results),
            "micro_pool_recall": (
                sum(
                    int(result["arms"][arm]["pool_matched_gold_count"])
                    for result in results
                )
                / total_gold
                if total_gold
                else 0.0
            ),
            "micro_pool_corpus_recall": (
                sum(
                    int(result["arms"][arm]["pool_matched_corpus_gold_count"])
                    for result in results
                )
                / total_corpus_gold
                if total_corpus_gold
                else 0.0
            ),
            "mean_candidates_after_filter": _mean(
                float(result["arms"][arm]["candidates_after_filter"])
                for result in results
            ),
            "mean_candidate_inflation_vs_original": _mean(
                float(result["arms"][arm]["candidate_inflation_vs_original"])
                for result in results
            ),
            "total_marginal_gold_hits_at": {
                str(k): sum(
                    int(result["arms"][arm]["marginal_gold_hits_at"][str(k)])
                    for result in results
                )
                for k in TOP_K_VALUES
            },
            "total_retrieval_latency_ms": sum(
                float(result["arms"][arm]["retrieval_latency_ms"])
                for result in results
            ),
            "micro_pre_citation_pool_recall": (
                sum(
                    int(
                        result["arms"][arm][
                            "pre_citation_pool_matched_gold_count"
                        ]
                    )
                    for result in results
                )
                / total_gold
                if total_gold
                else 0.0
            ),
            "total_citation_added_gold": sum(
                int(result["arms"][arm]["citation_added_gold_count"])
                for result in results
            ),
            "total_citation_api_calls": sum(
                int(result["arms"][arm]["citation_expansion"]["api_calls"])
                for result in results
            ),
            "total_citation_latency_ms": sum(
                float(result["arms"][arm]["citation_expansion"]["latency_ms"])
                for result in results
            ),
            "citation_stop_reasons": {
                reason: sum(
                    result["arms"][arm]["citation_expansion"]["stop_reason"]
                    == reason
                    for result in results
                )
                for reason in sorted(
                    {
                        str(
                            result["arms"][arm]["citation_expansion"][
                                "stop_reason"
                            ]
                        )
                        for result in results
                    }
                )
            },
        }
        round_summaries: dict[str, Any] = {}
        for label, selector in (
            ("round_1", lambda rounds: rounds[0]),
            ("round_2_or_final", lambda rounds: rounds[-1]),
        ):
            selected = [
                selector(result["arms"][arm]["rounds"])
                for result in results
                if result["arms"][arm]["rounds"]
            ]
            round_summaries[label] = {
                "query_count": len(selected),
                "actually_executed_round_2": sum(
                    int(item["round_number"] == 2) for item in selected
                ),
                "micro_pool_recall": (
                    sum(int(item["pool_matched_gold_count"]) for item in selected)
                    / total_gold
                    if total_gold
                    else 0.0
                ),
                "micro_pool_corpus_recall": (
                    sum(
                        int(item["pool_matched_corpus_gold_count"])
                        for item in selected
                    )
                    / total_corpus_gold
                    if total_corpus_gold
                    else 0.0
                ),
                "micro_recall_at": {
                    str(k): (
                        sum(
                            int(item["matched_gold_count_at"][str(k)])
                            for item in selected
                        )
                        / total_gold
                        if total_gold
                        else 0.0
                    )
                    for k in ROUND_TOP_K_VALUES
                },
                "mean_new_candidate_rate": _mean(
                    float(item["new_candidate_rate"]) for item in selected
                ),
                "mean_duplicate_rate": _mean(
                    float(item["duplicate_rate"]) for item in selected
                ),
                "total_latency_ms": sum(
                    float(item["latency_ms"]) for item in selected
                ),
                "stop_reasons": {
                    reason: sum(item["stop_reason"] == reason for item in selected)
                    for reason in sorted(
                        {
                            str(item["stop_reason"])
                            for item in selected
                            if item["stop_reason"] is not None
                        }
                    )
                },
            }
        arms[arm]["rounds"] = round_summaries
    return {
        "query_count": len(results),
        "gold_total": total_gold,
        "corpus_gold_total": total_corpus_gold,
        "corpus_coverage": total_corpus_gold / total_gold if total_gold else 0.0,
        "arms": arms,
    }


async def async_main() -> int:
    args = parse_args()
    if args.start < 0 or args.limit <= 0:
        raise ValueError("--start must be >= 0 and --limit must be > 0")
    if args.max_concurrency <= 0 or args.route_timeout <= 0:
        raise ValueError("concurrency and timeout must be positive")
    if not 1 <= args.first_round_limit < args.max_route_limit <= 100:
        raise ValueError(
            "adaptive limits must satisfy 1 <= first-round-limit < "
            "max-route-limit <= 100"
        )
    if not 0.0 <= args.min_new_candidate_rate <= 1.0:
        raise ValueError("--min-new-candidate-rate must be between 0 and 1")
    if min(
        args.citation_seed_top_k,
        args.citation_forward_top_k,
        args.citation_backward_top_k,
        args.citation_fetch_limit,
    ) < 0:
        raise ValueError("citation budgets must be non-negative")
    if not 0.0 <= args.citation_min_seed_score <= 1.0:
        raise ValueError("--citation-min-seed-score must be between 0 and 1")
    if not 0.0 <= args.citation_min_new_rate <= 1.0:
        raise ValueError("--citation-min-new-rate must be between 0 and 1")
    if not 0.0 <= args.citation_channel_weight <= 1.0:
        raise ValueError("--citation-channel-weight must be between 0 and 1")

    dataset_path = require_file(args.dataset, "RealScholarQuery dataset")
    cache_path = require_file(args.cache, "Frozen query cache")
    dataset = load_dataset(dataset_path)
    cache = load_cache(cache_path)
    end = min(len(dataset), args.start + args.limit)
    indices = list(range(args.start, end))
    if not indices:
        raise ValueError("No dataset records selected")
    dataset_sha = _sha256(dataset_path)
    cache_protocol = validate_selected_cache(
        dataset_records=dataset,
        cache_records=cache,
        selected_indices=indices,
        dataset_sha256=dataset_sha,
    )

    adapters = build_adapters(
        [SearchSource.PASA_LOCAL],
        profile=RetrievalProfile.FROZEN_LOCAL,
        timeout_seconds=args.route_timeout,
    )
    local_adapter = adapters[SearchSource.PASA_LOCAL]
    assert isinstance(local_adapter, PaSaLocalAdapter)
    citation_graph_ready = local_adapter.citation_graph.is_ready()
    citation_graph_metadata = (
        local_adapter.citation_graph.metadata() if citation_graph_ready else {}
    )
    citation_graph_sha = (
        _sha256(local_adapter.citation_graph.graph_path)
        if args.citation_snowball and citation_graph_ready
        else None
    )
    mode_budgets: dict[RetrievalMode, RetrievalModeBudget] = dict(MODE_BUDGETS)
    mode_budgets[RetrievalMode.COMPREHENSIVE] = replace(
        mode_budgets[RetrievalMode.COMPREHENSIVE],
        first_round_limit=args.first_round_limit,
        max_route_limit=args.max_route_limit,
        min_new_candidate_rate=args.min_new_candidate_rate,
    )
    pipeline = ScholarRetrievalPipeline(
        enabled_sources=[SearchSource.PASA_LOCAL],
        adapters=adapters,
        max_concurrency=args.max_concurrency,
        route_timeout_seconds=args.route_timeout,
        max_attempts=1,
        mode_budgets=mode_budgets,
        citation_config=CitationExpansionConfig(
            enabled=args.citation_snowball,
            channel_weight=args.citation_channel_weight,
            seed_top_k=args.citation_seed_top_k,
            forward_top_k=args.citation_forward_top_k,
            backward_top_k=args.citation_backward_top_k,
            fetch_limit_per_seed=args.citation_fetch_limit,
            min_seed_score=args.citation_min_seed_score,
            min_new_candidate_rate=args.citation_min_new_rate,
            max_concurrency=args.max_concurrency,
        ),
    )

    print("=" * 88)
    print("PaSa Formal Retrieval Pipeline: frozen cache + PASA_LOCAL")
    print("=" * 88)
    print(f"cache             = {cache_path}")
    print(f"range             = [{args.start}, {end})")
    print("profile           = frozen_local")
    print("source            = pasa_local")
    print("pipeline          = route -> orchestrator -> pool -> filter -> prerank")
    print("LLM/network       = disabled")
    print("selector          = not run (retrieval Recall evaluation)")
    print(
        "adaptive budget   = "
        f"{args.first_round_limit}->{args.max_route_limit}, "
        f"min_new_rate={args.min_new_candidate_rate:.3f}"
    )
    print(
        "citation snowball = "
        f"{args.citation_snowball}, seeds={args.citation_seed_top_k}, "
        f"forward={args.citation_forward_top_k}, "
        f"backward={args.citation_backward_top_k}, "
        f"channel_weight={args.citation_channel_weight:.2f}"
    )
    print(f"citation graph    = ready:{citation_graph_ready}")

    results: list[dict[str, Any]] = []
    for index in indices:
        result = await evaluate_query(
            index=index,
            source_record=dataset[index],
            cache_record=cache[index],
            pipeline=pipeline,
            local_adapter=local_adapter,
        )
        results.append(result)
        print(f"\n#{index} {result['qid']} gold={result['gold_count']}")
        for arm in ARM_KINDS:
            item = result["arms"][arm]
            print(
                f"  {arm:<16} routes={item['executed_routes']} "
                f"candidates={item['candidates_after_filter']} "
                f"R@20/50/100="
                f"{item['recall_at']['20']:.4f}/"
                f"{item['recall_at']['50']:.4f}/"
                f"{item['recall_at']['100']:.4f} "
                f"PoolR={item['pool_recall']:.4f} MRR={item['mrr']:.4f}"
            )
            citation = item["citation_expansion"]
            print(
                f"    citation seeds={len(citation['seed_candidate_ids'])} "
                f"new={citation['new_unique_candidates']} "
                f"new_rate={citation['new_candidate_rate']:.4f} "
                f"dup_rate={citation['duplicate_rate']:.4f} "
                f"PoolR={item['pre_citation_pool_recall']:.4f}"
                f"->{item['pool_recall']:.4f} "
                f"added_gold={item['citation_added_gold_count']} "
                f"stop={citation['stop_reason']}"
            )
            for round_item in item["rounds"]:
                print(
                    f"    round={round_item['round_number']} "
                    f"new={round_item['added_unique_candidates']} "
                    f"new_rate={round_item['new_candidate_rate']:.4f} "
                    f"dup_rate={round_item['duplicate_rate']:.4f} "
                    f"PoolR={round_item['pool_recall']:.4f} "
                    f"Top40/100={round_item['recall_at']['40']:.4f}/"
                    f"{round_item['recall_at']['100']:.4f} "
                    f"latency_ms={round_item['latency_ms']:.2f} "
                    f"stop={round_item['stop_reason']}"
                )

    summary = aggregate(results)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else args.results_dir.expanduser().resolve()
        / f"formal_local_{args.start}_{end - 1}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "name": "pasa_formal_retrieval_pipeline_v3_citation_snowball",
            "retrieval_profile": RetrievalProfile.FROZEN_LOCAL.value,
            "query_execution_mode": "frozen",
            "dataset": str(dataset_path),
            "dataset_sha256": dataset_sha,
            "query_cache": str(cache_path),
            "query_cache_protocol_sha256": cache_protocol,
            "enabled_sources": [SearchSource.PASA_LOCAL.value],
            "shared_pipeline_stages": [
                "RetrievalRouteBuilder",
                "RetrievalOrchestrator",
                "CandidatePool.from_hits",
                "CandidatePool.apply_hard_filters",
                "CandidatePool.prerank",
            ],
            "selector_runtime": False,
            "selector_reason": "Gold Recall is measured before optional semantic selection.",
            "llm_runtime": False,
            "network": False,
            "gold_used_for_retrieval_or_ranking": False,
            "adaptive_depth_budget": {
                "first_round_limit": args.first_round_limit,
                "max_route_limit": args.max_route_limit,
                "min_new_candidate_rate": args.min_new_candidate_rate,
            },
            "citation_snowball": {
                "enabled": args.citation_snowball,
                "channel_weight": args.citation_channel_weight,
                "seed_top_k": args.citation_seed_top_k,
                "forward_top_k": args.citation_forward_top_k,
                "backward_top_k": args.citation_backward_top_k,
                "fetch_limit_per_seed": args.citation_fetch_limit,
                "min_seed_score": args.citation_min_seed_score,
                "min_new_candidate_rate": args.citation_min_new_rate,
                "max_hops": 1,
                "graph_path": str(local_adapter.citation_graph.graph_path),
                "graph_ready": citation_graph_ready,
                "graph_sha256": citation_graph_sha,
                "graph_metadata": citation_graph_metadata,
            },
        },
        "range": {"start": args.start, "end_exclusive": end},
        "index_metadata": local_adapter.corpus.index_metadata(),
        "summary": summary,
        "results": results,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 88)
    print("FORMAL LOCAL SUMMARY")
    print("=" * 88)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to: {output_path}")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
