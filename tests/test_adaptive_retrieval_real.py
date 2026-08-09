# tests/test_adaptive_retrieval_real.py
"""
Static Routing vs Adaptive Routing 真实 A/B 集成验收。

设计目标
--------
1. 公平 A/B：
   - 每个 Case 只调用一次 Query Generation LLM；
   - Static / Adaptive 共享完全相同的 QueryAnalysisResult、QueryGenerationOutput、
     Native RetrievalPlan；
   - 两边使用相同 max routes / per-route limit；
   - 唯一核心变量是 Route Selection / Execution policy。

2. 不把实时网络波动误判成算法失败：
   - 默认只做结构、预算、数据自洽等硬断言；
   - Adaptive 是否“优于”Static 默认只打印比较，不作为硬断言；
   - 稳定重复若干轮后，可开启 ADAPTIVE_AB_STRICT=1 做非退化验收。

3. 比赛导向：
   - 记录 Route/API/Retry/Latency；
   - 记录 raw hits / unique candidates / duplicate ratio；
   - 记录 unique candidates per retrieval API call；
   - 记录 Candidate Pool Top-K 的 BGE routing-semantic proxy；
   - Adaptive 额外打印 probe routes、source yield、expected gain、
     unique credit、semantic gain、semantic scoring latency。

4. 后续无需改文件即可扩展：
   - 默认只跑 Complex Relation；
   - ADAPTIVE_AB_CASES=all 可跑 complex / or_exclude / broad 三组；
   - BGE device/batch/seq_len、strict 阈值均可由环境变量控制。

注意
----
- 这里没有 gold relevance labels，因此不能把任何 proxy 称为真实 Precision/Recall/F1。
- BGE semantic proxy 与 Adaptive Routing 使用同一 scorer，只用于诊断，不是独立评测器。
- Static baseline 继续沿用 test_retrieval_pipeline_real.py 中的 source caps；
  当前 AdaptiveRouterConfig 尚未实现 source_route_caps，因此 Adaptive 只受总 Route 数
  和 per-route limit 约束。这一差异会明确打印，后续若生产代码加入 source caps，
  再把两边约束统一即可。

推荐运行
--------
# Linux / 远程服务器
RUN_REAL_ADAPTIVE_RETRIEVAL_TESTS=1 \
python -m pytest -s tests/test_adaptive_retrieval_real.py

# Windows PowerShell
$env:RUN_REAL_ADAPTIVE_RETRIEVAL_TESTS="1"
python -m pytest -s tests/test_adaptive_retrieval_real.py

# 跑全部 3 个 case
$env:ADAPTIVE_AB_CASES="all"
python -m pytest -s tests/test_adaptive_retrieval_real.py

# 稳定重复验证后，再开启严格非退化门槛
$env:ADAPTIVE_AB_STRICT="1"
python -m pytest -s tests/test_adaptive_retrieval_real.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import pytest

from scholar_agent_v2.candidate_pool.dedup import normalize_title
from scholar_agent_v2.candidate_pool.pool import CandidatePool
from scholar_agent_v2.query_generation.models import QueryGenerationOutput
from scholar_agent_v2.query_understanding.schema import QueryAnalysisResult
from scholar_agent_v2.retrieval.adaptive import (
    AdaptiveRetrievalResult,
    AdaptiveRoutePlanner,
    AdaptiveRouterConfig,
    SemanticScorer,
)
from scholar_agent_v2.retrieval.config import get_enabled_sources
from scholar_agent_v2.retrieval.models import RetrievalOutput, RetrievalPlan
from scholar_agent_v2.retrieval.route_builder import RetrievalRouteBuilder

# 复用原真实 Pipeline 的“同一套”构建器、预算、健康断言与打印函数，
# 避免 A/B 测试复制出第二套标准，后续维护时两边漂移。
from tests.test_retrieval_pipeline_real import (
    ALL_CASES,
    MAX_EXEC_ROUTES,
    MAX_RETRIEVAL_API_CALLS,
    PER_ROUTE_LIMIT,
    SOURCE_ROUTE_CAPS,
    TOP_N,
    PipelineCase,
    _assert_candidate_pool_health,
    _assert_execution_plan_coverage,
    _assert_generation_safety,
    _assert_native_plan_semantics,
    _assert_retrieval_health,
    _build_adapters,
    _build_generator,
    _build_orchestrator,
    _make_execution_plan,
    _print_generation,
    _print_plan,
    _print_pool,
    _print_retrieval,
)


# =============================================================================
# Runtime configuration
# =============================================================================


RUN_REAL = os.getenv("RUN_REAL_ADAPTIVE_RETRIEVAL_TESTS", "0") == "1"
CASE_SELECTOR = os.getenv("ADAPTIVE_AB_CASES", "complex").strip().lower()

BGE_DEVICE = os.getenv("ADAPTIVE_BGE_DEVICE", "").strip()
BGE_BATCH_SIZE = int(os.getenv("ADAPTIVE_BGE_BATCH_SIZE", "16"))
BGE_MAX_SEQ_LENGTH = int(os.getenv("ADAPTIVE_BGE_MAX_SEQ_LENGTH", "512"))

ADAPTIVE_PROBE_ROUTES = int(os.getenv("ADAPTIVE_PROBE_ROUTES", "3"))
AB_ORDER = os.getenv("ADAPTIVE_AB_ORDER", "static_first").strip().lower()
AB_PAUSE_SECONDS = float(os.getenv("ADAPTIVE_AB_PAUSE_SECONDS", "0"))

STRICT = os.getenv("ADAPTIVE_AB_STRICT", "0") == "1"
STRICT_MIN_UNIQUE_RETENTION = float(
    os.getenv("ADAPTIVE_AB_MIN_UNIQUE_RETENTION", "0.90")
)
STRICT_MAX_API_CALL_DELTA = int(
    os.getenv("ADAPTIVE_AB_MAX_API_CALL_DELTA", "1")
)

SEMANTIC_TOP_K = int(os.getenv("ADAPTIVE_AB_SEMANTIC_TOP_K", "10"))


# =============================================================================
# Case selection
# =============================================================================


def _selected_cases() -> list[PipelineCase]:
    if CASE_SELECTOR in {"all", "*"}:
        return list(ALL_CASES.values())

    names = [x.strip() for x in CASE_SELECTOR.split(",") if x.strip()]
    unknown = [x for x in names if x not in ALL_CASES]
    if unknown:
        raise ValueError(
            f"Unknown ADAPTIVE_AB_CASES={unknown}; "
            f"valid={list(ALL_CASES)} or all"
        )
    return [ALL_CASES[name] for name in names]


# =============================================================================
# BGE scorer
# =============================================================================


@pytest.fixture(scope="session")
def semantic_scorer() -> SemanticScorer:
    """Load local BGE-M3 once for the whole A/B test session.

    Warmup is intentionally done before timing either A/B arm, so model load
    does not contaminate Adaptive routing latency.
    """
    try:
        import torch
    except ImportError as exc:
        pytest.fail(f"PyTorch is required for real Adaptive A/B: {exc}")

    device = BGE_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

    scorer = SemanticScorer(
        device=device,
        max_seq_length=BGE_MAX_SEQ_LENGTH,
        batch_size=BGE_BATCH_SIZE,
    )

    print("\n========== LOCAL BGE-M3 ==========")
    print(f"model_path          = {scorer.model_name}")
    print(f"device              = {device}")
    print(f"batch_size          = {BGE_BATCH_SIZE}")
    print(f"max_seq_length      = {BGE_MAX_SEQ_LENGTH}")
    if torch.cuda.is_available():
        print(f"cuda_device         = {torch.cuda.get_device_name(0)}")

    started = time.perf_counter()
    scorer.warmup()
    warmup_ms = (time.perf_counter() - started) * 1000.0
    print(f"warmup_ms           = {warmup_ms:.2f}")

    return scorer


# =============================================================================
# Result containers / metrics
# =============================================================================


@dataclass
class ArmResult:
    label: str
    retrieval: RetrievalOutput
    pool: CandidatePool
    wall_latency_ms: float
    routing_overhead_ms: float = 0.0
    adaptive: AdaptiveRetrievalResult | None = None


def _candidate_text(candidate) -> str:
    return SemanticScorer.paper_text(
        getattr(candidate, "title", None),
        getattr(candidate, "abstract", None),
    )


def _candidate_title_set(pool: CandidatePool) -> set[str]:
    return {
        title
        for candidate in pool.candidates
        if (title := normalize_title(getattr(candidate, "title", None)))
    }


def _duplicate_ratio(raw_hits: int, unique_candidates: int) -> float:
    if raw_hits <= 0:
        return 0.0
    return max(0.0, 1.0 - unique_candidates / raw_hits)


def _semantic_proxy(
    pool: CandidatePool,
    analysis: QueryAnalysisResult,
    scorer: SemanticScorer,
    *,
    top_k: int,
) -> dict[str, float | int]:
    """BGE semantic diagnostic on already-preranked Candidate Pool.

    This is NOT independent relevance evaluation because the same BGE scorer
    participates in Adaptive routing. It is only a useful route-quality proxy.
    """
    candidates = pool.top(min(top_k, len(pool.candidates)))
    if not candidates:
        return {
            "top_k": 0,
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    query_text = (analysis.content or analysis.original_query).strip()
    texts = [_candidate_text(candidate) for candidate in candidates]
    valid = [text for text in texts if text]
    if not valid:
        return {
            "top_k": 0,
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    query_emb = scorer.encode([query_text])[0]
    paper_embs = scorer.encode(valid)
    sims = paper_embs @ query_emb

    return {
        "top_k": len(valid),
        "avg": float(sims.mean()),
        "min": float(sims.min()),
        "max": float(sims.max()),
    }


def _arm_summary(
    arm: ArmResult,
    analysis: QueryAnalysisResult,
    scorer: SemanticScorer,
) -> dict[str, Any]:
    retrieval = arm.retrieval
    pool = arm.pool
    m = retrieval.metrics

    unique_candidates = pool.stats.candidates_before_filter
    retrieval_api_calls = m.api_calls

    return {
        "routes": m.executed_routes,
        "successful_routes": m.successful_routes,
        "empty_routes": m.empty_routes,
        "failed_routes": m.failed_routes,
        "timeout_routes": m.timeout_routes,
        "retrieval_api_calls": retrieval_api_calls,
        "retries": m.retries,
        "raw_hits": m.raw_hits,
        "unique_candidates": unique_candidates,
        "duplicate_ratio": _duplicate_ratio(
            m.raw_hits,
            unique_candidates,
        ),
        "unique_per_retrieval_api_call": (
            unique_candidates / retrieval_api_calls
            if retrieval_api_calls > 0
            else 0.0
        ),
        "hits_per_retrieval_api_call": (
            m.raw_hits / retrieval_api_calls
            if retrieval_api_calls > 0
            else 0.0
        ),
        "retrieval_network_ms": m.total_latency_ms,
        "routing_overhead_ms": arm.routing_overhead_ms,
        "arm_wall_ms": arm.wall_latency_ms,
        "semantic_proxy": _semantic_proxy(
            pool,
            analysis,
            scorer,
            top_k=SEMANTIC_TOP_K,
        ),
        "source_hit_counts": m.source_hit_counts,
        "source_api_calls": m.source_api_calls,
    }


def _delta(adaptive: float, static: float) -> float:
    return adaptive - static


def _pct_change(adaptive: float, static: float) -> float | None:
    if abs(static) <= 1e-12:
        return None
    return 100.0 * (adaptive - static) / static


# =============================================================================
# Candidate Pool
# =============================================================================


def _build_pool(
    retrieval: RetrievalOutput,
    analysis: QueryAnalysisResult,
    generation: QueryGenerationOutput,
    *,
    num_enabled_sources: int,
) -> CandidatePool:
    pool = CandidatePool.from_hits(retrieval.hits)
    pool.apply_hard_filters(analysis)
    pool.prerank(
        analysis,
        generation,
        num_enabled_sources=num_enabled_sources,
        num_planned_routes=len(retrieval.plan.routes),
    )
    _assert_candidate_pool_health(pool)
    return pool


# =============================================================================
# A/B arms
# =============================================================================


async def _run_static_arm(
    *,
    native_plan: RetrievalPlan,
    analysis: QueryAnalysisResult,
    generation: QueryGenerationOutput,
    enabled_sources,
) -> ArmResult:
    execution_plan = _make_execution_plan(native_plan)
    _assert_execution_plan_coverage(
        native_plan,
        execution_plan,
        enabled_sources,
    )

    print("\n\n############################################################")
    print("# STATIC ARM")
    print("############################################################")
    _print_plan("STATIC EXECUTION PLAN", execution_plan)

    adapters = _build_adapters(enabled_sources)
    orchestrator = _build_orchestrator(adapters)

    started = time.perf_counter()
    retrieval = await orchestrator.run(execution_plan)
    wall_ms = (time.perf_counter() - started) * 1000.0

    _assert_retrieval_health(retrieval, execution_plan)
    _print_retrieval(retrieval)

    pool = _build_pool(
        retrieval,
        analysis,
        generation,
        num_enabled_sources=len(enabled_sources),
    )
    _print_pool(pool, TOP_N)

    return ArmResult(
        label="static",
        retrieval=retrieval,
        pool=pool,
        wall_latency_ms=wall_ms,
    )


async def _run_adaptive_arm(
    *,
    native_plan: RetrievalPlan,
    analysis: QueryAnalysisResult,
    generation: QueryGenerationOutput,
    enabled_sources,
    scorer: SemanticScorer,
) -> ArmResult:
    print("\n\n############################################################")
    print("# ADAPTIVE ARM")
    print("############################################################")

    adapters = _build_adapters(enabled_sources)
    orchestrator = _build_orchestrator(adapters)
    planner = AdaptiveRoutePlanner(
        orchestrator,
        scorer,
        config=AdaptiveRouterConfig(
            max_exec_routes=MAX_EXEC_ROUTES,
            probe_routes=ADAPTIVE_PROBE_ROUTES,
            per_route_limit=PER_ROUTE_LIMIT,
        ),
    )

    started = time.perf_counter()
    adaptive_result = await planner.run(native_plan, analysis)
    wall_ms = (time.perf_counter() - started) * 1000.0
    retrieval = adaptive_result.retrieval

    # Adaptive final plan itself is the actual execution plan.
    assert 1 <= len(retrieval.plan.routes) <= MAX_EXEC_ROUTES
    assert all(route.limit <= PER_ROUTE_LIMIT for route in retrieval.plan.routes)
    _assert_retrieval_health(retrieval, retrieval.plan)

    _print_plan("ADAPTIVE FINAL EXECUTION PLAN", retrieval.plan)
    _print_adaptive_diagnostics(adaptive_result)
    _print_retrieval(retrieval)

    pool = _build_pool(
        retrieval,
        analysis,
        generation,
        num_enabled_sources=len(enabled_sources),
    )
    _print_pool(pool, TOP_N)

    return ArmResult(
        label="adaptive",
        retrieval=retrieval,
        pool=pool,
        wall_latency_ms=wall_ms,
        routing_overhead_ms=adaptive_result.diagnostics.semantic_scoring_ms,
        adaptive=adaptive_result,
    )


# =============================================================================
# Diagnostics
# =============================================================================


def _print_adaptive_diagnostics(result: AdaptiveRetrievalResult) -> None:
    d = result.diagnostics

    print("\n========== ADAPTIVE ROUTING DIAGNOSTICS ==========")
    print(f"probe_route_ids           = {d.probe_route_ids}")
    print(f"adaptive_route_ids        = {d.adaptive_route_ids}")
    print(f"source_yield              = {json.dumps(d.source_yield, ensure_ascii=False)}")
    print(
        "expected_gain             = "
        + json.dumps(
            {
                k: round(v, 6)
                for k, v in sorted(
                    d.expected_gain.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            ensure_ascii=False,
        )
    )
    print(f"semantic_scoring_ms       = {d.semantic_scoring_ms:.2f}")
    print(f"total_adaptive_latency_ms = {d.total_adaptive_latency_ms:.2f}")

    print("\n---------- PROBE FEATURES ----------")
    for route_id in d.probe_route_ids:
        f = d.probe_features.get(route_id)
        if f is None:
            print(f"[{route_id}] <NO FEATURES>")
            continue
        print(
            f"[{route_id}] source={f.source.value:<18} "
            f"hits={f.hit_count:<3} "
            f"avg_sim={f.avg_sim:.4f} "
            f"max_sim={f.max_sim:.4f} "
            f"var_sim={f.var_sim:.6f}"
        )
        print(
            f"    overall_sim={f.overall_sim:.4f} "
            f"cross_retriever_sim={f.cross_retriever_sim:.4f} "
            f"unique_credit={f.unique_credit:.4f} "
            f"semantic_gain={f.semantic_gain:.4f}"
        )


def _print_ab_comparison(
    static: ArmResult,
    adaptive: ArmResult,
    analysis: QueryAnalysisResult,
    scorer: SemanticScorer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    s = _arm_summary(static, analysis, scorer)
    a = _arm_summary(adaptive, analysis, scorer)

    static_titles = _candidate_title_set(static.pool)
    adaptive_titles = _candidate_title_set(adaptive.pool)
    intersection = static_titles & adaptive_titles
    union = static_titles | adaptive_titles

    overlap = {
        "static_unique_titles": len(static_titles),
        "adaptive_unique_titles": len(adaptive_titles),
        "shared_titles": len(intersection),
        "adaptive_only_titles": len(adaptive_titles - static_titles),
        "static_only_titles": len(static_titles - adaptive_titles),
        "jaccard": len(intersection) / len(union) if union else 1.0,
    }

    print("\n\n============================================================")
    print("STATIC vs ADAPTIVE — A/B SUMMARY")
    print("============================================================")
    print(
        "NOTE: no gold labels; semantic_proxy is diagnostic only, "
        "not true Precision/Recall/F1."
    )
    print(
        "NOTE: Static currently uses source caps "
        f"{ {k.value: v for k, v in SOURCE_ROUTE_CAPS.items()} }; "
        "Adaptive currently has no per-source cap in production config."
    )
    print(f"AB order                    = {AB_ORDER}")

    rows = [
        ("executed_routes", s["routes"], a["routes"]),
        ("successful_routes", s["successful_routes"], a["successful_routes"]),
        ("empty_routes", s["empty_routes"], a["empty_routes"]),
        ("retrieval_api_calls", s["retrieval_api_calls"], a["retrieval_api_calls"]),
        ("retries", s["retries"], a["retries"]),
        ("raw_hits", s["raw_hits"], a["raw_hits"]),
        ("unique_candidates", s["unique_candidates"], a["unique_candidates"]),
        ("duplicate_ratio", s["duplicate_ratio"], a["duplicate_ratio"]),
        (
            "unique/API",
            s["unique_per_retrieval_api_call"],
            a["unique_per_retrieval_api_call"],
        ),
        (
            "semantic_topk_avg",
            s["semantic_proxy"]["avg"],
            a["semantic_proxy"]["avg"],
        ),
        ("retrieval_network_ms", s["retrieval_network_ms"], a["retrieval_network_ms"]),
        ("routing_overhead_ms", s["routing_overhead_ms"], a["routing_overhead_ms"]),
        ("arm_wall_ms", s["arm_wall_ms"], a["arm_wall_ms"]),
    ]

    print(f"{'metric':<27}{'static':>14}{'adaptive':>14}{'delta(A-S)':>14}")
    print("-" * 69)
    for name, sv, av in rows:
        if isinstance(sv, float) or isinstance(av, float):
            print(f"{name:<27}{float(sv):>14.4f}{float(av):>14.4f}{float(av-sv):>14.4f}")
        else:
            print(f"{name:<27}{sv:>14}{av:>14}{av-sv:>14}")

    print("\n---------- RELATIVE CHANGES ----------")
    for key in (
        "retrieval_api_calls",
        "raw_hits",
        "unique_candidates",
        "unique_per_retrieval_api_call",
        "retrieval_network_ms",
        "arm_wall_ms",
    ):
        change = _pct_change(float(a[key]), float(s[key]))
        text = "N/A" if change is None else f"{change:+.2f}%"
        print(f"{key:<31}= {text}")

    print("\n---------- CANDIDATE OVERLAP ----------")
    print(json.dumps(overlap, indent=2, ensure_ascii=False))

    print("\n---------- SOURCE DISTRIBUTION ----------")
    print(
        "static_source_hits         = "
        + json.dumps(s["source_hit_counts"], ensure_ascii=False)
    )
    print(
        "adaptive_source_hits       = "
        + json.dumps(a["source_hit_counts"], ensure_ascii=False)
    )
    print(
        "static_source_api_calls    = "
        + json.dumps(s["source_api_calls"], ensure_ascii=False)
    )
    print(
        "adaptive_source_api_calls  = "
        + json.dumps(a["source_api_calls"], ensure_ascii=False)
    )

    return s, a


# =============================================================================
# Assertions
# =============================================================================


def _assert_common_ab_invariants(
    static: ArmResult,
    adaptive: ArmResult,
) -> None:
    """Assertions that should remain valid despite live API result drift."""
    s = static.retrieval
    a = adaptive.retrieval

    assert s.metrics.executed_routes <= MAX_EXEC_ROUTES
    assert a.metrics.executed_routes <= MAX_EXEC_ROUTES

    assert all(route.limit <= PER_ROUTE_LIMIT for route in s.plan.routes)
    assert all(route.limit <= PER_ROUTE_LIMIT for route in a.plan.routes)

    assert s.metrics.api_calls <= MAX_RETRIEVAL_API_CALLS
    assert a.metrics.api_calls <= MAX_RETRIEVAL_API_CALLS

    assert s.metrics.raw_hits == len(s.hits)
    assert a.metrics.raw_hits == len(a.hits)

    assert static.pool.stats.candidates_before_filter > 0
    assert adaptive.pool.stats.candidates_before_filter > 0

    # Adaptive diagnostics must describe exactly the selected two-stage plan.
    assert adaptive.adaptive is not None
    d = adaptive.adaptive.diagnostics
    selected = d.probe_route_ids + d.adaptive_route_ids
    final_ids = [route.route_id for route in a.plan.routes]
    assert selected == final_ids
    assert len(selected) == len(set(selected))
    assert len(d.probe_route_ids) <= ADAPTIVE_PROBE_ROUTES


def _assert_optional_strict_non_regression(
    static_summary: dict[str, Any],
    adaptive_summary: dict[str, Any],
) -> None:
    """Optional gate for later, after repeated A/B runs show stable behavior.

    Disabled by default because a single live-network run is not statistically
    sufficient evidence of algorithmic superiority.
    """
    if not STRICT:
        return

    static_unique = int(static_summary["unique_candidates"])
    adaptive_unique = int(adaptive_summary["unique_candidates"])

    assert adaptive_unique >= static_unique * STRICT_MIN_UNIQUE_RETENTION, (
        "Adaptive unique-candidate retention fell below strict threshold: "
        f"adaptive={adaptive_unique}, static={static_unique}, "
        f"required>={STRICT_MIN_UNIQUE_RETENTION:.2f}x static"
    )

    assert (
        int(adaptive_summary["retrieval_api_calls"])
        <= int(static_summary["retrieval_api_calls"]) + STRICT_MAX_API_CALL_DELTA
    ), (
        "Adaptive retrieval API calls exceeded strict budget relative to Static: "
        f"adaptive={adaptive_summary['retrieval_api_calls']}, "
        f"static={static_summary['retrieval_api_calls']}, "
        f"allowed_delta={STRICT_MAX_API_CALL_DELTA}"
    )


# =============================================================================
# Main runner
# =============================================================================


async def _run_ab_case(
    case: PipelineCase,
    scorer: SemanticScorer,
):
    case_started = time.perf_counter()

    analysis = case.factory()
    enabled_sources = get_enabled_sources()
    assert enabled_sources, "At least one retrieval source must be enabled"

    print("\n\n################################################################")
    print(f"# A/B CASE: {case.name}")
    print("################################################################")
    print(f"enabled_sources            = {[s.value for s in enabled_sources]}")
    print(f"max_exec_routes            = {MAX_EXEC_ROUTES}")
    print(f"per_route_limit            = {PER_ROUTE_LIMIT}")
    print(f"adaptive_probe_routes      = {ADAPTIVE_PROBE_ROUTES}")
    print(f"strict_non_regression      = {STRICT}")

    # -------------------------------------------------------------------------
    # 1) Generate ONCE. Both arms share this exact output.
    # -------------------------------------------------------------------------
    generator = _build_generator()
    generation = await generator.generate(analysis)
    _assert_generation_safety(analysis, generation)
    _print_generation(case.name, generation)

    # -------------------------------------------------------------------------
    # 2) Build ONCE. Both arms share this exact native plan.
    # -------------------------------------------------------------------------
    builder = RetrievalRouteBuilder(enabled_sources=enabled_sources)
    native_plan = builder.build(analysis, generation)
    _assert_native_plan_semantics(
        analysis,
        generation,
        native_plan,
        enabled_sources,
    )
    _print_plan("SHARED NATIVE ROUTE PLAN", native_plan)

    # -------------------------------------------------------------------------
    # 3) Run A/B in configured deterministic order.
    #    Separate adapters/orchestrators are created per arm to avoid state leak.
    # -------------------------------------------------------------------------
    if AB_ORDER not in {"static_first", "adaptive_first"}:
        raise ValueError(
            "ADAPTIVE_AB_ORDER must be 'static_first' or 'adaptive_first', "
            f"got {AB_ORDER!r}"
        )

    if AB_ORDER == "static_first":
        static = await _run_static_arm(
            native_plan=native_plan,
            analysis=analysis,
            generation=generation,
            enabled_sources=enabled_sources,
        )
        if AB_PAUSE_SECONDS > 0:
            await asyncio.sleep(AB_PAUSE_SECONDS)
        adaptive = await _run_adaptive_arm(
            native_plan=native_plan,
            analysis=analysis,
            generation=generation,
            enabled_sources=enabled_sources,
            scorer=scorer,
        )
    else:
        adaptive = await _run_adaptive_arm(
            native_plan=native_plan,
            analysis=analysis,
            generation=generation,
            enabled_sources=enabled_sources,
            scorer=scorer,
        )
        if AB_PAUSE_SECONDS > 0:
            await asyncio.sleep(AB_PAUSE_SECONDS)
        static = await _run_static_arm(
            native_plan=native_plan,
            analysis=analysis,
            generation=generation,
            enabled_sources=enabled_sources,
        )

    _assert_common_ab_invariants(static, adaptive)

    static_summary, adaptive_summary = _print_ab_comparison(
        static,
        adaptive,
        analysis,
        scorer,
    )
    _assert_optional_strict_non_regression(
        static_summary,
        adaptive_summary,
    )

    case_total_ms = (time.perf_counter() - case_started) * 1000.0
    print("\n---------- CASE TOTAL ----------")
    print(f"query_generation_ms        = {generation.metrics.latency_ms:.2f}")
    print(f"static_arm_wall_ms         = {static.wall_latency_ms:.2f}")
    print(f"adaptive_arm_wall_ms       = {adaptive.wall_latency_ms:.2f}")
    print(f"whole_ab_case_ms           = {case_total_ms:.2f}")

    print(
        f"\n[PASS] {case.name}: shared Query Generation + shared Native Plan -> "
        "Static vs Adaptive real A/B completed."
    )

    return {
        "analysis": analysis,
        "generation": generation,
        "native_plan": native_plan,
        "static": static,
        "adaptive": adaptive,
        "static_summary": static_summary,
        "adaptive_summary": adaptive_summary,
    }


# =============================================================================
# Pytest entry
# =============================================================================


@pytest.mark.skipif(
    not RUN_REAL,
    reason=(
        "Set RUN_REAL_ADAPTIVE_RETRIEVAL_TESTS=1 "
        "to run real Static-vs-Adaptive retrieval A/B."
    ),
)
@pytest.mark.parametrize(
    "case",
    _selected_cases(),
    ids=lambda c: c.name.replace(" ", "_"),
)
def test_real_static_vs_adaptive_retrieval(
    case: PipelineCase,
    semantic_scorer: SemanticScorer,
):
    result = asyncio.run(_run_ab_case(case, semantic_scorer))

    # Final guard: same generation and native plan are literally shared objects
    # inside one A/B execution; no second LLM generation is allowed here.
    assert result["generation"].metrics.api_calls <= 1
    assert result["native_plan"].routes

