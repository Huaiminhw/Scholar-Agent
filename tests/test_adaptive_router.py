from __future__ import annotations

import asyncio

import numpy as np
import pytest

from scholar_agent_v2.query_generation.models import QueryKind
from scholar_agent_v2.query_understanding.schema import (
    QueryAnalysisResult,
    QueryIntent,
    QueryType,
)
from scholar_agent_v2.retrieval.adaptive import (
    AdaptiveRoutePlanner,
    AdaptiveRouterConfig,
)
from scholar_agent_v2.retrieval.adaptive.marginal_gain import (
    aggregate_source_yield,
    expected_route_gain,
)
from scholar_agent_v2.retrieval.adaptive.route_features import (
    RouteSemanticFeatures,
    compute_probe_features,
)
from scholar_agent_v2.retrieval.models import (
    RetrievalHit,
    RetrievalMetrics,
    RetrievalOrigin,
    RetrievalOutput,
    RetrievalPlan,
    RetrievalStrategy,
    RouteExecutionResult,
    RouteSpec,
    RouteStatus,
    SearchSource,
)


# =============================================================================
# Test doubles
# =============================================================================


class FakeSemanticScorer:
    """Deterministic local scorer for unit tests.

    It intentionally does NOT load sentence-transformers / BGE-M3.
    The purpose of this file is to test routing logic, not model quality.
    """

    TERMS = (
        "reinforcement",
        "learning",
        "academic",
        "search",
        "agent",
        "scientific",
        "query",
        "dataset",
        "evaluation",
    )

    @staticmethod
    def paper_text(title: str | None, abstract: str | None) -> str:
        title = (title or "").strip()
        abstract = (abstract or "").strip()
        if title and abstract:
            return f"Title: {title}\nAbstract: {abstract}"
        return title or abstract

    def _vector(self, text: str) -> np.ndarray:
        text = (text or "").casefold()
        vector = np.asarray(
            [float(text.count(term)) for term in self.TERMS],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            vector = vector / norm
        return vector

    def encode(self, texts) -> np.ndarray:
        items = [str(x) for x in texts]
        if not items:
            return np.empty((0, len(self.TERMS)), dtype=np.float32)
        return np.stack([self._vector(x) for x in items], axis=0)

    def similarity(self, left: str, right: str) -> float:
        emb = self.encode([left, right])
        return float(np.dot(emb[0], emb[1]))


class ScriptedOrchestrator:
    """Fake orchestrator with a deliberately productive OpenAlex source.

    - OpenAlex returns two semantically relevant hits per route.
    - Semantic Scholar returns zero hits.
    - arXiv returns zero hits.

    This lets us verify that Stage 2 reallocates route budget toward the
    productive source after the three-source probe.
    """

    def __init__(self) -> None:
        self.plans: list[RetrievalPlan] = []

    async def run(self, plan: RetrievalPlan) -> RetrievalOutput:
        self.plans.append(plan)

        executions: list[RouteExecutionResult] = []
        merged_hits: list[RetrievalHit] = []

        for route in plan.routes:
            hits = self._hits_for_route(route)
            status = RouteStatus.SUCCESS if hits else RouteStatus.EMPTY
            execution = RouteExecutionResult(
                route_id=route.route_id,
                source=route.source,
                status=status,
                hits=hits,
                latency_ms=10.0,
                attempts=1,
                api_calls=1,
            )
            executions.append(execution)
            merged_hits.extend(hits)

        source_hit_counts: dict[str, int] = {}
        source_api_calls: dict[str, int] = {}
        source_latencies_ms: dict[str, float] = {}

        for execution in executions:
            source = execution.source.value
            source_hit_counts[source] = source_hit_counts.get(source, 0) + len(
                execution.hits
            )
            source_api_calls[source] = source_api_calls.get(source, 0) + execution.api_calls
            source_latencies_ms[source] = (
                source_latencies_ms.get(source, 0.0) + execution.latency_ms
            )

        metrics = RetrievalMetrics(
            planned_routes=len(plan.routes),
            executed_routes=len(executions),
            successful_routes=sum(
                ex.status in {RouteStatus.SUCCESS, RouteStatus.PARTIAL}
                for ex in executions
            ),
            empty_routes=sum(ex.status == RouteStatus.EMPTY for ex in executions),
            failed_routes=0,
            timeout_routes=0,
            api_calls=sum(ex.api_calls for ex in executions),
            retries=0,
            raw_hits=len(merged_hits),
            source_hit_counts=source_hit_counts,
            source_api_calls=source_api_calls,
            source_latencies_ms=source_latencies_ms,
            total_latency_ms=10.0 * len(executions),
        )

        return RetrievalOutput(
            plan=plan,
            executions=executions,
            hits=merged_hits,
            metrics=metrics,
            warnings=[],
        )

    @staticmethod
    def _hits_for_route(route: RouteSpec) -> list[RetrievalHit]:
        if route.source != SearchSource.OPENALEX:
            return []

        # One common paper appears on every productive OpenAlex route.
        # This also exercises de-dup/shared-credit behavior in realistic data.
        common = _hit(
            route,
            source_id=f"common-{route.route_id}",
            title="Reinforcement Learning for LLM Academic Search Agents",
            abstract=(
                "An academic search agent optimized with reinforcement learning "
                "and evaluated on real scientific query datasets."
            ),
            rank=1,
            doi="10.1000/common-adaptive-paper",
        )

        route_specific = _hit(
            route,
            source_id=f"specific-{route.route_id}",
            title=f"Academic Search Agent Evaluation {route.route_id}",
            abstract=(
                "Scientific query dataset evaluation for an academic search agent "
                "using reinforcement learning."
            ),
            rank=2,
            doi=f"10.1000/{route.route_id.casefold()}",
        )
        return [common, route_specific]


# =============================================================================
# Fixtures / builders
# =============================================================================


def _analysis() -> QueryAnalysisResult:
    return QueryAnalysisResult(
        original_query=(
            "找2023年以来使用强化学习优化LLM学术搜索Agent，"
            "并在真实科研查询数据集上验证的论文。"
        ),
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(),
        content=(
            "reinforcement learning optimization of LLM academic search agent "
            "evaluated on real scientific query datasets"
        ),
    )


def _route(
    route_id: str,
    source: SearchSource,
    kind: QueryKind,
    query_text: str,
    *,
    query_id: str,
    priority: float,
    branch: str | None = "B1",
    limit: int = 28,
) -> RouteSpec:
    return RouteSpec(
        route_id=route_id,
        query_id=query_id,
        query_text=query_text,
        query_kind=kind,
        source=source,
        strategy=RetrievalStrategy.RELEVANCE,
        priority=priority,
        limit=limit,
        logic_branch_id=branch,
    )


def _native_plan() -> RetrievalPlan:
    """Mirrors the structure of the previous Complex Relation baseline."""

    routes = [
        _route(
            "RT001",
            SearchSource.SEMANTIC_SCHOLAR,
            QueryKind.CORE,
            "reinforcement learning optimization of LLM academic search agent "
            "evaluated on real scientific query datasets",
            query_id="QC1",
            priority=0.950,
        ),
        _route(
            "RT002",
            SearchSource.OPENALEX,
            QueryKind.CORE,
            "reinforcement learning optimization of LLM academic search agent "
            "evaluated on real scientific query datasets",
            query_id="QC1",
            priority=0.912,
        ),
        _route(
            "RT003",
            SearchSource.SEMANTIC_SCHOLAR,
            QueryKind.RELATION,
            "reinforcement learning optimization LLM academic search agent",
            query_id="QR1",
            priority=0.846,
        ),
        _route(
            "RT004",
            SearchSource.SEMANTIC_SCHOLAR,
            QueryKind.RELATION,
            "LLM academic search agent evaluation real scientific query datasets",
            query_id="QR2",
            priority=0.846,
        ),
        _route(
            "RT005",
            SearchSource.ARXIV,
            QueryKind.CORE,
            "reinforcement learning optimization of LLM academic search agent "
            "evaluated on real scientific query datasets",
            query_id="QC1",
            priority=0.836,
        ),
        _route(
            "RT006",
            SearchSource.OPENALEX,
            QueryKind.RELATION,
            "reinforcement learning optimization LLM academic search agent",
            query_id="QR1",
            priority=0.812,
        ),
        _route(
            "RT007",
            SearchSource.OPENALEX,
            QueryKind.RELATION,
            "LLM academic search agent evaluation real scientific query datasets",
            query_id="QR2",
            priority=0.812,
        ),
        _route(
            "RT008",
            SearchSource.SEMANTIC_SCHOLAR,
            QueryKind.EXPANDED,
            "reinforcement learning LLM search agent scientific literature queries",
            query_id="QE1",
            priority=0.533,
        ),
        _route(
            "RT009",
            SearchSource.OPENALEX,
            QueryKind.EXPANDED,
            "reinforcement learning LLM search agent scientific literature queries",
            query_id="QE1",
            priority=0.511,
        ),
    ]

    return RetrievalPlan(
        routes=routes,
        max_routes=10,
        max_total_hits=280,
    )


def _hit(
    route: RouteSpec,
    *,
    source_id: str,
    title: str,
    abstract: str,
    rank: int,
    doi: str | None = None,
) -> RetrievalHit:
    origin = RetrievalOrigin(
        source=route.source,
        route_id=route.route_id,
        query_id=route.query_id,
        query_text=route.query_text,
        query_kind=route.query_kind,
        strategy=route.strategy,
        logic_branch_id=route.logic_branch_id,
        rank=rank,
    )
    return RetrievalHit(
        source=route.source,
        source_id=source_id,
        title=title,
        abstract=abstract,
        doi=doi,
        origin=origin,
    )


def _execution(
    route: RouteSpec,
    hits: list[RetrievalHit],
    *,
    api_calls: int = 1,
) -> RouteExecutionResult:
    return RouteExecutionResult(
        route_id=route.route_id,
        source=route.source,
        status=RouteStatus.SUCCESS if hits else RouteStatus.EMPTY,
        hits=hits,
        latency_ms=1.0,
        attempts=1,
        api_calls=api_calls,
    )


# =============================================================================
# 1. Probe selection
# =============================================================================


def test_probe_selection_covers_three_sources_and_keeps_core_anchor():
    planner = AdaptiveRoutePlanner(
        ScriptedOrchestrator(),
        FakeSemanticScorer(),
        config=AdaptiveRouterConfig(
            max_exec_routes=6,
            probe_routes=3,
            per_route_limit=10,
        ),
    )

    probes = planner._select_probe_routes(_native_plan().routes, 3)

    # For the current Complex baseline structure the deterministic selection is:
    # S2 CORE -> OpenAlex CORE -> arXiv CORE.
    assert [route.route_id for route in probes] == ["RT001", "RT002", "RT005"]
    assert {route.source for route in probes} == {
        SearchSource.SEMANTIC_SCHOLAR,
        SearchSource.OPENALEX,
        SearchSource.ARXIV,
    }
    assert all(route.query_kind == QueryKind.CORE for route in probes)


# =============================================================================
# 2. Shared-credit duplicate handling
# =============================================================================


def test_shared_credit_splits_one_duplicate_cluster_across_routes():
    scorer = FakeSemanticScorer()
    query = "reinforcement learning academic search agent"

    oa_route = _route(
        "RT_OA",
        SearchSource.OPENALEX,
        QueryKind.CORE,
        query,
        query_id="QC1",
        priority=1.0,
    )
    s2_route = _route(
        "RT_S2",
        SearchSource.SEMANTIC_SCHOLAR,
        QueryKind.CORE,
        query,
        query_id="QC1",
        priority=1.0,
    )

    title = "Reinforcement Learning for Academic Search Agents"
    abstract = "Academic search agent evaluated with scientific query datasets."

    oa_hit = _hit(
        oa_route,
        source_id="W1",
        title=title,
        abstract=abstract,
        rank=1,
        doi="10.1000/shared-paper",
    )
    s2_hit = _hit(
        s2_route,
        source_id="S2-1",
        title=title,
        abstract=abstract,
        rank=1,
        doi="https://doi.org/10.1000/shared-paper",
    )

    features = compute_probe_features(
        query,
        [
            _execution(oa_route, [oa_hit]),
            _execution(s2_route, [s2_hit]),
        ],
        scorer,
    )

    assert features["RT_OA"].unique_credit == pytest.approx(0.5)
    assert features["RT_S2"].unique_credit == pytest.approx(0.5)

    expected_total_gain = scorer.similarity(
        query,
        scorer.paper_text(title, abstract),
    )
    actual_total_gain = (
        features["RT_OA"].semantic_gain
        + features["RT_S2"].semantic_gain
    )

    # One duplicate paper contributes only one unit of semantic evidence in total.
    assert actual_total_gain == pytest.approx(expected_total_gain, abs=1e-6)
    assert features["RT_OA"].semantic_gain == pytest.approx(
        features["RT_S2"].semantic_gain,
        abs=1e-6,
    )


# =============================================================================
# 3. Source yield
# =============================================================================


def test_source_yield_is_semantic_gain_per_api_call():
    oa_route = _route(
        "RT_OA",
        SearchSource.OPENALEX,
        QueryKind.CORE,
        "academic search agent",
        query_id="QC1",
        priority=1.0,
    )
    s2_route = _route(
        "RT_S2",
        SearchSource.SEMANTIC_SCHOLAR,
        QueryKind.CORE,
        "academic search agent",
        query_id="QC1",
        priority=1.0,
    )

    features = {
        "RT_OA": RouteSemanticFeatures(
            route_id="RT_OA",
            source=SearchSource.OPENALEX,
            hit_count=10,
            avg_sim=0.7,
            max_sim=0.9,
            var_sim=0.01,
            overall_sim=0.8,
            cross_retriever_sim=0.2,
            unique_credit=4.0,
            semantic_gain=1.2,
        ),
        "RT_S2": RouteSemanticFeatures(
            route_id="RT_S2",
            source=SearchSource.SEMANTIC_SCHOLAR,
            hit_count=0,
            avg_sim=0.0,
            max_sim=0.0,
            var_sim=0.0,
            overall_sim=0.0,
            cross_retriever_sim=0.0,
            unique_credit=0.0,
            semantic_gain=0.0,
        ),
    }

    yields = aggregate_source_yield(
        features,
        [
            _execution(oa_route, [], api_calls=2),
            _execution(s2_route, [], api_calls=1),
        ],
    )

    assert yields[SearchSource.OPENALEX] == pytest.approx(0.6)
    assert yields[SearchSource.SEMANTIC_SCHOLAR] == pytest.approx(0.0)


# =============================================================================
# 4. Expected gain
# =============================================================================


def test_expected_route_gain_is_alignment_times_observed_source_yield():
    scorer = FakeSemanticScorer()
    semantic_query = "reinforcement learning academic search agent"

    oa_route = _route(
        "RT_OA",
        SearchSource.OPENALEX,
        QueryKind.RELATION,
        "reinforcement learning optimization academic search agent",
        query_id="QR1",
        priority=0.8,
    )
    s2_route = _route(
        "RT_S2",
        SearchSource.SEMANTIC_SCHOLAR,
        QueryKind.RELATION,
        "reinforcement learning optimization academic search agent",
        query_id="QR1",
        priority=0.8,
    )

    source_yield = {SearchSource.OPENALEX: 2.0}
    alignment = scorer.similarity(semantic_query, oa_route.query_text)

    assert expected_route_gain(
        semantic_query,
        oa_route,
        source_yield,
        scorer,
    ) == pytest.approx(alignment * 2.0)

    # No observed yield for S2 -> expected gain must be zero.
    assert expected_route_gain(
        semantic_query,
        s2_route,
        source_yield,
        scorer,
    ) == pytest.approx(0.0)


# =============================================================================
# 5. Full two-stage adaptive run
# =============================================================================


def test_full_adaptive_run_reallocates_remaining_budget_to_productive_source():
    orchestrator = ScriptedOrchestrator()
    planner = AdaptiveRoutePlanner(
        orchestrator,
        FakeSemanticScorer(),
        config=AdaptiveRouterConfig(
            max_exec_routes=6,
            probe_routes=3,
            per_route_limit=10,
        ),
    )

    result = asyncio.run(planner.run(_native_plan(), _analysis()))
    retrieval = result.retrieval
    diagnostics = result.diagnostics

    # Stage 1: three-source coverage probe.
    assert diagnostics.probe_route_ids == ["RT001", "RT002", "RT005"]

    # OpenAlex is the only productive probe source, so the remaining three routes
    # should all be selected from OpenAlex.
    assert set(diagnostics.adaptive_route_ids) == {"RT006", "RT007", "RT009"}

    assert diagnostics.source_yield["openalex"] > 0.0
    assert diagnostics.source_yield["semantic_scholar"] == pytest.approx(0.0)
    assert diagnostics.source_yield["arxiv"] == pytest.approx(0.0)

    # Two separate orchestrator calls: probe stage + adaptive stage.
    assert len(orchestrator.plans) == 2
    assert len(orchestrator.plans[0].routes) == 3
    assert len(orchestrator.plans[1].routes) == 3
    assert all(
        route.source == SearchSource.OPENALEX
        for route in orchestrator.plans[1].routes
    )

    # Competition budget must remain intact.
    assert len(retrieval.plan.routes) == 6
    assert retrieval.metrics.planned_routes == 6
    assert retrieval.metrics.executed_routes == 6
    assert retrieval.metrics.api_calls == 6
    assert all(route.limit <= 10 for route in retrieval.plan.routes)

    # Network-retrieval latency is the sum of the two fake retrieval stages.
    assert retrieval.metrics.total_latency_ms == pytest.approx(60.0)

    assert "adaptive routing applied" in retrieval.plan.warnings
    assert diagnostics.semantic_scoring_ms >= 0.0
    assert diagnostics.total_adaptive_latency_ms >= 0.0


# =============================================================================
# 6. No-route safety
# =============================================================================


def test_empty_native_plan_returns_safely_without_orchestrator_call():
    orchestrator = ScriptedOrchestrator()
    planner = AdaptiveRoutePlanner(
        orchestrator,
        FakeSemanticScorer(),
        config=AdaptiveRouterConfig(
            max_exec_routes=6,
            probe_routes=3,
            per_route_limit=10,
        ),
    )

    empty_plan = RetrievalPlan(
        routes=[],
        max_routes=1,
        max_total_hits=1,
    )

    result = asyncio.run(planner.run(empty_plan, _analysis()))

    assert result.retrieval.executions == []
    assert result.retrieval.hits == []
    assert result.retrieval.metrics.executed_routes == 0
    assert result.diagnostics.probe_route_ids == []
    assert result.diagnostics.adaptive_route_ids == []
    assert orchestrator.plans == []
