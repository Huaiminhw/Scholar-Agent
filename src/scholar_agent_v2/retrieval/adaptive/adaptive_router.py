from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from ...query_generation.models import QueryKind
from ...query_understanding.schema import QueryAnalysisResult
from ..models import (
    RetrievalMetrics,
    RetrievalOutput,
    RetrievalPlan,
    RouteExecutionResult,
    RouteSpec,
    RouteStatus,
    SearchSource,
)
from ..orchestrator import RetrievalOrchestrator
from .marginal_gain import aggregate_source_yield, expected_route_gain
from .route_features import RouteSemanticFeatures, compute_probe_features
from .semantic_scorer import SemanticScorer


@dataclass(frozen=True)
class AdaptiveRouterConfig:
    max_exec_routes: int = 6
    probe_routes: int = 3
    per_route_limit: int = 10


@dataclass
class AdaptiveRoutingDiagnostics:
    probe_route_ids: list[str] = field(default_factory=list)
    adaptive_route_ids: list[str] = field(default_factory=list)
    source_yield: dict[str, float] = field(default_factory=dict)
    expected_gain: dict[str, float] = field(default_factory=dict)
    probe_features: dict[str, RouteSemanticFeatures] = field(default_factory=dict)
    semantic_scoring_ms: float = 0.0
    total_adaptive_latency_ms: float = 0.0


@dataclass
class AdaptiveRetrievalResult:
    retrieval: RetrievalOutput
    diagnostics: AdaptiveRoutingDiagnostics


class AdaptiveRoutePlanner:
    """Two-stage, training-free adaptive route executor.

    Stage 1: execute coverage-oriented probe routes.
    Stage 2: use BGE-M3 + LTRR-inspired features + de-duplicated marginal gain
             to select the remaining routes.
    """

    def __init__(
        self,
        orchestrator: RetrievalOrchestrator,
        scorer: SemanticScorer,
        *,
        config: AdaptiveRouterConfig | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.scorer = scorer
        self.config = config or AdaptiveRouterConfig()

    async def run(
        self,
        native_plan: RetrievalPlan,
        analysis: QueryAnalysisResult,
    ) -> AdaptiveRetrievalResult:
        started = time.perf_counter()
        routes = [r for r in native_plan.routes if r.enabled]
        if not routes:
            empty = RetrievalOutput(plan=native_plan)
            return AdaptiveRetrievalResult(empty, AdaptiveRoutingDiagnostics())

        max_exec = min(self.config.max_exec_routes, len(routes))
        probe_n = min(self.config.probe_routes, max_exec)
        semantic_query = (analysis.content or analysis.original_query).strip()

        probe_routes = self._select_probe_routes(routes, probe_n)
        probe_plan = self._subplan(native_plan, probe_routes)
        probe_output = await self.orchestrator.run(probe_plan)

        semantic_started = time.perf_counter()
        probe_features = compute_probe_features(
            semantic_query,
            probe_output.executions,
            self.scorer,
        )
        source_yield = aggregate_source_yield(
            probe_features,
            probe_output.executions,
        )

        selected_ids = {r.route_id for r in probe_routes}
        remaining = [r for r in routes if r.route_id not in selected_ids]
        remaining_slots = max_exec - len(probe_routes)

        expected: dict[str, float] = {}
        for route in remaining:
            expected[route.route_id] = expected_route_gain(
                semantic_query,
                route,
                source_yield,
                self.scorer,
            )

        second_routes = self._select_adaptive_routes(
            remaining,
            remaining_slots,
            already_selected=probe_routes,
            expected_gain=expected,
        )
        semantic_ms = (time.perf_counter() - semantic_started) * 1000

        if second_routes:
            second_plan = self._subplan(native_plan, second_routes)
            second_output = await self.orchestrator.run(second_plan)
        else:
            second_output = RetrievalOutput(plan=self._subplan(native_plan, []))

        final_routes = [*probe_routes, *second_routes]
        final_output = self._merge_outputs(
            native_plan,
            final_routes,
            [probe_output, second_output],
        )

        diagnostics = AdaptiveRoutingDiagnostics(
            probe_route_ids=[r.route_id for r in probe_routes],
            adaptive_route_ids=[r.route_id for r in second_routes],
            source_yield={k.value: float(v) for k, v in source_yield.items()},
            expected_gain={k: float(v) for k, v in expected.items()},
            probe_features=probe_features,
            semantic_scoring_ms=semantic_ms,
            total_adaptive_latency_ms=(time.perf_counter() - started) * 1000,
        )
        return AdaptiveRetrievalResult(final_output, diagnostics)

    def _cap(self, route: RouteSpec) -> RouteSpec:
        return route.model_copy(
            update={"limit": min(route.limit, self.config.per_route_limit)}
        )

    def _subplan(
        self,
        native_plan: RetrievalPlan,
        routes: list[RouteSpec],
    ) -> RetrievalPlan:
        capped = [self._cap(r) for r in routes]
        max_hits = max(1, min(
            native_plan.max_total_hits,
            sum(r.limit for r in capped) if capped else 1,
        ))
        return RetrievalPlan(
            routes=capped,
            max_routes=max(1, len(capped)),
            max_total_hits=max_hits,
            warnings=list(native_plan.warnings),
        )

    @staticmethod
    def _is_anchor(route: RouteSpec) -> bool:
        return route.query_kind in {QueryKind.CORE, QueryKind.EXACT}

    def _select_probe_routes(
        self,
        routes: list[RouteSpec],
        count: int,
    ) -> list[RouteSpec]:
        """Coverage-first probe selection.

        With 3 sources and probe_routes=3 this naturally attempts one route/source,
        while also preserving logic-branch and CORE/EXACT coverage.
        """
        chosen: list[RouteSpec] = []
        remaining = list(routes)
        covered_sources: set[SearchSource] = set()
        covered_branches: set[str | None] = set()
        has_anchor = False

        while remaining and len(chosen) < count:
            best = max(
                remaining,
                key=lambda r: (
                    int(r.source not in covered_sources),
                    int(r.logic_branch_id not in covered_branches),
                    int(self._is_anchor(r) and not has_anchor),
                    r.priority,
                ),
            )
            chosen.append(best)
            remaining.remove(best)
            covered_sources.add(best.source)
            covered_branches.add(best.logic_branch_id)
            has_anchor = has_anchor or self._is_anchor(best)

        return chosen

    def _select_adaptive_routes(
        self,
        routes: list[RouteSpec],
        count: int,
        *,
        already_selected: list[RouteSpec],
        expected_gain: dict[str, float],
    ) -> list[RouteSpec]:
        if count <= 0:
            return []

        chosen: list[RouteSpec] = []
        remaining = list(routes)
        covered_sources = {r.source for r in already_selected}
        covered_branches = {r.logic_branch_id for r in already_selected}
        covered_kinds = {r.query_kind for r in already_selected}

        while remaining and len(chosen) < count:
            # Hierarchical/lexicographic decision: no hand-tuned 4/3/2 weights.
            best = max(
                remaining,
                key=lambda r: (
                    int(r.logic_branch_id not in covered_branches),
                    int(r.query_kind not in covered_kinds),
                    int(r.source not in covered_sources),
                    expected_gain.get(r.route_id, 0.0),
                    r.priority,
                ),
            )
            chosen.append(best)
            remaining.remove(best)
            covered_sources.add(best.source)
            covered_branches.add(best.logic_branch_id)
            covered_kinds.add(best.query_kind)

        return chosen

    def _merge_outputs(
        self,
        native_plan: RetrievalPlan,
        selected_routes: list[RouteSpec],
        outputs: list[RetrievalOutput],
    ) -> RetrievalOutput:
        capped_routes = [self._cap(r) for r in selected_routes]
        final_plan = RetrievalPlan(
            routes=capped_routes,
            max_routes=max(1, len(capped_routes)),
            max_total_hits=max(
                1,
                min(
                    native_plan.max_total_hits,
                    sum(r.limit for r in capped_routes) if capped_routes else 1,
                ),
            ),
            warnings=[*native_plan.warnings, "adaptive routing applied"],
        )

        route_order = {r.route_id: i for i, r in enumerate(capped_routes)}
        executions: list[RouteExecutionResult] = [
            ex for output in outputs for ex in output.executions
        ]
        executions.sort(key=lambda ex: route_order.get(ex.route_id, 10**9))

        hits = []
        for ex in executions:
            hits.extend(sorted(ex.hits, key=lambda h: h.origin.rank))
        hits = hits[: final_plan.max_total_hits]

        source_hit_counts: dict[str, int] = defaultdict(int)
        source_api_calls: dict[str, int] = defaultdict(int)
        source_latencies: dict[str, float] = defaultdict(float)
        for ex in executions:
            source_hit_counts[ex.source.value] += len(ex.hits)
            source_api_calls[ex.source.value] += ex.api_calls
            source_latencies[ex.source.value] += ex.latency_ms

        metrics = RetrievalMetrics(
            planned_routes=len(capped_routes),
            executed_routes=len(executions),
            successful_routes=sum(
                ex.status in {RouteStatus.SUCCESS, RouteStatus.PARTIAL}
                for ex in executions
            ),
            empty_routes=sum(ex.status == RouteStatus.EMPTY for ex in executions),
            failed_routes=sum(ex.status == RouteStatus.FAILED for ex in executions),
            timeout_routes=sum(ex.status == RouteStatus.TIMEOUT for ex in executions),
            api_calls=sum(ex.api_calls for ex in executions),
            retries=sum(max(0, ex.attempts - 1) for ex in executions),
            raw_hits=len(hits),
            source_hit_counts=dict(source_hit_counts),
            source_api_calls=dict(source_api_calls),
            source_latencies_ms=dict(source_latencies),
            # Keep this field as retrieval-network latency. End-to-end adaptive
            # latency (including BGE) is reported separately in diagnostics.
            total_latency_ms=sum(o.metrics.total_latency_ms for o in outputs),
        )

        warnings = [*final_plan.warnings]
        warnings.extend(w for output in outputs for w in output.warnings)
        return RetrievalOutput(
            plan=final_plan,
            executions=executions,
            hits=hits,
            metrics=metrics,
            warnings=warnings,
        )


__all__ = [
    "AdaptiveRouterConfig",
    "AdaptiveRoutingDiagnostics",
    "AdaptiveRetrievalResult",
    "AdaptiveRoutePlanner",
]
