from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Mapping

from ..candidate_pool.dedup import cluster_hits
from .models import (
    RetrievalMetrics,
    RetrievalOutput,
    RetrievalPlan,
    RetrievalRoundMetrics,
    RetrievalRoundResult,
    RouteExecutionResult,
    RouteSpec,
    RouteStatus,
    SearchSource,
)
from .sources.base import AcademicSourceAdapter, SourceRateLimitError


class RetrievalFailedError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    """并发执行 RetrievalPlan；单路线失败不影响其它路线。"""

    def __init__(
        self,
        adapters: Mapping[SearchSource, AcademicSourceAdapter],
        *,
        max_concurrency: int = 6,
        route_timeout_seconds: float = 20.0,
        max_attempts: int = 2,
        retry_backoff_seconds: float = 0.8,
        raise_if_all_failed: bool = True,
    ) -> None:
        self.adapters = dict(adapters)
        self.max_concurrency = max_concurrency
        self.route_timeout_seconds = route_timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.raise_if_all_failed = raise_if_all_failed

    async def run(self, plan: RetrievalPlan) -> RetrievalOutput:
        if plan.depth_budget.enabled:
            return await self._run_adaptive_depth(plan)

        output = await self._run_once(plan)
        unique = self._unique_count(output.hits)
        duplicate_rate = (
            max(0, len(output.hits) - unique) / len(output.hits)
            if output.hits
            else 0.0
        )
        stop_reason = "adaptive_depth_disabled"
        output.metrics.retrieval_rounds = 1
        output.metrics.stop_reason = stop_reason
        output.rounds = [
            RetrievalRoundResult(
                metrics=RetrievalRoundMetrics(
                    round_number=1,
                    requested_route_limit=max(
                        (route.limit for route in plan.routes if route.enabled),
                        default=1,
                    ),
                    route_ids=[route.route_id for route in plan.routes if route.enabled],
                    executed_routes=output.metrics.executed_routes,
                    raw_hits=len(output.hits),
                    incremental_hits=len(output.hits),
                    cumulative_unique_candidates=unique,
                    added_unique_candidates=unique,
                    new_candidate_rate=(unique / len(output.hits) if output.hits else 0.0),
                    duplicate_rate=duplicate_rate,
                    latency_ms=output.metrics.total_latency_ms,
                    stop_reason=stop_reason,
                ),
                hits=list(output.hits),
            )
        ]
        self._log_round(output.rounds[0].metrics)
        return output

    async def _run_once(
        self,
        plan: RetrievalPlan,
        *,
        raise_if_all_failed: bool | None = None,
    ) -> RetrievalOutput:
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        enabled_routes = [r for r in plan.routes if r.enabled]

        async def execute(route):
            async with semaphore:
                return await self._execute_route(route)

        executions = await asyncio.gather(*(execute(route) for route in enabled_routes))

        # 保持 RetrievalPlan 的确定性优先级，而不是按网络完成顺序合并。
        route_order = {route.route_id: i for i, route in enumerate(enabled_routes)}
        executions.sort(key=lambda x: route_order.get(x.route_id, 10**9))

        merged_hits = []
        for execution in executions:
            merged_hits.extend(sorted(execution.hits, key=lambda h: h.origin.rank))
        if len(merged_hits) > plan.max_total_hits:
            merged_hits = merged_hits[: plan.max_total_hits]

        source_hit_counts: dict[str, int] = defaultdict(int)
        source_api_calls: dict[str, int] = defaultdict(int)
        source_latencies: dict[str, float] = defaultdict(float)
        for ex in executions:
            source_hit_counts[ex.source.value] += len(ex.hits)
            source_api_calls[ex.source.value] += ex.api_calls
            source_latencies[ex.source.value] += ex.latency_ms

        metrics = RetrievalMetrics(
            planned_routes=len(enabled_routes),
            executed_routes=len(executions),
            successful_routes=sum(ex.status in {RouteStatus.SUCCESS, RouteStatus.PARTIAL} for ex in executions),
            empty_routes=sum(ex.status == RouteStatus.EMPTY for ex in executions),
            failed_routes=sum(ex.status == RouteStatus.FAILED for ex in executions),
            timeout_routes=sum(ex.status == RouteStatus.TIMEOUT for ex in executions),
            api_calls=sum(ex.api_calls for ex in executions),
            retries=sum(max(0, ex.attempts - 1) for ex in executions),
            raw_hits=len(merged_hits),
            source_hit_counts=dict(source_hit_counts),
            source_api_calls=dict(source_api_calls),
            source_latencies_ms=dict(source_latencies),
            total_latency_ms=(time.perf_counter() - started) * 1000,
        )

        warnings = list(plan.warnings)
        warnings.extend(w for ex in executions for w in ex.warnings)

        all_failed = bool(executions) and all(ex.status in {RouteStatus.FAILED, RouteStatus.TIMEOUT} for ex in executions)
        should_raise = self.raise_if_all_failed if raise_if_all_failed is None else raise_if_all_failed
        if should_raise and all_failed:
            details = "; ".join(f"{ex.route_id}:{ex.error_type or ex.status.value}" for ex in executions)
            raise RetrievalFailedError(f"All retrieval routes failed: {details}")

        return RetrievalOutput(
            plan=plan,
            executions=executions,
            hits=merged_hits,
            metrics=metrics,
            warnings=warnings,
        )

    async def _run_adaptive_depth(self, plan: RetrievalPlan) -> RetrievalOutput:
        budget = plan.depth_budget
        first_routes = [
            route.model_copy(
                update={"limit": min(route.limit, budget.first_round_limit)}
            )
            for route in plan.routes
            if route.enabled
        ]
        first_plan = self._subplan(plan, first_routes)
        first = await self._run_once(first_plan)
        first_unique = self._unique_count(first.hits)
        first_duplicate_rate = (
            max(0, len(first.hits) - first_unique) / len(first.hits)
            if first.hits
            else 0.0
        )

        route_yields = self._shared_candidate_yield(first.executions)
        saturated: list[RouteSpec] = []
        for route in plan.routes:
            execution = next(
                (item for item in first.executions if item.route_id == route.route_id),
                None,
            )
            if execution is None or execution.status not in {
                RouteStatus.SUCCESS,
                RouteStatus.PARTIAL,
            }:
                continue
            first_limit = min(route.limit, budget.first_round_limit)
            if (
                route.limit > first_limit
                and len(execution.hits) >= first_limit
                and route_yields.get(route.route_id, 0.0)
                >= budget.min_new_candidate_rate
            ):
                saturated.append(
                    route.model_copy(
                        update={"limit": min(route.limit, budget.max_route_limit)}
                    )
                )

        if saturated:
            first_stop = "continue_productive_saturated_routes"
        elif any(
            len(execution.hits) >= budget.first_round_limit
            for execution in first.executions
        ):
            first_stop = "low_candidate_novelty"
        else:
            first_stop = "all_routes_exhausted"

        first_round = RetrievalRoundResult(
            metrics=RetrievalRoundMetrics(
                round_number=1,
                requested_route_limit=budget.first_round_limit,
                route_ids=[route.route_id for route in first_routes],
                executed_routes=first.metrics.executed_routes,
                raw_hits=len(first.hits),
                incremental_hits=len(first.hits),
                cumulative_unique_candidates=first_unique,
                added_unique_candidates=first_unique,
                new_candidate_rate=(first_unique / len(first.hits) if first.hits else 0.0),
                duplicate_rate=first_duplicate_rate,
                latency_ms=first.metrics.total_latency_ms,
                stop_reason=first_stop,
            ),
            hits=list(first.hits),
        )
        self._log_round(first_round.metrics)

        if not saturated:
            first.plan = plan
            first.plan.warnings.append("adaptive depth stopped after round 1")
            first.metrics.retrieval_rounds = 1
            first.metrics.stop_reason = first_stop
            first.rounds = [first_round]
            return first

        second_plan = self._subplan(plan, saturated)
        second = await self._run_once(second_plan, raise_if_all_failed=False)
        first_by_route = {item.route_id: item for item in first.executions}
        second_by_route = {item.route_id: item for item in second.executions}
        final_executions: list[RouteExecutionResult] = []
        incremental_hits = []
        refetched = 0

        for route in [item for item in plan.routes if item.enabled]:
            initial = first_by_route.get(route.route_id)
            deeper = second_by_route.get(route.route_id)
            if initial is None:
                continue
            if deeper is None:
                final_executions.append(initial)
                continue

            new_hits = [
                hit
                for hit in deeper.hits
                if hit.origin.rank > min(route.limit, budget.first_round_limit)
            ]
            incremental_hits.extend(new_hits)
            refetched += max(0, len(deeper.hits) - len(new_hits))
            if (
                deeper.status in {RouteStatus.SUCCESS, RouteStatus.PARTIAL}
                and len(deeper.hits) >= len(initial.hits)
            ):
                final_hits = deeper.hits
                final_status = deeper.status
            else:
                final_hits = initial.hits
                final_status = RouteStatus.PARTIAL if initial.hits else deeper.status
            final_executions.append(
                RouteExecutionResult(
                    route_id=route.route_id,
                    source=route.source,
                    status=final_status,
                    hits=final_hits,
                    latency_ms=initial.latency_ms + deeper.latency_ms,
                    attempts=initial.attempts + deeper.attempts,
                    api_calls=initial.api_calls + deeper.api_calls,
                    warnings=[*initial.warnings, *deeper.warnings],
                    error_type=deeper.error_type,
                    error_message=deeper.error_message,
                )
            )

        final = self._output_from_executions(
            plan,
            final_executions,
            total_latency_ms=first.metrics.total_latency_ms + second.metrics.total_latency_ms,
            retries=first.metrics.retries + second.metrics.retries,
        )
        final_unique = self._unique_count(final.hits)
        added_unique = max(0, final_unique - first_unique)
        second_duplicate_rate = (
            max(0, len(incremental_hits) - added_unique) / len(incremental_hits)
            if incremental_hits
            else 0.0
        )
        second_succeeded = any(
            item.status in {RouteStatus.SUCCESS, RouteStatus.EMPTY, RouteStatus.PARTIAL}
            for item in second.executions
        )
        stop_reason = (
            "max_route_limit_reached" if second_succeeded else "second_round_failed"
        )
        second_round = RetrievalRoundResult(
            metrics=RetrievalRoundMetrics(
                round_number=2,
                requested_route_limit=budget.max_route_limit,
                route_ids=[route.route_id for route in saturated],
                executed_routes=second.metrics.executed_routes,
                raw_hits=len(second.hits),
                incremental_hits=len(incremental_hits),
                refetched_hits_discarded=refetched,
                cumulative_unique_candidates=final_unique,
                added_unique_candidates=added_unique,
                new_candidate_rate=(
                    added_unique / len(incremental_hits) if incremental_hits else 0.0
                ),
                duplicate_rate=second_duplicate_rate,
                latency_ms=second.metrics.total_latency_ms,
                stop_reason=stop_reason,
            ),
            hits=list(final.hits),
        )
        self._log_round(second_round.metrics)
        final.rounds = [first_round, second_round]
        final.metrics.retrieval_rounds = 2
        final.metrics.stop_reason = stop_reason
        final.warnings.extend(first.warnings)
        final.warnings.extend(second.warnings)
        return final

    @staticmethod
    def _subplan(plan: RetrievalPlan, routes: list[RouteSpec]) -> RetrievalPlan:
        return RetrievalPlan(
            routes=routes,
            max_routes=max(1, len(routes)),
            max_total_hits=max(
                1,
                min(plan.max_total_hits, sum(route.limit for route in routes)),
            ),
            depth_budget=plan.depth_budget,
            warnings=list(plan.warnings),
        )

    @staticmethod
    def _unique_count(hits) -> int:
        return len(cluster_hits(hits))

    @staticmethod
    def _shared_candidate_yield(
        executions: list[RouteExecutionResult],
    ) -> dict[str, float]:
        all_hits = [hit for execution in executions for hit in execution.hits]
        credits: dict[str, float] = defaultdict(float)
        for cluster in cluster_hits(all_hits):
            route_ids = {hit.origin.route_id for hit in cluster}
            if not route_ids:
                continue
            credit = 1.0 / len(route_ids)
            for route_id in route_ids:
                credits[route_id] += credit
        hit_counts = {
            execution.route_id: len(execution.hits) for execution in executions
        }
        return {
            route_id: credits.get(route_id, 0.0) / count if count else 0.0
            for route_id, count in hit_counts.items()
        }

    @staticmethod
    def _log_round(metrics: RetrievalRoundMetrics) -> None:
        logger.info(
            "retrieval_round=%s limit=%s routes=%s raw_hits=%s incremental_hits=%s "
            "new_candidates=%s new_rate=%.4f duplicate_rate=%.4f latency_ms=%.2f "
            "stop_reason=%s",
            metrics.round_number,
            metrics.requested_route_limit,
            len(metrics.route_ids),
            metrics.raw_hits,
            metrics.incremental_hits,
            metrics.added_unique_candidates,
            metrics.new_candidate_rate,
            metrics.duplicate_rate,
            metrics.latency_ms,
            metrics.stop_reason,
        )

    @staticmethod
    def _output_from_executions(
        plan: RetrievalPlan,
        executions: list[RouteExecutionResult],
        *,
        total_latency_ms: float,
        retries: int = 0,
    ) -> RetrievalOutput:
        merged_hits = [
            hit
            for execution in executions
            for hit in sorted(execution.hits, key=lambda item: item.origin.rank)
        ][: plan.max_total_hits]
        source_hit_counts: dict[str, int] = defaultdict(int)
        source_api_calls: dict[str, int] = defaultdict(int)
        source_latencies: dict[str, float] = defaultdict(float)
        for execution in executions:
            source_hit_counts[execution.source.value] += len(execution.hits)
            source_api_calls[execution.source.value] += execution.api_calls
            source_latencies[execution.source.value] += execution.latency_ms
        metrics = RetrievalMetrics(
            planned_routes=len([route for route in plan.routes if route.enabled]),
            executed_routes=len(executions),
            successful_routes=sum(
                item.status in {RouteStatus.SUCCESS, RouteStatus.PARTIAL}
                for item in executions
            ),
            empty_routes=sum(item.status == RouteStatus.EMPTY for item in executions),
            failed_routes=sum(item.status == RouteStatus.FAILED for item in executions),
            timeout_routes=sum(item.status == RouteStatus.TIMEOUT for item in executions),
            api_calls=sum(item.api_calls for item in executions),
            retries=retries,
            raw_hits=len(merged_hits),
            source_hit_counts=dict(source_hit_counts),
            source_api_calls=dict(source_api_calls),
            source_latencies_ms=dict(source_latencies),
            total_latency_ms=total_latency_ms,
        )
        return RetrievalOutput(
            plan=plan,
            executions=executions,
            hits=merged_hits,
            metrics=metrics,
            warnings=list(plan.warnings),
        )

    async def _execute_route(self, route) -> RouteExecutionResult:
        started = time.perf_counter()
        adapter = self.adapters.get(route.source)
        if adapter is None:
            return RouteExecutionResult(
                route_id=route.route_id,
                source=route.source,
                status=RouteStatus.SKIPPED,
                attempts=0,
                api_calls=0,
                error_type="missing_adapter",
                error_message=f"No adapter registered for {route.source.value}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        attempts = 0
        api_calls = 0
        last_error: Exception | None = None
        warnings: list[str] = []

        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            with adapter.track_http_requests() as request_counter:
                try:
                    result = await asyncio.wait_for(
                        adapter.search(route),
                        timeout=self.route_timeout_seconds,
                    )
                    # 真实 HTTP 计数优先；result.api_calls 为自定义/缓存 Adapter 保留兼容兜底。
                    api_calls += max(request_counter.count, result.api_calls)
                    warnings.extend(result.warnings)
                    status = RouteStatus.SUCCESS if result.hits else RouteStatus.EMPTY
                    return RouteExecutionResult(
                        route_id=route.route_id,
                        source=route.source,
                        status=status,
                        hits=result.hits,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        attempts=attempts,
                        api_calls=api_calls,
                        warnings=warnings,
                    )
                except asyncio.TimeoutError as exc:
                    api_calls += request_counter.count
                    last_error = exc
                    if attempt < self.max_attempts:
                        await asyncio.sleep(self.retry_backoff_seconds * attempt)
                        continue
                    return RouteExecutionResult(
                        route_id=route.route_id,
                        source=route.source,
                        status=RouteStatus.TIMEOUT,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        attempts=attempts,
                        api_calls=api_calls,
                        error_type="timeout",
                        error_message=f"route exceeded {self.route_timeout_seconds}s",
                    )
                except SourceRateLimitError as exc:
                    api_calls += request_counter.count
                    last_error = exc
                    if attempt < self.max_attempts:
                        wait_for = exc.retry_after if exc.retry_after is not None else self.retry_backoff_seconds * attempt
                        await asyncio.sleep(min(max(wait_for, 0.0), 10.0))
                        continue
                    break
                except Exception as exc:  # 单路线容错边界
                    api_calls += request_counter.count
                    last_error = exc
                    if attempt < self.max_attempts:
                        await asyncio.sleep(self.retry_backoff_seconds * attempt)
                        continue
                    break

        return RouteExecutionResult(
            route_id=route.route_id,
            source=route.source,
            status=RouteStatus.FAILED,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=attempts,
            api_calls=api_calls,
            error_type=type(last_error).__name__ if last_error else "unknown",
            error_message=str(last_error) if last_error else "unknown retrieval failure",
        )


__all__ = ["RetrievalOrchestrator", "RetrievalFailedError"]
