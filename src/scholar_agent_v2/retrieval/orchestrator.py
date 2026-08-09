from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Mapping

from .models import (
    RetrievalMetrics,
    RetrievalOutput,
    RetrievalPlan,
    RouteExecutionResult,
    RouteStatus,
    SearchSource,
)
from .sources.base import AcademicSourceAdapter, SourceRateLimitError


class RetrievalFailedError(RuntimeError):
    pass


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
        if self.raise_if_all_failed and all_failed:
            details = "; ".join(f"{ex.route_id}:{ex.error_type or ex.status.value}" for ex in executions)
            raise RetrievalFailedError(f"All retrieval routes failed: {details}")

        return RetrievalOutput(
            plan=plan,
            executions=executions,
            hits=merged_hits,
            metrics=metrics,
            warnings=warnings,
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
