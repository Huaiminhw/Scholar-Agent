from __future__ import annotations

import asyncio

from scholar_agent_v2.query_generation.models import QueryKind
from scholar_agent_v2.retrieval.models import (
    AdaptiveDepthBudget,
    AdapterFetchResult,
    AdapterSearchResult,
    RetrievalPlan,
    RetrievalHit,
    RetrievalOrigin,
    RetrievalStrategy,
    RouteSpec,
    SearchSource,
    SourceCapabilities,
)
from scholar_agent_v2.retrieval.orchestrator import RetrievalOrchestrator
from scholar_agent_v2.retrieval.sources.base import (
    AcademicSourceAdapter,
    HTTPResponse,
    SourceRateLimitError,
)


class RetryOnceAdapter(AcademicSourceAdapter):
    source = SearchSource.SEMANTIC_SCHOLAR
    capabilities = SourceCapabilities(relevance_search=True)

    def __init__(self) -> None:
        super().__init__()
        self.search_attempts = 0

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        self.search_attempts += 1
        await self._get("https://example.invalid/search")
        return AdapterSearchResult(hits=[], api_calls=1)

    async def fetch_details(self, ids: list[str]) -> AdapterFetchResult:
        return AdapterFetchResult()

    def _request_sync(self, method, url, body, headers) -> HTTPResponse:
        if self.search_attempts == 1:
            raise SourceRateLimitError(429, "test rate limit", retry_after=0.0)
        return HTTPResponse(status=200, body=b"{}", headers={})


def test_failed_http_request_is_included_in_api_call_metrics():
    route = RouteSpec(
        route_id="RT_S2",
        query_id="QC1",
        query_text="multimodal large language models",
        query_kind=QueryKind.CORE,
        source=SearchSource.SEMANTIC_SCHOLAR,
        strategy=RetrievalStrategy.RELEVANCE,
    )
    plan = RetrievalPlan(routes=[route])
    adapter = RetryOnceAdapter()
    orchestrator = RetrievalOrchestrator(
        {SearchSource.SEMANTIC_SCHOLAR: adapter},
        max_attempts=2,
        retry_backoff_seconds=0.0,
    )

    output = asyncio.run(orchestrator.run(plan))
    execution = output.executions[0]

    assert execution.attempts == 2
    assert execution.api_calls == 2
    assert output.metrics.api_calls == 2
    assert output.metrics.source_api_calls == {"semantic_scholar": 2}
    assert output.metrics.retries == 1


class DepthAdapter(AcademicSourceAdapter):
    source = SearchSource.SEMANTIC_SCHOLAR
    capabilities = SourceCapabilities(relevance_search=True)

    def __init__(self, *, shared_results: bool = False) -> None:
        super().__init__()
        self.shared_results = shared_results
        self.requested_limits: list[tuple[str, int]] = []

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        self.requested_limits.append((route.route_id, route.limit))
        prefix = "shared" if self.shared_results else route.route_id
        hits = [
            RetrievalHit(
                source=self.source,
                source_id=f"{prefix}-{rank}",
                title=f"Paper {prefix} {rank}",
                year=2024,
                origin=RetrievalOrigin(
                    source=self.source,
                    route_id=route.route_id,
                    query_id=route.query_id,
                    query_text=route.query_text,
                    query_kind=route.query_kind,
                    strategy=route.strategy,
                    rank=rank,
                ),
            )
            for rank in range(1, route.limit + 1)
        ]
        return AdapterSearchResult(hits=hits, api_calls=1)

    async def fetch_details(self, ids: list[str]) -> AdapterFetchResult:
        return AdapterFetchResult()


def _depth_route(route_id: str) -> RouteSpec:
    return RouteSpec(
        route_id=route_id,
        query_id=f"Q-{route_id}",
        query_text="adaptive retrieval depth",
        query_kind=QueryKind.CORE,
        source=SearchSource.SEMANTIC_SCHOLAR,
        strategy=RetrievalStrategy.RELEVANCE,
        limit=50,
    )


def test_adaptive_depth_expands_productive_route_from_budget() -> None:
    plan = RetrievalPlan(
        routes=[_depth_route("RT1")],
        max_total_hits=50,
        depth_budget=AdaptiveDepthBudget(
            enabled=True,
            first_round_limit=28,
            max_route_limit=50,
            min_new_candidate_rate=0.15,
        ),
    )
    adapter = DepthAdapter()
    output = asyncio.run(
        RetrievalOrchestrator(
            {SearchSource.SEMANTIC_SCHOLAR: adapter},
            max_attempts=1,
        ).run(plan)
    )

    assert adapter.requested_limits == [("RT1", 28), ("RT1", 50)]
    assert len(output.hits) == 50
    assert output.metrics.retrieval_rounds == 2
    assert output.metrics.stop_reason == "max_route_limit_reached"
    assert output.rounds[0].metrics.added_unique_candidates == 28
    assert output.rounds[1].metrics.incremental_hits == 22
    assert output.rounds[1].metrics.refetched_hits_discarded == 28
    assert output.rounds[1].metrics.added_unique_candidates == 22
    assert output.rounds[1].metrics.duplicate_rate == 0.0


def test_adaptive_depth_stops_when_shared_candidate_yield_is_too_low() -> None:
    plan = RetrievalPlan(
        routes=[_depth_route("RT1"), _depth_route("RT2")],
        max_routes=2,
        max_total_hits=100,
        depth_budget=AdaptiveDepthBudget(
            enabled=True,
            first_round_limit=28,
            max_route_limit=50,
            # Identical routes receive 0.5 shared credit each.
            min_new_candidate_rate=0.75,
        ),
    )
    adapter = DepthAdapter(shared_results=True)
    output = asyncio.run(
        RetrievalOrchestrator(
            {SearchSource.SEMANTIC_SCHOLAR: adapter},
            max_attempts=1,
        ).run(plan)
    )

    assert adapter.requested_limits == [("RT1", 28), ("RT2", 28)]
    assert output.metrics.retrieval_rounds == 1
    assert output.metrics.stop_reason == "low_candidate_novelty"
    assert output.rounds[0].metrics.cumulative_unique_candidates == 28
    assert output.rounds[0].metrics.duplicate_rate == 0.5
