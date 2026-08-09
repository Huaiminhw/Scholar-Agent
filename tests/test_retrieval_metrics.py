from __future__ import annotations

import asyncio

from scholar_agent_v2.query_generation.models import QueryKind
from scholar_agent_v2.retrieval.models import (
    AdapterFetchResult,
    AdapterSearchResult,
    RetrievalPlan,
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
