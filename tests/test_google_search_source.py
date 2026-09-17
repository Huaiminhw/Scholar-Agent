from __future__ import annotations

import asyncio
import json

import pytest

from scholar_agent_v2.query_generation.models import QueryKind
from scholar_agent_v2.retrieval.config import get_enabled_sources
from scholar_agent_v2.retrieval.models import (
    RetrievalStrategy,
    RouteSpec,
    SearchSource,
)
from scholar_agent_v2.retrieval.sources.base import HTTPResponse, SourceConfigurationError
from scholar_agent_v2.retrieval.sources.google_search import GoogleSearchAdapter
from tests.test_retrieval_candidate_pool import _analysis_with_or, _generation
from scholar_agent_v2.retrieval.route_builder import build_retrieval_plan


def _route(*, limit: int = 12) -> RouteSpec:
    return RouteSpec(
        route_id="RT_GOOGLE",
        query_id="Q1",
        query_text="large language model retrieval",
        query_kind=QueryKind.CORE,
        source=SearchSource.GOOGLE_SEARCH,
        strategy=RetrievalStrategy.RELEVANCE,
        limit=limit,
    )


def test_google_search_paginates_and_normalizes_metadata():
    adapter = GoogleSearchAdapter(api_key="test-key", engine_id="test-cx")
    requests: list[dict] = []

    async def fake_get(url, *, params=None, headers=None):
        requests.append(dict(params or {}))
        start = params["start"]
        count = params["num"]
        items = []
        for offset in range(count):
            rank = start + offset
            items.append(
                {
                    "title": f"Paper {rank}",
                    "link": f"https://arxiv.org/abs/2401.{rank:04d}",
                    "snippet": f"Snippet {rank}",
                    "pagemap": {
                        "metatags": [
                            {
                                "citation_title": f"Normalized Paper {rank}",
                                "citation_author": "Ada Lovelace; Alan Turing",
                                "citation_publication_date": "2024-01-10",
                                "citation_doi": f"10.1000/example.{rank}",
                            }
                        ]
                    },
                }
            )
        return HTTPResponse(
            status=200,
            body=json.dumps({"items": items}).encode("utf-8"),
            headers={},
        )

    adapter._get = fake_get  # type: ignore[method-assign]
    result = asyncio.run(adapter.search(_route()))

    assert len(result.hits) == 12
    assert result.api_calls == 2
    assert [request["start"] for request in requests] == [1, 11]
    assert [request["num"] for request in requests] == [10, 2]
    first = result.hits[0]
    assert first.source == SearchSource.GOOGLE_SEARCH
    assert first.title == "Normalized Paper 1"
    assert first.authors == ["Ada Lovelace", "Alan Turing"]
    assert first.year == 2024
    assert first.doi == "10.1000/example.1"
    assert first.arxiv_id == "2401.0001"
    assert first.origin.rank == 1


def test_google_search_requires_key_and_engine_id(monkeypatch):
    monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_SEARCH_ENGINE_ID", raising=False)
    adapter = GoogleSearchAdapter(api_key="", engine_id="")

    with pytest.raises(SourceConfigurationError, match="GOOGLE_SEARCH_API_KEY"):
        asyncio.run(adapter.search(_route(limit=1)))


def test_google_search_routes_are_opt_in():
    default_plan = build_retrieval_plan(_analysis_with_or(), _generation())
    assert all(route.source != SearchSource.GOOGLE_SEARCH for route in default_plan.routes)

    enabled_plan = build_retrieval_plan(
        _analysis_with_or(),
        _generation(),
        enabled_sources=list(SearchSource),
    )
    assert any(route.source == SearchSource.GOOGLE_SEARCH for route in enabled_plan.routes)


def test_google_search_can_be_enabled_from_environment(monkeypatch):
    monkeypatch.setenv("USE_OPENALEX", "false")
    monkeypatch.setenv("USE_SEMANTIC_SCHOLAR", "false")
    monkeypatch.setenv("USE_ARXIV", "false")
    monkeypatch.setenv("USE_GOOGLE_SEARCH", "true")

    assert get_enabled_sources() == [SearchSource.GOOGLE_SEARCH]
