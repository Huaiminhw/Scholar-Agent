from __future__ import annotations

import asyncio

from scholar_agent_v2.query_generation.models import QueryKind
from scholar_agent_v2.retrieval.models import (
    RetrievalStrategy,
    RouteSpec,
    SearchSource,
)
from scholar_agent_v2.retrieval.sources.arxiv import ArxivAdapter
from scholar_agent_v2.retrieval.sources.base import HTTPResponse


def _route(query: str) -> RouteSpec:
    return RouteSpec(
        route_id="RT-AX",
        query_id="QC1",
        query_text=query,
        query_kind=QueryKind.CORE,
        source=SearchSource.ARXIV,
        strategy=RetrievalStrategy.RELEVANCE,
        priority=1.0,
        limit=5,
    )


def _feed(*, with_entry: bool) -> bytes:
    entry = """
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <updated>2024-01-01T00:00:00Z</updated>
        <published>2024-01-01T00:00:00Z</published>
        <title>Multimodal Language Models</title>
        <summary>Reinforcement learning for multimodal models.</summary>
        <author><name>Test Author</name></author>
        <link href="https://arxiv.org/abs/2401.00001" rel="alternate" />
        <category term="cs.CL" />
      </entry>
    """ if with_entry else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">{entry}</feed>
    """.encode()


def test_relevance_query_is_tokenized_instead_of_one_long_phrase() -> None:
    adapter = ArxivAdapter()
    route = _route(
        "Find papers about multimodal large language models using reinforcement learning"
    )

    assert adapter._build_query(route) == (
        'all:"multimodal" AND all:"large" AND all:"language" '
        'AND all:"models" AND all:"reinforcement" AND all:"learning"'
    )
    assert adapter._build_query(route, operator="OR") == (
        'all:"multimodal" OR all:"large" OR all:"language" '
        'OR all:"models" OR all:"reinforcement" OR all:"learning"'
    )


def test_relevance_search_retries_with_or_after_empty_and_result() -> None:
    class FakeArxivAdapter(ArxivAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[str] = []

        async def _get(self, url, *, params=None, headers=None):
            self.queries.append(params["search_query"])
            return HTTPResponse(
                status=200,
                body=_feed(with_entry=len(self.queries) == 2),
                headers={},
            )

    adapter = FakeArxivAdapter()
    output = asyncio.run(adapter.search(_route("multimodal language models")))

    assert len(adapter.queries) == 2
    assert " AND " in adapter.queries[0]
    assert " OR " in adapter.queries[1]
    assert output.api_calls == 2
    assert len(output.hits) == 1
    assert output.hits[0].arxiv_id == "2401.00001"
    assert output.warnings == [
        "arXiv AND query returned no results; retried with OR"
    ]


def test_title_query_remains_an_exact_title_phrase() -> None:
    route = _route("Attention Is All You Need").model_copy(
        update={"strategy": RetrievalStrategy.TITLE}
    )
    assert ArxivAdapter()._build_query(route) == 'ti:"Attention Is All You Need"'
