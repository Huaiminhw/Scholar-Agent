from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

from scholar_agent_v2.query_generation.generator import QueryGenerator
from scholar_agent_v2.query_generation.models import QueryKind
from scholar_agent_v2.query_understanding.analyzer import (
    AnalyzerConfig,
    LLMCallError,
    LLMResponse,
    QueryAnalyzer,
)
from scholar_agent_v2.query_understanding.router import (
    QueryUnderstandingRouter,
    UnderstandingRoute,
)
from scholar_agent_v2.query_understanding.schema import QueryType


class FailIfCalledClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        raise AssertionError("deterministic query must not call the LLM")


def test_doi_lookup_uses_deterministic_identifier_route() -> None:
    decision = QueryUnderstandingRouter().route(
        "查找DOI为10.48550/arXiv.1706.03762的论文。"
    )

    assert decision.route == UnderstandingRoute.DETERMINISTIC_IDENTIFIER
    assert decision.result is not None
    assert decision.result.query_type == QueryType.SPECIFIC_BY_IDENTIFIER
    assert decision.result.content == ""
    assert decision.result.constraints.hard[0].value == (
        "10.48550/arxiv.1706.03762"
    )
    assert decision.result.logic.hard is not None
    assert decision.result.logic.hard.ref_id == "C1"


def test_supported_identifier_forms_use_fast_path() -> None:
    router = QueryUnderstandingRouter()
    cases = {
        "请查询 arXiv:1706.03762": "1706.03762",
        "查找 PMID: 31452104 的论文": "PMID:31452104",
        "定位 https://openalex.org/W2741809807": (
            "https://openalex.org/W2741809807"
        ),
        "Find paper CorpusID:13756489": "CorpusID:13756489",
    }

    for query, expected in cases.items():
        decision = router.route(query)
        assert decision.route == UnderstandingRoute.DETERMINISTIC_IDENTIFIER
        assert decision.result is not None
        assert decision.result.constraints.hard[0].value == expected


def test_identifier_with_extra_intent_still_uses_llm() -> None:
    router = QueryUnderstandingRouter()

    for query in (
        "查找引用 DOI:10.1000/example 的论文",
        "找与 arXiv:1706.03762 相似的论文",
        "查找 2023 年后讨论 DOI:10.1000/example 的文章",
    ):
        decision = router.route(query)
        assert decision.route == UnderstandingRoute.LLM
        assert decision.result is None


def test_explicit_exact_title_uses_deterministic_title_route() -> None:
    decision = QueryUnderstandingRouter().route(
        "查找《Attention Is All You Need》这篇论文"
    )

    assert decision.route == UnderstandingRoute.DETERMINISTIC_TITLE
    assert decision.result is not None
    assert decision.result.query_type == QueryType.SPECIFIC_BY_TITLE
    assert decision.result.content == "Attention Is All You Need"
    assert decision.result.properties.specific_paper_name == (
        "Attention Is All You Need"
    )


def test_broad_quoted_query_does_not_accidentally_bypass_llm() -> None:
    decision = QueryUnderstandingRouter().route(
        "查找讨论“注意力机制”且引用量高的最新论文"
    )
    assert decision.route == UnderstandingRoute.LLM


def test_analyzer_bypasses_llm_and_feeds_existing_query_generator() -> None:
    client = FailIfCalledClient()
    analyzer = QueryAnalyzer(client)

    output = asyncio.run(
        analyzer.analyze("查找DOI为10.48550/arXiv.1706.03762的论文。")
    )

    assert client.calls == 0
    assert output.validation.is_valid
    assert output.understanding_route == "deterministic_identifier"
    assert output.metrics.llm_bypassed is True
    assert output.metrics.api_calls == 0
    assert output.metrics.total_tokens == 0

    generation = asyncio.run(QueryGenerator().generate(output.result))
    assert len(generation.result.queries) == 1
    assert generation.result.queries[0].kind == QueryKind.EXACT
    assert generation.result.queries[0].text == "10.48550/arxiv.1706.03762"


def test_router_can_be_disabled_from_analyzer_config() -> None:
    client = FailIfCalledClient()
    analyzer = QueryAnalyzer(
        client,
        config=AnalyzerConfig(enable_deterministic_router=False),
    )

    try:
        asyncio.run(analyzer.analyze("查找 PMID: 31452104 的论文"))
    except LLMCallError as exc:
        assert "must not call" in str(exc)
    else:
        raise AssertionError("disabled router should enter the LLM path")
    assert client.calls == 1
