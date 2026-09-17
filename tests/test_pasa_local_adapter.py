from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scholar_agent_v2.query_generation.models import (
    ExpansionBudget,
    GeneratedQuery,
    GenerationMode,
    QueryGenerationPlan,
    QueryGenerationResult,
    QueryKind,
)
from scholar_agent_v2.query_understanding.schema import (
    QueryAnalysisResult,
    QueryIntent,
    QueryType,
    RetrievalMode,
)
from scholar_agent_v2.retrieval.config import (
    QueryExecutionMode,
    RetrievalProfile,
    get_enabled_sources,
    get_query_execution_mode,
    get_retrieval_profile,
)
from scholar_agent_v2.retrieval.factory import build_adapters
from scholar_agent_v2.retrieval.models import (
    RetrievalStrategy,
    RouteSpec,
    SearchSource,
)
from scholar_agent_v2.retrieval.pipeline import ScholarRetrievalPipeline
from scholar_agent_v2.retrieval.route_builder import RetrievalRouteBuilder
from scholar_agent_v2.retrieval.sources.base import SourceConfigurationError
from scholar_agent_v2.retrieval.sources.pasa_corpus import PaSaBM25Result
from scholar_agent_v2.retrieval.sources.pasa_local import PaSaLocalAdapter


class _FakeCorpus:
    index_path = Path("fake.sqlite3")

    @staticmethod
    def is_index_ready() -> bool:
        return True

    @staticmethod
    def _paper() -> PaSaBM25Result:
        return PaSaBM25Result(
            rank=1,
            corpus_key="paper.json",
            title="Attention Is All You Need",
            abstract="A transformer architecture.",
            arxiv_id="1706.03762",
            year=2017,
            bm25_score=12.5,
            sources=["arxiv"],
        )

    def search(self, query: str, *, limit: int, operator: str):
        assert query
        assert limit > 0
        assert operator == "or"
        return [self._paper()]

    def get_by_arxiv_id(self, arxiv_id: str):
        return self._paper() if arxiv_id == "1706.03762" else None


def _route(*, strategy: RetrievalStrategy, text: str) -> RouteSpec:
    return RouteSpec(
        route_id="RT1",
        query_id="Q1",
        query_text=text,
        query_kind=QueryKind.CORE,
        source=SearchSource.PASA_LOCAL,
        strategy=strategy,
        limit=10,
    )


def test_pasa_local_adapter_returns_standard_retrieval_hit() -> None:
    adapter = PaSaLocalAdapter(corpus=_FakeCorpus())
    result = asyncio.run(
        adapter.search(_route(strategy=RetrievalStrategy.RELEVANCE, text="transformer"))
    )

    assert result.api_calls == 0
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.source == SearchSource.PASA_LOCAL
    assert hit.origin.source == SearchSource.PASA_LOCAL
    assert hit.origin.rank == 1
    assert hit.arxiv_id == "1706.03762"
    assert hit.abstract


def test_pasa_local_adapter_supports_arxiv_doi_lookup() -> None:
    adapter = PaSaLocalAdapter(corpus=_FakeCorpus())
    result = asyncio.run(
        adapter.search(
            _route(
                strategy=RetrievalStrategy.IDENTIFIER,
                text="10.48550/arXiv.1706.03762",
            )
        )
    )

    assert [hit.arxiv_id for hit in result.hits] == ["1706.03762"]


def test_frozen_profile_selects_only_pasa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_PROFILE", "frozen_local")
    monkeypatch.setenv("QUERY_EXECUTION_MODE", "frozen")
    assert get_retrieval_profile() == RetrievalProfile.FROZEN_LOCAL
    assert get_query_execution_mode() == QueryExecutionMode.FROZEN
    assert get_enabled_sources() == [SearchSource.PASA_LOCAL]


def test_profile_factory_rejects_local_online_mixing() -> None:
    with pytest.raises(SourceConfigurationError, match="cannot include PASA_LOCAL"):
        build_adapters(
            [SearchSource.PASA_LOCAL, SearchSource.ARXIV],
            profile=RetrievalProfile.COMPETITION_ONLINE,
        )


def test_route_builder_creates_only_pasa_routes_in_local_profile() -> None:
    analysis = QueryAnalysisResult(
        original_query="transformers for video generation",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(mode=RetrievalMode.COMPREHENSIVE),
        content="transformers for video generation",
    )
    generation = QueryGenerationResult(
        original_query=analysis.original_query,
        core_content=analysis.content,
        plan=QueryGenerationPlan(
            mode=GenerationMode.BROAD,
            budget=ExpansionBudget(core=1, expanded=1),
            use_llm=True,
            max_total_queries=5,
        ),
        queries=[
            GeneratedQuery(
                query_id="QC1",
                text="transformers for video generation",
                kind=QueryKind.CORE,
                priority=0.95,
            ),
            GeneratedQuery(
                query_id="QE1",
                text="autoregressive video synthesis transformers",
                kind=QueryKind.EXPANDED,
                priority=0.72,
            ),
        ],
        llm_used=True,
    )

    plan = RetrievalRouteBuilder(
        enabled_sources=[SearchSource.PASA_LOCAL]
    ).build(analysis, generation)

    assert len(plan.routes) == 2
    assert {route.source for route in plan.routes} == {SearchSource.PASA_LOCAL}
    assert {route.query_kind for route in plan.routes} == {
        QueryKind.CORE,
        QueryKind.EXPANDED,
    }
    assert all(route.limit == 50 for route in plan.routes)
    assert plan.max_total_hits == 500
    assert plan.depth_budget.enabled is True
    assert plan.depth_budget.first_round_limit == 28
    assert plan.depth_budget.max_route_limit == 50
    assert all(not route.pushed_constraint_ids for route in plan.routes)


def test_shared_pipeline_runs_pasa_through_candidate_pool_and_prerank() -> None:
    analysis = QueryAnalysisResult(
        original_query="transformer",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(mode=RetrievalMode.PRECISE),
        content="transformer",
    )
    generation = QueryGenerationResult(
        original_query="transformer",
        core_content="transformer",
        plan=QueryGenerationPlan(
            mode=GenerationMode.STANDARD,
            budget=ExpansionBudget(core=1),
            max_total_queries=1,
        ),
        queries=[
            GeneratedQuery(
                query_id="QC1",
                text="transformer",
                kind=QueryKind.CORE,
                priority=0.95,
            )
        ],
    )
    adapter = PaSaLocalAdapter(corpus=_FakeCorpus())
    pipeline = ScholarRetrievalPipeline(
        enabled_sources=[SearchSource.PASA_LOCAL],
        adapters={SearchSource.PASA_LOCAL: adapter},
        max_attempts=1,
    )

    output = asyncio.run(pipeline.run(analysis, generation))

    assert output.retrieval.metrics.api_calls == 0
    assert output.retrieval.metrics.executed_routes == 1
    assert len(output.candidate_pool.candidates) == 1
    candidate = output.candidate_pool.candidates[0]
    assert candidate.identifiers.arxiv_id == "1706.03762"
    assert candidate.retrieval_score is not None
    assert "rrf" in candidate.score_components
