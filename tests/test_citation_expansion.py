from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path

import pytest

from scholar_agent_v2.candidate_pool.models import CandidatePaper, PaperIdentifiers
from scholar_agent_v2.candidate_pool.pool import CandidatePool
from scholar_agent_v2.query_understanding.schema import (
    ExtractedProperties,
    QueryAnalysisResult,
    QueryIntent,
    QueryType,
    RetrievalMode,
)
from scholar_agent_v2.retrieval.citation_expansion import (
    CitationExpansionConfig,
    CitationSnowballExpander,
)
from scholar_agent_v2.retrieval.models import (
    AdapterFetchResult,
    AdapterSearchResult,
    CitationEdge,
    CitationLookupKey,
    CitationNeighborResult,
    SearchSource,
    SourceCapabilities,
    SourcePaperRecord,
)
from scholar_agent_v2.retrieval.sources.base import AcademicSourceAdapter
from scholar_agent_v2.retrieval.sources.openalex import OpenAlexAdapter
from scholar_agent_v2.retrieval.sources.pasa_citations import PaSaCitationGraph
from scholar_agent_v2.retrieval.sources.semantic_scholar import SemanticScholarAdapter


def test_pasa_sections_build_backward_and_forward_graph(tmp_path: Path) -> None:
    paper_zip = tmp_path / "papers.zip"
    id_map = tmp_path / "id2paper.json"
    graph_path = tmp_path / "citations.sqlite3"
    records = {
        "a": {
            "title": "Paper A",
            "abstract": "",
            "sections": {"Related Work": ["Paper B", "Paper C"]},
            "source": [],
        },
        "b": {
            "title": "Paper B",
            "abstract": "",
            "sections": {},
            "source": [],
        },
        "c": {
            "title": "Paper C",
            "abstract": "",
            "sections": {},
            "source": [],
        },
        "d": {
            "title": "Paper D",
            "abstract": "",
            "sections": {"Intro": ["Paper B", "Unknown Paper"]},
            "source": [],
        },
    }
    with zipfile.ZipFile(paper_zip, "w") as archive:
        for name, record in records.items():
            archive.writestr(name, json.dumps(record))
    id_map.write_text(
        json.dumps(
            {
                "2401.00001": "Paper A",
                "2401.00002": "Paper B",
                "2401.00003": "Paper C",
                "2401.00004": "Paper D",
            }
        ),
        encoding="utf-8",
    )

    graph = PaSaCitationGraph(
        paper_zip_path=paper_zip,
        id_map_path=id_map,
        graph_path=graph_path,
    )
    stats = graph.build(progress=None)

    assert stats.complete is True
    assert stats.resolved_edges == 3
    assert stats.unresolved_reference_titles == 1
    assert graph.is_ready()
    assert graph.neighbors(
        "2401.00001", direction="backward", limit=10
    ) == ["2401.00002", "2401.00003"]
    assert graph.neighbors(
        "2401.00002", direction="forward", limit=10
    ) == ["2401.00001", "2401.00004"]


class _CitationAdapter(AcademicSourceAdapter):
    source = SearchSource.SEMANTIC_SCHOLAR
    capabilities = SourceCapabilities(
        relevance_search=True,
        citation_search=True,
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    async def search(self, route) -> AdapterSearchResult:
        return AdapterSearchResult()

    async def fetch_details(self, ids: list[str]) -> AdapterFetchResult:
        return AdapterFetchResult()

    async def citation_neighbors(
        self,
        seed: CitationLookupKey,
        *,
        direction: str,
        limit: int,
    ) -> CitationNeighborResult:
        self.calls.append((seed.candidate_id, direction))
        shared = SourcePaperRecord(
            source=self.source,
            source_id="shared",
            title="Shared Citation Candidate",
            semantic_scholar_id="shared",
        )
        unique = SourcePaperRecord(
            source=self.source,
            source_id=f"unique-{seed.candidate_id}",
            title=f"Unique for {seed.candidate_id}",
            semantic_scholar_id=f"unique-{seed.candidate_id}",
        )
        return CitationNeighborResult(
            edges=[
                CitationEdge(
                    source=self.source,
                    direction=direction,
                    seed_candidate_id=seed.candidate_id,
                    target=shared,
                    is_influential=True,
                ),
                CitationEdge(
                    source=self.source,
                    direction=direction,
                    seed_candidate_id=seed.candidate_id,
                    target=unique,
                ),
            ],
            api_calls=1,
        )


def _seed(candidate_id: str, score: float) -> CandidatePaper:
    return CandidatePaper(
        candidate_id=candidate_id,
        title=candidate_id,
        identifiers=PaperIdentifiers(semantic_scholar_id=candidate_id),
        retrieval_score=score,
    )


def test_snowball_uses_relevant_seeds_and_aggregates_shared_candidate() -> None:
    adapter = _CitationAdapter()
    pool = CandidatePool(
        candidates=[
            _seed("seed-1", 0.80),
            _seed("seed-2", 0.70),
            _seed("below-threshold", 0.10),
        ]
    )
    analysis = QueryAnalysisResult(
        original_query="papers about agents",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(mode=RetrievalMode.COMPREHENSIVE),
        content="agents",
    )
    expander = CitationSnowballExpander(
        {SearchSource.SEMANTIC_SCHOLAR: adapter},
        config=CitationExpansionConfig(
            enabled=True,
            seed_top_k=2,
            min_seed_score=0.20,
            forward_top_k=3,
            backward_top_k=0,
            fetch_limit_per_seed=10,
        ),
    )

    output = asyncio.run(expander.run(pool, analysis))

    assert output.diagnostics.seed_candidate_ids == ["seed-1", "seed-2"]
    assert output.diagnostics.raw_edges == 4
    assert output.diagnostics.new_unique_candidates == 3
    assert output.diagnostics.api_calls == 2
    assert output.diagnostics.stop_reason == "citation_budget_reached"
    citation_pool = CandidatePool.from_hits(output.hits)
    ranked = sorted(
        citation_pool.candidates,
        key=lambda candidate: max(
            origin.source_score or 0.0 for origin in candidate.origins
        ),
        reverse=True,
    )
    shared = next(
        candidate
        for candidate in ranked
        if candidate.identifiers.semantic_scholar_id == "shared"
    )
    assert shared.route_count == 2
    assert {origin.citation_seed_id for origin in shared.origins} == {
        "seed-1",
        "seed-2",
    }


def test_citation_channel_weight_is_bounded() -> None:
    with pytest.raises(ValueError, match="channel_weight"):
        CitationExpansionConfig(channel_weight=1.01)


def test_recent_query_runs_forward_only() -> None:
    adapter = _CitationAdapter()
    pool = CandidatePool(candidates=[_seed("seed-1", 0.80)])
    analysis = QueryAnalysisResult(
        original_query="latest agent papers",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(mode=RetrievalMode.LATEST),
        content="agents",
        properties=ExtractedProperties(recent_first=True),
    )
    output = asyncio.run(
        CitationSnowballExpander(
            {SearchSource.SEMANTIC_SCHOLAR: adapter},
            config=CitationExpansionConfig(
                enabled=True,
                seed_top_k=1,
                forward_top_k=5,
                backward_top_k=5,
            ),
        ).run(pool, analysis)
    )

    assert adapter.calls == [("seed-1", "forward")]
    assert [item.direction for item in output.diagnostics.directions] == ["forward"]


class _JSONResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def json(self):
        return self.payload


class _FakeSemanticScholar(SemanticScholarAdapter):
    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.requests = []

    async def _get(self, url, *, params=None, headers=None):
        self.requests.append((url, params))
        return _JSONResponse(
            {
                "data": [
                    {
                        "isInfluential": True,
                        "contexts": ["context"],
                        "citingPaper": {
                            "paperId": "target-s2",
                            "title": "Forward Target",
                            "externalIds": {"ArXiv": "2401.12345"},
                            "authors": [],
                        },
                    }
                ]
            }
        )


def test_semantic_scholar_forward_citation_response_is_normalized() -> None:
    adapter = _FakeSemanticScholar()
    result = asyncio.run(
        adapter.citation_neighbors(
            CitationLookupKey(
                candidate_id="seed",
                arxiv_id="2301.00001",
            ),
            direction="forward",
            limit=20,
        )
    )

    assert result.api_calls == 1
    assert len(result.edges) == 1
    assert result.edges[0].is_influential is True
    assert result.edges[0].target.semantic_scholar_id == "target-s2"
    assert "/paper/ARXIV:2301.00001/citations" in adapter.requests[0][0]


class _FakeOpenAlex(OpenAlexAdapter):
    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.requests = []

    async def _get(self, url, *, params=None, headers=None):
        self.requests.append((url, params))
        if "/works/WSEED" in url:
            return _JSONResponse(
                {
                    "id": "https://openalex.org/WSEED",
                    "display_name": "Seed",
                    "referenced_works": ["https://openalex.org/WTARGET"],
                }
            )
        return _JSONResponse(
            {
                "results": [
                    {
                        "id": "https://openalex.org/WTARGET",
                        "display_name": "Backward Target",
                        "authorships": [],
                        "ids": {},
                    }
                ]
            }
        )


def test_openalex_backward_references_are_fetched_by_work_id() -> None:
    adapter = _FakeOpenAlex()
    result = asyncio.run(
        adapter.citation_neighbors(
            CitationLookupKey(candidate_id="seed", openalex_id="WSEED"),
            direction="backward",
            limit=10,
        )
    )

    assert result.api_calls == 2
    assert [edge.target.openalex_id for edge in result.edges] == ["WTARGET"]
    assert adapter.requests[1][1]["filter"] == "openalex_id:WTARGET"
