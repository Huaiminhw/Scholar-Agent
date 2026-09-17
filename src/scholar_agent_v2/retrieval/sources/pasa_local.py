from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal

from ..models import (
    AdapterFetchResult,
    AdapterSearchResult,
    CitationEdge,
    CitationLookupKey,
    CitationNeighborResult,
    RetrievalHit,
    RetrievalOrigin,
    RetrievalStrategy,
    RouteSpec,
    SearchSource,
    SourceCapabilities,
    SourcePaperRecord,
)
from .base import AcademicSourceAdapter, SourceConfigurationError
from .pasa_corpus import PaSaBM25Result, PaSaCorpusBM25, normalize_arxiv_id
from .pasa_citations import PaSaCitationGraph


_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.([^\s]+)", re.IGNORECASE)
_ARXIV_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:arxiv:\s*)?((?:\d{4}\.\d{4,5})|(?:[a-z-]+/\d{7}))",
    re.IGNORECASE,
)


class PaSaLocalAdapter(AcademicSourceAdapter):
    """Formal retrieval adapter for the frozen local PaSa BM25 corpus."""

    source = SearchSource.PASA_LOCAL
    capabilities = SourceCapabilities(
        relevance_search=True,
        title_search=True,
        identifier_lookup=True,
        author_search=False,
        filter_only=False,
        citation_search=True,
        # The current SQLite search does not push metadata filters into SQL.
        supported_filters=set(),
    )

    def __init__(
        self,
        *,
        corpus: PaSaCorpusBM25 | None = None,
        paper_zip_path: str | Path | None = None,
        id_map_path: str | Path | None = None,
        index_path: str | Path | None = None,
        citation_graph_path: str | Path | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.corpus = corpus or PaSaCorpusBM25(
            paper_zip_path=paper_zip_path,
            id_map_path=id_map_path,
            index_path=index_path,
        )
        if not self.corpus.is_index_ready():
            raise SourceConfigurationError(
                f"PaSa local BM25 index is not ready: {self.corpus.index_path}"
            )
        self.citation_graph = PaSaCitationGraph(
            paper_zip_path=getattr(self.corpus, "paper_zip_path", None),
            id_map_path=getattr(self.corpus, "id_map_path", None),
            graph_path=citation_graph_path,
        )

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        if route.strategy == RetrievalStrategy.AUTHOR:
            return AdapterSearchResult(
                api_calls=0,
                warnings=["PaSa local corpus does not support author search."],
            )
        if route.strategy == RetrievalStrategy.FILTER_ONLY:
            return AdapterSearchResult(
                api_calls=0,
                warnings=["PaSa local corpus does not support filter-only search."],
            )

        if route.strategy == RetrievalStrategy.IDENTIFIER:
            identifier = self._extract_arxiv_id(route.query_text)
            item = (
                await asyncio.to_thread(self.corpus.get_by_arxiv_id, identifier)
                if identifier
                else None
            )
            results = [item] if item is not None else []
        else:
            results = await asyncio.to_thread(
                self.corpus.search,
                route.query_text,
                limit=route.limit,
                operator="or",
            )

        return AdapterSearchResult(
            hits=[self._to_hit(route, item) for item in results],
            api_calls=0,
        )

    async def fetch_details(self, ids: list[str]) -> AdapterFetchResult:
        records: list[SourcePaperRecord] = []
        for value in dict.fromkeys(ids):
            identifier = self._extract_arxiv_id(value)
            if not identifier:
                continue
            item = await asyncio.to_thread(self.corpus.get_by_arxiv_id, identifier)
            if item is not None:
                records.append(self._to_record(item))
        return AdapterFetchResult(records=records, api_calls=0)

    async def citation_neighbors(
        self,
        seed: CitationLookupKey,
        *,
        direction: Literal["forward", "backward"],
        limit: int,
    ) -> CitationNeighborResult:
        if not seed.arxiv_id:
            return CitationNeighborResult(
                warnings=[f"Citation seed {seed.candidate_id} has no arXiv ID"]
            )
        if not self.citation_graph.is_ready():
            return CitationNeighborResult(
                warnings=[
                    "PaSa citation graph is not ready; run "
                    "scripts/build_pasa_citation_graph.py"
                ]
            )
        neighbor_ids = await asyncio.to_thread(
            self.citation_graph.neighbors,
            seed.arxiv_id,
            direction=direction,
            limit=limit,
        )
        papers = await asyncio.to_thread(
            self.corpus.fetch_by_arxiv_ids,
            neighbor_ids,
        )
        by_id = {paper.arxiv_id: paper for paper in papers if paper.arxiv_id}
        edges: list[CitationEdge] = []
        for neighbor_id in neighbor_ids:
            paper = by_id.get(neighbor_id)
            if paper is None:
                continue
            edges.append(
                CitationEdge(
                    source=SearchSource.PASA_LOCAL,
                    direction=direction,
                    seed_candidate_id=seed.candidate_id,
                    target=self._to_record(paper),
                )
            )
        return CitationNeighborResult(edges=edges, api_calls=0)

    @staticmethod
    def _extract_arxiv_id(value: str) -> str | None:
        text = (value or "").strip()
        doi_match = _ARXIV_DOI_RE.search(text)
        if doi_match:
            text = doi_match.group(1)
        else:
            arxiv_match = _ARXIV_ID_RE.search(text)
            if arxiv_match:
                text = arxiv_match.group(1)
        return normalize_arxiv_id(text)

    @classmethod
    def _to_hit(cls, route: RouteSpec, item: PaSaBM25Result) -> RetrievalHit:
        return RetrievalHit(
            **cls._record_fields(item),
            origin=RetrievalOrigin(
                source=SearchSource.PASA_LOCAL,
                route_id=route.route_id,
                query_id=route.query_id,
                query_text=route.query_text,
                query_kind=route.query_kind,
                strategy=route.strategy,
                logic_branch_id=route.logic_branch_id,
                rank=item.rank,
                source_score=item.bm25_score,
                applied_constraint_ids=list(route.pushed_constraint_ids),
            ),
        )

    @classmethod
    def _to_record(cls, item: PaSaBM25Result) -> SourcePaperRecord:
        return SourcePaperRecord(**cls._record_fields(item))

    @staticmethod
    def _record_fields(item: PaSaBM25Result) -> dict:
        return {
            "source": SearchSource.PASA_LOCAL,
            "source_id": item.arxiv_id or item.corpus_key,
            "title": item.title,
            "abstract": item.abstract,
            "year": item.year,
            "arxiv_id": item.arxiv_id,
            "urls": (
                [f"https://arxiv.org/abs/{item.arxiv_id}"]
                if item.arxiv_id
                else []
            ),
            "raw": {"corpus_key": item.corpus_key, "sources": item.sources},
        }


__all__ = ["PaSaLocalAdapter"]
