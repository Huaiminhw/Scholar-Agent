from __future__ import annotations

import os
import re
from typing import Any, Literal

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
)
from ..normalizer import normalize_arxiv_id, normalize_doi, normalize_s2_paper

from .base import AcademicSourceAdapter


S2_FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "venue",
        "year",
        "authors",
        "citationCount",
        "publicationTypes",
        "fieldsOfStudy",
        "openAccessPdf",
    ]
)


class SemanticScholarAdapter(AcademicSourceAdapter):
    source = SearchSource.SEMANTIC_SCHOLAR
    capabilities = SourceCapabilities(
        relevance_search=True,
        title_search=True,
        identifier_lookup=True,
        author_search=True,
        filter_only=False,
        citation_search=True,
        supported_filters={"year", "venue", "publication_type", "field_of_study", "min_citations"},
    )

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 15.0,
        user_agent: str | None = None,
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            user_agent=user_agent or os.getenv("RETRIEVAL_USER_AGENT", "ScholarAgent/1.0"),
        )
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        if route.strategy == RetrievalStrategy.IDENTIFIER:
            return await self._identifier_lookup(route)
        if route.strategy == RetrievalStrategy.AUTHOR:
            return await self._author_search(route)
        if route.strategy == RetrievalStrategy.FILTER_ONLY:
            raise ValueError("Semantic Scholar V1 adapter does not use filter-only routes")

        params: dict[str, Any] = {
            "query": route.query_text,
            "limit": route.limit,
            "fields": S2_FIELDS,
        }
        self._apply_filters(params, route)
        response = await self._get(f"{self.BASE_URL}/paper/search", params=params, headers=self._headers())
        payload = response.json()
        raws = payload.get("data", []) or []
        records = [normalize_s2_paper(x) for x in raws]

        # TITLE 路线不在 API 端冒险做硬 exact；保留 API 召回后由 Candidate/Selector 判断。
        hits = [self._to_hit(route, r, i, raw) for i, (r, raw) in enumerate(zip(records, raws), 1)]
        return AdapterSearchResult(hits=hits, api_calls=1)

    async def _identifier_lookup(self, route: RouteSpec) -> AdapterSearchResult:
        identifier = self._s2_identifier(route.query_text)
        response = await self._get(
            f"{self.BASE_URL}/paper/{identifier}",
            params={"fields": S2_FIELDS},
            headers=self._headers(),
        )
        raw = response.json()
        record = normalize_s2_paper(raw)
        return AdapterSearchResult(hits=[self._to_hit(route, record, 1, raw)], api_calls=1)

    async def _author_search(self, route: RouteSpec) -> AdapterSearchResult:
        author_name = route.filters.authors[0] if route.filters.authors else route.query_text
        author_resp = await self._get(
            f"{self.BASE_URL}/author/search",
            params={"query": author_name, "limit": 3, "fields": "name,paperCount"},
            headers=self._headers(),
        )
        authors = author_resp.json().get("data", []) or []
        if not authors:
            return AdapterSearchResult(hits=[], api_calls=1)

        # 为避免同名作者爆炸，V1 只取 API 最靠前的作者实体；Candidate/Selector 仍负责最终判断。
        author_id = authors[0].get("authorId")
        if not author_id:
            return AdapterSearchResult(hits=[], api_calls=1, warnings=["S2 author search returned no authorId"])

        papers_resp = await self._get(
            f"{self.BASE_URL}/author/{author_id}/papers",
            params={"limit": route.limit, "fields": S2_FIELDS},
            headers=self._headers(),
        )
        raws = papers_resp.json().get("data", []) or []
        records = [normalize_s2_paper(x) for x in raws]
        hits = [self._to_hit(route, r, i, raw) for i, (r, raw) in enumerate(zip(records, raws), 1)]
        return AdapterSearchResult(hits=hits, api_calls=2)

    async def fetch_details(self, ids: list[str]) -> AdapterFetchResult:
        unique = list(dict.fromkeys(self._s2_identifier(x) for x in ids if x))
        if not unique:
            return AdapterFetchResult(records=[], api_calls=0)

        records = []
        warnings: list[str] = []
        api_calls = 0
        for i in range(0, len(unique), 100):
            batch = unique[i : i + 100]
            response = await self._post_json(
                f"{self.BASE_URL}/paper/batch",
                {"ids": batch},
                params={"fields": S2_FIELDS},
                headers=self._headers(),
            )
            api_calls += 1
            payload = response.json()
            for raw in payload or []:
                if raw:
                    records.append(normalize_s2_paper(raw))
                else:
                    warnings.append("S2 batch returned null paper record")
        return AdapterFetchResult(records=records, api_calls=api_calls, warnings=warnings)

    async def citation_neighbors(
        self,
        seed: CitationLookupKey,
        *,
        direction: Literal["forward", "backward"],
        limit: int,
    ) -> CitationNeighborResult:
        identifier = self._citation_identifier(seed)
        if not identifier or limit <= 0:
            return CitationNeighborResult(
                warnings=[f"S2 could not resolve citation seed {seed.candidate_id}"]
            )
        endpoint = "citations" if direction == "forward" else "references"
        paper_field = "citingPaper" if direction == "forward" else "citedPaper"
        response = await self._get(
            f"{self.BASE_URL}/paper/{identifier}/{endpoint}",
            params={"limit": min(limit, 1000), "fields": S2_FIELDS},
            headers=self._headers(),
        )
        rows = response.json().get("data", []) or []
        edges: list[CitationEdge] = []
        for row in rows:
            raw = row.get(paper_field) if isinstance(row, dict) else None
            if not isinstance(raw, dict) or not raw.get("paperId"):
                continue
            contexts = row.get("contexts") or []
            edges.append(
                CitationEdge(
                    source=self.source,
                    direction=direction,
                    seed_candidate_id=seed.candidate_id,
                    target=normalize_s2_paper(raw),
                    is_influential=bool(row.get("isInfluential")),
                    num_contexts=max(1, len(contexts)),
                )
            )
        return CitationNeighborResult(edges=edges, api_calls=1)

    def _apply_filters(self, params: dict[str, Any], route: RouteSpec) -> None:
        f = route.filters
        if f.year_start is not None or f.year_end is not None:
            if f.year_start is not None and f.year_end is not None:
                params["year"] = f"{f.year_start}-{f.year_end}"
            elif f.year_start is not None:
                params["year"] = f"{f.year_start}-"
            else:
                params["year"] = f"-{f.year_end}"
        if f.venues:
            params["venue"] = ",".join(f.venues)
        if f.publication_types:
            params["publicationTypes"] = ",".join(f.publication_types)
        if f.fields_of_study:
            params["fieldsOfStudy"] = ",".join(f.fields_of_study)
        if f.min_citations is not None:
            params["minCitationCount"] = f.min_citations

    @staticmethod
    def _s2_identifier(value: str) -> str:
        text = value.strip()
        doi = normalize_doi(text)
        if doi:
            return f"DOI:{doi}"
        arxiv = normalize_arxiv_id(text)
        if arxiv and ("arxiv" in text.casefold() or re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", text, re.I)):
            return f"ARXIV:{arxiv}"
        if text.upper().startswith(("DOI:", "ARXIV:", "CORPUSID:", "PMID:", "MAG:", "ACL:")):
            return text
        return text

    @classmethod
    def _citation_identifier(cls, seed: CitationLookupKey) -> str | None:
        if seed.semantic_scholar_id:
            return seed.semantic_scholar_id
        if seed.doi:
            return f"DOI:{seed.doi}"
        if seed.arxiv_id:
            return f"ARXIV:{seed.arxiv_id}"
        return None

    @staticmethod
    def _to_hit(route: RouteSpec, record, rank: int, raw: dict[str, Any]) -> RetrievalHit:
        score = raw.get("matchScore")
        return RetrievalHit(
            **record.model_dump(),
            origin=RetrievalOrigin(
                source=SearchSource.SEMANTIC_SCHOLAR,
                route_id=route.route_id,
                query_id=route.query_id,
                query_text=route.query_text,
                query_kind=route.query_kind,
                strategy=route.strategy,
                logic_branch_id=route.logic_branch_id,
                rank=rank,
                source_score=float(score) if score is not None else None,
                applied_constraint_ids=list(route.pushed_constraint_ids),
            ),
        )


__all__ = ["SemanticScholarAdapter"]
