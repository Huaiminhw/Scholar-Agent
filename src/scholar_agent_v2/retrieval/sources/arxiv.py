from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from ..models import (
    AdapterFetchResult,
    AdapterSearchResult,
    RetrievalHit,
    RetrievalOrigin,
    RetrievalStrategy,
    RouteSpec,
    SearchSource,
    SourceCapabilities,
)
from ..normalizer import normalize_arxiv_entry, normalize_arxiv_id

from .base import AcademicSourceAdapter


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


class ArxivAdapter(AcademicSourceAdapter):
    source = SearchSource.ARXIV
    capabilities = SourceCapabilities(
        relevance_search=True,
        title_search=True,
        identifier_lookup=True,
        author_search=True,
        filter_only=False,
        supported_filters=set(),
    )

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        user_agent: str | None = None,
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            user_agent=user_agent or os.getenv("RETRIEVAL_USER_AGENT", "ScholarAgent/1.0"),
        )

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        if route.strategy == RetrievalStrategy.IDENTIFIER:
            arxiv_id = normalize_arxiv_id(route.query_text)
            if not arxiv_id:
                return AdapterSearchResult(hits=[], api_calls=0, warnings=["Query is not an arXiv identifier"])
            params = {"id_list": arxiv_id, "max_results": 1}
        else:
            query = self._build_query(route)
            params = {
                "search_query": query,
                "start": 0,
                "max_results": route.limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

        response = await self._get(self.BASE_URL, params=params, headers={"Accept": "application/atom+xml"})
        entries = self._parse_feed(response.body)
        records = [normalize_arxiv_entry(x) for x in entries]
        hits = [self._to_hit(route, r, i) for i, r in enumerate(records, 1)]
        return AdapterSearchResult(hits=hits, api_calls=1)

    async def fetch_details(self, ids: list[str]) -> AdapterFetchResult:
        unique = list(dict.fromkeys(x for x in (normalize_arxiv_id(v) for v in ids) if x))
        if not unique:
            return AdapterFetchResult(records=[], api_calls=0)

        records = []
        api_calls = 0
        for i in range(0, len(unique), 50):
            batch = unique[i : i + 50]
            response = await self._get(
                self.BASE_URL,
                params={"id_list": ",".join(batch), "max_results": len(batch)},
                headers={"Accept": "application/atom+xml"},
            )
            api_calls += 1
            records.extend(normalize_arxiv_entry(x) for x in self._parse_feed(response.body))
        return AdapterFetchResult(records=records, api_calls=api_calls)

    def _build_query(self, route: RouteSpec) -> str:
        if route.strategy == RetrievalStrategy.TITLE:
            return f'ti:"{self._escape(route.query_text)}"'
        if route.strategy == RetrievalStrategy.AUTHOR:
            author = route.filters.authors[0] if route.filters.authors else route.query_text
            return f'au:"{self._escape(author)}"'
        # 直接 all:<query>，让 arXiv parser 自己处理词项；不额外生成新语义。
        return f'all:"{self._escape(route.query_text)}"'

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace('"', " ").strip()

    @staticmethod
    def _parse_feed(body: bytes) -> list[dict[str, Any]]:
        root = ET.fromstring(body)
        out: list[dict[str, Any]] = []
        for entry in root.findall(f"{ATOM}entry"):
            authors = [
                (a.findtext(f"{ATOM}name") or "").strip()
                for a in entry.findall(f"{ATOM}author")
            ]
            categories = [c.attrib.get("term", "") for c in entry.findall(f"{ATOM}category")]
            urls = [link.attrib.get("href", "") for link in entry.findall(f"{ATOM}link") if link.attrib.get("href")]
            out.append(
                {
                    "id": entry.findtext(f"{ATOM}id"),
                    "title": entry.findtext(f"{ATOM}title"),
                    "summary": entry.findtext(f"{ATOM}summary"),
                    "published": entry.findtext(f"{ATOM}published"),
                    "updated": entry.findtext(f"{ATOM}updated"),
                    "authors": authors,
                    "categories": categories,
                    "doi": entry.findtext(f"{ARXIV}doi"),
                    "journal_ref": entry.findtext(f"{ARXIV}journal_ref"),
                    "urls": urls,
                }
            )
        return out

    @staticmethod
    def _to_hit(route: RouteSpec, record, rank: int) -> RetrievalHit:
        return RetrievalHit(
            **record.model_dump(),
            origin=RetrievalOrigin(
                source=SearchSource.ARXIV,
                route_id=route.route_id,
                query_id=route.query_id,
                query_text=route.query_text,
                query_kind=route.query_kind,
                strategy=route.strategy,
                logic_branch_id=route.logic_branch_id,
                rank=rank,
                source_score=None,
                applied_constraint_ids=list(route.pushed_constraint_ids),
            ),
        )


__all__ = ["ArxivAdapter"]
