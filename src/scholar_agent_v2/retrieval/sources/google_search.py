from __future__ import annotations

import os
import re
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
    SourcePaperRecord,
)
from ..normalizer import normalize_arxiv_id, normalize_doi
from .base import AcademicSourceAdapter, SourceConfigurationError


class GoogleSearchAdapter(AcademicSourceAdapter):
    """Google Programmable Search / Custom Search JSON API adapter.

    Google Search returns webpages rather than a scholarly-work schema.  The
    adapter therefore extracts citation metadata from ``pagemap.metatags``
    when available and keeps unsupported hard filters deferred for Candidate
    Pool evaluation.
    """

    source = SearchSource.GOOGLE_SEARCH
    capabilities = SourceCapabilities(
        relevance_search=True,
        title_search=True,
        identifier_lookup=True,
        author_search=True,
        filter_only=False,
        supported_filters=set(),
    )

    BASE_URL = "https://customsearch.googleapis.com/customsearch/v1"
    PAGE_SIZE = 10
    MAX_RESULTS = 100

    def __init__(
        self,
        api_key: str | None = None,
        engine_id: str | None = None,
        *,
        timeout_seconds: float = 20.0,
        user_agent: str | None = None,
        query_suffix: str | None = None,
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            user_agent=user_agent or os.getenv("RETRIEVAL_USER_AGENT", "ScholarAgent/1.0"),
        )
        self.api_key = api_key or os.getenv("GOOGLE_SEARCH_API_KEY")
        self.engine_id = engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        configured_suffix = (
            query_suffix if query_suffix is not None else os.getenv("GOOGLE_SEARCH_QUERY_SUFFIX", "")
        )
        self.query_suffix = configured_suffix.strip()

    def _auth_params(self) -> dict[str, str]:
        missing = []
        if not self.api_key:
            missing.append("GOOGLE_SEARCH_API_KEY")
        if not self.engine_id:
            missing.append("GOOGLE_SEARCH_ENGINE_ID")
        if missing:
            raise SourceConfigurationError(
                f"Google Search configuration is missing: {', '.join(missing)}"
            )
        return {"key": self.api_key, "cx": self.engine_id}

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        query = self._build_query(route)
        if not query:
            return AdapterSearchResult(
                hits=[],
                api_calls=0,
                warnings=["Google Search route produced an empty query."],
            )

        target = min(route.limit, self.MAX_RESULTS)
        hits: list[RetrievalHit] = []
        api_calls = 0
        seen_urls: set[str] = set()
        next_start = 1

        while len(hits) < target and next_start <= self.MAX_RESULTS:
            page_size = min(self.PAGE_SIZE, target - len(hits))
            params: dict[str, Any] = {
                **self._auth_params(),
                "q": query,
                "start": next_start,
                "num": page_size,
                "safe": "off",
            }
            language = self._language_restrict(route.filters.language)
            if language:
                params["lr"] = language

            response = await self._get(self.BASE_URL, params=params)
            api_calls += 1
            payload = response.json()
            items = payload.get("items") or []
            if not items:
                break
            next_start += len(items)

            added = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("link") or "").strip()
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                record = self._normalize_item(item)
                hits.append(self._to_hit(route, record, len(hits) + 1))
                added += 1
                if len(hits) >= target:
                    break

            # Google may return fewer than requested at the end of the result
            # set.  Stop instead of repeatedly requesting the same page.
            if len(items) < page_size or added == 0:
                break

        warnings = []
        if route.limit > self.MAX_RESULTS:
            warnings.append(
                f"Google Search caps a query at {self.MAX_RESULTS} results; "
                f"requested {route.limit}."
            )
        return AdapterSearchResult(hits=hits, api_calls=api_calls, warnings=warnings)

    async def fetch_details(self, ids: list[str]) -> AdapterFetchResult:
        # Custom Search has no scholarly-record detail endpoint.  Returning an
        # explicit no-op prevents enrichment from pretending metadata was fetched.
        if not ids:
            return AdapterFetchResult(records=[], api_calls=0)
        return AdapterFetchResult(
            records=[],
            api_calls=0,
            warnings=["Google Search does not provide a paper detail endpoint."],
        )

    def _build_query(self, route: RouteSpec) -> str:
        text = route.query_text.replace('"', " ").strip()
        if route.strategy == RetrievalStrategy.TITLE:
            query = f'"{text}"'
        elif route.strategy == RetrievalStrategy.AUTHOR:
            author = route.filters.authors[0] if route.filters.authors else text
            author = author.replace('"', " ").strip()
            if text and text.casefold() != author.casefold():
                query = f'{text} "{author}"'
            else:
                query = f'"{author}"'
        elif route.strategy in {
            RetrievalStrategy.RELEVANCE,
            RetrievalStrategy.IDENTIFIER,
        }:
            query = text
        else:
            raise ValueError(f"Unsupported Google Search strategy: {route.strategy}")

        if self.query_suffix:
            query = f"{query} {self.query_suffix}".strip()
        return query

    @classmethod
    def _normalize_item(cls, item: dict[str, Any]) -> SourcePaperRecord:
        pagemap = item.get("pagemap") or {}
        raw_metatags = pagemap.get("metatags") or []
        meta: dict[str, Any] = {}
        for entry in raw_metatags:
            if isinstance(entry, dict):
                for key, value in entry.items():
                    meta.setdefault(str(key).casefold(), value)

        link = cls._clean_text(item.get("link"))
        title = cls._first_text(
            meta,
            "citation_title",
            "dc.title",
            "og:title",
        ) or cls._clean_text(item.get("title"))
        abstract = cls._first_text(
            meta,
            "citation_abstract",
            "dc.description",
            "description",
            "og:description",
        ) or cls._clean_text(item.get("snippet"))

        authors = cls._extract_authors(meta, pagemap)
        date_text = cls._first_text(
            meta,
            "citation_publication_date",
            "citation_date",
            "dc.date",
            "article:published_time",
            "date",
        )
        year = cls._extract_year(date_text)
        venue = cls._first_text(
            meta,
            "citation_journal_title",
            "citation_conference_title",
            "citation_publisher",
        )
        doi = normalize_doi(
            cls._first_text(meta, "citation_doi", "dc.identifier", "doi")
            or link
            or abstract
        )
        arxiv_id = normalize_arxiv_id(link) or normalize_arxiv_id(
            cls._first_text(meta, "citation_arxiv_id", "arxiv_id")
        )
        pdf_url = cls._first_text(meta, "citation_pdf_url")
        urls = cls._dedup([link, pdf_url])
        source_id = cls._clean_text(item.get("cacheId")) or link or doi or arxiv_id or "unknown"

        return SourcePaperRecord(
            source=SearchSource.GOOGLE_SEARCH,
            source_id=source_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            arxiv_id=arxiv_id,
            citation_count=None,
            publication_types=[],
            fields_of_study=[],
            language=None,
            urls=urls,
            raw=item,
        )

    @staticmethod
    def _to_hit(route: RouteSpec, record: SourcePaperRecord, rank: int) -> RetrievalHit:
        return RetrievalHit(
            **record.model_dump(),
            origin=RetrievalOrigin(
                source=SearchSource.GOOGLE_SEARCH,
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

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text or None

    @classmethod
    def _first_text(cls, mapping: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = cls._clean_text(mapping.get(key.casefold()))
            if value:
                return value
        return None

    @classmethod
    def _extract_authors(cls, meta: dict[str, Any], pagemap: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("citation_author", "dc.creator", "author"):
            value = cls._clean_text(meta.get(key))
            if value:
                values.extend(part.strip() for part in value.split(";") if part.strip())
        for person in pagemap.get("person") or []:
            if isinstance(person, dict):
                name = cls._clean_text(person.get("name"))
                if name:
                    values.append(name)
        return cls._dedup(values)

    @staticmethod
    def _extract_year(value: str | None) -> int | None:
        if not value:
            return None
        match = re.search(r"\b(19|20)\d{2}\b", value)
        return int(match.group(0)) if match else None

    @staticmethod
    def _language_restrict(language: str | None) -> str | None:
        if not language:
            return None
        value = language.strip().casefold().replace("_", "-")
        mapping = {
            "en": "lang_en",
            "english": "lang_en",
            "zh": "lang_zh-CN",
            "zh-cn": "lang_zh-CN",
            "chinese": "lang_zh-CN",
            "中文": "lang_zh-CN",
        }
        return mapping.get(value)

    @classmethod
    def _dedup(cls, values: list[str | None]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            text = cls._clean_text(value)
            if text and text.casefold() not in seen:
                seen.add(text.casefold())
                output.append(text)
        return output


__all__ = ["GoogleSearchAdapter"]
