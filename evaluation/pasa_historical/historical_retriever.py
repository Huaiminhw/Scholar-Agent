"""Historical wrappers for the three existing open-source adapters.

The wrappers live only in the PaSa evaluation lane.  Production adapters are
not modified.  Every accepted record must be strictly older than 2024-10-01.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, Mapping

from scholar_agent_v2.retrieval.models import (
    AdapterSearchResult,
    RetrievalStrategy,
    RouteSpec,
    SearchSource,
)
from scholar_agent_v2.retrieval.normalizer import (
    normalize_arxiv_entry,
    normalize_doi,
    normalize_openalex_work,
    normalize_s2_paper,
)
from scholar_agent_v2.retrieval.sources.arxiv import ArxivAdapter
from scholar_agent_v2.retrieval.sources.openalex import OpenAlexAdapter
from scholar_agent_v2.retrieval.sources.semantic_scholar import (
    S2_FIELDS,
    SemanticScholarAdapter,
)

from .config import HistoricalEvalConfig


HISTORICAL_S2_FIELDS = f"{S2_FIELDS},publicationDate"


@dataclass
class HistoricalRouteDiagnostics:
    route_id: str
    source: str
    cutoff_date: str
    server_filter: str
    pages_fetched: int = 0
    raw_records: int = 0
    accepted_records: int = 0
    post_cutoff_filtered: int = 0
    unknown_date_filtered: int = 0
    exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _HistoricalDiagnosticsMixin:
    cutoff_date: date
    diagnostics: dict[str, HistoricalRouteDiagnostics]

    def route_diagnostics(self, route_id: str) -> dict[str, Any] | None:
        item = self.diagnostics.get(route_id)
        return item.to_dict() if item is not None else None


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _raw_is_historical(
    raw: Mapping[str, Any],
    *,
    cutoff: date,
    date_keys: tuple[str, ...],
    year_key: str | None,
) -> tuple[bool, str]:
    for key in date_keys:
        parsed = _parse_date(raw.get(key))
        if parsed is not None:
            return parsed < cutoff, "date"
    year = raw.get(year_key) if year_key else None
    try:
        numeric_year = int(year) if year is not None else None
    except (TypeError, ValueError):
        numeric_year = None
    if numeric_year is not None and numeric_year < cutoff.year:
        return True, "year"
    if numeric_year is not None and numeric_year > cutoff.year:
        return False, "post_cutoff"
    # A 2024 record without month/day cannot prove it predates 2024-10-01.
    return False, "unknown_date"


class HistoricalOpenAlexAdapter(_HistoricalDiagnosticsMixin, OpenAlexAdapter):
    """OpenAlex wrapper with an exact server-side publication-date ceiling."""

    def __init__(self, *, cutoff_date: date, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cutoff_date = cutoff_date
        self.diagnostics = {}

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        if route.strategy == RetrievalStrategy.IDENTIFIER:
            text = route.query_text.strip()
            raw_records: list[dict[str, Any]] = []
            api_calls = 0
            warnings: list[str] = []
            doi = normalize_doi(text)
            oa_match = re.search(r"\bW\d+\b", text, re.IGNORECASE)
            if doi:
                target = f"https://doi.org/{doi}"
                try:
                    response = await self._get(
                        f"{self.BASE_URL}/works/{target}",
                        params=self._auth_params(),
                    )
                    raw_records = [response.json()]
                    api_calls = 1
                except Exception:
                    response = await self._get(
                        f"{self.BASE_URL}/works",
                        params={
                            **self._auth_params(),
                            "filter": f"doi:https://doi.org/{doi}",
                            "per-page": min(route.limit, 10),
                        },
                    )
                    raw_records = response.json().get("results", []) or []
                    api_calls = 2
                    warnings.append(
                        "OpenAlex DOI singleton failed; used DOI filter fallback."
                    )
            elif oa_match:
                oa_id = oa_match.group(0).upper()
                response = await self._get(
                    f"{self.BASE_URL}/works/{oa_id}",
                    params=self._auth_params(),
                )
                raw_records = [response.json()]
                api_calls = 1
            else:
                response = await self._get(
                    f"{self.BASE_URL}/works",
                    params={
                        **self._auth_params(),
                        "search.exact": text,
                        "per-page": min(route.limit, 10),
                    },
                )
                raw_records = response.json().get("results", []) or []
                api_calls = 1
                warnings.append(
                    "Identifier was not OpenAlex ID/DOI; used exact search fallback."
                )

            accepted_raws: list[dict[str, Any]] = []
            post_cutoff = 0
            unknown_date = 0
            for raw in raw_records:
                allowed, reason = _raw_is_historical(
                    raw,
                    cutoff=self.cutoff_date,
                    date_keys=("publication_date",),
                    year_key="publication_year",
                )
                if allowed:
                    accepted_raws.append(raw)
                elif reason == "unknown_date":
                    unknown_date += 1
                else:
                    post_cutoff += 1
            records = [normalize_openalex_work(raw) for raw in accepted_raws]
            accepted = [
                self._to_hit(route, record, rank, raw)
                for rank, (record, raw) in enumerate(
                    zip(records, accepted_raws), start=1
                )
            ]
            diag = HistoricalRouteDiagnostics(
                route_id=route.route_id,
                source=self.source.value,
                cutoff_date=self.cutoff_date.isoformat(),
                server_filter="identifier lookup + publication_date post-filter",
                pages_fetched=api_calls,
                raw_records=len(raw_records),
                accepted_records=len(accepted),
                post_cutoff_filtered=post_cutoff,
                unknown_date_filtered=unknown_date,
                exhausted=True,
            )
            self.diagnostics[route.route_id] = diag
            return AdapterSearchResult(
                hits=accepted,
                api_calls=api_calls,
                warnings=warnings,
            )

        cutoff_last_day = self.cutoff_date - timedelta(days=1)
        params: dict[str, Any] = {**self._auth_params(), "per-page": route.limit}
        filters = self._compile_filters(route)
        filters = [item for item in filters if not item.startswith("to_publication_date:")]
        filters.append(f"to_publication_date:{cutoff_last_day.isoformat()}")
        if filters:
            params["filter"] = ",".join(filters)

        if route.strategy == RetrievalStrategy.RELEVANCE:
            params["search"] = route.query_text
        elif route.strategy == RetrievalStrategy.TITLE:
            params["search.exact"] = route.query_text
        elif route.strategy == RetrievalStrategy.AUTHOR:
            author = route.filters.authors[0] if route.filters.authors else route.query_text
            author_expr = f'raw_author_name.search:"{self._escape_filter_phrase(author)}"'
            params["filter"] = ",".join(
                item for item in [params.get("filter"), author_expr] if item
            )
            if route.query_text and route.query_text.casefold() != author.casefold():
                params["search"] = route.query_text
        elif route.strategy != RetrievalStrategy.FILTER_ONLY:
            raise ValueError(f"Unsupported OpenAlex strategy: {route.strategy}")

        response = await self._get(f"{self.BASE_URL}/works", params=params)
        payload = response.json()
        raws = payload.get("results", []) or []
        accepted_raws: list[dict[str, Any]] = []
        post_cutoff = 0
        unknown_date = 0
        for raw in raws:
            allowed, reason = _raw_is_historical(
                raw,
                cutoff=self.cutoff_date,
                date_keys=("publication_date",),
                year_key="publication_year",
            )
            if allowed:
                accepted_raws.append(raw)
            elif reason == "unknown_date":
                unknown_date += 1
            else:
                post_cutoff += 1
        records = [normalize_openalex_work(raw) for raw in accepted_raws]
        hits = [
            self._to_hit(route, record, rank, raw)
            for rank, (record, raw) in enumerate(zip(records, accepted_raws), start=1)
        ]
        self.diagnostics[route.route_id] = HistoricalRouteDiagnostics(
            route_id=route.route_id,
            source=self.source.value,
            cutoff_date=self.cutoff_date.isoformat(),
            server_filter=f"to_publication_date:{cutoff_last_day.isoformat()}",
            pages_fetched=1,
            raw_records=len(raws),
            accepted_records=len(hits),
            post_cutoff_filtered=post_cutoff,
            unknown_date_filtered=unknown_date,
            exhausted=len(raws) < route.limit,
        )
        return AdapterSearchResult(hits=hits, api_calls=1)


class HistoricalSemanticScholarAdapter(
    _HistoricalDiagnosticsMixin, SemanticScholarAdapter
):
    """S2 wrapper: server-side year ceiling plus date filtering and pagination."""

    def __init__(
        self,
        *,
        cutoff_date: date,
        page_size: int,
        max_pages: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cutoff_date = cutoff_date
        self.page_size = page_size
        self.max_pages = max_pages
        self.diagnostics = {}

    def _historical_year_filter(self, route: RouteSpec) -> str | None:
        start = route.filters.year_start
        end = min(route.filters.year_end or self.cutoff_date.year, self.cutoff_date.year)
        if start is not None and start > end:
            return None
        return f"{start or ''}-{end}"

    async def _paged_search(
        self,
        route: RouteSpec,
        *,
        endpoint: str,
        base_params: dict[str, Any],
        initial_api_calls: int = 0,
        server_filter: str,
    ) -> AdapterSearchResult:
        accepted_raws: list[dict[str, Any]] = []
        offset = 0
        pages = 0
        raw_count = 0
        post_cutoff = 0
        unknown_date = 0
        exhausted = False

        while len(accepted_raws) < route.limit and pages < self.max_pages:
            # Fetch a full page even when only a few accepted records are still
            # needed.  Otherwise repeated post-cutoff removals can degrade into
            # many tiny requests and hit max_pages before Top-K is replenished.
            request_limit = self.page_size
            params = {
                **base_params,
                "limit": request_limit,
                "offset": offset,
                "fields": HISTORICAL_S2_FIELDS,
            }
            response = await self._get(endpoint, params=params, headers=self._headers())
            payload = response.json()
            raws = payload.get("data", []) or []
            pages += 1
            raw_count += len(raws)
            for raw in raws:
                allowed, reason = _raw_is_historical(
                    raw,
                    cutoff=self.cutoff_date,
                    date_keys=("publicationDate",),
                    year_key="year",
                )
                if allowed:
                    accepted_raws.append(raw)
                    if len(accepted_raws) >= route.limit:
                        break
                elif reason == "unknown_date":
                    unknown_date += 1
                else:
                    post_cutoff += 1

            next_offset = payload.get("next")
            if not raws or next_offset is None:
                exhausted = True
                break
            try:
                next_offset_int = int(next_offset)
            except (TypeError, ValueError):
                next_offset_int = offset + len(raws)
            if next_offset_int <= offset:
                exhausted = True
                break
            offset = next_offset_int

        accepted_raws = accepted_raws[: route.limit]
        records = [normalize_s2_paper(raw) for raw in accepted_raws]
        hits = [
            self._to_hit(route, record, rank, raw)
            for rank, (record, raw) in enumerate(zip(records, accepted_raws), start=1)
        ]
        diag = HistoricalRouteDiagnostics(
            route_id=route.route_id,
            source=self.source.value,
            cutoff_date=self.cutoff_date.isoformat(),
            server_filter=server_filter,
            pages_fetched=pages + initial_api_calls,
            raw_records=raw_count,
            accepted_records=len(hits),
            post_cutoff_filtered=post_cutoff,
            unknown_date_filtered=unknown_date,
            exhausted=exhausted,
        )
        self.diagnostics[route.route_id] = diag
        warnings: list[str] = []
        if len(hits) < route.limit and not exhausted and pages >= self.max_pages:
            warnings.append(
                f"S2 historical pagination reached max_pages={self.max_pages} "
                f"with {len(hits)}/{route.limit} accepted hits."
            )
        return AdapterSearchResult(
            hits=hits,
            api_calls=pages + initial_api_calls,
            warnings=warnings,
        )

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        if route.strategy == RetrievalStrategy.IDENTIFIER:
            identifier = self._s2_identifier(route.query_text)
            response = await self._get(
                f"{self.BASE_URL}/paper/{identifier}",
                params={"fields": HISTORICAL_S2_FIELDS},
                headers=self._headers(),
            )
            raw = response.json()
            allowed, reason = _raw_is_historical(
                raw,
                cutoff=self.cutoff_date,
                date_keys=("publicationDate",),
                year_key="year",
            )
            hits = []
            if allowed:
                record = normalize_s2_paper(raw)
                hits = [self._to_hit(route, record, 1, raw)]
            self.diagnostics[route.route_id] = HistoricalRouteDiagnostics(
                route_id=route.route_id,
                source=self.source.value,
                cutoff_date=self.cutoff_date.isoformat(),
                server_filter="identifier lookup + publicationDate post-filter",
                pages_fetched=1,
                raw_records=1,
                accepted_records=len(hits),
                post_cutoff_filtered=int(not allowed and reason != "unknown_date"),
                unknown_date_filtered=int(reason == "unknown_date"),
                exhausted=True,
            )
            return AdapterSearchResult(hits=hits, api_calls=1)

        if route.strategy == RetrievalStrategy.AUTHOR:
            author = route.filters.authors[0] if route.filters.authors else route.query_text
            author_response = await self._get(
                f"{self.BASE_URL}/author/search",
                params={"query": author, "limit": 3, "fields": "name,paperCount"},
                headers=self._headers(),
            )
            authors = author_response.json().get("data", []) or []
            if not authors or not authors[0].get("authorId"):
                self.diagnostics[route.route_id] = HistoricalRouteDiagnostics(
                    route_id=route.route_id,
                    source=self.source.value,
                    cutoff_date=self.cutoff_date.isoformat(),
                    server_filter="author lookup + publicationDate post-filter",
                    pages_fetched=1,
                    exhausted=True,
                )
                return AdapterSearchResult(hits=[], api_calls=1)
            author_id = authors[0]["authorId"]
            return await self._paged_search(
                route,
                endpoint=f"{self.BASE_URL}/author/{author_id}/papers",
                base_params={},
                initial_api_calls=1,
                server_filter="author papers + publicationDate post-filter",
            )

        if route.strategy == RetrievalStrategy.FILTER_ONLY:
            raise ValueError("Semantic Scholar does not support filter-only routes")

        year_filter = self._historical_year_filter(route)
        if year_filter is None:
            self.diagnostics[route.route_id] = HistoricalRouteDiagnostics(
                route_id=route.route_id,
                source=self.source.value,
                cutoff_date=self.cutoff_date.isoformat(),
                server_filter="empty year intersection",
                exhausted=True,
            )
            return AdapterSearchResult(hits=[], api_calls=0)

        params: dict[str, Any] = {"query": route.query_text}
        self._apply_filters(params, route)
        params["year"] = year_filter
        return await self._paged_search(
            route,
            endpoint=f"{self.BASE_URL}/paper/search",
            base_params=params,
            server_filter=f"year={year_filter}; publicationDate<{self.cutoff_date.isoformat()}",
        )


class HistoricalArxivAdapter(_HistoricalDiagnosticsMixin, ArxivAdapter):
    """arXiv wrapper with submittedDate filtering in the server query."""

    def __init__(self, *, cutoff_date: date, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cutoff_date = cutoff_date
        self.diagnostics = {}

    @property
    def submitted_date_filter(self) -> str:
        last_day = self.cutoff_date - timedelta(days=1)
        return f"submittedDate:[000000000000 TO {last_day:%Y%m%d}2359]"

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        if route.strategy == RetrievalStrategy.IDENTIFIER:
            from scholar_agent_v2.retrieval.normalizer import normalize_arxiv_id

            arxiv_id = normalize_arxiv_id(route.query_text)
            if not arxiv_id:
                return AdapterSearchResult(
                    hits=[],
                    api_calls=0,
                    warnings=["Query is not an arXiv identifier"],
                )
            response = await self._get(
                self.BASE_URL,
                params={"id_list": arxiv_id, "max_results": 1},
                headers={"Accept": "application/atom+xml"},
            )
            raws = self._parse_feed(response.body)
            accepted_raws = []
            post_cutoff = 0
            unknown_date = 0
            for raw in raws:
                allowed, reason = _raw_is_historical(
                    raw,
                    cutoff=self.cutoff_date,
                    date_keys=("published",),
                    year_key=None,
                )
                if allowed:
                    accepted_raws.append(raw)
                elif reason == "unknown_date":
                    unknown_date += 1
                else:
                    post_cutoff += 1
            records = [normalize_arxiv_entry(raw) for raw in accepted_raws]
            accepted = [
                self._to_hit(route, record, rank)
                for rank, record in enumerate(records, start=1)
            ]
            self.diagnostics[route.route_id] = HistoricalRouteDiagnostics(
                route_id=route.route_id,
                source=self.source.value,
                cutoff_date=self.cutoff_date.isoformat(),
                server_filter="identifier lookup + published date post-filter",
                pages_fetched=1,
                raw_records=len(raws),
                accepted_records=len(accepted),
                post_cutoff_filtered=post_cutoff,
                unknown_date_filtered=unknown_date,
                exhausted=True,
            )
            return AdapterSearchResult(hits=accepted, api_calls=1)

        base_query = self._build_query(route)
        search_query = f"({base_query}) AND {self.submitted_date_filter}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": route.limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = await self._get(
            self.BASE_URL,
            params=params,
            headers={"Accept": "application/atom+xml"},
        )
        raws = self._parse_feed(response.body)
        accepted_raws: list[dict[str, Any]] = []
        post_cutoff = 0
        unknown_date = 0
        for raw in raws:
            allowed, reason = _raw_is_historical(
                raw,
                cutoff=self.cutoff_date,
                date_keys=("published",),
                year_key=None,
            )
            if allowed:
                accepted_raws.append(raw)
            elif reason == "unknown_date":
                unknown_date += 1
            else:
                post_cutoff += 1
        records = [normalize_arxiv_entry(raw) for raw in accepted_raws]
        hits = [
            self._to_hit(route, record, rank)
            for rank, record in enumerate(records, start=1)
        ]
        self.diagnostics[route.route_id] = HistoricalRouteDiagnostics(
            route_id=route.route_id,
            source=self.source.value,
            cutoff_date=self.cutoff_date.isoformat(),
            server_filter=self.submitted_date_filter,
            pages_fetched=1,
            raw_records=len(raws),
            accepted_records=len(hits),
            post_cutoff_filtered=post_cutoff,
            unknown_date_filtered=unknown_date,
            exhausted=len(raws) < route.limit,
        )
        return AdapterSearchResult(hits=hits, api_calls=1)


def build_historical_adapters(
    config: HistoricalEvalConfig,
) -> dict[SearchSource, Any]:
    adapters: dict[SearchSource, Any] = {}
    if config.use_openalex:
        api_key = os.getenv("OPENALEX_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "PASA_HISTORICAL_USE_OPENALEX=true but OPENALEX_API_KEY is missing"
            )
        adapters[SearchSource.OPENALEX] = HistoricalOpenAlexAdapter(
            cutoff_date=config.cutoff_date,
            api_key=api_key,
            timeout_seconds=config.timeout_seconds,
        )
    if config.use_semantic_scholar:
        adapters[SearchSource.SEMANTIC_SCHOLAR] = HistoricalSemanticScholarAdapter(
            cutoff_date=config.cutoff_date,
            page_size=config.semantic_scholar_page_size,
            max_pages=config.semantic_scholar_max_pages,
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None,
            timeout_seconds=config.timeout_seconds,
        )
    if config.use_arxiv:
        adapters[SearchSource.ARXIV] = HistoricalArxivAdapter(
            cutoff_date=config.cutoff_date,
            timeout_seconds=config.timeout_seconds,
        )
    return adapters


def collect_route_diagnostics(
    adapters: Mapping[SearchSource, Any],
    route_id: str,
) -> dict[str, Any] | None:
    for adapter in adapters.values():
        item = adapter.route_diagnostics(route_id)
        if item is not None:
            return item
    return None


__all__ = [
    "HistoricalArxivAdapter",
    "HistoricalOpenAlexAdapter",
    "HistoricalRouteDiagnostics",
    "HistoricalSemanticScholarAdapter",
    "build_historical_adapters",
    "collect_route_diagnostics",
]
