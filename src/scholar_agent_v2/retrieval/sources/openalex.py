from __future__ import annotations

import asyncio
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
from ..normalizer import normalize_doi, normalize_openalex_work

from .base import AcademicSourceAdapter, SourceConfigurationError


class OpenAlexAdapter(AcademicSourceAdapter):
    source = SearchSource.OPENALEX
    capabilities = SourceCapabilities(
        relevance_search=True,
        title_search=True,
        identifier_lookup=True,
        author_search=True,
        filter_only=True,
        citation_search=True,
        supported_filters={"year", "author", "publication_type", "min_citations"},
    )

    BASE_URL = "https://api.openalex.org"

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
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY")

    def _auth_params(self) -> dict[str, str]:
        if not self.api_key:
            raise SourceConfigurationError(
                "OPENALEX_API_KEY is missing. Current OpenAlex API access requires an API key."
            )
        return {"api_key": self.api_key}

    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        if route.strategy == RetrievalStrategy.IDENTIFIER:
            return await self._identifier_lookup(route)

        params: dict[str, Any] = {
            **self._auth_params(),
            "per-page": route.limit,
        }
        filters = self._compile_filters(route)
        if filters:
            params["filter"] = ",".join(filters)

        if route.strategy == RetrievalStrategy.RELEVANCE:
            params["search"] = route.query_text
        elif route.strategy == RetrievalStrategy.TITLE:
            params["search.exact"] = route.query_text
        elif route.strategy == RetrievalStrategy.AUTHOR:
            author = route.filters.authors[0] if route.filters.authors else route.query_text
            author_expr = f'raw_author_name.search:"{self._escape_filter_phrase(author)}"'
            params["filter"] = ",".join([x for x in [params.get("filter"), author_expr] if x])
            # 如果 BY_AUTHOR 还包含主题内容，则同时保留内容 search；否则只按作者过滤。
            if route.query_text and route.query_text.casefold() != author.casefold():
                params["search"] = route.query_text
        elif route.strategy == RetrievalStrategy.FILTER_ONLY:
            pass
        else:
            raise ValueError(f"Unsupported OpenAlex strategy: {route.strategy}")

        response = await self._get(f"{self.BASE_URL}/works", params=params)
        payload = response.json()
        records = [normalize_openalex_work(x) for x in payload.get("results", [])]
        hits = [self._to_hit(route, record, rank, raw) for rank, (record, raw) in enumerate(zip(records, payload.get("results", [])), 1)]
        return AdapterSearchResult(hits=hits, api_calls=1)

    async def _identifier_lookup(self, route: RouteSpec) -> AdapterSearchResult:
        text = route.query_text.strip()
        params = self._auth_params()

        doi = normalize_doi(text)
        if doi:
            # OpenAlex 支持 DOI 作为 work singleton 外部 ID。
            target = f"https://doi.org/{doi}"
            try:
                response = await self._get(f"{self.BASE_URL}/works/{target}", params=params)
                raw = response.json()
                record = normalize_openalex_work(raw)
                return AdapterSearchResult(hits=[self._to_hit(route, record, 1, raw)], api_calls=1)
            except Exception:
                # 某些代理/编码环境下 singleton 外部 ID 可能失败，退回 DOI filter。
                response = await self._get(
                    f"{self.BASE_URL}/works",
                    params={**params, "filter": f"doi:https://doi.org/{doi}", "per-page": min(route.limit, 10)},
                )
                payload = response.json()
                raws = payload.get("results", [])
                records = [normalize_openalex_work(x) for x in raws]
                return AdapterSearchResult(
                    hits=[self._to_hit(route, r, i, raw) for i, (r, raw) in enumerate(zip(records, raws), 1)],
                    api_calls=2,
                    warnings=["OpenAlex DOI singleton failed; used DOI filter fallback."],
                )

        oa_match = re.search(r"\bW\d+\b", text, re.I)
        if oa_match:
            oa_id = oa_match.group(0).upper()
            response = await self._get(f"{self.BASE_URL}/works/{oa_id}", params=params)
            raw = response.json()
            record = normalize_openalex_work(raw)
            return AdapterSearchResult(hits=[self._to_hit(route, record, 1, raw)], api_calls=1)

        # 未识别为 OpenAlex/DOI 时，用精确文本搜索做保守 fallback。
        response = await self._get(
            f"{self.BASE_URL}/works",
            params={**params, "search.exact": text, "per-page": min(route.limit, 10)},
        )
        payload = response.json()
        raws = payload.get("results", [])
        records = [normalize_openalex_work(x) for x in raws]
        return AdapterSearchResult(
            hits=[self._to_hit(route, r, i, raw) for i, (r, raw) in enumerate(zip(records, raws), 1)],
            api_calls=1,
            warnings=["Identifier was not OpenAlex ID/DOI; used exact search fallback."],
        )

    async def fetch_details(self, ids: list[str]) -> AdapterFetchResult:
        unique = list(dict.fromkeys(x for x in ids if x))
        if not unique:
            return AdapterFetchResult(records=[], api_calls=0)

        params = self._auth_params()
        sem = asyncio.Semaphore(6)

        async def one(identifier: str):
            async with sem:
                target = identifier
                doi = normalize_doi(identifier)
                if doi:
                    target = f"https://doi.org/{doi}"
                response = await self._get(f"{self.BASE_URL}/works/{target}", params=params)
                return normalize_openalex_work(response.json())

        results = await asyncio.gather(*(one(x) for x in unique), return_exceptions=True)
        records = [x for x in results if not isinstance(x, Exception)]
        warnings = [f"OpenAlex enrichment failed for {identifier}: {result}" for identifier, result in zip(unique, results) if isinstance(result, Exception)]
        return AdapterFetchResult(records=records, api_calls=len(unique), warnings=warnings)

    async def citation_neighbors(
        self,
        seed: CitationLookupKey,
        *,
        direction: Literal["forward", "backward"],
        limit: int,
    ) -> CitationNeighborResult:
        if limit <= 0:
            return CitationNeighborResult()
        work_id, seed_raw, resolve_calls = await self._resolve_citation_seed(seed)
        if not work_id:
            return CitationNeighborResult(
                api_calls=resolve_calls,
                warnings=[f"OpenAlex could not resolve citation seed {seed.candidate_id}"],
            )

        params: dict[str, Any] = {
            **self._auth_params(),
            "per-page": min(limit, 100),
        }
        api_calls = resolve_calls
        if direction == "forward":
            params["filter"] = f"cites:{work_id}"
            response = await self._get(f"{self.BASE_URL}/works", params=params)
            api_calls += 1
            raws = response.json().get("results", []) or []
        else:
            referenced = list((seed_raw or {}).get("referenced_works") or [])[:limit]
            if not referenced:
                return CitationNeighborResult(api_calls=api_calls)
            short_ids = [value.rstrip("/").split("/")[-1] for value in referenced]
            params["filter"] = "openalex_id:" + "|".join(short_ids)
            response = await self._get(f"{self.BASE_URL}/works", params=params)
            api_calls += 1
            raws = response.json().get("results", []) or []

        edges = [
            CitationEdge(
                source=self.source,
                direction=direction,
                seed_candidate_id=seed.candidate_id,
                target=normalize_openalex_work(raw),
            )
            for raw in raws
            if isinstance(raw, dict) and raw.get("id")
        ]
        return CitationNeighborResult(edges=edges, api_calls=api_calls)

    async def _resolve_citation_seed(
        self,
        seed: CitationLookupKey,
    ) -> tuple[str | None, dict[str, Any] | None, int]:
        target: str | None = seed.openalex_id
        if target:
            target = target.rstrip("/").split("/")[-1]
        elif seed.doi:
            target = f"https://doi.org/{seed.doi}"
        if not target:
            return None, None, 0
        response = await self._get(
            f"{self.BASE_URL}/works/{target}",
            params=self._auth_params(),
        )
        raw = response.json()
        raw_id = raw.get("id")
        work_id = raw_id.rstrip("/").split("/")[-1] if raw_id else None
        return work_id, raw, 1

    def _compile_filters(self, route: RouteSpec) -> list[str]:
        f = route.filters
        out: list[str] = []
        if f.year_start is not None:
            out.append(f"from_publication_date:{f.year_start}-01-01")
        if f.year_end is not None:
            out.append(f"to_publication_date:{f.year_end}-12-31")
        if f.authors and route.strategy != RetrievalStrategy.AUTHOR:
            # 多作者 hard AND：重复 raw_author_name.search；RouteBuilder 只在明确支持时才写入。
            out.extend(f'raw_author_name.search:"{self._escape_filter_phrase(a)}"' for a in f.authors)
        if f.publication_types:
            values = "|".join(self._oa_type(x) for x in f.publication_types if self._oa_type(x))
            if values:
                out.append(f"type:{values}")
        if f.min_citations is not None:
            # OpenAlex 数值过滤使用 >；为了表达 >=N，写 >N-1。
            out.append(f"cited_by_count:>{max(0, f.min_citations - 1)}")
        return out

    @staticmethod
    def _oa_type(value: str) -> str:
        normalized = value.strip().casefold().replace("_", " ")
        mapping = {
            "journal article": "article",
            "article": "article",
            "conference paper": "article",
            "proceedings article": "article",
            "book": "book",
            "book chapter": "book-chapter",
            "preprint": "preprint",
            "review": "review",
            "survey": "review",
        }
        return mapping.get(normalized, normalized.replace(" ", "-"))

    @staticmethod
    def _escape_filter_phrase(value: str) -> str:
        return value.replace('"', "").strip()

    @staticmethod
    def _to_hit(route: RouteSpec, record, rank: int, raw: dict[str, Any]) -> RetrievalHit:
        score = raw.get("relevance_score")
        return RetrievalHit(
            **record.model_dump(),
            origin=RetrievalOrigin(
                source=SearchSource.OPENALEX,
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


__all__ = ["OpenAlexAdapter"]
