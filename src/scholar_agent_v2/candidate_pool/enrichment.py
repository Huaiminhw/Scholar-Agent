from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Mapping

from ..retrieval.models import SearchSource, SourcePaperRecord
from ..retrieval.sources.base import AcademicSourceAdapter

from .fusion import enrich_candidate
from .models import CandidatePaper


class CandidateEnricher:
    """轻量 Lazy Enrichment：只对 Top-N 候选补缺失元数据。"""

    def __init__(
        self,
        adapters: Mapping[SearchSource, AcademicSourceAdapter],
        *,
        preferred_sources: tuple[SearchSource, ...] = (
            SearchSource.SEMANTIC_SCHOLAR,
            SearchSource.OPENALEX,
            SearchSource.ARXIV,
        ),
    ) -> None:
        self.adapters = dict(adapters)
        self.preferred_sources = preferred_sources

    async def enrich(
        self,
        candidates: list[CandidatePaper],
        *,
        top_n: int = 80,
        required_fields: tuple[str, ...] = (
            "abstract",
            "venue",
            "citation_count",
            "publication_types",
            "fields_of_study",
        ),
    ) -> tuple[list[CandidatePaper], int, list[str]]:
        if not candidates:
            return candidates, 0, []

        target_indices = [
            i
            for i, c in enumerate(candidates[:top_n])
            if self._needs_enrichment(c, required_fields)
        ]
        if not target_indices:
            return candidates, 0, []

        groups: dict[SearchSource, list[tuple[int, str]]] = defaultdict(list)
        for idx in target_indices:
            candidate = candidates[idx]
            chosen = self._choose_identifier(candidate)
            if chosen:
                source, identifier = chosen
                groups[source].append((idx, identifier))

        tasks = []
        task_meta = []
        for source, items in groups.items():
            adapter = self.adapters.get(source)
            if not adapter:
                continue
            ids = [identifier for _, identifier in items]
            tasks.append(adapter.fetch_details(ids))
            task_meta.append((source, items))

        if not tasks:
            return candidates, 0, ["No enrichment-capable source identifier found for target candidates."]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        output = list(candidates)
        total_api_calls = 0
        warnings: list[str] = []

        for (source, items), result in zip(task_meta, results):
            if isinstance(result, Exception):
                warnings.append(f"{source.value} enrichment failed: {result}")
                continue
            total_api_calls += result.api_calls
            warnings.extend(result.warnings)
            index_by_identifier = {identifier.casefold(): idx for idx, identifier in items}

            for record in result.records:
                idx = self._match_record(record, index_by_identifier, output)
                if idx is not None:
                    output[idx] = enrich_candidate(output[idx], record)

        return output, total_api_calls, warnings

    def _choose_identifier(self, candidate: CandidatePaper) -> tuple[SearchSource, str] | None:
        ids = candidate.identifiers
        for source in self.preferred_sources:
            if source == SearchSource.SEMANTIC_SCHOLAR:
                if ids.semantic_scholar_id:
                    return source, ids.semantic_scholar_id
                if ids.doi:
                    return source, f"DOI:{ids.doi}"
                if ids.arxiv_id:
                    return source, f"ARXIV:{ids.arxiv_id}"
            elif source == SearchSource.OPENALEX:
                if ids.openalex_id:
                    return source, ids.openalex_id
                if ids.doi:
                    return source, ids.doi
            elif source == SearchSource.ARXIV and ids.arxiv_id:
                return source, ids.arxiv_id
        return None

    @staticmethod
    def _needs_enrichment(candidate: CandidatePaper, fields: tuple[str, ...]) -> bool:
        for field in fields:
            value = getattr(candidate, field, None)
            if value is None or value == "" or value == []:
                return True
        return False

    @staticmethod
    def _match_record(
        record: SourcePaperRecord,
        index_by_identifier: dict[str, int],
        candidates: list[CandidatePaper],
    ) -> int | None:
        keys = [
            record.source_id,
            record.semantic_scholar_id,
            record.openalex_id,
            record.doi,
            record.arxiv_id,
            f"DOI:{record.doi}" if record.doi else None,
            f"ARXIV:{record.arxiv_id}" if record.arxiv_id else None,
        ]
        for key in keys:
            if key and key.casefold() in index_by_identifier:
                return index_by_identifier[key.casefold()]

        # fallback: 通过强标识符在候选集里匹配。
        for idx, candidate in enumerate(candidates):
            ids = candidate.identifiers
            if record.doi and ids.doi and record.doi.casefold() == ids.doi.casefold():
                return idx
            if record.arxiv_id and ids.arxiv_id and record.arxiv_id.casefold() == ids.arxiv_id.casefold():
                return idx
            if record.semantic_scholar_id and ids.semantic_scholar_id == record.semantic_scholar_id:
                return idx
            if record.openalex_id and ids.openalex_id == record.openalex_id:
                return idx
        return None


__all__ = ["CandidateEnricher"]
