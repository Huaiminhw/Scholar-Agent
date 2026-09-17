from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .config import RetrievalProfile, get_enabled_sources, get_retrieval_profile
from .models import SearchSource
from .sources import (
    AcademicSourceAdapter,
    ArxivAdapter,
    GoogleSearchAdapter,
    OpenAlexAdapter,
    PaSaLocalAdapter,
    SemanticScholarAdapter,
)
from .sources.base import SourceConfigurationError


def build_adapters(
    enabled_sources: Iterable[SearchSource] | None = None,
    *,
    profile: RetrievalProfile | None = None,
    timeout_seconds: float = 20.0,
    pasa_index_path: str | Path | None = None,
    pasa_paper_zip_path: str | Path | None = None,
    pasa_id_map_path: str | Path | None = None,
    pasa_citation_graph_path: str | Path | None = None,
) -> dict[SearchSource, AcademicSourceAdapter]:
    """Build source adapters and reject accidental local/online profile mixing."""

    selected_profile = profile or get_retrieval_profile()
    sources = list(enabled_sources) if enabled_sources is not None else get_enabled_sources()
    source_set = set(sources)

    if selected_profile == RetrievalProfile.FROZEN_LOCAL:
        if source_set != {SearchSource.PASA_LOCAL}:
            raise SourceConfigurationError(
                "frozen_local profile must use PASA_LOCAL exclusively"
            )
    elif SearchSource.PASA_LOCAL in source_set:
        raise SourceConfigurationError(
            "competition_online profile cannot include PASA_LOCAL"
        )

    adapters: dict[SearchSource, AcademicSourceAdapter] = {}
    if SearchSource.PASA_LOCAL in source_set:
        adapters[SearchSource.PASA_LOCAL] = PaSaLocalAdapter(
            index_path=pasa_index_path,
            paper_zip_path=pasa_paper_zip_path,
            id_map_path=pasa_id_map_path,
            citation_graph_path=pasa_citation_graph_path,
            timeout_seconds=max(timeout_seconds, 60.0),
        )
    if SearchSource.OPENALEX in source_set:
        key = os.getenv("OPENALEX_API_KEY", "").strip()
        if not key:
            raise SourceConfigurationError(
                "USE_OPENALEX=true but OPENALEX_API_KEY is missing"
            )
        adapters[SearchSource.OPENALEX] = OpenAlexAdapter(
            api_key=key,
            timeout_seconds=timeout_seconds,
        )
    if SearchSource.SEMANTIC_SCHOLAR in source_set:
        adapters[SearchSource.SEMANTIC_SCHOLAR] = SemanticScholarAdapter(
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None,
            timeout_seconds=timeout_seconds,
        )
    if SearchSource.ARXIV in source_set:
        adapters[SearchSource.ARXIV] = ArxivAdapter(
            timeout_seconds=timeout_seconds,
        )
    if SearchSource.GOOGLE_SEARCH in source_set:
        key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
        engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "").strip()
        if not key or not engine_id:
            raise SourceConfigurationError(
                "USE_GOOGLE_SEARCH=true but GOOGLE_SEARCH_API_KEY or "
                "GOOGLE_SEARCH_ENGINE_ID is missing"
            )
        adapters[SearchSource.GOOGLE_SEARCH] = GoogleSearchAdapter(
            api_key=key,
            engine_id=engine_id,
            timeout_seconds=timeout_seconds,
        )
    return adapters


__all__ = ["build_adapters"]
