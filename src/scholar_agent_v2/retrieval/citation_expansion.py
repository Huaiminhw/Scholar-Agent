from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from ..candidate_pool.models import CandidatePaper
from ..candidate_pool.pool import CandidatePool
from ..query_generation.models import QueryKind
from ..query_understanding.schema import QueryAnalysisResult, RetrievalMode
from .models import (
    CitationEdge,
    CitationLookupKey,
    RetrievalHit,
    RetrievalOrigin,
    RetrievalStrategy,
    SearchSource,
)
from .sources.base import AcademicSourceAdapter


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CitationExpansionConfig:
    enabled: bool = False
    # Multiplier applied to citation-only candidates during the shared
    # candidate-pool prerank.  It controls ranking, not graph traversal.
    channel_weight: float = 0.70
    seed_top_k: int = 3
    min_seed_score: float = 0.20
    forward_top_k: int = 30
    backward_top_k: int = 30
    fetch_limit_per_seed: int = 50
    min_new_candidate_rate: float = 0.10
    max_concurrency: int = 4
    recent_forward_only: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.channel_weight <= 1.0:
            raise ValueError("channel_weight must be between 0 and 1")
        if self.seed_top_k < 0:
            raise ValueError("seed_top_k must be non-negative")
        if not 0.0 <= self.min_seed_score <= 1.0:
            raise ValueError("min_seed_score must be between 0 and 1")
        if min(self.forward_top_k, self.backward_top_k, self.fetch_limit_per_seed) < 0:
            raise ValueError("citation limits must be non-negative")
        if not 0.0 <= self.min_new_candidate_rate <= 1.0:
            raise ValueError("min_new_candidate_rate must be between 0 and 1")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CitationDirectionDiagnostics(_StrictModel):
    direction: Literal["forward", "backward"]
    requested_top_k: int = 0
    fetched_edges: int = 0
    unique_candidates: int = 0
    selected_candidates: int = 0


class CitationExpansionDiagnostics(_StrictModel):
    enabled: bool = False
    seed_candidate_ids: list[str] = Field(default_factory=list)
    seed_scores: dict[str, float] = Field(default_factory=dict)
    directions: list[CitationDirectionDiagnostics] = Field(default_factory=list)
    api_calls: int = 0
    raw_edges: int = 0
    existing_candidate_edges: int = 0
    fetched_new_unique_candidates: int = 0
    new_unique_candidates: int = 0
    added_hits: int = 0
    new_candidate_rate: float = 0.0
    duplicate_rate: float = 0.0
    latency_ms: float = 0.0
    stop_reason: str = "disabled"
    warnings: list[str] = Field(default_factory=list)


class CitationExpansionOutput(_StrictModel):
    hits: list[RetrievalHit] = Field(default_factory=list)
    diagnostics: CitationExpansionDiagnostics


class CitationSnowballExpander:
    """One-hop, relevance-seeded citation expansion with global Top-K."""

    def __init__(
        self,
        adapters: Mapping[SearchSource, AcademicSourceAdapter],
        *,
        config: CitationExpansionConfig | None = None,
    ) -> None:
        self.adapters = dict(adapters)
        self.config = config or CitationExpansionConfig()

    async def run(
        self,
        pool: CandidatePool,
        analysis: QueryAnalysisResult,
    ) -> CitationExpansionOutput:
        started = time.perf_counter()
        if not self.config.enabled:
            return self._empty("disabled", started)

        adapter_order = self._citation_adapters()
        if not adapter_order:
            return self._empty("no_citation_adapter", started)

        seeds: list[tuple[CandidatePaper, AcademicSourceAdapter, CitationLookupKey]] = []
        for candidate in pool.top(len(pool.candidates)):
            score = float(candidate.retrieval_score or 0.0)
            if score < self.config.min_seed_score:
                continue
            key = self._lookup_key(candidate)
            adapter = next(
                (item for item in adapter_order if self._adapter_can_resolve(item, key)),
                None,
            )
            if adapter is None:
                continue
            seeds.append((candidate, adapter, key))
            if len(seeds) >= self.config.seed_top_k:
                break
        if not seeds:
            return self._empty("no_relevant_seed", started)

        directions: list[Literal["forward", "backward"]] = []
        if self.config.forward_top_k > 0:
            directions.append("forward")
        forward_only = (
            self.config.recent_forward_only
            and (analysis.properties.recent_first or analysis.intent.mode == RetrievalMode.LATEST)
            and not analysis.properties.central_first
            and not analysis.properties.central_last
        )
        if self.config.backward_top_k > 0 and not forward_only:
            directions.append("backward")
        if not directions:
            return self._empty("zero_direction_budget", started)

        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def fetch(candidate, adapter, key, direction):
            async with semaphore:
                try:
                    result = await adapter.citation_neighbors(
                        key,
                        direction=direction,
                        limit=self.config.fetch_limit_per_seed,
                    )
                    return direction, result, None
                except Exception as exc:  # one seed/direction must not fail the run
                    return direction, None, exc

        fetched = await asyncio.gather(
            *(
                fetch(candidate, adapter, key, direction)
                for candidate, adapter, key in seeds
                for direction in directions
            )
        )
        seed_scores = {
            candidate.candidate_id: float(candidate.retrieval_score or 0.0)
            for candidate, _, _ in seeds
        }
        edges_by_direction: dict[str, list[CitationEdge]] = {
            direction: [] for direction in directions
        }
        api_calls = 0
        warnings: list[str] = []
        for direction, result, error in fetched:
            if error is not None:
                warnings.append(f"{direction} citation expansion failed: {error}")
                continue
            assert result is not None
            api_calls += result.api_calls
            warnings.extend(result.warnings)
            edges_by_direction[direction].extend(result.edges)

        existing_ids = {candidate.candidate_id for candidate in pool.candidates}
        selected_hits: list[RetrievalHit] = []
        direction_diagnostics: list[CitationDirectionDiagnostics] = []
        raw_edges = 0
        existing_edges = 0
        fetched_new_ids: set[str] = set()
        selected_new_ids: set[str] = set()

        for direction in directions:
            edges = edges_by_direction[direction]
            raw_edges += len(edges)
            edge_hits = [
                self._edge_hit(
                    edge,
                    edge_rank=index,
                    query_text=analysis.content or analysis.original_query,
                    seed_score=seed_scores.get(edge.seed_candidate_id, 0.0),
                )
                for index, edge in enumerate(edges, start=1)
            ]
            edge_pool = CandidatePool.from_hits(edge_hits)
            scored = []
            for candidate in edge_pool.candidates:
                score = sum(
                    float(origin.source_score or 0.0) for origin in candidate.origins
                )
                # Match the reference implementation's light anti-popularity
                # bias while keeping multi-seed relevance as the dominant term.
                score -= 0.0005 * float(candidate.citation_count or 0)
                scored.append((score, candidate))
            scored.sort(key=lambda item: (item[0], item[1].candidate_id), reverse=True)
            new_scored = []
            for score, candidate in scored:
                if candidate.candidate_id in existing_ids:
                    existing_edges += max(1, candidate.hit_count)
                    continue
                new_scored.append((score, candidate))
                fetched_new_ids.add(candidate.candidate_id)
            top_k = (
                self.config.forward_top_k
                if direction == "forward"
                else self.config.backward_top_k
            )
            selected = new_scored[:top_k]
            for rank, (score, candidate) in enumerate(selected, start=1):
                selected_hits.extend(
                    self._candidate_hits(candidate, rank=rank, score=score)
                )
                selected_new_ids.add(candidate.candidate_id)
            direction_diagnostics.append(
                CitationDirectionDiagnostics(
                    direction=direction,
                    requested_top_k=top_k,
                    fetched_edges=len(edges),
                    unique_candidates=len(scored),
                    selected_candidates=len(selected),
                )
            )

        new_rate = len(fetched_new_ids) / raw_edges if raw_edges else 0.0
        duplicate_rate = max(0.0, 1.0 - new_rate) if raw_edges else 0.0
        if raw_edges == 0:
            stop_reason = "no_citation_neighbors"
        elif not selected_new_ids:
            stop_reason = "no_new_candidates"
        elif new_rate < self.config.min_new_candidate_rate:
            stop_reason = "low_candidate_novelty"
        else:
            stop_reason = "citation_budget_reached"
        diagnostics = CitationExpansionDiagnostics(
            enabled=True,
            seed_candidate_ids=[candidate.candidate_id for candidate, _, _ in seeds],
            seed_scores=seed_scores,
            directions=direction_diagnostics,
            api_calls=api_calls,
            raw_edges=raw_edges,
            existing_candidate_edges=existing_edges,
            fetched_new_unique_candidates=len(fetched_new_ids),
            new_unique_candidates=len(selected_new_ids),
            added_hits=len(selected_hits),
            new_candidate_rate=new_rate,
            duplicate_rate=duplicate_rate,
            latency_ms=(time.perf_counter() - started) * 1000,
            stop_reason=stop_reason,
            warnings=warnings,
        )
        logger.info(
            "citation_snowball seeds=%s raw_edges=%s new_candidates=%s "
            "new_rate=%.4f duplicate_rate=%.4f api_calls=%s latency_ms=%.2f "
            "stop_reason=%s",
            len(seeds),
            raw_edges,
            len(selected_new_ids),
            new_rate,
            duplicate_rate,
            api_calls,
            diagnostics.latency_ms,
            stop_reason,
        )
        return CitationExpansionOutput(hits=selected_hits, diagnostics=diagnostics)

    def _citation_adapters(self) -> list[AcademicSourceAdapter]:
        priority = (
            SearchSource.PASA_LOCAL,
            SearchSource.SEMANTIC_SCHOLAR,
            SearchSource.OPENALEX,
        )
        return [
            self.adapters[source]
            for source in priority
            if source in self.adapters
            and self.adapters[source].capabilities.citation_search
        ]

    @staticmethod
    def _lookup_key(candidate: CandidatePaper) -> CitationLookupKey:
        return CitationLookupKey(
            candidate_id=candidate.candidate_id,
            title=candidate.title,
            doi=candidate.identifiers.doi,
            arxiv_id=candidate.identifiers.arxiv_id,
            semantic_scholar_id=candidate.identifiers.semantic_scholar_id,
            openalex_id=candidate.identifiers.openalex_id,
        )

    @staticmethod
    def _adapter_can_resolve(
        adapter: AcademicSourceAdapter,
        key: CitationLookupKey,
    ) -> bool:
        if adapter.source == SearchSource.PASA_LOCAL:
            return bool(key.arxiv_id)
        if adapter.source == SearchSource.SEMANTIC_SCHOLAR:
            return bool(key.semantic_scholar_id or key.doi or key.arxiv_id)
        if adapter.source == SearchSource.OPENALEX:
            return bool(key.openalex_id or key.doi)
        return False

    @staticmethod
    def _edge_hit(
        edge: CitationEdge,
        *,
        edge_rank: int,
        query_text: str,
        seed_score: float,
    ) -> RetrievalHit:
        contribution = seed_score + (0.1 if edge.is_influential else 0.0)
        return RetrievalHit(
            **edge.target.model_dump(),
            origin=RetrievalOrigin(
                source=edge.source,
                route_id=(
                    f"CIT-{edge.direction[0].upper()}-{edge.seed_candidate_id}"
                ),
                query_id="CITATION",
                query_text=query_text,
                query_kind=QueryKind.EXPANDED,
                strategy=RetrievalStrategy.CITATION,
                rank=edge_rank,
                source_score=contribution,
                citation_direction=edge.direction,
                citation_seed_id=edge.seed_candidate_id,
                citation_hop=1,
                search_iteration=1,
            ),
        )

    @staticmethod
    def _candidate_hits(
        candidate: CandidatePaper,
        *,
        rank: int,
        score: float,
    ) -> list[RetrievalHit]:
        source_id = (
            candidate.identifiers.arxiv_id
            or candidate.identifiers.semantic_scholar_id
            or candidate.identifiers.openalex_id
            or candidate.identifiers.doi
            or candidate.candidate_id
        )
        output: list[RetrievalHit] = []
        for origin in candidate.origins:
            output.append(
                RetrievalHit(
                    source=origin.source,
                    source_id=source_id,
                    title=candidate.title,
                    abstract=candidate.abstract,
                    authors=list(candidate.authors),
                    year=candidate.year,
                    venue=candidate.venue,
                    doi=candidate.identifiers.doi,
                    arxiv_id=candidate.identifiers.arxiv_id,
                    semantic_scholar_id=candidate.identifiers.semantic_scholar_id,
                    openalex_id=candidate.identifiers.openalex_id,
                    pmid=candidate.identifiers.pmid,
                    citation_count=candidate.citation_count,
                    publication_types=list(candidate.publication_types),
                    fields_of_study=list(candidate.fields_of_study),
                    language=candidate.language,
                    urls=list(candidate.urls),
                    origin=origin.model_copy(
                        update={"rank": rank, "source_score": score}
                    ),
                )
            )
        return output

    def _empty(self, reason: str, started: float) -> CitationExpansionOutput:
        return CitationExpansionOutput(
            diagnostics=CitationExpansionDiagnostics(
                enabled=self.config.enabled,
                latency_ms=(time.perf_counter() - started) * 1000,
                stop_reason=reason,
            )
        )


__all__ = [
    "CitationExpansionConfig",
    "CitationDirectionDiagnostics",
    "CitationExpansionDiagnostics",
    "CitationExpansionOutput",
    "CitationSnowballExpander",
]
