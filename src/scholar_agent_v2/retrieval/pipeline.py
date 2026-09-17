from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence

from ..candidate_pool.pool import CandidatePool
from ..candidate_pool.rerank import RerankConfig
from ..query_generation.models import QueryGenerationOutput, QueryGenerationResult
from ..query_understanding.schema import QueryAnalysisResult
from ..query_understanding.schema import RetrievalMode
from .factory import build_adapters
from .config import env_bool, get_enabled_sources
from .citation_expansion import (
    CitationExpansionConfig,
    CitationExpansionOutput,
    CitationSnowballExpander,
)
from .models import RetrievalOutput, SearchSource
from .orchestrator import RetrievalOrchestrator
from .route_builder import RetrievalModeBudget, RetrievalRouteBuilder
from .sources.base import AcademicSourceAdapter


@dataclass
class RetrievalPipelineOutput:
    retrieval: RetrievalOutput
    candidate_pool: CandidatePool
    citation_expansion: CitationExpansionOutput


class ScholarRetrievalPipeline:
    """Shared retrieval-to-prerank pipeline for local and online profiles."""

    def __init__(
        self,
        *,
        enabled_sources: Sequence[SearchSource],
        adapters: Mapping[SearchSource, AcademicSourceAdapter] | None = None,
        max_concurrency: int = 6,
        route_timeout_seconds: float = 20.0,
        max_attempts: int = 2,
        mode_budgets: Mapping[RetrievalMode, RetrievalModeBudget] | None = None,
        citation_config: CitationExpansionConfig | None = None,
    ) -> None:
        self.enabled_sources = list(enabled_sources)
        self.adapters = dict(adapters or build_adapters(self.enabled_sources))
        missing = set(self.enabled_sources) - set(self.adapters)
        if missing:
            names = ", ".join(sorted(source.value for source in missing))
            raise ValueError(f"Missing adapters for enabled sources: {names}")
        self.route_builder = RetrievalRouteBuilder(
            enabled_sources=self.enabled_sources,
            mode_budgets=dict(mode_budgets) if mode_budgets is not None else None,
        )
        self.orchestrator = RetrievalOrchestrator(
            self.adapters,
            max_concurrency=max_concurrency,
            route_timeout_seconds=route_timeout_seconds,
            max_attempts=max_attempts,
        )
        self.citation_config = citation_config or CitationExpansionConfig()
        self.citation_expander = CitationSnowballExpander(
            self.adapters,
            config=self.citation_config,
        )

    @classmethod
    def from_environment(
        cls,
        *,
        max_concurrency: int = 6,
        route_timeout_seconds: float = 20.0,
        max_attempts: int = 2,
        mode_budgets: Mapping[RetrievalMode, RetrievalModeBudget] | None = None,
        citation_config: CitationExpansionConfig | None = None,
    ) -> "ScholarRetrievalPipeline":
        """Build the same pipeline from the active retrieval profile."""
        sources = get_enabled_sources()
        adapters = build_adapters(
            sources,
            timeout_seconds=route_timeout_seconds,
        )
        configured_citation = citation_config or CitationExpansionConfig(
            enabled=env_bool("USE_CITATION_SNOWBALL", False),
            channel_weight=float(os.getenv("CITATION_CHANNEL_WEIGHT", "0.70")),
            seed_top_k=int(os.getenv("CITATION_SEED_TOP_K", "3")),
            forward_top_k=int(os.getenv("CITATION_FORWARD_TOP_K", "30")),
            backward_top_k=int(os.getenv("CITATION_BACKWARD_TOP_K", "30")),
            fetch_limit_per_seed=int(
                os.getenv("CITATION_FETCH_LIMIT_PER_SEED", "50")
            ),
            min_seed_score=float(os.getenv("CITATION_MIN_SEED_SCORE", "0.20")),
            min_new_candidate_rate=float(
                os.getenv("CITATION_MIN_NEW_CANDIDATE_RATE", "0.10")
            ),
        )
        return cls(
            enabled_sources=sources,
            adapters=adapters,
            max_concurrency=max_concurrency,
            route_timeout_seconds=route_timeout_seconds,
            max_attempts=max_attempts,
            mode_budgets=mode_budgets,
            citation_config=configured_citation,
        )

    async def run(
        self,
        analysis: QueryAnalysisResult,
        generation: QueryGenerationResult | QueryGenerationOutput,
        *,
        apply_hard_filters: bool = True,
    ) -> RetrievalPipelineOutput:
        plan = self.route_builder.build(analysis, generation)
        retrieval = await self.orchestrator.run(plan)
        pool = CandidatePool.from_hits(retrieval.hits)
        if apply_hard_filters:
            pool.apply_hard_filters(analysis)
        pool.prerank(
            analysis,
            generation,
            num_enabled_sources=len(self.enabled_sources),
            num_planned_routes=len(plan.routes),
        )
        citation = await self.citation_expander.run(pool, analysis)
        if citation.hits:
            pool = CandidatePool.from_hits([*retrieval.hits, *citation.hits])
            if apply_hard_filters:
                pool.apply_hard_filters(analysis)
            pool.prerank(
                analysis,
                generation,
                num_enabled_sources=len(self.enabled_sources),
                num_planned_routes=len(plan.routes),
                config=RerankConfig(
                    citation_channel_weight=self.citation_config.channel_weight
                ),
            )
        pool.warnings.extend(citation.diagnostics.warnings)
        return RetrievalPipelineOutput(
            retrieval=retrieval,
            candidate_pool=pool,
            citation_expansion=citation,
        )


__all__ = ["RetrievalPipelineOutput", "ScholarRetrievalPipeline"]
