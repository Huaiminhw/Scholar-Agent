from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ..query_generation.models import QueryGenerationOutput, QueryGenerationResult
from ..query_understanding.schema import QueryAnalysisResult
from ..retrieval.models import RetrievalHit, SearchSource
from ..retrieval.sources.base import AcademicSourceAdapter

from .dedup import cluster_hits
from .enrichment import CandidateEnricher
from .filter import apply_filters
from .fusion import fuse_hit_cluster
from .models import CandidatePaper, CandidatePoolResult, CandidatePoolStats, MatchStatus
from .rerank import RerankConfig, rerank_candidates


@dataclass
class CandidatePool:
    candidates: list[CandidatePaper] = field(default_factory=list)
    rejected: list[CandidatePaper] = field(default_factory=list)
    stats: CandidatePoolStats = field(default_factory=CandidatePoolStats)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_hits(cls, hits: Iterable[RetrievalHit]) -> "CandidatePool":
        hit_list = list(hits)
        clusters = cluster_hits(hit_list)
        candidates = [fuse_hit_cluster(cluster) for cluster in clusters]
        stats = CandidatePoolStats(
            raw_hits=len(hit_list),
            dedup_clusters=len(clusters),
            candidates_before_filter=len(candidates),
            candidates_after_filter=len(candidates),
        )
        return cls(candidates=candidates, stats=stats)

    def apply_hard_filters(
        self,
        analysis: QueryAnalysisResult,
        *,
        strict_field_of_study: bool = False,
    ) -> "CandidatePool":
        kept, rejected = apply_filters(
            self.candidates,
            analysis,
            strict_field_of_study=strict_field_of_study,
        )
        self.candidates = kept
        self.rejected.extend(rejected)
        self.stats.rejected_by_hard_filter = len(self.rejected)
        self.stats.unknown_after_filter = sum(
            c.hard_filter_status == MatchStatus.UNKNOWN for c in self.candidates
        )
        self.stats.candidates_after_filter = len(self.candidates)
        return self

    def prerank(
        self,
        analysis: QueryAnalysisResult,
        generation: QueryGenerationResult | QueryGenerationOutput,
        *,
        num_enabled_sources: int = 3,
        num_planned_routes: int | None = None,
        config: RerankConfig | None = None,
    ) -> "CandidatePool":
        self.candidates = rerank_candidates(
            self.candidates,
            analysis,
            generation,
            num_enabled_sources=num_enabled_sources,
            num_planned_routes=num_planned_routes,
            config=config,
        )
        return self

    async def enrich(
        self,
        adapters: Mapping[SearchSource, AcademicSourceAdapter],
        *,
        top_n: int = 80,
        required_fields: tuple[str, ...] = (
            "abstract",
            "venue",
            "citation_count",
            "publication_types",
            "fields_of_study",
        ),
    ) -> int:
        enricher = CandidateEnricher(adapters)
        before = [set(c.enriched_fields) for c in self.candidates]
        self.candidates, api_calls, warnings = await enricher.enrich(
            self.candidates,
            top_n=top_n,
            required_fields=required_fields,
        )
        self.warnings.extend(warnings)
        self.stats.enriched_candidates = sum(
            bool(c.enriched_fields - old)
            for c, old in zip(self.candidates, before)
        )
        return api_calls

    def top(self, n: int) -> list[CandidatePaper]:
        if n <= 0:
            return []
        if any(c.retrieval_score is not None for c in self.candidates):
            return sorted(
                self.candidates,
                key=lambda c: c.retrieval_score if c.retrieval_score is not None else -1.0,
                reverse=True,
            )[:n]
        return sorted(self.candidates, key=lambda c: c.best_rank or 10**9)[:n]

    def to_result(self) -> CandidatePoolResult:
        return CandidatePoolResult(
            candidates=self.candidates,
            rejected=self.rejected,
            stats=self.stats,
            warnings=self.warnings,
        )


def build_candidate_pool(hits: Iterable[RetrievalHit]) -> CandidatePool:
    return CandidatePool.from_hits(hits)


__all__ = ["CandidatePool", "build_candidate_pool"]
