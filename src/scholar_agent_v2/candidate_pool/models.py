from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..retrieval.models import RetrievalOrigin


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MatchStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class PaperIdentifiers(StrictBaseModel):
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None
    pmid: str | None = None

    def non_empty(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v}


class FilterEvidence(StrictBaseModel):
    ref_id: str
    status: MatchStatus
    message: str
    field: str | None = None
    observed: Any | None = None


class CandidatePaper(StrictBaseModel):
    candidate_id: str

    title: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None

    publication_types: list[str] = Field(default_factory=list)
    fields_of_study: list[str] = Field(default_factory=list)
    language: str | None = None
    citation_count: int | None = Field(default=None, ge=0)

    identifiers: PaperIdentifiers = Field(default_factory=PaperIdentifiers)
    urls: list[str] = Field(default_factory=list)
    origins: list[RetrievalOrigin] = Field(default_factory=list)

    # 字段来自哪些数据源；便于追踪融合依据。
    field_sources: dict[str, list[str]] = Field(default_factory=dict)
    # 冲突不静默覆盖。
    metadata_conflicts: dict[str, list[Any]] = Field(default_factory=dict)

    source_count: int = Field(default=0, ge=0)
    route_count: int = Field(default=0, ge=0)
    hit_count: int = Field(default=0, ge=0)
    best_rank: int | None = Field(default=None, ge=1)

    hard_filter_status: MatchStatus | None = None
    hard_filter_evidence: list[FilterEvidence] = Field(default_factory=list)
    soft_match_score: float = Field(default=0.0, ge=0.0, le=1.0)

    retrieval_score: float | None = None
    score_components: dict[str, float] = Field(default_factory=dict)
    enriched_fields: set[str] = Field(default_factory=set)


class CandidatePoolStats(StrictBaseModel):
    raw_hits: int = 0
    dedup_clusters: int = 0
    candidates_before_filter: int = 0
    rejected_by_hard_filter: int = 0
    unknown_after_filter: int = 0
    candidates_after_filter: int = 0
    enriched_candidates: int = 0


class CandidatePoolResult(StrictBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    candidates: list[CandidatePaper] = Field(default_factory=list)
    rejected: list[CandidatePaper] = Field(default_factory=list)
    stats: CandidatePoolStats = Field(default_factory=CandidatePoolStats)
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "MatchStatus",
    "PaperIdentifiers",
    "FilterEvidence",
    "CandidatePaper",
    "CandidatePoolStats",
    "CandidatePoolResult",
]
