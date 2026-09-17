from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..candidate_pool.models import FilterEvidence, MatchStatus
from ..query_understanding.schema import (
    ConstraintLevel,
    ConstraintOperator,
    ConstraintType,
    ConstraintValue,
)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CriterionKind(str, Enum):
    CONTENT = "content"
    CONSTRAINT = "constraint"
    RELATION = "relation"


class SelectorCriterion(StrictBaseModel):
    criterion_id: str
    kind: CriterionKind
    level: ConstraintLevel
    description: str
    source_text: str = ""

    constraint_type: ConstraintType | None = None
    operator: ConstraintOperator | None = None
    expected_values: list[ConstraintValue] = Field(default_factory=list)

    subject: str | None = None
    predicate: str | None = None
    object: str | None = None

    synthetic: bool = False


class CriterionJudgement(StrictBaseModel):
    criterion_id: str
    status: MatchStatus
    evidence: str | None = None
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_verified: bool = False
    evidence_source: Literal["title", "abstract", "mixed"] | None = None

    backend_name: str | None = None
    model_name: str | None = None
    warnings: list[str] = Field(default_factory=list)


class BackendPaperJudgement(StrictBaseModel):
    candidate_id: str
    judgements: list[CriterionJudgement] = Field(default_factory=list)
    summary: str | None = None
    backend_name: str
    model_name: str | None = None
    warnings: list[str] = Field(default_factory=list)


class SelectorBackendMetrics(StrictBaseModel):
    api_calls: int = 0
    repair_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model_names: list[str] = Field(default_factory=list)
    failed_batches: int = 0


class PaperSelectionResult(StrictBaseModel):
    candidate_id: str
    final_status: MatchStatus
    hard_status: MatchStatus
    exclude_match_status: MatchStatus
    soft_match_score: float = Field(default=0.0, ge=0.0, le=1.0)

    semantic_judgements: list[CriterionJudgement] = Field(default_factory=list)
    metadata_evidence: list[FilterEvidence] = Field(default_factory=list)

    prior_score: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_score: float | None = None
    semantic_known_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    summary: str | None = None
    warnings: list[str] = Field(default_factory=list)


class SelectorMetrics(StrictBaseModel):
    candidates_seen: int = 0
    criteria_count: int = 0

    pass_count: int = 0
    unknown_count: int = 0
    fail_count: int = 0

    llm_api_calls: int = 0
    llm_repair_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_latency_ms: float = 0.0
    llm_failed_batches: int = 0
    llm_models: list[str] = Field(default_factory=list)

    prior_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class SelectorOutput(StrictBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    results: list[PaperSelectionResult] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    uncertain_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)
    metrics: SelectorMetrics = Field(default_factory=SelectorMetrics)
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "CriterionKind",
    "SelectorCriterion",
    "CriterionJudgement",
    "BackendPaperJudgement",
    "SelectorBackendMetrics",
    "PaperSelectionResult",
    "SelectorMetrics",
    "SelectorOutput",
]
