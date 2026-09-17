from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..query_generation.models import QueryKind


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchSource(str, Enum):
    PASA_LOCAL = "pasa_local"
    OPENALEX = "openalex"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ARXIV = "arxiv"
    GOOGLE_SEARCH = "google_search"


class RetrievalStrategy(str, Enum):
    RELEVANCE = "relevance"
    TITLE = "title"
    IDENTIFIER = "identifier"
    AUTHOR = "author"
    FILTER_ONLY = "filter_only"
    CITATION = "citation"  # V2 预留；V1 route_builder 默认不会生成


class RouteStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    FAILED = "failed"
    SKIPPED = "skipped"


class RetrievalFilters(StrictBaseModel):
    """跨数据源的中立过滤表达。

    Route Builder 只把“可安全下推”的 Hard Metadata 写入这里；
    各 Source Adapter 再转换为该 API 的真实参数。
    """

    year_start: int | None = Field(default=None, ge=0)
    year_end: int | None = Field(default=None, ge=0)
    authors: list[str] = Field(default_factory=list)
    venues: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    fields_of_study: list[str] = Field(default_factory=list)
    language: str | None = None
    min_citations: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_years(self):
        if self.year_start is not None and self.year_end is not None and self.year_start > self.year_end:
            raise ValueError("year_start cannot be greater than year_end")
        return self


class SourceCapabilities(StrictBaseModel):
    relevance_search: bool = True
    title_search: bool = False
    identifier_lookup: bool = False
    author_search: bool = False
    filter_only: bool = False
    citation_search: bool = False
    supported_filters: set[str] = Field(default_factory=set)

    def supports_strategy(self, strategy: RetrievalStrategy) -> bool:
        return {
            RetrievalStrategy.RELEVANCE: self.relevance_search,
            RetrievalStrategy.TITLE: self.title_search,
            RetrievalStrategy.IDENTIFIER: self.identifier_lookup,
            RetrievalStrategy.AUTHOR: self.author_search,
            RetrievalStrategy.FILTER_ONLY: self.filter_only,
            RetrievalStrategy.CITATION: self.citation_search,
        }[strategy]


class RouteSpec(StrictBaseModel):
    route_id: str
    query_id: str
    query_text: str
    query_kind: QueryKind

    source: SearchSource
    strategy: RetrievalStrategy

    priority: float = Field(default=1.0, ge=0.0, le=2.0)
    limit: int = Field(default=20, ge=1, le=100)

    logic_branch_id: str | None = None
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)

    pushed_constraint_ids: list[str] = Field(default_factory=list)
    deferred_constraint_ids: list[str] = Field(default_factory=list)

    enabled: bool = True
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_route(self):
        if self.strategy != RetrievalStrategy.FILTER_ONLY and not self.query_text.strip():
            raise ValueError("non-filter-only route must have non-empty query_text")
        return self


class AdaptiveDepthBudget(StrictBaseModel):
    """Two-round per-route retrieval depth budget.

    ``first_round_limit`` is the inexpensive probe depth.  Routes that fill
    that depth and provide enough de-duplicated candidate yield may be
    re-issued at ``max_route_limit``.  Keeping these values in the plan makes
    local and online runs inspectable and removes 28/50 policy constants from
    the executor.
    """

    enabled: bool = False
    first_round_limit: int = Field(default=20, ge=1, le=100)
    max_route_limit: int = Field(default=20, ge=1, le=100)
    min_new_candidate_rate: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_limits(self):
        if self.first_round_limit > self.max_route_limit:
            raise ValueError("first_round_limit cannot exceed max_route_limit")
        if self.enabled and self.first_round_limit == self.max_route_limit:
            raise ValueError("adaptive depth requires first_round_limit < max_route_limit")
        return self


class RetrievalPlan(StrictBaseModel):
    schema_version: Literal["1.0"] = "1.0"
    routes: list[RouteSpec] = Field(default_factory=list)
    max_routes: int = Field(default=10, ge=1, le=30)
    max_total_hits: int = Field(default=300, ge=1, le=2000)
    depth_budget: AdaptiveDepthBudget = Field(default_factory=AdaptiveDepthBudget)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_route_count(self):
        enabled = [r for r in self.routes if r.enabled]
        if len(enabled) > self.max_routes:
            raise ValueError(f"enabled routes={len(enabled)} exceeds max_routes={self.max_routes}")
        ids = [r.route_id for r in self.routes]
        if len(ids) != len(set(ids)):
            raise ValueError("route_id must be globally unique")
        return self


class RetrievalOrigin(StrictBaseModel):
    source: SearchSource
    route_id: str
    query_id: str
    query_text: str
    query_kind: QueryKind
    strategy: RetrievalStrategy
    logic_branch_id: str | None = None
    rank: int = Field(ge=1)
    source_score: float | None = None
    applied_constraint_ids: list[str] = Field(default_factory=list)
    citation_direction: Literal["forward", "backward"] | None = None
    citation_seed_id: str | None = None
    citation_hop: int | None = Field(default=None, ge=1)
    search_iteration: int | None = Field(default=None, ge=1)


class SourcePaperRecord(StrictBaseModel):
    """各数据源归一化后的论文元数据，不包含“由哪条检索路线召回”。"""

    source: SearchSource
    source_id: str

    title: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None

    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None
    pmid: str | None = None

    citation_count: int | None = Field(default=None, ge=0)
    publication_types: list[str] = Field(default_factory=list)
    fields_of_study: list[str] = Field(default_factory=list)
    language: str | None = None

    urls: list[str] = Field(default_factory=list)
    raw: dict[str, Any] | None = None


class CitationLookupKey(StrictBaseModel):
    """Source-neutral identifiers used to resolve a snowball seed."""

    candidate_id: str
    title: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None


class CitationEdge(StrictBaseModel):
    source: SearchSource
    direction: Literal["forward", "backward"]
    seed_candidate_id: str
    target: SourcePaperRecord
    is_influential: bool = False
    num_contexts: int = Field(default=1, ge=1)


class CitationNeighborResult(StrictBaseModel):
    edges: list[CitationEdge] = Field(default_factory=list)
    api_calls: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class RetrievalHit(SourcePaperRecord):
    origin: RetrievalOrigin


class AdapterSearchResult(StrictBaseModel):
    hits: list[RetrievalHit] = Field(default_factory=list)
    api_calls: int = Field(default=1, ge=0)
    warnings: list[str] = Field(default_factory=list)


class AdapterFetchResult(StrictBaseModel):
    records: list[SourcePaperRecord] = Field(default_factory=list)
    api_calls: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class RouteExecutionResult(StrictBaseModel):
    route_id: str
    source: SearchSource
    status: RouteStatus
    hits: list[RetrievalHit] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=1, ge=0)
    api_calls: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class RetrievalMetrics(StrictBaseModel):
    planned_routes: int = 0
    executed_routes: int = 0
    successful_routes: int = 0
    empty_routes: int = 0
    failed_routes: int = 0
    timeout_routes: int = 0

    api_calls: int = 0
    retries: int = 0
    raw_hits: int = 0

    source_hit_counts: dict[str, int] = Field(default_factory=dict)
    source_api_calls: dict[str, int] = Field(default_factory=dict)
    source_latencies_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    retrieval_rounds: int = 1
    stop_reason: str | None = None


class RetrievalRoundMetrics(StrictBaseModel):
    round_number: int = Field(ge=1)
    requested_route_limit: int = Field(ge=1, le=100)
    route_ids: list[str] = Field(default_factory=list)
    executed_routes: int = 0
    raw_hits: int = 0
    incremental_hits: int = 0
    refetched_hits_discarded: int = 0
    cumulative_unique_candidates: int = 0
    added_unique_candidates: int = 0
    new_candidate_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    stop_reason: str | None = None


class RetrievalRoundResult(StrictBaseModel):
    metrics: RetrievalRoundMetrics
    # Cumulative snapshot after this round, used by reproducible evaluations.
    hits: list[RetrievalHit] = Field(default_factory=list)


class RetrievalOutput(StrictBaseModel):
    plan: RetrievalPlan
    executions: list[RouteExecutionResult] = Field(default_factory=list)
    hits: list[RetrievalHit] = Field(default_factory=list)
    rounds: list[RetrievalRoundResult] = Field(default_factory=list)
    metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "SearchSource",
    "RetrievalStrategy",
    "RouteStatus",
    "RetrievalFilters",
    "SourceCapabilities",
    "RouteSpec",
    "AdaptiveDepthBudget",
    "RetrievalPlan",
    "RetrievalOrigin",
    "SourcePaperRecord",
    "CitationLookupKey",
    "CitationEdge",
    "CitationNeighborResult",
    "RetrievalHit",
    "AdapterSearchResult",
    "AdapterFetchResult",
    "RouteExecutionResult",
    "RetrievalMetrics",
    "RetrievalRoundMetrics",
    "RetrievalRoundResult",
    "RetrievalOutput",
]
