from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictBaseModel(BaseModel):
    """query_generation 内部统一严格模型。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class QueryKind(str, Enum):
    """生成查询的用途。"""

    EXACT = "exact"
    # 精确标题 / 唯一标识符定位；不做语义扩展

    CORE = "core"
    # 最接近 QueryAnalysisResult.content 的核心主题查询

    RELATION = "relation"
    # 保留关键 subject-predicate-object 关系的查询

    ALIAS = "alias"
    # 仅使用 Query Understanding 已确认 aliases 的低风险变体

    EXPANDED = "expanded"
    # 最多一次 LLM 调用生成的高置信度学术等价表达


class GenerationMode(str, Enum):
    DIRECT_LOOKUP = "direct_lookup"
    METADATA_ONLY = "metadata_only"
    LIGHT = "light"
    STANDARD = "standard"
    BROAD = "broad"


class ExpansionBudget(StrictBaseModel):
    """不同类型查询的数量预算。总量由 planner 保证不超过 max_total_queries。"""

    exact: int = Field(default=0, ge=0, le=2)
    core: int = Field(default=1, ge=0, le=2)
    relation: int = Field(default=0, ge=0, le=3)
    alias: int = Field(default=0, ge=0, le=2)
    expanded: int = Field(default=0, ge=0, le=3)

    @property
    def total(self) -> int:
        return self.exact + self.core + self.relation + self.alias + self.expanded


class QueryGenerationPlan(StrictBaseModel):
    """planner.py 的确定性输出，不调用 LLM。"""

    mode: GenerationMode
    budget: ExpansionBudget

    use_llm: bool = False
    max_total_queries: int = Field(default=5, ge=1, le=10)

    # 如果用户查询本身无法/不需要形成内容检索式，可为 False。
    include_original_content: bool = True

    reason: str = ""

    @model_validator(mode="after")
    def check_budget(self):
        if self.budget.total > self.max_total_queries:
            raise ValueError(
                f"query budget={self.budget.total} exceeds max_total_queries={self.max_total_queries}"
            )
        if self.use_llm and self.budget.expanded <= 0:
            raise ValueError("use_llm=true 时 expanded budget 必须 > 0")
        return self


class GeneratedQuery(StrictBaseModel):
    """交给后续 Route Builder 的标准查询对象。"""

    query_id: str
    text: str
    kind: QueryKind

    # 越大越优先；只用于路线预算/调度，不表示论文相关性。
    priority: float = Field(default=0.5, ge=0.0, le=1.0)

    # 可追踪“为什么生成这条 Query”。
    based_on_entity_names: list[str] = Field(default_factory=list)
    based_on_relation_ids: list[str] = Field(default_factory=list)
    based_on_constraint_ids: list[str] = Field(default_factory=list)

    # LLM 扩展时的简短解释；本地规则生成可留空。
    reason: str = ""

    # validator 可能补充非阻断性风险说明。
    warnings: list[str] = Field(default_factory=list)


class QueryGenerationResult(StrictBaseModel):
    """query_generation 模块的唯一标准输出。"""

    schema_version: Literal["1.0"] = "1.0"

    original_query: str
    core_content: str
    plan: QueryGenerationPlan

    queries: list[GeneratedQuery] = Field(default_factory=list)

    llm_used: bool = False
    degraded: bool = False
    # degraded=True 表示 LLM 扩展失败，但本地 CORE/RELATION/ALIAS 仍可继续用于检索。

    warnings: list[str] = Field(default_factory=list)


class LLMExpandedQuery(StrictBaseModel):
    """LLM 只允许返回非常小的、受约束的扩展对象。"""

    query: str
    reason: str = ""


class LLMExpansionPayload(StrictBaseModel):
    expanded_queries: list[LLMExpandedQuery] = Field(default_factory=list)


class GenerationMetrics(StrictBaseModel):
    """与比赛运行效率指标对应。"""

    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    model_names: list[str] = Field(default_factory=list)


class QueryValidationIssue(StrictBaseModel):
    code: str
    message: str
    query_id: str | None = None
    severity: Literal["error", "warning"] = "warning"


class QueryValidationReport(StrictBaseModel):
    issues: list[QueryValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[QueryValidationIssue]:
        return [x for x in self.issues if x.severity == "error"]

    @property
    def warnings(self) -> list[QueryValidationIssue]:
        return [x for x in self.issues if x.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors


class QueryGenerationOutput(StrictBaseModel):
    result: QueryGenerationResult
    validation: QueryValidationReport
    metrics: GenerationMetrics


__all__ = [
    "QueryKind",
    "GenerationMode",
    "ExpansionBudget",
    "QueryGenerationPlan",
    "GeneratedQuery",
    "QueryGenerationResult",
    "LLMExpandedQuery",
    "LLMExpansionPayload",
    "GenerationMetrics",
    "QueryValidationIssue",
    "QueryValidationReport",
    "QueryGenerationOutput",
]
