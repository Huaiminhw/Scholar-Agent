# query_understanding/schema.py

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================
# 0. 基础配置
# ============================================================

class StrictBaseModel(BaseModel):
    """
    所有 Schema 的基础类：
    - extra="forbid"：禁止 LLM 输出未定义字段
    - str_strip_whitespace=True：自动清理字符串首尾空格
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


# ============================================================
# 1. Query Type（查询类型）
#    参考 Ai2 Paper Finder 的路由思想
# ============================================================

class QueryType(str, Enum):
    BROAD_BY_DESCRIPTION = "BROAD_BY_DESCRIPTION"
    # 按自然语言主题描述查找一组论文

    SPECIFIC_BY_TITLE = "SPECIFIC_BY_TITLE"
    # 用户明确给出了论文标题

    SPECIFIC_BY_NAME = "SPECIFIC_BY_NAME"
    # 用户给出了模型、方法、数据集或论文简称

    SPECIFIC_BY_IDENTIFIER = "SPECIFIC_BY_IDENTIFIER"
    # DOI / arXiv ID / Corpus ID 等精确标识符

    BY_AUTHOR = "BY_AUTHOR"
    # 按作者找论文

    METADATA_ONLY_NO_AUTHOR = "METADATA_ONLY_NO_AUTHOR"
    # 基本只有年份、会议、期刊等元数据条件

    UNKNOWN = "UNKNOWN"
    # 无法可靠识别


# ============================================================
# 2. Query Intent（查询意图）
#    主要参考 SPAR 的 Query Intent Analysis（查询意图分析）
# ============================================================

class SearchTask(str, Enum):
    PAPER_SEARCH = "paper_search"
    # 查找一组论文

    PAPER_LOOKUP = "paper_lookup"
    # 查找某一篇具体论文

    AUTHOR_SEARCH = "author_search"
    # 查找某作者相关论文

    CITATION_SEARCH = "citation_search"
    # 引用 / 被引用关系搜索

    LITERATURE_REVIEW = "literature_review"
    # 文献综述型搜索


class RetrievalMode(str, Enum):
    COMPREHENSIVE = "comprehensive"
    # 尽可能全面，偏重 Recall（召回率）

    PRECISE = "precise"
    # 尽可能严格，偏重 Precision（精确率）

    LATEST = "latest"
    # 最新研究优先

    SEMINAL = "seminal"
    # 经典 / 高影响力论文优先

    OVERVIEW = "overview"
    # 综述式搜索

    SPECIFIC = "specific"
    # 精确定位论文


class QueryIntent(StrictBaseModel):
    task: SearchTask = SearchTask.PAPER_SEARCH

    mode: RetrievalMode = RetrievalMode.COMPREHENSIVE

    goal: str = Field(
        default="",
        description="对用户检索目标的简洁概括，不增加用户没有表达的信息"
    )


# ============================================================
# 3. Constraint（查询约束）
#    主要参考 Ai2 Paper Finder 的 Specifications /
#    Relevance Criteria 思想
# ============================================================

class ConstraintType(str, Enum):
    TOPIC = "topic"
    # 研究主题

    TASK = "task"
    # 论文解决的任务

    METHOD = "method"
    # 方法 / 算法

    TECHNOLOGY = "technology"
    # 技术体系，例如 LLM Agent

    DATASET = "dataset"
    # 数据集

    EVALUATION = "evaluation"
    # 实验 / 验证要求

    AUTHOR = "author"
    # 作者

    VENUE = "venue"
    # 会议 / 期刊

    YEAR = "year"
    # 年份

    PUBLICATION_TYPE = "publication_type"
    # journal / conference / survey 等

    MIN_CITATIONS = "min_citations"
    # 最低引用量

    FIELD_OF_STUDY = "field_of_study"
    # 学科

    LANGUAGE = "language"
    # 论文语言

    IDENTIFIER = "identifier"
    # DOI / arXiv ID 等

    CODE_AVAILABILITY = "code_availability"
    # 是否要求开源代码

    DATA_AVAILABILITY = "data_availability"
    # 是否要求开放数据

    OTHER = "other"


class ConstraintOperator(str, Enum):
    EQ = "eq"
    # 等于

    NE = "ne"
    # 不等于

    GT = "gt"

    GTE = "gte"

    LT = "lt"

    LTE = "lte"

    ANY_OF = "any_of"
    # 多个值满足任意一个即可：A OR B

    ALL_OF = "all_of"
    # 多个值必须全部满足：A AND B

    CONTAINS = "contains"

    NOT_CONTAINS = "not_contains"

    BETWEEN = "between"


ConstraintValue = str | int | float | bool


class ConstraintItem(StrictBaseModel):
    """
    普通约束。

    示例：
        year >= 2023
        venue = ACL OR NeurIPS
        dataset = real scientific queries
    """

    id: str = Field(
        description="全局唯一编号，例如 C1、C2、P1、E1"
    )

    type: ConstraintType

    operator: ConstraintOperator = ConstraintOperator.EQ

    value: ConstraintValue | None = Field(
        default=None,
        description="单值约束"
    )

    values: list[ConstraintValue] = Field(
        default_factory=list,
        description="多值约束，例如 ACL OR NeurIPS"
    )

    source_text: str = Field(
        default="",
        description="该约束在用户原始查询中的对应文本"
    )

    normalized_text: str | None = Field(
        default=None,
        description="标准化后的表达；不能改变原始语义"
    )


class ConstraintSet(StrictBaseModel):
    """
    Hard / Soft / Exclude（硬约束 / 软约束 / 排除约束）
    """

    hard: list[ConstraintItem] = Field(
        default_factory=list,
        description="论文必须满足，否则 Selector 应判定为不符合"
    )

    soft: list[ConstraintItem] = Field(
        default_factory=list,
        description="最好满足，用于 Ranking（排序），不能直接过滤"
    )

    exclude: list[ConstraintItem] = Field(
        default_factory=list,
        description="满足这些条件的论文需要排除"
    )


# ============================================================
# 4. Relation Constraint（关系约束）
#    参考 Ai2：不能只保留关键词，还必须保留关系
# ============================================================

class ConstraintLevel(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    EXCLUDE = "exclude"


class RelationConstraint(StrictBaseModel):
    """
    示例：

    reinforcement learning
           ↓ optimize
    LLM academic search agent
    """

    id: str = Field(
        description="关系编号，例如 R1"
    )

    subject: str

    predicate: str

    object: str

    level: ConstraintLevel = ConstraintLevel.HARD

    source_text: str = Field(
        default="",
        description="用户原话中表达该关系的文本"
    )


# ============================================================
# 5. Boolean Logic（布尔逻辑）
#    用于真正保存 AND / OR / NOT
# ============================================================

class LogicNode(StrictBaseModel):
    """
    逻辑树节点。

    LEAF:
        ref_id = "C1"

    AND:
        children = [C1, C2]

    OR:
        children = [C3, C4]
    """

    operator: Literal["LEAF", "AND", "OR", "NOT"]

    ref_id: str | None = Field(
        default=None,
        description="LEAF 节点引用 Constraint / Relation 的 ID"
    )

    children: list["LogicNode"] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_logic_node(self):

        if self.operator == "LEAF":
            if not self.ref_id:
                raise ValueError("LEAF 节点必须提供 ref_id")

            if self.children:
                raise ValueError("LEAF 节点不能包含 children")

        else:
            if self.ref_id is not None:
                raise ValueError(
                    f"{self.operator} 节点不能设置 ref_id"
                )

            if not self.children:
                raise ValueError(
                    f"{self.operator} 节点必须包含 children"
                )

            if self.operator == "NOT" and len(self.children) != 1:
                raise ValueError("NOT 节点必须只有一个 child")

        return self


class ConstraintLogic(StrictBaseModel):
    """
    如果没有显式填写：
    hard 默认 AND；
    soft 默认作为排序加分项；
    exclude 默认命中任意一个即排除。
    """

    hard: LogicNode | None = None

    soft: LogicNode | None = None

    exclude: LogicNode | None = None


# ============================================================
# 6. Extracted Properties（检索附加偏好）
#    参考 Ai2 ExtractedProperties
# ============================================================

class ExtractedProperties(StrictBaseModel):

    recent_first: bool = False
    # 最新论文优先

    recent_last: bool = False
    # 早期论文优先

    central_first: bool = False
    # 高引用 / 高影响力论文优先

    central_last: bool = False
    # 边缘 / 低引用论文优先

    specific_paper_name: str | None = None
    # 核心论文、方法、模型、数据集名称

    suitable_for_by_citing: bool | None = None
    # 是否适合从核心论文进行 Citation Expansion（引用扩展）

    @model_validator(mode="after")
    def check_conflicts(self):

        if self.recent_first and self.recent_last:
            raise ValueError(
                "recent_first 和 recent_last 不能同时为 true"
            )

        if self.central_first and self.central_last:
            raise ValueError(
                "central_first 和 central_last 不能同时为 true"
            )

        return self


# ============================================================
# 7. Entity（实体与标准化表达）
#    为后续 Query Generator（查询生成器）服务
# ============================================================

class EntityType(str, Enum):
    TOPIC = "topic"
    METHOD = "method"
    MODEL = "model"
    DATASET = "dataset"
    AUTHOR = "author"
    VENUE = "venue"
    PAPER = "paper"
    OTHER = "other"


class QueryEntity(StrictBaseModel):

    type: EntityType

    canonical_name: str = Field(
        description="标准名称"
    )

    aliases: list[str] = Field(
        default_factory=list,
        description="高置信度同义词 / 缩写，不允许无依据扩展"
    )

    source_text: str = ""


# ============================================================
# 8. Domain（学科领域）
#    参考 Ai2 DomainsIdentified
# ============================================================

class DomainInfo(StrictBaseModel):

    main_field: str | None = Field(
        default=None,
        description="主要研究领域"
    )

    secondary_fields: list[str] = Field(
        default_factory=list,
        description="次级相关领域"
    )


# ============================================================
# 9. Clarification（查询澄清）
#    为真正存在关键歧义的 Query 保留
# ============================================================

class ClarificationInfo(StrictBaseModel):

    needed: bool = False

    unresolved_ambiguities: list[str] = Field(
        default_factory=list,
        description="当前不能可靠确定、且会明显影响检索结果的问题"
    )

    questions: list[str] = Field(
        default_factory=list,
        description="必要时向用户提出的澄清问题"
    )


# ============================================================
# 10. Result Requirements（结果要求）
#     用户如果明确要求“20篇、列表、关系图、带DOI”等，
#     也需要传递给后面的 Planner
# ============================================================

class OutputStructure(str, Enum):
    LIST = "list"
    TABLE = "table"
    RELATION_GRAPH = "relation_graph"
    MIXED = "mixed"
    UNSPECIFIED = "unspecified"


class ResultRequirements(StrictBaseModel):

    requested_top_k: int | None = Field(
        default=None,
        ge=1,
        description="用户明确要求的论文数量"
    )

    required_fields: list[str] = Field(
        default_factory=list,
        description="例如 title、authors、year、doi、url"
    )

    output_structure: OutputStructure = OutputStructure.UNSPECIFIED

    need_relation_graph: bool = False

    need_open_access: bool = False

    need_code: bool = False


# ============================================================
# 11. Parse Status（解析状态）
# ============================================================

class ParseStatus(str, Enum):
    SUCCESS = "success"

    NEEDS_CLARIFICATION = "needs_clarification"

    UNSUPPORTED = "unsupported"


# ============================================================
# 12. 查询理解模块最终统一输出
# ============================================================

class QueryAnalysisResult(StrictBaseModel):
    """
    整个 Query Understanding（查询理解）模块的唯一标准输出。

    User Query
        ↓
    1 次 LLM
        ↓
    QueryAnalysisResult
        ↓
    Retrieval Planner / Query Generator / Selector / Ranker
    """

    schema_version: Literal["1.0"] = "1.0"

    status: ParseStatus = ParseStatus.SUCCESS

    original_query: str

    query_type: QueryType

    intent: QueryIntent

    content: str = Field(
        description=(
            "去除年份、作者、会议等元数据后，"
            "保留关系和核心语义的内容检索表达"
        )
    )

    constraints: ConstraintSet = Field(
        default_factory=ConstraintSet
    )

    relations: list[RelationConstraint] = Field(
        default_factory=list
    )

    logic: ConstraintLogic = Field(
        default_factory=ConstraintLogic
    )

    properties: ExtractedProperties = Field(
        default_factory=ExtractedProperties
    )

    entities: list[QueryEntity] = Field(
        default_factory=list
    )

    domains: DomainInfo = Field(
        default_factory=DomainInfo
    )

    clarification: ClarificationInfo = Field(
        default_factory=ClarificationInfo
    )

    result_requirements: ResultRequirements = Field(
        default_factory=ResultRequirements
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="解析过程中发现但不足以导致失败的问题"
    )


# Pydantic 处理递归 LogicNode
LogicNode.model_rebuild()
