from __future__ import annotations

from dataclasses import dataclass

from ..query_understanding.schema import (
    QueryAnalysisResult,
    QueryType,
    RetrievalMode,
    SearchTask,
)

from .models import ExpansionBudget, GenerationMode, QueryGenerationPlan


@dataclass(frozen=True)
class PlannerConfig:
    """V1 默认控制在 3~5 条查询；特殊精确查询更少。"""

    max_total_queries: int = 5
    enable_llm_expansion: bool = True


class QueryGenerationPlanner:
    """
    纯规则 Planner：
    - 不重复调用 LLM 分析 intent/domain；
    - 直接消费 QueryAnalysisResult；
    - 只决定“生成哪些类型、各多少条”。
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def plan(self, analysis: QueryAnalysisResult) -> QueryGenerationPlan:
        qtype = analysis.query_type
        mode = analysis.intent.mode

        # 1) DOI / arXiv ID / PMID 等唯一标识符：只做精确定位。
        if qtype == QueryType.SPECIFIC_BY_IDENTIFIER:
            return QueryGenerationPlan(
                mode=GenerationMode.DIRECT_LOOKUP,
                budget=ExpansionBudget(exact=1, core=0),
                use_llm=False,
                max_total_queries=self.config.max_total_queries,
                include_original_content=False,
                reason="唯一标识符查询无需语义扩展。",
            )

        # 2) 精确标题：优先 exact；不做自由扩展，避免漂移。
        if qtype == QueryType.SPECIFIC_BY_TITLE:
            return QueryGenerationPlan(
                mode=GenerationMode.DIRECT_LOOKUP,
                budget=ExpansionBudget(exact=1, core=0),
                use_llm=False,
                max_total_queries=self.config.max_total_queries,
                include_original_content=False,
                reason="精确标题查询以直接定位为主。",
            )

        # 3) 只有 metadata 且没有主题内容：Query Generation 不应硬造内容查询。
        if qtype == QueryType.METADATA_ONLY_NO_AUTHOR and not analysis.content.strip():
            return QueryGenerationPlan(
                mode=GenerationMode.METADATA_ONLY,
                budget=ExpansionBudget(core=0),
                use_llm=False,
                max_total_queries=self.config.max_total_queries,
                include_original_content=False,
                reason="纯元数据查询由 Route Builder/API Filter 执行。",
            )

        # 4) 明确精确/特定查询：轻量生成。
        if mode in {RetrievalMode.SPECIFIC, RetrievalMode.PRECISE}:
            relation_num = 1 if analysis.relations else 0
            alias_num = 1 if _has_useful_aliases(analysis) else 0
            expanded_num = 0

            # SPECIFIC_BY_NAME 可以允许 1 条高置信度等价表达。
            if (
                qtype == QueryType.SPECIFIC_BY_NAME
                and self.config.enable_llm_expansion
            ):
                expanded_num = 1

            return QueryGenerationPlan(
                mode=GenerationMode.LIGHT,
                budget=_fit_budget(
                    ExpansionBudget(
                        core=1,
                        relation=relation_num,
                        alias=alias_num,
                        expanded=expanded_num,
                    ),
                    self.config.max_total_queries,
                ),
                use_llm=expanded_num > 0,
                max_total_queries=self.config.max_total_queries,
                reason="精确型查询限制扩展宽度，优先保持 Precision。",
            )

        # 5) 作者检索：作者本身由 metadata filter 处理；主题只轻量扩展。
        if qtype == QueryType.BY_AUTHOR or analysis.intent.task == SearchTask.AUTHOR_SEARCH:
            alias_num = 1 if _has_useful_aliases(analysis) else 0
            expanded_num = 1 if self.config.enable_llm_expansion and analysis.content.strip() else 0
            return QueryGenerationPlan(
                mode=GenerationMode.LIGHT,
                budget=_fit_budget(
                    ExpansionBudget(core=1 if analysis.content.strip() else 0, alias=alias_num, expanded=expanded_num),
                    self.config.max_total_queries,
                ),
                use_llm=expanded_num > 0,
                max_total_queries=self.config.max_total_queries,
                reason="作者作为过滤条件处理，主题部分仅轻量扩展。",
            )

        # 6) 综述/overview：不是强制给每条 query 加 survey；只放宽路线数量。
        if mode == RetrievalMode.OVERVIEW or analysis.intent.task == SearchTask.LITERATURE_REVIEW:
            return QueryGenerationPlan(
                mode=GenerationMode.BROAD,
                budget=_fit_budget(
                    ExpansionBudget(
                        core=1,
                        relation=min(2, len(analysis.relations)),
                        alias=1 if _has_useful_aliases(analysis) else 0,
                        expanded=2 if self.config.enable_llm_expansion else 0,
                    ),
                    self.config.max_total_queries,
                ),
                use_llm=self.config.enable_llm_expansion,
                max_total_queries=self.config.max_total_queries,
                reason="综述型任务允许更宽覆盖，但仍保持扩展受原始语义约束。",
            )

        # 7) 普通 broad/comprehensive 查询：主路径。
        broad = mode == RetrievalMode.COMPREHENSIVE
        relation_num = min(2 if broad else 1, len(analysis.relations))
        alias_num = 1 if _has_useful_aliases(analysis) else 0
        expanded_num = 2 if broad else 1
        if not self.config.enable_llm_expansion:
            expanded_num = 0

        budget = _fit_budget(
            ExpansionBudget(
                core=1 if analysis.content.strip() else 0,
                relation=relation_num,
                alias=alias_num,
                expanded=expanded_num,
            ),
            self.config.max_total_queries,
        )

        return QueryGenerationPlan(
            mode=GenerationMode.BROAD if broad else GenerationMode.STANDARD,
            budget=budget,
            use_llm=budget.expanded > 0,
            max_total_queries=self.config.max_total_queries,
            reason="普通学术查询采用 CORE + RELATION + ALIAS + 少量 LLM EXPANDED 的受控多路生成。",
        )


def _has_useful_aliases(analysis: QueryAnalysisResult) -> bool:
    return any(entity.aliases for entity in analysis.entities)


def _fit_budget(budget: ExpansionBudget, max_total: int) -> ExpansionBudget:
    """
    超预算时先牺牲 Alias，再收缩重复的 Relation/Expansion 角度；
    至少一条 CORE/EXACT 和显式 Relation 最后保留。
    """
    values = budget.model_dump()
    while sum(values.values()) > max_total:
        if values["alias"] > 0:
            values["alias"] -= 1
        elif values["relation"] > 1:
            values["relation"] -= 1
        elif values["expanded"] > 1:
            values["expanded"] -= 1
        elif values["expanded"] > 0:
            values["expanded"] -= 1
        elif values["relation"] > 0:
            values["relation"] -= 1
        elif values["core"] > 1:
            values["core"] -= 1
        elif values["exact"] > 1:
            values["exact"] -= 1
        else:
            break
    return ExpansionBudget(**values)


__all__ = ["PlannerConfig", "QueryGenerationPlanner"]
