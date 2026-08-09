from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from ..query_generation.models import (
    GeneratedQuery,
    QueryGenerationOutput,
    QueryGenerationResult,
    QueryKind,
)
from ..query_understanding.schema import (
    ConstraintItem,
    ConstraintLevel,
    ConstraintOperator,
    ConstraintType,
    LogicNode,
    EntityType,
    QueryAnalysisResult,
    QueryType,
    RetrievalMode,
)

from .models import (
    RetrievalFilters,
    RetrievalPlan,
    RetrievalStrategy,
    RouteSpec,
    SearchSource,
    SourceCapabilities,
)


DEFAULT_CAPABILITIES: dict[SearchSource, SourceCapabilities] = {
    SearchSource.OPENALEX: SourceCapabilities(
        relevance_search=True,
        title_search=True,
        identifier_lookup=True,
        author_search=True,
        filter_only=True,
        supported_filters={
            ConstraintType.YEAR.value,
            ConstraintType.AUTHOR.value,
            ConstraintType.PUBLICATION_TYPE.value,
            ConstraintType.MIN_CITATIONS.value,
        },
    ),
    SearchSource.SEMANTIC_SCHOLAR: SourceCapabilities(
        relevance_search=True,
        title_search=True,
        identifier_lookup=True,
        author_search=True,
        filter_only=False,
        supported_filters={
            ConstraintType.YEAR.value,
            ConstraintType.VENUE.value,
            ConstraintType.PUBLICATION_TYPE.value,
            ConstraintType.FIELD_OF_STUDY.value,
            ConstraintType.MIN_CITATIONS.value,
        },
    ),
    SearchSource.ARXIV: SourceCapabilities(
        relevance_search=True,
        title_search=True,
        identifier_lookup=True,
        author_search=True,
        filter_only=False,
        supported_filters=set(),
    ),
}


@dataclass(frozen=True)
class _Budget:
    max_routes: int
    route_limit: int
    max_total_hits: int


MODE_BUDGETS: dict[RetrievalMode, _Budget] = {
    RetrievalMode.SPECIFIC: _Budget(3, 8, 24),
    RetrievalMode.PRECISE: _Budget(6, 18, 110),
    RetrievalMode.LATEST: _Budget(8, 24, 190),
    RetrievalMode.SEMINAL: _Budget(8, 24, 190),
    RetrievalMode.OVERVIEW: _Budget(8, 25, 200),
    RetrievalMode.COMPREHENSIVE: _Budget(10, 28, 280),
}


@dataclass
class _LogicBranch:
    ref_ids: set[str]
    positive_ref_ids: set[str]


@dataclass
class _RouteCandidate:
    query: GeneratedQuery
    source: SearchSource
    strategy: RetrievalStrategy
    branch_id: str
    branch: _LogicBranch
    priority: float


class RetrievalRouteBuilder:
    """把 Query Understanding + Query Generation 编译成可执行 RetrievalPlan。

    设计原则：
    1. Soft 约束不下推为主检索硬过滤；
    2. Hard 约束仅在对应数据源明确支持时下推；
    3. 显式 OR 的元数据配对通过 logic_branch_id 保留；
    4. 避免 query × source 的无脑全笛卡尔积；
    5. Relation/Topic/Method 等语义约束留给 Candidate Pool / Selector。
    """

    def __init__(
        self,
        capabilities: dict[SearchSource, SourceCapabilities] | None = None,
        enabled_sources: Iterable[SearchSource] | None = None,
    ) -> None:
        self.capabilities = capabilities or DEFAULT_CAPABILITIES
        self.enabled_sources = list(enabled_sources or list(SearchSource))

    def build(
        self,
        analysis: QueryAnalysisResult,
        generation: QueryGenerationResult | QueryGenerationOutput,
    ) -> RetrievalPlan:
        gen = generation.result if isinstance(generation, QueryGenerationOutput) else generation
        budget = MODE_BUDGETS[analysis.intent.mode]
        branches = self._hard_logic_branches(analysis)
        queries = self._select_queries(analysis, gen)

        route_candidates: list[_RouteCandidate] = []
        for branch_idx, branch in enumerate(branches, start=1):
            branch_id = f"B{branch_idx}"
            for query in queries:
                strategy = self._strategy_for(analysis, query)
                for source in self._preferred_sources(analysis.intent.mode, query.kind, strategy):
                    if source not in self.enabled_sources:
                        continue
                    caps = self.capabilities[source]
                    if not caps.supports_strategy(strategy):
                        continue
                    route_candidates.append(
                        _RouteCandidate(
                            query=query,
                            source=source,
                            strategy=strategy,
                            branch_id=branch_id,
                            branch=branch,
                            priority=self._route_priority(analysis.intent.mode, query, source),
                        )
                    )

        # Metadata-only 且没有内容 Query：允许 OpenAlex FILTER_ONLY。
        if not route_candidates and analysis.query_type == QueryType.METADATA_ONLY_NO_AUTHOR:
            for branch_idx, branch in enumerate(branches, start=1):
                source = SearchSource.OPENALEX
                if source in self.enabled_sources and self.capabilities[source].filter_only:
                    pseudo = GeneratedQuery(
                        query_id="QF1",
                        text="",
                        kind=QueryKind.CORE,
                        priority=0.9,
                        reason="metadata-only filter route",
                    )
                    route_candidates.append(
                        _RouteCandidate(
                            query=pseudo,
                            source=source,
                            strategy=RetrievalStrategy.FILTER_ONLY,
                            branch_id=f"B{branch_idx}",
                            branch=branch,
                            priority=0.9,
                        )
                    )

        selected = self._allocate(route_candidates, budget.max_routes)
        routes: list[RouteSpec] = []
        warnings: list[str] = []

        hard_constraint_map = {c.id: c for c in analysis.constraints.hard}
        hard_relation_ids = {r.id for r in analysis.relations if r.level == ConstraintLevel.HARD}

        for i, candidate in enumerate(selected, start=1):
            source = candidate.source
            branch_constraint_ids = {
                rid for rid in candidate.branch.ref_ids if rid in hard_constraint_map
            }
            positive_constraint_ids = {
                rid for rid in candidate.branch.positive_ref_ids if rid in hard_constraint_map
            }
            branch_constraints = [hard_constraint_map[rid] for rid in positive_constraint_ids]
            filters, pushed = self._compile_filters(branch_constraints, source)

            # BY_AUTHOR 在极少数解析结果中作者可能只存在 entities 而未重复写入 constraints；
            # 这里做无语义扩展的安全回填，避免把主题 query_text 错当作者名。
            if candidate.strategy == RetrievalStrategy.AUTHOR and not filters.authors:
                author_names = self._author_names(analysis)
                if author_names:
                    filters = filters.model_copy(update={"authors": author_names})

            # 该逻辑分支中没有安全下推的 hard 条件继续由后端判断。
            deferred = sorted((candidate.branch.ref_ids - set(pushed)) | hard_relation_ids)

            if candidate.strategy == RetrievalStrategy.FILTER_ONLY and not pushed:
                warnings.append(
                    f"{candidate.branch_id}: metadata-only route has no safely pushable filter; skipped"
                )
                continue

            route_limit = min(budget.route_limit, max(1, budget.max_total_hits // max(1, len(selected))))
            routes.append(
                RouteSpec(
                    route_id=f"RT{i:03d}",
                    query_id=candidate.query.query_id,
                    query_text=candidate.query.text,
                    query_kind=candidate.query.kind,
                    source=source,
                    strategy=candidate.strategy,
                    priority=min(candidate.priority, 2.0),
                    limit=route_limit,
                    logic_branch_id=candidate.branch_id,
                    filters=filters,
                    pushed_constraint_ids=sorted(pushed),
                    deferred_constraint_ids=deferred,
                )
            )

        if not routes:
            warnings.append("No executable retrieval route was produced.")

        return RetrievalPlan(
            routes=routes,
            max_routes=budget.max_routes,
            max_total_hits=budget.max_total_hits,
            warnings=warnings,
        )

    def _select_queries(
        self, analysis: QueryAnalysisResult, generation: QueryGenerationResult
    ) -> list[GeneratedQuery]:
        queries = list(generation.queries)
        if analysis.query_type == QueryType.SPECIFIC_BY_IDENTIFIER:
            exact = [q for q in queries if q.kind == QueryKind.EXACT]
            return exact[:1] or queries[:1]
        if analysis.query_type == QueryType.SPECIFIC_BY_TITLE:
            exact = [q for q in queries if q.kind in {QueryKind.EXACT, QueryKind.CORE}]
            return exact[:1] or queries[:1]
        if analysis.query_type == QueryType.BY_AUTHOR:
            # 内容主题可以继续用于作者论文中的后续排序，但 route 主体由 author strategy 控制。
            return sorted(queries, key=lambda q: q.priority, reverse=True)[:2]
        return sorted(queries, key=lambda q: q.priority, reverse=True)

    def _strategy_for(self, analysis: QueryAnalysisResult, query: GeneratedQuery) -> RetrievalStrategy:
        if analysis.query_type == QueryType.SPECIFIC_BY_IDENTIFIER or query.kind == QueryKind.EXACT:
            # SPECIFIC_BY_TITLE 的 EXACT 也可能存在，所以优先判断 query_type。
            if analysis.query_type == QueryType.SPECIFIC_BY_TITLE:
                return RetrievalStrategy.TITLE
            return RetrievalStrategy.IDENTIFIER
        if analysis.query_type == QueryType.SPECIFIC_BY_TITLE:
            return RetrievalStrategy.TITLE
        if analysis.query_type == QueryType.BY_AUTHOR:
            return RetrievalStrategy.AUTHOR
        return RetrievalStrategy.RELEVANCE

    def _preferred_sources(
        self,
        mode: RetrievalMode,
        kind: QueryKind,
        strategy: RetrievalStrategy,
    ) -> list[SearchSource]:
        if strategy in {RetrievalStrategy.IDENTIFIER, RetrievalStrategy.TITLE}:
            return [
                SearchSource.SEMANTIC_SCHOLAR,
                SearchSource.OPENALEX,
                SearchSource.ARXIV,
            ]
        if strategy == RetrievalStrategy.AUTHOR:
            return [
                SearchSource.OPENALEX,
                SearchSource.SEMANTIC_SCHOLAR,
                SearchSource.ARXIV,
            ]

        if mode == RetrievalMode.LATEST:
            base = [SearchSource.ARXIV, SearchSource.SEMANTIC_SCHOLAR, SearchSource.OPENALEX]
        else:
            base = [SearchSource.SEMANTIC_SCHOLAR, SearchSource.OPENALEX, SearchSource.ARXIV]

        if kind == QueryKind.CORE:
            return base
        if kind == QueryKind.RELATION:
            return [s for s in base if s != SearchSource.ARXIV]
        if kind == QueryKind.ALIAS:
            return [SearchSource.SEMANTIC_SCHOLAR, SearchSource.OPENALEX]
        if kind == QueryKind.EXPANDED:
            return [SearchSource.SEMANTIC_SCHOLAR, SearchSource.OPENALEX]
        return base

    def _route_priority(
        self, mode: RetrievalMode, query: GeneratedQuery, source: SearchSource
    ) -> float:
        kind_weight = {
            QueryKind.EXACT: 1.10,
            QueryKind.CORE: 1.00,
            QueryKind.RELATION: 0.94,
            QueryKind.ALIAS: 0.84,
            QueryKind.EXPANDED: 0.74,
        }[query.kind]
        source_weight = {
            SearchSource.SEMANTIC_SCHOLAR: 1.00,
            SearchSource.OPENALEX: 0.96,
            SearchSource.ARXIV: 0.88,
        }[source]
        if mode == RetrievalMode.LATEST and source == SearchSource.ARXIV:
            source_weight = 1.03
        return query.priority * kind_weight * source_weight

    def _allocate(self, candidates: list[_RouteCandidate], max_routes: int) -> list[_RouteCandidate]:
        if not candidates:
            return []

        # 先保证每个显式逻辑分支至少有一个 CORE/EXACT 高质量路线。
        chosen: list[_RouteCandidate] = []
        chosen_keys: set[tuple] = set()
        branches = sorted({c.branch_id for c in candidates})
        for branch in branches:
            branch_candidates = [c for c in candidates if c.branch_id == branch]
            branch_candidates.sort(
                key=lambda c: (
                    c.query.kind not in {QueryKind.EXACT, QueryKind.CORE},
                    -c.priority,
                )
            )
            if branch_candidates:
                c = branch_candidates[0]
                chosen.append(c)
                chosen_keys.add(self._candidate_key(c))
                if len(chosen) >= max_routes:
                    return chosen

        for c in sorted(candidates, key=lambda x: x.priority, reverse=True):
            key = self._candidate_key(c)
            if key in chosen_keys:
                continue
            chosen.append(c)
            chosen_keys.add(key)
            if len(chosen) >= max_routes:
                break
        return chosen

    @staticmethod
    def _candidate_key(c: _RouteCandidate) -> tuple:
        return (c.branch_id, c.query.query_id, c.source, c.strategy)

    def _hard_logic_branches(self, analysis: QueryAnalysisResult) -> list[_LogicBranch]:
        hard_ids = {c.id for c in analysis.constraints.hard}
        hard_relation_ids = {r.id for r in analysis.relations if r.level == ConstraintLevel.HARD}

        if analysis.logic.hard is None:
            return [
                _LogicBranch(
                    ref_ids=set(hard_ids) | set(hard_relation_ids),
                    positive_ref_ids=set(hard_ids),
                )
            ]

        branches = self._logic_to_dnf(analysis.logic.hard)
        referenced = set().union(*(b.ref_ids for b in branches)) if branches else set()

        # 显式逻辑树外遗漏的 hard 条件，按默认 AND 加到每个分支。
        missing_constraints = hard_ids - referenced
        missing_relations = hard_relation_ids - referenced
        for branch in branches:
            branch.ref_ids |= missing_constraints | missing_relations
            branch.positive_ref_ids |= missing_constraints

        # 去重。
        unique: dict[tuple[tuple[str, ...], tuple[str, ...]], _LogicBranch] = {}
        for b in branches:
            key = (tuple(sorted(b.ref_ids)), tuple(sorted(b.positive_ref_ids)))
            unique[key] = b
        return list(unique.values()) or [_LogicBranch(set(), set())]

    def _logic_to_dnf(self, node: LogicNode) -> list[_LogicBranch]:
        if node.operator == "LEAF":
            assert node.ref_id is not None
            return [_LogicBranch({node.ref_id}, {node.ref_id})]

        if node.operator == "NOT":
            # V1 不把 NOT 安全下推到各 API；保留为 deferred。
            refs = self._collect_refs(node.children[0])
            return [_LogicBranch(set(refs), set())]

        child_branches = [self._logic_to_dnf(child) for child in node.children]
        if node.operator == "OR":
            return [b for group in child_branches for b in group]

        # AND: 笛卡尔积组合每个 child 的 DNF 分支。
        output: list[_LogicBranch] = []
        for combo in product(*child_branches):
            output.append(
                _LogicBranch(
                    ref_ids=set().union(*(b.ref_ids for b in combo)),
                    positive_ref_ids=set().union(*(b.positive_ref_ids for b in combo)),
                )
            )
        return output

    def _collect_refs(self, node: LogicNode) -> set[str]:
        if node.operator == "LEAF":
            return {node.ref_id} if node.ref_id else set()
        out: set[str] = set()
        for child in node.children:
            out |= self._collect_refs(child)
        return out

    def _compile_filters(
        self,
        constraints: list[ConstraintItem],
        source: SearchSource,
    ) -> tuple[RetrievalFilters, list[str]]:
        supported = self.capabilities[source].supported_filters
        pushed: list[str] = []

        year_start: int | None = None
        year_end: int | None = None
        authors: list[str] = []
        venues: list[str] = []
        publication_types: list[str] = []
        fields: list[str] = []
        language: str | None = None
        min_citations: int | None = None

        for c in constraints:
            if c.type.value not in supported:
                continue

            values = self._constraint_values(c)
            if c.type == ConstraintType.YEAR:
                bounds = self._year_bounds(c)
                if bounds is None:
                    continue
                lo, hi = bounds
                year_start = max(x for x in [year_start, lo] if x is not None) if any(
                    x is not None for x in [year_start, lo]
                ) else None
                year_end = min(x for x in [year_end, hi] if x is not None) if any(
                    x is not None for x in [year_end, hi]
                ) else None
                if year_start is not None and year_end is not None and year_start > year_end:
                    continue
                pushed.append(c.id)
            elif c.type == ConstraintType.AUTHOR and values:
                authors.extend(str(v) for v in values)
                pushed.append(c.id)
            elif c.type == ConstraintType.VENUE and values:
                venues.extend(str(v) for v in values)
                pushed.append(c.id)
            elif c.type == ConstraintType.PUBLICATION_TYPE and values:
                publication_types.extend(str(v) for v in values)
                pushed.append(c.id)
            elif c.type == ConstraintType.FIELD_OF_STUDY and values:
                fields.extend(str(v) for v in values)
                pushed.append(c.id)
            elif c.type == ConstraintType.LANGUAGE and values:
                language = str(values[0])
                pushed.append(c.id)
            elif c.type == ConstraintType.MIN_CITATIONS:
                numeric = self._numeric_lower_bound(c)
                if numeric is not None:
                    min_citations = max(min_citations or 0, numeric)
                    pushed.append(c.id)

        return (
            RetrievalFilters(
                year_start=year_start,
                year_end=year_end,
                authors=self._dedup_str(authors),
                venues=self._dedup_str(venues),
                publication_types=self._dedup_str(publication_types),
                fields_of_study=self._dedup_str(fields),
                language=language,
                min_citations=min_citations,
            ),
            pushed,
        )


    @staticmethod
    def _author_names(analysis: QueryAnalysisResult) -> list[str]:
        names: list[str] = []
        for c in analysis.constraints.hard:
            if c.type == ConstraintType.AUTHOR:
                names.extend(str(v) for v in RetrievalRouteBuilder._constraint_values(c))
        for entity in analysis.entities:
            if entity.type == EntityType.AUTHOR and entity.canonical_name:
                names.append(entity.canonical_name)
        return RetrievalRouteBuilder._dedup_str(names)

    @staticmethod
    def _constraint_values(c: ConstraintItem) -> list[object]:
        if c.values:
            return list(c.values)
        return [c.value] if c.value is not None else []

    @staticmethod
    def _year_bounds(c: ConstraintItem) -> tuple[int | None, int | None] | None:
        vals = RetrievalRouteBuilder._constraint_values(c)
        try:
            if c.operator == ConstraintOperator.BETWEEN and len(vals) >= 2:
                a, b = int(vals[0]), int(vals[1])
                return min(a, b), max(a, b)
            if not vals:
                return None
            y = int(vals[0])
            if c.operator == ConstraintOperator.EQ:
                return y, y
            if c.operator == ConstraintOperator.GTE:
                return y, None
            if c.operator == ConstraintOperator.GT:
                return y + 1, None
            if c.operator == ConstraintOperator.LTE:
                return None, y
            if c.operator == ConstraintOperator.LT:
                return None, y - 1
            if c.operator == ConstraintOperator.ANY_OF:
                ys = sorted(int(v) for v in vals)
                # 离散 ANY_OF 不能安全变成连续范围，否则引入额外年份。
                if len(ys) == 1:
                    return ys[0], ys[0]
                return None
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _numeric_lower_bound(c: ConstraintItem) -> int | None:
        vals = RetrievalRouteBuilder._constraint_values(c)
        if not vals:
            return None
        try:
            v = int(vals[0])
        except (TypeError, ValueError):
            return None
        if c.operator in {ConstraintOperator.EQ, ConstraintOperator.GTE}:
            return v
        if c.operator == ConstraintOperator.GT:
            return v + 1
        return None

    @staticmethod
    def _dedup_str(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            key = value.strip().casefold()
            if key and key not in seen:
                seen.add(key)
                out.append(value.strip())
        return out


def build_retrieval_plan(
    analysis: QueryAnalysisResult,
    generation: QueryGenerationResult | QueryGenerationOutput,
    *,
    capabilities: dict[SearchSource, SourceCapabilities] | None = None,
    enabled_sources: Iterable[SearchSource] | None = None,
) -> RetrievalPlan:
    return RetrievalRouteBuilder(capabilities, enabled_sources).build(analysis, generation)


__all__ = ["RetrievalRouteBuilder", "build_retrieval_plan", "DEFAULT_CAPABILITIES"]
