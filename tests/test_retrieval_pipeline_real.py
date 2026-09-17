# tests/test_retrieval_pipeline_real.py
"""
Retrieval + Candidate Pool V1.0 冻结前的真实多 Query / 多 Route 集成验收。

目标：
1. F1 导向：尽量扩大 Recall，同时防止 Query Drift、Soft→Hard、OR 配对破坏、
   缺失元数据误删等 Precision/Recall 风险；
2. 运行效率导向：限制真实执行 Route/API 数量，记录 LLM Token、API Calls、Retry、Latency；
3. 工程稳定性：真实 OpenAlex + Semantic Scholar + arXiv，多路线并发，允许外部 API
   的有限重试，但任何语义错误必须直接失败；
4. Candidate Pool：真实跨源 Dedup / Fusion / Origins / Hard Filter / Weighted RRF。

重要说明：
- Query Understanding 已有独立真实测试并冻结，本文件从人工构造、语义已知的
  QueryAnalysisResult 开始，避免把 Query Understanding 的 LLM 延迟/随机性混进
  Retrieval V1.0 的验收。
- Query Generation 使用真实 LLM，验证 QC/QR/QA/QE 能真正进入 Route Builder。
- 本测试没有 gold relevance labels，因此不能伪造“真实 F1”；会输出 F1-oriented
  proxy diagnostics（召回增益、硬过滤、Top-N、跨源/跨路线证据）。真正 F1 应在
  ASTA/PASA/赛题标注集上计算。

推荐运行：

# 先跑最重要的 Complex case（默认）
$env:RUN_REAL_RETRIEVAL_PIPELINE_TESTS="1"
python -m pytest -s tests/test_retrieval_pipeline_real.py

# 冻结 V1.0 前完整跑 3 个 case
$env:RUN_REAL_RETRIEVAL_PIPELINE_TESTS="1"
$env:REAL_PIPELINE_CASES="all"
python -m pytest -s tests/test_retrieval_pipeline_real.py

可调但默认已经偏比赛效率：
REAL_PIPELINE_MAX_ROUTES=6
REAL_PIPELINE_PER_ROUTE_LIMIT=10
REAL_PIPELINE_TOP_N=40
REAL_PIPELINE_MAX_RETRIEVAL_API_CALLS=12
REAL_PIPELINE_MAX_QGEN_TOKENS=6000
REAL_PIPELINE_MAX_LATENCY_MS=0   # 0=不做网络延迟硬断言，只记录
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from scholar_agent_v2.candidate_pool.models import MatchStatus
from scholar_agent_v2.candidate_pool.pool import CandidatePool
from scholar_agent_v2.query_generation.generator import GeneratorConfig, QueryGenerator
from scholar_agent_v2.query_generation.models import QueryGenerationOutput, QueryKind
from scholar_agent_v2.query_understanding.schema import (
    ConstraintItem,
    ConstraintLevel,
    ConstraintLogic,
    ConstraintOperator,
    ConstraintSet,
    ConstraintType,
    EntityType,
    LogicNode,
    QueryAnalysisResult,
    QueryEntity,
    QueryIntent,
    QueryType,
    RelationConstraint,
    RetrievalMode,
    SearchTask,
)
from scholar_agent_v2.retrieval.config import get_enabled_sources
from scholar_agent_v2.retrieval.models import (
    RetrievalOutput,
    RetrievalPlan,
    RouteSpec,
    RouteStatus,
    SearchSource,
)
from scholar_agent_v2.retrieval.orchestrator import RetrievalOrchestrator
from scholar_agent_v2.retrieval.route_builder import RetrievalRouteBuilder
from scholar_agent_v2.retrieval.sources.arxiv import ArxivAdapter
from scholar_agent_v2.retrieval.sources.google_search import GoogleSearchAdapter
from scholar_agent_v2.retrieval.sources.openalex import OpenAlexAdapter
from scholar_agent_v2.retrieval.sources.semantic_scholar import SemanticScholarAdapter
from tests.real_llm_client import build_client


# =============================================================================
# Runtime configuration
# =============================================================================

RUN_REAL = os.getenv("RUN_REAL_RETRIEVAL_PIPELINE_TESTS", "0") == "1"
CASE_SELECTOR = os.getenv("REAL_PIPELINE_CASES", "complex").strip().lower()

MAX_EXEC_ROUTES = int(os.getenv("REAL_PIPELINE_MAX_ROUTES", "6"))
PER_ROUTE_LIMIT = int(os.getenv("REAL_PIPELINE_PER_ROUTE_LIMIT", "10"))
TOP_N = int(os.getenv("REAL_PIPELINE_TOP_N", "40"))

MAX_RETRIEVAL_API_CALLS = int(
    os.getenv("REAL_PIPELINE_MAX_RETRIEVAL_API_CALLS", "12")
)
MAX_QGEN_TOKENS = int(os.getenv("REAL_PIPELINE_MAX_QGEN_TOKENS", "6000"))
MAX_LATENCY_MS = float(os.getenv("REAL_PIPELINE_MAX_LATENCY_MS", "0"))

ROUTE_TIMEOUT_SECONDS = float(
    os.getenv("REAL_PIPELINE_ROUTE_TIMEOUT_SECONDS", "35")
)
MAX_ATTEMPTS = int(os.getenv("REAL_PIPELINE_MAX_ATTEMPTS", "2"))
RETRY_BACKOFF_SECONDS = float(
    os.getenv("REAL_PIPELINE_RETRY_BACKOFF_SECONDS", "1.0")
)

# 真实 API 的 source-aware 执行上限：
# - S2 key 的常见速率限制较严格，所以默认只执行 1 条 S2 route；
# - arXiv 也避免同一测试连续发多条请求；
# - 多 Query 的额外 Recall 主要由 OpenAlex route 承担。
# 这只是“真实验收执行预算”，不会修改/掩盖 Route Builder 生成的 native plan；
# native plan 会完整打印并做语义断言。
SOURCE_ROUTE_CAPS: dict[SearchSource, int] = {
    SearchSource.OPENALEX: int(os.getenv("REAL_PIPELINE_OPENALEX_ROUTES", "4")),
    SearchSource.SEMANTIC_SCHOLAR: int(os.getenv("REAL_PIPELINE_S2_ROUTES", "1")),
    SearchSource.ARXIV: int(os.getenv("REAL_PIPELINE_ARXIV_ROUTES", "1")),
    SearchSource.GOOGLE_SEARCH: int(os.getenv("REAL_PIPELINE_GOOGLE_ROUTES", "1")),
}


# =============================================================================
# Query fixtures: 与 Query Generation 真实测试保持同一语义标准
# =============================================================================


def make_complex_analysis() -> QueryAnalysisResult:
    """Year + Soft Venue + 两条 Hard Relation。"""
    return QueryAnalysisResult(
        original_query=(
            "找2023年以来使用强化学习优化LLM学术搜索Agent，并在真实科研查询数据集上验证的论文，"
            "最好来自ACL、NeurIPS。"
        ),
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(
            task=SearchTask.PAPER_SEARCH,
            mode=RetrievalMode.COMPREHENSIVE,
            goal=(
                "Find papers using reinforcement learning to optimize LLM academic "
                "search agents and evaluating them on real research queries."
            ),
        ),
        content=(
            "reinforcement learning optimization of LLM academic search agent "
            "evaluated on real scientific query datasets"
        ),
        constraints=ConstraintSet(
            hard=[
                ConstraintItem(
                    id="C1",
                    type=ConstraintType.YEAR,
                    operator=ConstraintOperator.GTE,
                    value=2023,
                    source_text="2023年以来",
                ),
                ConstraintItem(
                    id="C2",
                    type=ConstraintType.EVALUATION,
                    operator=ConstraintOperator.CONTAINS,
                    value="real scientific query datasets",
                    source_text="在真实科研查询数据集上验证",
                ),
            ],
            soft=[
                ConstraintItem(
                    id="P1",
                    type=ConstraintType.VENUE,
                    operator=ConstraintOperator.ANY_OF,
                    values=["ACL", "NeurIPS"],
                    source_text="最好来自ACL、NeurIPS",
                )
            ],
        ),
        relations=[
            RelationConstraint(
                id="R1",
                subject="reinforcement learning",
                predicate="optimize",
                object="LLM academic search agent",
                level=ConstraintLevel.HARD,
                source_text="使用强化学习优化LLM学术搜索Agent",
            ),
            RelationConstraint(
                id="R2",
                subject="LLM academic search agent",
                predicate="evaluated_on",
                object="real scientific query datasets",
                level=ConstraintLevel.HARD,
                source_text="在真实科研查询数据集上验证",
            ),
        ],
        entities=[
            QueryEntity(
                type=EntityType.METHOD,
                canonical_name="reinforcement learning",
                aliases=["RL"],
                source_text="强化学习",
            ),
            QueryEntity(
                type=EntityType.MODEL,
                canonical_name="LLM academic search agent",
                aliases=[],
                source_text="LLM学术搜索Agent",
            ),
            QueryEntity(
                type=EntityType.DATASET,
                canonical_name="real scientific query datasets",
                aliases=[],
                source_text="真实科研查询数据集",
            ),
        ],
    )


def make_or_exclude_analysis() -> QueryAnalysisResult:
    """显式 OR 配对 + Exclude survey/review。"""
    return QueryAnalysisResult(
        original_query="找 ACL 2024 或 NeurIPS 2023 的多模态大模型论文，不要综述。",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(
            task=SearchTask.PAPER_SEARCH,
            mode=RetrievalMode.COMPREHENSIVE,
            goal=(
                "Find multimodal large language model papers from either ACL 2024 "
                "or NeurIPS 2023, excluding survey/review papers."
            ),
        ),
        content="multimodal large language models",
        constraints=ConstraintSet(
            hard=[
                ConstraintItem(
                    id="C1",
                    type=ConstraintType.VENUE,
                    operator=ConstraintOperator.EQ,
                    value="ACL",
                    source_text="ACL 2024",
                ),
                ConstraintItem(
                    id="C2",
                    type=ConstraintType.YEAR,
                    operator=ConstraintOperator.EQ,
                    value=2024,
                    source_text="ACL 2024",
                ),
                ConstraintItem(
                    id="C3",
                    type=ConstraintType.VENUE,
                    operator=ConstraintOperator.EQ,
                    value="NeurIPS",
                    source_text="NeurIPS 2023",
                ),
                ConstraintItem(
                    id="C4",
                    type=ConstraintType.YEAR,
                    operator=ConstraintOperator.EQ,
                    value=2023,
                    source_text="NeurIPS 2023",
                ),
            ],
            exclude=[
                ConstraintItem(
                    id="E1",
                    type=ConstraintType.PUBLICATION_TYPE,
                    operator=ConstraintOperator.ANY_OF,
                    values=["survey", "review"],
                    source_text="不要综述",
                )
            ],
        ),
        logic=ConstraintLogic(
            hard=LogicNode(
                operator="OR",
                children=[
                    LogicNode(
                        operator="AND",
                        children=[
                            LogicNode(operator="LEAF", ref_id="C1"),
                            LogicNode(operator="LEAF", ref_id="C2"),
                        ],
                    ),
                    LogicNode(
                        operator="AND",
                        children=[
                            LogicNode(operator="LEAF", ref_id="C3"),
                            LogicNode(operator="LEAF", ref_id="C4"),
                        ],
                    ),
                ],
            )
        ),
        entities=[
            QueryEntity(
                type=EntityType.TOPIC,
                canonical_name="multimodal large language models",
                aliases=["MLLM"],
                source_text="多模态大模型",
            )
        ],
    )


def make_broad_forgery_analysis() -> QueryAnalysisResult:
    """Broad Query：主要观察 Expansion 的 Recall 增益。"""
    return QueryAnalysisResult(
        original_query="给我多模态伪造检测相关的研究论文。",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(
            task=SearchTask.PAPER_SEARCH,
            mode=RetrievalMode.COMPREHENSIVE,
            goal="Find research papers on multimodal forgery detection.",
        ),
        content="multimodal forgery detection",
        constraints=ConstraintSet(),
        entities=[
            QueryEntity(
                type=EntityType.TOPIC,
                canonical_name="multimodal forgery detection",
                aliases=[],
                source_text="多模态伪造检测",
            )
        ],
    )


@dataclass(frozen=True)
class PipelineCase:
    name: str
    factory: Callable[[], QueryAnalysisResult]


ALL_CASES = {
    "complex": PipelineCase("Complex Relation", make_complex_analysis),
    "or_exclude": PipelineCase("OR + Exclude", make_or_exclude_analysis),
    "broad": PipelineCase("Broad Multimodal Forgery", make_broad_forgery_analysis),
}


def _selected_cases() -> list[PipelineCase]:
    if CASE_SELECTOR in {"all", "*"}:
        return list(ALL_CASES.values())

    names = [x.strip() for x in CASE_SELECTOR.split(",") if x.strip()]
    unknown = [x for x in names if x not in ALL_CASES]
    if unknown:
        raise ValueError(
            f"Unknown REAL_PIPELINE_CASES={unknown}; valid={list(ALL_CASES)} or all"
        )
    return [ALL_CASES[x] for x in names]


# =============================================================================
# Builders
# =============================================================================


def _build_generator() -> QueryGenerator:
    return QueryGenerator(
        client=build_client(),
        config=GeneratorConfig(
            temperature=0.0,
            use_structured_output=False,
            enable_llm_expansion=True,
            max_transport_retries=0,
            max_total_queries=5,
            near_duplicate_threshold=0.88,
        ),
    )


def _build_adapters(enabled_sources: list[SearchSource]):
    enabled = set(enabled_sources)
    adapters = {}

    if SearchSource.OPENALEX in enabled:
        key = os.getenv("OPENALEX_API_KEY", "").strip()
        if not key:
            pytest.fail(
                "USE_OPENALEX=true，但 OPENALEX_API_KEY 未配置。"
                "请配置 key，或暂时设置 USE_OPENALEX=false。"
            )
        adapters[SearchSource.OPENALEX] = OpenAlexAdapter(
            api_key=key,
            timeout_seconds=25.0,
        )

    if SearchSource.SEMANTIC_SCHOLAR in enabled:
        key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if not key:
            pytest.fail(
                "USE_SEMANTIC_SCHOLAR=true，但 SEMANTIC_SCHOLAR_API_KEY 未配置。"
                "真实 full-pipeline 验收不建议匿名 S2；请配置 key，或关闭该源。"
            )
        adapters[SearchSource.SEMANTIC_SCHOLAR] = SemanticScholarAdapter(
            api_key=key,
            timeout_seconds=25.0,
        )

    if SearchSource.ARXIV in enabled:
        adapters[SearchSource.ARXIV] = ArxivAdapter(timeout_seconds=30.0)

    if SearchSource.GOOGLE_SEARCH in enabled:
        key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
        engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "").strip()
        if not key or not engine_id:
            pytest.fail(
                "USE_GOOGLE_SEARCH=true，但 GOOGLE_SEARCH_API_KEY 或 "
                "GOOGLE_SEARCH_ENGINE_ID 未配置。"
            )
        adapters[SearchSource.GOOGLE_SEARCH] = GoogleSearchAdapter(
            api_key=key,
            engine_id=engine_id,
            timeout_seconds=25.0,
        )

    return adapters


def _build_orchestrator(adapters) -> RetrievalOrchestrator:
    # route-level 并发控制；真实执行计划同时又限制 S2/arXiv 每次 case 的 route 数。
    return RetrievalOrchestrator(
        adapters,
        max_concurrency=min(3, max(1, len(adapters))),
        route_timeout_seconds=ROUTE_TIMEOUT_SECONDS,
        max_attempts=MAX_ATTEMPTS,
        retry_backoff_seconds=RETRY_BACKOFF_SECONDS,
        raise_if_all_failed=True,
    )


# =============================================================================
# Competition-aware route execution budget
# =============================================================================


def _route_key(route: RouteSpec) -> tuple:
    return (
        route.logic_branch_id,
        route.query_id,
        route.source,
        route.strategy,
    )


def _cap_route(route: RouteSpec) -> RouteSpec:
    return route.model_copy(update={"limit": min(route.limit, PER_ROUTE_LIMIT)})


def _make_execution_plan(native_plan: RetrievalPlan) -> RetrievalPlan:
    """从完整 native plan 中选一个受控、覆盖充分的真实执行子集。

    为什么这么做：
    - native plan 完整用于检查 F1 相关语义（OR、Soft、Relation、预算）；
    - 真实 API 不应为了测试无脑执行 10 routes × 28 hits；
    - 用 coverage-greedy 保证优先覆盖 branch / source / query kind，再按 priority 填满。

    这不会改生产 Route Builder；只是 integration test 的“执行预算层”。
    """
    routes = [r for r in native_plan.routes if r.enabled]
    if not routes:
        return native_plan

    max_routes = max(1, min(MAX_EXEC_ROUTES, len(routes)))
    source_counts: Counter[SearchSource] = Counter()
    selected: list[RouteSpec] = []
    selected_keys: set[tuple] = set()

    all_branches = {r.logic_branch_id for r in routes}
    all_sources = {r.source for r in routes}
    all_kinds = {r.query_kind for r in routes}

    covered_branches: set[str | None] = set()
    covered_sources: set[SearchSource] = set()
    covered_kinds: set[QueryKind] = set()

    remaining = sorted(routes, key=lambda r: r.priority, reverse=True)

    while remaining and len(selected) < max_routes:
        best = None
        best_score = None

        for route in remaining:
            cap = SOURCE_ROUTE_CAPS.get(route.source, max_routes)
            if source_counts[route.source] >= cap:
                continue

            gain = 0.0
            if route.logic_branch_id not in covered_branches:
                gain += 4.0
            if route.source not in covered_sources:
                gain += 3.0
            if route.query_kind not in covered_kinds:
                gain += 2.0

            # CORE/EXACT 是 Precision/Recall 的主锚点，稍微优先。
            if route.query_kind in {QueryKind.CORE, QueryKind.EXACT}:
                gain += 0.4
            elif route.query_kind == QueryKind.RELATION:
                gain += 0.3
            elif route.query_kind == QueryKind.EXPANDED:
                gain += 0.2

            score = (gain, route.priority)
            if best_score is None or score > best_score:
                best_score = score
                best = route

        if best is None:
            break

        selected.append(_cap_route(best))
        selected_keys.add(_route_key(best))
        source_counts[best.source] += 1
        covered_branches.add(best.logic_branch_id)
        covered_sources.add(best.source)
        covered_kinds.add(best.query_kind)
        remaining = [r for r in remaining if _route_key(r) != _route_key(best)]

    # 如果 caps 太严格导致没有任何 route，直接暴露配置问题，而不是静默运行空计划。
    if not selected:
        raise AssertionError(
            f"No executable route after source caps: {SOURCE_ROUTE_CAPS}"
        )

    max_total_hits = min(
        native_plan.max_total_hits,
        sum(route.limit for route in selected),
    )

    return RetrievalPlan(
        routes=selected,
        max_routes=max(len(selected), 1),
        max_total_hits=max(max_total_hits, 1),
        warnings=[
            *native_plan.warnings,
            (
                "real-test execution budget applied: "
                f"native_routes={len(native_plan.routes)}, executed_routes={len(selected)}, "
                f"per_route_limit<={PER_ROUTE_LIMIT}"
            ),
        ],
    )


# =============================================================================
# F1-oriented invariants
# =============================================================================


def _assert_generation_safety(
    analysis: QueryAnalysisResult,
    generation: QueryGenerationOutput,
) -> None:
    assert not generation.validation.errors, (
        "Query Generation 出现 Query Drift：\n"
        + "\n".join(
            f"{x.code}: {x.message}" for x in generation.validation.errors
        )
    )

    assert 1 <= len(generation.result.queries) <= 5
    assert generation.metrics.api_calls <= 1, (
        "Query Generation V1 不应超过 1 次 LLM 调用。"
    )
    assert generation.metrics.total_tokens <= MAX_QGEN_TOKENS, (
        f"Query Generation token={generation.metrics.total_tokens} > "
        f"budget={MAX_QGEN_TOKENS}"
    )

    joined = " ".join(q.text.casefold() for q in generation.result.queries)

    # Metadata 不应污染内容 Query；这些应由 Route Builder/filter 处理。
    metadata_terms: list[str] = []
    for c in analysis.constraints.hard + analysis.constraints.soft:
        if c.type in {ConstraintType.VENUE, ConstraintType.AUTHOR}:
            vals = list(c.values) if c.values else ([c.value] if c.value is not None else [])
            metadata_terms.extend(str(v).casefold() for v in vals)
        if c.type == ConstraintType.YEAR:
            vals = list(c.values) if c.values else ([c.value] if c.value is not None else [])
            metadata_terms.extend(str(v) for v in vals)

    for term in metadata_terms:
        if term:
            assert term not in joined, (
                f"metadata term {term!r} leaked into generated content query"
            )

    # Exclude survey/review 不允许反向变成 expansion。
    has_review_exclude = any(
        c.type == ConstraintType.PUBLICATION_TYPE
        and any(
            str(v).casefold() in {"survey", "review"}
            for v in (list(c.values) if c.values else ([c.value] if c.value else []))
        )
        for c in analysis.constraints.exclude
    )
    if has_review_exclude:
        assert "survey" not in joined
        assert "review" not in joined


def _assert_native_plan_semantics(
    analysis: QueryAnalysisResult,
    generation: QueryGenerationOutput,
    plan: RetrievalPlan,
    enabled_sources: list[SearchSource],
) -> None:
    assert plan.routes, "Route Builder produced no route"
    assert len(plan.routes) <= plan.max_routes
    assert plan.max_routes <= 10, "V1 Comprehensive route budget should stay bounded"

    generated_ids = {q.query_id for q in generation.result.queries}
    hard_ids = {c.id for c in analysis.constraints.hard}
    soft_ids = {c.id for c in analysis.constraints.soft}

    for route in plan.routes:
        assert route.source in enabled_sources
        assert route.query_id in generated_ids or route.query_id == "QF1"
        assert set(route.pushed_constraint_ids) <= hard_ids

        # F1 核心：Soft 只排序加分，绝不下推为硬过滤。
        assert not (set(route.pushed_constraint_ids) & soft_ids)

    # 至少有一个 Core/Exact 锚点。
    assert any(
        r.query_kind in {QueryKind.CORE, QueryKind.EXACT}
        for r in plan.routes
    )

    # 有 Relation Query 时，至少 native plan 中有 Relation route。
    generated_kinds = {q.kind for q in generation.result.queries}
    if QueryKind.RELATION in generated_kinds:
        assert any(r.query_kind == QueryKind.RELATION for r in plan.routes)

    # 非笛卡尔积/预算保护：理论全笛卡尔积大于预算时必须被裁剪。
    theoretical = len(generation.result.queries) * len(enabled_sources)
    if theoretical > plan.max_routes:
        assert len(plan.routes) < theoretical

    # OR Pairing：B1=(ACL,2024), B2=(NeurIPS,2023)，不能交叉组合。
    if analysis.original_query.startswith("找 ACL 2024"):
        branch_expected = {
            "B1": {"C1", "C2"},
            "B2": {"C3", "C4"},
        }
        branch_forbidden = {
            "B1": {"C3", "C4"},
            "B2": {"C1", "C2"},
        }
        seen_branches = set()
        for route in plan.routes:
            if route.logic_branch_id not in branch_expected:
                continue
            seen_branches.add(route.logic_branch_id)
            refs = set(route.pushed_constraint_ids) | set(route.deferred_constraint_ids)
            assert branch_expected[route.logic_branch_id] <= refs
            assert not (branch_forbidden[route.logic_branch_id] & refs)
        assert seen_branches == {"B1", "B2"}

    # Complex：Soft Venue P1 不能变 Hard；两条 relation 要 deferred 给后端 Selector。
    if analysis.original_query.startswith("找2023年以来使用强化学习"):
        for route in plan.routes:
            assert "P1" not in route.pushed_constraint_ids
        relation_ids = {"R1", "R2"}
        assert any(
            relation_ids <= set(route.deferred_constraint_ids)
            for route in plan.routes
        )


def _assert_execution_plan_coverage(
    native_plan: RetrievalPlan,
    execution_plan: RetrievalPlan,
    enabled_sources: list[SearchSource],
) -> None:
    assert 1 <= len(execution_plan.routes) <= MAX_EXEC_ROUTES
    assert all(r.limit <= PER_ROUTE_LIMIT for r in execution_plan.routes)

    native_sources = {r.source for r in native_plan.routes}
    exec_sources = {r.source for r in execution_plan.routes}
    expected_source_coverage = native_sources & set(enabled_sources)

    # 如果 budget 容得下，就要求所有启用且 native plan 使用的数据源都被真实执行。
    if len(expected_source_coverage) <= MAX_EXEC_ROUTES:
        assert expected_source_coverage <= exec_sources

    native_branches = {r.logic_branch_id for r in native_plan.routes}
    exec_branches = {r.logic_branch_id for r in execution_plan.routes}
    if len(native_branches) <= MAX_EXEC_ROUTES:
        assert native_branches <= exec_branches

    native_kinds = {r.query_kind for r in native_plan.routes}
    exec_kinds = {r.query_kind for r in execution_plan.routes}
    # CORE 必须执行；Relation/Expanded 在 budget 足够时应尽量真实验证。
    assert QueryKind.CORE in exec_kinds or QueryKind.EXACT in exec_kinds
    if QueryKind.RELATION in native_kinds and MAX_EXEC_ROUTES >= 4:
        assert QueryKind.RELATION in exec_kinds
    if QueryKind.EXPANDED in native_kinds and MAX_EXEC_ROUTES >= 5:
        assert QueryKind.EXPANDED in exec_kinds


def _assert_retrieval_health(
    retrieval: RetrievalOutput,
    execution_plan: RetrievalPlan,
) -> None:
    m = retrieval.metrics
    assert m.planned_routes == len(execution_plan.routes)
    assert m.executed_routes == len(execution_plan.routes)
    assert m.raw_hits == len(retrieval.hits)
    assert m.raw_hits > 0

    # 外部源单路短暂失败可以容错，但至少 2/3 以上路线应成功；
    # 对当前受控执行计划，这比“所有 route 必须 100% 成功”更符合真实工程。
    min_success = max(1, (2 * m.executed_routes + 2) // 3)
    assert m.successful_routes >= min_success, (
        f"retrieval success routes={m.successful_routes}/{m.executed_routes}"
    )

    assert m.api_calls <= MAX_RETRIEVAL_API_CALLS, (
        f"retrieval api_calls={m.api_calls} > budget={MAX_RETRIEVAL_API_CALLS}"
    )
    assert m.raw_hits <= execution_plan.max_total_hits

    # Metrics 必须自洽；重试也要真实计入 api_calls。
    assert m.api_calls == sum(ex.api_calls for ex in retrieval.executions)
    assert m.retries == sum(max(0, ex.attempts - 1) for ex in retrieval.executions)

    executed_route_ids = {r.route_id for r in execution_plan.routes}
    for hit in retrieval.hits:
        assert hit.origin.route_id in executed_route_ids
        assert hit.origin.source == hit.source
        assert hit.origin.rank >= 1
        assert hit.title


def _assert_candidate_pool_health(pool: CandidatePool) -> None:
    assert pool.stats.raw_hits > 0
    assert pool.stats.candidates_before_filter <= pool.stats.raw_hits
    assert pool.stats.candidates_after_filter == len(pool.candidates)
    assert pool.stats.rejected_by_hard_filter == len(pool.rejected)

    # Recall 保护：UNKNOWN 必须留在 pool；只有明确 FAIL 才 rejected。
    assert all(c.hard_filter_status != MatchStatus.FAIL for c in pool.candidates)
    assert all(c.hard_filter_status == MatchStatus.FAIL for c in pool.rejected)

    for c in [*pool.candidates, *pool.rejected]:
        assert c.origins
        assert c.hit_count >= len(c.origins)
        assert c.source_count == len({o.source for o in c.origins})
        assert c.route_count == len({o.route_id for o in c.origins})
        assert c.best_rank == min(o.rank for o in c.origins)

    # RRF/cheap rerank 已执行后，每个保留候选必须有可解释 score components。
    for c in pool.candidates:
        assert c.retrieval_score is not None
        assert {
            "rrf",
            "lexical",
            "source_coverage",
            "route_coverage",
            "soft_match",
            "preference",
        } <= set(c.score_components)


# =============================================================================
# Diagnostics
# =============================================================================


def _query_kind_diagnostics(pool: CandidatePool) -> dict[str, dict[str, int]]:
    """统计各 QueryKind 对 Recall 的实际贡献。

    marginal_non_core = 没有任何 CORE/EXACT origin、但由该 kind 找到的 Candidate 数。
    这个指标比“某 kind hits > 0”更能看出 expansion 是否真的带来新增 Recall。
    """
    result: dict[str, dict[str, int]] = {}
    for kind in QueryKind:
        candidates_with_kind = 0
        marginal_non_core = 0
        origin_count = 0
        for c in pool.candidates:
            kinds = {o.query_kind for o in c.origins}
            n = sum(o.query_kind == kind for o in c.origins)
            origin_count += n
            if kind in kinds:
                candidates_with_kind += 1
                if (
                    kind not in {QueryKind.CORE, QueryKind.EXACT}
                    and QueryKind.CORE not in kinds
                    and QueryKind.EXACT not in kinds
                ):
                    marginal_non_core += 1
        result[kind.value] = {
            "origin_count": origin_count,
            "candidate_count": candidates_with_kind,
            "marginal_non_core": marginal_non_core,
        }
    return result


def _source_diagnostics(pool: CandidatePool) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for c in pool.candidates:
        for source in {o.source.value for o in c.origins}:
            counts[source] += 1
    return dict(counts)


def _print_generation(case_name: str, g: QueryGenerationOutput) -> None:
    print(f"\n========== {case_name}: QUERY GENERATION ==========")
    print(f"llm_api_calls      = {g.metrics.api_calls}")
    print(f"prompt_tokens      = {g.metrics.prompt_tokens}")
    print(f"completion_tokens  = {g.metrics.completion_tokens}")
    print(f"total_tokens       = {g.metrics.total_tokens}")
    print(f"llm_latency_ms     = {g.metrics.llm_latency_ms:.2f}")
    print(f"generation_ms      = {g.metrics.latency_ms:.2f}")
    print(f"query_count        = {len(g.result.queries)}")
    for q in g.result.queries:
        print(
            f"[{q.query_id}] kind={q.kind.value:<9} "
            f"priority={q.priority:.2f} text={q.text}"
        )


def _print_plan(label: str, plan: RetrievalPlan) -> None:
    print(f"\n---------- {label} ----------")
    print(f"routes={len(plan.routes)} max_routes={plan.max_routes} max_total_hits={plan.max_total_hits}")
    for r in plan.routes:
        f = r.filters
        print(
            f"[{r.route_id}] {r.source.value:<18} {r.query_kind.value:<9} "
            f"q={r.query_id:<4} branch={r.logic_branch_id} limit={r.limit:<3} "
            f"priority={r.priority:.3f}"
        )
        print(
            "    filters="
            f"year=({f.year_start},{f.year_end}) venues={f.venues} authors={f.authors} "
            f"types={f.publication_types} min_citations={f.min_citations}"
        )
        print(
            f"    pushed={r.pushed_constraint_ids} deferred={r.deferred_constraint_ids}"
        )


def _print_retrieval(retrieval: RetrievalOutput) -> None:
    m = retrieval.metrics
    print("\n========== REAL MULTI-ROUTE RETRIEVAL ==========")
    print(f"planned_routes      = {m.planned_routes}")
    print(f"executed_routes     = {m.executed_routes}")
    print(f"successful_routes   = {m.successful_routes}")
    print(f"empty_routes        = {m.empty_routes}")
    print(f"failed_routes       = {m.failed_routes}")
    print(f"timeout_routes      = {m.timeout_routes}")
    print(f"api_calls           = {m.api_calls}")
    print(f"retries             = {m.retries}")
    print(f"raw_hits            = {m.raw_hits}")
    print(f"source_hit_counts   = {m.source_hit_counts}")
    print(f"source_api_calls    = {m.source_api_calls}")
    print(f"source_latencies_ms = {m.source_latencies_ms}")
    print(f"retrieval_ms        = {m.total_latency_ms:.2f}")

    print("\n---------- ROUTE EXECUTIONS ----------")
    for ex in retrieval.executions:
        print(
            f"[{ex.route_id}] source={ex.source.value:<18} "
            f"status={ex.status.value:<8} hits={len(ex.hits):<3} "
            f"api_calls={ex.api_calls:<2} attempts={ex.attempts:<2} "
            f"latency_ms={ex.latency_ms:.2f}"
        )
        if ex.error_type or ex.error_message:
            print(f"    error={ex.error_type}: {ex.error_message}")


def _print_pool(pool: CandidatePool, top_n: int) -> None:
    qdiag = _query_kind_diagnostics(pool)
    sdiag = _source_diagnostics(pool)

    print("\n========== CANDIDATE POOL / F1 PROXIES ==========")
    print(f"raw_hits                  = {pool.stats.raw_hits}")
    print(f"dedup_clusters            = {pool.stats.dedup_clusters}")
    print(f"candidates_before_filter  = {pool.stats.candidates_before_filter}")
    print(f"rejected_by_hard_filter   = {pool.stats.rejected_by_hard_filter}")
    print(f"unknown_after_filter      = {pool.stats.unknown_after_filter}")
    print(f"candidates_after_filter   = {pool.stats.candidates_after_filter}")
    print(f"query_kind_contribution   = {json.dumps(qdiag, ensure_ascii=False)}")
    print(f"source_candidate_coverage = {json.dumps(sdiag, ensure_ascii=False)}")
    print(
        "f1_note                    = integration test has no gold labels; "
        "these are recall/precision proxies, not true F1"
    )

    print(f"\n---------- TOP {min(top_n, len(pool.candidates))} ----------")
    for idx, c in enumerate(pool.top(top_n), start=1):
        title = (c.title or "<NO TITLE>").replace("\n", " ").strip()
        kinds = sorted({o.query_kind.value for o in c.origins})
        sources = sorted({o.source.value for o in c.origins})
        print(
            f"[{idx}] score={c.retrieval_score:.4f} "
            f"status={c.hard_filter_status.value if c.hard_filter_status else None} "
            f"sources={sources} kinds={kinds} best_rank={c.best_rank}\n"
            f"    {title}\n"
            f"    components={c.score_components}"
        )
        for o in sorted(c.origins, key=lambda x: (x.rank, x.source.value)):
            print(
                f"    origin={o.source.value}/{o.query_id}/{o.query_kind.value}/"
                f"rank={o.rank}/route={o.route_id}/branch={o.logic_branch_id}"
            )


def _print_efficiency_summary(
    generation: QueryGenerationOutput,
    retrieval: RetrievalOutput,
    pool: CandidatePool,
    total_latency_ms: float,
) -> None:
    unique = max(1, pool.stats.candidates_before_filter)
    total_api_calls = generation.metrics.api_calls + retrieval.metrics.api_calls

    summary = {
        "llm_api_calls": generation.metrics.api_calls,
        "retrieval_api_calls": retrieval.metrics.api_calls,
        "total_api_calls": total_api_calls,
        "llm_tokens": generation.metrics.total_tokens,
        "generation_latency_ms": round(generation.metrics.latency_ms, 2),
        "retrieval_latency_ms": round(retrieval.metrics.total_latency_ms, 2),
        "pipeline_latency_ms": round(total_latency_ms, 2),
        "raw_hits": retrieval.metrics.raw_hits,
        "unique_candidates": pool.stats.candidates_before_filter,
        "api_calls_per_unique_candidate": round(total_api_calls / unique, 4),
        "ms_per_unique_candidate": round(total_latency_ms / unique, 2),
        "selector_input_upper_bound": min(TOP_N, len(pool.candidates)),
    }
    print("\n========== COMPETITION EFFICIENCY SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# =============================================================================
# Test runner
# =============================================================================


async def _run_case(case: PipelineCase):
    started = time.perf_counter()

    analysis = case.factory()
    enabled_sources = get_enabled_sources()
    assert enabled_sources, "At least one retrieval source must be enabled"

    # 1) Real Query Generation
    generator = _build_generator()
    generation = await generator.generate(analysis)
    _assert_generation_safety(analysis, generation)
    _print_generation(case.name, generation)

    # 2) Native Route Builder plan: 完整语义验收
    builder = RetrievalRouteBuilder(enabled_sources=enabled_sources)
    native_plan = builder.build(analysis, generation)
    _assert_native_plan_semantics(
        analysis,
        generation,
        native_plan,
        enabled_sources,
    )
    _print_plan("NATIVE ROUTE PLAN", native_plan)

    # 3) Competition-aware execution plan: 控制真实 API 成本
    execution_plan = _make_execution_plan(native_plan)
    _assert_execution_plan_coverage(native_plan, execution_plan, enabled_sources)
    _print_plan("REAL EXECUTION PLAN", execution_plan)

    # 4) Real multi-source/multi-route retrieval
    adapters = _build_adapters(enabled_sources)
    orchestrator = _build_orchestrator(adapters)
    retrieval = await orchestrator.run(execution_plan)
    _assert_retrieval_health(retrieval, execution_plan)
    _print_retrieval(retrieval)

    # 5) Candidate Pool: Dedup/Fusion -> Hard Filter -> Weighted RRF/Cheap Rerank
    pool = CandidatePool.from_hits(retrieval.hits)
    pool.apply_hard_filters(analysis)
    pool.prerank(
        analysis,
        generation,
        num_enabled_sources=len(enabled_sources),
        num_planned_routes=len(execution_plan.routes),
    )
    _assert_candidate_pool_health(pool)
    _print_pool(pool, TOP_N)

    total_latency_ms = (time.perf_counter() - started) * 1000.0
    _print_efficiency_summary(generation, retrieval, pool, total_latency_ms)

    if MAX_LATENCY_MS > 0:
        assert total_latency_ms <= MAX_LATENCY_MS, (
            f"pipeline latency={total_latency_ms:.2f}ms > budget={MAX_LATENCY_MS:.2f}ms"
        )

    return analysis, generation, native_plan, execution_plan, retrieval, pool


@pytest.mark.skipif(
    not RUN_REAL,
    reason=(
        "Set RUN_REAL_RETRIEVAL_PIPELINE_TESTS=1 to run real multi-route pipeline tests."
    ),
)
@pytest.mark.parametrize(
    "case",
    _selected_cases(),
    ids=lambda c: c.name.replace(" ", "_"),
)
def test_real_multi_query_multi_route_pipeline(case: PipelineCase):
    """Retrieval + Candidate Pool V1.0 冻结前主验收。"""
    analysis, generation, native_plan, execution_plan, retrieval, pool = asyncio.run(
        _run_case(case)
    )

    # 至少真正执行了“多 Route”；Complex case 在默认 budget 下还必须是“多 Query”。
    assert len(execution_plan.routes) >= min(2, len(native_plan.routes))
    executed_query_ids = {r.query_id for r in execution_plan.routes}
    if case.name == "Complex Relation" and len(generation.result.queries) >= 2:
        assert len(executed_query_ids) >= 2

    # Top-N 必须是 retrieval_score 单调不增，避免 Selector 接收到未排序集合。
    top = pool.top(TOP_N)
    scores = [c.retrieval_score for c in top]
    assert scores == sorted(scores, reverse=True)

    print(
        f"\n[PASS] {case.name}: Query Generation -> Route Builder -> "
        "OpenAlex/S2/arXiv -> Candidate Pool -> Hard Filter -> RRF is healthy."
    )
