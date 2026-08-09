# tests/test_query_generation_real.py
"""
Query Generation（查询生成）真实 LLM 小规模语义验收。

特点：
- 默认跳过，不消耗真实 API；
- 设置 RUN_REAL_LLM_TESTS=1 后启用；
- 只测试 Query Generation，不重复调用 Query Understanding；
- 每个测试最多 1 次真实 LLM 调用；
- 重点检查 Query Drift、查询数量、Token、延迟和结构安全。

运行：
    $env:RUN_REAL_LLM_TESTS="1"
    python -m pytest -s tests/test_query_generation_real.py
"""

from __future__ import annotations

import asyncio
import os
import re

import pytest

from scholar_agent_v2.query_generation.generator import GeneratorConfig, QueryGenerator
from scholar_agent_v2.query_generation.models import QueryKind
from scholar_agent_v2.query_understanding.schema import (
    ConstraintItem,
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
    RetrievalMode,
    SearchTask,
)
from tests.real_llm_client import build_client
from tests.test_query_generation import make_complex_analysis


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1",
    reason=(
        "默认跳过真实 LLM 测试。"
        "设置 RUN_REAL_LLM_TESTS=1 后启用。"
    ),
)


# ============================================================
# Helpers
# ============================================================


def run_real_generation(analysis: QueryAnalysisResult):
    generator = QueryGenerator(
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
    return asyncio.run(generator.generate(analysis))


def print_output(case_name: str, output) -> None:
    print(f"\n========== REAL QUERY GENERATION: {case_name} ==========")
    print(f"api_calls          = {output.metrics.api_calls}")
    print(f"prompt_tokens      = {output.metrics.prompt_tokens}")
    print(f"completion_tokens  = {output.metrics.completion_tokens}")
    print(f"total_tokens       = {output.metrics.total_tokens}")
    print(f"llm_latency_ms     = {output.metrics.llm_latency_ms:.2f}")
    print(f"total_latency_ms   = {output.metrics.latency_ms:.2f}")
    print(f"models             = {output.metrics.model_names}")
    print(f"llm_used           = {output.result.llm_used}")
    print(f"degraded           = {output.result.degraded}")
    print(f"query_count        = {len(output.result.queries)}")

    print("\n---------- QUERIES ----------")
    for query in output.result.queries:
        print(
            f"[{query.query_id}] "
            f"kind={query.kind.value:<9} "
            f"priority={query.priority:.2f} "
            f"text={query.text}"
        )
        if query.reason:
            print(f"    reason={query.reason}")
        if query.warnings:
            print(f"    warnings={query.warnings}")

    print("\n---------- VALIDATION ----------")
    if not output.validation.issues:
        print("OK")
    else:
        for issue in output.validation.issues:
            print(
                f"{issue.severity.upper()} "
                f"{issue.code} "
                f"query_id={issue.query_id}: "
                f"{issue.message}"
            )


def final_query_text(output) -> str:
    return " ".join(q.text.lower() for q in output.result.queries)


def assert_common_acceptance(output) -> None:
    """所有真实 Query Generation Case 的公共验收条件。"""
    assert output.metrics.api_calls == 1, "V1 Query Expansion 最多且应只调用 1 次 LLM。"
    assert output.result.llm_used is True
    assert output.result.degraded is False

    assert 1 <= len(output.result.queries) <= 5
    assert all(q.text.strip() for q in output.result.queries)

    # 真实模型自身不应产生被 Drift Validator 判为 ERROR 的扩展。
    # 如果这里失败，说明虽然系统可能把坏 Query 过滤掉了，但 Prompt 仍需优化。
    assert not output.validation.errors, (
        "真实 LLM 产生了 Query Drift / 违反约束的 Query：\n"
        + "\n".join(
            f"{x.code}: {x.message}"
            for x in output.validation.errors
        )
    )


# ============================================================
# Fixtures
# ============================================================


def make_or_exclude_analysis() -> QueryAnalysisResult:
    """ACL 2024 OR NeurIPS 2023 + 排除综述。"""
    return QueryAnalysisResult(
        original_query="找 ACL 2024 或 NeurIPS 2023 的多模态大模型论文，不要综述。",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(
            task=SearchTask.PAPER_SEARCH,
            mode=RetrievalMode.COMPREHENSIVE,
            goal=(
                "Find multimodal large language model papers from either "
                "ACL 2024 or NeurIPS 2023, excluding survey/review papers."
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
                    value=None,
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


def make_broad_multimodal_forgery_analysis() -> QueryAnalysisResult:
    """普通 Broad Query：测试术语扩展能力。"""
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


# ============================================================
# 1. Complex Relation
# ============================================================


def test_real_llm_complex_relation_expansion():
    analysis = make_complex_analysis()
    output = run_real_generation(analysis)
    print_output("Complex Relation", output)

    assert_common_acceptance(output)

    kinds = {q.kind for q in output.result.queries}
    assert QueryKind.CORE in kinds
    assert QueryKind.RELATION in kinds
    assert QueryKind.EXPANDED in kinds, (
        "真实 LLM 调用了但没有留下有效 EXPANDED Query；"
        "说明扩展可能与本地 Query 高度重复或被 Validator 拦截。"
    )

    joined = final_query_text(output)

    # Soft venue 不能偷偷进入内容 Query。
    assert "acl" not in joined
    assert "neurips" not in joined

    # 年份由后续 Route Builder/API Filter 处理，不污染内容 Query。
    assert "2023" not in joined

    # 用户没有要求综述。
    assert "survey" not in joined
    assert "review" not in joined

    # broad method 不能擅自收窄成具体 RL 算法。
    tokens = set(re.findall(r"[a-zA-Z0-9\-]+", joined))
    assert not ({"ppo", "grpo", "dqn"} & tokens)

    # 两个核心关系主题必须还在最终路线集合里。
    assert "reinforcement" in joined
    assert any(term in joined for term in ("academic", "scholarly", "literature"))
    assert any(term in joined for term in ("query", "queries", "dataset", "evaluation"))


# ============================================================
# 2. OR + Exclude
# ============================================================


def test_real_llm_or_exclude_expansion():
    analysis = make_or_exclude_analysis()
    output = run_real_generation(analysis)
    print_output("OR + Exclude", output)

    assert_common_acceptance(output)

    joined = final_query_text(output)

    # Query Generation 只生成内容检索表达；ACL/NeurIPS/年份留给 Route Builder。
    assert "acl" not in joined
    assert "neurips" not in joined
    assert "2024" not in joined
    assert "2023" not in joined

    # “不要综述”绝不能被反向扩成 survey/review Route。
    assert "survey" not in joined
    assert "review" not in joined

    assert "multimodal" in joined
    assert any(
        term in joined
        for term in ("large language model", "language models", "mllm")
    )


# ============================================================
# 3. Broad Query
# ============================================================


def test_real_llm_broad_multimodal_forgery_expansion():
    analysis = make_broad_multimodal_forgery_analysis()
    output = run_real_generation(analysis)
    print_output("Broad Multimodal Forgery", output)

    assert_common_acceptance(output)

    kinds = {q.kind for q in output.result.queries}
    assert QueryKind.CORE in kinds
    assert QueryKind.EXPANDED in kinds

    joined = final_query_text(output)

    # 无时间要求，不允许擅自生成年份。
    assert not re.search(r"\b(?:19|20)\d{2}\b", joined)

    # 用户没有要求综述。
    assert "survey" not in joined
    assert "review" not in joined

    # 必须围绕 multimodal + forgery/deepfake/manipulation detection 语义。
    assert "multimodal" in joined
    assert any(
        term in joined
        for term in ("forgery", "deepfake", "manipulation", "tampering", "fake")
    )
    assert any(
        term in joined
        for term in ("detection", "detect", "forensics", "verification")
    )
