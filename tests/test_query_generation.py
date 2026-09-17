from __future__ import annotations

import asyncio
import json

from scholar_agent_v2.query_generation.dedup import deduplicate_queries
from scholar_agent_v2.query_generation.generator import GeneratorConfig, QueryGenerator
from scholar_agent_v2.query_generation.models import GeneratedQuery, QueryKind
from scholar_agent_v2.query_generation.prompt import (
    QUERY_EXPANSION_SYSTEM_PROMPT,
    build_expansion_user_prompt,
)
from scholar_agent_v2.query_understanding.analyzer import LLMResponse, LLMUsage
from scholar_agent_v2.query_understanding.schema import (
    ConstraintItem,
    ConstraintLevel,
    ConstraintLogic,
    ConstraintOperator,
    ConstraintSet,
    ConstraintType,
    LogicNode,
    QueryAnalysisResult,
    QueryEntity,
    QueryIntent,
    QueryType,
    RelationConstraint,
    RetrievalMode,
    SearchTask,
    EntityType,
)


# ============================================================
# Fake LLM
# ============================================================

class FakeExpansionLLM:
    """用于 Query Generation 测试；不访问真实 API。"""

    def __init__(self, expanded_queries=None, *, error: Exception | None = None):
        self.expanded_queries = expanded_queries or []
        self.error = error
        self.calls = 0

    async def complete(
        self,
        *,
        messages,
        temperature: float = 0.0,
        response_schema=None,
    ) -> LLMResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error

        payload = {
            "expanded_queries": [
                {"query": query, "reason": "fake test expansion"}
                for query in self.expanded_queries
            ]
        }
        return LLMResponse(
            content=payload,
            usage=LLMUsage(
                prompt_tokens=100,
                completion_tokens=40,
                total_tokens=140,
            ),
            model="fake-query-expander",
        )


def run_generate(generator: QueryGenerator, analysis: QueryAnalysisResult):
    return asyncio.run(generator.generate(analysis))


# ============================================================
# QueryAnalysisResult Fixtures
# ============================================================


def make_complex_analysis() -> QueryAnalysisResult:
    """复杂查询：Year + Soft Venue + 两条 Relation + Alias。"""
    return QueryAnalysisResult(
        original_query=(
            "找2023年以来使用强化学习优化LLM学术搜索Agent，并在真实科研查询数据集上验证的论文，"
            "最好来自ACL、NeurIPS。"
        ),
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(
            task=SearchTask.PAPER_SEARCH,
            mode=RetrievalMode.COMPREHENSIVE,
            goal="Find papers using reinforcement learning to optimize LLM academic search agents and evaluating them on real research queries.",
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
                    value=None,
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


def make_doi_analysis() -> QueryAnalysisResult:
    return QueryAnalysisResult(
        original_query="帮我找 DOI 10.1109/TETCI.2026.3654456 这篇论文",
        query_type=QueryType.SPECIFIC_BY_IDENTIFIER,
        intent=QueryIntent(
            task=SearchTask.PAPER_LOOKUP,
            mode=RetrievalMode.SPECIFIC,
            goal="Locate the paper by DOI.",
        ),
        content="",
        constraints=ConstraintSet(
            hard=[
                ConstraintItem(
                    id="C1",
                    type=ConstraintType.IDENTIFIER,
                    operator=ConstraintOperator.EQ,
                    value="10.1109/TETCI.2026.3654456",
                    source_text="10.1109/TETCI.2026.3654456",
                )
            ]
        ),
    )


def make_or_analysis() -> QueryAnalysisResult:
    return QueryAnalysisResult(
        original_query="找 ACL 2024 或 NeurIPS 2023 的多模态大模型论文",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(
            task=SearchTask.PAPER_SEARCH,
            mode=RetrievalMode.COMPREHENSIVE,
            goal="Find multimodal large language model papers in either ACL 2024 or NeurIPS 2023.",
        ),
        content="multimodal large language models",
        constraints=ConstraintSet(
            hard=[
                ConstraintItem(id="C1", type=ConstraintType.VENUE, value="ACL", source_text="ACL 2024"),
                ConstraintItem(id="C2", type=ConstraintType.YEAR, value=2024, source_text="ACL 2024"),
                ConstraintItem(id="C3", type=ConstraintType.VENUE, value="NeurIPS", source_text="NeurIPS 2023"),
                ConstraintItem(id="C4", type=ConstraintType.YEAR, value=2023, source_text="NeurIPS 2023"),
            ]
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


def make_exclude_survey_analysis() -> QueryAnalysisResult:
    return QueryAnalysisResult(
        original_query="找多模态大模型相关论文，不要综述",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(
            task=SearchTask.PAPER_SEARCH,
            mode=RetrievalMode.COMPREHENSIVE,
            goal="Find primary research papers on multimodal large language models, excluding surveys.",
        ),
        content="multimodal large language models",
        constraints=ConstraintSet(
            exclude=[
                ConstraintItem(
                    id="E1",
                    type=ConstraintType.PUBLICATION_TYPE,
                    operator=ConstraintOperator.ANY_OF,
                    value=None,
                    values=["survey", "review"],
                    source_text="不要综述",
                )
            ]
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


# ============================================================
# 1. Local deterministic generation
# ============================================================


def test_complex_local_generation_without_llm():
    analysis = make_complex_analysis()
    generator = QueryGenerator(
        client=None,
        config=GeneratorConfig(enable_llm_expansion=False, max_total_queries=5),
    )

    output = run_generate(generator, analysis)
    kinds = {q.kind for q in output.result.queries}

    assert output.metrics.api_calls == 0
    assert output.result.llm_used is False
    assert output.result.degraded is False
    assert len(output.result.queries) <= 5
    assert QueryKind.CORE in kinds
    assert QueryKind.RELATION in kinds
    assert all(q.text.strip() for q in output.result.queries)


# ============================================================
# 2. One-shot LLM expansion + metrics
# ============================================================


def test_llm_expansion_calls_at_most_once_and_tracks_metrics():
    analysis = make_complex_analysis()
    client = FakeExpansionLLM(
        [
            "reinforcement learning scholarly literature search agents",
            "RL scientific paper retrieval agents",
        ]
    )
    generator = QueryGenerator(client=client)

    output = run_generate(generator, analysis)

    assert client.calls == 1
    assert output.metrics.api_calls == 1
    assert output.result.llm_used is True
    assert output.metrics.total_tokens == 140
    assert output.metrics.model_names == ["fake-query-expander"]
    assert len(output.result.queries) <= 5
    assert output.result.plan.budget.alias == 0
    assert output.result.plan.budget.relation == 2
    assert output.result.plan.budget.expanded == 2


# ============================================================
# 3. Query Drift: invented year must be removed
# ============================================================


def test_invented_year_is_filtered():
    analysis = make_complex_analysis()
    client = FakeExpansionLLM(
        ["2025 reinforcement learning academic search agent"]
    )
    generator = QueryGenerator(client=client)

    output = run_generate(generator, analysis)

    assert all("2025" not in q.text for q in output.result.queries)
    assert any(issue.code == "INVENTED_YEAR" for issue in output.validation.issues)


def test_supported_hard_year_stays_in_downstream_filter_not_expansion():
    analysis = make_complex_analysis()
    client = FakeExpansionLLM(
        ["2023 reinforcement learning academic search agent"]
    )
    output = run_generate(QueryGenerator(client=client), analysis)

    assert all("2023" not in q.text for q in output.result.queries)
    assert any(
        issue.code == "HARD_METADATA_EMBEDDED"
        for issue in output.validation.issues
    )


def test_expansion_may_omit_a_semantic_hard_constraint_for_recall():
    analysis = make_complex_analysis()
    client = FakeExpansionLLM(
        ["reinforcement learning for scholarly search agents"]
    )
    output = run_generate(QueryGenerator(client=client), analysis)

    assert any(
        q.kind == QueryKind.EXPANDED
        and q.text == "reinforcement learning for scholarly search agents"
        for q in output.result.queries
    )


# ============================================================
# 4. Query Drift: survey/review must not be invented
# ============================================================


def test_unrequested_review_intent_is_filtered():
    analysis = make_complex_analysis()
    client = FakeExpansionLLM(
        ["survey of reinforcement learning academic search agents"]
    )
    generator = QueryGenerator(client=client)

    output = run_generate(generator, analysis)

    assert all("survey" not in q.text.lower() for q in output.result.queries)
    assert any(issue.code == "INVENTED_REVIEW_INTENT" for issue in output.validation.issues)


# ============================================================
# 5. Soft metadata cannot become a hard search term
# ============================================================


def test_soft_venue_is_not_hardened_into_query_text():
    analysis = make_complex_analysis()
    client = FakeExpansionLLM(
        ["ACL reinforcement learning academic search agent"]
    )
    generator = QueryGenerator(client=client)

    output = run_generate(generator, analysis)

    assert all("acl" not in q.text.lower().split() for q in output.result.queries)
    assert any(issue.code == "SOFT_METADATA_HARDENED" for issue in output.validation.issues)


# ============================================================
# 6. DOI direct lookup: no expansion, no LLM
# ============================================================


def test_doi_query_uses_exact_lookup_and_zero_llm_calls():
    analysis = make_doi_analysis()
    client = FakeExpansionLLM(["should never be called"])
    generator = QueryGenerator(client=client)

    output = run_generate(generator, analysis)

    assert client.calls == 0
    assert output.metrics.api_calls == 0
    assert len(output.result.queries) == 1
    assert output.result.queries[0].kind == QueryKind.EXACT
    assert output.result.queries[0].text == "10.1109/TETCI.2026.3654456"


# ============================================================
# 7. OR metadata stays in logic/filter layer, not content query
# ============================================================


def test_or_metadata_pairing_is_not_flattened_into_content_query():
    analysis = make_or_analysis()
    generator = QueryGenerator(
        client=None,
        config=GeneratorConfig(enable_llm_expansion=False),
    )

    output = run_generate(generator, analysis)
    joined = " ".join(q.text.lower() for q in output.result.queries)

    assert "acl" not in joined
    assert "neurips" not in joined
    assert "2024" not in joined
    assert "2023" not in joined
    assert any(q.kind == QueryKind.CORE for q in output.result.queries)


# ============================================================
# 8. LLM failure degrades gracefully instead of killing retrieval
# ============================================================


def test_llm_failure_degrades_to_local_queries():
    analysis = make_complex_analysis()
    client = FakeExpansionLLM(error=TimeoutError("fake timeout"))
    generator = QueryGenerator(client=client)

    output = run_generate(generator, analysis)

    assert client.calls == 1
    assert output.metrics.api_calls == 1
    assert output.result.degraded is True
    assert output.result.queries
    assert any(q.kind in {QueryKind.CORE, QueryKind.RELATION} for q in output.result.queries)


def test_expansion_prompt_treats_hard_constraints_as_selection_boundaries():
    analysis = make_complex_analysis()
    prompt = build_expansion_user_prompt(
        analysis,
        existing_queries=[analysis.content],
        max_queries=2,
    )
    json_text = prompt.split("\n\n", 1)[1].rsplit("\n\nReturn exactly:", 1)[0]
    payload = json.loads(json_text)

    assert "recall probe" in QUERY_EXPANSION_SYSTEM_PROMPT
    assert "do NOT pack every requirement into every query" in (
        QUERY_EXPANSION_SYSTEM_PROMPT
    )
    constraints = payload["constraints"]
    assert "final_selection_requirements" in constraints
    assert "hard" not in constraints
    assert "original_query" not in payload
    assert "goal" not in payload["intent"]
    assert all(
        "source_text" not in item and "id" not in item
        for bucket in constraints.values()
        for item in bucket
    )


# ============================================================
# 9. Dedup: keep CORE over a duplicate EXPANDED query
# ============================================================


def test_dedup_prefers_core_over_expanded_duplicate():
    queries = [
        GeneratedQuery(
            query_id="QC1",
            text="reinforcement learning academic search agent",
            kind=QueryKind.CORE,
            priority=0.95,
        ),
        GeneratedQuery(
            query_id="QE1",
            text="reinforcement learning academic search agent",
            kind=QueryKind.EXPANDED,
            priority=0.72,
        ),
    ]

    deduped = deduplicate_queries(queries)

    assert len(deduped) == 1
    assert deduped[0].kind == QueryKind.CORE
    assert deduped[0].query_id == "QC1"


# ============================================================
# 10. Specification test: excluded survey must NEVER become positive expansion
#     If this fails, validator's review-intent detection is treating EXCLUDE as positive intent.
# ============================================================


def test_excluded_review_does_not_authorize_review_expansion():
    analysis = make_exclude_survey_analysis()
    client = FakeExpansionLLM(
        ["systematic review MLLM architectures"]
    )
    generator = QueryGenerator(client=client)

    output = run_generate(generator, analysis)

    assert all(
        "survey" not in q.text.lower() and "review" not in q.text.lower()
        for q in output.result.queries
    ), (
        "用户明确说‘不要综述’，survey/review expansion 必须被过滤。"
    )


# ============================================================
# 11. Specification test: broad method must not be narrowed to invented submethod
#     If this fails, Query Drift validator still lacks named-method invention detection.
# ============================================================


def test_broad_method_is_not_narrowed_to_invented_submethod():
    analysis = make_complex_analysis()
    client = FakeExpansionLLM(
        ["PPO reinforcement learning academic search agent"]
    )
    generator = QueryGenerator(client=client)

    output = run_generate(generator, analysis)

    assert all("ppo" not in q.text.lower().split() for q in output.result.queries), (
        "原查询只有 reinforcement learning，不能由扩展模型擅自缩窄成 PPO。"
    )
