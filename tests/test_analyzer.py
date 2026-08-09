# tests/test_analyzer.py
"""
Query Understanding（查询理解）模块测试。

测试分两部分：

A. 单元/工程测试（默认执行，不消耗真实 LLM API）
   - 主分析 1 次调用成功
   - Hard / Soft / Exclude 结构
   - Relation（关系约束）
   - AND / OR 组合逻辑
   - JSON 容错
   - Validator 失败后 Repair
   - API / Token 统计
   - original_query 安全恢复

B. 真实 LLM 语义测试（默认跳过）
   - 当你实现 tests/real_llm_client.py 后：
       RUN_REAL_LLM_TESTS=1 python -m pytest -s tests/test_analyzer.py
"""

from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from typing import Any, Mapping, Sequence

import pytest

from scholar_agent_v2.query_understanding.analyzer import (
    AnalyzerConfig,
    LLMResponse,
    LLMUsage,
    QueryAnalyzer,
    parse_json_payload,
)
from scholar_agent_v2.query_understanding.schema import (
    ConstraintOperator,
    ConstraintType,
    QueryType,
)


# ============================================================
# 1. Fake LLM Client（假模型客户端）
#    用于测试整个工程链路，不花 API。
# ============================================================

class FakeLLMClient:
    def __init__(
        self,
        responses: list[str | Mapping[str, Any]],
        *,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        model: str = "fake-model",
    ) -> None:
        self.responses = list(responses)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.model = model

        self.calls = 0
        self.received_messages: list[Sequence[Mapping[str, str]]] = []
        self.received_schemas: list[Mapping[str, Any] | None] = []

    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:

        self.calls += 1
        self.received_messages.append(messages)
        self.received_schemas.append(response_schema)

        if not self.responses:
            raise RuntimeError("FakeLLMClient 没有剩余测试响应。")

        content = self.responses.pop(0)

        return LLMResponse(
            content=content,
            usage=LLMUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            ),
            model=self.model,
        )


# ============================================================
# 2. 构造标准测试 JSON
# ============================================================

COMPLEX_QUERY = (
    "找2023年以来使用强化学习优化LLM学术搜索Agent，"
    "并在真实科研查询数据集上验证的论文，"
    "最好来自ACL、NeurIPS等会议。"
)


def make_base_payload(
    original_query: str = COMPLEX_QUERY,
) -> dict[str, Any]:
    """
    一份符合我们当前 QueryAnalysisResult Schema 的标准输出。
    后续测试基于它做局部修改。
    """
    return {
        "schema_version": "1.0",
        "status": "success",
        "original_query": original_query,

        "query_type": "BROAD_BY_DESCRIPTION",

        "intent": {
            "task": "paper_search",
            "mode": "comprehensive",
            "goal": "寻找使用强化学习优化LLM学术搜索Agent并进行真实科研查询验证的论文",
        },

        "content": (
            "reinforcement learning used to optimize "
            "LLM-based academic literature search agents "
            "evaluated on real scientific queries"
        ),

        "constraints": {
            "hard": [
                {
                    "id": "C1",
                    "type": "year",
                    "operator": "gte",
                    "value": 2023,
                    "values": [],
                    "source_text": "2023年以来",
                    "normalized_text": None,
                },
                {
                    "id": "C2",
                    "type": "technology",
                    "operator": "eq",
                    "value": "LLM academic search agent",
                    "values": [],
                    "source_text": "LLM学术搜索Agent",
                    "normalized_text": "LLM-based academic literature search agent",
                },
                {
                    "id": "C3",
                    "type": "evaluation",
                    "operator": "eq",
                    "value": "real scientific research query dataset",
                    "values": [],
                    "source_text": "真实科研查询数据集上验证",
                    "normalized_text": None,
                },
            ],

            "soft": [
                {
                    "id": "P1",
                    "type": "venue",
                    "operator": "any_of",
                    "value": None,
                    "values": ["ACL", "NeurIPS"],
                    "source_text": "最好来自ACL、NeurIPS等会议",
                    "normalized_text": None,
                }
            ],

            "exclude": [],
        },

        "relations": [
            {
                "id": "R1",
                "subject": "reinforcement learning",
                "predicate": "optimize",
                "object": "LLM academic search agent",
                "level": "hard",
                "source_text": "使用强化学习优化LLM学术搜索Agent",
            }
        ],

        # 默认 hard 为 AND、soft 为排序偏好，因此无需显式 Logic Tree。
        "logic": {
            "hard": None,
            "soft": None,
            "exclude": None,
        },

        "properties": {
            # 只有“2023年以来”，没有“最新/最近”，所以这里不能擅自 recent_first=True
            "recent_first": False,
            "recent_last": False,
            "central_first": False,
            "central_last": False,
            "specific_paper_name": None,
            "suitable_for_by_citing": None,
        },

        "entities": [
            {
                "type": "method",
                "canonical_name": "reinforcement learning",
                "aliases": ["RL"],
                "source_text": "强化学习",
            },
            {
                "type": "topic",
                "canonical_name": "academic literature search agent",
                "aliases": [
                    "academic paper search agent",
                    "scholarly retrieval agent",
                ],
                "source_text": "学术搜索Agent",
            },
        ],

        "domains": {
            "main_field": "Computer Science",
            "secondary_fields": [],
        },

        "clarification": {
            "needed": False,
            "unresolved_ambiguities": [],
            "questions": [],
        },

        "result_requirements": {
            "requested_top_k": None,
            "required_fields": [],
            "output_structure": "unspecified",
            "need_relation_graph": False,
            "need_open_access": False,
            "need_code": False,
        },

        "warnings": [],
    }


def run_analyzer(
    client: FakeLLMClient,
    *,
    config: AnalyzerConfig | None = None,
    query: str = COMPLEX_QUERY,
):
    analyzer = QueryAnalyzer(
        client,
        config=config or AnalyzerConfig(
            use_structured_output=False,
            prompt_schema_mode="summary",
            enable_repair=True,
        ),
    )

    return asyncio.run(
        analyzer.analyze(query)
    )


# ============================================================
# 3. 核心测试：标准复杂查询
# ============================================================

def test_complex_query_primary_success():
    """
    最重要的一条：
    检查一次 LLM 调用能否完整形成可用 QueryAnalysisResult。
    """
    payload = make_base_payload()

    client = FakeLLMClient([payload])

    output = run_analyzer(client)

    # ---------- 工程链路 ----------
    assert output.repaired is False
    assert output.validation.is_valid is True

    assert output.metrics.api_calls == 1
    assert output.metrics.repair_calls == 0

    # ---------- Query Type ----------
    assert output.result.query_type == QueryType.BROAD_BY_DESCRIPTION

    # ---------- Hard ----------
    hard = output.result.constraints.hard

    year = next(
        item
        for item in hard
        if item.type == ConstraintType.YEAR
    )

    assert year.operator == ConstraintOperator.GTE
    assert year.value == 2023

    # ---------- Soft ----------
    soft = output.result.constraints.soft

    venue = next(
        item
        for item in soft
        if item.type == ConstraintType.VENUE
    )

    assert venue.operator == ConstraintOperator.ANY_OF
    assert set(venue.values) == {"ACL", "NeurIPS"}

    # “最好”绝不能被当成 Hard
    assert all(
        not (
            item.type == ConstraintType.VENUE
            and "ACL" in str(item.value)
        )
        for item in hard
    )

    # ---------- Relation ----------
    assert len(output.result.relations) >= 1

    relation = output.result.relations[0]

    assert "reinforcement" in relation.subject.lower()
    assert relation.predicate.lower() in {
        "optimize",
        "optimise",
        "train",
        "optimize_or_train",
    }

    assert "agent" in relation.object.lower()
    assert relation.level.value == "hard"

    # ---------- 不允许把绝对年份偷换成“最新优先” ----------
    assert output.result.properties.recent_first is False


# ============================================================
# 4. Token / API / 延迟指标
# ============================================================

def test_metrics_are_recorded():
    client = FakeLLMClient(
        [make_base_payload()],
        prompt_tokens=123,
        completion_tokens=45,
        model="test-model",
    )

    output = run_analyzer(client)

    assert output.metrics.api_calls == 1
    assert output.metrics.prompt_tokens == 123
    assert output.metrics.completion_tokens == 45
    assert output.metrics.total_tokens == 168

    assert output.metrics.latency_ms >= 0
    assert output.metrics.primary_latency_ms >= 0

    assert output.metrics.model_names == ["test-model"]


# ============================================================
# 5. JSON 容错
# ============================================================

def test_json_code_fence_can_be_parsed():
    payload = make_base_payload()

    raw = (
        "```json\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n```"
    )

    parsed = parse_json_payload(raw)

    assert parsed["query_type"] == "BROAD_BY_DESCRIPTION"
    assert parsed["constraints"]["hard"][0]["value"] == 2023


def test_json_with_extra_text_can_be_parsed():
    payload = make_base_payload()

    raw = (
        "Here is the result:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\nDone."
    )

    parsed = parse_json_payload(raw)

    assert parsed["schema_version"] == "1.0"


# ============================================================
# 6. Validator 错误 -> Repair
# ============================================================

def test_validator_failure_triggers_repair():
    """
    第一次：
        any_of 只有一个值 -> 业务逻辑不合法

    第二次 Repair：
        改为 eq -> 应通过
    """
    bad = make_base_payload()

    bad["constraints"]["soft"][0] = {
        "id": "P1",
        "type": "venue",
        "operator": "any_of",
        "value": None,
        "values": ["ACL"],  # 故意错误：any_of 至少两个
        "source_text": "最好来自ACL",
        "normalized_text": None,
    }

    repaired = make_base_payload()

    repaired["constraints"]["soft"][0] = {
        "id": "P1",
        "type": "venue",
        "operator": "eq",
        "value": "ACL",
        "values": [],
        "source_text": "最好来自ACL",
        "normalized_text": None,
    }

    client = FakeLLMClient(
        [bad, repaired],
        prompt_tokens=100,
        completion_tokens=50,
    )

    output = run_analyzer(client)

    assert output.repaired is True
    assert output.validation.is_valid is True

    assert output.metrics.api_calls == 2
    assert output.metrics.repair_calls == 1

    # 两次调用 Token 都应该被统计
    assert output.metrics.total_tokens == 300


def test_malformed_json_triggers_repair():
    repaired = make_base_payload()

    client = FakeLLMClient(
        [
            "This is not valid JSON.",
            repaired,
        ]
    )

    output = run_analyzer(client)

    assert output.repaired is True
    assert output.metrics.api_calls == 2
    assert output.validation.is_valid is True


# ============================================================
# 7. original_query 不允许被模型改写
# ============================================================

def test_original_query_is_restored_without_extra_llm_call():
    payload = make_base_payload(
        original_query="LLM偷偷改写后的query"
    )

    client = FakeLLMClient([payload])

    output = run_analyzer(client)

    # safe_normalize 应确定性恢复原始 Query，
    # 不应因此浪费 Repair API。
    assert output.result.original_query == COMPLEX_QUERY
    assert output.metrics.api_calls == 1
    assert output.repaired is False


# ============================================================
# 8. AND / OR 配对逻辑
# ============================================================

def test_paired_venue_year_or_logic():
    """
    用户：
        找 ACL 2024 或 NeurIPS 2023 的多模态大模型论文

    正确：
        OR(
          AND(ACL, 2024),
          AND(NeurIPS, 2023)
        )

    错误：
        venue in [ACL, NeurIPS]
        AND year in [2023, 2024]
    """
    query = "找ACL 2024或NeurIPS 2023发表的多模态大模型论文"

    payload = make_base_payload(query)

    payload["content"] = "multimodal large language models"

    payload["constraints"] = {
        "hard": [
            {
                "id": "C1",
                "type": "venue",
                "operator": "eq",
                "value": "ACL",
                "values": [],
                "source_text": "ACL 2024",
                "normalized_text": None,
            },
            {
                "id": "C2",
                "type": "year",
                "operator": "eq",
                "value": 2024,
                "values": [],
                "source_text": "ACL 2024",
                "normalized_text": None,
            },
            {
                "id": "C3",
                "type": "venue",
                "operator": "eq",
                "value": "NeurIPS",
                "values": [],
                "source_text": "NeurIPS 2023",
                "normalized_text": None,
            },
            {
                "id": "C4",
                "type": "year",
                "operator": "eq",
                "value": 2023,
                "values": [],
                "source_text": "NeurIPS 2023",
                "normalized_text": None,
            },
            {
                "id": "C5",
                "type": "topic",
                "operator": "eq",
                "value": "multimodal large language models",
                "values": [],
                "source_text": "多模态大模型",
                "normalized_text": "multimodal large language models",
            },
        ],
        "soft": [],
        "exclude": [],
    }

    payload["relations"] = []

    payload["logic"] = {
        "hard": {
            "operator": "AND",
            "ref_id": None,
            "children": [
                {
                    "operator": "OR",
                    "ref_id": None,
                    "children": [
                        {
                            "operator": "AND",
                            "ref_id": None,
                            "children": [
                                {
                                    "operator": "LEAF",
                                    "ref_id": "C1",
                                    "children": [],
                                },
                                {
                                    "operator": "LEAF",
                                    "ref_id": "C2",
                                    "children": [],
                                },
                            ],
                        },
                        {
                            "operator": "AND",
                            "ref_id": None,
                            "children": [
                                {
                                    "operator": "LEAF",
                                    "ref_id": "C3",
                                    "children": [],
                                },
                                {
                                    "operator": "LEAF",
                                    "ref_id": "C4",
                                    "children": [],
                                },
                            ],
                        },
                    ],
                },
                {
                    "operator": "LEAF",
                    "ref_id": "C5",
                    "children": [],
                },
            ],
        },
        "soft": None,
        "exclude": None,
    }

    client = FakeLLMClient([payload])

    output = run_analyzer(
        client,
        query=query,
    )

    assert output.validation.is_valid is True
    assert output.result.logic.hard is not None

    root = output.result.logic.hard

    assert root.operator == "AND"
    assert root.children[0].operator == "OR"


# ============================================================
# 9. Exclude（排除条件）
# ============================================================

def test_exclude_constraint():
    query = "找2022年以来的多模态大模型论文，不要综述文章"

    payload = make_base_payload(query)
    payload["content"] = "multimodal large language models"

    payload["constraints"] = {
        "hard": [
            {
                "id": "C1",
                "type": "year",
                "operator": "gte",
                "value": 2022,
                "values": [],
                "source_text": "2022年以来",
                "normalized_text": None,
            },
            {
                "id": "C2",
                "type": "topic",
                "operator": "eq",
                "value": "multimodal large language models",
                "values": [],
                "source_text": "多模态大模型",
                "normalized_text": None,
            },
        ],

        "soft": [],

        "exclude": [
            {
                "id": "E1",
                "type": "publication_type",
                "operator": "eq",
                "value": "survey",
                "values": [],
                "source_text": "不要综述文章",
                "normalized_text": "survey/review",
            }
        ],
    }

    payload["relations"] = []

    client = FakeLLMClient([payload])

    output = run_analyzer(
        client,
        query=query,
    )

    assert output.validation.is_valid is True

    assert len(output.result.constraints.exclude) == 1

    excluded = output.result.constraints.exclude[0]

    assert excluded.type == ConstraintType.PUBLICATION_TYPE
    assert excluded.id == "E1"


# ============================================================
# 10. 可选：真实 LLM 语义验收
# ============================================================

@pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1",
    reason=(
        "默认跳过真实 LLM 测试。"
        "设置 RUN_REAL_LLM_TESTS=1 后启用。"
    ),
)
def test_real_llm_semantic_acceptance():
    """
    这个测试才是真正检查“模型有没有理解对”。

    需要你后续建立：
        tests/real_llm_client.py

    并实现：
        build_client() -> 满足 analyzer.LLMClient Protocol 的客户端

    该测试不会强行要求模型输出一模一样的字符串，
    只检查真正影响检索 F1 的关键语义。
    """
    from tests.real_llm_client import build_client

    analyzer = QueryAnalyzer(
        build_client(),
        config=AnalyzerConfig(
            use_structured_output=False,
            prompt_schema_mode="summary",
            enable_repair=True,
            max_transport_retries=0,
        ),
    )

    output = asyncio.run(
        analyzer.analyze(COMPLEX_QUERY)
    )

    print("\n========== REAL LLM METRICS ==========")
    print(f"api_calls             = {output.metrics.api_calls}")
    print(f"repair_calls          = {output.metrics.repair_calls}")
    print(f"repaired              = {output.repaired}")

    print(f"prompt_tokens         = {output.metrics.prompt_tokens}")
    print(f"completion_tokens     = {output.metrics.completion_tokens}")
    print(f"total_tokens          = {output.metrics.total_tokens}")

    print(f"primary_latency_ms    = {output.metrics.primary_latency_ms:.2f}")
    print(f"repair_latency_ms     = {output.metrics.repair_latency_ms:.2f}")
    print(f"total_latency_ms      = {output.metrics.latency_ms:.2f}")
    print(f"models                = {output.metrics.model_names}")

    print("\n========== VALIDATION ==========")
    print(output.validation.summary())

    print("\n========== RESULT ==========")
    print(output.result.model_dump_json(indent=2))

    result = output.result

    # ---------- Query Type ----------
    assert result.query_type == QueryType.BROAD_BY_DESCRIPTION

    # ---------- Year >= 2023 ----------
    assert any(
        item.type == ConstraintType.YEAR
        and item.operator == ConstraintOperator.GTE
        and item.value == 2023
        for item in result.constraints.hard
    )

    # ---------- ACL / NeurIPS 必须是 Soft ----------
    soft_venue_text = " ".join(
        [
            str(item.value or "")
            + " "
            + " ".join(map(str, item.values))
            for item in result.constraints.soft
            if item.type == ConstraintType.VENUE
        ]
    ).lower()

    assert "acl" in soft_venue_text
    assert "neurips" in soft_venue_text

    hard_venue_text = " ".join(
        [
            str(item.value or "")
            + " "
            + " ".join(map(str, item.values))
            for item in result.constraints.hard
            if item.type == ConstraintType.VENUE
        ]
    ).lower()

    assert "acl" not in hard_venue_text
    assert "neurips" not in hard_venue_text

    # ---------- 必须保存 RL -> optimize -> Agent 的关系 ----------
    assert any(
        (
            "reinforcement" in rel.subject.lower()
            or rel.subject.lower().strip() == "rl"
        )
        and (
            "optim" in rel.predicate.lower()
            or "train" in rel.predicate.lower()
        )
        and "agent" in rel.object.lower()
        and rel.level.value == "hard"
        for rel in result.relations
    )

    # ---------- “2023年以来”不能擅自变为“最新优先” ----------
    assert result.properties.recent_first is False

    # ---------- 工程要求 ----------
    assert output.validation.is_valid
    assert output.metrics.api_calls <= 2



# ============================================================
# 11. 真实 LLM 小规模语义回归测试
#     注意：以下内容仅新增，不修改前面的任何测试
# ============================================================

def _build_real_analyzer():
    """
    仅供下面新增的真实 LLM 测试复用。
    不影响现有 test_real_llm_semantic_acceptance。
    """
    from tests.real_llm_client import build_client

    return QueryAnalyzer(
        build_client(),
        config=AnalyzerConfig(
            use_structured_output=False,
            prompt_schema_mode="summary",
            enable_repair=True,
            max_transport_retries=0,
        ),
    )


def _print_real_result(name: str, output):
    """
    统一打印指标，方便后续比较不同模型。
    """
    print(f"\n========== {name} ==========")

    print(f"api_calls          = {output.metrics.api_calls}")
    print(f"repair_calls       = {output.metrics.repair_calls}")
    print(f"repaired           = {output.repaired}")

    print(f"prompt_tokens      = {output.metrics.prompt_tokens}")
    print(f"completion_tokens  = {output.metrics.completion_tokens}")
    print(f"total_tokens       = {output.metrics.total_tokens}")

    print(f"latency_ms         = {output.metrics.latency_ms:.2f}")
    print(f"models             = {output.metrics.model_names}")

    print("\nRESULT:")
    print(output.result.model_dump_json(indent=2))


# ============================================================
# 11.1 OR：RAG 或 Agentic Retrieval
# ============================================================

@pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1",
    reason="设置 RUN_REAL_LLM_TESTS=1 后启用真实 LLM 测试。",
)
def test_real_llm_or_semantics():

    query = (
        "找使用RAG或Agentic Retrieval"
        "提升LLM学术检索能力的论文。"
    )

    analyzer = _build_real_analyzer()

    output = asyncio.run(
        analyzer.analyze(query)
    )

    _print_real_result(
        "REAL LLM OR TEST",
        output,
    )

    result = output.result

    raw = result.model_dump_json().lower()

    # 两个条件都必须保留下来
    assert "rag" in raw
    assert "agentic retrieval" in raw

    # --------------------------------------------------------
    # OR 允许两种合法表达：
    #
    # 1. ConstraintOperator.ANY_OF
    # 2. LogicNode(operator="OR")
    #
    # 不强迫模型只能使用其中一种。
    # --------------------------------------------------------

    has_any_of = any(
        item.operator == ConstraintOperator.ANY_OF
        and (
            "rag" in " ".join(
                map(str, item.values)
            ).lower()
        )
        and (
            "agentic retrieval" in " ".join(
                map(str, item.values)
            ).lower()
        )
        for item in result.constraints.hard
    )

    def contains_or(node):
        if node is None:
            return False

        if node.operator == "OR":
            return True

        return any(
            contains_or(child)
            for child in node.children
        )

    has_logic_or = contains_or(
        result.logic.hard
    )

    assert has_any_of or has_logic_or

    assert output.validation.is_valid
    assert output.metrics.api_calls <= 2


# ============================================================
# 11.2 Exclude / NOT：不要使用 RAG
# ============================================================

@pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1",
    reason="设置 RUN_REAL_LLM_TESTS=1 后启用真实 LLM 测试。",
)
def test_real_llm_exclude_semantics():

    query = (
        "找研究LLM Agent学术搜索的方法，"
        "但不要使用RAG。"
    )

    analyzer = _build_real_analyzer()

    output = asyncio.run(
        analyzer.analyze(query)
    )

    _print_real_result(
        "REAL LLM EXCLUDE TEST",
        output,
    )

    result = output.result

    # 必须识别 LLM Agent 主主题
    raw = result.model_dump_json().lower()

    assert "agent" in raw
    assert "rag" in raw

    # --------------------------------------------------------
    # RAG 必须体现为排除语义。
    # 可以进入 constraints.exclude，
    # 也可以进入显式 NOT logic。
    # --------------------------------------------------------

    exclude_text = " ".join(
        (
            str(item.value or "")
            + " "
            + " ".join(map(str, item.values))
        )
        for item in result.constraints.exclude
    ).lower()

    has_exclude_constraint = (
        "rag" in exclude_text
    )

    def contains_not(node):
        if node is None:
            return False

        if node.operator == "NOT":
            return True

        return any(
            contains_not(child)
            for child in node.children
        )

    has_not_logic = (
        contains_not(result.logic.hard)
        or contains_not(result.logic.exclude)
    )

    assert (
        has_exclude_constraint
        or has_not_logic
    )

    assert output.validation.is_valid
    assert output.metrics.api_calls <= 2


# ============================================================
# 11.3 Hard Metadata：2024 + ACL
# ============================================================

@pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1",
    reason="设置 RUN_REAL_LLM_TESTS=1 后启用真实 LLM 测试。",
)
def test_real_llm_hard_metadata_semantics():

    query = (
        "找2024年发表在ACL的"
        "LLM Agent论文。"
    )

    analyzer = _build_real_analyzer()

    output = asyncio.run(
        analyzer.analyze(query)
    )

    _print_real_result(
        "REAL LLM HARD METADATA TEST",
        output,
    )

    result = output.result

    # 2024 必须是 Hard
    assert any(
        item.type == ConstraintType.YEAR
        and item.value == 2024
        for item in result.constraints.hard
    )

    # ACL 不是“最好”，而是明确限制，因此必须 Hard
    hard_venue_text = " ".join(
        (
            str(item.value or "")
            + " "
            + " ".join(map(str, item.values))
        )
        for item in result.constraints.hard
        if item.type == ConstraintType.VENUE
    ).lower()

    assert "acl" in hard_venue_text

    # 不应该错误放入 Soft
    soft_venue_text = " ".join(
        (
            str(item.value or "")
            + " "
            + " ".join(map(str, item.values))
        )
        for item in result.constraints.soft
        if item.type == ConstraintType.VENUE
    ).lower()

    assert "acl" not in soft_venue_text

    assert output.validation.is_valid
    assert output.metrics.api_calls <= 2


# ============================================================
# 11.4 Author + Year
# ============================================================

@pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1",
    reason="设置 RUN_REAL_LLM_TESTS=1 后启用真实 LLM 测试。",
)
def test_real_llm_author_semantics():

    query = (
        "找Yann LeCun "
        "2020年以来发表的论文。"
    )

    analyzer = _build_real_analyzer()

    output = asyncio.run(
        analyzer.analyze(query)
    )

    _print_real_result(
        "REAL LLM AUTHOR TEST",
        output,
    )

    result = output.result

    # 查询类型
    assert result.query_type == QueryType.BY_AUTHOR

    raw = result.model_dump_json().lower()

    assert "yann" in raw
    assert "lecun" in raw

    # 年份必须 >= 2020
    assert any(
        item.type == ConstraintType.YEAR
        and item.operator == ConstraintOperator.GTE
        and item.value == 2020
        for item in result.constraints.hard
    )

    assert output.validation.is_valid
    assert output.metrics.api_calls <= 2


# ============================================================
# 11.5 DOI Identifier 精确查找
# ============================================================

@pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1",
    reason="设置 RUN_REAL_LLM_TESTS=1 后启用真实 LLM 测试。",
)
def test_real_llm_identifier_semantics():

    doi = "10.1145/1234567.1234568"

    query = (
        f"找 DOI 为 {doi} 的论文。"
    )

    analyzer = _build_real_analyzer()

    output = asyncio.run(
        analyzer.analyze(query)
    )

    _print_real_result(
        "REAL LLM IDENTIFIER TEST",
        output,
    )

    result = output.result

    # 必须进入精确标识符查询路线
    assert (
        result.query_type
        == QueryType.SPECIFIC_BY_IDENTIFIER
    )

    raw = result.model_dump_json().lower()

    # DOI 不能丢
    assert doi.lower() in raw

    # 最好还能显式形成 IDENTIFIER Constraint
    assert any(
        item.type == ConstraintType.IDENTIFIER
        and doi.lower()
        in (
            str(item.value or "")
            + " "
            + " ".join(map(str, item.values))
        ).lower()
        for item in result.constraints.hard
    )

    assert output.validation.is_valid
    assert output.metrics.api_calls <= 2


# ============================================================
# 11.6 方法关系 + Benchmark Evaluation
# ============================================================

@pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1",
    reason="设置 RUN_REAL_LLM_TESTS=1 后启用真实 LLM 测试。",
)
def test_real_llm_relation_chain_semantics():

    query = (
        "找使用知识蒸馏压缩大语言模型，"
        "并在MMLU上进行评估的论文。"
    )

    analyzer = _build_real_analyzer()

    output = asyncio.run(
        analyzer.analyze(query)
    )

    _print_real_result(
        "REAL LLM RELATION CHAIN TEST",
        output,
    )

    result = output.result

    raw = result.model_dump_json().lower()

    # 基本实体不能丢
    assert (
        "distill" in raw
        or "知识蒸馏" in raw
    )

    assert "mmlu" in raw

    # --------------------------------------------------------
    # 至少应保留：
    # 大语言模型 -> evaluated_on -> MMLU
    #
    # predicate 可以是：
    # evaluate / evaluated_on / validate / benchmark 等
    # --------------------------------------------------------

    has_mmlu_evaluation = any(
        (
            "evaluat" in rel.predicate.lower()
            or "validat" in rel.predicate.lower()
            or "benchmark" in rel.predicate.lower()
            or "test" in rel.predicate.lower()
        )
        and "mmlu" in rel.object.lower()
        and rel.level.value == "hard"
        for rel in result.relations
    )

    assert has_mmlu_evaluation

    # --------------------------------------------------------
    # 知识蒸馏与模型压缩的关系也应该保存。
    # 不要求 predicate 字符串完全固定。
    # --------------------------------------------------------

    has_distillation_relation = any(
        (
            "distill" in rel.subject.lower()
            or "distill" in rel.predicate.lower()
            or "知识蒸馏" in rel.subject
        )
        and (
            "model" in rel.object.lower()
            or "language" in rel.object.lower()
            or "llm" in rel.object.lower()
        )
        for rel in result.relations
    )

    assert has_distillation_relation

    assert output.validation.is_valid
    assert output.metrics.api_calls <= 2