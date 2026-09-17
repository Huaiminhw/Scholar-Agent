from __future__ import annotations

import asyncio

from scholar_agent_v2.query_understanding.analyzer import (
    AnalyzerConfig,
    LLMResponse,
    LLMUsage,
    QueryAnalyzer,
)
from scholar_agent_v2.query_understanding.normalizer import (
    normalize_identifier,
    normalize_query_analysis,
)
from scholar_agent_v2.query_understanding.schema import (
    ClarificationInfo,
    ConstraintItem,
    ConstraintLogic,
    ConstraintOperator,
    ConstraintSet,
    ConstraintType,
    DomainInfo,
    EntityType,
    LogicNode,
    ParseStatus,
    QueryAnalysisResult,
    QueryEntity,
    QueryIntent,
    QueryType,
    RelationConstraint,
    ResultRequirements,
)
from scholar_agent_v2.query_understanding.validator import validate_query_analysis


def _analysis() -> QueryAnalysisResult:
    return QueryAnalysisResult(
        original_query="model supplied query",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(goal="  Find\n multimodal   retrieval papers  "),
        content="  multimodal\u3000LLM\n retrieval  ",
        constraints=ConstraintSet(
            hard=[
                ConstraintItem(
                    id="C1",
                    type=ConstraintType.YEAR,
                    operator=ConstraintOperator.GTE,
                    value="２０２３",
                    source_text="  ２０２３ 年以后  ",
                ),
                ConstraintItem(
                    id="C2",
                    type=ConstraintType.IDENTIFIER,
                    value="https://doi.org/10.1000/XYZ",
                    source_text=" DOI 10.1000/XYZ ",
                    normalized_text="DOI:10.1000/XYZ",
                ),
                ConstraintItem(
                    id="C3",
                    type=ConstraintType.VENUE,
                    operator=ConstraintOperator.ANY_OF,
                    value="ACL",
                    values=[" ACL ", "NeurIPS", "neurips"],
                    source_text="ACL or NeurIPS",
                ),
            ]
        ),
        relations=[
            RelationConstraint(
                id="R1",
                subject="  reinforcement   learning ",
                predicate=" optimize ",
                object=" LLM\nagent ",
                source_text=" RL optimizes LLM agents ",
            )
        ],
        logic=ConstraintLogic(
            hard=LogicNode(
                operator="AND",
                children=[
                    LogicNode(operator="LEAF", ref_id="C1"),
                    LogicNode(operator="LEAF", ref_id="C2"),
                    LogicNode(operator="LEAF", ref_id="C3"),
                    LogicNode(operator="LEAF", ref_id="R1"),
                ],
            )
        ),
        entities=[
            QueryEntity(
                type=EntityType.MODEL,
                canonical_name=" Large   Language Model ",
                aliases=[" LLM ", "llm", "Large Language Model", ""],
                source_text=" large language model ",
            ),
            QueryEntity(
                type=EntityType.MODEL,
                canonical_name="large language model",
                aliases=["Large-language model", "LLM"],
            ),
        ],
        domains=DomainInfo(
            main_field=" Computer   Science ",
            secondary_fields=["computer science", " NLP ", "nlp"],
        ),
        clarification=ClarificationInfo(
            needed=True,
            unresolved_ambiguities=[" model size ", "Model  size"],
            questions=[" Which size? ", "Which   size?"],
        ),
        result_requirements=ResultRequirements(
            required_fields=[" title ", "TITLE", "doi"]
        ),
        warnings=[" metadata incomplete ", "Metadata   incomplete"],
    )


def test_normalize_query_analysis_is_semantics_preserving_and_idempotent():
    authoritative_query = "  用户原始查询不能被改写  "
    original = _analysis()

    normalized = normalize_query_analysis(
        original,
        original_query=authoritative_query,
    )

    assert normalized.original_query == authoritative_query
    assert normalized.content == "multimodal LLM retrieval"
    assert normalized.intent.goal == "Find multimodal retrieval papers"
    assert normalized.status == ParseStatus.NEEDS_CLARIFICATION

    year, identifier, venues = normalized.constraints.hard
    assert year.value == 2023
    assert identifier.value == "10.1000/xyz"
    assert identifier.normalized_text == "10.1000/xyz"
    assert venues.value is None
    assert venues.values == ["ACL", "NeurIPS"]

    assert [item.id for item in normalized.constraints.hard] == ["C1", "C2", "C3"]
    assert normalized.logic == original.logic
    assert normalized.relations[0].subject == "reinforcement learning"
    assert normalized.relations[0].predicate == "optimize"
    assert normalized.relations[0].object == "LLM agent"

    assert len(normalized.entities) == 1
    assert normalized.entities[0].canonical_name == "Large Language Model"
    assert normalized.entities[0].aliases == ["LLM", "Large-language model"]
    assert normalized.domains.main_field == "Computer Science"
    assert normalized.domains.secondary_fields == ["NLP"]
    assert normalized.result_requirements.required_fields == ["title", "doi"]
    assert normalized.warnings == ["metadata incomplete"]

    normalized_twice = normalize_query_analysis(
        normalized,
        original_query=authoritative_query,
    )
    assert normalized_twice.model_dump() == normalized.model_dump()


def test_normalizer_keeps_real_shape_conflict_for_validator():
    analysis = _analysis()
    venue = analysis.constraints.hard[2]
    venue.value = "ICML"
    venue.values = ["ACL", "NeurIPS"]

    normalized = normalize_query_analysis(
        analysis,
        original_query=analysis.original_query,
    )
    report = validate_query_analysis(
        normalized,
        original_query=analysis.original_query,
    )

    assert normalized.constraints.hard[2].value == "ICML"
    assert not report.is_valid
    assert any(issue.code == "MULTI_VALUE_OP_HAS_VALUE" for issue in report.errors)


def test_identifier_normalization_only_changes_recognized_forms():
    assert normalize_identifier("DOI:10.1000/XYZ") == "10.1000/xyz"
    assert normalize_identifier("arXiv:2401.12345v2") == "2401.12345"
    assert normalize_identifier("CorpusID: 12345") == "CorpusID: 12345"


class _FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    async def complete(self, **_kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=self.payload,
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model="fake",
        )


def test_query_analyzer_uses_normalizer_before_business_validation():
    query = "Find ACL or NeurIPS papers since 2023"
    analysis = _analysis()
    analysis.clarification = ClarificationInfo()
    analysis.status = ParseStatus.SUCCESS
    payload = analysis.model_dump(mode="json")
    payload["original_query"] = "LLM rewrote this"

    # Redundant value duplicates a valid ANY_OF member.  The normalizer can
    # remove this representation noise safely, avoiding a Repair LLM call.
    payload["constraints"]["hard"][2]["value"] = "ACL"
    payload["constraints"]["hard"][2]["values"] = ["ACL", "NeurIPS", "neurips"]

    client = _FakeClient(payload)
    analyzer = QueryAnalyzer(client, config=AnalyzerConfig(enable_repair=True))
    output = asyncio.run(analyzer.analyze(query))

    assert client.calls == 1
    assert output.repaired is False
    assert output.validation.is_valid
    assert output.result.original_query == query
    assert output.result.constraints.hard[0].value == 2023
    assert output.result.constraints.hard[2].value is None
    assert output.result.constraints.hard[2].values == ["ACL", "NeurIPS"]
