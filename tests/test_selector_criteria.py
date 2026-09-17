from scholar_agent_v2.query_understanding.schema import (
    ConstraintItem,
    ConstraintLevel,
    ConstraintOperator,
    ConstraintSet,
    ConstraintType,
    QueryAnalysisResult,
    QueryIntent,
    QueryType,
    RelationConstraint,
    RetrievalMode,
    SearchTask,
)
from scholar_agent_v2.selector.criteria import build_selector_criteria
from scholar_agent_v2.selector.models import CriterionKind


def _base_analysis(**kwargs):
    payload = dict(
        original_query="papers about multimodal forgery detection",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(task=SearchTask.PAPER_SEARCH, mode=RetrievalMode.COMPREHENSIVE),
        content="multimodal forgery detection",
    )
    payload.update(kwargs)
    return QueryAnalysisResult(**payload)


def test_content_is_fallback_when_no_hard_semantic_criteria():
    analysis = _base_analysis(
        constraints=ConstraintSet(
            hard=[ConstraintItem(id="C1", type=ConstraintType.YEAR, operator=ConstraintOperator.GTE, value=2023)]
        )
    )
    criteria = build_selector_criteria(analysis)
    assert [criterion.criterion_id for criterion in criteria] == ["Q_CONTENT"]
    assert criteria[0].kind == CriterionKind.CONTENT
    assert criteria[0].synthetic is True


def test_complex_relations_prevent_redundant_content_criterion():
    analysis = _base_analysis(
        content="reinforcement learning optimization of LLM academic search agent evaluated on real scientific query datasets",
        constraints=ConstraintSet(
            hard=[
                ConstraintItem(id="C1", type=ConstraintType.YEAR, operator=ConstraintOperator.GTE, value=2023),
                ConstraintItem(
                    id="C2",
                    type=ConstraintType.EVALUATION,
                    operator=ConstraintOperator.CONTAINS,
                    value="real scientific query datasets",
                ),
            ]
        ),
        relations=[
            RelationConstraint(
                id="R1",
                subject="reinforcement learning",
                predicate="optimize",
                object="LLM academic search agent",
                level=ConstraintLevel.HARD,
            ),
            RelationConstraint(
                id="R2",
                subject="LLM academic search agent",
                predicate="evaluated_on",
                object="real scientific query datasets",
                level=ConstraintLevel.HARD,
            ),
        ],
    )
    criteria = build_selector_criteria(analysis)
    ids = [criterion.criterion_id for criterion in criteria]
    assert "C1" not in ids  # deterministic metadata remains local
    assert "Q_CONTENT" not in ids  # avoid redundant whole-query judgement
    assert ids == ["C2", "R1", "R2"]


def test_metadata_exclude_is_not_sent_to_llm():
    analysis = _base_analysis(
        constraints=ConstraintSet(
            exclude=[
                ConstraintItem(
                    id="E1",
                    type=ConstraintType.PUBLICATION_TYPE,
                    value="survey",
                )
            ]
        )
    )
    criteria = build_selector_criteria(analysis)
    assert [criterion.criterion_id for criterion in criteria] == ["Q_CONTENT"]
