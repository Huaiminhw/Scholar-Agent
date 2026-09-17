from scholar_agent_v2.candidate_pool.models import CandidatePaper, MatchStatus
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
from scholar_agent_v2.selector.evaluator import evaluate_paper_selection
from scholar_agent_v2.selector.models import CriterionJudgement


def make_complex_analysis():
    return QueryAnalysisResult(
        original_query="找2023年以来使用强化学习优化LLM学术搜索Agent，并在真实科研查询数据集上验证的论文",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(task=SearchTask.PAPER_SEARCH, mode=RetrievalMode.COMPREHENSIVE),
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
            RelationConstraint(id="R1", subject="reinforcement learning", predicate="optimize", object="LLM academic search agent"),
            RelationConstraint(id="R2", subject="LLM academic search agent", predicate="evaluated_on", object="real scientific query datasets"),
        ],
    )


def _judgement(criterion_id, status):
    return CriterionJudgement(criterion_id=criterion_id, status=status, evidence_verified=True)


def test_all_hard_semantics_pass_means_pass():
    analysis = make_complex_analysis()
    criteria = build_selector_criteria(analysis)
    candidate = CandidatePaper(candidate_id="P1", year=2024)
    result = evaluate_paper_selection(
        candidate,
        analysis,
        criteria,
        [_judgement("C2", MatchStatus.PASS), _judgement("R1", MatchStatus.PASS), _judgement("R2", MatchStatus.PASS)],
    )
    assert result.hard_status == MatchStatus.PASS
    assert result.final_status == MatchStatus.PASS


def test_unknown_semantic_condition_protects_recall():
    analysis = make_complex_analysis()
    criteria = build_selector_criteria(analysis)
    candidate = CandidatePaper(candidate_id="P1", year=2024)
    result = evaluate_paper_selection(
        candidate,
        analysis,
        criteria,
        [_judgement("C2", MatchStatus.UNKNOWN), _judgement("R1", MatchStatus.PASS), _judgement("R2", MatchStatus.PASS)],
    )
    assert result.final_status == MatchStatus.UNKNOWN


def test_explicit_hard_failure_rejects():
    analysis = make_complex_analysis()
    criteria = build_selector_criteria(analysis)
    candidate = CandidatePaper(candidate_id="P1", year=2024)
    result = evaluate_paper_selection(
        candidate,
        analysis,
        criteria,
        [_judgement("C2", MatchStatus.PASS), _judgement("R1", MatchStatus.FAIL), _judgement("R2", MatchStatus.PASS)],
    )
    assert result.final_status == MatchStatus.FAIL
