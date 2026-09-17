from __future__ import annotations

from collections.abc import Sequence

from ..candidate_pool.filter import evaluate_constraint
from ..candidate_pool.models import CandidatePaper, FilterEvidence, MatchStatus
from ..candidate_pool.tri_logic import collect_refs, evaluate_logic, tri_and, tri_or
from ..query_understanding.schema import ConstraintLevel, QueryAnalysisResult
from .models import CriterionJudgement, PaperSelectionResult, SelectorCriterion


def evaluate_paper_selection(
    candidate: CandidatePaper,
    analysis: QueryAnalysisResult,
    criteria: Sequence[SelectorCriterion],
    semantic_judgements: Sequence[CriterionJudgement],
    *,
    prior_score: float | None = None,
    summary: str | None = None,
    strict_field_of_study: bool = False,
) -> PaperSelectionResult:
    constraints = {
        constraint.id: constraint
        for bucket in (analysis.constraints.hard, analysis.constraints.soft, analysis.constraints.exclude)
        for constraint in bucket
    }
    relations = {relation.id: relation for relation in analysis.relations}
    semantic_map = {judgement.criterion_id: judgement for judgement in semantic_judgements}
    metadata_map: dict[str, FilterEvidence] = {}

    def leaf_eval(ref_id: str) -> MatchStatus:
        semantic = semantic_map.get(ref_id)
        if semantic is not None:
            return semantic.status
        if ref_id in metadata_map:
            return metadata_map[ref_id].status
        if ref_id in constraints:
            evidence = evaluate_constraint(
                candidate,
                constraints[ref_id],
                analysis,
                strict_field_of_study=strict_field_of_study,
            )
            metadata_map[ref_id] = evidence
            return evidence.status
        if ref_id in relations:
            return MatchStatus.UNKNOWN
        return MatchStatus.UNKNOWN

    if analysis.logic.hard is not None:
        hard_status = evaluate_logic(analysis.logic.hard, leaf_eval)
        referenced = collect_refs(analysis.logic.hard)
        missing_hard = [constraint.id for constraint in analysis.constraints.hard if constraint.id not in referenced]
        missing_rel = [
            relation.id
            for relation in analysis.relations
            if relation.level == ConstraintLevel.HARD and relation.id not in referenced
        ]
        hard_status = tri_and([hard_status] + [leaf_eval(ref) for ref in missing_hard + missing_rel])
    else:
        hard_status = tri_and(
            [leaf_eval(constraint.id) for constraint in analysis.constraints.hard]
            + [
                leaf_eval(relation.id)
                for relation in analysis.relations
                if relation.level == ConstraintLevel.HARD
            ]
        )

    synthetic_hard_ids = [
        criterion.criterion_id
        for criterion in criteria
        if criterion.synthetic and criterion.level == ConstraintLevel.HARD
    ]
    if synthetic_hard_ids:
        hard_status = tri_and([hard_status] + [leaf_eval(ref_id) for ref_id in synthetic_hard_ids])

    if analysis.logic.exclude is not None:
        exclude_match = evaluate_logic(analysis.logic.exclude, leaf_eval)
        referenced = collect_refs(analysis.logic.exclude)
        missing_exclude = [
            constraint.id for constraint in analysis.constraints.exclude if constraint.id not in referenced
        ]
        missing_rel = [
            relation.id
            for relation in analysis.relations
            if relation.level == ConstraintLevel.EXCLUDE and relation.id not in referenced
        ]
        if missing_exclude or missing_rel:
            exclude_match = tri_or(
                [exclude_match] + [leaf_eval(ref) for ref in missing_exclude + missing_rel]
            )
    else:
        exclude_values = [leaf_eval(constraint.id) for constraint in analysis.constraints.exclude]
        exclude_values.extend(
            leaf_eval(relation.id)
            for relation in analysis.relations
            if relation.level == ConstraintLevel.EXCLUDE
        )
        exclude_match = tri_or(exclude_values) if exclude_values else MatchStatus.FAIL

    if hard_status == MatchStatus.FAIL or exclude_match == MatchStatus.PASS:
        final_status = MatchStatus.FAIL
    elif hard_status == MatchStatus.PASS and exclude_match == MatchStatus.FAIL:
        final_status = MatchStatus.PASS
    else:
        final_status = MatchStatus.UNKNOWN

    soft_ids = [constraint.id for constraint in analysis.constraints.soft]
    soft_ids.extend(
        relation.id for relation in analysis.relations if relation.level == ConstraintLevel.SOFT
    )
    soft_values = [leaf_eval(ref_id) for ref_id in soft_ids]
    soft_match_score = (
        sum(value == MatchStatus.PASS for value in soft_values) / len(soft_values)
        if soft_values
        else 0.0
    )

    known = sum(judgement.status != MatchStatus.UNKNOWN for judgement in semantic_judgements)
    semantic_known_fraction = known / len(semantic_judgements) if semantic_judgements else 1.0

    warnings: list[str] = []
    for judgement in semantic_judgements:
        warnings.extend(judgement.warnings)

    return PaperSelectionResult(
        candidate_id=candidate.candidate_id,
        final_status=final_status,
        hard_status=hard_status,
        exclude_match_status=exclude_match,
        soft_match_score=float(soft_match_score),
        semantic_judgements=list(semantic_judgements),
        metadata_evidence=list(metadata_map.values()),
        prior_score=prior_score,
        retrieval_score=candidate.retrieval_score,
        semantic_known_fraction=float(semantic_known_fraction),
        summary=summary,
        warnings=warnings,
    )


__all__ = ["evaluate_paper_selection"]
