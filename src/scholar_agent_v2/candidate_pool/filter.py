from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from ..query_understanding.schema import (
    ConstraintItem,
    ConstraintLevel,
    ConstraintOperator,
    ConstraintType,
    EntityType,
    LogicNode,
    QueryAnalysisResult,
)

from .models import CandidatePaper, FilterEvidence, MatchStatus


@dataclass
class FilterDecision:
    status: MatchStatus
    evidence: list[FilterEvidence]
    soft_match_score: float = 0.0


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _constraint_values(c: ConstraintItem) -> list[Any]:
    return list(c.values) if c.values else ([c.value] if c.value is not None else [])


def _tri_and(values: list[MatchStatus]) -> MatchStatus:
    if any(v == MatchStatus.FAIL for v in values):
        return MatchStatus.FAIL
    if values and all(v == MatchStatus.PASS for v in values):
        return MatchStatus.PASS
    return MatchStatus.UNKNOWN if values else MatchStatus.PASS


def _tri_or(values: list[MatchStatus]) -> MatchStatus:
    if any(v == MatchStatus.PASS for v in values):
        return MatchStatus.PASS
    if values and all(v == MatchStatus.FAIL for v in values):
        return MatchStatus.FAIL
    return MatchStatus.UNKNOWN if values else MatchStatus.FAIL


def _tri_not(value: MatchStatus) -> MatchStatus:
    if value == MatchStatus.PASS:
        return MatchStatus.FAIL
    if value == MatchStatus.FAIL:
        return MatchStatus.PASS
    return MatchStatus.UNKNOWN


def _compare_numeric(observed: int | float | None, c: ConstraintItem) -> MatchStatus:
    if observed is None:
        return MatchStatus.UNKNOWN
    vals = _constraint_values(c)
    try:
        nums = [float(x) for x in vals]
        x = float(observed)
    except (TypeError, ValueError):
        return MatchStatus.UNKNOWN
    if not nums:
        return MatchStatus.UNKNOWN

    op = c.operator
    if op == ConstraintOperator.EQ:
        return MatchStatus.PASS if x == nums[0] else MatchStatus.FAIL
    if op == ConstraintOperator.NE:
        return MatchStatus.PASS if x != nums[0] else MatchStatus.FAIL
    if op == ConstraintOperator.GT:
        return MatchStatus.PASS if x > nums[0] else MatchStatus.FAIL
    if op == ConstraintOperator.GTE:
        return MatchStatus.PASS if x >= nums[0] else MatchStatus.FAIL
    if op == ConstraintOperator.LT:
        return MatchStatus.PASS if x < nums[0] else MatchStatus.FAIL
    if op == ConstraintOperator.LTE:
        return MatchStatus.PASS if x <= nums[0] else MatchStatus.FAIL
    if op == ConstraintOperator.BETWEEN and len(nums) >= 2:
        lo, hi = min(nums[0], nums[1]), max(nums[0], nums[1])
        return MatchStatus.PASS if lo <= x <= hi else MatchStatus.FAIL
    if op == ConstraintOperator.ANY_OF:
        return MatchStatus.PASS if x in nums else MatchStatus.FAIL
    if op == ConstraintOperator.ALL_OF:
        return MatchStatus.PASS if all(x == n for n in nums) else MatchStatus.FAIL
    return MatchStatus.UNKNOWN


def _string_match_one(observed: str, expected: str, operator: ConstraintOperator) -> bool:
    o, e = _norm(observed), _norm(expected)
    if not o or not e:
        return False
    if operator in {ConstraintOperator.CONTAINS, ConstraintOperator.EQ, ConstraintOperator.ANY_OF, ConstraintOperator.ALL_OF}:
        return o == e or e in o
    if operator in {ConstraintOperator.NOT_CONTAINS, ConstraintOperator.NE}:
        return not (o == e or e in o)
    return o == e


def _compare_string_list(
    observed: list[str] | None,
    c: ConstraintItem,
    *,
    aliases: dict[str, set[str]] | None = None,
) -> MatchStatus:
    if not observed:
        return MatchStatus.UNKNOWN
    vals = [str(v) for v in _constraint_values(c)]
    if not vals:
        return MatchStatus.UNKNOWN

    aliases = aliases or {}
    expanded: list[set[str]] = []
    for value in vals:
        n = _norm(value)
        expanded.append({n} | aliases.get(n, set()))

    observed_norm = [_norm(x) for x in observed if _norm(x)]
    if not observed_norm:
        return MatchStatus.UNKNOWN

    def matches(group: set[str]) -> bool:
        for o in observed_norm:
            for e in group:
                if o == e or e in o:
                    return True
        return False

    if c.operator in {ConstraintOperator.EQ, ConstraintOperator.CONTAINS}:
        return MatchStatus.PASS if matches(expanded[0]) else MatchStatus.FAIL
    if c.operator == ConstraintOperator.ANY_OF:
        return MatchStatus.PASS if any(matches(g) for g in expanded) else MatchStatus.FAIL
    if c.operator == ConstraintOperator.ALL_OF:
        return MatchStatus.PASS if all(matches(g) for g in expanded) else MatchStatus.FAIL
    if c.operator in {ConstraintOperator.NE, ConstraintOperator.NOT_CONTAINS}:
        return MatchStatus.FAIL if matches(expanded[0]) else MatchStatus.PASS
    return MatchStatus.UNKNOWN


def _publication_type_observations(candidate: CandidatePaper) -> list[str]:
    values = list(candidate.publication_types)
    title = _norm(candidate.title)
    if re.search(r"\b(systematic review|literature review|survey|meta analysis|review)\b", title):
        values.extend(["review", "survey"])
    return values


def _identifier_values(candidate: CandidatePaper) -> list[str]:
    return [v for v in candidate.identifiers.non_empty().values()]


def _alias_map(analysis: QueryAnalysisResult, entity_type: EntityType) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for entity in analysis.entities:
        if entity.type != entity_type:
            continue
        names = {_norm(entity.canonical_name), _norm(entity.source_text)} | {_norm(a) for a in entity.aliases}
        names.discard("")
        for name in names:
            out.setdefault(name, set()).update(names)
    return out


def evaluate_constraint(
    candidate: CandidatePaper,
    c: ConstraintItem,
    analysis: QueryAnalysisResult,
    *,
    strict_field_of_study: bool = False,
) -> FilterEvidence:
    status = MatchStatus.UNKNOWN
    field: str | None = None
    observed: Any = None

    if c.type == ConstraintType.YEAR:
        field, observed = "year", candidate.year
        status = _compare_numeric(candidate.year, c)
    elif c.type == ConstraintType.MIN_CITATIONS:
        field, observed = "citation_count", candidate.citation_count
        status = _compare_numeric(candidate.citation_count, c)
    elif c.type == ConstraintType.AUTHOR:
        field, observed = "authors", candidate.authors
        status = _compare_string_list(candidate.authors, c, aliases=_alias_map(analysis, EntityType.AUTHOR))
    elif c.type == ConstraintType.VENUE:
        field, observed = "venue", candidate.venue
        status = _compare_string_list(
            [candidate.venue] if candidate.venue else None,
            c,
            aliases=_alias_map(analysis, EntityType.VENUE),
        )
    elif c.type == ConstraintType.PUBLICATION_TYPE:
        field, observed = "publication_types", candidate.publication_types
        status = _compare_string_list(_publication_type_observations(candidate), c)
    elif c.type == ConstraintType.IDENTIFIER:
        field, observed = "identifiers", candidate.identifiers.non_empty()
        status = _compare_string_list(_identifier_values(candidate), c)
    elif c.type == ConstraintType.LANGUAGE:
        field, observed = "language", candidate.language
        status = _compare_string_list([candidate.language] if candidate.language else None, c)
    elif c.type == ConstraintType.FIELD_OF_STUDY:
        field, observed = "fields_of_study", candidate.fields_of_study
        if strict_field_of_study:
            status = _compare_string_list(candidate.fields_of_study, c)
        else:
            # 不同数据库的 FoS taxonomy 差异较大，V1 默认不以此确定 FAIL。
            if candidate.fields_of_study:
                raw = _compare_string_list(candidate.fields_of_study, c)
                status = MatchStatus.PASS if raw == MatchStatus.PASS else MatchStatus.UNKNOWN
    else:
        # topic/task/method/technology/dataset/evaluation/code/data 等语义条件留给 Selector。
        status = MatchStatus.UNKNOWN

    msg = f"{c.type.value} -> {status.value}"
    return FilterEvidence(ref_id=c.id, status=status, message=msg, field=field, observed=observed)


def _evaluate_logic(
    node: LogicNode,
    leaf_eval: Callable[[str], MatchStatus],
) -> MatchStatus:
    if node.operator == "LEAF":
        assert node.ref_id is not None
        return leaf_eval(node.ref_id)
    child_values = [_evaluate_logic(child, leaf_eval) for child in node.children]
    if node.operator == "AND":
        return _tri_and(child_values)
    if node.operator == "OR":
        return _tri_or(child_values)
    if node.operator == "NOT":
        return _tri_not(child_values[0])
    return MatchStatus.UNKNOWN


def evaluate_candidate(
    candidate: CandidatePaper,
    analysis: QueryAnalysisResult,
    *,
    strict_field_of_study: bool = False,
) -> FilterDecision:
    constraints = {
        c.id: c
        for bucket in (analysis.constraints.hard, analysis.constraints.soft, analysis.constraints.exclude)
        for c in bucket
    }
    relations = {r.id: r for r in analysis.relations}
    evidence_map: dict[str, FilterEvidence] = {}

    def leaf_eval(ref_id: str) -> MatchStatus:
        if ref_id in evidence_map:
            return evidence_map[ref_id].status
        if ref_id in constraints:
            ev = evaluate_constraint(
                candidate,
                constraints[ref_id],
                analysis,
                strict_field_of_study=strict_field_of_study,
            )
        elif ref_id in relations:
            ev = FilterEvidence(
                ref_id=ref_id,
                status=MatchStatus.UNKNOWN,
                message="relation constraint deferred to Selector",
            )
        else:
            ev = FilterEvidence(
                ref_id=ref_id,
                status=MatchStatus.UNKNOWN,
                message="unknown logic reference",
            )
        evidence_map[ref_id] = ev
        return ev.status

    # Hard: 有显式 logic 就按三值逻辑，否则默认 AND。
    if analysis.logic.hard is not None:
        hard_status = _evaluate_logic(analysis.logic.hard, leaf_eval)
        referenced = _collect_refs(analysis.logic.hard)
        missing_hard = [c.id for c in analysis.constraints.hard if c.id not in referenced]
        missing_rel = [r.id for r in analysis.relations if r.level == ConstraintLevel.HARD and r.id not in referenced]
        hard_status = _tri_and([hard_status] + [leaf_eval(x) for x in missing_hard + missing_rel])
    else:
        hard_values = [leaf_eval(c.id) for c in analysis.constraints.hard]
        hard_values += [leaf_eval(r.id) for r in analysis.relations if r.level == ConstraintLevel.HARD]
        hard_status = _tri_and(hard_values)

    # Exclude: “匹配排除条件”意味着最终 FAIL。默认多个 exclude 为 OR。
    if analysis.logic.exclude is not None:
        exclude_match = _evaluate_logic(analysis.logic.exclude, leaf_eval)
        referenced = _collect_refs(analysis.logic.exclude)
        missing_ex = [c.id for c in analysis.constraints.exclude if c.id not in referenced]
        missing_rel = [r.id for r in analysis.relations if r.level == ConstraintLevel.EXCLUDE and r.id not in referenced]
        if missing_ex or missing_rel:
            exclude_match = _tri_or([exclude_match] + [leaf_eval(x) for x in missing_ex + missing_rel])
    else:
        exclude_values = [leaf_eval(c.id) for c in analysis.constraints.exclude]
        exclude_values += [leaf_eval(r.id) for r in analysis.relations if r.level == ConstraintLevel.EXCLUDE]
        exclude_match = _tri_or(exclude_values) if exclude_values else MatchStatus.FAIL

    if hard_status == MatchStatus.FAIL or exclude_match == MatchStatus.PASS:
        final = MatchStatus.FAIL
    elif hard_status == MatchStatus.PASS and exclude_match == MatchStatus.FAIL:
        final = MatchStatus.PASS
    else:
        final = MatchStatus.UNKNOWN

    # Soft 只做加分，不过滤。
    soft_statuses = [leaf_eval(c.id) for c in analysis.constraints.soft]
    soft_statuses += [leaf_eval(r.id) for r in analysis.relations if r.level == ConstraintLevel.SOFT]
    if soft_statuses:
        # UNKNOWN 不惩罚成负分，只是不加分。
        soft_score = sum(x == MatchStatus.PASS for x in soft_statuses) / len(soft_statuses)
    else:
        soft_score = 0.0

    return FilterDecision(
        status=final,
        evidence=list(evidence_map.values()),
        soft_match_score=float(soft_score),
    )


def _collect_refs(node: LogicNode) -> set[str]:
    if node.operator == "LEAF":
        return {node.ref_id} if node.ref_id else set()
    out: set[str] = set()
    for child in node.children:
        out |= _collect_refs(child)
    return out


def apply_filters(
    candidates: list[CandidatePaper],
    analysis: QueryAnalysisResult,
    *,
    strict_field_of_study: bool = False,
) -> tuple[list[CandidatePaper], list[CandidatePaper]]:
    kept: list[CandidatePaper] = []
    rejected: list[CandidatePaper] = []
    for candidate in candidates:
        decision = evaluate_candidate(candidate, analysis, strict_field_of_study=strict_field_of_study)
        candidate.hard_filter_status = decision.status
        candidate.hard_filter_evidence = decision.evidence
        candidate.soft_match_score = decision.soft_match_score
        if decision.status == MatchStatus.FAIL:
            rejected.append(candidate)
        else:
            kept.append(candidate)
    return kept, rejected


__all__ = [
    "FilterDecision",
    "evaluate_constraint",
    "evaluate_candidate",
    "apply_filters",
]
