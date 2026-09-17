from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..query_understanding.schema import (
    ConstraintItem,
    ConstraintLevel,
    ConstraintOperator,
    ConstraintType,
    QueryAnalysisResult,
    RelationConstraint,
)
from .models import CriterionKind, SelectorCriterion


DETERMINISTIC_METADATA_TYPES: frozenset[ConstraintType] = frozenset(
    {
        ConstraintType.AUTHOR,
        ConstraintType.VENUE,
        ConstraintType.YEAR,
        ConstraintType.PUBLICATION_TYPE,
        ConstraintType.MIN_CITATIONS,
        ConstraintType.FIELD_OF_STUDY,
        ConstraintType.LANGUAGE,
        ConstraintType.IDENTIFIER,
    }
)

SEMANTIC_CONSTRAINT_TYPES: frozenset[ConstraintType] = frozenset(
    set(ConstraintType) - DETERMINISTIC_METADATA_TYPES
)


@dataclass(frozen=True)
class CriteriaBuilderConfig:
    # fallback mirrors Ai2's safe behavior: use content as a criterion only when
    # Query Understanding did not already produce any hard semantic criterion/relation.
    content_criterion_mode: Literal["fallback", "always", "never"] = "fallback"
    include_soft: bool = True
    include_exclude: bool = True


def _constraint_values(item: ConstraintItem) -> list[str | int | float | bool]:
    if item.values:
        return list(item.values)
    if item.value is not None:
        return [item.value]
    return []


def _operator_phrase(operator: ConstraintOperator) -> str:
    return {
        ConstraintOperator.EQ: "matches",
        ConstraintOperator.NE: "does not match",
        ConstraintOperator.GT: "is greater than",
        ConstraintOperator.GTE: "is at least",
        ConstraintOperator.LT: "is less than",
        ConstraintOperator.LTE: "is at most",
        ConstraintOperator.ANY_OF: "matches at least one of",
        ConstraintOperator.ALL_OF: "matches all of",
        ConstraintOperator.CONTAINS: "contains or substantively demonstrates",
        ConstraintOperator.NOT_CONTAINS: "does not contain or demonstrate",
        ConstraintOperator.BETWEEN: "falls between",
    }.get(operator, "matches")


def _constraint_description(item: ConstraintItem) -> str:
    values = _constraint_values(item)
    rendered = ", ".join(str(value) for value in values) if values else "the requested property"
    return (
        f"The paper's {item.type.value} {_operator_phrase(item.operator)}: {rendered}. "
        "Judge the scientific substance, not mere keyword co-occurrence."
    )


def _relation_description(relation: RelationConstraint) -> str:
    return (
        "The paper must substantively establish the directed relation: "
        f'"{relation.subject}" --[{relation.predicate}]--> "{relation.object}". '
        "The subject, predicate, object, and direction all matter."
    )


def _content_description(content: str) -> str:
    return (
        "The paper's main scientific contribution, studied problem, or evaluation must be substantially about: "
        f'"{content}". Superficial keyword overlap is insufficient.'
    )


def build_selector_criteria(
    analysis: QueryAnalysisResult,
    *,
    config: CriteriaBuilderConfig | None = None,
) -> list[SelectorCriterion]:
    config = config or CriteriaBuilderConfig()
    criteria: list[SelectorCriterion] = []

    buckets: list[tuple[ConstraintLevel, list[ConstraintItem]]] = [
        (ConstraintLevel.HARD, analysis.constraints.hard),
    ]
    if config.include_soft:
        buckets.append((ConstraintLevel.SOFT, analysis.constraints.soft))
    if config.include_exclude:
        buckets.append((ConstraintLevel.EXCLUDE, analysis.constraints.exclude))

    for level, items in buckets:
        for item in items:
            if item.type in DETERMINISTIC_METADATA_TYPES:
                continue
            criteria.append(
                SelectorCriterion(
                    criterion_id=item.id,
                    kind=CriterionKind.CONSTRAINT,
                    level=level,
                    description=_constraint_description(item),
                    source_text=item.source_text,
                    constraint_type=item.type,
                    operator=item.operator,
                    expected_values=_constraint_values(item),
                )
            )

    for relation in analysis.relations:
        if relation.level == ConstraintLevel.SOFT and not config.include_soft:
            continue
        if relation.level == ConstraintLevel.EXCLUDE and not config.include_exclude:
            continue
        criteria.append(
            SelectorCriterion(
                criterion_id=relation.id,
                kind=CriterionKind.RELATION,
                level=relation.level,
                description=_relation_description(relation),
                source_text=relation.source_text,
                subject=relation.subject,
                predicate=relation.predicate,
                object=relation.object,
            )
        )


    has_hard_semantic = any(criterion.level == ConstraintLevel.HARD for criterion in criteria)
    should_add_content = (
        config.content_criterion_mode == "always"
        or (config.content_criterion_mode == "fallback" and not has_hard_semantic)
    )
    if should_add_content and analysis.content.strip():
        criteria.insert(
            0,
            SelectorCriterion(
                criterion_id="Q_CONTENT",
                kind=CriterionKind.CONTENT,
                level=ConstraintLevel.HARD,
                description=_content_description(analysis.content.strip()),
                source_text=analysis.content.strip(),
                synthetic=True,
            ),
        )

    seen: set[str] = set()
    deduped: list[SelectorCriterion] = []
    for criterion in criteria:
        if criterion.criterion_id in seen:
            continue
        seen.add(criterion.criterion_id)
        deduped.append(criterion)
    return deduped


__all__ = [
    "DETERMINISTIC_METADATA_TYPES",
    "SEMANTIC_CONSTRAINT_TYPES",
    "CriteriaBuilderConfig",
    "build_selector_criteria",
]
