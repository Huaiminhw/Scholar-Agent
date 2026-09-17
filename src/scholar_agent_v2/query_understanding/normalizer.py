"""Deterministic, semantics-preserving normalization for query understanding.

The LLM is responsible for understanding the user's intent.  This module is
deliberately narrower: it canonicalizes harmless representation differences
before business validation and downstream planning.  It must never guess a
constraint, move an item between hard/soft/exclude, reverse a relation, or
rewrite Boolean logic.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from .schema import (
    ConstraintItem,
    ConstraintOperator,
    ConstraintType,
    ConstraintValue,
    ParseStatus,
    QueryAnalysisResult,
    QueryEntity,
)


_SPACE_RE = re.compile(r"\s+")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.\-]+/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)

def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace while preserving spelling and case."""

    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def _text_key(value: str) -> str:
    return normalize_text(value).casefold()


def dedupe_strings(
    values: Iterable[str],
    *,
    exclude_keys: Iterable[str] = (),
) -> list[str]:
    """Stable, Unicode-aware and case-insensitive string deduplication."""

    excluded = {_text_key(value) for value in exclude_keys if normalize_text(value)}
    output: list[str] = []
    seen: set[str] = set(excluded)
    for value in values:
        cleaned = normalize_text(value)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def normalize_identifier(value: str) -> str:
    """Canonicalize recognized DOI/arXiv forms; leave unknown IDs untouched."""

    cleaned = normalize_text(value)
    lowered = cleaned.casefold()

    doi = lowered
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :].strip()
            break
    if _DOI_RE.fullmatch(doi):
        return doi.rstrip(".,;)")

    arxiv = lowered
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv:"):
        if arxiv.startswith(prefix):
            arxiv = arxiv[len(prefix) :].strip()
            break
    arxiv = arxiv.rstrip("/.,;)")
    if _ARXIV_RE.fullmatch(arxiv):
        return _ARXIV_VERSION_RE.sub("", arxiv)

    return cleaned


def _normalize_numeric(value: ConstraintValue) -> ConstraintValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        cleaned = normalize_text(value)
        compact = cleaned.replace(",", "")
        if _INTEGER_RE.fullmatch(compact):
            return int(compact)
        return cleaned
    return value


def _normalize_constraint_value(
    item: ConstraintItem,
    value: ConstraintValue,
) -> ConstraintValue:
    if isinstance(value, str):
        value = normalize_text(value)
    if item.type in {ConstraintType.YEAR, ConstraintType.MIN_CITATIONS}:
        return _normalize_numeric(value)
    if item.type == ConstraintType.IDENTIFIER and isinstance(value, str):
        return normalize_identifier(value)
    return value


def _value_key(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        return ("str", _text_key(value))
    return (type(value).__name__, repr(value))


def _dedupe_values(values: Iterable[ConstraintValue]) -> list[ConstraintValue]:
    output: list[ConstraintValue] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = _value_key(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _normalize_constraint(item: ConstraintItem) -> None:
    item.source_text = normalize_text(item.source_text)
    if item.normalized_text is not None:
        normalized_text = normalize_text(item.normalized_text)
        if item.type == ConstraintType.IDENTIFIER and normalized_text:
            normalized_text = normalize_identifier(normalized_text)
        item.normalized_text = normalized_text or None

    if item.value is not None:
        item.value = _normalize_constraint_value(item, item.value)
    item.values = _dedupe_values(
        _normalize_constraint_value(item, value) for value in item.values
    )

    # Repair only redundant shape noise.  Conflicting value/values remain so
    # validator.py can expose the ambiguity instead of silently choosing one.
    if item.value is not None and item.values:
        value_key = _value_key(item.value)
        value_keys = {_value_key(value) for value in item.values}
        if item.operator in {
            ConstraintOperator.ANY_OF,
            ConstraintOperator.ALL_OF,
            ConstraintOperator.BETWEEN,
        }:
            if value_key in value_keys:
                item.value = None
        elif value_keys == {value_key}:
            item.values = []


def _normalize_entity(entity: QueryEntity) -> None:
    entity.canonical_name = normalize_text(entity.canonical_name)
    entity.source_text = normalize_text(entity.source_text)
    entity.aliases = dedupe_strings(
        entity.aliases,
        exclude_keys=(entity.canonical_name,),
    )


def _merge_duplicate_entities(entities: Iterable[QueryEntity]) -> list[QueryEntity]:
    """Merge exact semantic duplicates without inventing cross-entity aliases."""

    output: list[QueryEntity] = []
    index_by_key: dict[tuple[str, str], int] = {}
    for entity in entities:
        _normalize_entity(entity)
        key = (entity.type.value, _text_key(entity.canonical_name))
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(output)
            output.append(entity)
            continue

        existing = output[existing_index]
        existing.aliases = dedupe_strings(
            [*existing.aliases, *entity.aliases],
            exclude_keys=(existing.canonical_name,),
        )
        if not existing.source_text and entity.source_text:
            existing.source_text = entity.source_text
    return output


def normalize_query_analysis(
    result: QueryAnalysisResult,
    *,
    original_query: str | None = None,
) -> QueryAnalysisResult:
    """Return a deep-normalized copy of a parsed query-analysis result.

    Safe transformations:
    - restore the authoritative user query;
    - normalize Unicode/whitespace in generated descriptive fields;
    - canonicalize unambiguous year/citation integers and DOI/arXiv IDs;
    - deduplicate values, aliases, entities and other set-like string lists;
    - synchronize ``success`` to ``needs_clarification`` when the LLM already
      set ``clarification.needed=true``.

    Intentionally untouched:
    - query type and retrieval intent;
    - hard/soft/exclude membership and item IDs;
    - relation direction/level;
    - Boolean logic structure and references;
    - missing or contradictory semantic information.
    """

    normalized = result.model_copy(deep=True)

    if original_query is not None:
        # The caller owns this value.  Do not NFKC/case-fold/collapse it.
        normalized.original_query = original_query

    normalized.intent.goal = normalize_text(normalized.intent.goal)
    normalized.content = normalize_text(normalized.content)

    for bucket_name in ("hard", "soft", "exclude"):
        for item in getattr(normalized.constraints, bucket_name):
            _normalize_constraint(item)

    for relation in normalized.relations:
        relation.subject = normalize_text(relation.subject)
        relation.predicate = normalize_text(relation.predicate)
        relation.object = normalize_text(relation.object)
        relation.source_text = normalize_text(relation.source_text)

    if normalized.properties.specific_paper_name is not None:
        paper_name = normalize_text(normalized.properties.specific_paper_name)
        normalized.properties.specific_paper_name = paper_name or None

    normalized.entities = _merge_duplicate_entities(normalized.entities)

    if normalized.domains.main_field is not None:
        main_field = normalize_text(normalized.domains.main_field)
        normalized.domains.main_field = main_field or None
    normalized.domains.secondary_fields = dedupe_strings(
        normalized.domains.secondary_fields,
        exclude_keys=(normalized.domains.main_field or "",),
    )

    normalized.clarification.unresolved_ambiguities = dedupe_strings(
        normalized.clarification.unresolved_ambiguities
    )
    normalized.clarification.questions = dedupe_strings(
        normalized.clarification.questions
    )
    normalized.result_requirements.required_fields = dedupe_strings(
        normalized.result_requirements.required_fields
    )
    normalized.warnings = dedupe_strings(normalized.warnings)

    if (
        normalized.clarification.needed
        and normalized.status == ParseStatus.SUCCESS
    ):
        normalized.status = ParseStatus.NEEDS_CLARIFICATION

    return normalized


# Backwards-compatible name used by earlier code and external callers.
safe_normalize = normalize_query_analysis


__all__ = [
    "normalize_text",
    "normalize_identifier",
    "dedupe_strings",
    "normalize_query_analysis",
    "safe_normalize",
]
