from __future__ import annotations

from collections.abc import Callable, Iterable

from ..query_understanding.schema import LogicNode
from .models import MatchStatus


LeafEvaluator = Callable[[str], MatchStatus]


def tri_and(values: Iterable[MatchStatus]) -> MatchStatus:
    values = list(values)
    if any(value == MatchStatus.FAIL for value in values):
        return MatchStatus.FAIL
    if values and all(value == MatchStatus.PASS for value in values):
        return MatchStatus.PASS
    return MatchStatus.UNKNOWN if values else MatchStatus.PASS


def tri_or(values: Iterable[MatchStatus]) -> MatchStatus:
    values = list(values)
    if any(value == MatchStatus.PASS for value in values):
        return MatchStatus.PASS
    if values and all(value == MatchStatus.FAIL for value in values):
        return MatchStatus.FAIL
    return MatchStatus.UNKNOWN if values else MatchStatus.FAIL


def tri_not(value: MatchStatus) -> MatchStatus:
    if value == MatchStatus.PASS:
        return MatchStatus.FAIL
    if value == MatchStatus.FAIL:
        return MatchStatus.PASS
    return MatchStatus.UNKNOWN


def evaluate_logic(node: LogicNode, leaf_eval: LeafEvaluator) -> MatchStatus:
    if node.operator == "LEAF":
        if node.ref_id is None:
            return MatchStatus.UNKNOWN
        return leaf_eval(node.ref_id)

    child_values = [evaluate_logic(child, leaf_eval) for child in node.children]
    if node.operator == "AND":
        return tri_and(child_values)
    if node.operator == "OR":
        return tri_or(child_values)
    if node.operator == "NOT":
        return tri_not(child_values[0]) if child_values else MatchStatus.UNKNOWN
    return MatchStatus.UNKNOWN


def collect_refs(node: LogicNode) -> set[str]:
    if node.operator == "LEAF":
        return {node.ref_id} if node.ref_id else set()

    refs: set[str] = set()
    for child in node.children:
        refs.update(collect_refs(child))
    return refs


__all__ = [
    "LeafEvaluator",
    "tri_and",
    "tri_or",
    "tri_not",
    "evaluate_logic",
    "collect_refs",
]
