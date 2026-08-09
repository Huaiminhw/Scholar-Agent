from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..models import RouteExecutionResult, RouteSpec, SearchSource
from .route_features import RouteSemanticFeatures
from .semantic_scorer import SemanticScorer


def aggregate_source_yield(
    features: dict[str, RouteSemanticFeatures],
    executions: Iterable[RouteExecutionResult],
) -> dict[SearchSource, float]:
    """Estimate useful semantic return per retrieval API call for each source.

    semantic_gain already applies shared credit to duplicate clusters, therefore
    repeatedly retrieving the same papers cannot earn full reward multiple times.
    """

    gain_by_source: dict[SearchSource, float] = defaultdict(float)
    calls_by_source: dict[SearchSource, int] = defaultdict(int)

    for execution in executions:
        f = features.get(execution.route_id)
        if f is not None:
            gain_by_source[execution.source] += max(0.0, f.semantic_gain)
        calls_by_source[execution.source] += max(1, execution.api_calls)

    return {
        source: gain_by_source[source] / max(1, calls_by_source[source])
        for source in set(gain_by_source) | set(calls_by_source)
    }


def route_alignment(
    semantic_query: str,
    route: RouteSpec,
    scorer: SemanticScorer,
) -> float:
    if not route.query_text.strip():
        # FILTER_ONLY route has no semantic query; do not punish it as irrelevant.
        return 1.0
    return max(0.0, scorer.similarity(semantic_query, route.query_text))


def expected_route_gain(
    semantic_query: str,
    route: RouteSpec,
    source_yield: dict[SearchSource, float],
    scorer: SemanticScorer,
) -> float:
    """Training-free estimate for an unexecuted route.

    No hand-tuned weighted sum:
        expected gain = query-route semantic alignment × observed source yield.
    """

    return route_alignment(semantic_query, route, scorer) * max(
        0.0, source_yield.get(route.source, 0.0)
    )


__all__ = [
    "aggregate_source_yield",
    "route_alignment",
    "expected_route_gain",
]
