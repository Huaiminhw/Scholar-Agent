from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ...candidate_pool.dedup import cluster_hits
from ..models import RetrievalHit, RouteExecutionResult, SearchSource
from .semantic_scorer import SemanticScorer


@dataclass(frozen=True)
class RouteSemanticFeatures:
    route_id: str
    source: SearchSource
    hit_count: int
    avg_sim: float
    max_sim: float
    var_sim: float
    overall_sim: float
    cross_retriever_sim: float
    unique_credit: float
    semantic_gain: float


def _safe_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def compute_probe_features(
    query_text: str,
    executions: Iterable[RouteExecutionResult],
    scorer: SemanticScorer,
) -> dict[str, RouteSemanticFeatures]:
    """Compute LTRR-inspired post-retrieval features for executed probe routes.

    Decision-oriented fields:
    - avg_sim: average query-paper semantic relevance.
    - semantic_gain: relevance-weighted de-duplicated route contribution.

    Diagnostics:
    - max_sim / var_sim / overall_sim / cross_retriever_sim.
    """

    executions = list(executions)
    query_emb = scorer.encode([query_text])[0]

    route_sims: dict[str, np.ndarray] = {}
    route_centroids: dict[str, np.ndarray] = {}
    route_source: dict[str, SearchSource] = {}
    route_hit_count: dict[str, int] = {}

    for execution in executions:
        route_id = execution.route_id
        route_source[route_id] = execution.source
        route_hit_count[route_id] = len(execution.hits)

        texts = [scorer.paper_text(h.title, h.abstract) for h in execution.hits]
        texts = [t for t in texts if t]
        if not texts:
            route_sims[route_id] = np.empty((0,), dtype=np.float32)
            route_centroids[route_id] = np.zeros_like(query_emb)
            continue

        paper_embs = scorer.encode(texts)
        sims = paper_embs @ query_emb
        centroid = _safe_normalize(paper_embs.mean(axis=0))
        route_sims[route_id] = np.asarray(sims, dtype=np.float32)
        route_centroids[route_id] = centroid

    # Cross-retriever similarity: mean centroid similarity to other executed routes.
    cross: dict[str, float] = {}
    for route_id, centroid in route_centroids.items():
        others = [
            float(np.dot(centroid, other))
            for other_id, other in route_centroids.items()
            if other_id != route_id and np.any(other)
        ]
        cross[route_id] = float(np.mean(others)) if others else 0.0

    # Shared-credit de-duplicated semantic gain.
    # One duplicate paper cluster contributes only one unit of total credit across routes.
    all_hits: list[RetrievalHit] = [h for ex in executions for h in ex.hits]
    unique_credit: dict[str, float] = {ex.route_id: 0.0 for ex in executions}
    semantic_gain: dict[str, float] = {ex.route_id: 0.0 for ex in executions}

    for cluster in cluster_hits(all_hits):
        route_ids = sorted({h.origin.route_id for h in cluster})
        if not route_ids:
            continue

        # Representative with richest text; avoids encoding every duplicate again.
        rep = max(cluster, key=lambda h: len((h.title or "")) + len((h.abstract or "")))
        text = scorer.paper_text(rep.title, rep.abstract)
        sim = max(0.0, scorer.similarity(query_text, text)) if text else 0.0
        credit = 1.0 / len(route_ids)

        for route_id in route_ids:
            unique_credit[route_id] = unique_credit.get(route_id, 0.0) + credit
            semantic_gain[route_id] = semantic_gain.get(route_id, 0.0) + sim * credit

    out: dict[str, RouteSemanticFeatures] = {}
    for execution in executions:
        route_id = execution.route_id
        sims = route_sims[route_id]
        centroid = route_centroids[route_id]
        out[route_id] = RouteSemanticFeatures(
            route_id=route_id,
            source=route_source[route_id],
            hit_count=route_hit_count[route_id],
            avg_sim=float(np.mean(sims)) if len(sims) else 0.0,
            max_sim=float(np.max(sims)) if len(sims) else 0.0,
            var_sim=float(np.var(sims)) if len(sims) else 0.0,
            overall_sim=float(np.dot(query_emb, centroid)) if np.any(centroid) else 0.0,
            cross_retriever_sim=cross.get(route_id, 0.0),
            unique_credit=unique_credit.get(route_id, 0.0),
            semantic_gain=semantic_gain.get(route_id, 0.0),
        )
    return out


__all__ = ["RouteSemanticFeatures", "compute_probe_features"]
