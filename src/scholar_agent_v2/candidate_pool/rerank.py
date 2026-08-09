from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from ..query_generation.models import QueryGenerationOutput, QueryGenerationResult, QueryKind
from ..query_understanding.schema import QueryAnalysisResult, RetrievalMode

from .models import CandidatePaper


@dataclass(frozen=True)
class RerankConfig:
    # 权重是可调参数，不属于 schema contract；后续用 benchmark 调参。
    rrf_weight: float = 0.42
    lexical_weight: float = 0.28
    source_coverage_weight: float = 0.10
    route_coverage_weight: float = 0.08
    soft_match_weight: float = 0.06
    preference_weight: float = 0.06
    rrf_k: int = 60


KIND_WEIGHTS = {
    QueryKind.EXACT: 1.10,
    QueryKind.CORE: 1.00,
    QueryKind.RELATION: 0.90,
    QueryKind.ALIAS: 0.80,
    QueryKind.EXPANDED: 0.70,
}


_WORD_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]", re.I)


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return set(_WORD_RE.findall(normalized))


def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def _weighted_rrf(candidate: CandidatePaper, k: int) -> float:
    score = 0.0
    for origin in candidate.origins:
        weight = KIND_WEIGHTS.get(origin.query_kind, 0.7)
        score += weight / (k + origin.rank)
    return score


def _lexical_score(candidate: CandidatePaper, generation: QueryGenerationResult) -> float:
    title_tokens = _tokens(candidate.title)
    abstract_tokens = _tokens(candidate.abstract)
    if not title_tokens and not abstract_tokens:
        return 0.0

    query_scores = []
    for q in generation.queries:
        qt = _tokens(q.text)
        if not qt:
            continue
        title_score = _dice(qt, title_tokens)
        abstract_score = _dice(qt, abstract_tokens)
        q_score = 0.68 * title_score + 0.32 * abstract_score
        q_score *= min(1.0, KIND_WEIGHTS.get(q.kind, 0.7))
        query_scores.append(q_score)

    core_tokens = _tokens(generation.core_content)
    if core_tokens:
        query_scores.append(0.72 * _dice(core_tokens, title_tokens) + 0.28 * _dice(core_tokens, abstract_tokens))
    return max(query_scores, default=0.0)


def _preference_score(
    candidate: CandidatePaper,
    analysis: QueryAnalysisResult,
    *,
    min_year: int | None,
    max_year: int | None,
) -> float:
    # 仅作很轻的排序偏好，不承担 Hard Constraint。
    components: list[float] = []
    if analysis.properties.recent_first or analysis.intent.mode == RetrievalMode.LATEST:
        if candidate.year is None:
            components.append(0.0)
        elif min_year is not None and max_year is not None and max_year > min_year:
            # 使用候选集内部年份归一化，避免硬编码当前年份。
            components.append((candidate.year - min_year) / (max_year - min_year))
        else:
            components.append(1.0)
    if analysis.properties.central_first or analysis.intent.mode == RetrievalMode.SEMINAL:
        if candidate.citation_count is not None:
            components.append(min(1.0, math.log1p(candidate.citation_count) / math.log1p(1000)))
        else:
            components.append(0.0)
    return sum(components) / len(components) if components else 0.0


def rerank_candidates(
    candidates: list[CandidatePaper],
    analysis: QueryAnalysisResult,
    generation: QueryGenerationResult | QueryGenerationOutput,
    *,
    num_enabled_sources: int = 3,
    num_planned_routes: int | None = None,
    config: RerankConfig | None = None,
) -> list[CandidatePaper]:
    if not candidates:
        return candidates
    cfg = config or RerankConfig()
    gen = generation.result if isinstance(generation, QueryGenerationOutput) else generation
    route_den = max(1, num_planned_routes or max((c.route_count for c in candidates), default=1))

    raw_rrf = [_weighted_rrf(c, cfg.rrf_k) for c in candidates]
    max_rrf = max(raw_rrf, default=0.0) or 1.0

    years = [c.year for c in candidates if c.year is not None]
    min_year = min(years) if years else None
    max_year = max(years) if years else None

    for candidate, rrf_raw in zip(candidates, raw_rrf):
        rrf = rrf_raw / max_rrf
        lexical = _lexical_score(candidate, gen)
        source_cov = min(1.0, candidate.source_count / max(1, num_enabled_sources))
        route_cov = min(1.0, candidate.route_count / route_den)
        soft = candidate.soft_match_score
        preference = _preference_score(
            candidate,
            analysis,
            min_year=min_year,
            max_year=max_year,
        )

        score = (
            cfg.rrf_weight * rrf
            + cfg.lexical_weight * lexical
            + cfg.source_coverage_weight * source_cov
            + cfg.route_coverage_weight * route_cov
            + cfg.soft_match_weight * soft
            + cfg.preference_weight * preference
        )
        candidate.score_components = {
            "rrf": rrf,
            "lexical": lexical,
            "source_coverage": source_cov,
            "route_coverage": route_cov,
            "soft_match": soft,
            "preference": preference,
        }
        candidate.retrieval_score = float(score)

    return sorted(
        candidates,
        key=lambda c: (
            c.retrieval_score if c.retrieval_score is not None else -1.0,
            -(c.best_rank or 10**9),
        ),
        reverse=True,
    )


__all__ = ["RerankConfig", "rerank_candidates"]
