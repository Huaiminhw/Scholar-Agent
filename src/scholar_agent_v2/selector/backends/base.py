from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ...candidate_pool.models import CandidatePaper
from ...query_understanding.schema import QueryAnalysisResult
from ..models import BackendPaperJudgement, SelectorBackendMetrics, SelectorCriterion


class CriteriaJudgeBackend(Protocol):
    async def judge(
        self,
        candidates: Sequence[CandidatePaper],
        criteria: Sequence[SelectorCriterion],
    ) -> tuple[list[BackendPaperJudgement], SelectorBackendMetrics]:
        ...


class PaperPriorScorer(Protocol):
    def score(
        self,
        analysis: QueryAnalysisResult,
        candidates: Sequence[CandidatePaper],
    ) -> tuple[dict[str, float], float]:
        """Return candidate_id -> score and latency_ms."""
        ...


__all__ = ["CriteriaJudgeBackend", "PaperPriorScorer"]
