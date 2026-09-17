from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal, Sequence

from ..candidate_pool.models import CandidatePaper, MatchStatus
from ..query_understanding.schema import QueryAnalysisResult
from .backends.base import CriteriaJudgeBackend, PaperPriorScorer
from .criteria import CriteriaBuilderConfig, build_selector_criteria
from .evaluator import evaluate_paper_selection
from .evidence import verify_evidence
from .models import (
    BackendPaperJudgement,
    CriterionJudgement,
    SelectorMetrics,
    SelectorOutput,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectorConfig:
    max_candidates: int = 40
    content_criterion_mode: Literal["fallback", "always", "never"] = "fallback"
    include_soft: bool = True
    include_exclude: bool = True

    require_verified_evidence_for_decisive: bool = True
    min_evidence_normalized_chars: int = 8

    strict_field_of_study: bool = False
    fail_open_prior: bool = True


class CandidateSelector:
    def __init__(
        self,
        backend: CriteriaJudgeBackend,
        *,
        prior_scorer: PaperPriorScorer | None = None,
        config: SelectorConfig | None = None,
    ) -> None:
        self.backend = backend
        self.prior_scorer = prior_scorer
        self.config = config or SelectorConfig()
        if self.config.max_candidates <= 0:
            raise ValueError("max_candidates must be > 0")

    async def select(
        self,
        analysis: QueryAnalysisResult,
        candidates: Sequence[CandidatePaper],
    ) -> SelectorOutput:
        started = time.perf_counter()
        candidates = list(candidates[: self.config.max_candidates])
        criteria = build_selector_criteria(
            analysis,
            config=CriteriaBuilderConfig(
                content_criterion_mode=self.config.content_criterion_mode,
                include_soft=self.config.include_soft,
                include_exclude=self.config.include_exclude,
            ),
        )

        metrics = SelectorMetrics(
            candidates_seen=len(candidates),
            criteria_count=len(criteria),
        )
        if not candidates:
            metrics.total_latency_ms = (time.perf_counter() - started) * 1000.0
            return SelectorOutput(metrics=metrics)

        backend_task = asyncio.create_task(self.backend.judge(candidates, criteria))
        prior_task = None
        if self.prior_scorer is not None:
            prior_task = asyncio.create_task(
                asyncio.to_thread(self.prior_scorer.score, analysis, candidates)
            )

        backend_judgements, backend_metrics = await backend_task
        prior_scores: dict[str, float] = {}
        warnings: list[str] = []

        if prior_task is not None:
            try:
                prior_scores, prior_latency_ms = await prior_task
                metrics.prior_latency_ms = prior_latency_ms
            except Exception as exc:
                if not self.config.fail_open_prior:
                    raise
                warning = f"prior scorer failed; continuing without prior: {type(exc).__name__}: {exc}"
                logger.warning(warning)
                warnings.append(warning)

        metrics.llm_api_calls = backend_metrics.api_calls
        metrics.llm_repair_calls = backend_metrics.repair_calls
        metrics.llm_prompt_tokens = backend_metrics.prompt_tokens
        metrics.llm_completion_tokens = backend_metrics.completion_tokens
        metrics.llm_total_tokens = backend_metrics.total_tokens
        metrics.llm_latency_ms = backend_metrics.latency_ms
        metrics.llm_failed_batches = backend_metrics.failed_batches
        metrics.llm_models = list(backend_metrics.model_names)

        backend_map = {item.candidate_id: item for item in backend_judgements}
        results = []

        for candidate in candidates:
            raw = backend_map.get(candidate.candidate_id)
            if raw is None:
                raw = BackendPaperJudgement(
                    candidate_id=candidate.candidate_id,
                    judgements=[
                        CriterionJudgement(
                            criterion_id=criterion.criterion_id,
                            status=MatchStatus.UNKNOWN,
                            reason="backend omitted paper",
                            warnings=["backend omitted paper"],
                        )
                        for criterion in criteria
                    ],
                    backend_name="missing",
                    warnings=["backend omitted paper"],
                )

            verified = [
                self._verify_and_sanitize_judgement(candidate, judgement)
                for judgement in raw.judgements
            ]
            result = evaluate_paper_selection(
                candidate,
                analysis,
                criteria,
                verified,
                prior_score=prior_scores.get(candidate.candidate_id),
                summary=raw.summary,
                strict_field_of_study=self.config.strict_field_of_study,
            )
            result.warnings.extend(raw.warnings)
            results.append(result)

        selected_ids = [result.candidate_id for result in results if result.final_status == MatchStatus.PASS]
        uncertain_ids = [result.candidate_id for result in results if result.final_status == MatchStatus.UNKNOWN]
        rejected_ids = [result.candidate_id for result in results if result.final_status == MatchStatus.FAIL]

        metrics.pass_count = len(selected_ids)
        metrics.unknown_count = len(uncertain_ids)
        metrics.fail_count = len(rejected_ids)
        metrics.total_latency_ms = (time.perf_counter() - started) * 1000.0

        return SelectorOutput(
            results=results,
            selected_ids=selected_ids,
            uncertain_ids=uncertain_ids,
            rejected_ids=rejected_ids,
            metrics=metrics,
            warnings=warnings,
        )

    def _verify_and_sanitize_judgement(
        self,
        candidate: CandidatePaper,
        judgement: CriterionJudgement,
    ) -> CriterionJudgement:
        verification = verify_evidence(
            candidate,
            judgement.evidence,
            min_normalized_chars=self.config.min_evidence_normalized_chars,
        )
        update = {
            "evidence_verified": verification.verified,
            "evidence_source": verification.source,
        }
        warnings = list(judgement.warnings)

        if (
            self.config.require_verified_evidence_for_decisive
            and judgement.status in {MatchStatus.PASS, MatchStatus.FAIL}
            and not verification.verified
        ):
            warnings.append(
                f"{judgement.criterion_id}: decisive judgement downgraded to UNKNOWN because evidence was not verified"
            )
            update["status"] = MatchStatus.UNKNOWN

        update["warnings"] = warnings
        return judgement.model_copy(update=update)


__all__ = ["SelectorConfig", "CandidateSelector"]
