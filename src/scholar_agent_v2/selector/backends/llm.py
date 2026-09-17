from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...candidate_pool.models import CandidatePaper, MatchStatus
from ...query_understanding.analyzer import LLMClient, LLMResponse, parse_json_payload
from ..models import (
    BackendPaperJudgement,
    CriterionJudgement,
    SelectorBackendMetrics,
    SelectorCriterion,
)
from ..prompt import build_selector_messages, build_selector_repair_messages

logger = logging.getLogger(__name__)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _LLMCriterionPayload(_StrictModel):
    criterion_id: str
    status: Literal["pass", "fail", "unknown"]
    evidence: str | None = None
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class _LLMPaperPayload(_StrictModel):
    candidate_id: str
    summary: str | None = None
    judgements: list[_LLMCriterionPayload]


class _LLMBatchPayload(_StrictModel):
    papers: list[_LLMPaperPayload]


@dataclass(frozen=True)
class LLMCriteriaBackendConfig:
    batch_size: int = 5
    max_concurrency: int = 2
    temperature: float = 0.0
    use_structured_output: bool = False

    enable_repair: bool = True
    max_transport_retries: int = 0
    retry_backoff_seconds: float = 0.5

    fail_open: bool = True
    backend_name: str = "llm_criteria"


@dataclass
class _BatchExecution:
    judgements: list[BackendPaperJudgement]
    metrics: SelectorBackendMetrics
    warnings: list[str]


class LLMCriteriaBackend:
    """Provider-agnostic batched criterion judge; works with the project's LLMClient protocol."""

    def __init__(self, client: LLMClient, *, config: LLMCriteriaBackendConfig | None = None) -> None:
        self.client = client
        self.config = config or LLMCriteriaBackendConfig()
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.config.max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")

    async def judge(
        self,
        candidates: Sequence[CandidatePaper],
        criteria: Sequence[SelectorCriterion],
    ) -> tuple[list[BackendPaperJudgement], SelectorBackendMetrics]:
        if not candidates:
            return [], SelectorBackendMetrics()
        if not criteria:
            return [
                BackendPaperJudgement(
                    candidate_id=candidate.candidate_id,
                    judgements=[],
                    backend_name=self.config.backend_name,
                )
                for candidate in candidates
            ], SelectorBackendMetrics()

        batches = [
            list(candidates[index : index + self.config.batch_size])
            for index in range(0, len(candidates), self.config.batch_size)
        ]
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def run_batch(batch: list[CandidatePaper]) -> _BatchExecution:
            async with semaphore:
                return await self._judge_batch(batch, criteria)

        executions = await asyncio.gather(*(run_batch(batch) for batch in batches))
        all_judgements: list[BackendPaperJudgement] = []
        metrics = SelectorBackendMetrics()
        for execution in executions:
            all_judgements.extend(execution.judgements)
            self._merge_metrics(metrics, execution.metrics)
        return all_judgements, metrics

    @staticmethod
    def _merge_metrics(target: SelectorBackendMetrics, source: SelectorBackendMetrics) -> None:
        target.api_calls += source.api_calls
        target.repair_calls += source.repair_calls
        target.prompt_tokens += source.prompt_tokens
        target.completion_tokens += source.completion_tokens
        target.total_tokens += source.total_tokens
        target.latency_ms += source.latency_ms
        target.failed_batches += source.failed_batches
        for name in source.model_names:
            if name not in target.model_names:
                target.model_names.append(name)

    async def _judge_batch(
        self,
        batch: Sequence[CandidatePaper],
        criteria: Sequence[SelectorCriterion],
    ) -> _BatchExecution:
        metrics = SelectorBackendMetrics()
        messages = build_selector_messages(batch, criteria)
        response_schema: Mapping[str, Any] | None = (
            _LLMBatchPayload.model_json_schema() if self.config.use_structured_output else None
        )

        try:
            response = await self._call_llm(messages, metrics, response_schema=response_schema, is_repair=False)
        except Exception as call_error:
            # Repair is for malformed semantic output, not for transport/API failures.
            return self._handle_batch_failure(batch, criteria, metrics, call_error)

        try:
            payload = self._parse_payload(response)
        except Exception as primary_error:
            if not self.config.enable_repair:
                return self._handle_batch_failure(batch, criteria, metrics, primary_error)

            raw_output = (
                response.content
                if isinstance(response.content, str)
                else json.dumps(response.content, ensure_ascii=False)
            )
            repair_messages = build_selector_repair_messages(
                raw_output=raw_output,
                candidates=batch,
                criteria=criteria,
                error=str(primary_error),
            )
            try:
                repaired = await self._call_llm(
                    repair_messages,
                    metrics,
                    response_schema=response_schema,
                    is_repair=True,
                )
                payload = self._parse_payload(repaired)
            except Exception as repair_error:
                return self._handle_batch_failure(batch, criteria, metrics, repair_error)

        judgements, warnings = self._normalize_payload(batch, criteria, payload, metrics.model_names[-1] if metrics.model_names else None)
        return _BatchExecution(judgements=judgements, metrics=metrics, warnings=warnings)

    async def _call_llm(
        self,
        messages: Sequence[Mapping[str, str]],
        metrics: SelectorBackendMetrics,
        *,
        response_schema: Mapping[str, Any] | None,
        is_repair: bool,
    ) -> LLMResponse:
        attempts = 1 + max(0, self.config.max_transport_retries)
        last_error: Exception | None = None

        for attempt in range(attempts):
            started = time.perf_counter()
            metrics.api_calls += 1
            if is_repair:
                metrics.repair_calls += 1
            try:
                response = await self.client.complete(
                    messages=messages,
                    temperature=self.config.temperature,
                    response_schema=response_schema,
                )
            except Exception as exc:
                metrics.latency_ms += (time.perf_counter() - started) * 1000.0
                last_error = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(self.config.retry_backoff_seconds * (2**attempt))
                    continue
                raise

            metrics.latency_ms += (time.perf_counter() - started) * 1000.0
            metrics.prompt_tokens += response.usage.prompt_tokens or 0
            metrics.completion_tokens += response.usage.completion_tokens or 0
            metrics.total_tokens += response.usage.total_tokens or 0
            if response.model and response.model not in metrics.model_names:
                metrics.model_names.append(response.model)
            return response

        raise RuntimeError(f"LLM selector call failed: {last_error}")

    @staticmethod
    def _parse_payload(response: LLMResponse) -> _LLMBatchPayload:
        payload = parse_json_payload(response.content)
        try:
            return _LLMBatchPayload.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Selector LLM payload schema invalid: {exc}") from exc

    def _normalize_payload(
        self,
        batch: Sequence[CandidatePaper],
        criteria: Sequence[SelectorCriterion],
        payload: _LLMBatchPayload,
        model_name: str | None,
    ) -> tuple[list[BackendPaperJudgement], list[str]]:
        expected_candidates = {candidate.candidate_id for candidate in batch}
        expected_criteria = {criterion.criterion_id for criterion in criteria}
        warnings: list[str] = []

        paper_map: dict[str, _LLMPaperPayload] = {}
        for paper in payload.papers:
            if paper.candidate_id not in expected_candidates:
                warnings.append(f"unexpected candidate_id from LLM: {paper.candidate_id}")
                continue
            if paper.candidate_id in paper_map:
                warnings.append(f"duplicate candidate_id from LLM: {paper.candidate_id}")
                continue
            paper_map[paper.candidate_id] = paper

        results: list[BackendPaperJudgement] = []
        for candidate in batch:
            paper = paper_map.get(candidate.candidate_id)
            if paper is None:
                warnings.append(f"missing candidate_id from LLM: {candidate.candidate_id}")
                results.append(self._unknown_paper(candidate, criteria, model_name, "missing paper judgement"))
                continue

            judgement_map: dict[str, _LLMCriterionPayload] = {}
            local_warnings: list[str] = []
            for judgement in paper.judgements:
                if judgement.criterion_id not in expected_criteria:
                    local_warnings.append(f"unexpected criterion_id: {judgement.criterion_id}")
                    continue
                if judgement.criterion_id in judgement_map:
                    local_warnings.append(f"duplicate criterion_id: {judgement.criterion_id}")
                    continue
                judgement_map[judgement.criterion_id] = judgement

            normalized: list[CriterionJudgement] = []
            for criterion in criteria:
                raw = judgement_map.get(criterion.criterion_id)
                if raw is None:
                    local_warnings.append(f"missing criterion_id: {criterion.criterion_id}")
                    normalized.append(
                        CriterionJudgement(
                            criterion_id=criterion.criterion_id,
                            status=MatchStatus.UNKNOWN,
                            reason="LLM omitted this criterion.",
                            backend_name=self.config.backend_name,
                            model_name=model_name,
                        )
                    )
                    continue
                normalized.append(
                    CriterionJudgement(
                        criterion_id=criterion.criterion_id,
                        status=MatchStatus(raw.status),
                        evidence=raw.evidence,
                        reason=raw.reason,
                        confidence=raw.confidence,
                        backend_name=self.config.backend_name,
                        model_name=model_name,
                    )
                )

            results.append(
                BackendPaperJudgement(
                    candidate_id=candidate.candidate_id,
                    judgements=normalized,
                    summary=paper.summary,
                    backend_name=self.config.backend_name,
                    model_name=model_name,
                    warnings=local_warnings,
                )
            )

        return results, warnings

    def _unknown_paper(
        self,
        candidate: CandidatePaper,
        criteria: Sequence[SelectorCriterion],
        model_name: str | None,
        reason: str,
    ) -> BackendPaperJudgement:
        return BackendPaperJudgement(
            candidate_id=candidate.candidate_id,
            judgements=[
                CriterionJudgement(
                    criterion_id=criterion.criterion_id,
                    status=MatchStatus.UNKNOWN,
                    reason=reason,
                    backend_name=self.config.backend_name,
                    model_name=model_name,
                )
                for criterion in criteria
            ],
            backend_name=self.config.backend_name,
            model_name=model_name,
            warnings=[reason],
        )

    def _handle_batch_failure(
        self,
        batch: Sequence[CandidatePaper],
        criteria: Sequence[SelectorCriterion],
        metrics: SelectorBackendMetrics,
        error: Exception,
    ) -> _BatchExecution:
        metrics.failed_batches += 1
        if not self.config.fail_open:
            raise error
        message = f"selector batch failed; downgraded to UNKNOWN: {type(error).__name__}: {error}"
        logger.warning(message)
        model_name = metrics.model_names[-1] if metrics.model_names else None
        return _BatchExecution(
            judgements=[self._unknown_paper(candidate, criteria, model_name, message) for candidate in batch],
            metrics=metrics,
            warnings=[message],
        )


QwenCriteriaBackend = LLMCriteriaBackend
QwenCriteriaBackendConfig = LLMCriteriaBackendConfig


__all__ = [
    "LLMCriteriaBackendConfig",
    "LLMCriteriaBackend",
    "QwenCriteriaBackendConfig",
    "QwenCriteriaBackend",
]
