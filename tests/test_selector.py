from __future__ import annotations

import asyncio
from typing import Sequence

from scholar_agent_v2.candidate_pool.models import CandidatePaper, MatchStatus
from scholar_agent_v2.query_understanding.schema import (
    QueryAnalysisResult,
    QueryIntent,
    QueryType,
    RetrievalMode,
    SearchTask,
)
from scholar_agent_v2.selector.models import (
    BackendPaperJudgement,
    CriterionJudgement,
    SelectorBackendMetrics,
    SelectorCriterion,
)
from scholar_agent_v2.selector.selector import CandidateSelector


class RuleBackend:
    async def judge(
        self,
        candidates: Sequence[CandidatePaper],
        criteria: Sequence[SelectorCriterion],
    ):
        outputs = []
        for candidate in candidates:
            relevant = "deepfake" in ((candidate.title or "") + " " + (candidate.abstract or "")).lower()
            status = MatchStatus.PASS if relevant else MatchStatus.FAIL
            evidence = candidate.title if candidate.title else candidate.abstract
            outputs.append(
                BackendPaperJudgement(
                    candidate_id=candidate.candidate_id,
                    judgements=[
                        CriterionJudgement(
                            criterion_id=criterion.criterion_id,
                            status=status,
                            evidence=evidence,
                            reason="rule",
                            backend_name="rule",
                        )
                        for criterion in criteria
                    ],
                    backend_name="rule",
                )
            )
        return outputs, SelectorBackendMetrics()


def test_selector_verifies_evidence_and_separates_pass_fail():
    analysis = QueryAnalysisResult(
        original_query="找多模态伪造检测论文",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(task=SearchTask.PAPER_SEARCH, mode=RetrievalMode.COMPREHENSIVE),
        content="multimodal deepfake detection",
    )
    candidates = [
        CandidatePaper(candidate_id="good", title="Audio-Visual Deepfake Detection", abstract="Detects multimodal deepfakes."),
        CandidatePaper(candidate_id="bad", title="Motor Learning with Multimodal Feedback", abstract="Studies motor learning."),
    ]
    output = asyncio.run(CandidateSelector(RuleBackend()).select(analysis, candidates))
    assert output.selected_ids == ["good"]
    assert output.rejected_ids == ["bad"]
    assert output.uncertain_ids == []


class HallucinatedEvidenceBackend(RuleBackend):
    async def judge(self, candidates, criteria):
        outputs = [
            BackendPaperJudgement(
                candidate_id=candidate.candidate_id,
                judgements=[
                    CriterionJudgement(
                        criterion_id=criterion.criterion_id,
                        status=MatchStatus.FAIL,
                        evidence="This sentence does not exist in the supplied paper.",
                        reason="fake negative",
                    )
                    for criterion in criteria
                ],
                backend_name="hallucinated",
            )
            for candidate in candidates
        ]
        return outputs, SelectorBackendMetrics()


def test_unverified_decisive_evidence_downgrades_to_unknown():
    analysis = QueryAnalysisResult(
        original_query="找多模态伪造检测论文",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(task=SearchTask.PAPER_SEARCH, mode=RetrievalMode.COMPREHENSIVE),
        content="multimodal deepfake detection",
    )
    candidate = CandidatePaper(candidate_id="P1", title="A Paper", abstract="No useful evidence.")
    output = asyncio.run(CandidateSelector(HallucinatedEvidenceBackend()).select(analysis, [candidate]))
    assert output.uncertain_ids == ["P1"]
    assert output.rejected_ids == []
