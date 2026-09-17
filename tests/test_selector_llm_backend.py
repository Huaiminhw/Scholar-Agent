from __future__ import annotations

import asyncio
import json
from typing import Mapping, Sequence

from scholar_agent_v2.candidate_pool.models import CandidatePaper, MatchStatus
from scholar_agent_v2.query_understanding.analyzer import LLMResponse, LLMUsage
from scholar_agent_v2.selector.backends.llm import LLMCriteriaBackend, LLMCriteriaBackendConfig
from scholar_agent_v2.selector.models import CriterionKind, SelectorCriterion
from scholar_agent_v2.query_understanding.schema import ConstraintLevel


class EchoSelectorClient:
    def __init__(self):
        self.calls = 0

    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
        response_schema=None,
    ) -> LLMResponse:
        self.calls += 1
        request = json.loads(messages[-1]["content"])
        papers = []
        for paper in request["papers"]:
            judgements = []
            for criterion in request["criteria"]:
                evidence = paper["title"] or paper["abstract"]
                judgements.append(
                    {
                        "criterion_id": criterion["criterion_id"],
                        "status": "pass",
                        "evidence": evidence,
                        "reason": "fake",
                        "confidence": 0.9,
                    }
                )
            papers.append({"candidate_id": paper["candidate_id"], "summary": None, "judgements": judgements})
        return LLMResponse(
            content={"papers": papers},
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="fake-selector",
        )


def test_batching_caps_api_calls():
    client = EchoSelectorClient()
    backend = LLMCriteriaBackend(client, config=LLMCriteriaBackendConfig(batch_size=5, max_concurrency=2))
    candidates = [CandidatePaper(candidate_id=f"P{i}", title=f"Paper {i}") for i in range(11)]
    criteria = [
        SelectorCriterion(
            criterion_id="Q_CONTENT",
            kind=CriterionKind.CONTENT,
            level=ConstraintLevel.HARD,
            description="topic",
        )
    ]
    results, metrics = asyncio.run(backend.judge(candidates, criteria))
    assert len(results) == 11
    assert client.calls == 3
    assert metrics.api_calls == 3
    assert metrics.total_tokens == 450
    assert all(item.judgements[0].status == MatchStatus.PASS for item in results)

class RepairClient:
    def __init__(self):
        self.calls = 0

    async def complete(self, *, messages, temperature=0.0, response_schema=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="not valid json", model="fake-selector")
        requested = json.loads(messages[-1]["content"])
        candidate_ids = requested["candidate_ids"]
        criterion_ids = requested["criterion_ids"]
        return LLMResponse(
            content={
                "papers": [
                    {
                        "candidate_id": cid,
                        "summary": None,
                        "judgements": [
                            {
                                "criterion_id": criterion_id,
                                "status": "unknown",
                                "evidence": None,
                                "reason": "repaired",
                                "confidence": 0.5,
                            }
                            for criterion_id in criterion_ids
                        ],
                    }
                    for cid in candidate_ids
                ]
            },
            model="fake-selector",
        )


def test_malformed_output_uses_single_repair_call():
    client = RepairClient()
    backend = LLMCriteriaBackend(client, config=LLMCriteriaBackendConfig(batch_size=5, enable_repair=True))
    candidates = [CandidatePaper(candidate_id="P1", title="Paper")]
    criteria = [
        SelectorCriterion(
            criterion_id="Q_CONTENT",
            kind=CriterionKind.CONTENT,
            level=ConstraintLevel.HARD,
            description="topic",
        )
    ]
    results, metrics = asyncio.run(backend.judge(candidates, criteria))
    assert client.calls == 2
    assert metrics.api_calls == 2
    assert metrics.repair_calls == 1
    assert results[0].judgements[0].status == MatchStatus.UNKNOWN


class TransportFailClient:
    def __init__(self):
        self.calls = 0

    async def complete(self, *, messages, temperature=0.0, response_schema=None):
        self.calls += 1
        raise TimeoutError("network timeout")


def test_transport_failure_does_not_waste_repair_call_and_fails_open():
    client = TransportFailClient()
    backend = LLMCriteriaBackend(
        client,
        config=LLMCriteriaBackendConfig(
            batch_size=5,
            enable_repair=True,
            max_transport_retries=0,
            fail_open=True,
        ),
    )
    candidates = [CandidatePaper(candidate_id="P1", title="Paper")]
    criteria = [
        SelectorCriterion(
            criterion_id="Q_CONTENT",
            kind=CriterionKind.CONTENT,
            level=ConstraintLevel.HARD,
            description="topic",
        )
    ]
    results, metrics = asyncio.run(backend.judge(candidates, criteria))
    assert client.calls == 1
    assert metrics.api_calls == 1
    assert metrics.repair_calls == 0
    assert metrics.failed_batches == 1
    assert results[0].judgements[0].status == MatchStatus.UNKNOWN
