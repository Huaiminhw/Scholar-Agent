from __future__ import annotations

import asyncio
import os

import pytest

from scholar_agent_v2.candidate_pool.models import CandidatePaper
from scholar_agent_v2.query_understanding.schema import (
    QueryAnalysisResult,
    QueryIntent,
    QueryType,
    RetrievalMode,
    SearchTask,
)
from scholar_agent_v2.selector.backends.llm import LLMCriteriaBackend, LLMCriteriaBackendConfig
from scholar_agent_v2.selector.selector import CandidateSelector, SelectorConfig
from tests.real_llm_client import build_client


RUN_REAL = os.getenv("RUN_REAL_SELECTOR_TESTS", "0") == "1"


@pytest.mark.skipif(
    not RUN_REAL,
    reason="Set RUN_REAL_SELECTOR_TESTS=1 to run the real Selector LLM smoke test.",
)
def test_real_selector_semantic_smoke():
    analysis = QueryAnalysisResult(
        original_query="找多模态伪造检测方面的论文",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(task=SearchTask.PAPER_SEARCH, mode=RetrievalMode.COMPREHENSIVE),
        content="multimodal forgery and deepfake detection",
    )
    candidates = [
        CandidatePaper(
            candidate_id="good",
            title="Audio-Visual Joint Learning for Deepfake Detection",
            abstract=(
                "We propose an audio-visual framework for multimodal deepfake detection. "
                "The method jointly models visual and audio forgery cues for detecting manipulated videos."
            ),
        ),
        CandidatePaper(
            candidate_id="bad",
            title="Augmented Visual, Auditory, Haptic, and Multimodal Feedback in Motor Learning",
            abstract=(
                "This review studies augmented visual, auditory and haptic feedback for motor learning "
                "and human movement training."
            ),
        ),
    ]

    backend = LLMCriteriaBackend(
        build_client(),
        config=LLMCriteriaBackendConfig(
            batch_size=5,
            max_concurrency=1,
            temperature=0.0,
            use_structured_output=False,
            enable_repair=True,
            max_transport_retries=0,
        ),
    )
    selector = CandidateSelector(
        backend,
        config=SelectorConfig(max_candidates=10),
    )
    output = asyncio.run(selector.select(analysis, candidates))

    print("\n========== REAL SELECTOR SMOKE ==========")
    print(output.model_dump_json(indent=2))

    assert len(output.results) == 2
    assert output.metrics.llm_api_calls <= 2  # one normal call; at most one repair on malformed JSON
    # Semantic assertions are deliberately recall-safe for live-model variability.
    assert "good" not in output.rejected_ids
    assert "bad" not in output.selected_ids
