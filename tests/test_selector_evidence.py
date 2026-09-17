from scholar_agent_v2.candidate_pool.models import CandidatePaper
from scholar_agent_v2.selector.evidence import verify_evidence


def test_verify_exact_abstract_evidence_with_punctuation_normalization():
    candidate = CandidatePaper(
        candidate_id="P1",
        title="A Search Agent",
        abstract="We optimize the search policy using reinforcement-learning on real scientific queries.",
    )
    result = verify_evidence(
        candidate,
        "optimize the search policy using reinforcement learning",
    )
    assert result.verified is True
    assert result.source == "abstract"


def test_verify_multi_part_evidence():
    candidate = CandidatePaper(
        candidate_id="P1",
        title="RL Search Agent",
        abstract="We evaluate the agent on real scientific queries collected from researchers.",
    )
    result = verify_evidence(candidate, "RL Search Agent ... real scientific queries")
    assert result.verified is True
    assert result.source == "mixed"


def test_hallucinated_evidence_is_rejected():
    candidate = CandidatePaper(candidate_id="P1", title="A Paper", abstract="Unrelated abstract.")
    result = verify_evidence(candidate, "we use reinforcement learning")
    assert result.verified is False
