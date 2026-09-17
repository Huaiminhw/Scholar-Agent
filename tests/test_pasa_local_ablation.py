from __future__ import annotations

from scripts.eval_pasa_frozen_v2_queries import (
    CandidateOrigin,
    DeduplicatedCandidate,
    GoldPaper,
    evaluate_ablation_arms,
    rank_candidates_for_arm,
)


def _candidate(
    arxiv_id: str,
    *,
    route_index: int,
    kind: str,
    source: str = "generated",
    rank: int = 1,
) -> DeduplicatedCandidate:
    return DeduplicatedCandidate(
        arxiv_id=arxiv_id,
        title=f"paper {arxiv_id}",
        year=2024,
        corpus_key=arxiv_id,
        first_seen_order=route_index,
        origins=[
            CandidateOrigin(
                route_index=route_index,
                query_id=f"Q{route_index}",
                query_kind=kind,
                query_source=source,
                query_text=f"query {route_index}",
                rank=rank,
                bm25_score=1.0,
            )
        ],
    )


def _route(route_index: int, kind: str, source: str = "generated") -> dict:
    return {
        "route_index": route_index,
        "query_kind": kind,
        "query_source": source,
        "retrieved_count": 1,
        "missing_arxiv_id_hits": 0,
    }


def test_ablation_arms_do_not_leak_query_kinds() -> None:
    candidates = [
        _candidate("2401.00001", route_index=0, kind="original", source="original"),
        _candidate("2401.00002", route_index=1, kind="core"),
        _candidate("2401.00003", route_index=2, kind="relation"),
        _candidate("2401.00004", route_index=3, kind="alias"),
        _candidate("2401.00005", route_index=4, kind="expanded"),
    ]

    assert [item.arxiv_id for item in rank_candidates_for_arm(candidates, arm="A_original")] == [
        "2401.00001"
    ]
    assert {
        item.arxiv_id
        for item in rank_candidates_for_arm(candidates, arm="B_deterministic")
    } == {"2401.00001", "2401.00002", "2401.00003"}
    assert len(rank_candidates_for_arm(candidates, arm="C_full")) == 5


def test_ablation_metrics_measure_ranked_marginal_gain() -> None:
    candidates = [
        _candidate("2401.00001", route_index=0, kind="original", source="original"),
        _candidate("2401.00002", route_index=1, kind="core"),
        _candidate("2401.00003", route_index=2, kind="expanded"),
    ]
    routes = [
        _route(0, "original", "original"),
        _route(1, "core"),
        _route(2, "expanded"),
    ]
    gold = [
        GoldPaper(title="one", arxiv_id="2401.00001"),
        GoldPaper(title="two", arxiv_id="2401.00002"),
        GoldPaper(title="three", arxiv_id="2401.00003"),
    ]

    results = evaluate_ablation_arms(
        candidates=candidates,
        route_results=routes,
        gold=gold,
        available_gold_indices={0, 1, 2},
    )

    assert results["A_original"].recall_at["100"] == 1 / 3
    assert results["B_deterministic"].recall_at["100"] == 2 / 3
    assert results["C_full"].recall_at["100"] == 1.0
    assert results["B_deterministic"].marginal_gold_hits_at["100"] == 1
    assert results["C_full"].marginal_gold_hits_at["100"] == 2
    assert results["A_original"].candidate_inflation_vs_original == 0.0
    assert results["C_full"].candidate_inflation_vs_original == 2.0


def test_weighted_rrf_rewards_support_from_multiple_routes() -> None:
    supported = _candidate("2401.00001", route_index=0, kind="original", source="original", rank=8)
    supported.origins.append(
        CandidateOrigin(
            route_index=1,
            query_id="Q1",
            query_kind="core",
            query_source="generated",
            query_text="core query",
            rank=8,
            bm25_score=1.0,
        )
    )
    single = _candidate("2401.00002", route_index=2, kind="original", source="original", rank=1)

    ranked = rank_candidates_for_arm([single, supported], arm="B_deterministic")

    assert ranked[0].arxiv_id == "2401.00001"
    assert ranked[0].matched_routes == 2
