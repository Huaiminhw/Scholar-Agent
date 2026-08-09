from __future__ import annotations

from scholar_agent_v2.query_generation.models import (
    ExpansionBudget,
    GeneratedQuery,
    GenerationMode,
    QueryGenerationPlan,
    QueryGenerationResult,
    QueryKind,
)
from scholar_agent_v2.query_understanding.schema import (
    ConstraintItem,
    ConstraintLogic,
    ConstraintOperator,
    ConstraintSet,
    ConstraintType,
    LogicNode,
    QueryAnalysisResult,
    QueryIntent,
    QueryType,
    RetrievalMode,
)

from scholar_agent_v2.candidate_pool.filter import evaluate_candidate
from scholar_agent_v2.candidate_pool.models import CandidatePaper, MatchStatus, PaperIdentifiers
from scholar_agent_v2.candidate_pool.pool import CandidatePool
from scholar_agent_v2.retrieval.models import RetrievalHit, RetrievalOrigin, RetrievalStrategy, SearchSource
from scholar_agent_v2.retrieval.route_builder import build_retrieval_plan


def _generation(text: str = "multimodal large language models") -> QueryGenerationResult:
    plan = QueryGenerationPlan(
        mode=GenerationMode.STANDARD,
        budget=ExpansionBudget(core=1, relation=0, alias=0, expanded=1),
        use_llm=True,
        max_total_queries=5,
    )
    return QueryGenerationResult(
        original_query=text,
        core_content=text,
        plan=plan,
        queries=[
            GeneratedQuery(query_id="QC1", text=text, kind=QueryKind.CORE, priority=0.95),
            GeneratedQuery(query_id="QE1", text="multimodal LLMs", kind=QueryKind.EXPANDED, priority=0.72),
        ],
        llm_used=True,
    )


def _analysis_with_or() -> QueryAnalysisResult:
    c1 = ConstraintItem(id="C1", type=ConstraintType.VENUE, operator=ConstraintOperator.EQ, value="ACL")
    c2 = ConstraintItem(id="C2", type=ConstraintType.YEAR, operator=ConstraintOperator.EQ, value=2024)
    c3 = ConstraintItem(id="C3", type=ConstraintType.VENUE, operator=ConstraintOperator.EQ, value="NeurIPS")
    c4 = ConstraintItem(id="C4", type=ConstraintType.YEAR, operator=ConstraintOperator.EQ, value=2023)
    logic = LogicNode(
        operator="OR",
        children=[
            LogicNode(operator="AND", children=[LogicNode(operator="LEAF", ref_id="C1"), LogicNode(operator="LEAF", ref_id="C2")]),
            LogicNode(operator="AND", children=[LogicNode(operator="LEAF", ref_id="C3"), LogicNode(operator="LEAF", ref_id="C4")]),
        ],
    )
    return QueryAnalysisResult(
        original_query="multimodal LLM papers in ACL 2024 or NeurIPS 2023",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(mode=RetrievalMode.COMPREHENSIVE),
        content="multimodal large language models",
        constraints=ConstraintSet(hard=[c1, c2, c3, c4]),
        logic=ConstraintLogic(hard=logic),
    )


def _origin(source: SearchSource, route: str, qid: str, kind: QueryKind, rank: int) -> RetrievalOrigin:
    return RetrievalOrigin(
        source=source,
        route_id=route,
        query_id=qid,
        query_text="multimodal large language models",
        query_kind=kind,
        strategy=RetrievalStrategy.RELEVANCE,
        rank=rank,
    )


def test_or_pairing_is_preserved_in_routes():
    analysis = _analysis_with_or()
    plan = build_retrieval_plan(analysis, _generation())
    branches = {}
    for route in plan.routes:
        branches.setdefault(route.logic_branch_id, []).append(route)
    assert set(branches) == {"B1", "B2"}

    # S2 支持 year + venue，必须保持 ACL-2024 / NeurIPS-2023 的配对，不能交叉。
    s2 = [r for r in plan.routes if r.source == SearchSource.SEMANTIC_SCHOLAR]
    pairs = {(tuple(r.filters.venues), r.filters.year_start, r.filters.year_end) for r in s2}
    assert (("ACL",), 2024, 2024) in pairs
    assert (("NeurIPS",), 2023, 2023) in pairs
    assert (("ACL",), 2023, 2023) not in pairs
    assert (("NeurIPS",), 2024, 2024) not in pairs


def test_soft_venue_is_not_pushed():
    soft = ConstraintItem(id="S1", type=ConstraintType.VENUE, operator=ConstraintOperator.ANY_OF, values=["ACL", "NeurIPS"])
    analysis = QueryAnalysisResult(
        original_query="multimodal LLMs, preferably ACL or NeurIPS",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(mode=RetrievalMode.COMPREHENSIVE),
        content="multimodal large language models",
        constraints=ConstraintSet(soft=[soft]),
    )
    plan = build_retrieval_plan(analysis, _generation())
    assert all(not r.filters.venues for r in plan.routes)
    assert all("S1" not in r.pushed_constraint_ids for r in plan.routes)


def test_cross_source_doi_dedup_and_origin_fusion():
    h1 = RetrievalHit(
        source=SearchSource.SEMANTIC_SCHOLAR,
        source_id="s2a",
        title="A Great Paper",
        authors=["Alice Smith"],
        year=2024,
        doi="10.1000/XYZ",
        semantic_scholar_id="s2a",
        origin=_origin(SearchSource.SEMANTIC_SCHOLAR, "RT1", "QC1", QueryKind.CORE, 2),
    )
    h2 = RetrievalHit(
        source=SearchSource.OPENALEX,
        source_id="W123",
        title="A Great Paper",
        authors=["Alice Smith"],
        year=2024,
        doi="https://doi.org/10.1000/xyz",
        openalex_id="W123",
        origin=_origin(SearchSource.OPENALEX, "RT2", "QC1", QueryKind.CORE, 4),
    )
    pool = CandidatePool.from_hits([h1, h2])
    assert len(pool.candidates) == 1
    c = pool.candidates[0]
    assert c.source_count == 2
    assert c.route_count == 2
    assert len(c.origins) == 2
    assert c.identifiers.doi == "10.1000/xyz"


def test_missing_hard_metadata_is_unknown_not_fail():
    year = ConstraintItem(id="C1", type=ConstraintType.YEAR, operator=ConstraintOperator.GTE, value=2023)
    analysis = QueryAnalysisResult(
        original_query="papers since 2023",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(),
        content="papers",
        constraints=ConstraintSet(hard=[year]),
    )
    candidate = CandidatePaper(candidate_id="x", title="Unknown year", identifiers=PaperIdentifiers())
    decision = evaluate_candidate(candidate, analysis)
    assert decision.status == MatchStatus.UNKNOWN


def test_definite_hard_metadata_failure_is_removed():
    year = ConstraintItem(id="C1", type=ConstraintType.YEAR, operator=ConstraintOperator.GTE, value=2023)
    analysis = QueryAnalysisResult(
        original_query="papers since 2023",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(),
        content="papers",
        constraints=ConstraintSet(hard=[year]),
    )
    candidate = CandidatePaper(candidate_id="x", title="Old", year=2021, identifiers=PaperIdentifiers())
    decision = evaluate_candidate(candidate, analysis)
    assert decision.status == MatchStatus.FAIL


def test_multi_origin_core_candidate_ranks_above_single_expanded_hit():
    analysis = QueryAnalysisResult(
        original_query="multimodal large language models",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(),
        content="multimodal large language models",
    )
    a1 = RetrievalHit(
        source=SearchSource.SEMANTIC_SCHOLAR, source_id="a", title="Multimodal Large Language Models", abstract="multimodal large language models",
        origin=_origin(SearchSource.SEMANTIC_SCHOLAR, "RT1", "QC1", QueryKind.CORE, 2),
    )
    a2 = RetrievalHit(
        source=SearchSource.OPENALEX, source_id="oa", title="Multimodal Large Language Models", abstract="multimodal large language models",
        origin=_origin(SearchSource.OPENALEX, "RT2", "QC1", QueryKind.CORE, 3),
    )
    b = RetrievalHit(
        source=SearchSource.SEMANTIC_SCHOLAR, source_id="b", title="Large Models for Vision", abstract="models and vision",
        origin=_origin(SearchSource.SEMANTIC_SCHOLAR, "RT3", "QE1", QueryKind.EXPANDED, 15),
    )
    pool = CandidatePool.from_hits([a1, a2, b])
    pool.prerank(analysis, _generation(), num_planned_routes=3)
    assert pool.candidates[0].title == "Multimodal Large Language Models"


def test_latest_mode_gives_newer_paper_a_higher_preference_score():
    analysis = QueryAnalysisResult(
        original_query="latest multimodal large language model papers",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(mode=RetrievalMode.LATEST),
        content="multimodal large language models",
    )
    older = CandidatePaper(
        candidate_id="older",
        title="Multimodal Large Language Models",
        year=2021,
    )
    newer = CandidatePaper(
        candidate_id="newer",
        title="Multimodal Large Language Models",
        year=2025,
    )
    pool = CandidatePool(candidates=[older, newer])

    pool.prerank(analysis, _generation())

    assert pool.candidates[0].candidate_id == "newer"
    assert newer.score_components["preference"] > older.score_components["preference"]


def test_exclude_survey_can_use_strong_title_evidence():
    exclude = ConstraintItem(
        id="E1",
        type=ConstraintType.PUBLICATION_TYPE,
        operator=ConstraintOperator.EQ,
        value="survey",
    )
    analysis = QueryAnalysisResult(
        original_query="multimodal LLM papers, no surveys",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(),
        content="multimodal large language models",
        constraints=ConstraintSet(exclude=[exclude]),
    )
    candidate = CandidatePaper(
        candidate_id="survey",
        title="A Survey of Multimodal Large Language Models",
        identifiers=PaperIdentifiers(),
    )
    assert evaluate_candidate(candidate, analysis).status == MatchStatus.FAIL


def test_semantic_hard_constraint_is_unknown_not_deleted():
    method = ConstraintItem(
        id="C1",
        type=ConstraintType.METHOD,
        operator=ConstraintOperator.EQ,
        value="reinforcement learning",
    )
    analysis = QueryAnalysisResult(
        original_query="reinforcement learning methods",
        query_type=QueryType.BROAD_BY_DESCRIPTION,
        intent=QueryIntent(),
        content="reinforcement learning methods",
        constraints=ConstraintSet(hard=[method]),
    )
    candidate = CandidatePaper(
        candidate_id="x",
        title="Reinforcement Learning for Search",
        identifiers=PaperIdentifiers(),
    )
    assert evaluate_candidate(candidate, analysis).status == MatchStatus.UNKNOWN
