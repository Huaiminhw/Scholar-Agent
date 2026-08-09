from __future__ import annotations

import asyncio
import os

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from scholar_agent_v2.candidate_pool.pool import CandidatePool
from scholar_agent_v2.query_generation.models import QueryKind
from scholar_agent_v2.retrieval.config import get_enabled_sources
from scholar_agent_v2.retrieval.models import (
    RetrievalPlan,
    RetrievalStrategy,
    RouteSpec,
    RouteStatus,
    SearchSource,
)
from scholar_agent_v2.retrieval.orchestrator import RetrievalOrchestrator
from scholar_agent_v2.retrieval.sources.arxiv import ArxivAdapter
from scholar_agent_v2.retrieval.sources.openalex import OpenAlexAdapter
from scholar_agent_v2.retrieval.sources.semantic_scholar import SemanticScholarAdapter


RUN_REAL = os.getenv("RUN_REAL_RETRIEVAL_TESTS", "0") == "1"
TEST_QUERY = os.getenv(
    "REAL_RETRIEVAL_TEST_QUERY",
    "multimodal large language models",
)
PER_SOURCE_LIMIT = int(os.getenv("REAL_RETRIEVAL_LIMIT", "10"))

_ROUTE_IDS = {
    SearchSource.OPENALEX: "RT_OA",
    SearchSource.SEMANTIC_SCHOLAR: "RT_S2",
    SearchSource.ARXIV: "RT_AX",
}


def _build_plan(enabled_sources: list[SearchSource]) -> RetrievalPlan:
    routes = [
        RouteSpec(
            route_id=_ROUTE_IDS[source],
            query_id="QC1",
            query_text=TEST_QUERY,
            query_kind=QueryKind.CORE,
            source=source,
            strategy=RetrievalStrategy.RELEVANCE,
            priority=1.0,
            limit=PER_SOURCE_LIMIT,
        )
        for source in enabled_sources
    ]

    return RetrievalPlan(
        routes=routes,
        max_routes=len(routes),
        max_total_hits=PER_SOURCE_LIMIT * len(routes),
    )


def _build_adapters(enabled_sources: list[SearchSource]):
    enabled = set(enabled_sources)
    adapters = {}

    if SearchSource.OPENALEX in enabled:
        openalex_key = os.getenv("OPENALEX_API_KEY", "").strip()
        if not openalex_key:
            pytest.fail(
                "USE_OPENALEX=true，但 OPENALEX_API_KEY 未配置。\n"
                "请设置 OPENALEX_API_KEY，或将 USE_OPENALEX=false。"
            )

        adapters[SearchSource.OPENALEX] = OpenAlexAdapter(
            api_key=openalex_key,
            timeout_seconds=25.0,
        )

    if SearchSource.SEMANTIC_SCHOLAR in enabled:
        s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None
        if not s2_key:
            print(
                "\n[WARN] USE_SEMANTIC_SCHOLAR=true，但 "
                "SEMANTIC_SCHOLAR_API_KEY 未配置，将尝试匿名调用。"
            )

        adapters[SearchSource.SEMANTIC_SCHOLAR] = SemanticScholarAdapter(
            api_key=s2_key,
            timeout_seconds=25.0,
        )

    if SearchSource.ARXIV in enabled:
        adapters[SearchSource.ARXIV] = ArxivAdapter(
            timeout_seconds=30.0,
        )

    return adapters


async def _run_enabled_retrieval():
    enabled_sources = get_enabled_sources()
    adapters = _build_adapters(enabled_sources)

    orchestrator = RetrievalOrchestrator(
        adapters,
        max_concurrency=max(1, len(enabled_sources)),
        route_timeout_seconds=35.0,
        max_attempts=2,
        retry_backoff_seconds=1.0,
        raise_if_all_failed=True,
    )

    plan = _build_plan(enabled_sources)
    output = await orchestrator.run(plan)
    return enabled_sources, output


def _print_retrieval_output(
    enabled_sources: list[SearchSource],
    output,
) -> None:
    metrics = output.metrics

    print("\n========== REAL RETRIEVAL: ENABLED SOURCES ==========")
    print(f"query               = {TEST_QUERY}")
    print(f"enabled_sources     = {[s.value for s in enabled_sources]}")
    print(f"per_source_limit    = {PER_SOURCE_LIMIT}")
    print(f"planned_routes      = {metrics.planned_routes}")
    print(f"executed_routes     = {metrics.executed_routes}")
    print(f"successful_routes   = {metrics.successful_routes}")
    print(f"empty_routes        = {metrics.empty_routes}")
    print(f"failed_routes       = {metrics.failed_routes}")
    print(f"timeout_routes      = {metrics.timeout_routes}")
    print(f"api_calls           = {metrics.api_calls}")
    print(f"retries             = {metrics.retries}")
    print(f"raw_hits            = {metrics.raw_hits}")
    print(f"source_hit_counts   = {metrics.source_hit_counts}")
    print(f"source_api_calls    = {metrics.source_api_calls}")
    print(f"source_latencies_ms = {metrics.source_latencies_ms}")
    print(f"total_latency_ms    = {metrics.total_latency_ms:.2f}")

    print("\n---------- ROUTE EXECUTIONS ----------")
    for ex in output.executions:
        print(
            f"[{ex.route_id}] source={ex.source.value:<18} "
            f"status={ex.status.value:<8} hits={len(ex.hits):<3} "
            f"api_calls={ex.api_calls:<2} attempts={ex.attempts:<2} "
            f"latency_ms={ex.latency_ms:.2f}"
        )
        if ex.error_type or ex.error_message:
            print(f"    error={ex.error_type}: {ex.error_message}")


def _print_candidate_pool(pool: CandidatePool, raw_hit_count: int) -> None:
    unique_count = len(pool.candidates)
    multi_source = [c for c in pool.candidates if c.source_count >= 2]
    multi_route = [c for c in pool.candidates if c.route_count >= 2]

    print("\n========== CANDIDATE POOL ==========")
    print(f"raw_hits             = {raw_hit_count}")
    print(f"unique_candidates    = {unique_count}")
    print(f"deduplicated_hits    = {raw_hit_count - unique_count}")
    print(f"multi_source_papers  = {len(multi_source)}")
    print(f"multi_route_papers   = {len(multi_route)}")

    print("\n---------- TOP CANDIDATES / ORIGINS ----------")
    candidates = sorted(
        pool.candidates,
        key=lambda c: c.best_rank or 10**9,
    )[:10]

    for idx, candidate in enumerate(candidates, start=1):
        title = (candidate.title or "<NO TITLE>").replace("\n", " ").strip()
        print(
            f"[{idx}] {title}\n"
            f"    candidate_id={candidate.candidate_id}\n"
            f"    year={candidate.year} venue={candidate.venue}\n"
            f"    source_count={candidate.source_count} "
            f"route_count={candidate.route_count} "
            f"hit_count={candidate.hit_count} "
            f"best_rank={candidate.best_rank}"
        )

        for origin in sorted(
            candidate.origins,
            key=lambda x: (x.source.value, x.rank),
        ):
            print(
                "    origin="
                f"{origin.source.value} / {origin.query_id} / "
                f"{origin.strategy.value} / rank={origin.rank} / "
                f"route={origin.route_id}"
            )


@pytest.mark.skipif(
    not RUN_REAL,
    reason="Set RUN_REAL_RETRIEVAL_TESTS=1 to run real network retrieval tests.",
)
def test_real_enabled_sources_smoke():
    """
    真实 Retrieval + Candidate Pool 冒烟测试。

    数据源完全由：
      USE_OPENALEX
      USE_SEMANTIC_SCHOLAR
      USE_ARXIV
    控制。

    Semantic Scholar 被关闭时：
    - 不生成 S2 Route
    - 不实例化 S2 Adapter
    - 不发送 S2 HTTP 请求
    - 但 S2 代码和接口完整保留
    """
    enabled_sources, output = asyncio.run(_run_enabled_retrieval())
    _print_retrieval_output(enabled_sources, output)

    expected_sources = set(enabled_sources)
    execution_by_source = {ex.source: ex for ex in output.executions}

    assert set(execution_by_source) == expected_sources
    assert output.metrics.planned_routes == len(enabled_sources)
    assert output.metrics.executed_routes == len(enabled_sources)

    for source in enabled_sources:
        ex = execution_by_source[source]
        assert ex.status == RouteStatus.SUCCESS, (
            f"{source.value} route is {ex.status.value}; "
            f"error={ex.error_type}: {ex.error_message}"
        )
        assert len(ex.hits) > 0, f"{source.value} returned zero hits"

    assert output.metrics.successful_routes == len(enabled_sources)
    assert output.metrics.failed_routes == 0
    assert output.metrics.timeout_routes == 0
    assert output.metrics.raw_hits == len(output.hits)
    assert output.metrics.raw_hits > 0

    # 禁用的源必须完全没有执行记录。
    disabled_sources = set(SearchSource) - expected_sources
    assert all(ex.source not in disabled_sources for ex in output.executions)

    for hit in output.hits:
        assert hit.origin.source == hit.source
        assert hit.origin.query_id == "QC1"
        assert hit.origin.rank >= 1
        assert hit.title

    pool = CandidatePool.from_hits(output.hits)
    _print_candidate_pool(pool, raw_hit_count=len(output.hits))

    assert len(pool.candidates) > 0
    assert len(pool.candidates) <= len(output.hits)
    assert pool.stats.raw_hits == len(output.hits)

    for candidate in pool.candidates:
        assert candidate.origins
        assert candidate.hit_count >= len(candidate.origins)
        assert candidate.source_count == len({o.source for o in candidate.origins})
        assert candidate.route_count == len({o.route_id for o in candidate.origins})
        assert candidate.best_rank == min(o.rank for o in candidate.origins)

    print(
        "\n[PASS] Real retrieval pipeline is healthy for enabled sources: "
        + ", ".join(source.value for source in enabled_sources)
    )
