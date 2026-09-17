# tests/test_selector_pipeline_real.py
"""
Static Retrieval -> Candidate Pool -> Qwen Semantic Selector 真实端到端集成测试。

目的
----
1. 复用已经冻结的 Static Retrieval（静态路由）真实 Pipeline，不复制第二套路由逻辑；
2. 在真实 OpenAlex / Semantic Scholar / arXiv Candidate Pool 上运行 Qwen Selector；
3. 验证：
   - Metadata Hard Filter（元数据硬过滤）仍然保持 Recall-safe；
   - Semantic Selector（语义判别器）只补齐语义条件/关系条件；
   - Evidence Verification（证据验证）正常；
   - PASS / UNKNOWN / FAIL 三值逻辑最终能闭环；
   - Selector API Calls / Tokens / Latency 可观测；
4. 本测试没有 Gold Labels，因此这里不能宣称真实 Precision / Recall / F1。
   真正 F1 必须在人工 Gold Set / 比赛标注集上计算。

推荐先跑 Broad Case
-------------------
Linux / 远程服务器：

    export RUN_REAL_SELECTOR_PIPELINE_TESTS=1
    export SELECTOR_PIPELINE_CASES=broad
    python -m pytest -s tests/test_selector_pipeline_real.py

Windows PowerShell：

    $env:RUN_REAL_SELECTOR_PIPELINE_TESTS="1"
    $env:SELECTOR_PIPELINE_CASES="broad"
    python -m pytest -s tests/test_selector_pipeline_real.py

其他 Case：

    # Complex
    export SELECTOR_PIPELINE_CASES=complex

    # OR + Exclude
    export SELECTOR_PIPELINE_CASES=or_exclude

    # 全部
    export SELECTOR_PIPELINE_CASES=all

可调参数
--------
SELECTOR_PIPELINE_TOP_N=30
SELECTOR_PIPELINE_BATCH_SIZE=5
SELECTOR_PIPELINE_MAX_CONCURRENCY=2
SELECTOR_PIPELINE_ENABLE_REPAIR=1
SELECTOR_PIPELINE_MAX_FAILED_BATCHES=0
SELECTOR_PIPELINE_MAX_LLM_API_CALLS=0
SELECTOR_PIPELINE_MAX_LATENCY_MS=0

说明：
- MAX_LLM_API_CALLS=0：不做 Selector API 次数硬上限，只打印；
- MAX_LATENCY_MS=0：不做端到端延迟硬上限，只打印；
- 默认 max_failed_batches=0，真实验收时任何 Selector batch 失败都暴露出来；
- Selector 默认 fail-open：若模型批次失败，Candidate 会降级为 UNKNOWN，不会误删。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from scholar_agent_v2.candidate_pool.models import CandidatePaper, MatchStatus
from scholar_agent_v2.retrieval.config import get_enabled_sources
from scholar_agent_v2.selector import CandidateSelector, SelectorConfig, SelectorOutput
from scholar_agent_v2.selector.backends import (
    QwenCriteriaBackend,
    QwenCriteriaBackendConfig,
)
from tests.real_llm_client import build_client

# 复用已经验证过的 Static Retrieval 真实 Pipeline。
# 这里故意不复制 Route Builder / source caps / retrieval health 逻辑，
# 避免未来 Static baseline 与 Selector Pipeline 的检索阶段发生漂移。
from tests.test_retrieval_pipeline_real import (
    ALL_CASES,
    PipelineCase,
    _build_adapters,
    _run_case,
)


# =============================================================================
# Runtime configuration
# =============================================================================

RUN_REAL = os.getenv("RUN_REAL_SELECTOR_PIPELINE_TESTS", "0") == "1"
CASE_SELECTOR = os.getenv("SELECTOR_PIPELINE_CASES", "broad").strip().lower()

SELECTOR_TOP_N = int(os.getenv("SELECTOR_PIPELINE_TOP_N", "30"))
SELECTOR_BATCH_SIZE = int(os.getenv("SELECTOR_PIPELINE_BATCH_SIZE", "5"))
SELECTOR_MAX_CONCURRENCY = int(
    os.getenv("SELECTOR_PIPELINE_MAX_CONCURRENCY", "2")
)
SELECTOR_ENABLE_REPAIR = (
    os.getenv("SELECTOR_PIPELINE_ENABLE_REPAIR", "1").strip() == "1"
)
SELECTOR_MAX_FAILED_BATCHES = int(
    os.getenv("SELECTOR_PIPELINE_MAX_FAILED_BATCHES", "0")
)

# 0 = 只记录，不做硬断言
SELECTOR_MAX_LLM_API_CALLS = int(
    os.getenv("SELECTOR_PIPELINE_MAX_LLM_API_CALLS", "0")
)
SELECTOR_MAX_LATENCY_MS = float(
    os.getenv("SELECTOR_PIPELINE_MAX_LATENCY_MS", "0")
)

# Selector 前只补齐这些最有价值的字段。
# CandidateEnricher 会按 Candidate 当前缺失字段决定是否真的发请求。
ENRICH_REQUIRED_FIELDS = (
    "abstract",
    "venue",
    "citation_count",
    "publication_types",
    "fields_of_study",
)


# =============================================================================
# Small result container
# =============================================================================


@dataclass
class SelectorPipelineRun:
    case: PipelineCase
    analysis: object
    generation: object
    execution_plan: object
    retrieval: object
    pool: object
    selector_input: list[CandidatePaper]
    selector_output: SelectorOutput
    enrichment_api_calls: int
    enrichment_ms: float
    total_wall_ms: float


# =============================================================================
# Case selection
# =============================================================================


def _selected_cases() -> list[PipelineCase]:
    if CASE_SELECTOR in {"all", "*"}:
        return list(ALL_CASES.values())

    names = [x.strip() for x in CASE_SELECTOR.split(",") if x.strip()]
    unknown = [x for x in names if x not in ALL_CASES]
    if unknown:
        raise ValueError(
            f"Unknown SELECTOR_PIPELINE_CASES={unknown}; "
            f"valid={list(ALL_CASES)} or all"
        )

    return [ALL_CASES[name] for name in names]


# =============================================================================
# Builders
# =============================================================================


def _build_selector() -> CandidateSelector:
    backend = QwenCriteriaBackend(
        build_client(),
        config=QwenCriteriaBackendConfig(
            batch_size=SELECTOR_BATCH_SIZE,
            max_concurrency=SELECTOR_MAX_CONCURRENCY,
            temperature=0.0,
            use_structured_output=False,
            enable_repair=SELECTOR_ENABLE_REPAIR,
            # 比赛效率统计中避免底层自动多次请求；格式问题最多走一次 Repair。
            max_transport_retries=0,
            # 真实 Pipeline 中宁可 UNKNOWN，也不要因为单个 batch 异常误删 Candidate。
            fail_open=True,
        ),
    )

    return CandidateSelector(
        backend,
        config=SelectorConfig(
            max_candidates=SELECTOR_TOP_N,
            # Broad Query 没有显式 semantic hard criterion 时，
            # 自动用 analysis.content 构造 Q_CONTENT；
            # Complex Query 已有 C/R 条件时不重复判断整个 content。
            content_criterion_mode="fallback",
            include_soft=True,
            include_exclude=True,
            # 决定性的 PASS/FAIL 必须能在 title/abstract 中验证到 evidence。
            require_verified_evidence_for_decisive=True,
            min_evidence_normalized_chars=8,
            strict_field_of_study=False,
            fail_open_prior=True,
        ),
    )


# =============================================================================
# Diagnostics helpers
# =============================================================================


def _safe_title(candidate: CandidatePaper) -> str:
    return (candidate.title or "<NO TITLE>").replace("\n", " ").strip()


def _candidate_map(candidates: Iterable[CandidatePaper]) -> dict[str, CandidatePaper]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def _short(text: str | None, limit: int = 180) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _print_selector_input(candidates: list[CandidatePaper]) -> None:
    print("\n========== SELECTOR INPUT ==========")
    print(f"candidate_count      = {len(candidates)}")
    print(f"batch_size           = {SELECTOR_BATCH_SIZE}")
    print(f"max_concurrency      = {SELECTOR_MAX_CONCURRENCY}")
    print(
        "expected_batches     = "
        f"{math.ceil(len(candidates) / SELECTOR_BATCH_SIZE) if candidates else 0}"
    )

    for idx, candidate in enumerate(candidates, start=1):
        abstract_state = "yes" if (candidate.abstract or "").strip() else "no"
        score = (
            f"{candidate.retrieval_score:.4f}"
            if candidate.retrieval_score is not None
            else "None"
        )
        print(
            f"[{idx:02d}] id={candidate.candidate_id} "
            f"score={score} abstract={abstract_state} "
            f"hard={candidate.hard_filter_status.value if candidate.hard_filter_status else None}\n"
            f"     {_safe_title(candidate)}"
        )


def _print_selector_output(
    output: SelectorOutput,
    candidates: list[CandidatePaper],
) -> None:
    by_id = _candidate_map(candidates)

    print("\n========== QWEN SELECTOR RESULT ==========")
    print(f"PASS                 = {len(output.selected_ids)}")
    print(f"UNKNOWN              = {len(output.uncertain_ids)}")
    print(f"FAIL                 = {len(output.rejected_ids)}")
    print(f"llm_api_calls        = {output.metrics.llm_api_calls}")
    print(f"llm_repair_calls     = {output.metrics.llm_repair_calls}")
    print(f"llm_failed_batches   = {output.metrics.llm_failed_batches}")
    print(f"prompt_tokens        = {output.metrics.llm_prompt_tokens}")
    print(f"completion_tokens    = {output.metrics.llm_completion_tokens}")
    print(f"total_tokens         = {output.metrics.llm_total_tokens}")
    print(f"llm_latency_ms       = {output.metrics.llm_latency_ms:.2f}")
    print(f"selector_total_ms    = {output.metrics.total_latency_ms:.2f}")
    print(f"models               = {output.metrics.llm_models}")

    known_values = [
        result.semantic_known_fraction
        for result in output.results
    ]
    avg_known = (
        sum(known_values) / len(known_values)
        if known_values
        else 1.0
    )
    print(f"avg_semantic_known   = {avg_known:.4f}")

    print("\n---------- PER PAPER JUDGEMENTS ----------")
    for idx, result in enumerate(output.results, start=1):
        candidate = by_id.get(result.candidate_id)
        title = _safe_title(candidate) if candidate else "<MISSING CANDIDATE>"

        print(
            f"\n[{idx:02d}] {result.final_status.value.upper():<7} "
            f"id={result.candidate_id} "
            f"retrieval_score={result.retrieval_score} "
            f"known={result.semantic_known_fraction:.3f}"
        )
        print(f"     {title}")
        print(
            f"     hard={result.hard_status.value} "
            f"exclude_match={result.exclude_match_status.value} "
            f"soft={result.soft_match_score:.3f}"
        )
        if result.prior_score is not None:
            print(f"     prior_score={result.prior_score:.4f}")

        for judgement in result.semantic_judgements:
            print(
                f"     - {judgement.criterion_id}: "
                f"{judgement.status.value} "
                f"confidence={judgement.confidence} "
                f"evidence_verified={judgement.evidence_verified}"
            )
            if judgement.evidence:
                print(f"       evidence={_short(judgement.evidence)}")
            if judgement.reason:
                print(f"       reason={_short(judgement.reason)}")
            if judgement.warnings:
                print(
                    "       warnings="
                    + json.dumps(judgement.warnings, ensure_ascii=False)
                )

        if result.summary:
            print(f"     summary={_short(result.summary, 240)}")
        if result.warnings:
            print(
                "     paper_warnings="
                + json.dumps(result.warnings, ensure_ascii=False)
            )

    def print_bucket(label: str, ids: list[str]) -> None:
        print(f"\n---------- {label} ({len(ids)}) ----------")
        if not ids:
            print("<empty>")
            return
        for idx, candidate_id in enumerate(ids, start=1):
            candidate = by_id.get(candidate_id)
            title = _safe_title(candidate) if candidate else "<MISSING CANDIDATE>"
            print(f"[{idx:02d}] {candidate_id} | {title}")

    print_bucket("SELECTED / PASS", output.selected_ids)
    print_bucket("UNCERTAIN / UNKNOWN", output.uncertain_ids)
    print_bucket("REJECTED / FAIL", output.rejected_ids)


def _print_combined_efficiency(run: SelectorPipelineRun) -> None:
    generation = run.generation
    retrieval = run.retrieval
    selector = run.selector_output

    qgen_calls = generation.metrics.api_calls
    retrieval_calls = retrieval.metrics.api_calls
    enrichment_calls = run.enrichment_api_calls
    selector_calls = selector.metrics.llm_api_calls

    total_calls = (
        qgen_calls
        + retrieval_calls
        + enrichment_calls
        + selector_calls
    )

    final_retained = (
        len(selector.selected_ids)
        + len(selector.uncertain_ids)
    )
    input_count = max(1, len(run.selector_input))

    summary = {
        "query_generation_api_calls": qgen_calls,
        "retrieval_api_calls": retrieval_calls,
        "enrichment_api_calls": enrichment_calls,
        "selector_llm_api_calls": selector_calls,
        "total_api_calls": total_calls,
        "query_generation_tokens": generation.metrics.total_tokens,
        "selector_tokens": selector.metrics.llm_total_tokens,
        "raw_hits": retrieval.metrics.raw_hits,
        "candidate_pool_after_hard_filter": len(run.pool.candidates),
        "selector_input": len(run.selector_input),
        "selector_pass": len(selector.selected_ids),
        "selector_unknown": len(selector.uncertain_ids),
        "selector_fail": len(selector.rejected_ids),
        "selector_retain_count": final_retained,
        "selector_reject_rate": round(
            len(selector.rejected_ids) / input_count,
            4,
        ),
        "enrichment_ms": round(run.enrichment_ms, 2),
        "selector_ms": round(selector.metrics.total_latency_ms, 2),
        "whole_pipeline_wall_ms": round(run.total_wall_ms, 2),
    }

    print("\n========== END-TO-END EFFICIENCY SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        "f1_note = no gold labels in this integration test; "
        "PASS/FAIL counts are diagnostics, not true Precision/Recall/F1."
    )


# =============================================================================
# Health assertions
# =============================================================================


def _assert_selector_health(
    output: SelectorOutput,
    candidates: list[CandidatePaper],
) -> None:
    input_ids = [candidate.candidate_id for candidate in candidates]
    input_set = set(input_ids)

    assert len(input_ids) == len(input_set), (
        "Selector input candidate_id must be unique."
    )

    result_ids = [result.candidate_id for result in output.results]
    assert len(result_ids) == len(candidates)
    assert set(result_ids) == input_set

    selected = set(output.selected_ids)
    uncertain = set(output.uncertain_ids)
    rejected = set(output.rejected_ids)

    # 三个 bucket 必须互斥且完整覆盖输入。
    assert not (selected & uncertain)
    assert not (selected & rejected)
    assert not (uncertain & rejected)
    assert selected | uncertain | rejected == input_set

    assert output.metrics.candidates_seen == len(candidates)
    assert output.metrics.pass_count == len(selected)
    assert output.metrics.unknown_count == len(uncertain)
    assert output.metrics.fail_count == len(rejected)

    # 默认真实验收不接受悄悄失败的 LLM batch。
    assert output.metrics.llm_failed_batches <= SELECTOR_MAX_FAILED_BATCHES, (
        f"selector failed_batches={output.metrics.llm_failed_batches} > "
        f"allowed={SELECTOR_MAX_FAILED_BATCHES}"
    )

    if SELECTOR_MAX_LLM_API_CALLS > 0:
        assert output.metrics.llm_api_calls <= SELECTOR_MAX_LLM_API_CALLS, (
            f"selector llm_api_calls={output.metrics.llm_api_calls} > "
            f"budget={SELECTOR_MAX_LLM_API_CALLS}"
        )

    if SELECTOR_MAX_LATENCY_MS > 0:
        assert output.metrics.total_latency_ms <= SELECTOR_MAX_LATENCY_MS, (
            f"selector latency={output.metrics.total_latency_ms:.2f}ms > "
            f"budget={SELECTOR_MAX_LATENCY_MS:.2f}ms"
        )

    # 在 require_verified_evidence_for_decisive=True 下：
    # Semantic judgement 只要仍然是 PASS/FAIL，就必须有已验证证据。
    # 无法验证的 decisive judgement 应已被 Selector 自动降级为 UNKNOWN。
    for result in output.results:
        for judgement in result.semantic_judgements:
            if judgement.status in {MatchStatus.PASS, MatchStatus.FAIL}:
                assert judgement.evidence_verified, (
                    f"{result.candidate_id}/{judgement.criterion_id} is "
                    f"{judgement.status.value} without verified evidence"
                )
                assert (judgement.evidence or "").strip(), (
                    f"{result.candidate_id}/{judgement.criterion_id} has "
                    "decisive status but empty evidence"
                )

    # API 次数的自然上限：
    # 每个 batch 正常一次；若 JSON malformed，最多再一次 Repair。
    expected_batches = (
        math.ceil(len(candidates) / SELECTOR_BATCH_SIZE)
        if candidates
        else 0
    )
    natural_upper_bound = expected_batches * (
        2 if SELECTOR_ENABLE_REPAIR else 1
    )
    assert output.metrics.llm_api_calls <= natural_upper_bound, (
        f"selector api calls={output.metrics.llm_api_calls} exceed "
        f"natural batch/repair upper bound={natural_upper_bound}"
    )


# =============================================================================
# Pipeline
# =============================================================================


async def _run_selector_pipeline(case: PipelineCase) -> SelectorPipelineRun:
    total_started = time.perf_counter()

    # -------------------------------------------------------------------------
    # 1) Frozen Static Retrieval + Candidate Pool
    # -------------------------------------------------------------------------
    (
        analysis,
        generation,
        _native_plan,
        execution_plan,
        retrieval,
        pool,
    ) = await _run_case(case)

    assert pool.candidates, (
        f"{case.name}: Candidate Pool is empty after hard filter."
    )

    # -------------------------------------------------------------------------
    # 2) Enrichment before semantic judgement
    #
    # 复用 CandidateEnricher；只给 Selector Top-N 附近的论文补 abstract/metadata。
    # 原 Static Pipeline 已经 prerank 过一次，所以当前 pool.candidates 有稳定排序依据。
    # Enrich 后再 prerank 一次，让新补到的 metadata 能进入最终 cheap ranking。
    # 最终 Selector evaluator 还会重新执行 metadata constraints，因此 enrichment 后
    # 新获得的 year/venue/type 等字段不会被旧 hard_filter_status“锁死”。
    # -------------------------------------------------------------------------
    enabled_sources = get_enabled_sources()
    adapters = _build_adapters(enabled_sources)

    enrichment_started = time.perf_counter()
    enrichment_api_calls = await pool.enrich(
        adapters,
        top_n=min(SELECTOR_TOP_N, len(pool.candidates)),
        required_fields=ENRICH_REQUIRED_FIELDS,
    )
    enrichment_ms = (
        time.perf_counter() - enrichment_started
    ) * 1000.0

    pool.prerank(
        analysis,
        generation,
        num_enabled_sources=len(enabled_sources),
        num_planned_routes=len(execution_plan.routes),
    )

    selector_input = pool.top(
        min(SELECTOR_TOP_N, len(pool.candidates))
    )
    assert selector_input, (
        f"{case.name}: no candidate available for Selector."
    )

    _print_selector_input(selector_input)

    if pool.warnings:
        print("\n---------- CANDIDATE POOL / ENRICH WARNINGS ----------")
        for warning in pool.warnings:
            print(f"- {warning}")

    print("\n========== ENRICHMENT ==========")
    print(f"api_calls            = {enrichment_api_calls}")
    print(f"enriched_candidates  = {pool.stats.enriched_candidates}")
    print(f"latency_ms           = {enrichment_ms:.2f}")

    # -------------------------------------------------------------------------
    # 3) Real Qwen Semantic Selector
    # -------------------------------------------------------------------------
    selector = _build_selector()
    selector_output = await selector.select(
        analysis,
        selector_input,
    )

    _assert_selector_health(
        selector_output,
        selector_input,
    )
    _print_selector_output(
        selector_output,
        selector_input,
    )

    total_wall_ms = (
        time.perf_counter() - total_started
    ) * 1000.0

    run = SelectorPipelineRun(
        case=case,
        analysis=analysis,
        generation=generation,
        execution_plan=execution_plan,
        retrieval=retrieval,
        pool=pool,
        selector_input=selector_input,
        selector_output=selector_output,
        enrichment_api_calls=enrichment_api_calls,
        enrichment_ms=enrichment_ms,
        total_wall_ms=total_wall_ms,
    )
    _print_combined_efficiency(run)

    return run


# =============================================================================
# Real test
# =============================================================================


@pytest.mark.skipif(
    not RUN_REAL,
    reason=(
        "Set RUN_REAL_SELECTOR_PIPELINE_TESTS=1 "
        "to run Static Retrieval -> Qwen Selector real pipeline tests."
    ),
)
@pytest.mark.parametrize(
    "case",
    _selected_cases(),
    ids=lambda case: case.name.replace(" ", "_"),
)
def test_real_static_retrieval_to_qwen_selector(case: PipelineCase):
    run = asyncio.run(_run_selector_pipeline(case))

    output = run.selector_output

    # 最终至少产生一个可解释的 Selector result。
    assert output.results

    # 不要求 live model 必须固定 PASS/FAIL 某一篇；
    # 真实模型/API存在小幅波动。这里验收结构与逻辑健康性。
    assert (
        len(output.selected_ids)
        + len(output.uncertain_ids)
        + len(output.rejected_ids)
        == len(run.selector_input)
    )

    print(
        f"\n[PASS] {case.name}: Static Retrieval -> Candidate Pool -> "
        "Enrichment -> Qwen Semantic Selector -> "
        "PASS/UNKNOWN/FAIL completed."
    )
