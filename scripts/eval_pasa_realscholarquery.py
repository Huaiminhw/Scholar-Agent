# scripts/eval_pasa_realscholarquery.py
r"""
用当前 Scholar-Agent-V2 完整主链路评测 PaSa RealScholarQuery。

默认：
- 数据集：PaSa RealScholarQuery/test.jsonl
- 测试前 10 条
- 完整链路：
    Live Query Understanding
      -> Live Query Generation
      -> ScholarRetrievalPipeline
      -> adaptive 28->50 online retrieval
      -> Citation Snowball + channel weight
      -> Candidate Pool + Hard Filter + RRF
      -> Enrichment
      -> LLM Semantic Selector
      -> PASS / UNKNOWN / FAIL
- 主指标：
    * Retrieval Recall（Selector 前）
    * Recall@20 / Recall@50 / Recall@100
    * Selector PASS Precision / Recall / F1
    * Recall-safe PASS+UNKNOWN Precision / Recall / F1（诊断）
- 标题匹配方式：
    与 PaSa 官方 metrics.py 保持一致：仅保留字母并 lowercase。

注意：
1. 这是“当前系统 vs PaSa Gold Answers”的测试，不是 PaSa 原版 Agent 复现。
2. PaSa 原版运行时会使用 source_meta.published_time - 7 days 作为搜索截止时间；
   我们当前 OpenAlex/S2/arXiv Static Pipeline 没有统一的日级 benchmark cutoff，
   因此默认不偷偷改变生产检索逻辑。结果会明确标记 temporal_cutoff_applied=False。
3. 每条 query 都会立即保存 checkpoint；中途 Ctrl+C / API 异常不会丢掉已完成结果。

Windows PowerShell 示例：

    $env:PASA_REALSCHOLARQUERY_PATH="D:\data\pasa\data\RealScholarQuery\test.jsonl"
    $env:PASA_EVAL_LIMIT="10"
    python scripts/eval_pasa_realscholarquery.py

如果数据就在项目目录：
    data\RealScholarQuery\test.jsonl
则无需设置 PASA_REALSCHOLARQUERY_PATH。

可调参数：
    PASA_EVAL_START=0
    PASA_EVAL_LIMIT=10
    PASA_SELECTOR_TOP_N=30
    PASA_ENRICH_TOP_N=30
    PASA_SLEEP_SECONDS=1.0
    PASA_RESULTS_DIR=results/pasa_eval

若 Semantic Scholar 持续 429，可临时：
    $env:USE_SEMANTIC_SCHOLAR="false"

但正式比较时请固定所有检索源配置，避免 A/B 不公平。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 确保直接 `python scripts/...py` 时项目根目录可 import。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scholar_agent_v2.candidate_pool.rerank import RerankConfig
from scholar_agent_v2.query_generation.generator import GeneratorConfig, QueryGenerator
from scholar_agent_v2.query_understanding.analyzer import AnalyzerConfig, QueryAnalyzer
from scholar_agent_v2.retrieval.config import (
    RetrievalProfile,
    get_enabled_sources,
    get_retrieval_profile,
)
from scholar_agent_v2.retrieval.pipeline import ScholarRetrievalPipeline
from scholar_agent_v2.selector import CandidateSelector, SelectorConfig
from scholar_agent_v2.selector.backends import (
    LLMCriteriaBackend,
    LLMCriteriaBackendConfig,
)

from tests.real_llm_client import build_client


ENRICH_REQUIRED_FIELDS = (
    "abstract",
    "venue",
    "citation_count",
    "publication_types",
    "fields_of_study",
)


# =============================================================================
# Configuration
# =============================================================================

START = int(os.getenv("PASA_EVAL_START", "0"))
LIMIT = int(os.getenv("PASA_EVAL_LIMIT", "10"))
SELECTOR_TOP_N = int(os.getenv("PASA_SELECTOR_TOP_N", "30"))
ENRICH_TOP_N = int(os.getenv("PASA_ENRICH_TOP_N", str(SELECTOR_TOP_N)))
SLEEP_SECONDS = float(os.getenv("PASA_SLEEP_SECONDS", "1.0"))
RETRIEVAL_MAX_CONCURRENCY = int(
    os.getenv("RETRIEVAL_MAX_CONCURRENCY", "6")
)
RETRIEVAL_ROUTE_TIMEOUT = float(
    os.getenv("RETRIEVAL_ROUTE_TIMEOUT", "30")
)
RETRIEVAL_MAX_ATTEMPTS = int(
    os.getenv("RETRIEVAL_MAX_ATTEMPTS", "2")
)
SELECTOR_BATCH_SIZE = int(os.getenv("SELECTOR_BATCH_SIZE", "5"))
SELECTOR_MAX_CONCURRENCY = int(
    os.getenv("SELECTOR_MAX_CONCURRENCY", "2")
)

RESULTS_DIR = Path(
    os.getenv(
        "PASA_RESULTS_DIR",
        str(PROJECT_ROOT / "results" / "pasa_eval"),
    )
)

DATASET_ENV = os.getenv("PASA_REALSCHOLARQUERY_PATH", "").strip()

DEFAULT_DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "RealScholarQuery" / "test.jsonl",
    PROJECT_ROOT / "pasa" / "data" / "RealScholarQuery" / "test.jsonl",
    PROJECT_ROOT.parent / "pasa" / "data" / "RealScholarQuery" / "test.jsonl",
]


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class PRF:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


@dataclass
class QueryEvalResult:
    index: int
    question: str
    gold_count: int
    gold_titles: list[str]

    status: str = "success"
    failed_stage: str | None = None
    error: str | None = None

    source_published_time: str | None = None
    temporal_cutoff_applied: bool = False

    query_analysis_api_calls: int = 0
    query_analysis_tokens: int = 0
    query_analysis_ms: float = 0.0
    query_analysis_repaired: bool = False

    query_generation_api_calls: int = 0
    query_generation_tokens: int = 0
    query_generation_ms: float = 0.0
    generated_queries: list[dict[str, Any]] = field(default_factory=list)
    generation_validation_errors: list[str] = field(default_factory=list)

    retrieval_api_calls: int = 0
    retrieval_ms: float = 0.0
    retrieval_rounds: int = 0
    retrieval_stop_reason: str | None = None
    retrieval_source_hit_counts: dict[str, int] = field(default_factory=dict)
    retrieval_source_api_calls: dict[str, int] = field(default_factory=dict)
    retrieval_source_latencies_ms: dict[str, float] = field(default_factory=dict)
    retrieval_round_details: list[dict[str, Any]] = field(default_factory=list)
    raw_hits: int = 0
    unique_candidates: int = 0
    candidates_after_hard_filter: int = 0
    citation_expansion: dict[str, Any] = field(default_factory=dict)
    top_candidates: list[dict[str, Any]] = field(default_factory=list)

    enrichment_api_calls: int = 0
    enrichment_ms: float = 0.0

    selector_input_count: int = 0
    selector_pass_count: int = 0
    selector_unknown_count: int = 0
    selector_fail_count: int = 0
    selector_api_calls: int = 0
    selector_tokens: int = 0
    selector_ms: float = 0.0

    retrieval_recall: float = 0.0
    recall_at_20: float = 0.0
    recall_at_50: float = 0.0
    recall_at_100: float = 0.0

    strict_pass: PRF | None = None
    recall_safe_pass_unknown: PRF | None = None

    pass_titles: list[str] = field(default_factory=list)
    unknown_titles: list[str] = field(default_factory=list)
    fail_titles: list[str] = field(default_factory=list)

    matched_gold_pass: list[str] = field(default_factory=list)
    missed_gold: list[str] = field(default_factory=list)
    false_positive_pass: list[str] = field(default_factory=list)

    total_wall_ms: float = 0.0


# =============================================================================
# PaSa-compatible title matching
# =============================================================================


def keep_letters(text: str) -> str:
    """
    与 PaSa 官方 utils.keep_letters() 语义一致：
    仅保留 isalpha() 字符并 lowercase。
    """
    return "".join(ch for ch in str(text) if ch.isalpha()).lower()


def normalized_title_map(titles: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for title in titles:
        if not isinstance(title, str):
            continue
        title = title.strip()
        if not title:
            continue
        key = keep_letters(title)
        if key:
            out.setdefault(key, title)
    return out


def prf(pred_titles: Iterable[str], gold_titles: Iterable[str]) -> PRF:
    pred = set(normalized_title_map(pred_titles))
    gold = set(normalized_title_map(gold_titles))

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return PRF(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def recall_at_k(
    ranked_titles: list[str],
    gold_titles: list[str],
    k: int,
) -> float:
    gold = set(normalized_title_map(gold_titles))
    if not gold:
        return 0.0

    pred = set(normalized_title_map(ranked_titles[:k]))
    return len(pred & gold) / len(gold)


# =============================================================================
# Dataset
# =============================================================================


def resolve_dataset_path() -> Path:
    if DATASET_ENV:
        path = Path(DATASET_ENV).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(
            f"PASA_REALSCHOLARQUERY_PATH does not exist: {path}"
        )

    for path in DEFAULT_DATASET_CANDIDATES:
        if path.is_file():
            return path

    searched = "\n".join(f"  - {p}" for p in DEFAULT_DATASET_CANDIDATES)
    raise FileNotFoundError(
        "RealScholarQuery/test.jsonl not found.\n"
        "Set PASA_REALSCHOLARQUERY_PATH to the real local file.\n"
        f"Auto-checked:\n{searched}"
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_no}: {exc}"
                ) from exc

            if "question" not in obj:
                raise ValueError(
                    f"RealScholarQuery line {line_no} has no 'question'. "
                    f"keys={list(obj)}"
                )
            if "answer" not in obj:
                raise ValueError(
                    f"RealScholarQuery line {line_no} has no 'answer'. "
                    f"keys={list(obj)}"
                )
            if not isinstance(obj["answer"], list):
                raise ValueError(
                    f"RealScholarQuery line {line_no} 'answer' must be list."
                )
            records.append(obj)

    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


# =============================================================================
# Builders
# =============================================================================


def build_analyzer(client) -> QueryAnalyzer:
    return QueryAnalyzer(
        client,
        config=AnalyzerConfig(
            temperature=0.0,
            enable_deterministic_router=True,
            use_structured_output=False,
            prompt_schema_mode="summary",
            include_few_shot_examples=False,
            enable_repair=True,
            repair_schema_mode="full",
            max_transport_retries=0,
            retain_raw_output=False,
        ),
    )


def build_generator(client) -> QueryGenerator:
    return QueryGenerator(
        client=client,
        config=GeneratorConfig(
            temperature=0.0,
            use_structured_output=False,
            enable_llm_expansion=True,
            max_transport_retries=0,
            max_total_queries=5,
            near_duplicate_threshold=0.88,
        ),
    )


def build_pipeline() -> ScholarRetrievalPipeline:
    if get_retrieval_profile() != RetrievalProfile.COMPETITION_ONLINE:
        raise RuntimeError(
            "PaSa live evaluation requires "
            "RETRIEVAL_PROFILE=competition_online"
        )
    return ScholarRetrievalPipeline.from_environment(
        max_concurrency=RETRIEVAL_MAX_CONCURRENCY,
        route_timeout_seconds=RETRIEVAL_ROUTE_TIMEOUT,
        max_attempts=RETRIEVAL_MAX_ATTEMPTS,
    )


def build_selector(client) -> CandidateSelector:
    backend = LLMCriteriaBackend(
        client,
        config=LLMCriteriaBackendConfig(
            batch_size=SELECTOR_BATCH_SIZE,
            max_concurrency=SELECTOR_MAX_CONCURRENCY,
            temperature=0.0,
            use_structured_output=False,
            enable_repair=True,
            max_transport_retries=0,
            fail_open=True,
        ),
    )
    return CandidateSelector(
        backend,
        config=SelectorConfig(max_candidates=SELECTOR_TOP_N),
    )


# =============================================================================
# One-query evaluation
# =============================================================================


async def evaluate_one(
    *,
    index: int,
    record: dict[str, Any],
    analyzer: QueryAnalyzer,
    generator: QueryGenerator,
    pipeline: ScholarRetrievalPipeline,
    selector: CandidateSelector,
) -> QueryEvalResult:
    question = str(record["question"]).strip()
    gold_titles = [
        str(x).strip()
        for x in record.get("answer", [])
        if str(x).strip()
    ]
    source_meta = record.get("source_meta") or {}
    source_published_time = source_meta.get("published_time")

    result = QueryEvalResult(
        index=index,
        question=question,
        gold_count=len(gold_titles),
        gold_titles=gold_titles,
        source_published_time=source_published_time,
        temporal_cutoff_applied=False,
    )

    total_started = time.perf_counter()
    current_stage = "query_understanding"

    try:
        # ---------------------------------------------------------------------
        # 1) Real Query Understanding
        # ---------------------------------------------------------------------
        qa = await analyzer.analyze(question)
        analysis = qa.result

        result.query_analysis_api_calls = qa.metrics.api_calls
        result.query_analysis_tokens = qa.metrics.total_tokens
        result.query_analysis_ms = qa.metrics.latency_ms
        result.query_analysis_repaired = qa.repaired

        # ---------------------------------------------------------------------
        # 2) Real Query Generation
        # ---------------------------------------------------------------------
        current_stage = "query_generation"
        generation = await generator.generate(analysis)

        result.query_generation_api_calls = generation.metrics.api_calls
        result.query_generation_tokens = generation.metrics.total_tokens
        result.query_generation_ms = generation.metrics.latency_ms
        result.generated_queries = [
            {
                "query_id": q.query_id,
                "kind": q.kind.value,
                "priority": q.priority,
                "text": q.text,
            }
            for q in generation.result.queries
        ]
        result.generation_validation_errors = [
            f"{issue.code}: {issue.message}"
            for issue in generation.validation.errors
        ]

        # 对 benchmark 不让单条坏 Query 把前 10 条全部打断；
        # 但验证错误必须被记录成该 query 的系统失败，不能静默忽略。
        if generation.validation.errors:
            raise RuntimeError(
                "Query Generation validation error(s): "
                + " | ".join(result.generation_validation_errors)
            )

        # ---------------------------------------------------------------------
        # 3) Current production pipeline + real online retrieval
        # ---------------------------------------------------------------------
        current_stage = "retrieval"
        pipeline_output = await pipeline.run(analysis, generation.result)
        retrieval = pipeline_output.retrieval
        pool = pipeline_output.candidate_pool
        result.retrieval_api_calls = retrieval.metrics.api_calls
        result.retrieval_ms = retrieval.metrics.total_latency_ms
        result.retrieval_rounds = retrieval.metrics.retrieval_rounds
        result.retrieval_stop_reason = retrieval.metrics.stop_reason
        result.retrieval_source_hit_counts = dict(
            retrieval.metrics.source_hit_counts
        )
        result.retrieval_source_api_calls = dict(
            retrieval.metrics.source_api_calls
        )
        result.retrieval_source_latencies_ms = dict(
            retrieval.metrics.source_latencies_ms
        )
        result.retrieval_round_details = [
            round_result.metrics.model_dump(mode="json")
            for round_result in retrieval.rounds
        ]
        result.raw_hits = retrieval.metrics.raw_hits
        result.citation_expansion = (
            pipeline_output.citation_expansion.diagnostics.model_dump(
                mode="json"
            )
        )

        if not retrieval.hits and not pipeline_output.citation_expansion.hits:
            raise RuntimeError("Retrieval returned zero hits.")

        # ---------------------------------------------------------------------
        # 4) Candidate Pool + hard filter + cheap rerank
        # ---------------------------------------------------------------------
        current_stage = "candidate_pool"
        result.unique_candidates = pool.stats.candidates_before_filter
        result.candidates_after_hard_filter = len(pool.candidates)

        ranked_before_selector = pool.top(
            max(100, len(pool.candidates))
        )
        ranked_titles = [
            c.title
            for c in ranked_before_selector
            if c.title
        ]
        result.top_candidates = [
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "title": candidate.title,
                "year": candidate.year,
                "identifiers": candidate.identifiers.model_dump(mode="json"),
                "retrieval_score": candidate.retrieval_score,
                "score_components": candidate.score_components,
                "sources": sorted(
                    {origin.source.value for origin in candidate.origins}
                ),
                "route_ids": sorted(
                    {origin.route_id for origin in candidate.origins}
                ),
                "citation_expanded": any(
                    origin.strategy.value == "citation"
                    for origin in candidate.origins
                ),
            }
            for rank, candidate in enumerate(
                ranked_before_selector[:100],
                start=1,
            )
        ]

        result.retrieval_recall = recall_at_k(
            ranked_titles,
            gold_titles,
            max(1, len(ranked_titles)),
        )
        result.recall_at_20 = recall_at_k(
            ranked_titles,
            gold_titles,
            20,
        )
        result.recall_at_50 = recall_at_k(
            ranked_titles,
            gold_titles,
            50,
        )
        result.recall_at_100 = recall_at_k(
            ranked_titles,
            gold_titles,
            100,
        )

        if not pool.candidates:
            # 仍然产生有效的 0 分 benchmark 记录。
            result.strict_pass = prf([], gold_titles)
            result.recall_safe_pass_unknown = prf([], gold_titles)
            result.total_wall_ms = (
                time.perf_counter() - total_started
            ) * 1000.0
            return result

        # ---------------------------------------------------------------------
        # 5) Enrichment
        # ---------------------------------------------------------------------
        current_stage = "enrichment"
        enrichment_started = time.perf_counter()
        try:
            result.enrichment_api_calls = await pool.enrich(
                pipeline.adapters,
                top_n=min(
                    ENRICH_TOP_N,
                    len(pool.candidates),
                ),
                required_fields=ENRICH_REQUIRED_FIELDS,
            )
        finally:
            result.enrichment_ms = (
                time.perf_counter() - enrichment_started
            ) * 1000.0

        # Enrichment 后重新便宜排序。
        pool.prerank(
            analysis,
            generation.result,
            num_enabled_sources=len(pipeline.enabled_sources),
            num_planned_routes=len(retrieval.plan.routes),
            config=RerankConfig(
                citation_channel_weight=(
                    pipeline.citation_config.channel_weight
                )
            ),
        )

        # ---------------------------------------------------------------------
        # 6) Qwen Semantic Selector
        # ---------------------------------------------------------------------
        current_stage = "selector"
        selector_input = pool.top(
            min(
                SELECTOR_TOP_N,
                len(pool.candidates),
            )
        )
        result.selector_input_count = len(selector_input)

        selector_output = await selector.select(
            analysis,
            selector_input,
        )

        result.selector_pass_count = len(
            selector_output.selected_ids
        )
        result.selector_unknown_count = len(
            selector_output.uncertain_ids
        )
        result.selector_fail_count = len(
            selector_output.rejected_ids
        )
        result.selector_api_calls = (
            selector_output.metrics.llm_api_calls
        )
        result.selector_tokens = (
            selector_output.metrics.llm_total_tokens
        )
        result.selector_ms = (
            selector_output.metrics.total_latency_ms
        )

        by_id = {
            c.candidate_id: c
            for c in selector_input
        }

        pass_titles = [
            by_id[cid].title
            for cid in selector_output.selected_ids
            if cid in by_id and by_id[cid].title
        ]
        unknown_titles = [
            by_id[cid].title
            for cid in selector_output.uncertain_ids
            if cid in by_id and by_id[cid].title
        ]
        fail_titles = [
            by_id[cid].title
            for cid in selector_output.rejected_ids
            if cid in by_id and by_id[cid].title
        ]

        result.pass_titles = pass_titles
        result.unknown_titles = unknown_titles
        result.fail_titles = fail_titles

        # 主指标：PASS only。
        result.strict_pass = prf(
            pass_titles,
            gold_titles,
        )

        # 诊断指标：PASS + UNKNOWN。
        result.recall_safe_pass_unknown = prf(
            [*pass_titles, *unknown_titles],
            gold_titles,
        )

        gold_map = normalized_title_map(gold_titles)
        pass_map = normalized_title_map(pass_titles)

        matched_keys = set(gold_map) & set(pass_map)
        missed_keys = set(gold_map) - set(pass_map)
        fp_keys = set(pass_map) - set(gold_map)

        result.matched_gold_pass = [
            gold_map[k]
            for k in sorted(matched_keys)
        ]
        result.missed_gold = [
            gold_map[k]
            for k in sorted(missed_keys)
        ]
        result.false_positive_pass = [
            pass_map[k]
            for k in sorted(fp_keys)
        ]

        result.total_wall_ms = (
            time.perf_counter() - total_started
        ) * 1000.0
        return result

    except Exception as exc:
        result.status = "failed"
        result.failed_stage = current_stage
        result.error = (
            f"{type(exc).__name__}: {exc}"
        )
        result.total_wall_ms = (
            time.perf_counter() - total_started
        ) * 1000.0

        # 保留 traceback 到 console，JSON 只存紧凑错误。
        print(
            f"\n[ERROR] query #{index} failed at "
            f"{current_stage}: {result.error}"
        )
        traceback.print_exc()
        return result


# =============================================================================
# Aggregate
# =============================================================================


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(results: list[QueryEvalResult]) -> dict[str, Any]:
    successful = [
        r
        for r in results
        if r.status == "success"
    ]
    failed = [
        r
        for r in results
        if r.status != "success"
    ]

    strict = [
        r.strict_pass
        for r in successful
        if r.strict_pass is not None
    ]
    lenient = [
        r.recall_safe_pass_unknown
        for r in successful
        if r.recall_safe_pass_unknown is not None
    ]

    strict_tp = sum(x.tp for x in strict)
    strict_fp = sum(x.fp for x in strict)
    strict_fn = sum(x.fn for x in strict)

    micro_precision = (
        strict_tp / (strict_tp + strict_fp)
        if (strict_tp + strict_fp)
        else 0.0
    )
    micro_recall = (
        strict_tp / (strict_tp + strict_fn)
        if (strict_tp + strict_fn)
        else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    source_names = sorted(
        {
            source
            for result in results
            for source in result.retrieval_source_api_calls
        }
    )

    return {
        "requested_queries": len(results),
        "successful_queries": len(successful),
        "failed_queries": len(failed),
        "failed_indices": [r.index for r in failed],

        # PaSa metrics.py 的主要展示是“逐 query 指标后取平均”；
        # 我们在这里同时给 macro 和 global micro。
        "macro": {
            "retrieval_recall": mean(
                [r.retrieval_recall for r in successful]
            ),
            "recall_at_20": mean(
                [r.recall_at_20 for r in successful]
            ),
            "recall_at_50": mean(
                [r.recall_at_50 for r in successful]
            ),
            "recall_at_100": mean(
                [r.recall_at_100 for r in successful]
            ),
            "selector_pass_precision": mean(
                [x.precision for x in strict]
            ),
            "selector_pass_recall": mean(
                [x.recall for x in strict]
            ),
            "selector_pass_f1": mean(
                [x.f1 for x in strict]
            ),
            "pass_plus_unknown_precision": mean(
                [x.precision for x in lenient]
            ),
            "pass_plus_unknown_recall": mean(
                [x.recall for x in lenient]
            ),
            "pass_plus_unknown_f1": mean(
                [x.f1 for x in lenient]
            ),
        },

        "global_micro_selector_pass": {
            "tp": strict_tp,
            "fp": strict_fp,
            "fn": strict_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },

        "efficiency": {
            "query_analysis_api_calls": sum(
                r.query_analysis_api_calls
                for r in results
            ),
            "query_generation_api_calls": sum(
                r.query_generation_api_calls
                for r in results
            ),
            "retrieval_api_calls": sum(
                r.retrieval_api_calls
                for r in results
            ),
            "citation_api_calls": sum(
                int(r.citation_expansion.get("api_calls", 0))
                for r in results
            ),
            "citation_latency_ms": sum(
                float(r.citation_expansion.get("latency_ms", 0.0))
                for r in results
            ),
            "retrieval_source_api_calls": {
                source: sum(
                    int(r.retrieval_source_api_calls.get(source, 0))
                    for r in results
                )
                for source in source_names
            },
            "retrieval_source_hits": {
                source: sum(
                    int(r.retrieval_source_hit_counts.get(source, 0))
                    for r in results
                )
                for source in source_names
            },
            "enrichment_api_calls": sum(
                r.enrichment_api_calls
                for r in results
            ),
            "selector_api_calls": sum(
                r.selector_api_calls
                for r in results
            ),
            "llm_total_tokens": sum(
                r.query_analysis_tokens
                + r.query_generation_tokens
                + r.selector_tokens
                for r in results
            ),
            "total_wall_seconds": sum(
                r.total_wall_ms
                for r in results
            ) / 1000.0,
            "mean_wall_seconds_per_query": mean(
                [
                    r.total_wall_ms / 1000.0
                    for r in results
                ]
            ),
        },

        "benchmark_notes": {
            "dataset": "PaSa RealScholarQuery",
            "pipeline": "ScholarRetrievalPipeline",
            "query_execution_mode": "live",
            "retrieval_profile": "competition_online",
            "title_normalization": (
                "PaSa-compatible keep_letters: "
                "alphabetic chars only + lowercase"
            ),
            "primary_selector_metric": "PASS only",
            "unknown_metric": (
                "PASS+UNKNOWN is diagnostic only"
            ),
            "temporal_cutoff_applied": False,
            "warning": (
                "PaSa original runner uses "
                "source_meta.published_time minus 7 days "
                "as search end_date. Current Scholar-Agent "
                "pipeline is evaluated against current scholarly "
                "indexes without a unified day-level cutoff."
            ),
        },
    }


# =============================================================================
# Output
# =============================================================================


def format_prf(x: PRF | None) -> str:
    if x is None:
        return "N/A"
    return (
        f"P={x.precision:.4f} "
        f"R={x.recall:.4f} "
        f"F1={x.f1:.4f} "
        f"(TP={x.tp}, FP={x.fp}, FN={x.fn})"
    )


def print_query_result(result: QueryEvalResult) -> None:
    print("\n" + "=" * 88)
    print(
        f"PaSa RealScholarQuery #{result.index}"
    )
    print("=" * 88)
    print(f"question = {result.question}")
    print(f"gold_count = {result.gold_count}")
    print(
        "source_published_time = "
        f"{result.source_published_time}"
    )
    print(f"status = {result.status}")

    if result.status != "success":
        print(
            f"failed_stage = {result.failed_stage}"
        )
        print(f"error = {result.error}")
        print(
            f"wall_s = {result.total_wall_ms / 1000.0:.2f}"
        )
        return

    print("\n---------- RETRIEVAL ----------")
    print(f"raw_hits              = {result.raw_hits}")
    print(
        f"unique_candidates     = "
        f"{result.unique_candidates}"
    )
    print(
        f"after_hard_filter     = "
        f"{result.candidates_after_hard_filter}"
    )
    print(
        f"retrieval_recall      = "
        f"{result.retrieval_recall:.4f}"
    )
    print(f"Recall@20             = {result.recall_at_20:.4f}")
    print(f"Recall@50             = {result.recall_at_50:.4f}")
    print(f"Recall@100            = {result.recall_at_100:.4f}")
    print(
        f"adaptive rounds/stop  = {result.retrieval_rounds}/"
        f"{result.retrieval_stop_reason}"
    )
    print(
        "source hits            = "
        f"{result.retrieval_source_hit_counts}"
    )
    print(
        "source API calls       = "
        f"{result.retrieval_source_api_calls}"
    )
    if result.citation_expansion:
        print(
            "citation new/stop      = "
            f"{result.citation_expansion.get('new_unique_candidates', 0)}/"
            f"{result.citation_expansion.get('stop_reason')}"
        )

    print("\n---------- SELECTOR ----------")
    print(
        f"input/pass/unknown/fail = "
        f"{result.selector_input_count}/"
        f"{result.selector_pass_count}/"
        f"{result.selector_unknown_count}/"
        f"{result.selector_fail_count}"
    )
    print(
        "PASS only             = "
        + format_prf(result.strict_pass)
    )
    print(
        "PASS + UNKNOWN        = "
        + format_prf(result.recall_safe_pass_unknown)
    )

    print("\n---------- GOLD MATCH ----------")
    print(
        f"matched_gold_pass = "
        f"{len(result.matched_gold_pass)}"
    )
    for title in result.matched_gold_pass[:20]:
        print(f"  [TP] {title}")

    print(
        f"missed_gold = {len(result.missed_gold)}"
    )
    for title in result.missed_gold[:20]:
        print(f"  [FN] {title}")

    print(
        f"false_positive_pass = "
        f"{len(result.false_positive_pass)}"
    )
    for title in result.false_positive_pass[:20]:
        print(f"  [FP] {title}")

    print("\n---------- EFFICIENCY ----------")
    print(
        "QA/QGen/Retrieval/Enrich/Selector calls = "
        f"{result.query_analysis_api_calls}/"
        f"{result.query_generation_api_calls}/"
        f"{result.retrieval_api_calls}/"
        f"{result.enrichment_api_calls}/"
        f"{result.selector_api_calls}"
    )
    print(
        "QA/QGen/Selector tokens = "
        f"{result.query_analysis_tokens}/"
        f"{result.query_generation_tokens}/"
        f"{result.selector_tokens}"
    )
    print(
        f"wall_s = "
        f"{result.total_wall_ms / 1000.0:.2f}"
    )


def save_checkpoint(
    *,
    dataset_path: Path,
    results: list[QueryEvalResult],
    output_path: Path,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "protocol": {
            "name": "pasa_live_competition_pipeline_v4",
            "query_execution_mode": "live",
            "retrieval_profile": "competition_online",
            "llm_runtime": True,
            "network": True,
            "pipeline": "ScholarRetrievalPipeline",
            "adaptive_depth": True,
            "citation_snowball": os.getenv(
                "USE_CITATION_SNOWBALL", "false"
            ).strip().lower() in {"1", "true", "yes", "on"},
            "citation_channel_weight": float(
                os.getenv("CITATION_CHANNEL_WEIGHT", "0.70")
            ),
            "temporal_cutoff_applied": False,
        },
        "dataset_path": str(dataset_path),
        "start": START,
        "limit": LIMIT,
        "selector_top_n": SELECTOR_TOP_N,
        "enabled_sources": [
            s.value
            for s in get_enabled_sources()
        ],
        "results": [
            asdict(r)
            for r in results
        ],
        "aggregate": aggregate(results),
    }
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# Main
# =============================================================================


async def main() -> int:
    dataset_path = resolve_dataset_path()
    records = load_records(dataset_path)

    start = max(0, START)
    end = min(
        len(records),
        start + max(0, LIMIT),
    )
    selected_records = records[start:end]

    if not selected_records:
        raise ValueError(
            f"No records selected: start={start}, "
            f"limit={LIMIT}, dataset_size={len(records)}"
        )

    client = build_client()
    analyzer = build_analyzer(client)
    generator = build_generator(client)
    pipeline = build_pipeline()
    selector = build_selector(client)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    output_path = (
        RESULTS_DIR
        / f"realscholarquery_live_v4_{start}_{end - 1}_{stamp}.json"
    )

    print("=" * 88)
    print("PaSa RealScholarQuery -> Scholar-Agent-V2 Benchmark")
    print("=" * 88)
    print(f"dataset       = {dataset_path}")
    print(f"dataset_size  = {len(records)}")
    print(f"range         = [{start}, {end})")
    print(f"query_count   = {len(selected_records)}")
    print(f"selector_topN = {SELECTOR_TOP_N}")
    print(
        "sources       = "
        f"{[s.value for s in get_enabled_sources()]}"
    )
    print(
        "temporal_cutoff_applied = False "
        "(see benchmark note)"
    )
    print(
        "citation      = "
        f"enabled:{pipeline.citation_config.enabled}, "
        f"weight:{pipeline.citation_config.channel_weight:.2f}"
    )
    print(f"checkpoint    = {output_path}")

    results: list[QueryEvalResult] = []

    for offset, record in enumerate(
        selected_records,
        start=start,
    ):
        print("\n" + "#" * 88)
        print(
            f"RUNNING QUERY {offset} "
            f"({len(results) + 1}/{len(selected_records)})"
        )
        print("#" * 88)

        result = await evaluate_one(
            index=offset,
            record=record,
            analyzer=analyzer,
            generator=generator,
            pipeline=pipeline,
            selector=selector,
        )
        results.append(result)

        print_query_result(result)

        # 每题保存一次。
        save_checkpoint(
            dataset_path=dataset_path,
            results=results,
            output_path=output_path,
        )

        if (
            SLEEP_SECONDS > 0
            and len(results) < len(selected_records)
        ):
            await asyncio.sleep(SLEEP_SECONDS)

    agg = aggregate(results)

    print("\n" + "=" * 88)
    print("FINAL FIRST-N SUMMARY")
    print("=" * 88)
    print(
        json.dumps(
            agg,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nSaved to: {output_path}")

    # Benchmark runner 本身不因为某几题失败而丢失结果；
    # 但 process exit code=1 提醒 CI / 用户本轮并非全成功。
    return 0 if agg["failed_queries"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
