# scripts/eval_scholarquest.py
r"""
ScholarQuest -> Scholar-Agent-V2 真实端到端评测。

默认跑 ScholarQuest.jsonl 的前 10 条，完整走当前系统：

    Query Understanding
      -> Query Generation
      -> Static Routing
      -> OpenAlex / Semantic Scholar / arXiv
      -> Candidate Pool
      -> Hard Filter + RRF
      -> Enrichment
      -> Qwen Semantic Selector
      -> arXiv-ID Gold Evaluation

ScholarQuest 官方主指标是 Recall@25 / Recall@100 / Recall@All。
本脚本因此把 Recall 作为主指标，并额外给出：
- Candidate Pool 进入 Selector 前的 Recall
- Selector PASS Recall
- PASS + UNKNOWN Recall（Recall-safe 诊断）
- Selector conditional recall（只在已进入 Selector 的 Gold 上计算）
- 可识别 arXiv-ID 的 PASS precision（诊断项，不是 ScholarQuest 官方主指标）

重要公平性说明
--------------
ScholarQuest 论文的正式实验使用固定 ScholarBase 环境（约 3M arXiv papers），
支持 BM25 / BGE-M3 dense / RRF hybrid 和 citation/reference traversal。
我们当前系统使用实时 OpenAlex / Semantic Scholar / arXiv API，因此：
    * 这次结果非常适合做“我们自己的真实诊断 / A-B”
    * 不能直接把绝对 Recall 数字与 ScholarQuest 论文表格视为严格同环境对比

数据格式（官方 datasets/ScholarQuest.jsonl）：
{
  "query_id": "BQ_000001",
  "final_query": "...",
  "answer_arxiv_ids": ["xxxx.xxxxx", ...],
  "final_answer_count": 12,
  "source_dir": "..."
}

Windows PowerShell
------------------
如果数据在项目内：
    data\ScholarQuest\ScholarQuest.jsonl
或：
    data\ScholarQuest.jsonl

直接：
    $env:SCHOLARQUEST_EVAL_START="0"
    $env:SCHOLARQUEST_EVAL_LIMIT="10"
    python scripts/eval_scholarquest.py

如果数据在别处：
    $env:SCHOLARQUEST_DATASET_PATH="D:\data\ScholarQuest\datasets\ScholarQuest.jsonl"
    $env:SCHOLARQUEST_METADATA_PATH="D:\data\ScholarQuest\datasets\query_metadata.jsonl"
    $env:SCHOLARQUEST_EVAL_START="0"
    $env:SCHOLARQUEST_EVAL_LIMIT="10"
    python scripts/eval_scholarquest.py

可调：
    SCHOLARQUEST_SELECTOR_TOP_N=30
    SCHOLARQUEST_ENRICH_TOP_N=30
    SCHOLARQUEST_SLEEP_SECONDS=1.0
    SCHOLARQUEST_RESULTS_DIR=results/scholarquest_eval

如果 S2 持续 429，可临时关闭：
    $env:USE_SEMANTIC_SCHOLAR="false"

但用于不同版本 A/B 时，请固定相同 sources / topN / route budget。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Runtime config -- read BEFORE importing test_selector_pipeline_real
# =============================================================================

START = int(os.getenv("SCHOLARQUEST_EVAL_START", "0"))
LIMIT = int(os.getenv("SCHOLARQUEST_EVAL_LIMIT", "10"))
SELECTOR_TOP_N = int(os.getenv("SCHOLARQUEST_SELECTOR_TOP_N", "30"))
ENRICH_TOP_N = int(
    os.getenv("SCHOLARQUEST_ENRICH_TOP_N", str(SELECTOR_TOP_N))
)
SLEEP_SECONDS = float(os.getenv("SCHOLARQUEST_SLEEP_SECONDS", "1.0"))

RESULTS_DIR = Path(
    os.getenv(
        "SCHOLARQUEST_RESULTS_DIR",
        str(PROJECT_ROOT / "results" / "scholarquest_eval"),
    )
)

# 让复用的 _build_selector() 与本脚本 topN 一致。
os.environ["SELECTOR_PIPELINE_TOP_N"] = str(SELECTOR_TOP_N)

from scholar_agent_v2.candidate_pool.models import CandidatePaper
from scholar_agent_v2.candidate_pool.pool import CandidatePool
from scholar_agent_v2.query_generation.generator import (
    GeneratorConfig,
    QueryGenerator,
)
from scholar_agent_v2.query_understanding.analyzer import (
    AnalyzerConfig,
    QueryAnalyzer,
)
from scholar_agent_v2.retrieval.config import get_enabled_sources
from scholar_agent_v2.retrieval.route_builder import RetrievalRouteBuilder

from tests.real_llm_client import build_client
from tests.test_retrieval_pipeline_real import (
    _build_adapters,
    _build_orchestrator,
    _make_execution_plan,
)
from tests.test_selector_pipeline_real import (
    ENRICH_REQUIRED_FIELDS,
    _build_selector,
)


# =============================================================================
# Dataset discovery
# =============================================================================

DATASET_ENV = os.getenv("SCHOLARQUEST_DATASET_PATH", "").strip()
METADATA_ENV = os.getenv("SCHOLARQUEST_METADATA_PATH", "").strip()

DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "ScholarQuest" / "ScholarQuest.jsonl",
    PROJECT_ROOT / "data" / "ScholarQuest.jsonl",
    PROJECT_ROOT / "datasets" / "ScholarQuest.jsonl",
    PROJECT_ROOT / "ScholarQuest" / "datasets" / "ScholarQuest.jsonl",
    PROJECT_ROOT.parent / "ScholarQuest" / "datasets" / "ScholarQuest.jsonl",
]

METADATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "ScholarQuest" / "query_metadata.jsonl",
    PROJECT_ROOT / "data" / "query_metadata.jsonl",
    PROJECT_ROOT / "datasets" / "query_metadata.jsonl",
    PROJECT_ROOT / "ScholarQuest" / "datasets" / "query_metadata.jsonl",
    PROJECT_ROOT.parent / "ScholarQuest" / "datasets" / "query_metadata.jsonl",
]


def _resolve_file(
    env_value: str,
    candidates: list[Path],
    *,
    required: bool,
    label: str,
) -> Path | None:
    if env_value:
        p = Path(env_value).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"{label} does not exist: {p}")

    for p in candidates:
        if p.is_file():
            return p

    if required:
        checked = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            f"{label} not found.\n"
            f"Set SCHOLARQUEST_DATASET_PATH to the real file.\n"
            f"Auto-checked:\n{checked}"
        )
    return None


def resolve_dataset_path() -> Path:
    p = _resolve_file(
        DATASET_ENV,
        DATASET_CANDIDATES,
        required=True,
        label="ScholarQuest.jsonl",
    )
    assert p is not None
    return p


def resolve_metadata_path() -> Path | None:
    return _resolve_file(
        METADATA_ENV,
        METADATA_CANDIDATES,
        required=False,
        label="query_metadata.jsonl",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL: {path}, line={line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object at {path}:{line_no}, got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for i, row in enumerate(rows):
        missing = [
            key
            for key in ("query_id", "final_query", "answer_arxiv_ids")
            if key not in row
        ]
        if missing:
            raise ValueError(
                f"ScholarQuest row {i} missing fields: {missing}; "
                f"keys={list(row)}"
            )
        if not isinstance(row["answer_arxiv_ids"], list):
            raise ValueError(
                f"row {i}: answer_arxiv_ids must be list"
            )
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def load_metadata_by_query_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        qid = str(row.get("query_id", "")).strip()
        if qid and qid not in out:
            out[qid] = row
    return out


# =============================================================================
# arXiv ID normalization
# =============================================================================

_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([^?#/]+)",
    flags=re.IGNORECASE,
)
_VERSION_RE = re.compile(r"v\d+$", flags=re.IGNORECASE)


def normalize_arxiv_id(value: Any) -> str | None:
    """
    Normalize:
      arXiv:2401.12345v2
      https://arxiv.org/abs/2401.12345
      https://arxiv.org/pdf/2401.12345.pdf
      2401.12345v1
      hep-th/9901001v2
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    m = _ARXIV_URL_RE.search(s)
    if m:
        s = m.group(1)

    s = re.sub(r"^arxiv\s*:\s*", "", s, flags=re.IGNORECASE)
    s = s.split("?")[0].split("#")[0].strip()
    if s.lower().endswith(".pdf"):
        s = s[:-4]
    s = _VERSION_RE.sub("", s)
    s = s.strip().lower()

    return s or None


def normalize_arxiv_set(values: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for value in values:
        n = normalize_arxiv_id(value)
        if n:
            out.add(n)
    return out


def candidate_arxiv_id(candidate: CandidatePaper) -> str | None:
    # 1) Structured identifier
    n = normalize_arxiv_id(candidate.identifiers.arxiv_id)
    if n:
        return n

    # 2) candidate_id may itself be arxiv:...
    cid = str(candidate.candidate_id or "")
    if cid.lower().startswith("arxiv:"):
        n = normalize_arxiv_id(cid)
        if n:
            return n

    # 3) URLs
    for url in candidate.urls:
        n = normalize_arxiv_id(url) if "arxiv.org/" in url.lower() else None
        if n:
            return n

    return None


def candidate_id_map(
    candidates: Iterable[CandidatePaper],
) -> dict[str, CandidatePaper]:
    out: dict[str, CandidatePaper] = {}
    for c in candidates:
        aid = candidate_arxiv_id(c)
        if aid and aid not in out:
            out[aid] = c
    return out


# =============================================================================
# Metrics
# =============================================================================


def recall(pred: set[str], gold: set[str]) -> float:
    return len(pred & gold) / len(gold) if gold else 0.0


def precision(pred: set[str], gold: set[str]) -> float:
    return len(pred & gold) / len(pred) if pred else 0.0


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def recall_at_k(
    ranked_candidates: list[CandidatePaper],
    gold: set[str],
    k: int,
) -> float:
    ids = {
        aid
        for c in ranked_candidates[:k]
        if (aid := candidate_arxiv_id(c))
    }
    return recall(ids, gold)


def matched_gold_titles_or_ids(
    ids: set[str],
    gold: set[str],
) -> list[str]:
    return sorted(ids & gold)


# =============================================================================
# Results
# =============================================================================


@dataclass
class QueryResult:
    index: int
    query_id: str
    query: str
    gold_count: int
    gold_arxiv_ids: list[str]

    category: str | None = None
    domain: str | None = None
    topic_seed: str | None = None
    constraint_kind: str | None = None
    constraint_value: Any | None = None

    status: str = "success"
    failed_stage: str | None = None
    error: str | None = None

    qa_api_calls: int = 0
    qa_tokens: int = 0
    qa_ms: float = 0.0
    qa_repaired: bool = False

    qgen_api_calls: int = 0
    qgen_tokens: int = 0
    qgen_ms: float = 0.0
    generated_queries: list[dict[str, Any]] = field(default_factory=list)

    retrieval_api_calls: int = 0
    retrieval_ms: float = 0.0
    raw_hits: int = 0

    unique_candidates: int = 0
    hard_rejected: int = 0
    after_hard_filter: int = 0

    candidate_arxiv_identifiable: int = 0
    candidate_arxiv_identifiable_fraction: float = 0.0

    retrieval_recall_25: float = 0.0
    retrieval_recall_100: float = 0.0
    retrieval_recall_all: float = 0.0

    gold_found_before_filter: int = 0
    gold_found_after_filter: int = 0
    gold_lost_by_hard_filter: list[str] = field(default_factory=list)

    enrichment_api_calls: int = 0
    enrichment_ms: float = 0.0

    selector_input_count: int = 0
    selector_input_gold_count: int = 0
    selector_input_recall: float = 0.0

    selector_pass_count: int = 0
    selector_unknown_count: int = 0
    selector_fail_count: int = 0
    selector_api_calls: int = 0
    selector_tokens: int = 0
    selector_ms: float = 0.0

    selector_pass_recall: float = 0.0
    selector_pass_unknown_recall: float = 0.0
    selector_conditional_recall: float | None = None

    # Precision 只在“能识别 arXiv ID”的 PASS prediction 上计算。
    selector_pass_identifiable_count: int = 0
    selector_pass_precision_diagnostic: float = 0.0
    selector_pass_f1_diagnostic: float = 0.0

    matched_gold_pass: list[str] = field(default_factory=list)
    matched_gold_unknown: list[str] = field(default_factory=list)
    missed_gold: list[str] = field(default_factory=list)

    pass_papers: list[dict[str, Any]] = field(default_factory=list)
    unknown_papers: list[dict[str, Any]] = field(default_factory=list)
    fail_papers: list[dict[str, Any]] = field(default_factory=list)

    total_wall_ms: float = 0.0


# =============================================================================
# Builders
# =============================================================================


def build_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer(
        build_client(),
        config=AnalyzerConfig(
            temperature=0.0,
            use_structured_output=False,
            prompt_schema_mode="summary",
            include_few_shot_examples=False,
            enable_repair=True,
            repair_schema_mode="full",
            max_transport_retries=0,
            retain_raw_output=False,
        ),
    )


def build_generator() -> QueryGenerator:
    return QueryGenerator(
        client=build_client(),
        config=GeneratorConfig(
            temperature=0.0,
            use_structured_output=False,
            enable_llm_expansion=True,
            max_transport_retries=0,
            max_total_queries=5,
            near_duplicate_threshold=0.88,
        ),
    )


# =============================================================================
# Evaluation
# =============================================================================


async def evaluate_one(
    index: int,
    row: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> QueryResult:
    query_id = str(row["query_id"])
    query = str(row["final_query"]).strip()
    gold = normalize_arxiv_set(row["answer_arxiv_ids"])

    meta = metadata or {}
    result = QueryResult(
        index=index,
        query_id=query_id,
        query=query,
        gold_count=len(gold),
        gold_arxiv_ids=sorted(gold),
        category=meta.get("category"),
        domain=meta.get("domain"),
        topic_seed=meta.get("topic_seed"),
        constraint_kind=meta.get("constraint_kind"),
        constraint_value=meta.get("constraint_value"),
    )

    overall = time.perf_counter()
    stage = "query_understanding"

    try:
        # ---------------------------------------------------------------------
        # 1) Query Understanding
        # ---------------------------------------------------------------------
        qa = await build_analyzer().analyze(query)
        analysis = qa.result

        result.qa_api_calls = qa.metrics.api_calls
        result.qa_tokens = qa.metrics.total_tokens
        result.qa_ms = qa.metrics.latency_ms
        result.qa_repaired = qa.repaired

        # ---------------------------------------------------------------------
        # 2) Query Generation
        # ---------------------------------------------------------------------
        stage = "query_generation"
        generation = await build_generator().generate(analysis)

        result.qgen_api_calls = generation.metrics.api_calls
        result.qgen_tokens = generation.metrics.total_tokens
        result.qgen_ms = generation.metrics.latency_ms
        result.generated_queries = [
            {
                "query_id": q.query_id,
                "kind": q.kind.value,
                "priority": q.priority,
                "text": q.text,
            }
            for q in generation.result.queries
        ]

        if generation.validation.errors:
            errors = [
                f"{x.code}: {x.message}"
                for x in generation.validation.errors
            ]
            raise RuntimeError(
                "Query Generation validation error(s): " + " | ".join(errors)
            )

        # ---------------------------------------------------------------------
        # 3) Static Retrieval
        # ---------------------------------------------------------------------
        stage = "retrieval"
        sources = get_enabled_sources()
        if not sources:
            raise RuntimeError("No retrieval source enabled.")

        native_plan = RetrievalRouteBuilder(
            enabled_sources=sources
        ).build(
            analysis,
            generation,
        )
        execution_plan = _make_execution_plan(native_plan)

        adapters = _build_adapters(sources)
        orchestrator = _build_orchestrator(adapters)

        retrieval = await orchestrator.run(execution_plan)
        result.retrieval_api_calls = retrieval.metrics.api_calls
        result.retrieval_ms = retrieval.metrics.total_latency_ms
        result.raw_hits = retrieval.metrics.raw_hits

        if not retrieval.hits:
            raise RuntimeError("Retrieval returned zero hits.")

        # ---------------------------------------------------------------------
        # 4) Candidate Pool before/after hard filter
        # ---------------------------------------------------------------------
        stage = "candidate_pool"
        pool = CandidatePool.from_hits(retrieval.hits)

        before_filter = list(pool.candidates)
        before_map = candidate_id_map(before_filter)
        before_ids = set(before_map)
        result.unique_candidates = len(before_filter)
        result.gold_found_before_filter = len(before_ids & gold)

        pool.apply_hard_filters(analysis)
        result.hard_rejected = len(pool.rejected)
        result.after_hard_filter = len(pool.candidates)

        after_filter = list(pool.candidates)
        after_map = candidate_id_map(after_filter)
        after_ids = set(after_map)
        result.gold_found_after_filter = len(after_ids & gold)
        result.gold_lost_by_hard_filter = sorted(
            (before_ids & gold) - after_ids
        )

        pool.prerank(
            analysis,
            generation,
            num_enabled_sources=len(sources),
            num_planned_routes=len(execution_plan.routes),
        )

        ranked = pool.top(max(100, len(pool.candidates)))
        ranked_map = candidate_id_map(ranked)

        result.candidate_arxiv_identifiable = len(ranked_map)
        result.candidate_arxiv_identifiable_fraction = (
            len(ranked_map) / len(ranked)
            if ranked
            else 0.0
        )

        result.retrieval_recall_25 = recall_at_k(
            ranked, gold, 25
        )
        result.retrieval_recall_100 = recall_at_k(
            ranked, gold, 100
        )
        result.retrieval_recall_all = recall(
            set(ranked_map), gold
        )

        if not pool.candidates:
            result.missed_gold = sorted(gold)
            result.total_wall_ms = (
                time.perf_counter() - overall
            ) * 1000.0
            return result

        # ---------------------------------------------------------------------
        # 5) Enrichment
        # ---------------------------------------------------------------------
        stage = "enrichment"
        enrich_started = time.perf_counter()
        try:
            result.enrichment_api_calls = await pool.enrich(
                adapters,
                top_n=min(ENRICH_TOP_N, len(pool.candidates)),
                required_fields=ENRICH_REQUIRED_FIELDS,
            )
        finally:
            result.enrichment_ms = (
                time.perf_counter() - enrich_started
            ) * 1000.0

        pool.prerank(
            analysis,
            generation,
            num_enabled_sources=len(sources),
            num_planned_routes=len(execution_plan.routes),
        )

        # ---------------------------------------------------------------------
        # 6) Selector
        # ---------------------------------------------------------------------
        stage = "selector"
        selector_input = pool.top(
            min(SELECTOR_TOP_N, len(pool.candidates))
        )
        result.selector_input_count = len(selector_input)

        input_map = candidate_id_map(selector_input)
        input_ids = set(input_map)
        input_gold = input_ids & gold

        result.selector_input_gold_count = len(input_gold)
        result.selector_input_recall = recall(input_ids, gold)

        output = await _build_selector().select(
            analysis,
            selector_input,
        )

        result.selector_pass_count = len(output.selected_ids)
        result.selector_unknown_count = len(output.uncertain_ids)
        result.selector_fail_count = len(output.rejected_ids)
        result.selector_api_calls = output.metrics.llm_api_calls
        result.selector_tokens = output.metrics.llm_total_tokens
        result.selector_ms = output.metrics.total_latency_ms

        by_candidate_id = {
            c.candidate_id: c
            for c in selector_input
        }

        def papers_for(ids: list[str]) -> list[dict[str, Any]]:
            rows_: list[dict[str, Any]] = []
            for cid in ids:
                c = by_candidate_id.get(cid)
                if c is None:
                    continue
                rows_.append(
                    {
                        "candidate_id": cid,
                        "arxiv_id": candidate_arxiv_id(c),
                        "title": c.title,
                        "retrieval_score": c.retrieval_score,
                    }
                )
            return rows_

        result.pass_papers = papers_for(output.selected_ids)
        result.unknown_papers = papers_for(output.uncertain_ids)
        result.fail_papers = papers_for(output.rejected_ids)

        pass_ids = {
            x["arxiv_id"]
            for x in result.pass_papers
            if x["arxiv_id"]
        }
        unknown_ids = {
            x["arxiv_id"]
            for x in result.unknown_papers
            if x["arxiv_id"]
        }

        result.selector_pass_identifiable_count = len(pass_ids)
        result.selector_pass_recall = recall(pass_ids, gold)
        result.selector_pass_unknown_recall = recall(
            pass_ids | unknown_ids,
            gold,
        )

        if input_gold:
            result.selector_conditional_recall = (
                len(pass_ids & input_gold) / len(input_gold)
            )
        else:
            result.selector_conditional_recall = None

        p = precision(pass_ids, gold)
        result.selector_pass_precision_diagnostic = p
        result.selector_pass_f1_diagnostic = f1(
            p,
            result.selector_pass_recall,
        )

        result.matched_gold_pass = sorted(pass_ids & gold)
        result.matched_gold_unknown = sorted(unknown_ids & gold)
        result.missed_gold = sorted(gold - pass_ids)

        result.total_wall_ms = (
            time.perf_counter() - overall
        ) * 1000.0
        return result

    except Exception as exc:
        result.status = "failed"
        result.failed_stage = stage
        result.error = f"{type(exc).__name__}: {exc}"
        result.total_wall_ms = (
            time.perf_counter() - overall
        ) * 1000.0

        print(
            f"\n[ERROR] #{index} {query_id} failed at "
            f"{stage}: {result.error}"
        )
        traceback.print_exc()
        return result


# =============================================================================
# Aggregate
# =============================================================================


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(results: list[QueryResult]) -> dict[str, Any]:
    ok = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]

    summary: dict[str, Any] = {
        "requested_queries": len(results),
        "successful_queries": len(ok),
        "failed_queries": len(failed),
        "failed_query_ids": [r.query_id for r in failed],

        "official_style_recall": {
            "macro_recall_at_25": mean(
                [r.retrieval_recall_25 for r in ok]
            ),
            "macro_recall_at_100": mean(
                [r.retrieval_recall_100 for r in ok]
            ),
            "macro_recall_all": mean(
                [r.retrieval_recall_all for r in ok]
            ),
            "zero_recall_all_queries": sum(
                r.retrieval_recall_all == 0.0
                for r in ok
            ),
        },

        "selector": {
            "macro_input_recall": mean(
                [r.selector_input_recall for r in ok]
            ),
            "macro_pass_recall": mean(
                [r.selector_pass_recall for r in ok]
            ),
            "macro_pass_plus_unknown_recall": mean(
                [r.selector_pass_unknown_recall for r in ok]
            ),
            "macro_conditional_recall_on_reached_gold": mean(
                [
                    r.selector_conditional_recall
                    for r in ok
                    if r.selector_conditional_recall is not None
                ]
            ),
            "macro_pass_precision_diagnostic": mean(
                [
                    r.selector_pass_precision_diagnostic
                    for r in ok
                ]
            ),
            "macro_pass_f1_diagnostic": mean(
                [r.selector_pass_f1_diagnostic for r in ok]
            ),
        },

        "candidate_diagnostics": {
            "mean_unique_candidates": mean(
                [float(r.unique_candidates) for r in ok]
            ),
            "mean_after_hard_filter": mean(
                [float(r.after_hard_filter) for r in ok]
            ),
            "mean_arxiv_identifiable_fraction": mean(
                [
                    r.candidate_arxiv_identifiable_fraction
                    for r in ok
                ]
            ),
            "gold_lost_by_hard_filter_total": sum(
                len(r.gold_lost_by_hard_filter)
                for r in ok
            ),
        },

        "efficiency": {
            "qa_api_calls": sum(r.qa_api_calls for r in results),
            "qgen_api_calls": sum(r.qgen_api_calls for r in results),
            "retrieval_api_calls": sum(
                r.retrieval_api_calls for r in results
            ),
            "enrichment_api_calls": sum(
                r.enrichment_api_calls for r in results
            ),
            "selector_api_calls": sum(
                r.selector_api_calls for r in results
            ),
            "llm_total_tokens": sum(
                r.qa_tokens + r.qgen_tokens + r.selector_tokens
                for r in results
            ),
            "total_wall_seconds": sum(
                r.total_wall_ms for r in results
            ) / 1000.0,
            "mean_wall_seconds_per_query": mean(
                [r.total_wall_ms / 1000.0 for r in results]
            ),
        },

        "environment_note": {
            "gold_key": "answer_arxiv_ids",
            "id_matching": (
                "normalized exact arXiv ID, version suffix removed"
            ),
            "primary_metric": (
                "ScholarQuest-style Recall@25 / Recall@100 / Recall@All"
            ),
            "backend": (
                "Scholar-Agent realtime OpenAlex/S2/arXiv, "
                "NOT official fixed ScholarBase"
            ),
            "strict_comparability_to_paper": False,
        },
    }

    # Optional category breakdown from query_metadata.jsonl.
    categories = sorted(
        {r.category for r in ok if r.category}
    )
    if categories:
        summary["by_category"] = {}
        for category in categories:
            subset = [
                r for r in ok if r.category == category
            ]
            summary["by_category"][category] = {
                "n": len(subset),
                "r25": mean(
                    [r.retrieval_recall_25 for r in subset]
                ),
                "r100": mean(
                    [r.retrieval_recall_100 for r in subset]
                ),
                "rall": mean(
                    [r.retrieval_recall_all for r in subset]
                ),
                "selector_pass_recall": mean(
                    [r.selector_pass_recall for r in subset]
                ),
            }

    return summary


# =============================================================================
# Printing / checkpoint
# =============================================================================


def _fmt_optional(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.4f}"


def print_result(r: QueryResult) -> None:
    print("\n" + "=" * 96)
    print(
        f"ScholarQuest #{r.index} | {r.query_id} "
        f"| category={r.category or 'N/A'}"
    )
    print("=" * 96)
    print(f"query = {r.query}")
    print(f"gold_count = {r.gold_count}")
    print(f"status = {r.status}")

    if r.status != "success":
        print(f"failed_stage = {r.failed_stage}")
        print(f"error = {r.error}")
        print(f"wall_s = {r.total_wall_ms / 1000.0:.2f}")
        return

    print("\n---------- RETRIEVAL (ScholarQuest primary) ----------")
    print(f"raw_hits                    = {r.raw_hits}")
    print(f"unique_candidates           = {r.unique_candidates}")
    print(f"hard_rejected               = {r.hard_rejected}")
    print(f"after_hard_filter           = {r.after_hard_filter}")
    print(
        f"arxiv_identifiable          = "
        f"{r.candidate_arxiv_identifiable}/"
        f"{r.after_hard_filter} "
        f"({r.candidate_arxiv_identifiable_fraction:.4f})"
    )
    print(f"Recall@25                   = {r.retrieval_recall_25:.4f}")
    print(f"Recall@100                  = {r.retrieval_recall_100:.4f}")
    print(f"Recall@All                  = {r.retrieval_recall_all:.4f}")
    print(
        f"gold before/after hard      = "
        f"{r.gold_found_before_filter}/"
        f"{r.gold_found_after_filter}"
    )
    if r.gold_lost_by_hard_filter:
        print(
            "gold_lost_by_hard_filter = "
            + json.dumps(r.gold_lost_by_hard_filter)
        )

    print("\n---------- SELECTOR ----------")
    print(
        f"input/pass/unknown/fail      = "
        f"{r.selector_input_count}/"
        f"{r.selector_pass_count}/"
        f"{r.selector_unknown_count}/"
        f"{r.selector_fail_count}"
    )
    print(f"selector input Recall        = {r.selector_input_recall:.4f}")
    print(f"PASS Recall                  = {r.selector_pass_recall:.4f}")
    print(
        f"PASS+UNKNOWN Recall          = "
        f"{r.selector_pass_unknown_recall:.4f}"
    )
    print(
        f"conditional Recall           = "
        f"{_fmt_optional(r.selector_conditional_recall)} "
        "(among Gold that reached Selector)"
    )
    print(
        f"PASS precision (diagnostic)  = "
        f"{r.selector_pass_precision_diagnostic:.4f} "
        f"(arXiv-identifiable PASS={r.selector_pass_identifiable_count})"
    )
    print(
        f"PASS F1 (diagnostic)         = "
        f"{r.selector_pass_f1_diagnostic:.4f}"
    )

    print("\n---------- GOLD DIAGNOSTIC ----------")
    print(f"matched PASS    = {len(r.matched_gold_pass)}")
    for aid in r.matched_gold_pass[:30]:
        print(f"  [TP/PASS] arXiv:{aid}")

    print(f"matched UNKNOWN = {len(r.matched_gold_unknown)}")
    for aid in r.matched_gold_unknown[:30]:
        print(f"  [GOLD/UNKNOWN] arXiv:{aid}")

    print(f"missed Gold     = {len(r.missed_gold)}")
    for aid in r.missed_gold[:30]:
        print(f"  [MISS] arXiv:{aid}")

    print("\n---------- EFFICIENCY ----------")
    print(
        "QA/QGen/Retrieval/Enrich/Selector calls = "
        f"{r.qa_api_calls}/"
        f"{r.qgen_api_calls}/"
        f"{r.retrieval_api_calls}/"
        f"{r.enrichment_api_calls}/"
        f"{r.selector_api_calls}"
    )
    print(
        "QA/QGen/Selector tokens = "
        f"{r.qa_tokens}/{r.qgen_tokens}/{r.selector_tokens}"
    )
    print(f"wall_s = {r.total_wall_ms / 1000.0:.2f}")


def save_checkpoint(
    *,
    dataset_path: Path,
    metadata_path: Path | None,
    results: list[QueryResult],
    output_path: Path,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "metadata_path": (
            str(metadata_path) if metadata_path else None
        ),
        "start": START,
        "limit": LIMIT,
        "selector_top_n": SELECTOR_TOP_N,
        "enabled_sources": [
            x.value for x in get_enabled_sources()
        ],
        "results": [asdict(r) for r in results],
        "aggregate": aggregate(results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    metadata_path = resolve_metadata_path()

    benchmark = load_benchmark(dataset_path)
    metadata_by_id = load_metadata_by_query_id(
        metadata_path
    )

    start = max(0, START)
    end = min(
        len(benchmark),
        start + max(0, LIMIT),
    )
    rows = benchmark[start:end]

    if not rows:
        raise ValueError(
            f"No ScholarQuest rows selected: "
            f"dataset_size={len(benchmark)}, "
            f"start={start}, limit={LIMIT}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        RESULTS_DIR
        / f"scholarquest_{start}_{end - 1}_{stamp}.json"
    )

    print("=" * 96)
    print("ScholarQuest -> Scholar-Agent-V2 Benchmark")
    print("=" * 96)
    print(f"dataset        = {dataset_path}")
    print(f"metadata       = {metadata_path}")
    print(f"dataset_size   = {len(benchmark)}")
    print(f"range          = [{start}, {end})")
    print(f"query_count    = {len(rows)}")
    print(f"selector_topN  = {SELECTOR_TOP_N}")
    print(
        "sources        = "
        f"{[x.value for x in get_enabled_sources()]}"
    )
    print(
        "gold matching  = exact normalized arXiv ID "
        "(version suffix removed)"
    )
    print(
        "IMPORTANT      = realtime OpenAlex/S2/arXiv backend; "
        "not official ScholarBase"
    )
    print(f"checkpoint     = {output_path}")

    results: list[QueryResult] = []

    for local_idx, row in enumerate(rows):
        global_idx = start + local_idx
        query_id = str(row["query_id"])

        print("\n" + "#" * 96)
        print(
            f"RUNNING {query_id} "
            f"({local_idx + 1}/{len(rows)})"
        )
        print("#" * 96)

        r = await evaluate_one(
            global_idx,
            row,
            metadata_by_id.get(query_id),
        )
        results.append(r)
        print_result(r)

        # Every query -> checkpoint.
        save_checkpoint(
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            results=results,
            output_path=output_path,
        )

        if (
            SLEEP_SECONDS > 0
            and local_idx + 1 < len(rows)
        ):
            await asyncio.sleep(SLEEP_SECONDS)

    agg = aggregate(results)

    print("\n" + "=" * 96)
    print("FINAL SCHOLARQUEST SUMMARY")
    print("=" * 96)
    print(
        json.dumps(
            agg,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nSaved to: {output_path}")

    return 0 if agg["failed_queries"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
