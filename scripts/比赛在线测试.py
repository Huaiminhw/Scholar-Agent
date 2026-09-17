"""Interactive competition pipeline: live LLM -> online retrieval -> JSON papers."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env", override=False)

from scholar_agent_v2.query_generation.generator import (  # noqa: E402
    GeneratorConfig,
    QueryGenerator,
)
from scholar_agent_v2.query_understanding.analyzer import (  # noqa: E402
    AnalyzerConfig,
    QueryAnalyzer,
)
from scholar_agent_v2.retrieval.config import (  # noqa: E402
    RetrievalProfile,
    get_retrieval_profile,
)
from scholar_agent_v2.retrieval.models import RetrievalStrategy  # noqa: E402
from scholar_agent_v2.retrieval.pipeline import ScholarRetrievalPipeline  # noqa: E402
from scholar_agent_v2.selector import CandidateSelector, SelectorConfig  # noqa: E402
from scholar_agent_v2.selector.backends import (  # noqa: E402
    LLMCriteriaBackend,
    LLMCriteriaBackendConfig,
)
from tests.real_llm_client import build_client  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="输入问题，运行在线比赛 Pipeline 并输出结构化论文 JSON。"
    )
    parser.add_argument("question", nargs="*", help="不提供时进入终端交互输入。")
    parser.add_argument("--top-k", type=int, default=20, help="输出论文数量。")
    parser.add_argument(
        "--selector-top-n",
        type=int,
        default=20,
        help="交给 LLM Selector 判断的候选数。",
    )
    parser.add_argument(
        "--no-selector",
        action="store_true",
        help="跳过最终 LLM Selector，只输出 Retrieval 排序。",
    )
    parser.add_argument(
        "--force-understanding-llm",
        action="store_true",
        help="绕过查询理解 Router，强制 QueryAnalyzer 调用 LLM。",
    )
    parser.add_argument("--output", type=Path, help="可选 JSON 输出文件。")
    return parser.parse_args()


def read_question(parts: list[str]) -> str:
    question = " ".join(parts).strip()
    if not question:
        question = input("请输入比赛学术检索问题：\n> ").strip()
    if not question:
        raise ValueError("问题不能为空。")
    return question


def _document(candidate, rank: int, selection) -> dict:
    origins = candidate.origins
    return {
        "rank": rank,
        "candidate_id": candidate.candidate_id,
        "title": candidate.title,
        "abstract": candidate.abstract,
        "authors": candidate.authors,
        "year": candidate.year,
        "venue": candidate.venue,
        "identifiers": candidate.identifiers.model_dump(mode="json"),
        "urls": candidate.urls,
        "citation_count": candidate.citation_count,
        "retrieval_score": candidate.retrieval_score,
        "score_components": candidate.score_components,
        "selection": (
            selection.model_dump(mode="json")
            if selection is not None
            else None
        ),
        "provenance": {
            "sources": sorted({origin.source.value for origin in origins}),
            "route_ids": sorted({origin.route_id for origin in origins}),
            "query_ids": sorted({origin.query_id for origin in origins}),
            "citation_expanded": any(
                origin.strategy == RetrievalStrategy.CITATION
                for origin in origins
            ),
        },
    }


async def run(args: argparse.Namespace, question: str) -> dict:
    if get_retrieval_profile() != RetrievalProfile.COMPETITION_ONLINE:
        raise RuntimeError(
            "比赛在线脚本要求 RETRIEVAL_PROFILE=competition_online；"
            "请不要使用 frozen_local。"
        )
    if args.top_k <= 0 or args.selector_top_n <= 0:
        raise ValueError("--top-k 和 --selector-top-n 必须大于 0。")

    started = time.perf_counter()
    client = build_client()
    analyzer = QueryAnalyzer(
        client,
        config=AnalyzerConfig(
            temperature=0.0,
            enable_deterministic_router=not args.force_understanding_llm,
            use_structured_output=False,
            prompt_schema_mode="summary",
            include_few_shot_examples=False,
            enable_repair=True,
            repair_schema_mode="full",
            max_transport_retries=0,
            retain_raw_output=False,
        ),
    )
    generator = QueryGenerator(
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

    print("\n[1/4] Query Understanding...", flush=True)
    analysis = await analyzer.analyze(question)
    print("[2/4] Query Expansion...", flush=True)
    generation = await generator.generate(analysis.result)

    pipeline = ScholarRetrievalPipeline.from_environment(
        max_concurrency=int(os.getenv("RETRIEVAL_MAX_CONCURRENCY", "6")),
        route_timeout_seconds=float(os.getenv("RETRIEVAL_ROUTE_TIMEOUT", "30")),
        max_attempts=int(os.getenv("RETRIEVAL_MAX_ATTEMPTS", "2")),
    )
    print(
        "[3/4] Online Retrieval: "
        + ", ".join(source.value for source in pipeline.enabled_sources)
        + "...",
        flush=True,
    )
    retrieval_output = await pipeline.run(analysis.result, generation.result)
    pool = retrieval_output.candidate_pool

    selector_output = None
    if not args.no_selector:
        selector_n = min(args.selector_top_n, len(pool.candidates))
        if selector_n:
            await pool.enrich(
                pipeline.adapters,
                top_n=selector_n,
                required_fields=(
                    "abstract",
                    "venue",
                    "citation_count",
                    "publication_types",
                    "fields_of_study",
                ),
            )
            print(f"[4/4] LLM Selector: top {selector_n}...", flush=True)
            selector = CandidateSelector(
                LLMCriteriaBackend(
                    client,
                    config=LLMCriteriaBackendConfig(
                        batch_size=int(os.getenv("SELECTOR_BATCH_SIZE", "5")),
                        max_concurrency=int(
                            os.getenv("SELECTOR_MAX_CONCURRENCY", "2")
                        ),
                        temperature=0.0,
                        use_structured_output=False,
                        enable_repair=True,
                        max_transport_retries=0,
                        fail_open=True,
                    ),
                ),
                config=SelectorConfig(max_candidates=selector_n),
            )
            selector_output = await selector.select(
                analysis.result,
                pool.top(selector_n),
            )
    else:
        print("[4/4] LLM Selector skipped.", flush=True)

    selection_map = (
        {item.candidate_id: item for item in selector_output.results}
        if selector_output is not None
        else {}
    )
    ranked = pool.top(args.top_k)
    return {
        "schema_version": "1.0",
        "status": "success",
        "profile": "competition_online",
        "original_query": question,
        "query_understanding": {
            "route": analysis.understanding_route,
            "route_reason": analysis.route_reason,
            "repaired": analysis.repaired,
            "repair_trigger": analysis.repair_trigger,
            "result": analysis.result.model_dump(mode="json"),
            "metrics": asdict(analysis.metrics),
        },
        "query_generation": {
            "result": generation.result.model_dump(mode="json"),
            "metrics": generation.metrics.model_dump(mode="json"),
        },
        "retrieval": {
            "sources": [source.value for source in pipeline.enabled_sources],
            "metrics": retrieval_output.retrieval.metrics.model_dump(mode="json"),
            "citation": retrieval_output.citation_expansion.diagnostics.model_dump(
                mode="json"
            ),
            "pool_stats": pool.stats.__dict__,
        },
        "selector": (
            {
                "enabled": True,
                "metrics": selector_output.metrics.model_dump(mode="json"),
                "warnings": selector_output.warnings,
            }
            if selector_output is not None
            else {"enabled": False}
        ),
        "documents": [
            _document(candidate, rank, selection_map.get(candidate.candidate_id))
            for rank, candidate in enumerate(ranked, start=1)
        ],
        "total_latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def main() -> int:
    args = parse_args()
    try:
        question = read_question(args.question)
        result = asyncio.run(run(args, question))
        rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        print("\n========== COMPETITION RESULT ==========\n")
        print(rendered)
        if args.output:
            output_path = args.output.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
            print(f"\n结果已保存：{output_path}")
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n比赛在线测试失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
