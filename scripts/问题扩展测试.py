"""在终端输入问题，查看 Query Understanding 与 Query Generation 扩展结果。

示例：
    python scripts/问题扩展测试.py
    python scripts/问题扩展测试.py "查找多模态大模型的安全对齐论文"
    python scripts/问题扩展测试.py --local-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent_v2.query_generation.generator import (  # noqa: E402
    GeneratorConfig,
    QueryGenerator,
)
from scholar_agent_v2.query_understanding.analyzer import (  # noqa: E402
    AnalyzerConfig,
    QueryAnalyzer,
)
from tests.real_llm_client import build_client  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="输入学术问题并打印查询理解、规划和最终扩展查询。"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="待扩展的问题；不提供时进入交互输入。",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="关闭 Query Generation 的 LLM 语义扩展，只看本地规则结果。",
    )
    parser.add_argument(
        "--force-understanding-llm",
        action="store_true",
        help="关闭查询理解快速路由，强制 QueryAnalyzer 调用 LLM。",
    )
    parser.add_argument(
        "--show-analysis",
        action="store_true",
        help="打印完整 QueryAnalysisResult，而不只显示摘要。",
    )
    return parser.parse_args()


def read_question(parts: list[str]) -> str:
    question = " ".join(parts).strip()
    if not question:
        question = input("请输入需要扩展的学术检索问题：\n> ").strip()
    if not question:
        raise ValueError("问题不能为空。")
    return question


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


async def run(
    question: str,
    *,
    local_only: bool,
    force_understanding_llm: bool,
    show_analysis: bool,
) -> None:
    client = build_client()
    analyzer = QueryAnalyzer(
        client,
        config=AnalyzerConfig(
            temperature=0.0,
            enable_deterministic_router=not force_understanding_llm,
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
        client=None if local_only else client,
        config=GeneratorConfig(
            temperature=0.0,
            use_structured_output=False,
            enable_llm_expansion=not local_only,
            max_transport_retries=0,
            max_total_queries=5,
            near_duplicate_threshold=0.88,
        ),
    )

    analysis = await analyzer.analyze(question)
    generation = await generator.generate(analysis.result)

    print("\n========== QUERY UNDERSTANDING ==========")
    print_json(
        {
            "route": analysis.understanding_route,
            "route_reason": analysis.route_reason,
            "repaired": analysis.repaired,
            "repair_trigger": analysis.repair_trigger,
            "query_type": analysis.result.query_type.value,
            "task": analysis.result.intent.task.value,
            "mode": analysis.result.intent.mode.value,
            "content": analysis.result.content,
            "constraints": analysis.result.constraints.model_dump(mode="json"),
            "relations": [
                relation.model_dump(mode="json")
                for relation in analysis.result.relations
            ],
        }
    )

    if show_analysis:
        print("\n========== FULL QUERY ANALYSIS RESULT ==========")
        print_json(analysis.result.model_dump(mode="json"))

    print("\n========== EXPANSION PLAN ==========")
    print_json(generation.result.plan.model_dump(mode="json"))

    print("\n========== EXPANDED QUERIES ==========")
    if not generation.result.queries:
        print("没有生成查询。")
    for query in generation.result.queries:
        print(
            f"[{query.query_id}] kind={query.kind.value:<8} "
            f"priority={query.priority:.2f}\n"
            f"  query : {query.text}\n"
            f"  reason: {query.reason or '-'}"
        )
        if query.warnings:
            print(f"  warnings: {query.warnings}")

    print("\n========== VALIDATION ==========")
    if not generation.validation.issues:
        print("OK")
    else:
        for issue in generation.validation.issues:
            print(
                f"{issue.severity.upper()} {issue.code} "
                f"query_id={issue.query_id}: {issue.message}"
            )

    print("\n========== METRICS ==========")
    print_json(
        {
            "query_understanding": asdict(analysis.metrics),
            "query_generation": generation.metrics.model_dump(mode="json"),
            "generation_llm_used": generation.result.llm_used,
            "generation_degraded": generation.result.degraded,
            "warnings": generation.result.warnings,
        }
    )


def main() -> int:
    args = parse_args()
    try:
        question = read_question(args.question)
        asyncio.run(
            run(
                question,
                local_only=args.local_only,
                force_understanding_llm=args.force_understanding_llm,
                show_analysis=args.show_analysis,
            )
        )
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n查询扩展失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
