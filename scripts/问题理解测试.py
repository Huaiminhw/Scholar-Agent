"""Interactively inspect Scholar Agent V2 query-understanding output.

Examples::

    python scripts/问题理解测试.py
    python scripts/问题理解测试.py "Find ACL papers after 2023, excluding surveys"

The script loads the OpenAI-compatible LLM configuration from the project
root ``.env`` through the existing real LLM client.
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

from scholar_agent_v2.query_understanding.analyzer import (  # noqa: E402
    AnalyzerConfig,
    QueryAnalyzer,
)
from tests.real_llm_client import build_client  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="输入自然语言学术查询并打印 QueryAnalysisResult。"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="待分析的问题；不提供时进入交互输入。",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="额外打印 LLM 原始输出，便于排查 JSON 或 Prompt 问题。",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="关闭查询理解 Repair LLM。",
    )
    return parser.parse_args()


def read_question(parts: list[str]) -> str:
    question = " ".join(parts).strip()
    if not question:
        question = input("请输入学术检索问题：\n> ").strip()
    if not question:
        raise ValueError("问题不能为空。")
    return question


async def run(question: str, *, show_raw: bool, enable_repair: bool) -> None:
    analyzer = QueryAnalyzer(
        build_client(),
        config=AnalyzerConfig(
            temperature=0.0,
            use_structured_output=False,
            prompt_schema_mode="summary",
            include_few_shot_examples=False,
            enable_repair=enable_repair,
            repair_schema_mode="full",
            max_transport_retries=0,
            retain_raw_output=show_raw,
        ),
    )
    output = await analyzer.analyze(question)

    print("\n========== QUERY ANALYSIS RESULT ==========")
    print(json.dumps(output.result.model_dump(mode="json"), ensure_ascii=False, indent=2))

    print("\n========== VALIDATION ==========")
    print(
        json.dumps(
            {
                "is_valid": output.validation.is_valid,
                "summary": output.validation.summary,
                "issues": [asdict(issue) for issue in output.validation.issues],
                "repaired": output.repaired,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print("\n========== METRICS ==========")
    print(json.dumps(asdict(output.metrics), ensure_ascii=False, indent=2))

    if show_raw:
        print("\n========== RAW PRIMARY OUTPUT ==========")
        print(output.raw_primary_output)
        if output.raw_repair_output is not None:
            print("\n========== RAW REPAIR OUTPUT ==========")
            print(output.raw_repair_output)


def main() -> int:
    args = parse_args()
    try:
        question = read_question(args.question)
        asyncio.run(
            run(
                question,
                show_raw=args.show_raw,
                enable_repair=not args.no_repair,
            )
        )
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n查询理解失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
