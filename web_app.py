"""Scholar Agent V2 的轻量 Web 界面（仅使用 Python 标准库）。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

from scholar_agent_v2.query_understanding.analyzer import (  # noqa: E402
    AnalyzerConfig,
    QueryAnalyzer,
)
from tests.real_llm_client import build_client  # noqa: E402


async def analyze_question(question: str) -> dict[str, Any]:
    analyzer = QueryAnalyzer(
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
    output = await analyzer.analyze(question)
    return {
        "status": "success",
        "mode": "understanding",
        "original_query": question,
        "query_understanding": {
            "route": output.understanding_route,
            "route_reason": output.route_reason,
            "repaired": output.repaired,
            "result": output.result.model_dump(mode="json"),
            "validation": {
                "is_valid": output.validation.is_valid,
                "summary": output.validation.summary,
                "issues": [asdict(item) for item in output.validation.issues],
            },
            "metrics": asdict(output.metrics),
        },
        "documents": [],
    }


async def search_papers(question: str, top_k: int, use_selector: bool) -> dict[str, Any]:
    from scripts.比赛在线测试 import run

    args = argparse.Namespace(
        top_k=top_k,
        selector_top_n=max(top_k, 20),
        no_selector=not use_selector,
        force_understanding_llm=False,
        output=None,
    )
    return await run(args, question)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self._json(404, {"status": "error", "message": "接口不存在"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            question = str(data.get("question", "")).strip()
            mode = str(data.get("mode", "search"))
            top_k = min(max(int(data.get("top_k", 10)), 1), 50)
            if not question:
                raise ValueError("请输入要检索的学术问题。")
            if mode == "understanding":
                result = asyncio.run(analyze_question(question))
            elif mode == "search":
                result = asyncio.run(
                    search_papers(question, top_k, bool(data.get("use_selector", True)))
                )
            else:
                raise ValueError("不支持的运行模式。")
            self._json(200, result)
        except Exception as exc:
            self._json(
                500,
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 Scholar Agent Web 界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Scholar Agent 已启动：http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
