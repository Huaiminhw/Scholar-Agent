# tests/real_llm_client.py
"""
真实 LLM 客户端适配器。

适配当前 .env：
    LLM_API_KEY=...
    LLM_BASE_URL=https://.../compatible-mode/v1
    LLM_MODEL=qwen3.8-max

特点：
- 使用 OpenAI-compatible（OpenAI 兼容接口）
- 自动从项目根目录寻找 .env
- 不把 API Key 写死在代码里
- 实现 scholar_agent_v2.query_understanding.analyzer.LLMClient Protocol
- 自动统计 prompt/completion/total tokens
- 默认不强制 Structured Output，兼容性更稳
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
from openai import AsyncOpenAI

from scholar_agent_v2.query_understanding.analyzer import (
    LLMResponse,
    LLMUsage,
)


# ============================================================
# 1. 查找并加载项目根目录 .env
# ============================================================

def _find_project_env() -> Path:
    """
    从当前文件位置向上寻找 .env。

    这样无论你从：
        项目根目录
        src/scholar_agent_v2
        tests
    哪个目录启动 pytest，都能正确找到项目根目录的 .env。
    """
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "没有找到 .env 文件。请确认项目根目录下存在 .env。"
    )


_ENV_PATH = _find_project_env()
load_dotenv(_ENV_PATH, override=False)


# ============================================================
# 2. 环境变量读取
# ============================================================

def _required_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"缺少环境变量 {name}。"
            f"请检查 {_ENV_PATH}。"
        )

    return value.strip()


def _normalize_base_url(value: str) -> str:
    """
    正常 .env 应直接写 URL。
    这里顺手容忍复制时产生的 <...> 或末尾 /。
    """
    value = value.strip()

    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()

    return value.rstrip("/")


# ============================================================
# 3. OpenAI-compatible LLM Client
# ============================================================

class RealLLMClient:
    """
    QueryAnalyzer 使用的真实模型客户端。

    当前模型由 .env 中 LLM_MODEL 决定，
    因此后续切换模型一般不用改代码。
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = _required_env("LLM_API_KEY")
        self.base_url = _normalize_base_url(
            _required_env("LLM_BASE_URL")
        )
        self.model = _required_env("LLM_MODEL")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """
        调用真实模型。

        注意：
        当前先采用普通 JSON Prompt 模式，
        不强制使用 response_schema。

        原因：
        不同 OpenAI-compatible 平台对
        json_schema / structured output 的实现存在差异。
        我们先验证查询理解准确率和稳定性。

        如果后续确认当前阿里云模型支持 JSON Schema，
        再开启 analyzer.py 中 use_structured_output=True。
        """

        # OpenAI SDK 接受 list[dict]。
        request_messages = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
        ]

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=request_messages,
            temperature=temperature,
        )

        if not completion.choices:
            raise RuntimeError(
                "LLM 返回结果中没有 choices。"
            )

        content = completion.choices[0].message.content

        if content is None or not content.strip():
            raise RuntimeError(
                "LLM 返回了空 content。"
            )

        usage = completion.usage

        if usage is None:
            llm_usage = LLMUsage()
        else:
            llm_usage = LLMUsage(
                prompt_tokens=getattr(
                    usage,
                    "prompt_tokens",
                    None,
                ),
                completion_tokens=getattr(
                    usage,
                    "completion_tokens",
                    None,
                ),
                total_tokens=getattr(
                    usage,
                    "total_tokens",
                    None,
                ),
            )

        return LLMResponse(
            content=content,
            usage=llm_usage,
            model=getattr(
                completion,
                "model",
                self.model,
            ),
            request_id=getattr(
                completion,
                "id",
                None,
            ),
        )


def build_client() -> RealLLMClient:
    """
    test_analyzer.py 中统一通过这个函数创建真实客户端。
    """
    return RealLLMClient()


__all__ = [
    "RealLLMClient",
    "build_client",
]
