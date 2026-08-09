# query_understanding/analyzer.py
"""
Query Understanding（查询理解）执行器。

主流程：
User Query
    ↓
Prompt 构造
    ↓
LLM 主分析（通常 1 次）
    ↓
JSON 解析
    ↓
Pydantic Schema 校验
    ↓
safe_normalize（安全归一化）
    ↓
validator.py 业务逻辑校验
    ↓
若失败：可选 Repair Prompt（最多额外 1 次 LLM）
    ↓
QueryAnalysisResult + Metrics

特点：
- 不绑定具体模型厂商；
- 支持 Structured Output（结构化输出）；
- Repair 只在失败时触发；
- 统一统计 API 次数、Token、延迟；
- 便于后续接 OpenAI、华为云、DeepSeek、Qwen 等客户端。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import ValidationError

from .prompt import (
    PromptConfig,
    build_query_analysis_messages,
    build_query_repair_messages,
    get_query_analysis_json_schema,
)
from .schema import QueryAnalysisResult
from .validator import (
    ValidationReport,
    safe_normalize,
    validate_query_analysis,
)

logger = logging.getLogger(__name__)

SchemaMode = Literal["none", "summary", "full"]
ChatMessage = Mapping[str, str]


# ============================================================
# 1. LLM Provider 抽象
# ============================================================

@dataclass
class LLMUsage:
    """
    不同厂商 usage 字段不同，由适配器统一映射到这里。
    拿不到的字段保持 None。
    """
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMResponse:
    """
    LLMClient.complete() 的统一返回。

    content:
    - 普通模式：JSON 字符串
    - Structured Output：也可以直接返回 dict
    """
    content: str | Mapping[str, Any]
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str | None = None
    request_id: str | None = None


class LLMClient(Protocol):
    """
    任意 LLM 客户端只要实现本接口即可接入 QueryAnalyzer。

    response_schema:
    - Provider 支持 JSON Schema / Structured Output 时使用；
    - 不支持时，适配器可忽略该参数。
    """

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        ...


# ============================================================
# 2. 配置与统计
# ============================================================

@dataclass(frozen=True)
class AnalyzerConfig:
    # 查询理解尽量确定性
    temperature: float = 0.0

    # 默认关闭，保证兼容普通 API。
    # 如果实际模型支持原生 Structured Output，建议打开。
    use_structured_output: bool = False

    # 普通 JSON 模式默认 summary；
    # 如果 use_structured_output=True，可改成 none 节省 Token。
    prompt_schema_mode: SchemaMode = "summary"

    # 默认不放 Few-shot，减少 Token。
    # 只有离线评测发现特定错误模式时再开启。
    include_few_shot_examples: bool = False

    # 主结果失败时允许一次结构/逻辑修复
    enable_repair: bool = True

    # Repair 更强调稳定性，因此默认 full
    repair_schema_mode: SchemaMode = "full"

    # 比赛有 API 次数评分，默认不做传输层自动重试。
    # 若线上网络不稳定，可改为 1。
    max_transport_retries: int = 0

    retry_backoff_seconds: float = 0.5

    # 开发阶段建议 True，方便错误分析；
    # 正式部署可关闭。
    retain_raw_output: bool = True


@dataclass
class AnalyzerMetrics:
    """
    与比赛运行效率指标直接对应。
    """
    api_calls: int = 0
    repair_calls: int = 0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    latency_ms: float = 0.0
    primary_latency_ms: float = 0.0
    repair_latency_ms: float = 0.0

    model_names: list[str] = field(default_factory=list)


@dataclass
class QueryAnalyzerOutput:
    """
    查询理解模块的完整工程输出。

    result:
        给下游 Retrieval Planner / Query Generator / Selector 使用。

    validation:
        本地业务规则检查报告。

    metrics:
        API 次数、Token、耗时。

    repaired:
        是否触发过第二次 Repair。
    """
    result: QueryAnalysisResult
    validation: ValidationReport
    metrics: AnalyzerMetrics

    repaired: bool = False

    raw_primary_output: str | Mapping[str, Any] | None = None
    raw_repair_output: str | Mapping[str, Any] | None = None


# ============================================================
# 3. 错误类型
# ============================================================

class QueryAnalyzerException(RuntimeError):
    pass


class LLMCallError(QueryAnalyzerException):
    def __init__(
        self,
        message: str,
        *,
        metrics: AnalyzerMetrics | None = None,
    ) -> None:
        super().__init__(message)
        self.metrics = metrics


class JSONExtractionError(QueryAnalyzerException):
    pass


class QueryAnalysisValidationError(QueryAnalyzerException):
    """
    主分析和（若开启）Repair 都没有得到可用结果。
    """

    def __init__(
        self,
        message: str,
        *,
        primary_error: str | None = None,
        repair_error: str | None = None,
        metrics: AnalyzerMetrics | None = None,
    ) -> None:
        super().__init__(message)
        self.primary_error = primary_error
        self.repair_error = repair_error
        self.metrics = metrics


# ============================================================
# 4. JSON 容错解析
# ============================================================

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    """
    模型偶尔会在 JSON 前后加一句解释。
    这里使用确定性状态机提取第一个完整 JSON Object，
    避免为了格式小错马上增加一次 LLM 调用。
    """
    start = text.find("{")

    if start < 0:
        raise JSONExtractionError(
            "模型输出中没有找到 JSON 对象起始符 '{'。"
        )

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[start : index + 1]

    raise JSONExtractionError(
        "模型输出中的 JSON 对象没有正确闭合。"
    )


def parse_json_payload(
    content: str | Mapping[str, Any],
) -> dict[str, Any]:
    """
    优先级：
    1. Provider 已经返回 dict
    2. 直接 json.loads
    3. 去除 ```json code fence 后解析
    4. 从附加文本中提取第一个完整 JSON 对象
    """
    if isinstance(content, Mapping):
        return dict(content)

    if not isinstance(content, str):
        raise JSONExtractionError(
            f"不支持的 LLM content 类型：{type(content).__name__}"
        )

    text = content.strip()

    if not text:
        raise JSONExtractionError("模型返回了空内容。")

    candidates: list[str] = [text]

    stripped = _strip_code_fence(text)
    if stripped != text:
        candidates.append(stripped)

    last_error: Exception | None = None

    for candidate in candidates:
        try:
            payload = json.loads(candidate)

            if not isinstance(payload, dict):
                raise JSONExtractionError(
                    "顶层 JSON 必须是 object/dict。"
                )

            return payload

        except (json.JSONDecodeError, JSONExtractionError) as exc:
            last_error = exc

    try:
        extracted = _extract_first_json_object(stripped)
        payload = json.loads(extracted)

        if not isinstance(payload, dict):
            raise JSONExtractionError(
                "顶层 JSON 必须是 object/dict。"
            )

        return payload

    except (json.JSONDecodeError, JSONExtractionError) as exc:
        last_error = exc

    raise JSONExtractionError(
        f"无法从模型输出解析 JSON：{last_error}"
    )


# ============================================================
# 5. Schema + Business Validation
# ============================================================

@dataclass
class _CandidateValidation:
    result: QueryAnalysisResult | None
    report: ValidationReport | None
    error_text: str | None


def _validate_candidate(
    payload: dict[str, Any],
    *,
    original_query: str,
) -> _CandidateValidation:
    """
    两层校验：
    1. Pydantic Schema
    2. validator.py 业务逻辑
    """
    try:
        result = QueryAnalysisResult.model_validate(payload)

    except ValidationError as exc:
        return _CandidateValidation(
            result=None,
            report=None,
            error_text=(
                "Pydantic schema validation failed:\n"
                + exc.json(indent=2)
            ),
        )

    # 先执行安全、本地、无语义变化的修正
    result = safe_normalize(
        result,
        original_query=original_query,
    )

    report = validate_query_analysis(
        result,
        original_query=original_query,
    )

    if not report.is_valid:
        return _CandidateValidation(
            result=result,
            report=report,
            error_text=(
                "Business validation failed:\n"
                + report.summary()
            ),
        )

    return _CandidateValidation(
        result=result,
        report=report,
        error_text=None,
    )


# ============================================================
# 6. QueryAnalyzer
# ============================================================

class QueryAnalyzer:
    def __init__(
        self,
        client: LLMClient,
        *,
        config: AnalyzerConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or AnalyzerConfig()

    def _prompt_config(self) -> PromptConfig:
        return PromptConfig(
            schema_mode=self.config.prompt_schema_mode,
            include_examples=self.config.include_few_shot_examples,
        )

    @staticmethod
    def _accumulate_usage(
        metrics: AnalyzerMetrics,
        response: LLMResponse,
    ) -> None:
        usage = response.usage

        metrics.prompt_tokens += usage.prompt_tokens or 0
        metrics.completion_tokens += usage.completion_tokens or 0
        metrics.total_tokens += usage.total_tokens or 0

        if (
            response.model
            and response.model not in metrics.model_names
        ):
            metrics.model_names.append(response.model)

    async def _call_llm(
        self,
        *,
        messages: Sequence[ChatMessage],
        metrics: AnalyzerMetrics,
        is_repair: bool,
        response_schema: Mapping[str, Any] | None,
    ) -> LLMResponse:
        """
        所有模型请求统一经过这里，
        确保 API 次数 / Token / 延迟统计不漏。
        """
        attempts = 1 + max(
            0,
            self.config.max_transport_retries,
        )

        last_error: Exception | None = None

        for attempt in range(attempts):
            started = time.perf_counter()

            # 真实请求尝试就计入 API 次数，
            # 即便后面网络报错，也不能“假装没有调用”。
            metrics.api_calls += 1

            if is_repair:
                metrics.repair_calls += 1

            try:
                response = await self.client.complete(
                    messages=messages,
                    temperature=self.config.temperature,
                    response_schema=response_schema,
                )

            except Exception as exc:
                elapsed = (
                    time.perf_counter() - started
                ) * 1000.0

                if is_repair:
                    metrics.repair_latency_ms += elapsed
                else:
                    metrics.primary_latency_ms += elapsed

                last_error = exc

                logger.warning(
                    "LLM request failed: attempt=%s/%s error=%s",
                    attempt + 1,
                    attempts,
                    exc,
                )

                if attempt + 1 < attempts:
                    await asyncio.sleep(
                        self.config.retry_backoff_seconds
                        * (2**attempt)
                    )
                    continue

                raise LLMCallError(
                    f"LLM 调用失败，已尝试 {attempts} 次：{exc}",
                    metrics=metrics,
                ) from exc

            elapsed = (
                time.perf_counter() - started
            ) * 1000.0

            if is_repair:
                metrics.repair_latency_ms += elapsed
            else:
                metrics.primary_latency_ms += elapsed

            self._accumulate_usage(
                metrics,
                response,
            )

            return response

        raise LLMCallError(
            f"LLM 调用失败：{last_error}",
            metrics=metrics,
        )

    async def analyze(
        self,
        user_query: str,
    ) -> QueryAnalyzerOutput:
        """
        正常情况：1 次 LLM
        校验失败：最多增加 1 次 Repair LLM
        """
        if not isinstance(user_query, str):
            raise TypeError("user_query 必须是字符串。")

        user_query = user_query.strip()

        if not user_query:
            raise ValueError(
                "user_query 不能为空。"
            )

        metrics = AnalyzerMetrics()
        overall_started = time.perf_counter()

        # ----------------------------------------------------
        # A. Primary Analysis
        # ----------------------------------------------------
        primary_messages = build_query_analysis_messages(
            user_query,
            self._prompt_config(),
        )

        response_schema = (
            get_query_analysis_json_schema()
            if self.config.use_structured_output
            else None
        )

        primary_response = await self._call_llm(
            messages=primary_messages,
            metrics=metrics,
            is_repair=False,
            response_schema=response_schema,
        )

        raw_primary = (
            primary_response.content
            if self.config.retain_raw_output
            else None
        )

        primary_error: str | None = None
        primary_validation: _CandidateValidation | None = None

        try:
            primary_payload = parse_json_payload(
                primary_response.content
            )

            primary_validation = _validate_candidate(
                primary_payload,
                original_query=user_query,
            )

            primary_error = (
                primary_validation.error_text
            )

        except QueryAnalyzerException as exc:
            primary_error = (
                f"{type(exc).__name__}: {exc}"
            )

        if (
            primary_validation is not None
            and primary_validation.result is not None
            and primary_validation.report is not None
            and primary_validation.report.is_valid
        ):
            metrics.latency_ms = (
                time.perf_counter() - overall_started
            ) * 1000.0

            return QueryAnalyzerOutput(
                result=primary_validation.result,
                validation=primary_validation.report,
                metrics=metrics,
                repaired=False,
                raw_primary_output=raw_primary,
            )

        # ----------------------------------------------------
        # B. No Repair
        # ----------------------------------------------------
        if not self.config.enable_repair:
            metrics.latency_ms = (
                time.perf_counter() - overall_started
            ) * 1000.0

            raise QueryAnalysisValidationError(
                "主查询理解结果未通过校验，且 Repair 已关闭。",
                primary_error=primary_error,
                metrics=metrics,
            )

        # ----------------------------------------------------
        # C. Repair
        # ----------------------------------------------------
        if isinstance(
            primary_response.content,
            str,
        ):
            raw_primary_for_repair = (
                primary_response.content
            )
        else:
            raw_primary_for_repair = json.dumps(
                primary_response.content,
                ensure_ascii=False,
                indent=2,
            )

        repair_messages = build_query_repair_messages(
            user_query=user_query,
            raw_output=raw_primary_for_repair,
            validation_error=(
                primary_error
                or "Unknown validation error"
            ),
            schema_mode=self.config.repair_schema_mode,
        )

        try:
            repair_response = await self._call_llm(
                messages=repair_messages,
                metrics=metrics,
                is_repair=True,
                response_schema=response_schema,
            )

        except LLMCallError as exc:
            metrics.latency_ms = (
                time.perf_counter() - overall_started
            ) * 1000.0

            raise QueryAnalysisValidationError(
                "主分析失败，且 Repair LLM 调用也失败。",
                primary_error=primary_error,
                repair_error=str(exc),
                metrics=metrics,
            ) from exc

        raw_repair = (
            repair_response.content
            if self.config.retain_raw_output
            else None
        )

        repair_error: str | None = None
        repair_validation: _CandidateValidation | None = None

        try:
            repair_payload = parse_json_payload(
                repair_response.content
            )

            repair_validation = _validate_candidate(
                repair_payload,
                original_query=user_query,
            )

            repair_error = (
                repair_validation.error_text
            )

        except QueryAnalyzerException as exc:
            repair_error = (
                f"{type(exc).__name__}: {exc}"
            )

        metrics.latency_ms = (
            time.perf_counter() - overall_started
        ) * 1000.0

        if (
            repair_validation is not None
            and repair_validation.result is not None
            and repair_validation.report is not None
            and repair_validation.report.is_valid
        ):
            return QueryAnalyzerOutput(
                result=repair_validation.result,
                validation=repair_validation.report,
                metrics=metrics,
                repaired=True,
                raw_primary_output=raw_primary,
                raw_repair_output=raw_repair,
            )

        raise QueryAnalysisValidationError(
            "主分析与 Repair 均未得到可用的 QueryAnalysisResult。",
            primary_error=primary_error,
            repair_error=repair_error,
            metrics=metrics,
        )


__all__ = [
    "LLMUsage",
    "LLMResponse",
    "LLMClient",
    "AnalyzerConfig",
    "AnalyzerMetrics",
    "QueryAnalyzerOutput",
    "QueryAnalyzerException",
    "LLMCallError",
    "JSONExtractionError",
    "QueryAnalysisValidationError",
    "parse_json_payload",
    "QueryAnalyzer",
]
