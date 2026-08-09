from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Mapping, Any, Protocol, Sequence

from pydantic import ValidationError

from ..query_understanding.analyzer import LLMResponse, parse_json_payload
from ..query_understanding.schema import (
    ConstraintType,
    QueryAnalysisResult,
    QueryType,
)

from .dedup import deduplicate_queries, normalize_query_text
from .models import (
    GeneratedQuery,
    GenerationMetrics,
    LLMExpansionPayload,
    QueryGenerationOutput,
    QueryGenerationResult,
    QueryKind,
)
from .planner import PlannerConfig, QueryGenerationPlanner
from .prompt import build_expansion_messages, get_llm_expansion_json_schema
from .validator import filter_invalid_queries, validate_queries

logger = logging.getLogger(__name__)

ChatMessage = Mapping[str, str]


class QueryGenerationLLMClient(Protocol):
    """与 Query Understanding 的 LLMClient 兼容。"""

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        ...


@dataclass(frozen=True)
class GeneratorConfig:
    temperature: float = 0.0
    use_structured_output: bool = False

    # V1 只允许一次语义扩展调用；失败就降级为本地查询，不做 Repair LLM。
    enable_llm_expansion: bool = True

    # 传输层重试默认 0；如线上网络不稳可设 1。
    max_transport_retries: int = 0
    retry_backoff_seconds: float = 0.5

    max_total_queries: int = 5
    near_duplicate_threshold: float = 0.88


class QueryGenerator:
    """
    QueryAnalysisResult -> QueryGenerationResult

    设计原则：
    1. 规则优先；CORE / RELATION / ALIAS 零 LLM；
    2. LLM 最多一次，且只补 1~2 条高置信度等价表达；
    3. LLM 失败不阻断检索，自动降级；
    4. Validator + Dedup 在进入 Retrieval 前完成。
    """

    def __init__(
        self,
        client: QueryGenerationLLMClient | None = None,
        *,
        config: GeneratorConfig | None = None,
        planner: QueryGenerationPlanner | None = None,
    ) -> None:
        self.client = client
        self.config = config or GeneratorConfig()
        self.planner = planner or QueryGenerationPlanner(
            PlannerConfig(
                max_total_queries=self.config.max_total_queries,
                enable_llm_expansion=self.config.enable_llm_expansion and client is not None,
            )
        )

    async def generate(self, analysis: QueryAnalysisResult) -> QueryGenerationOutput:
        started = time.perf_counter()
        metrics = GenerationMetrics()
        warnings: list[str] = []

        plan = self.planner.plan(analysis)
        queries: list[GeneratedQuery] = []

        # A. EXACT
        queries.extend(self._generate_exact(analysis, plan.budget.exact))

        # B. CORE
        queries.extend(self._generate_core(analysis, plan.budget.core))

        # C. RELATION
        queries.extend(self._generate_relation_queries(analysis, plan.budget.relation))

        # D. ALIAS
        queries.extend(self._generate_alias_queries(analysis, plan.budget.alias, queries))

        # 本地先去重，避免把重复 query 发给 LLM。
        queries = deduplicate_queries(
            queries,
            threshold=self.config.near_duplicate_threshold,
            max_queries=plan.max_total_queries,
        )

        # E. LLM EXPANDED：最多一次语义调用。
        degraded = False
        if plan.use_llm and plan.budget.expanded > 0:
            if self.client is None:
                degraded = True
                warnings.append("Planner 允许 LLM 扩展，但未提供 LLM client；已退化为本地查询。")
            else:
                try:
                    llm_queries = await self._generate_llm_expanded(
                        analysis,
                        existing=queries,
                        max_new=plan.budget.expanded,
                        metrics=metrics,
                    )
                    queries.extend(llm_queries)
                except Exception as exc:
                    degraded = True
                    warnings.append(f"LLM Query Expansion 失败，已使用本地查询继续检索：{exc}")
                    logger.warning("LLM query expansion degraded: %s", exc)

        # F. Query Drift 校验：ERROR query 删除，WARNING 保留。
        queries, validation = filter_invalid_queries(queries, analysis)

        # G. 最终近重复去重 + 总预算截断。
        queries = deduplicate_queries(
            queries,
            threshold=self.config.near_duplicate_threshold,
            max_queries=plan.max_total_queries,
        )

        # 去重后重新跑一次报告，确保 final report 对应最终 queries。
        final_validation = validate_queries(queries, analysis)
        # 保留之前被删除的错误信息，便于调试。
        final_validation.issues = [*validation.issues, *final_validation.issues]

        metrics.latency_ms = (time.perf_counter() - started) * 1000.0

        result = QueryGenerationResult(
            original_query=analysis.original_query,
            core_content=analysis.content,
            plan=plan,
            queries=queries,
            llm_used=metrics.api_calls > 0,
            degraded=degraded,
            warnings=warnings,
        )

        return QueryGenerationOutput(
            result=result,
            validation=final_validation,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Local deterministic generation
    # ------------------------------------------------------------------

    def _generate_exact(self, analysis: QueryAnalysisResult, budget: int) -> list[GeneratedQuery]:
        if budget <= 0:
            return []

        text = ""
        constraint_refs: list[str] = []

        if analysis.query_type == QueryType.SPECIFIC_BY_IDENTIFIER:
            for item in analysis.constraints.hard:
                if item.type == ConstraintType.IDENTIFIER:
                    if item.value is not None:
                        text = str(item.value)
                        constraint_refs.append(item.id)
                        break
                    if item.values:
                        text = str(item.values[0])
                        constraint_refs.append(item.id)
                        break

        elif analysis.query_type == QueryType.SPECIFIC_BY_TITLE:
            text = (
                analysis.properties.specific_paper_name
                or analysis.content
                or analysis.original_query
            )

        if not text.strip():
            return []

        return [
            GeneratedQuery(
                query_id="QX1",
                text=text.strip(),
                kind=QueryKind.EXACT,
                priority=1.0,
                based_on_constraint_ids=constraint_refs,
                reason="精确定位查询。",
            )
        ]

    def _generate_core(self, analysis: QueryAnalysisResult, budget: int) -> list[GeneratedQuery]:
        if budget <= 0 or not analysis.content.strip():
            return []

        return [
            GeneratedQuery(
                query_id="QC1",
                text=_clean_content_query(analysis.content),
                kind=QueryKind.CORE,
                priority=0.95,
                based_on_entity_names=[e.canonical_name for e in analysis.entities],
                reason="直接来自 QueryAnalysisResult.content 的核心主题查询。",
            )
        ]

    def _generate_relation_queries(
        self,
        analysis: QueryAnalysisResult,
        budget: int,
    ) -> list[GeneratedQuery]:
        if budget <= 0 or not analysis.relations:
            return []

        queries: list[GeneratedQuery] = []

        # hard relation 优先，其次 soft；exclude relation 不用于正向召回。
        relations = sorted(
            [r for r in analysis.relations if r.level.value != "exclude"],
            key=lambda r: 0 if r.level.value == "hard" else 1,
        )

        for relation in relations[:budget]:
            text = _relation_to_search_text(
                relation.subject,
                relation.predicate,
                relation.object,
            )
            if not text:
                continue

            queries.append(
                GeneratedQuery(
                    query_id=f"QR{len(queries)+1}",
                    text=text,
                    kind=QueryKind.RELATION,
                    priority=0.90 if relation.level.value == "hard" else 0.76,
                    based_on_relation_ids=[relation.id],
                    reason=f"保留关系语义：{relation.subject} -> {relation.predicate} -> {relation.object}",
                )
            )

        return queries

    def _generate_alias_queries(
        self,
        analysis: QueryAnalysisResult,
        budget: int,
        existing: list[GeneratedQuery],
    ) -> list[GeneratedQuery]:
        if budget <= 0:
            return []

        # 只使用 Query Understanding 已经确认过的 aliases，不在这里发明新同义词。
        alias_pairs: list[tuple[str, str]] = []
        for entity in analysis.entities:
            for alias in entity.aliases:
                if alias.strip() and normalize_query_text(alias) != normalize_query_text(entity.canonical_name):
                    alias_pairs.append((entity.canonical_name, alias))

        if not alias_pairs:
            return []

        bases = [q for q in existing if q.kind in {QueryKind.CORE, QueryKind.RELATION}]
        if not bases:
            return []

        output: list[GeneratedQuery] = []
        for canonical, alias in alias_pairs:
            for base in bases:
                replaced = _replace_phrase_ci(base.text, canonical, alias)
                if replaced == base.text:
                    # canonical 不在文本中时，尝试将 alias 与最相关 base 组合。
                    if base.kind == QueryKind.CORE:
                        replaced = f"{base.text} {alias}".strip()
                    else:
                        continue

                output.append(
                    GeneratedQuery(
                        query_id=f"QA{len(output)+1}",
                        text=replaced,
                        kind=QueryKind.ALIAS,
                        priority=0.82,
                        based_on_entity_names=[canonical, alias],
                        based_on_relation_ids=base.based_on_relation_ids,
                        reason=f"使用已确认 alias：{canonical} -> {alias}",
                    )
                )
                if len(output) >= budget:
                    return output

        return output

    # ------------------------------------------------------------------
    # One-shot LLM expansion
    # ------------------------------------------------------------------

    async def _generate_llm_expanded(
        self,
        analysis: QueryAnalysisResult,
        *,
        existing: list[GeneratedQuery],
        max_new: int,
        metrics: GenerationMetrics,
    ) -> list[GeneratedQuery]:
        assert self.client is not None

        messages = build_expansion_messages(
            analysis,
            existing_queries=[q.text for q in existing],
            max_queries=max_new,
        )

        response_schema = (
            get_llm_expansion_json_schema()
            if self.config.use_structured_output
            else None
        )

        response = await self._call_llm_once(
            messages=messages,
            response_schema=response_schema,
            metrics=metrics,
        )

        payload_dict = parse_json_payload(response.content)
        try:
            payload = LLMExpansionPayload.model_validate(payload_dict)
        except ValidationError as exc:
            raise ValueError(f"LLM expansion payload schema invalid: {exc}") from exc

        queries: list[GeneratedQuery] = []
        for item in payload.expanded_queries[:max_new]:
            if not item.query.strip():
                continue
            queries.append(
                GeneratedQuery(
                    query_id=f"QE{len(queries)+1}",
                    text=_clean_content_query(item.query),
                    kind=QueryKind.EXPANDED,
                    priority=0.72,
                    reason=item.reason,
                )
            )

        return queries

    async def _call_llm_once(
        self,
        *,
        messages: Sequence[ChatMessage],
        response_schema: Mapping[str, Any] | None,
        metrics: GenerationMetrics,
    ) -> LLMResponse:
        assert self.client is not None

        attempts = 1 + max(0, self.config.max_transport_retries)
        last_error: Exception | None = None

        for attempt in range(attempts):
            started = time.perf_counter()
            metrics.api_calls += 1
            try:
                response = await self.client.complete(
                    messages=messages,
                    temperature=self.config.temperature,
                    response_schema=response_schema,
                )
            except Exception as exc:
                metrics.llm_latency_ms += (time.perf_counter() - started) * 1000.0
                last_error = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(self.config.retry_backoff_seconds * (2**attempt))
                    continue
                raise

            metrics.llm_latency_ms += (time.perf_counter() - started) * 1000.0
            metrics.prompt_tokens += response.usage.prompt_tokens or 0
            metrics.completion_tokens += response.usage.completion_tokens or 0
            metrics.total_tokens += response.usage.total_tokens or 0
            if response.model and response.model not in metrics.model_names:
                metrics.model_names.append(response.model)
            return response

        raise RuntimeError(f"LLM expansion failed: {last_error}")


def _clean_content_query(text: str) -> str:
    text = " ".join(text.strip().split())
    # 去掉末尾问号等自然语言标点，但保留内部技术符号（如 GPT-4、RAG）。
    return text.strip(" ?？。.;；")


def _replace_phrase_ci(text: str, source: str, target: str) -> str:
    if not source.strip():
        return text
    pattern = re.compile(re.escape(source), flags=re.IGNORECASE)
    return pattern.sub(target, text, count=1)


def _relation_to_search_text(subject: str, predicate: str, object_: str) -> str:
    """
    学术搜索 API 对自然语言谓词支持有限，但完全删掉谓词又会损失关系。
    因此：
    - 对常见关系保留一个短谓词；
    - 其他关系使用 subject + predicate + object；
    - 不引入任何新术语。
    """
    subject = subject.strip()
    predicate = predicate.strip()
    object_ = object_.strip()
    if not subject or not object_:
        return ""

    pred_norm = predicate.lower().replace("_", " ")
    predicate_map = {
        "optimize": "optimization",
        "optimizes": "optimization",
        "optimized by": "optimization",
        "applied to": "applied to",
        "applied_to": "applied to",
        "evaluate on": "evaluation",
        "evaluated on": "evaluation",
        "evaluated_on": "evaluation",
        "compare with": "comparison",
        "compared with": "comparison",
        "compared_with": "comparison",
        "cites": "cites",
        "cited by": "cited by",
        "cited_by": "cited by",
    }
    connector = predicate_map.get(pred_norm, predicate)
    return _clean_content_query(f"{subject} {connector} {object_}")


__all__ = [
    "QueryGenerationLLMClient",
    "GeneratorConfig",
    "QueryGenerator",
]
