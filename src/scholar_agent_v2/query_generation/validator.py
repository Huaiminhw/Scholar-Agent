from __future__ import annotations

import re
from typing import Iterable

from ..query_understanding.schema import (
    ConstraintItem,
    ConstraintType,
    QueryAnalysisResult,
    SearchTask,
    RetrievalMode,
)

from .dedup import normalize_query_text, token_set
from .models import (
    GeneratedQuery,
    QueryKind,
    QueryValidationIssue,
    QueryValidationReport,
)


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")
_SURVEY_TERMS = {
    "survey",
    "review",
    "literature review",
    "systematic review",
    "state of the art",
    "state-of-the-art",
    "meta analysis",
    "meta-analysis",
    "taxonomy",
}


def _flatten_values(items: Iterable[ConstraintItem], ctype: ConstraintType) -> list[str]:
    output: list[str] = []
    for item in items:
        if item.type != ctype:
            continue
        if item.value is not None:
            output.append(str(item.value))
        output.extend(str(x) for x in item.values)
    return output


def _all_constraints(analysis: QueryAnalysisResult) -> list[ConstraintItem]:
    return [
        *analysis.constraints.hard,
        *analysis.constraints.soft,
        *analysis.constraints.exclude,
    ]


def _allowed_years(analysis: QueryAnalysisResult) -> set[str]:
    allowed = set(_YEAR_RE.findall(analysis.original_query))
    for value in _flatten_values(_all_constraints(analysis), ConstraintType.YEAR):
        allowed.update(_YEAR_RE.findall(value))
    return allowed


def _explicit_review_intent(analysis: QueryAnalysisResult) -> bool:
    if analysis.intent.task == SearchTask.LITERATURE_REVIEW:
        return True
    if analysis.intent.mode == RetrievalMode.OVERVIEW:
        return True

    # EXCLUDE 中的 survey/review 表示明确排除，不能被解释为正向综述意图。
    for item in [*analysis.constraints.hard, *analysis.constraints.soft]:
        if item.type != ConstraintType.PUBLICATION_TYPE:
            continue
        vals = ([item.value] if item.value is not None else []) + list(item.values)
        joined = " ".join(str(v).lower() for v in vals)
        if any(term in joined for term in ("survey", "review", "meta-analysis", "meta analysis")):
            return True
    return False


def _known_acronyms(analysis: QueryAnalysisResult) -> set[str]:
    """收集查询理解阶段已经确认的缩写，防止扩展擅自引入具体方法。"""
    trusted_texts = [analysis.original_query, analysis.content, analysis.intent.goal]

    for entity in analysis.entities:
        trusted_texts.extend(
            [entity.canonical_name, entity.source_text, *entity.aliases]
        )
    for relation in analysis.relations:
        trusted_texts.extend(
            [relation.subject, relation.predicate, relation.object, relation.source_text]
        )
    for item in _all_constraints(analysis):
        trusted_texts.append(item.source_text)
        if item.value is not None:
            trusted_texts.append(str(item.value))
        trusted_texts.extend(str(value) for value in item.values)

    return {
        match.group(0).lower()
        for text in trusted_texts
        for match in _ACRONYM_RE.finditer(text or "")
    }


def _known_soft_literals(analysis: QueryAnalysisResult) -> set[str]:
    literals: set[str] = set()
    for item in analysis.constraints.soft:
        if item.type not in {
            ConstraintType.VENUE,
            ConstraintType.YEAR,
            ConstraintType.AUTHOR,
            ConstraintType.PUBLICATION_TYPE,
            ConstraintType.MIN_CITATIONS,
        }:
            continue
        vals = ([item.value] if item.value is not None else []) + list(item.values)
        for value in vals:
            if value is not None:
                text = normalize_query_text(str(value))
                if text:
                    literals.add(text)
    return literals


def _core_anchors(analysis: QueryAnalysisResult) -> set[str]:
    anchors: set[str] = set()

    for entity in analysis.entities:
        anchors.update(token_set(entity.canonical_name))
        for alias in entity.aliases:
            anchors.update(token_set(alias))

    for relation in analysis.relations:
        anchors.update(token_set(relation.subject))
        anchors.update(token_set(relation.object))

    if not anchors:
        anchors.update(token_set(analysis.content))

    # 单字符/常见虚词不作为 anchor。
    return {x for x in anchors if len(x) > 1}


def validate_generated_query(
    query: GeneratedQuery,
    analysis: QueryAnalysisResult,
) -> QueryValidationReport:
    issues: list[QueryValidationIssue] = []
    text_norm = normalize_query_text(query.text)

    if not text_norm:
        issues.append(
            QueryValidationIssue(
                code="EMPTY_QUERY",
                message="生成查询不能为空。",
                query_id=query.query_id,
                severity="error",
            )
        )
        return QueryValidationReport(issues=issues)

    # 1) LLM Expanded 不允许擅自添加年份。
    if query.kind == QueryKind.EXPANDED:
        allowed_years = _allowed_years(analysis)
        introduced_years = set(_YEAR_RE.findall(query.text)) - allowed_years
        if introduced_years:
            issues.append(
                QueryValidationIssue(
                    code="INVENTED_YEAR",
                    message=f"扩展查询引入了原需求没有的年份：{sorted(introduced_years)}",
                    query_id=query.query_id,
                    severity="error",
                )
            )

    # 2) 用户未要求综述时，不允许 LLM 擅自改成 survey/review。
    if query.kind == QueryKind.EXPANDED and not _explicit_review_intent(analysis):
        if any(term in text_norm for term in _SURVEY_TERMS):
            issues.append(
                QueryValidationIssue(
                    code="INVENTED_REVIEW_INTENT",
                    message="用户未要求综述/Review，但扩展查询加入了 survey/review 类限制。",
                    query_id=query.query_id,
                    severity="error",
                )
            )

    # 3) soft metadata 不应被塞进主内容查询，否则等价于把 soft 变 hard。
    if query.kind in {QueryKind.CORE, QueryKind.RELATION, QueryKind.ALIAS, QueryKind.EXPANDED}:
        soft_literals = _known_soft_literals(analysis)
        for literal in soft_literals:
            if literal and literal in text_norm:
                issues.append(
                    QueryValidationIssue(
                        code="SOFT_METADATA_HARDENED",
                        message=f"查询文本包含 soft metadata {literal!r}，可能把偏好误变成硬检索条件。",
                        query_id=query.query_id,
                        severity="error",
                    )
                )

    # 4) 扩展 Query 至少要保留一部分核心 anchor，防明显 Query Drift。
    if query.kind == QueryKind.EXPANDED:
        anchors = _core_anchors(analysis)
        query_tokens = token_set(query.text)
        if anchors and not (anchors & query_tokens):
            issues.append(
                QueryValidationIssue(
                    code="CORE_ANCHOR_LOST",
                    message="扩展查询未保留任何已知核心实体/关系锚点，疑似 Query Drift。",
                    query_id=query.query_id,
                    severity="error",
                )
            )

        # 大写缩写通常代表具体模型、算法或数据集。只有查询理解阶段已经
        # 确认过的缩写才允许进入扩展，避免 RL 被擅自缩窄成 PPO 等子方法。
        introduced_acronyms = {
            match.group(0).lower() for match in _ACRONYM_RE.finditer(query.text)
        } - _known_acronyms(analysis)
        if introduced_acronyms:
            issues.append(
                QueryValidationIssue(
                    code="INVENTED_NAMED_METHOD",
                    message=(
                        "扩展查询引入了原需求未确认的模型、算法或数据集缩写："
                        f"{sorted(introduced_acronyms)}"
                    ),
                    query_id=query.query_id,
                    severity="error",
                )
            )

    # 5) Relation query 必须保留对应关系两端至少各一个词。
    if query.kind == QueryKind.RELATION and query.based_on_relation_ids:
        relation_map = {r.id: r for r in analysis.relations}
        qtokens = token_set(query.text)
        for relation_id in query.based_on_relation_ids:
            relation = relation_map.get(relation_id)
            if relation is None:
                issues.append(
                    QueryValidationIssue(
                        code="UNKNOWN_RELATION_REF",
                        message=f"查询引用了不存在的 relation id={relation_id}",
                        query_id=query.query_id,
                        severity="error",
                    )
                )
                continue

            subject_tokens = token_set(relation.subject)
            object_tokens = token_set(relation.object)
            if subject_tokens and not (subject_tokens & qtokens):
                issues.append(
                    QueryValidationIssue(
                        code="RELATION_SUBJECT_LOST",
                        message=f"Relation Query 丢失 subject：{relation.subject}",
                        query_id=query.query_id,
                        severity="error",
                    )
                )
            if object_tokens and not (object_tokens & qtokens):
                issues.append(
                    QueryValidationIssue(
                        code="RELATION_OBJECT_LOST",
                        message=f"Relation Query 丢失 object：{relation.object}",
                        query_id=query.query_id,
                        severity="error",
                    )
                )

    # 6) 过长查询通常不适合学术 API 搜索；只 warning，不阻断。
    if len(query.text.split()) > 18:
        issues.append(
            QueryValidationIssue(
                code="QUERY_TOO_LONG",
                message="查询超过 18 个空格分词，建议保持更简洁。",
                query_id=query.query_id,
                severity="warning",
            )
        )

    return QueryValidationReport(issues=issues)


def validate_queries(
    queries: list[GeneratedQuery],
    analysis: QueryAnalysisResult,
) -> QueryValidationReport:
    issues: list[QueryValidationIssue] = []
    for query in queries:
        issues.extend(validate_generated_query(query, analysis).issues)
    return QueryValidationReport(issues=issues)


def filter_invalid_queries(
    queries: list[GeneratedQuery],
    analysis: QueryAnalysisResult,
) -> tuple[list[GeneratedQuery], QueryValidationReport]:
    """删除 ERROR Query；WARNING 保留并写回 query.warnings。"""
    kept: list[GeneratedQuery] = []
    all_issues: list[QueryValidationIssue] = []

    for query in queries:
        report = validate_generated_query(query, analysis)
        all_issues.extend(report.issues)
        if report.errors:
            continue

        warnings = [x.message for x in report.warnings]
        kept.append(query.model_copy(update={"warnings": [*query.warnings, *warnings]}))

    return kept, QueryValidationReport(issues=all_issues)


__all__ = [
    "validate_generated_query",
    "validate_queries",
    "filter_invalid_queries",
]
