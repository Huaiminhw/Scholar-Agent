# query_understanding/validator.py
"""
QueryAnalysisResult 的业务逻辑校验与安全归一化。

职责：
- schema.py：字段/类型/枚举/Pydantic 模型级校验
- validator.py：跨字段、跨对象的业务一致性校验

本文件只把确定会影响下游执行的问题判为 ERROR；
对可能合理但值得观察的问题判为 WARNING，避免过度阻断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator

from .schema import (
    ConstraintItem,
    ConstraintOperator,
    ConstraintType,
    LogicNode,
    ParseStatus,
    QueryAnalysisResult,
    QueryType,
)


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: IssueSeverity
    path: str = ""
    refs: tuple[str, ...] = ()


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [x for x in self.issues if x.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [x for x in self.issues if x.severity == IssueSeverity.WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(
        self,
        code: str,
        message: str,
        severity: IssueSeverity,
        *,
        path: str = "",
        refs: Iterable[str] = (),
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=severity,
                path=path,
                refs=tuple(refs),
            )
        )

    def summary(self) -> str:
        if not self.issues:
            return "OK"

        lines: list[str] = []
        for issue in self.issues:
            loc = f" [{issue.path}]" if issue.path else ""
            refs = f" refs={list(issue.refs)}" if issue.refs else ""
            lines.append(
                f"{issue.severity.value.upper()} {issue.code}{loc}: "
                f"{issue.message}{refs}"
            )
        return "\n".join(lines)


def _iter_constraints(
    result: QueryAnalysisResult,
) -> Iterator[tuple[str, int, ConstraintItem]]:
    for bucket_name in ("hard", "soft", "exclude"):
        bucket = getattr(result.constraints, bucket_name)
        for index, item in enumerate(bucket):
            yield bucket_name, index, item


def _normalize_scalar(value: object) -> str:
    if value is None:
        return "<null>"
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return str(value).strip().lower()


def _constraint_signature(item: ConstraintItem) -> tuple:
    values = tuple(sorted(_normalize_scalar(v) for v in item.values))
    return (
        item.type.value,
        item.operator.value,
        _normalize_scalar(item.value),
        values,
    )


def _relation_signature(relation) -> tuple[str, str, str]:
    return (
        _normalize_scalar(relation.subject),
        _normalize_scalar(relation.predicate),
        _normalize_scalar(relation.object),
    )


def _walk_logic(
    node: LogicNode | None,
    path: str,
) -> Iterator[tuple[LogicNode, str]]:
    if node is None:
        return

    yield node, path

    for index, child in enumerate(node.children):
        yield from _walk_logic(child, f"{path}.children[{index}]")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_constraint_operator(
    item: ConstraintItem,
    path: str,
    report: ValidationReport,
) -> None:
    op = item.operator

    if not item.source_text.strip():
        report.add(
            "MISSING_SOURCE_TEXT",
            "建议保留该约束在用户原始查询中的对应文本，便于追溯与调试。",
            IssueSeverity.WARNING,
            path=f"{path}.source_text",
            refs=(item.id,),
        )

    if op in {ConstraintOperator.ANY_OF, ConstraintOperator.ALL_OF}:
        if item.value is not None:
            report.add(
                "MULTI_VALUE_OP_HAS_VALUE",
                f"{op.value} 应使用 values，value 应为 null。",
                IssueSeverity.ERROR,
                path=path,
                refs=(item.id,),
            )

        if len(item.values) < 2:
            report.add(
                "MULTI_VALUE_OP_TOO_FEW_VALUES",
                f"{op.value} 至少需要两个候选值。",
                IssueSeverity.ERROR,
                path=path,
                refs=(item.id,),
            )

        if item.type == ConstraintType.YEAR:
            if any(not isinstance(v, int) or isinstance(v, bool) for v in item.values):
                report.add(
                    "YEAR_VALUES_NOT_INTEGER",
                    "year 的多值约束必须使用整数年份。",
                    IssueSeverity.ERROR,
                    path=path,
                    refs=(item.id,),
                )
        return

    if op == ConstraintOperator.BETWEEN:
        if item.value is not None:
            report.add(
                "BETWEEN_HAS_VALUE",
                "between 应使用 values=[lower, upper]，value 应为 null。",
                IssueSeverity.ERROR,
                path=path,
                refs=(item.id,),
            )

        if len(item.values) != 2:
            report.add(
                "BETWEEN_BAD_ARITY",
                "between 必须且只能包含两个边界值。",
                IssueSeverity.ERROR,
                path=path,
                refs=(item.id,),
            )
            return

        if item.type == ConstraintType.YEAR:
            if any(not isinstance(v, int) or isinstance(v, bool) for v in item.values):
                report.add(
                    "YEAR_RANGE_NOT_INTEGER",
                    "year 的 between 两个边界必须是整数年份。",
                    IssueSeverity.ERROR,
                    path=path,
                    refs=(item.id,),
                )
            elif item.values[0] > item.values[1]:
                report.add(
                    "YEAR_RANGE_REVERSED",
                    "年份范围下界不能大于上界。",
                    IssueSeverity.ERROR,
                    path=path,
                    refs=(item.id,),
                )
        elif all(_is_number(v) for v in item.values):
            if item.values[0] > item.values[1]:
                report.add(
                    "BETWEEN_REVERSED",
                    "between 的下界不能大于上界。",
                    IssueSeverity.ERROR,
                    path=path,
                    refs=(item.id,),
                )
        return

    # 其余均按单值操作符处理
    if item.value is None:
        report.add(
            "SINGLE_VALUE_OP_MISSING_VALUE",
            f"{op.value} 需要设置 value。",
            IssueSeverity.ERROR,
            path=path,
            refs=(item.id,),
        )

    if item.values:
        report.add(
            "SINGLE_VALUE_OP_HAS_VALUES",
            f"{op.value} 应主要使用 value，不应同时携带 values。",
            IssueSeverity.WARNING,
            path=path,
            refs=(item.id,),
        )

    if op in {
        ConstraintOperator.GT,
        ConstraintOperator.GTE,
        ConstraintOperator.LT,
        ConstraintOperator.LTE,
    } and not _is_number(item.value):
        report.add(
            "NUMERIC_OPERATOR_NON_NUMERIC",
            f"{op.value} 应搭配数值型 value。",
            IssueSeverity.ERROR,
            path=path,
            refs=(item.id,),
        )

    if item.type == ConstraintType.YEAR:
        if item.value is not None and (
            not isinstance(item.value, int) or isinstance(item.value, bool)
        ):
            report.add(
                "YEAR_NOT_INTEGER",
                "year 约束的 value 应为整数年份。",
                IssueSeverity.ERROR,
                path=path,
                refs=(item.id,),
            )

    if item.type == ConstraintType.MIN_CITATIONS:
        if op in {
            ConstraintOperator.GT,
            ConstraintOperator.GTE,
            ConstraintOperator.LT,
            ConstraintOperator.LTE,
        } and not _is_number(item.value):
            report.add(
                "MIN_CITATIONS_NON_NUMERIC_THRESHOLD",
                "引用量比较阈值应为数字；'high' 这类表达应使用 eq。",
                IssueSeverity.ERROR,
                path=path,
                refs=(item.id,),
            )


def _validate_ids(
    result: QueryAnalysisResult,
    report: ValidationReport,
) -> None:
    seen: dict[str, str] = {}

    expected_prefix = {
        "hard": "C",
        "soft": "P",
        "exclude": "E",
    }

    for bucket, index, item in _iter_constraints(result):
        path = f"constraints.{bucket}[{index}].id"

        if item.id in seen:
            report.add(
                "DUPLICATE_ID",
                f"ID {item.id!r} 已在 {seen[item.id]} 使用。",
                IssueSeverity.ERROR,
                path=path,
                refs=(item.id,),
            )
        else:
            seen[item.id] = path

        if not item.id.startswith(expected_prefix[bucket]):
            report.add(
                "NONSTANDARD_CONSTRAINT_ID",
                f"{bucket} 约束建议使用 {expected_prefix[bucket]} 开头的 ID。",
                IssueSeverity.WARNING,
                path=path,
                refs=(item.id,),
            )

    for index, relation in enumerate(result.relations):
        path = f"relations[{index}].id"

        if relation.id in seen:
            report.add(
                "DUPLICATE_ID",
                f"ID {relation.id!r} 已在 {seen[relation.id]} 使用。",
                IssueSeverity.ERROR,
                path=path,
                refs=(relation.id,),
            )
        else:
            seen[relation.id] = path

        if not relation.id.startswith("R"):
            report.add(
                "NONSTANDARD_RELATION_ID",
                "关系约束建议使用 R 开头的 ID。",
                IssueSeverity.WARNING,
                path=path,
                refs=(relation.id,),
            )


def _validate_relations(
    result: QueryAnalysisResult,
    report: ValidationReport,
) -> None:
    for index, relation in enumerate(result.relations):
        path = f"relations[{index}]"

        if not relation.subject.strip():
            report.add(
                "EMPTY_RELATION_SUBJECT",
                "关系约束 subject 不能为空。",
                IssueSeverity.ERROR,
                path=f"{path}.subject",
                refs=(relation.id,),
            )

        if not relation.predicate.strip():
            report.add(
                "EMPTY_RELATION_PREDICATE",
                "关系约束 predicate 不能为空。",
                IssueSeverity.ERROR,
                path=f"{path}.predicate",
                refs=(relation.id,),
            )

        if not relation.object.strip():
            report.add(
                "EMPTY_RELATION_OBJECT",
                "关系约束 object 不能为空。",
                IssueSeverity.ERROR,
                path=f"{path}.object",
                refs=(relation.id,),
            )

        if not relation.source_text.strip():
            report.add(
                "MISSING_RELATION_SOURCE_TEXT",
                "建议保留该关系在原查询中的对应文本。",
                IssueSeverity.WARNING,
                path=f"{path}.source_text",
                refs=(relation.id,),
            )


def _validate_logic(
    result: QueryAnalysisResult,
    report: ValidationReport,
) -> None:
    constraint_level: dict[str, str] = {}
    for bucket, _, item in _iter_constraints(result):
        constraint_level[item.id] = bucket

    relation_level = {
        relation.id: relation.level.value
        for relation in result.relations
    }

    valid_refs = set(constraint_level) | set(relation_level)

    for bucket in ("hard", "soft", "exclude"):
        root = getattr(result.logic, bucket)

        for node, path in _walk_logic(root, f"logic.{bucket}"):
            if node.operator != "LEAF":
                continue

            ref_id = node.ref_id
            assert ref_id is not None  # schema.py 已保证

            if ref_id not in valid_refs:
                report.add(
                    "LOGIC_UNKNOWN_REFERENCE",
                    f"逻辑树引用了不存在的 ID {ref_id!r}。",
                    IssueSeverity.ERROR,
                    path=path,
                    refs=(ref_id,),
                )
                continue

            actual_level = (
                constraint_level.get(ref_id)
                or relation_level.get(ref_id)
            )

            if actual_level != bucket:
                report.add(
                    "LOGIC_CROSS_BUCKET_REFERENCE",
                    f"{bucket} 逻辑树引用了 level={actual_level} 的项目。",
                    IssueSeverity.ERROR,
                    path=path,
                    refs=(ref_id,),
                )


def _validate_cross_bucket_duplicates(
    result: QueryAnalysisResult,
    report: ValidationReport,
) -> None:
    signatures: dict[tuple, list[tuple[str, str]]] = {}

    for bucket, _, item in _iter_constraints(result):
        signatures.setdefault(
            _constraint_signature(item),
            [],
        ).append((bucket, item.id))

    for locations in signatures.values():
        buckets = {bucket for bucket, _ in locations}
        refs = [item_id for _, item_id in locations]

        if "hard" in buckets and "exclude" in buckets:
            report.add(
                "HARD_EXCLUDE_CONFLICT",
                "同一个约束同时出现在 hard 与 exclude 中。",
                IssueSeverity.ERROR,
                refs=refs,
            )
        elif "hard" in buckets and "soft" in buckets:
            report.add(
                "HARD_SOFT_DUPLICATE",
                "同一个约束同时出现在 hard 与 soft 中，通常属于冗余。",
                IssueSeverity.WARNING,
                refs=refs,
            )
        elif len(locations) > 1:
            report.add(
                "DUPLICATE_CONSTRAINT",
                "发现完全重复的约束。",
                IssueSeverity.WARNING,
                refs=refs,
            )

    relations: dict[tuple, list[tuple[str, str]]] = {}

    for relation in result.relations:
        relations.setdefault(
            _relation_signature(relation),
            [],
        ).append((relation.level.value, relation.id))

    for locations in relations.values():
        levels = {level for level, _ in locations}
        refs = [relation_id for _, relation_id in locations]

        if "hard" in levels and "exclude" in levels:
            report.add(
                "RELATION_HARD_EXCLUDE_CONFLICT",
                "同一语义关系同时被要求和排除。",
                IssueSeverity.ERROR,
                refs=refs,
            )
        elif len(locations) > 1:
            report.add(
                "DUPLICATE_RELATION",
                "发现重复的关系约束。",
                IssueSeverity.WARNING,
                refs=refs,
            )


def _validate_status_and_clarification(
    result: QueryAnalysisResult,
    report: ValidationReport,
) -> None:
    status = result.status
    needed = result.clarification.needed

    if status == ParseStatus.NEEDS_CLARIFICATION and not needed:
        report.add(
            "STATUS_CLARIFICATION_MISMATCH",
            "status=needs_clarification，但 clarification.needed=false。",
            IssueSeverity.ERROR,
            path="clarification.needed",
        )

    if status == ParseStatus.SUCCESS and needed:
        report.add(
            "STATUS_CLARIFICATION_MISMATCH",
            "status=success，但 clarification.needed=true。",
            IssueSeverity.ERROR,
            path="status",
        )

    if needed and not result.clarification.questions:
        report.add(
            "CLARIFICATION_WITHOUT_QUESTION",
            "clarification.needed=true 时至少应提供一个澄清问题。",
            IssueSeverity.ERROR,
            path="clarification.questions",
        )

    if not needed and result.clarification.questions:
        report.add(
            "UNNEEDED_CLARIFICATION_QUESTIONS",
            "clarification.needed=false，但仍提供了澄清问题。",
            IssueSeverity.WARNING,
            path="clarification.questions",
        )


def _validate_query_type_consistency(
    result: QueryAnalysisResult,
    report: ValidationReport,
) -> None:
    hard_types = {item.type for item in result.constraints.hard}

    if result.query_type == QueryType.SPECIFIC_BY_IDENTIFIER:
        if ConstraintType.IDENTIFIER not in hard_types:
            report.add(
                "IDENTIFIER_QUERY_WITHOUT_IDENTIFIER",
                "SPECIFIC_BY_IDENTIFIER 应包含硬性 identifier 约束。",
                IssueSeverity.ERROR,
                path="query_type",
            )

    elif result.query_type == QueryType.BY_AUTHOR:
        if ConstraintType.AUTHOR not in hard_types:
            report.add(
                "AUTHOR_QUERY_WITHOUT_AUTHOR_CONSTRAINT",
                "BY_AUTHOR 未发现硬性 author 约束。",
                IssueSeverity.WARNING,
                path="query_type",
            )

    elif result.query_type == QueryType.METADATA_ONLY_NO_AUTHOR:
        if result.content.strip():
            report.add(
                "METADATA_ONLY_HAS_CONTENT",
                "METADATA_ONLY_NO_AUTHOR 通常不应包含实质性 content。",
                IssueSeverity.WARNING,
                path="content",
            )

    elif result.query_type in {
        QueryType.SPECIFIC_BY_TITLE,
        QueryType.SPECIFIC_BY_NAME,
    }:
        if (
            not result.content.strip()
            and not result.properties.specific_paper_name
        ):
            report.add(
                "SPECIFIC_QUERY_WITHOUT_LOCATOR",
                "特定论文/名称查询缺少 content 或 specific_paper_name。",
                IssueSeverity.WARNING,
                path="query_type",
            )


def _validate_original_query(
    result: QueryAnalysisResult,
    original_query: str | None,
    report: ValidationReport,
) -> None:
    if original_query is not None and result.original_query != original_query:
        report.add(
            "ORIGINAL_QUERY_CHANGED",
            "original_query 与真实用户输入不一致。",
            IssueSeverity.ERROR,
            path="original_query",
        )


def _validate_entities(
    result: QueryAnalysisResult,
    report: ValidationReport,
) -> None:
    seen: set[tuple[str, str]] = set()

    for index, entity in enumerate(result.entities):
        key = (
            entity.type.value,
            _normalize_scalar(entity.canonical_name),
        )

        if key in seen:
            report.add(
                "DUPLICATE_ENTITY",
                "发现重复实体。",
                IssueSeverity.WARNING,
                path=f"entities[{index}]",
            )

        seen.add(key)


def validate_query_analysis(
    result: QueryAnalysisResult,
    *,
    original_query: str | None = None,
) -> ValidationReport:
    """
    对已经通过 Pydantic 的 QueryAnalysisResult 做业务逻辑校验。
    """
    report = ValidationReport()

    _validate_original_query(result, original_query, report)
    _validate_ids(result, report)

    for bucket, index, item in _iter_constraints(result):
        _validate_constraint_operator(
            item,
            f"constraints.{bucket}[{index}]",
            report,
        )

    _validate_relations(result, report)
    _validate_logic(result, report)
    _validate_cross_bucket_duplicates(result, report)
    _validate_status_and_clarification(result, report)
    _validate_query_type_consistency(result, report)
    _validate_entities(result, report)

    return report


def _dedupe_strings(values: list[str]) -> list[str]:
    """保持原始顺序、大小写不敏感去重。"""
    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = _normalize_scalar(value)

        if normalized in {"", "<null>"} or normalized in seen:
            continue

        seen.add(normalized)
        output.append(value.strip())

    return output


def safe_normalize(
    result: QueryAnalysisResult,
    *,
    original_query: str | None = None,
) -> QueryAnalysisResult:
    """
    只执行不会改变用户检索语义的安全修复。

    会做：
    - 恢复真实 original_query
    - aliases / secondary_fields / required_fields / warnings 去重
    - clarification.needed=true 且 status=success 时，将 status 调整为
      needs_clarification

    不会做：
    - hard ↔ soft 转换
    - 新增/删除语义约束
    - 猜测年份、作者、venue、方法
    - 自动修改 query_type
    """
    normalized = result.model_copy(deep=True)

    if original_query is not None:
        normalized.original_query = original_query

    normalized.domains.secondary_fields = _dedupe_strings(
        normalized.domains.secondary_fields
    )

    normalized.result_requirements.required_fields = _dedupe_strings(
        normalized.result_requirements.required_fields
    )

    normalized.warnings = _dedupe_strings(
        normalized.warnings
    )

    for entity in normalized.entities:
        entity.aliases = _dedupe_strings(entity.aliases)

    if (
        normalized.clarification.needed
        and normalized.status == ParseStatus.SUCCESS
    ):
        normalized.status = ParseStatus.NEEDS_CLARIFICATION

    return normalized


__all__ = [
    "IssueSeverity",
    "ValidationIssue",
    "ValidationReport",
    "validate_query_analysis",
    "safe_normalize",
]
