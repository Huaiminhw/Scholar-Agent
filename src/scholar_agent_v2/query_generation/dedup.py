from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from .models import GeneratedQuery, QueryKind


_TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)


_KIND_KEEP_PRIORITY = {
    QueryKind.EXACT: 6,
    QueryKind.CORE: 5,
    QueryKind.RELATION: 4,
    QueryKind.ALIAS: 3,
    QueryKind.EXPANDED: 2,
}


def normalize_query_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = re.sub(r"[\"'`“”‘’]+", " ", text)
    text = re.sub(r"[^\w\-\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def token_set(text: str) -> set[str]:
    normalized = normalize_query_text(text)
    return {m.group(0) for m in _TOKEN_RE.finditer(normalized)}


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def near_duplicate(a: str, b: str, *, threshold: float = 0.88) -> bool:
    na, nb = normalize_query_text(a), normalize_query_text(b)
    if na == nb:
        return True

    seq = SequenceMatcher(None, na, nb).ratio()
    jac = jaccard_similarity(na, nb)

    # 两个指标任一非常高，或二者都较高，即视为近重复。
    return seq >= 0.94 or jac >= threshold or (seq >= 0.88 and jac >= 0.80)


def _prefer(a: GeneratedQuery, b: GeneratedQuery) -> GeneratedQuery:
    """重复时优先保留更“原生”、优先级更高的 Query。"""
    ka = (_KIND_KEEP_PRIORITY[a.kind], a.priority)
    kb = (_KIND_KEEP_PRIORITY[b.kind], b.priority)
    return a if ka >= kb else b


def deduplicate_queries(
    queries: list[GeneratedQuery],
    *,
    threshold: float = 0.88,
    max_queries: int | None = None,
) -> list[GeneratedQuery]:
    """
    保持总体顺序稳定的近重复去重。
    CORE/EXACT 优先于 RELATION/ALIAS/EXPANDED。
    """
    kept: list[GeneratedQuery] = []

    for query in queries:
        duplicate_index: int | None = None
        for i, existing in enumerate(kept):
            if near_duplicate(query.text, existing.text, threshold=threshold):
                duplicate_index = i
                break

        if duplicate_index is None:
            kept.append(query)
        else:
            kept[duplicate_index] = _prefer(kept[duplicate_index], query)

    kept.sort(key=lambda q: q.priority, reverse=True)

    if max_queries is not None:
        kept = kept[:max_queries]

    # 去重后重新稳定编号，便于 Origins / Route 追踪。
    counters: dict[QueryKind, int] = {}
    output: list[GeneratedQuery] = []
    for q in kept:
        counters[q.kind] = counters.get(q.kind, 0) + 1
        prefix = {
            QueryKind.EXACT: "QX",
            QueryKind.CORE: "QC",
            QueryKind.RELATION: "QR",
            QueryKind.ALIAS: "QA",
            QueryKind.EXPANDED: "QE",
        }[q.kind]
        output.append(q.model_copy(update={"query_id": f"{prefix}{counters[q.kind]}"}))

    return output


__all__ = [
    "normalize_query_text",
    "token_set",
    "jaccard_similarity",
    "near_duplicate",
    "deduplicate_queries",
]
