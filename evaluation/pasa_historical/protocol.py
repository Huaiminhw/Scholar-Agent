"""Frozen cache loading and deterministic route construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scholar_agent_v2.query_generation.models import QueryKind
from scholar_agent_v2.retrieval.models import (
    RetrievalFilters,
    RetrievalPlan,
    RetrievalStrategy,
    RouteSpec,
    SearchSource,
)


@dataclass(frozen=True)
class CachedQueryCase:
    dataset_index: int
    qid: str
    question: str
    gold_titles: list[str]
    gold_arxiv_ids: list[str | None]
    cache_created_at: str | None
    cache_protocol_sha256: str
    academic_query_ir: dict[str, Any]
    generated_queries: list[dict[str, Any]]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_arxiv_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "/abs/" in text:
        text = text.split("/abs/", 1)[1]
    elif "/pdf/" in text:
        text = text.split("/pdf/", 1)[1]
    if text.endswith(".pdf"):
        text = text[:-4]
    import re

    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", text, flags=re.IGNORECASE) or None


def load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid {label} JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} line {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def load_cases(
    *,
    dataset_path: Path,
    cache_path: Path,
    start: int,
    limit: int,
) -> tuple[list[CachedQueryCase], str, str]:
    dataset = load_jsonl(dataset_path, label="RealScholarQuery dataset")
    cache_rows = load_jsonl(cache_path, label="V2 query cache")
    cache = {row.get("dataset_index"): row for row in cache_rows}
    if any(not isinstance(index, int) for index in cache):
        raise ValueError("Every cache record must have an integer dataset_index")

    dataset_hash = sha256_file(dataset_path)
    end = min(len(dataset), start + limit)
    indices = list(range(start, end))
    if not indices:
        raise ValueError(
            f"No records selected: start={start}, limit={limit}, size={len(dataset)}"
        )

    cases: list[CachedQueryCase] = []
    protocol_hashes: set[str] = set()
    problems: list[str] = []
    for index in indices:
        source = dataset[index]
        frozen = cache.get(index)
        if frozen is None:
            problems.append(f"dataset_index={index}: missing cache record")
            continue
        if frozen.get("status") != "success":
            problems.append(f"dataset_index={index}: cache is not successful")
            continue
        question = str(source.get("question", "")).strip()
        if not question or frozen.get("question") != question:
            problems.append(f"dataset_index={index}: question mismatch")
            continue
        if frozen.get("dataset_sha256") != dataset_hash:
            problems.append(f"dataset_index={index}: dataset SHA256 mismatch")
            continue
        protocol_hash = frozen.get("protocol_sha256")
        if not isinstance(protocol_hash, str) or not protocol_hash:
            problems.append(f"dataset_index={index}: missing protocol SHA256")
            continue
        protocol_hashes.add(protocol_hash)

        titles = source.get("answer")
        raw_ids = source.get("answer_arxiv_id")
        generated = frozen.get("search_queries")
        ir = frozen.get("academic_query_ir")
        if not isinstance(titles, list) or not isinstance(raw_ids, list):
            problems.append(f"dataset_index={index}: invalid Gold fields")
            continue
        if len(titles) != len(raw_ids):
            problems.append(f"dataset_index={index}: Gold title/ID length mismatch")
            continue
        if not isinstance(generated, list) or not isinstance(ir, dict):
            problems.append(f"dataset_index={index}: missing frozen V2 outputs")
            continue
        if any(not isinstance(item, dict) or not str(item.get("text", "")).strip() for item in generated):
            problems.append(f"dataset_index={index}: invalid generated query")
            continue

        cases.append(
            CachedQueryCase(
                dataset_index=index,
                qid=str(source.get("qid", f"RealScholarQuery_{index}")),
                question=question,
                gold_titles=[str(value).strip() for value in titles],
                gold_arxiv_ids=[normalize_arxiv_id(value) for value in raw_ids],
                cache_created_at=str(frozen.get("created_at", "")) or None,
                cache_protocol_sha256=protocol_hash,
                academic_query_ir=ir,
                generated_queries=generated,
            )
        )

    if len(protocol_hashes) > 1:
        problems.append("selected cache records mix multiple LLM/config protocols")
    if problems:
        raise RuntimeError(
            "Frozen query cache validation failed:\n  - " + "\n  - ".join(problems)
        )
    return cases, dataset_hash, next(iter(protocol_hashes))


def _strategy(case: CachedQueryCase, kind: QueryKind) -> RetrievalStrategy:
    query_type = str(case.academic_query_ir.get("query_type", ""))
    if query_type == "SPECIFIC_BY_IDENTIFIER":
        return RetrievalStrategy.IDENTIFIER
    if query_type == "SPECIFIC_BY_TITLE":
        return RetrievalStrategy.TITLE
    if query_type == "BY_AUTHOR":
        return RetrievalStrategy.AUTHOR
    if kind == QueryKind.EXACT:
        return RetrievalStrategy.TITLE
    return RetrievalStrategy.RELEVANCE


def build_plan(
    case: CachedQueryCase,
    *,
    sources: Sequence[SearchSource],
    per_route_limit: int,
) -> RetrievalPlan:
    query_specs: list[tuple[str, str, QueryKind, str]] = [
        ("QO", case.question, QueryKind.CORE, "original")
    ]
    for position, item in enumerate(case.generated_queries, start=1):
        try:
            kind = QueryKind(str(item.get("kind", QueryKind.CORE.value)))
        except ValueError as exc:
            raise ValueError(
                f"dataset_index={case.dataset_index}: invalid QueryKind {item.get('kind')!r}"
            ) from exc
        query_specs.append(
            (
                str(item.get("query_id", f"QG{position}")),
                str(item["text"]).strip(),
                kind,
                "generated",
            )
        )

    routes: list[RouteSpec] = []
    for query_position, (query_id, text, kind, query_source) in enumerate(query_specs):
        for source in sources:
            route_number = len(routes) + 1
            routes.append(
                RouteSpec(
                    route_id=f"H{case.dataset_index:02d}-RT{route_number:03d}",
                    query_id=f"{query_source}:{query_position}:{query_id}",
                    query_text=text,
                    query_kind=kind,
                    source=source,
                    strategy=_strategy(case, kind),
                    priority=1.0,
                    limit=per_route_limit,
                    filters=RetrievalFilters(),
                    notes=[f"frozen_query_source={query_source}"],
                )
            )

    return RetrievalPlan(
        routes=routes,
        max_routes=max(1, len(routes)),
        max_total_hits=max(1, len(routes) * per_route_limit),
        warnings=[],
    )


__all__ = [
    "CachedQueryCase",
    "build_plan",
    "load_cases",
    "normalize_arxiv_id",
    "sha256_file",
]
