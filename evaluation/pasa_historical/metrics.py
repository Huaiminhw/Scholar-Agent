"""Exact-arXiv-ID candidate-pool recall for PaSa historical evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .candidate_merge import HistoricalCandidate
from .protocol import CachedQueryCase, normalize_arxiv_id


@dataclass
class HistoricalRecallResult:
    dataset_index: int
    qid: str
    question: str
    gold_count: int
    matched_gold_count: int
    recall: float
    raw_hits: int
    candidate_count: int
    duplicate_hits_removed: int
    api_calls: int
    retrieval_latency_ms: float
    matched_gold: list[dict[str, Any]]
    missed_gold: list[dict[str, Any]]
    source_recall: dict[str, float]
    source_matched_gold_count: dict[str, int]
    routes: list[dict[str, Any]]
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_arxiv_ids(
    candidates: Iterable[HistoricalCandidate],
    *,
    source: str | None = None,
) -> set[str]:
    output: set[str] = set()
    for candidate in candidates:
        if source is not None and source not in candidate.sources:
            continue
        output.update(
            value
            for value in (normalize_arxiv_id(item) for item in candidate.arxiv_ids)
            if value
        )
    return output


def evaluate_candidate_pool(
    *,
    case: CachedQueryCase,
    candidates: Sequence[HistoricalCandidate],
    raw_hits: int,
    api_calls: int,
    retrieval_latency_ms: float,
    routes: list[dict[str, Any]],
    sources: Sequence[str],
) -> HistoricalRecallResult:
    candidate_ids = _candidate_arxiv_ids(candidates)
    matched_indices = {
        index
        for index, gold_id in enumerate(case.gold_arxiv_ids)
        if gold_id is not None and gold_id in candidate_ids
    }
    gold_count = len(case.gold_arxiv_ids)

    def gold_item(index: int) -> dict[str, Any]:
        return {
            "arxiv_id": case.gold_arxiv_ids[index],
            "title": case.gold_titles[index],
        }

    source_counts: dict[str, int] = {}
    source_recall: dict[str, float] = {}
    for source in sources:
        ids = _candidate_arxiv_ids(candidates, source=source)
        count = sum(
            gold_id is not None and gold_id in ids for gold_id in case.gold_arxiv_ids
        )
        source_counts[source] = count
        source_recall[source] = count / gold_count if gold_count else 0.0

    return HistoricalRecallResult(
        dataset_index=case.dataset_index,
        qid=case.qid,
        question=case.question,
        gold_count=gold_count,
        matched_gold_count=len(matched_indices),
        recall=len(matched_indices) / gold_count if gold_count else 0.0,
        raw_hits=raw_hits,
        candidate_count=len(candidates),
        duplicate_hits_removed=raw_hits - len(candidates),
        api_calls=api_calls,
        retrieval_latency_ms=retrieval_latency_ms,
        matched_gold=[gold_item(index) for index in sorted(matched_indices)],
        missed_gold=[
            gold_item(index) for index in range(gold_count) if index not in matched_indices
        ],
        source_recall=source_recall,
        source_matched_gold_count=source_counts,
        routes=routes,
        candidates=[candidate.to_dict() for candidate in candidates],
    )


def aggregate(results: Sequence[HistoricalRecallResult]) -> dict[str, Any]:
    gold_total = sum(result.gold_count for result in results)
    matched_total = sum(result.matched_gold_count for result in results)
    sources = sorted(
        {
            source
            for result in results
            for source in result.source_matched_gold_count
        }
    )
    source_hits = {
        source: sum(
            result.source_matched_gold_count.get(source, 0) for result in results
        )
        for source in sources
    }
    return {
        "query_count": len(results),
        "gold_total": gold_total,
        "matched_gold_total": matched_total,
        "micro_candidate_pool_recall": (
            matched_total / gold_total if gold_total else 0.0
        ),
        "macro_candidate_pool_recall": (
            sum(result.recall for result in results) / len(results) if results else 0.0
        ),
        "source_exact_id_hits": source_hits,
        "source_micro_recall": {
            source: hits / gold_total if gold_total else 0.0
            for source, hits in source_hits.items()
        },
        "total_raw_hits": sum(result.raw_hits for result in results),
        "total_candidates_after_dedup": sum(
            result.candidate_count for result in results
        ),
        "total_duplicate_hits_removed": sum(
            result.duplicate_hits_removed for result in results
        ),
        "total_api_calls": sum(result.api_calls for result in results),
        "total_retrieval_latency_ms": sum(
            result.retrieval_latency_ms for result in results
        ),
        "mean_retrieval_latency_ms": (
            sum(result.retrieval_latency_ms for result in results) / len(results)
            if results
            else 0.0
        ),
    }


def comparison_row(label: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a historical/live/frozen summary for later comparison tables."""

    recall = summary.get("micro_candidate_pool_recall")
    if recall is None:
        recall = summary.get("micro_union_recall")
    if recall is None:
        recall = summary.get("micro_recall_at", {}).get("100")
    return {
        "label": label,
        "gold_total": summary.get("gold_total"),
        "matched_gold_total": summary.get(
            "matched_gold_total",
            summary.get("union_exact_id_hits", summary.get("original_exact_id_hits_at_100")),
        ),
        "recall": recall,
    }


__all__ = [
    "HistoricalRecallResult",
    "aggregate",
    "comparison_row",
    "evaluate_candidate_pool",
]
