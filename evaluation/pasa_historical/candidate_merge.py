"""Deterministic union and conservative deduplication for historical hits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from scholar_agent_v2.candidate_pool.dedup import normalize_title
from scholar_agent_v2.retrieval.models import RetrievalHit
from scholar_agent_v2.retrieval.normalizer import normalize_arxiv_id, normalize_doi


@dataclass
class HistoricalCandidate:
    candidate_id: str
    title: str | None
    year: int | None
    arxiv_ids: list[str] = field(default_factory=list)
    dois: list[str] = field(default_factory=list)
    semantic_scholar_ids: list[str] = field(default_factory=list)
    openalex_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    origins: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strong_keys(hit: RetrievalHit) -> list[str]:
    keys: list[str] = []
    arxiv_id = normalize_arxiv_id(hit.arxiv_id)
    doi = normalize_doi(hit.doi)
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id.casefold()}")
    if doi:
        keys.append(f"doi:{doi}")
    if hit.semantic_scholar_id:
        keys.append(f"s2:{hit.semantic_scholar_id.casefold()}")
    if hit.openalex_id:
        keys.append(f"oa:{hit.openalex_id.casefold()}")
    if hit.pmid:
        keys.append(f"pmid:{hit.pmid.casefold()}")
    # Exact title+year is a conservative fallback for records whose APIs expose
    # no shared identifier.  No fuzzy-title merge is used in this first round.
    title = normalize_title(hit.title)
    if title and hit.year is not None:
        keys.append(f"title-year:{title}:{hit.year}")
    return keys


def merge_hits(hits: Iterable[RetrievalHit]) -> list[HistoricalCandidate]:
    values = list(hits)
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    key_owner: dict[str, int] = {}
    for index, hit in enumerate(values):
        for key in _strong_keys(hit):
            owner = key_owner.get(key)
            if owner is None:
                key_owner[key] = index
            else:
                union(owner, index)

    groups: dict[int, list[RetrievalHit]] = {}
    order: list[int] = []
    for index, hit in enumerate(values):
        root = find(index)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(hit)

    candidates: list[HistoricalCandidate] = []
    for candidate_number, root in enumerate(order, start=1):
        group = groups[root]
        arxiv_ids = sorted(
            {
                value
                for value in (normalize_arxiv_id(hit.arxiv_id) for hit in group)
                if value
            }
        )
        dois = sorted(
            {value for value in (normalize_doi(hit.doi) for hit in group) if value}
        )
        s2_ids = sorted({hit.semantic_scholar_id for hit in group if hit.semantic_scholar_id})
        oa_ids = sorted({hit.openalex_id for hit in group if hit.openalex_id})
        sources = sorted({hit.source.value for hit in group})
        title = next((hit.title for hit in group if hit.title), None)
        year = next((hit.year for hit in group if hit.year is not None), None)
        origins = [
            {
                "source": hit.origin.source.value,
                "route_id": hit.origin.route_id,
                "query_id": hit.origin.query_id,
                "query_text": hit.origin.query_text,
                "query_kind": hit.origin.query_kind.value,
                "strategy": hit.origin.strategy.value,
                "rank": hit.origin.rank,
                "source_score": hit.origin.source_score,
            }
            for hit in group
        ]
        preferred_id = (
            f"arxiv:{arxiv_ids[0]}"
            if arxiv_ids
            else f"doi:{dois[0]}"
            if dois
            else f"s2:{s2_ids[0]}"
            if s2_ids
            else f"oa:{oa_ids[0]}"
            if oa_ids
            else f"historical:{candidate_number}"
        )
        candidates.append(
            HistoricalCandidate(
                candidate_id=preferred_id,
                title=title,
                year=year,
                arxiv_ids=arxiv_ids,
                dois=dois,
                semantic_scholar_ids=s2_ids,
                openalex_ids=oa_ids,
                sources=sources,
                origins=origins,
            )
        )
    return candidates


__all__ = ["HistoricalCandidate", "merge_hits"]
