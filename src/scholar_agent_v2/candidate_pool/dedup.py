from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from ..retrieval.models import RetrievalHit
from ..retrieval.normalizer import normalize_arxiv_id, normalize_doi


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    text = unicodedata.normalize("NFKC", title).casefold()
    text = text.replace("–", "-").replace("—", "-")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_person(name: str | None) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", name).casefold()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _first_author(hit: RetrievalHit) -> str:
    return normalize_person(hit.authors[0]) if hit.authors else ""


def _year_compatible(a: int | None, b: int | None, *, tolerance: int = 1) -> bool:
    if a is None or b is None:
        return True
    return abs(a - b) <= tolerance


def _author_compatible(a: RetrievalHit, b: RetrievalHit) -> bool:
    aa, bb = _first_author(a), _first_author(b)
    if not aa or not bb:
        return True
    if aa == bb:
        return True
    a_tokens, b_tokens = set(aa.split()), set(bb.split())
    return bool(a_tokens & b_tokens) and SequenceMatcher(None, aa, bb).ratio() >= 0.78


def _identifier_map(hit: RetrievalHit) -> dict[str, str]:
    out: dict[str, str] = {}
    doi = normalize_doi(hit.doi)
    arxiv = normalize_arxiv_id(hit.arxiv_id)
    if doi:
        out["doi"] = doi
    if arxiv:
        out["arxiv"] = arxiv.casefold()
    if hit.semantic_scholar_id:
        out["s2"] = hit.semantic_scholar_id.casefold()
    if hit.openalex_id:
        out["openalex"] = hit.openalex_id.casefold()
    if hit.pmid:
        out["pmid"] = hit.pmid.casefold()
    return out


@dataclass(frozen=True)
class DuplicateDecision:
    is_duplicate: bool
    method: str | None = None
    score: float | None = None


def duplicate_decision(a: RetrievalHit, b: RetrievalHit, *, fuzzy_threshold: float = 0.965) -> DuplicateDecision:
    ia, ib = _identifier_map(a), _identifier_map(b)

    # DOI / arXiv 最适合跨源实体解析；其余 source-specific ID 也能在同一 ID 时直接合并。
    for key in ("doi", "arxiv", "s2", "openalex", "pmid"):
        if ia.get(key) and ia.get(key) == ib.get(key):
            return DuplicateDecision(True, key, 1.0)

    ta, tb = normalize_title(a.title), normalize_title(b.title)
    if not ta or not tb:
        return DuplicateDecision(False)

    if ta == tb:
        if _year_compatible(a.year, b.year) and _author_compatible(a, b):
            return DuplicateDecision(True, "exact_title+metadata", 1.0)
        return DuplicateDecision(False)

    # 模糊标题必须同时有年份和作者证据；宁可少合并，也不错误合并。
    if a.year is None or b.year is None or not a.authors or not b.authors:
        return DuplicateDecision(False)
    if not _year_compatible(a.year, b.year) or not _author_compatible(a, b):
        return DuplicateDecision(False)

    score = SequenceMatcher(None, ta, tb).ratio()
    if score >= fuzzy_threshold:
        return DuplicateDecision(True, "fuzzy_title+year+author", score)
    return DuplicateDecision(False, score=score)


def cluster_hits(hits: Iterable[RetrievalHit], *, fuzzy_threshold: float = 0.965) -> list[list[RetrievalHit]]:
    """保守聚类。先用强 ID/标题索引，最后才做小规模 fuzzy fallback。"""

    clusters: list[list[RetrievalHit]] = []
    doi_index: dict[str, int] = {}
    arxiv_index: dict[str, int] = {}
    s2_index: dict[str, int] = {}
    oa_index: dict[str, int] = {}
    pmid_index: dict[str, int] = {}
    title_index: dict[str, list[int]] = {}

    def register(idx: int, hit: RetrievalHit) -> None:
        ids = _identifier_map(hit)
        for key, index in (
            ("doi", doi_index),
            ("arxiv", arxiv_index),
            ("s2", s2_index),
            ("openalex", oa_index),
            ("pmid", pmid_index),
        ):
            if ids.get(key):
                index[ids[key]] = idx
        title = normalize_title(hit.title)
        if title:
            title_index.setdefault(title, []).append(idx)

    for hit in hits:
        ids = _identifier_map(hit)
        candidate_indices: list[int] = []
        for key, index in (
            ("doi", doi_index),
            ("arxiv", arxiv_index),
            ("s2", s2_index),
            ("openalex", oa_index),
            ("pmid", pmid_index),
        ):
            if ids.get(key) in index:
                candidate_indices.append(index[ids[key]])

        title = normalize_title(hit.title)
        candidate_indices.extend(title_index.get(title, []))

        matched_idx: int | None = None
        for idx in dict.fromkeys(candidate_indices):
            if any(duplicate_decision(hit, existing, fuzzy_threshold=fuzzy_threshold).is_duplicate for existing in clusters[idx]):
                matched_idx = idx
                break

        if matched_idx is None and title:
            # fuzzy fallback：只对标题长度接近的 cluster 代表项做比较，避免 O(N^2) 无限制放大。
            for idx, cluster in enumerate(clusters):
                rep = cluster[0]
                rep_title = normalize_title(rep.title)
                if not rep_title or abs(len(rep_title) - len(title)) > max(12, int(0.2 * len(title))):
                    continue
                if duplicate_decision(hit, rep, fuzzy_threshold=fuzzy_threshold).is_duplicate:
                    matched_idx = idx
                    break

        if matched_idx is None:
            matched_idx = len(clusters)
            clusters.append([hit])
        else:
            clusters[matched_idx].append(hit)

        register(matched_idx, hit)

    return clusters


__all__ = [
    "normalize_title",
    "normalize_person",
    "DuplicateDecision",
    "duplicate_decision",
    "cluster_hits",
]
