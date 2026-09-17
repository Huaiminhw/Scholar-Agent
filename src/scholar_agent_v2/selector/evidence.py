from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from ..candidate_pool.models import CandidatePaper


@dataclass(frozen=True)
class EvidenceVerification:
    verified: bool
    source: Literal["title", "abstract", "mixed"] | None = None
    matched_parts: int = 0
    total_parts: int = 0


def _normalize_with_index(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    decomposed = unicodedata.normalize("NFD", text)
    for index, char in enumerate(decomposed):
        if unicodedata.category(char) == "Mn":
            continue
        if char.isalnum():
            normalized_chars.append(char.casefold())
            index_map.append(index)
    return "".join(normalized_chars), index_map


def _contains_normalized(haystack: str | None, needle: str, *, min_chars: int) -> bool:
    if not haystack:
        return False
    normalized_haystack, _ = _normalize_with_index(haystack)
    normalized_needle, _ = _normalize_with_index(needle)
    if len(normalized_needle) < min_chars:
        return False
    return normalized_needle in normalized_haystack


def _split_evidence(evidence: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\s*(?:\.\.\.|…)+\s*", evidence) if part.strip()]
    return parts or ([evidence.strip()] if evidence.strip() else [])


def verify_evidence(
    candidate: CandidatePaper,
    evidence: str | None,
    *,
    min_normalized_chars: int = 8,
) -> EvidenceVerification:
    if not evidence or not evidence.strip():
        return EvidenceVerification(verified=False)

    parts = _split_evidence(evidence)
    if not parts:
        return EvidenceVerification(verified=False)

    title_hits = 0
    abstract_hits = 0
    matched = 0

    for part in parts:
        in_title = _contains_normalized(candidate.title, part, min_chars=min_normalized_chars)
        in_abstract = _contains_normalized(candidate.abstract, part, min_chars=min_normalized_chars)
        if in_title or in_abstract:
            matched += 1
            if in_title:
                title_hits += 1
            if in_abstract:
                abstract_hits += 1

    if matched != len(parts):
        return EvidenceVerification(
            verified=False,
            matched_parts=matched,
            total_parts=len(parts),
        )

    if title_hits == len(parts) and abstract_hits == 0:
        source: Literal["title", "abstract", "mixed"] = "title"
    elif abstract_hits == len(parts) and title_hits == 0:
        source = "abstract"
    else:
        source = "mixed"

    return EvidenceVerification(
        verified=True,
        source=source,
        matched_parts=matched,
        total_parts=len(parts),
    )


__all__ = ["EvidenceVerification", "verify_evidence"]
