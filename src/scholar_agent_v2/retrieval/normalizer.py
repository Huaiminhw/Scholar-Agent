from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .models import SearchSource, SourcePaperRecord

_DOI_RE = re.compile(r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
_ARXIV_RE = re.compile(r"(?:arxiv:)?((?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)", re.I)


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    match = _DOI_RE.search(value.strip())
    return match.group(1).rstrip(". ").lower() if match else None


def normalize_arxiv_id(value: str | None, *, drop_version: bool = True) -> str | None:
    if not value:
        return None
    text = value.strip()
    if "/abs/" in text:
        text = text.split("/abs/", 1)[1]
    elif "/pdf/" in text:
        text = text.split("/pdf/", 1)[1]
    text = text.removesuffix(".pdf")
    match = _ARXIV_RE.search(text)
    if not match:
        return None
    arxiv_id = match.group(1)
    if drop_version:
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id, flags=re.I)
    return arxiv_id


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _dedup(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


def reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for token, idxs in inverted_index.items():
        for idx in idxs or []:
            positions.append((int(idx), token))
    if not positions:
        return None
    positions.sort(key=lambda x: x[0])
    return " ".join(token for _, token in positions)


def normalize_openalex_work(raw: dict[str, Any], *, keep_raw: bool = False) -> SourcePaperRecord:
    oa_id = _clean_text(raw.get("id"))
    if oa_id and "/" in oa_id:
        oa_short = oa_id.rstrip("/").split("/")[-1]
    else:
        oa_short = oa_id

    authors = [
        a.get("author", {}).get("display_name", "")
        for a in raw.get("authorships", []) or []
        if isinstance(a, dict)
    ]

    location = raw.get("primary_location") or raw.get("best_oa_location") or {}
    source_info = location.get("source") or {}
    venue = source_info.get("display_name")

    ids = raw.get("ids") or {}
    doi = normalize_doi(raw.get("doi") or ids.get("doi"))
    arxiv_id = normalize_arxiv_id(ids.get("arxiv"))

    topics = []
    for topic in raw.get("topics", []) or []:
        if isinstance(topic, dict) and topic.get("display_name"):
            topics.append(topic["display_name"])

    urls = [u for u in [raw.get("id"), location.get("landing_page_url"), location.get("pdf_url")] if u]

    return SourcePaperRecord(
        source=SearchSource.OPENALEX,
        source_id=oa_short or doi or arxiv_id or "unknown",
        title=_clean_text(raw.get("display_name") or raw.get("title")),
        abstract=_clean_text(reconstruct_openalex_abstract(raw.get("abstract_inverted_index"))),
        authors=_dedup(authors),
        year=raw.get("publication_year"),
        venue=_clean_text(venue),
        doi=doi,
        arxiv_id=arxiv_id,
        openalex_id=oa_short,
        citation_count=raw.get("cited_by_count"),
        publication_types=_dedup([raw.get("type", "")]),
        fields_of_study=_dedup(topics),
        language=_clean_text(raw.get("language")),
        urls=_dedup(urls),
        raw=raw if keep_raw else None,
    )


def normalize_s2_paper(raw: dict[str, Any], *, keep_raw: bool = False) -> SourcePaperRecord:
    external = raw.get("externalIds") or {}
    paper_id = _clean_text(raw.get("paperId")) or "unknown"
    authors = [a.get("name", "") for a in raw.get("authors", []) or [] if isinstance(a, dict)]
    publication_types = raw.get("publicationTypes") or []
    if isinstance(publication_types, str):
        publication_types = [publication_types]
    fields = raw.get("fieldsOfStudy") or []
    if isinstance(fields, str):
        fields = [fields]

    urls = [u for u in [raw.get("url"), (raw.get("openAccessPdf") or {}).get("url")] if u]

    return SourcePaperRecord(
        source=SearchSource.SEMANTIC_SCHOLAR,
        source_id=paper_id,
        title=_clean_text(raw.get("title")),
        abstract=_clean_text(raw.get("abstract")),
        authors=_dedup(authors),
        year=raw.get("year"),
        venue=_clean_text(raw.get("venue")),
        doi=normalize_doi(external.get("DOI")),
        arxiv_id=normalize_arxiv_id(external.get("ArXiv")),
        semantic_scholar_id=paper_id if paper_id != "unknown" else None,
        pmid=_clean_text(external.get("PubMed")),
        citation_count=raw.get("citationCount"),
        publication_types=_dedup(publication_types),
        fields_of_study=_dedup(fields),
        urls=_dedup(urls),
        raw=raw if keep_raw else None,
    )


def normalize_arxiv_entry(raw: dict[str, Any], *, keep_raw: bool = False) -> SourcePaperRecord:
    arxiv_id = normalize_arxiv_id(raw.get("id") or raw.get("arxiv_id"))
    published = _clean_text(raw.get("published"))
    year = None
    if published:
        try:
            year = datetime.fromisoformat(published.replace("Z", "+00:00")).year
        except ValueError:
            match = re.match(r"(\d{4})", published)
            if match:
                year = int(match.group(1))

    categories = raw.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]
    authors = raw.get("authors") or []
    if authors and isinstance(authors[0], dict):
        authors = [a.get("name", "") for a in authors]

    urls = raw.get("urls") or []
    if raw.get("id"):
        urls = list(urls) + [raw["id"]]

    return SourcePaperRecord(
        source=SearchSource.ARXIV,
        source_id=arxiv_id or _clean_text(raw.get("id")) or "unknown",
        title=_clean_text(raw.get("title")),
        abstract=_clean_text(raw.get("summary") or raw.get("abstract")),
        authors=_dedup(list(authors)),
        year=year,
        venue=_clean_text(raw.get("journal_ref")) or "arXiv",
        doi=normalize_doi(raw.get("doi")),
        arxiv_id=arxiv_id,
        citation_count=None,
        publication_types=["preprint"],
        fields_of_study=_dedup(categories),
        urls=_dedup(list(urls)),
        raw=raw if keep_raw else None,
    )


__all__ = [
    "normalize_doi",
    "normalize_arxiv_id",
    "reconstruct_openalex_abstract",
    "normalize_openalex_work",
    "normalize_s2_paper",
    "normalize_arxiv_entry",
]
