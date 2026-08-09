from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Callable, Iterable

from ..retrieval.models import RetrievalHit, RetrievalOrigin, SearchSource, SourcePaperRecord
from ..retrieval.normalizer import normalize_arxiv_id, normalize_doi

from .dedup import normalize_person, normalize_title
from .models import CandidatePaper, PaperIdentifiers


def _dedup_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = (value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _distinct(values: Iterable[Any], normalizer: Callable[[Any], Any] | None = None) -> list[Any]:
    normalizer = normalizer or (lambda x: x)
    seen = set()
    out = []
    for value in values:
        if value is None or value == "" or value == []:
            continue
        key_obj = normalizer(value)
        try:
            key = json.dumps(key_obj, sort_keys=True, ensure_ascii=False, default=str)
        except TypeError:
            key = str(key_obj)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _choose_title(hits: list[RetrievalHit]) -> str | None:
    titles = [h.title for h in hits if h.title]
    if not titles:
        return None
    grouped: dict[str, list[str]] = {}
    for title in titles:
        grouped.setdefault(normalize_title(title), []).append(title)
    best_group = max(grouped.values(), key=lambda g: (len(g), max(len(x) for x in g)))
    return max(best_group, key=len)


def _choose_abstract(hits: list[RetrievalHit]) -> str | None:
    abstracts = [h.abstract.strip() for h in hits if h.abstract and h.abstract.strip()]
    return max(abstracts, key=len) if abstracts else None


def _choose_authors(hits: list[RetrievalHit]) -> list[str]:
    lists = [h.authors for h in hits if h.authors]
    if not lists:
        return []
    # 优先更完整的作者列表；同长度时优先 S2/OA，再 arXiv。
    source_priority = {
        SearchSource.SEMANTIC_SCHOLAR: 3,
        SearchSource.OPENALEX: 2,
        SearchSource.ARXIV: 1,
    }
    best_hit = max(
        (h for h in hits if h.authors),
        key=lambda h: (len(h.authors), source_priority[h.source]),
    )
    return _dedup_strings(best_hit.authors)


def _choose_mode(values: list[Any]) -> Any | None:
    clean = [v for v in values if v is not None and v != ""]
    if not clean:
        return None
    counts = Counter(clean)
    return max(clean, key=lambda x: (counts[x], clean.count(x)))


def _choose_venue(hits: list[RetrievalHit]) -> str | None:
    venues = [h.venue for h in hits if h.venue and h.venue.casefold() != "arxiv"]
    if not venues:
        venues = [h.venue for h in hits if h.venue]
    if not venues:
        return None
    norm_counts = Counter(v.casefold().strip() for v in venues)
    best_norm = max(norm_counts, key=lambda k: (norm_counts[k], len(k)))
    return max((v for v in venues if v.casefold().strip() == best_norm), key=len)


def _merge_origins(origins: Iterable[RetrievalOrigin]) -> list[RetrievalOrigin]:
    merged: dict[tuple, RetrievalOrigin] = {}
    for origin in origins:
        key = (
            origin.source,
            origin.route_id,
            origin.query_id,
            origin.strategy,
            origin.logic_branch_id,
        )
        if key not in merged:
            merged[key] = origin.model_copy(deep=True)
            continue
        current = merged[key]
        if origin.rank < current.rank:
            current.rank = origin.rank
        if origin.source_score is not None and (
            current.source_score is None or origin.source_score > current.source_score
        ):
            current.source_score = origin.source_score
        current.applied_constraint_ids = sorted(
            set(current.applied_constraint_ids) | set(origin.applied_constraint_ids)
        )
    return sorted(merged.values(), key=lambda o: (o.rank, o.route_id, o.source.value))


def _field_sources(hits: list[RetrievalHit]) -> dict[str, list[str]]:
    fields = [
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "citation_count",
        "publication_types",
        "fields_of_study",
        "language",
        "doi",
        "arxiv_id",
        "semantic_scholar_id",
        "openalex_id",
        "pmid",
    ]
    output: dict[str, list[str]] = {}
    for field in fields:
        sources = []
        for hit in hits:
            value = getattr(hit, field)
            if value not in (None, "", []):
                sources.append(hit.source.value)
        if sources:
            output[field] = _dedup_strings(sources)
    return output


def _conflicts(hits: list[RetrievalHit]) -> dict[str, list[Any]]:
    checks = {
        "title": lambda v: normalize_title(v),
        "year": lambda v: v,
        "venue": lambda v: str(v).casefold().strip(),
        "doi": lambda v: normalize_doi(v),
        "arxiv_id": lambda v: normalize_arxiv_id(v),
    }
    output: dict[str, list[Any]] = {}
    for field, normalizer in checks.items():
        vals = _distinct((getattr(h, field) for h in hits), normalizer)
        if len(vals) > 1:
            output[field] = vals
    return output


def _make_candidate_id(ids: PaperIdentifiers, title: str | None, year: int | None, authors: list[str]) -> str:
    if ids.doi:
        return f"doi:{ids.doi}"
    if ids.arxiv_id:
        return f"arxiv:{ids.arxiv_id}"
    if ids.semantic_scholar_id:
        return f"s2:{ids.semantic_scholar_id}"
    if ids.openalex_id:
        return f"openalex:{ids.openalex_id}"
    seed = "|".join(
        [
            normalize_title(title),
            str(year or ""),
            normalize_person(authors[0]) if authors else "",
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]
    return f"hash:{digest}"


def fuse_hit_cluster(hits: list[RetrievalHit]) -> CandidatePaper:
    if not hits:
        raise ValueError("cannot fuse empty hit cluster")

    title = _choose_title(hits)
    abstract = _choose_abstract(hits)
    authors = _choose_authors(hits)
    year = _choose_mode([h.year for h in hits])
    venue = _choose_venue(hits)

    doi = next((normalize_doi(h.doi) for h in hits if normalize_doi(h.doi)), None)
    arxiv_id = next((normalize_arxiv_id(h.arxiv_id) for h in hits if normalize_arxiv_id(h.arxiv_id)), None)
    s2_id = next((h.semantic_scholar_id for h in hits if h.semantic_scholar_id), None)
    oa_id = next((h.openalex_id for h in hits if h.openalex_id), None)
    pmid = next((h.pmid for h in hits if h.pmid), None)
    identifiers = PaperIdentifiers(
        doi=doi,
        arxiv_id=arxiv_id,
        semantic_scholar_id=s2_id,
        openalex_id=oa_id,
        pmid=pmid,
    )

    origins = _merge_origins(h.origin for h in hits)
    sources = {o.source.value for o in origins}
    routes = {o.route_id for o in origins}

    candidate = CandidatePaper(
        candidate_id=_make_candidate_id(identifiers, title, year, authors),
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        venue=venue,
        publication_types=_dedup_strings(x for h in hits for x in h.publication_types),
        fields_of_study=_dedup_strings(x for h in hits for x in h.fields_of_study),
        language=_choose_mode([h.language for h in hits]),
        citation_count=max((h.citation_count for h in hits if h.citation_count is not None), default=None),
        identifiers=identifiers,
        urls=_dedup_strings(x for h in hits for x in h.urls),
        origins=origins,
        field_sources=_field_sources(hits),
        metadata_conflicts=_conflicts(hits),
        source_count=len(sources),
        route_count=len(routes),
        hit_count=len(hits),
        best_rank=min((o.rank for o in origins), default=None),
    )
    return candidate


def enrich_candidate(candidate: CandidatePaper, record: SourcePaperRecord) -> CandidatePaper:
    """用延迟加载得到的元数据补 Candidate；不改变 candidate_id。"""

    c = candidate.model_copy(deep=True)
    source = record.source.value

    def add_source(field: str):
        c.field_sources[field] = _dedup_strings(c.field_sources.get(field, []) + [source])

    def fill_scalar(field: str, new_value: Any):
        if new_value in (None, ""):
            return
        old = getattr(c, field)
        if old in (None, ""):
            setattr(c, field, new_value)
            c.enriched_fields.add(field)
        elif old != new_value:
            vals = c.metadata_conflicts.setdefault(field, [])
            if old not in vals:
                vals.append(old)
            if new_value not in vals:
                vals.append(new_value)
        add_source(field)

    fill_scalar("title", record.title)
    if record.abstract and (not c.abstract or len(record.abstract) > len(c.abstract)):
        if c.abstract and c.abstract != record.abstract:
            c.metadata_conflicts.setdefault("abstract", [])
        c.abstract = record.abstract
        c.enriched_fields.add("abstract")
        add_source("abstract")
    fill_scalar("year", record.year)
    fill_scalar("venue", record.venue)
    fill_scalar("language", record.language)

    if record.authors:
        if not c.authors or len(record.authors) > len(c.authors):
            c.authors = _dedup_strings(record.authors)
            c.enriched_fields.add("authors")
        add_source("authors")

    if record.citation_count is not None:
        if c.citation_count is None or record.citation_count > c.citation_count:
            c.citation_count = record.citation_count
            c.enriched_fields.add("citation_count")
        add_source("citation_count")

    c.publication_types = _dedup_strings(c.publication_types + record.publication_types)
    c.fields_of_study = _dedup_strings(c.fields_of_study + record.fields_of_study)
    c.urls = _dedup_strings(c.urls + record.urls)

    for field in ["doi", "arxiv_id", "semantic_scholar_id", "openalex_id", "pmid"]:
        new_value = getattr(record, field)
        if new_value and not getattr(c.identifiers, field):
            setattr(c.identifiers, field, new_value)
            c.enriched_fields.add(field)
            add_source(field)

    return c


__all__ = ["fuse_hit_cluster", "enrich_candidate"]
