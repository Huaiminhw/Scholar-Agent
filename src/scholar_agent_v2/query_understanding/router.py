"""Conservative fast-path router for deterministic scholarly queries.

Only queries that can be parsed without semantic judgement bypass the LLM.
Everything else falls back to :class:`QueryAnalyzer`'s normal LLM path.  The
router returns the same ``QueryAnalysisResult`` contract as the LLM path, so
Query Generation and Retrieval do not need a second execution pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from .normalizer import normalize_identifier, normalize_text
from .schema import (
    ConstraintItem,
    ConstraintLogic,
    ConstraintOperator,
    ConstraintSet,
    ConstraintType,
    EntityType,
    ExtractedProperties,
    LogicNode,
    QueryAnalysisResult,
    QueryEntity,
    QueryIntent,
    QueryType,
    RetrievalMode,
    SearchTask,
)


ROUTER_VERSION = "1.0"


class UnderstandingRoute(str, Enum):
    LLM = "llm"
    DETERMINISTIC_IDENTIFIER = "deterministic_identifier"
    DETERMINISTIC_TITLE = "deterministic_title"


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool = True
    enable_identifier_fast_path: bool = True
    enable_exact_title_fast_path: bool = True


@dataclass(frozen=True)
class RouterDecision:
    route: UnderstandingRoute
    reason: str
    result: QueryAnalysisResult | None = None

    @property
    def use_llm(self) -> bool:
        return self.route == UnderstandingRoute.LLM


@dataclass(frozen=True)
class _IdentifierMatch:
    value: str
    source_text: str
    start: int
    end: int
    kind: str


_DOI_SEARCH_RE = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?"
    r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    re.IGNORECASE,
)
_ARXIV_SEARCH_RE = re.compile(
    r"(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:\s*)?"
    r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.\-]+/\d{7})(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)
_PMID_SEARCH_RE = re.compile(
    r"\bPMID\s*(?::|：|=|为|是)\s*\d+\b",
    re.IGNORECASE,
)
_PMCID_SEARCH_RE = re.compile(r"\bPMC\d+\b", re.IGNORECASE)
_OPENALEX_SEARCH_RE = re.compile(
    r"(?:https?://openalex\.org/)?W\d{5,}",
    re.IGNORECASE,
)
_CORPUS_ID_SEARCH_RE = re.compile(
    r"\bCorpusID\s*(?::|：|=|为|是)\s*\d+\b",
    re.IGNORECASE,
)

_PUNCTUATION_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_ENGLISH_LOOKUP_WORD_RE = re.compile(
    r"\b(?:please|find|look|up|search|for|locate|get|give|me|the|a|an|"
    r"paper|article|work|with|whose|identifier|number|id|is)\b",
    re.IGNORECASE,
)
_IDENTIFIER_LABELS = (
    "semantic scholar corpusid",
    "semantic scholar",
    "openalex",
    "corpusid",
    "identifier",
    "arxiv",
    "pmcid",
    "pmid",
    "doi",
)
_CHINESE_LOOKUP_PHRASES = tuple(
    sorted(
        {
            "请帮我查找",
            "请帮我查询",
            "请帮我搜索",
            "帮我查找",
            "帮我查询",
            "帮我搜索",
            "查找",
            "查询",
            "搜索",
            "找到",
            "定位",
            "给我",
            "请",
            "帮我",
            "标识符",
            "编号",
            "这篇",
            "这一篇",
            "一篇",
            "论文",
            "文章",
            "文献",
            "为",
            "是",
            "的",
            "号",
            "找",
        },
        key=len,
        reverse=True,
    )
)

_CHINESE_TITLE_RE = re.compile(
    r"^(?:请|请帮我|帮我)?\s*(?:查找|查询|搜索|找|找到|定位)\s*"
    r"(?:(?:题为|标题为|论文标题为)\s*)?"
    r"[《“「](?P<title>.+?)[》”」]\s*"
    r"(?:(?:这|这一)篇|的)?\s*(?:论文|文章)?\s*[。！？.!?]*$",
    re.IGNORECASE,
)
_ENGLISH_TITLE_RE = re.compile(
    r"^(?:please\s+)?(?:find|look\s+up|locate|search\s+for)\s+"
    r"(?:the\s+)?(?:paper|article)\s+(?:titled|named|called)\s+"
    r"[\"“](?P<title>.+?)[\"”]\s*[.!?]*$",
    re.IGNORECASE,
)
_EXPLICIT_TITLE_RE = re.compile(
    r"^(?:title|标题|论文标题)\s*[:：]\s*(?P<title>.+?)\s*$",
    re.IGNORECASE,
)


class QueryUnderstandingRouter:
    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def route(self, user_query: str) -> RouterDecision:
        query = user_query.strip()
        if not self.config.enabled:
            return self._llm("deterministic router disabled")

        if self.config.enable_identifier_fast_path:
            identifier = self._find_identifier(query)
            if identifier is not None:
                if self._is_simple_identifier_lookup(query, identifier):
                    return RouterDecision(
                        route=UnderstandingRoute.DETERMINISTIC_IDENTIFIER,
                        reason=f"unambiguous {identifier.kind} lookup",
                        result=self._identifier_result(query, identifier),
                    )
                return self._llm(
                    "identifier is present with additional semantic intent or constraints"
                )

        if self.config.enable_exact_title_fast_path:
            title = self._find_exact_title(query)
            if title is not None:
                return RouterDecision(
                    route=UnderstandingRoute.DETERMINISTIC_TITLE,
                    reason="explicit exact-title lookup",
                    result=self._title_result(query, title),
                )

        return self._llm("query requires semantic understanding")

    @staticmethod
    def _llm(reason: str) -> RouterDecision:
        return RouterDecision(route=UnderstandingRoute.LLM, reason=reason)

    @staticmethod
    def _find_identifier(query: str) -> _IdentifierMatch | None:
        patterns = (
            ("doi", _DOI_SEARCH_RE),
            ("pmid", _PMID_SEARCH_RE),
            ("pmcid", _PMCID_SEARCH_RE),
            ("openalex", _OPENALEX_SEARCH_RE),
            ("semantic_scholar_corpus_id", _CORPUS_ID_SEARCH_RE),
            ("arxiv", _ARXIV_SEARCH_RE),
        )
        for kind, pattern in patterns:
            match = pattern.search(query)
            if match is None:
                continue
            source_text = match.group(0).rstrip(".,;。；，")
            end = match.start() + len(source_text)
            value = QueryUnderstandingRouter._canonical_identifier(kind, source_text)
            return _IdentifierMatch(
                value=value,
                source_text=source_text,
                start=match.start(),
                end=end,
                kind=kind,
            )
        return None

    @staticmethod
    def _canonical_identifier(kind: str, source_text: str) -> str:
        if kind in {"doi", "arxiv"}:
            return normalize_identifier(source_text)
        compact = normalize_text(source_text)
        if kind == "pmid":
            return f"PMID:{QueryUnderstandingRouter._required_digits(compact)}"
        if kind == "pmcid":
            return compact.upper()
        if kind == "openalex":
            work_id = re.search(r"W\d+", compact, re.IGNORECASE)
            return f"https://openalex.org/{work_id.group(0).upper()}"  # type: ignore[union-attr]
        if kind == "semantic_scholar_corpus_id":
            return f"CorpusID:{QueryUnderstandingRouter._required_digits(compact)}"
        return compact

    @staticmethod
    def _required_digits(text: str) -> str:
        match = re.search(r"\d+", text)
        if match is None:  # Protected by the identifier regexes above.
            raise ValueError(f"identifier contains no digits: {text!r}")
        return match.group(0)

    @staticmethod
    def _is_simple_identifier_lookup(
        query: str,
        identifier: _IdentifierMatch,
    ) -> bool:
        remaining = query[: identifier.start] + query[identifier.end :]
        remaining = unicodedata.normalize("NFKC", remaining).casefold()
        for label in _IDENTIFIER_LABELS:
            remaining = remaining.replace(label, " ")
        for phrase in _CHINESE_LOOKUP_PHRASES:
            remaining = remaining.replace(phrase, " ")
        remaining = _ENGLISH_LOOKUP_WORD_RE.sub(" ", remaining)
        return not _PUNCTUATION_RE.sub("", remaining)

    @staticmethod
    def _find_exact_title(query: str) -> str | None:
        for pattern in (_CHINESE_TITLE_RE, _ENGLISH_TITLE_RE, _EXPLICIT_TITLE_RE):
            match = pattern.fullmatch(query)
            if match is None:
                continue
            title = normalize_text(match.group("title")).strip("\"'“”《》「」")
            if len(title) >= 4:
                return title
        return None

    @staticmethod
    def _identifier_result(
        query: str,
        identifier: _IdentifierMatch,
    ) -> QueryAnalysisResult:
        constraint = ConstraintItem(
            id="C1",
            type=ConstraintType.IDENTIFIER,
            operator=ConstraintOperator.EQ,
            value=identifier.value,
            source_text=identifier.source_text,
            normalized_text=identifier.value,
        )
        return QueryAnalysisResult(
            original_query=query,
            query_type=QueryType.SPECIFIC_BY_IDENTIFIER,
            intent=QueryIntent(
                task=SearchTask.PAPER_LOOKUP,
                mode=RetrievalMode.SPECIFIC,
                goal=f"Locate the paper identified by {identifier.value}.",
            ),
            content="",
            constraints=ConstraintSet(hard=[constraint]),
            logic=ConstraintLogic(
                hard=LogicNode(operator="LEAF", ref_id=constraint.id)
            ),
            properties=ExtractedProperties(suitable_for_by_citing=False),
            entities=[
                QueryEntity(
                    type=EntityType.OTHER,
                    canonical_name=identifier.value,
                    source_text=identifier.source_text,
                )
            ],
        )

    @staticmethod
    def _title_result(query: str, title: str) -> QueryAnalysisResult:
        return QueryAnalysisResult(
            original_query=query,
            query_type=QueryType.SPECIFIC_BY_TITLE,
            intent=QueryIntent(
                task=SearchTask.PAPER_LOOKUP,
                mode=RetrievalMode.SPECIFIC,
                goal=f'Locate the paper with the exact title "{title}".',
            ),
            content=title,
            properties=ExtractedProperties(
                specific_paper_name=title,
                suitable_for_by_citing=False,
            ),
            entities=[
                QueryEntity(
                    type=EntityType.PAPER,
                    canonical_name=title,
                    source_text=title,
                )
            ],
        )


__all__ = [
    "ROUTER_VERSION",
    "UnderstandingRoute",
    "RouterConfig",
    "RouterDecision",
    "QueryUnderstandingRouter",
]
