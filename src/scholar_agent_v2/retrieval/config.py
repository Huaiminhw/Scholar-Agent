from __future__ import annotations

import os
from enum import Enum

from .models import SearchSource


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


class RetrievalProfile(str, Enum):
    """Selects only the retrieval backend; upper pipeline stages stay shared."""

    FROZEN_LOCAL = "frozen_local"
    COMPETITION_ONLINE = "competition_online"


class QueryExecutionMode(str, Enum):
    """Controls whether understanding/generation are loaded or run live."""

    FROZEN = "frozen"
    LIVE = "live"


def env_bool(name: str, default: bool = False) -> bool:
    """
    严格读取布尔环境变量。

    true:  1 / true / yes / on
    false: 0 / false / no / off

    对 flase 等拼写错误直接报错，避免配置被静默忽略。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False

    raise ValueError(
        f"Invalid boolean value for {name}: {raw!r}. "
        f"Expected one of: {sorted(_TRUE_VALUES | _FALSE_VALUES)}"
    )


def get_retrieval_profile() -> RetrievalProfile:
    raw = os.getenv(
        "RETRIEVAL_PROFILE",
        RetrievalProfile.COMPETITION_ONLINE.value,
    ).strip().lower()
    try:
        return RetrievalProfile(raw)
    except ValueError as exc:
        expected = ", ".join(item.value for item in RetrievalProfile)
        raise ValueError(
            f"Invalid RETRIEVAL_PROFILE={raw!r}. Expected one of: {expected}."
        ) from exc


def get_query_execution_mode() -> QueryExecutionMode:
    raw = os.getenv(
        "QUERY_EXECUTION_MODE",
        QueryExecutionMode.LIVE.value,
    ).strip().lower()
    try:
        return QueryExecutionMode(raw)
    except ValueError as exc:
        expected = ", ".join(item.value for item in QueryExecutionMode)
        raise ValueError(
            f"Invalid QUERY_EXECUTION_MODE={raw!r}. Expected one of: {expected}."
        ) from exc


def get_enabled_sources() -> list[SearchSource]:
    """
    根据 .env 返回当前启用的数据源。

    默认：
    - OpenAlex: True
    - Semantic Scholar: False
    - arXiv: True
    - Google Search: False（需要 API Key 和 Programmable Search Engine ID）
    """
    profile = get_retrieval_profile()
    if profile == RetrievalProfile.FROZEN_LOCAL:
        return [SearchSource.PASA_LOCAL]

    sources: list[SearchSource] = []

    if env_bool("USE_OPENALEX", True):
        sources.append(SearchSource.OPENALEX)

    if env_bool("USE_SEMANTIC_SCHOLAR", False):
        sources.append(SearchSource.SEMANTIC_SCHOLAR)

    if env_bool("USE_ARXIV", True):
        sources.append(SearchSource.ARXIV)

    if env_bool("USE_GOOGLE_SEARCH", False):
        sources.append(SearchSource.GOOGLE_SEARCH)

    if not sources:
        raise RuntimeError(
            "No retrieval source is enabled. "
            "Enable at least one of USE_OPENALEX, "
            "USE_SEMANTIC_SCHOLAR, USE_ARXIV, USE_GOOGLE_SEARCH."
        )

    return sources


__all__ = [
    "RetrievalProfile",
    "QueryExecutionMode",
    "env_bool",
    "get_retrieval_profile",
    "get_query_execution_mode",
    "get_enabled_sources",
]
