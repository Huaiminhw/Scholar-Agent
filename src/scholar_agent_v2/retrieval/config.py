from __future__ import annotations

import os

from .models import SearchSource


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


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


def get_enabled_sources() -> list[SearchSource]:
    """
    根据 .env 返回当前启用的数据源。

    默认：
    - OpenAlex: True
    - Semantic Scholar: False
    - arXiv: True
    """
    sources: list[SearchSource] = []

    if env_bool("USE_OPENALEX", True):
        sources.append(SearchSource.OPENALEX)

    if env_bool("USE_SEMANTIC_SCHOLAR", False):
        sources.append(SearchSource.SEMANTIC_SCHOLAR)

    if env_bool("USE_ARXIV", True):
        sources.append(SearchSource.ARXIV)

    if not sources:
        raise RuntimeError(
            "No retrieval source is enabled. "
            "Enable at least one of USE_OPENALEX, "
            "USE_SEMANTIC_SCHOLAR, USE_ARXIV."
        )

    return sources
