"""Configuration for the isolated PaSa historical evaluation lane."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_PASA_CUTOFF_DATE = date(2024, 10, 1)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in {None, ""} else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in {None, ""} else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class HistoricalEvalConfig:
    dataset_path: Path
    query_cache_path: Path
    results_dir: Path
    output_path: Path | None

    start: int
    limit: int
    cutoff_date: date = OFFICIAL_PASA_CUTOFF_DATE
    per_route_limit: int = 100

    use_openalex: bool = True
    use_semantic_scholar: bool = True
    use_arxiv: bool = True

    timeout_seconds: float = 35.0
    route_timeout_seconds: float = 90.0
    max_concurrency: int = 3
    max_attempts: int = 1
    retry_backoff_seconds: float = 1.0

    semantic_scholar_page_size: int = 100
    semantic_scholar_max_pages: int = 5

    @classmethod
    def from_env(cls) -> "HistoricalEvalConfig":
        exact_output = os.getenv("PASA_HISTORICAL_OUTPUT", "").strip()
        config = cls(
            dataset_path=Path(
                os.getenv(
                    "PASA_REALSCHOLARQUERY_PATH",
                    str(PROJECT_ROOT / "data" / "RealScholarQuery" / "test.jsonl"),
                )
            ).expanduser().resolve(),
            query_cache_path=Path(
                os.getenv(
                    "PASA_V2_QUERY_CACHE_PATH",
                    str(
                        PROJECT_ROOT
                        / "results"
                        / "pasa_v2_query_cache"
                        / "realscholarquery_v2_queries.jsonl"
                    ),
                )
            ).expanduser().resolve(),
            results_dir=Path(
                os.getenv(
                    "PASA_HISTORICAL_RESULTS_DIR",
                    str(PROJECT_ROOT / "results" / "pasa_historical"),
                )
            ).expanduser().resolve(),
            output_path=(
                Path(exact_output).expanduser().resolve() if exact_output else None
            ),
            start=_env_int("PASA_HISTORICAL_START", 0),
            limit=_env_int("PASA_HISTORICAL_LIMIT", 50),
            per_route_limit=_env_int("PASA_HISTORICAL_PER_ROUTE_LIMIT", 100),
            use_openalex=_env_bool("PASA_HISTORICAL_USE_OPENALEX", True),
            use_semantic_scholar=_env_bool(
                "PASA_HISTORICAL_USE_SEMANTIC_SCHOLAR", True
            ),
            use_arxiv=_env_bool("PASA_HISTORICAL_USE_ARXIV", True),
            timeout_seconds=_env_float("PASA_HISTORICAL_HTTP_TIMEOUT_SECONDS", 35.0),
            route_timeout_seconds=_env_float(
                "PASA_HISTORICAL_ROUTE_TIMEOUT_SECONDS", 90.0
            ),
            max_concurrency=_env_int("PASA_HISTORICAL_MAX_CONCURRENCY", 3),
            max_attempts=_env_int("PASA_HISTORICAL_MAX_ATTEMPTS", 1),
            retry_backoff_seconds=_env_float(
                "PASA_HISTORICAL_RETRY_BACKOFF_SECONDS", 1.0
            ),
            semantic_scholar_page_size=_env_int(
                "PASA_HISTORICAL_S2_PAGE_SIZE", 100
            ),
            semantic_scholar_max_pages=_env_int(
                "PASA_HISTORICAL_S2_MAX_PAGES", 5
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.cutoff_date != OFFICIAL_PASA_CUTOFF_DATE:
            raise ValueError("PaSa historical cutoff must remain 2024-10-01")
        if self.start < 0 or self.limit <= 0:
            raise ValueError("start must be >= 0 and limit must be > 0")
        if not 1 <= self.per_route_limit <= 100:
            raise ValueError("per_route_limit must be between 1 and 100")
        if not any((self.use_openalex, self.use_semantic_scholar, self.use_arxiv)):
            raise ValueError("At least one historical source must be enabled")
        if self.timeout_seconds <= 0 or self.route_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_concurrency <= 0 or self.max_attempts <= 0:
            raise ValueError("max_concurrency and max_attempts must be positive")
        if not 1 <= self.semantic_scholar_page_size <= 100:
            raise ValueError("Semantic Scholar page size must be between 1 and 100")
        if self.semantic_scholar_max_pages <= 0:
            raise ValueError("Semantic Scholar max pages must be positive")


__all__ = [
    "HistoricalEvalConfig",
    "OFFICIAL_PASA_CUTOFF_DATE",
    "PROJECT_ROOT",
]
