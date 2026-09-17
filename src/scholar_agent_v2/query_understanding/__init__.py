"""查询理解、规范化与业务校验。"""

from .analyzer import AnalyzerConfig, QueryAnalyzer, QueryAnalyzerOutput
from .normalizer import (
    dedupe_strings,
    normalize_identifier,
    normalize_query_analysis,
    normalize_text,
)
from .router import (
    ROUTER_VERSION,
    QueryUnderstandingRouter,
    RouterConfig,
    RouterDecision,
    UnderstandingRoute,
)
from .schema import QueryAnalysisResult
from .validator import ValidationReport, validate_query_analysis

__all__ = [
    "AnalyzerConfig",
    "QueryAnalyzer",
    "QueryAnalyzerOutput",
    "QueryAnalysisResult",
    "ValidationReport",
    "normalize_text",
    "normalize_identifier",
    "dedupe_strings",
    "normalize_query_analysis",
    "validate_query_analysis",
    "ROUTER_VERSION",
    "UnderstandingRoute",
    "RouterConfig",
    "RouterDecision",
    "QueryUnderstandingRouter",
]
