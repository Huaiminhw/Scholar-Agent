"""Build a frozen Scholar-Agent-V2 query cache for PaSa RealScholarQuery.

This script has one job::

    RealScholarQuery/test.jsonl
        -> existing QueryAnalyzer
        -> existing query-understanding validator
        -> QueryAnalysisResult (AcademicQueryIR)
        -> existing QueryGenerator
        -> frozen search queries
        -> JSONL

It intentionally performs no retrieval and reads no Gold answers.  Successful
records are immutable during normal resume runs, so an LLM cannot silently
replace yesterday's Query A with today's Query B.  Use ``--force`` only when a
complete, intentional cache rebuild is required.

PowerShell example::

    $env:PASA_V2_QUERY_CACHE_START="0"
    $env:PASA_V2_QUERY_CACHE_LIMIT="50"
    python scripts/build_pasa_v2_query_cache.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from openai import AsyncOpenAI  # noqa: E402

from scholar_agent_v2.query_generation.generator import (  # noqa: E402
    GeneratorConfig,
    QueryGenerator,
)
from scholar_agent_v2.query_understanding.analyzer import (  # noqa: E402
    AnalyzerConfig,
    LLMResponse,
    LLMUsage,
    QueryAnalyzer,
)
from scholar_agent_v2.query_understanding.router import ROUTER_VERSION  # noqa: E402
from scholar_agent_v2.query_understanding.validator import (  # noqa: E402
    ValidationReport,
    validate_query_analysis,
)


CACHE_SCHEMA_VERSION = "1.0"
PROTOCOL_NAME = "pasa_frozen_v2_query_cache"

DEFAULT_DATASET = PROJECT_ROOT / "data" / "RealScholarQuery" / "test.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "pasa_v2_query_cache"
    / "realscholarquery_v2_queries.jsonl"
)


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


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing {name}. Configure it in {PROJECT_ROOT / '.env'}."
        )
    return value


def _normalize_base_url(value: str) -> str:
    value = value.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    return value.rstrip("/")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _utc_now() -> str:
    return datetime.now().astimezone().isoformat()


@dataclass(frozen=True)
class CacheConfig:
    dataset_path: Path
    output_path: Path
    start: int
    limit: int
    sleep_seconds: float
    timeout_seconds: float
    force: bool
    accept_degraded: bool


class OpenAICompatibleLLMClient:
    """Minimal adapter shared by the existing analyzer and generator."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self.model = _required_env("LLM_MODEL")
        self._client = AsyncOpenAI(
            api_key=_required_env("LLM_API_KEY"),
            base_url=_normalize_base_url(_required_env("LLM_BASE_URL")),
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        # The current OpenAI-compatible Qwen endpoint is used in ordinary JSON
        # prompt mode; response_schema is intentionally not forced on it.
        del response_schema
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
            temperature=temperature,
        )
        if not response.choices:
            raise RuntimeError("LLM response contains no choices")
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise RuntimeError("LLM response content is empty")

        usage = response.usage
        return LLMResponse(
            content=content,
            usage=LLMUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )
            if usage is not None
            else LLMUsage(),
            model=getattr(response, "model", self.model),
            request_id=getattr(response, "id", None),
        )

    async def close(self) -> None:
        await self._client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Scholar-Agent-V2 Query Understanding and Query Generation "
            "outputs for PaSa RealScholarQuery."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.getenv("PASA_REALSCHOLARQUERY_PATH", str(DEFAULT_DATASET))),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("PASA_V2_QUERY_CACHE_PATH", str(DEFAULT_OUTPUT))),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=_env_int("PASA_V2_QUERY_CACHE_START", 0),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_env_int("PASA_V2_QUERY_CACHE_LIMIT", 50),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=_env_float("PASA_V2_QUERY_CACHE_SLEEP_SECONDS", 0.5),
        help="Delay between records to reduce provider rate-limit pressure.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_env_float("PASA_V2_QUERY_CACHE_TIMEOUT_SECONDS", 120.0),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=_env_bool("PASA_V2_QUERY_CACHE_FORCE", False),
        help="Discard the existing cache and rebuild the selected range.",
    )
    parser.add_argument(
        "--accept-degraded",
        action="store_true",
        default=_env_bool("PASA_V2_QUERY_CACHE_ACCEPT_DEGRADED", False),
        help="Cache local fallback queries when LLM expansion fails.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CacheConfig:
    config = CacheConfig(
        dataset_path=args.dataset.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        start=args.start,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        force=args.force,
        accept_degraded=args.accept_degraded,
    )
    if config.start < 0:
        raise ValueError("--start must be >= 0")
    if config.limit <= 0:
        raise ValueError("--limit must be > 0")
    if config.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be >= 0")
    if config.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if not config.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {config.dataset_path}")
    return config


def load_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid dataset JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record.get("question"), str) or not record["question"].strip():
                raise ValueError(f"Dataset line {line_number} has no non-empty question")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def load_existing_cache(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid cache JSONL at line {line_number}: {exc}. "
                    "Do not continue with a corrupted cache."
                ) from exc
            index = record.get("dataset_index")
            if not isinstance(index, int):
                raise ValueError(f"Cache line {line_number} has no integer dataset_index")
            if index in records:
                raise ValueError(f"Duplicate dataset_index={index} in cache")
            records[index] = record
    return records


def atomic_write_jsonl(path: Path, records: Mapping[int, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in sorted(records):
            handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def validation_report_json(report: ValidationReport) -> dict[str, Any]:
    return {
        "is_valid": report.is_valid,
        "summary": report.summary(),
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "severity": issue.severity.value,
                "path": issue.path,
                "refs": list(issue.refs),
            }
            for issue in report.issues
        ],
    }


def cache_protocol(model: str, *, accept_degraded: bool) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": model,
        "temperature": 0.0,
        "analyzer": {
            "deterministic_router": {
                "version": ROUTER_VERSION,
                "enabled": True,
                "identifier_fast_path": True,
                "exact_title_fast_path": True,
            },
            "use_structured_output": False,
            "prompt_schema_mode": "summary",
            "include_few_shot_examples": False,
            "enable_repair": True,
            "repair_schema_mode": "full",
            "max_transport_retries": 0,
        },
        "generator": {
            "use_structured_output": False,
            "enable_llm_expansion": True,
            "max_transport_retries": 0,
            "max_total_queries": 5,
            "near_duplicate_threshold": 0.88,
            "accept_degraded": accept_degraded,
        },
    }


def validate_existing_cache(
    records: Mapping[int, Mapping[str, Any]],
    *,
    dataset_sha256: str,
    protocol_sha256: str,
) -> None:
    for index, record in records.items():
        if record.get("dataset_sha256") != dataset_sha256:
            raise RuntimeError(
                f"Existing cache record {index} belongs to a different dataset. "
                "Use another --output path or intentionally rebuild with --force."
            )
        if record.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(
                f"Existing cache record {index} uses another model/configuration. "
                "Mixing configurations would break Frozen-1. Use another --output "
                "path or intentionally rebuild with --force."
            )


def build_analyzer(client: OpenAICompatibleLLMClient) -> QueryAnalyzer:
    return QueryAnalyzer(
        client,
        config=AnalyzerConfig(
            temperature=0.0,
            use_structured_output=False,
            prompt_schema_mode="summary",
            include_few_shot_examples=False,
            enable_repair=True,
            repair_schema_mode="full",
            max_transport_retries=0,
            retain_raw_output=False,
        ),
    )


def build_generator(client: OpenAICompatibleLLMClient) -> QueryGenerator:
    return QueryGenerator(
        client=client,
        config=GeneratorConfig(
            temperature=0.0,
            use_structured_output=False,
            enable_llm_expansion=True,
            max_transport_retries=0,
            max_total_queries=5,
            near_duplicate_threshold=0.88,
        ),
    )


async def build_one_record(
    *,
    index: int,
    source_record: Mapping[str, Any],
    analyzer: QueryAnalyzer,
    generator: QueryGenerator,
    dataset_sha256: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    accept_degraded: bool,
) -> dict[str, Any]:
    question = str(source_record["question"]).strip()
    started = time.perf_counter()

    analysis_output = await analyzer.analyze(question)
    academic_query_ir = analysis_output.result

    # QueryAnalyzer already performs this validation internally.  Re-run the
    # public validator here so the cache boundary explicitly guarantees that
    # every serialized AcademicQueryIR is valid under the current V2 rules.
    analysis_validation = validate_query_analysis(
        academic_query_ir,
        original_query=question,
    )
    if not analysis_validation.is_valid:
        raise RuntimeError(
            "AcademicQueryIR failed explicit validation:\n"
            + analysis_validation.summary()
        )

    generation_output = await generator.generate(academic_query_ir)
    if generation_output.result.degraded and not accept_degraded:
        raise RuntimeError(
            "Query Generation degraded to local fallback. The record was not "
            "frozen; retry later or pass --accept-degraded intentionally."
        )
    if not generation_output.result.queries:
        raise RuntimeError("Query Generation produced no search queries")

    search_queries = [
        query.model_dump(mode="json")
        for query in generation_output.result.queries
    ]
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "status": "success",
        "dataset_index": index,
        "qid": str(source_record.get("qid", f"RealScholarQuery_{index}")),
        "question": question,
        "question_sha256": _sha256_bytes(question.encode("utf-8")),
        "source_published_time": str(
            (source_record.get("source_meta") or {}).get("published_time", "")
        )
        or None,
        "dataset_sha256": dataset_sha256,
        "protocol_sha256": protocol_sha256,
        "protocol": dict(protocol),
        "created_at": _utc_now(),
        "academic_query_ir": academic_query_ir.model_dump(mode="json"),
        "query_understanding": {
            "route": analysis_output.understanding_route,
            "route_reason": analysis_output.route_reason,
            "validation": validation_report_json(analysis_validation),
            "metrics": asdict(analysis_output.metrics),
            "repaired": analysis_output.repaired,
            "repair_trigger": analysis_output.repair_trigger,
        },
        "query_generation": generation_output.model_dump(mode="json"),
        "search_queries": search_queries,
        "query_texts": [query["text"] for query in search_queries],
        "total_wall_ms": (time.perf_counter() - started) * 1000.0,
    }


def error_record(
    *,
    index: int,
    source_record: Mapping[str, Any],
    dataset_sha256: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    exc: Exception,
) -> dict[str, Any]:
    question = str(source_record["question"]).strip()
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "status": "error",
        "dataset_index": index,
        "qid": str(source_record.get("qid", f"RealScholarQuery_{index}")),
        "question": question,
        "question_sha256": _sha256_bytes(question.encode("utf-8")),
        "dataset_sha256": dataset_sha256,
        "protocol_sha256": protocol_sha256,
        "protocol": dict(protocol),
        "created_at": _utc_now(),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


async def async_main() -> int:
    args = parse_args()
    config = build_config(args)
    dataset_bytes = config.dataset_path.read_bytes()
    dataset_sha256 = _sha256_bytes(dataset_bytes)
    records = load_dataset(config.dataset_path)
    end = min(len(records), config.start + config.limit)
    selected_indices = list(range(config.start, end))
    if not selected_indices:
        raise ValueError(
            f"No records selected: start={config.start}, limit={config.limit}, "
            f"dataset_size={len(records)}"
        )

    client = OpenAICompatibleLLMClient(timeout_seconds=config.timeout_seconds)
    protocol = cache_protocol(client.model, accept_degraded=config.accept_degraded)
    protocol_sha256 = _stable_hash(protocol)

    cached = {} if config.force else load_existing_cache(config.output_path)
    if cached:
        validate_existing_cache(
            cached,
            dataset_sha256=dataset_sha256,
            protocol_sha256=protocol_sha256,
        )

    analyzer = build_analyzer(client)
    generator = build_generator(client)
    completed = 0
    skipped = 0
    failed = 0

    print("=" * 88)
    print("PaSa Frozen-1 V2 Query Cache Builder")
    print("=" * 88)
    print(f"dataset          = {config.dataset_path}")
    print(f"dataset_sha256   = {dataset_sha256}")
    print(f"output           = {config.output_path}")
    print(f"range            = [{config.start}, {end})")
    print(f"model            = {client.model}")
    print(f"protocol_sha256  = {protocol_sha256}")
    print(f"resume           = {not config.force}")

    try:
        for position, index in enumerate(selected_indices):
            existing = cached.get(index)
            if existing is not None and existing.get("status") == "success":
                if existing.get("question") != records[index]["question"].strip():
                    raise RuntimeError(
                        f"Cache question mismatch at dataset_index={index}"
                    )
                skipped += 1
                print(f"[{index:02d}] cached success; skipped")
                continue

            print(f"[{index:02d}] building: {records[index]['question']}")
            try:
                cache_record = await build_one_record(
                    index=index,
                    source_record=records[index],
                    analyzer=analyzer,
                    generator=generator,
                    dataset_sha256=dataset_sha256,
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    accept_degraded=config.accept_degraded,
                )
            except Exception as exc:
                failed += 1
                cache_record = error_record(
                    index=index,
                    source_record=records[index],
                    dataset_sha256=dataset_sha256,
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    exc=exc,
                )
                print(f"      ERROR {type(exc).__name__}: {exc}")
            else:
                completed += 1
                analysis_metrics = cache_record["query_understanding"]["metrics"]
                generation_metrics = cache_record["query_generation"]["metrics"]
                print(
                    "      saved "
                    f"queries={len(cache_record['search_queries'])} "
                    f"api_calls={analysis_metrics['api_calls'] + generation_metrics['api_calls']} "
                    f"tokens={analysis_metrics['total_tokens'] + generation_metrics['total_tokens']} "
                    f"wall_ms={cache_record['total_wall_ms']:.2f}"
                )

            cached[index] = cache_record
            atomic_write_jsonl(config.output_path, cached)

            if position + 1 < len(selected_indices) and config.sleep_seconds:
                await asyncio.sleep(config.sleep_seconds)
    finally:
        await client.close()

    success_total = sum(item.get("status") == "success" for item in cached.values())
    error_total = sum(item.get("status") == "error" for item in cached.values())
    print("\n" + "=" * 88)
    print("CACHE SUMMARY")
    print("=" * 88)
    print(f"new_success      = {completed}")
    print(f"skipped_success  = {skipped}")
    print(f"new_failed       = {failed}")
    print(f"cache_success    = {success_total}")
    print(f"cache_errors     = {error_total}")
    print(f"saved_to         = {config.output_path}")
    return 1 if failed else 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
