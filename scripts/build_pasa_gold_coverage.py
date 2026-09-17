"""Build a PaSa RealScholarQuery Gold-to-corpus coverage report.

The script is intentionally independent from the retrieval pipeline. It only
reads the released PaSa files and answers one question: which Gold papers are
actually present in the local frozen paper corpus?

Default inputs:

    data/RealScholarQuery/test.jsonl
    data/paper_database/id2paper.json
    data/paper_database/cs_paper_2nd.zip

PowerShell example:

    python scripts/build_pasa_gold_coverage.py

Custom paths may be supplied with command-line arguments or the environment
variables PASA_REALSCHOLARQUERY_PATH, PASA_ID2PAPER_PATH,
PASA_PAPER_ZIP_PATH, and PASA_GOLD_COVERAGE_OUTPUT.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_path(env_name: str, relative_path: str) -> Path:
    configured = os.getenv(env_name, "").strip()
    return Path(configured).expanduser() if configured else PROJECT_ROOT / relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how many RealScholarQuery Gold papers exist in the "
            "released PaSa local paper corpus."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_path(
            "PASA_REALSCHOLARQUERY_PATH",
            "data/RealScholarQuery/test.jsonl",
        ),
        help="Path to RealScholarQuery/test.jsonl.",
    )
    parser.add_argument(
        "--id-map",
        type=Path,
        default=default_path(
            "PASA_ID2PAPER_PATH",
            "data/paper_database/id2paper.json",
        ),
        help="Path to PaSa id2paper.json.",
    )
    parser.add_argument(
        "--paper-zip",
        type=Path,
        default=default_path(
            "PASA_PAPER_ZIP_PATH",
            "data/paper_database/cs_paper_2nd.zip",
        ),
        help="Path to PaSa cs_paper_2nd.zip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. By default a timestamped report is written "
            "under results/pasa_gold_coverage/."
        ),
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def normalize_arxiv_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "/abs/" in text:
        text = text.split("/abs/", 1)[1]
    elif "/pdf/" in text:
        text = text.split("/pdf/", 1)[1]
    text = text.removesuffix(".pdf")
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", text, flags=re.IGNORECASE) or None


def normalize_title(value: Any) -> str:
    """Match the title-key convention used by the released PaSa ZIP."""

    return "".join(character for character in str(value) if character.isalpha()).lower()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Dataset line {line_number} is not a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def load_id_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"id2paper must contain a JSON object: {path}")

    normalized: dict[str, str] = {}
    for raw_id, raw_title in raw.items():
        arxiv_id = normalize_arxiv_id(raw_id)
        if arxiv_id and isinstance(raw_title, str) and raw_title.strip():
            normalized[arxiv_id] = raw_title.strip()
    return normalized


def load_zip_keys(path: Path) -> tuple[set[str], int]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        # Reuse ZipInfo.filename objects instead of archive.namelist(), which
        # would allocate another list for this 500k+ entry archive.
        keys = {info.filename.rstrip("/") for info in infos if not info.is_dir()}
        return keys, len(infos)


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def analyze(
    records: list[dict[str, Any]],
    id_to_title: dict[str, str],
    zip_keys: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []

    for query_index, record in enumerate(records):
        titles = record.get("answer")
        arxiv_ids = record.get("answer_arxiv_id")
        if not isinstance(titles, list) or not isinstance(arxiv_ids, list):
            raise ValueError(
                f"Query #{query_index} must contain answer and answer_arxiv_id lists"
            )
        if len(titles) != len(arxiv_ids):
            raise ValueError(
                f"Query #{query_index} has {len(titles)} titles but "
                f"{len(arxiv_ids)} arXiv IDs"
            )

        qid = str(record.get("qid", f"RealScholarQuery_{query_index}"))
        query_detail_start = len(details)
        for gold_index, (raw_title, raw_arxiv_id) in enumerate(zip(titles, arxiv_ids)):
            gold_title = str(raw_title).strip()
            arxiv_id = normalize_arxiv_id(raw_arxiv_id)
            mapped_title = id_to_title.get(arxiv_id or "")
            gold_zip_key = normalize_title(gold_title)
            mapped_zip_key = normalize_title(mapped_title) if mapped_title else None

            id_in_map = bool(arxiv_id and arxiv_id in id_to_title)
            content_via_mapped_title = bool(mapped_zip_key and mapped_zip_key in zip_keys)
            content_via_gold_title = bool(gold_zip_key and gold_zip_key in zip_keys)
            corpus_available = content_via_mapped_title or content_via_gold_title
            match_method = (
                "id_map_title"
                if content_via_mapped_title
                else "gold_title_fallback"
                if content_via_gold_title
                else None
            )

            details.append(
                {
                    "query_index": query_index,
                    "qid": qid,
                    "gold_index": gold_index,
                    "question": str(record.get("question", "")),
                    "source_published_time": str(
                        (record.get("source_meta") or {}).get("published_time", "")
                    )
                    or None,
                    "gold_title": gold_title,
                    "arxiv_id": arxiv_id,
                    "id_in_id2paper": id_in_map,
                    "mapped_title": mapped_title,
                    "mapped_title_differs": bool(
                        mapped_title
                        and normalize_title(mapped_title) != normalize_title(gold_title)
                    ),
                    "content_via_mapped_title": content_via_mapped_title,
                    "content_via_gold_title": content_via_gold_title,
                    "corpus_available": corpus_available,
                    "match_method": match_method,
                    "matched_zip_key": (
                        mapped_zip_key
                        if content_via_mapped_title
                        else gold_zip_key
                        if content_via_gold_title
                        else None
                    ),
                }
            )

        query_details = details[query_detail_start:]
        gold_count = len(query_details)
        available = sum(item["corpus_available"] for item in query_details)
        per_query.append(
            {
                "query_index": query_index,
                "qid": qid,
                "question": str(record.get("question", "")),
                "source_published_time": str(
                    (record.get("source_meta") or {}).get("published_time", "")
                )
                or None,
                "gold_count": gold_count,
                "corpus_available": available,
                "corpus_missing": gold_count - available,
                "corpus_coverage": rate(available, gold_count),
            }
        )

    total = len(details)
    id_covered = sum(item["id_in_id2paper"] for item in details)
    mapped_content = sum(item["content_via_mapped_title"] for item in details)
    gold_title_content = sum(item["content_via_gold_title"] for item in details)
    available = sum(item["corpus_available"] for item in details)
    methods = Counter(item["match_method"] or "missing" for item in details)
    published_times = Counter(item["source_published_time"] or "unknown" for item in details)

    summary = {
        "query_count": len(records),
        "gold_total": total,
        "id_in_id2paper": {
            "count": id_covered,
            "rate": rate(id_covered, total),
        },
        "content_via_id_map_title": {
            "count": mapped_content,
            "rate": rate(mapped_content, total),
        },
        "content_via_gold_title": {
            "count": gold_title_content,
            "rate": rate(gold_title_content, total),
        },
        "corpus_available_union": {
            "count": available,
            "rate": rate(available, total),
        },
        "corpus_missing": {
            "count": total - available,
            "rate": rate(total - available, total),
        },
        "theoretical_max_end_to_end_recall": rate(available, total),
        "match_methods": dict(methods),
        "source_published_times": dict(published_times),
        "per_query": per_query,
        "notes": [
            "corpus_available is true when the normalized id2paper title or Gold title exists in the ZIP.",
            "ZIP entries are keyed by normalized title; a filename match is strong coverage evidence but not a cryptographic identity proof.",
            "Corpus-restricted recall should use only corpus_available Gold papers as its denominator.",
            "End-to-end recall should continue using all Gold papers and is bounded by corpus coverage.",
        ],
    }
    return summary, details


def write_missing_csv(path: Path, details: list[dict[str, Any]]) -> None:
    missing = [item for item in details if not item["corpus_available"]]
    fieldnames = [
        "query_index",
        "qid",
        "gold_index",
        "arxiv_id",
        "gold_title",
        "id_in_id2paper",
        "mapped_title",
        "source_published_time",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(missing)


def main() -> int:
    args = parse_args()
    dataset_path = require_file(args.dataset, "RealScholarQuery dataset")
    id_map_path = require_file(args.id_map, "PaSa id2paper mapping")
    paper_zip_path = require_file(args.paper_zip, "PaSa paper corpus ZIP")

    print("=" * 84)
    print("PaSa RealScholarQuery Gold Coverage")
    print("=" * 84)
    print(f"dataset   = {dataset_path}")
    print(f"id map    = {id_map_path}")
    print(f"paper ZIP = {paper_zip_path}")

    print("\n[1/3] Loading RealScholarQuery...")
    records = load_jsonl(dataset_path)
    print(f"      queries = {len(records)}")

    print("[2/3] Loading id2paper mapping...")
    id_to_title = load_id_map(id_map_path)
    print(f"      normalized IDs = {len(id_to_title)}")

    print("[3/3] Reading ZIP central directory (paper bodies are not extracted)...")
    zip_keys, zip_entries = load_zip_keys(paper_zip_path)
    print(f"      ZIP entries = {zip_entries}; unique file keys = {len(zip_keys)}")

    summary, details = analyze(records, id_to_title, zip_keys)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configured_output = os.getenv("PASA_GOLD_COVERAGE_OUTPUT", "").strip()
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else Path(configured_output).expanduser()
        if configured_output
        else PROJECT_ROOT
        / "results"
        / "pasa_gold_coverage"
        / f"gold_coverage_{stamp}.json"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_csv_path = output_path.with_name(f"{output_path.stem}_missing.csv")

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "inputs": {
            "dataset": str(dataset_path),
            "dataset_size_bytes": dataset_path.stat().st_size,
            "id_map": str(id_map_path),
            "id_map_size_bytes": id_map_path.stat().st_size,
            "paper_zip": str(paper_zip_path),
            "paper_zip_size_bytes": paper_zip_path.stat().st_size,
            "id_map_entries": len(id_to_title),
            "zip_entries": zip_entries,
            "unique_zip_file_keys": len(zip_keys),
        },
        "summary": summary,
        "gold_details": details,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_missing_csv(missing_csv_path, details)

    total = summary["gold_total"]
    id_stats = summary["id_in_id2paper"]
    mapped_stats = summary["content_via_id_map_title"]
    available_stats = summary["corpus_available_union"]
    missing_stats = summary["corpus_missing"]
    print("\n" + "=" * 84)
    print("COVERAGE SUMMARY")
    print("=" * 84)
    print(f"Gold papers                 = {total}")
    print(f"ID present in id2paper      = {id_stats['count']} ({id_stats['rate']:.2%})")
    print(
        "Paper content via ID title  = "
        f"{mapped_stats['count']} ({mapped_stats['rate']:.2%})"
    )
    print(
        "Corpus available (union)    = "
        f"{available_stats['count']} ({available_stats['rate']:.2%})"
    )
    print(
        "Corpus missing              = "
        f"{missing_stats['count']} ({missing_stats['rate']:.2%})"
    )
    print(
        "Theoretical max E2E recall  = "
        f"{summary['theoretical_max_end_to_end_recall']:.2%}"
    )
    print(f"\nJSON report = {output_path}")
    print(f"Missing CSV = {missing_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
