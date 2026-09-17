"""Local BM25 retrieval over the released PaSa paper corpus.

This first version deliberately has one responsibility: build and query a
persistent sparse keyword index for ``cs_paper_2nd.zip``.  It does not call an
LLM, access the network, generate retrieval routes, or perform dense retrieval.

SQLite FTS5 is used because it provides a disk-backed BM25 implementation in
the Python standard library.  The source ZIP contains more than 500k papers;
keeping a Python postings dictionary in memory would be unnecessarily large.

Typical use::

    corpus = PaSaCorpusBM25()
    if not corpus.is_index_ready():
        corpus.build_index()
    results = corpus.search("large language model document ranking", limit=20)

The initial index build reads every JSON document in the ZIP and may take a
while.  Subsequent searches reuse the SQLite index.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
import zipfile
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[4]
INDEX_SCHEMA_VERSION = "1"

_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[._+\-][A-Za-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
_NEW_ARXIV_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)

# Query boilerplate contributes little retrieval value and can dominate short
# scholarly queries.  Domain terms are intentionally not included here.
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "all",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "find",
        "for",
        "from",
        "give",
        "help",
        "i",
        "in",
        "is",
        "it",
        "list",
        "looking",
        "me",
        "of",
        "on",
        "paper",
        "papers",
        "please",
        "provide",
        "research",
        "search",
        "show",
        "some",
        "that",
        "the",
        "these",
        "those",
        "to",
        "using",
        "what",
        "which",
        "with",
        "work",
        "works",
        "you",
    }
)


class PaSaCorpusError(RuntimeError):
    """Base error for local PaSa corpus operations."""


class PaSaCorpusConfigurationError(PaSaCorpusError):
    """Raised when a required corpus file or SQLite feature is unavailable."""


class PaSaCorpusIndexError(PaSaCorpusError):
    """Raised when the persistent index is missing, stale, or incomplete."""


@dataclass(frozen=True)
class PaSaBM25Result:
    rank: int
    corpus_key: str
    title: str
    abstract: str | None
    arxiv_id: str | None
    year: int | None
    bm25_score: float
    sources: list[str]


@dataclass(frozen=True)
class PaSaIndexBuildStats:
    index_path: str
    indexed_documents: int
    skipped_directories: int
    invalid_json_documents: int
    invalid_record_documents: int
    id_map_entries: int
    mapped_arxiv_ids: int
    elapsed_seconds: float
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaSaCorpusBM25:
    """Persistent BM25 index for the frozen PaSa corpus.

    The index contains title and abstract only.  Title receives a larger BM25
    weight during search.  Full paper sections remain in the source ZIP and can
    be added by a later retrieval version without changing this baseline.
    """

    def __init__(
        self,
        *,
        paper_zip_path: str | Path | None = None,
        id_map_path: str | Path | None = None,
        index_path: str | Path | None = None,
        title_weight: float = 3.0,
        abstract_weight: float = 1.0,
    ) -> None:
        self.paper_zip_path = self._resolve_path(
            paper_zip_path,
            "PASA_PAPER_ZIP_PATH",
            PROJECT_ROOT / "data" / "paper_database" / "cs_paper_2nd.zip",
        )
        self.id_map_path = self._resolve_path(
            id_map_path,
            "PASA_ID2PAPER_PATH",
            PROJECT_ROOT / "data" / "paper_database" / "id2paper.json",
        )
        self.index_path = self._resolve_path(
            index_path,
            "PASA_BM25_INDEX_PATH",
            PROJECT_ROOT / "models" / "pasa_bm25" / "pasa_corpus.sqlite3",
        )
        if title_weight <= 0 or abstract_weight <= 0:
            raise ValueError("BM25 title and abstract weights must be positive")
        self.title_weight = float(title_weight)
        self.abstract_weight = float(abstract_weight)

    @staticmethod
    def _resolve_path(
        configured: str | Path | None,
        env_name: str,
        default: Path,
    ) -> Path:
        if configured is not None:
            return Path(configured).expanduser().resolve()
        environment_value = os.getenv(env_name, "").strip()
        if environment_value:
            return Path(environment_value).expanduser().resolve()
        return default.resolve()

    def is_index_ready(self, *, require_current_sources: bool = True) -> bool:
        if not self.index_path.is_file():
            return False
        try:
            with closing(self._connect(read_only=True)) as connection:
                metadata = self._read_metadata(connection)
                if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
                    return False
                if metadata.get("complete") != "1":
                    return False
                if not require_current_sources:
                    return True
                return metadata.get("source_signature") == self._source_signature()
        except (OSError, sqlite3.Error, PaSaCorpusError):
            return False

    def index_metadata(self) -> dict[str, str]:
        self._require_ready()
        with closing(self._connect(read_only=True)) as connection:
            return self._read_metadata(connection)

    def build_index(
        self,
        *,
        force: bool = False,
        batch_size: int = 2_000,
        progress_every: int = 10_000,
        progress: Callable[[str], None] | None = print,
        max_documents: int | None = None,
    ) -> PaSaIndexBuildStats:
        """Build the persistent title+abstract BM25 index.

        ``max_documents`` exists for smoke tests only.  Such an index is marked
        incomplete and will not pass :meth:`is_index_ready`.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if progress_every <= 0:
            raise ValueError("progress_every must be positive")
        if max_documents is not None and max_documents <= 0:
            raise ValueError("max_documents must be positive when provided")
        self._require_source_files()
        self._require_fts5()

        if self.is_index_ready() and not force and max_documents is None:
            metadata = self.index_metadata()
            return PaSaIndexBuildStats(
                index_path=str(self.index_path),
                indexed_documents=int(metadata.get("indexed_documents", "0")),
                skipped_directories=0,
                invalid_json_documents=0,
                invalid_record_documents=0,
                id_map_entries=int(metadata.get("id_map_entries", "0")),
                mapped_arxiv_ids=int(metadata.get("mapped_arxiv_ids", "0")),
                elapsed_seconds=0.0,
                complete=True,
            )

        started = time.perf_counter()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_name(f"{self.index_path.name}.building")
        if temporary_path.exists():
            temporary_path.unlink()

        if progress:
            progress(f"Loading PaSa ID mapping: {self.id_map_path}")
        title_to_arxiv, id_map_entries = self._load_title_to_arxiv()

        indexed = 0
        skipped_directories = 0
        invalid_json = 0
        invalid_record = 0
        mapped_arxiv_ids = 0

        connection = sqlite3.connect(temporary_path)
        try:
            self._configure_build_connection(connection)
            self._create_schema(connection)
            if progress:
                progress(f"Building BM25 index from: {self.paper_zip_path}")

            batch: list[tuple[str, str | None, str, str, int | None, str]] = []
            with zipfile.ZipFile(self.paper_zip_path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        skipped_directories += 1
                        continue
                    if max_documents is not None and indexed >= max_documents:
                        break
                    try:
                        raw = json.loads(archive.read(info).decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                        invalid_json += 1
                        continue
                    if not isinstance(raw, dict):
                        invalid_record += 1
                        continue

                    title = self._clean_text(raw.get("title"))
                    if not title:
                        invalid_record += 1
                        continue
                    abstract = self._clean_text(raw.get("abstract")) or ""
                    corpus_key = info.filename.rstrip("/")
                    arxiv_id = title_to_arxiv.get(normalize_title(title))
                    if arxiv_id:
                        mapped_arxiv_ids += 1
                    year = infer_arxiv_year(arxiv_id)
                    raw_sources = raw.get("source") or []
                    if isinstance(raw_sources, str):
                        raw_sources = [raw_sources]
                    sources = [str(value) for value in raw_sources if value]
                    batch.append(
                        (
                            corpus_key,
                            arxiv_id,
                            title,
                            abstract,
                            year,
                            json.dumps(sources, ensure_ascii=False),
                        )
                    )
                    indexed += 1

                    if len(batch) >= batch_size:
                        self._insert_batch(connection, batch)
                        batch.clear()
                    if progress and indexed % progress_every == 0:
                        progress(f"  indexed {indexed:,} papers")

            if batch:
                self._insert_batch(connection, batch)
            connection.commit()

            complete = max_documents is None
            metadata = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "complete": "1" if complete else "0",
                "built_at": datetime.now().astimezone().isoformat(),
                "source_signature": self._source_signature(),
                "paper_zip_path": str(self.paper_zip_path),
                "id_map_path": str(self.id_map_path),
                "indexed_documents": str(indexed),
                "id_map_entries": str(id_map_entries),
                "mapped_arxiv_ids": str(mapped_arxiv_ids),
                "title_weight": str(self.title_weight),
                "abstract_weight": str(self.abstract_weight),
            }
            connection.executemany(
                "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            connection.commit()
            connection.execute("PRAGMA optimize")
        except Exception:
            connection.close()
            if temporary_path.exists():
                temporary_path.unlink()
            raise
        else:
            connection.close()

        os.replace(temporary_path, self.index_path)
        elapsed = time.perf_counter() - started
        stats = PaSaIndexBuildStats(
            index_path=str(self.index_path),
            indexed_documents=indexed,
            skipped_directories=skipped_directories,
            invalid_json_documents=invalid_json,
            invalid_record_documents=invalid_record,
            id_map_entries=id_map_entries,
            mapped_arxiv_ids=mapped_arxiv_ids,
            elapsed_seconds=elapsed,
            complete=max_documents is None,
        )
        if progress:
            progress(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
        return stats

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        operator: Literal["or", "and"] = "or",
    ) -> list[PaSaBM25Result]:
        """Search title and abstract with SQLite FTS5 BM25."""

        if limit <= 0:
            return []
        self._require_ready()
        fts_query = compile_fts_query(query, operator=operator)
        if not fts_query:
            return []

        sql = """
            SELECT
                p.corpus_key,
                p.title,
                p.abstract,
                p.arxiv_id,
                p.year,
                p.sources_json,
                bm25(papers_fts, ?, ?) AS bm25_rank
            FROM papers_fts
            JOIN papers AS p ON p.rowid = papers_fts.rowid
            WHERE papers_fts MATCH ?
            ORDER BY bm25_rank ASC, p.rowid ASC
            LIMIT ?
        """
        with closing(self._connect(read_only=True)) as connection:
            rows = connection.execute(
                sql,
                (self.title_weight, self.abstract_weight, fts_query, int(limit)),
            ).fetchall()
        return [self._row_to_result(row, rank) for rank, row in enumerate(rows, start=1)]

    def get_by_arxiv_id(self, arxiv_id: str) -> PaSaBM25Result | None:
        self._require_ready()
        normalized = normalize_arxiv_id(arxiv_id)
        if not normalized:
            return None
        with closing(self._connect(read_only=True)) as connection:
            row = connection.execute(
                """
                SELECT corpus_key, title, abstract, arxiv_id, year, sources_json, 0.0
                FROM papers
                WHERE arxiv_id = ?
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()
        return self._row_to_result(row, 1) if row else None

    def fetch_by_arxiv_ids(self, arxiv_ids: Iterable[str]) -> list[PaSaBM25Result]:
        self._require_ready()
        normalized = [
            value
            for raw in dict.fromkeys(arxiv_ids)
            if (value := normalize_arxiv_id(raw))
        ]
        if not normalized:
            return []
        output: list[PaSaBM25Result] = []
        with closing(self._connect(read_only=True)) as connection:
            for start in range(0, len(normalized), 500):
                batch = normalized[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""
                    SELECT corpus_key, title, abstract, arxiv_id, year,
                           sources_json, 0.0
                    FROM papers
                    WHERE arxiv_id IN ({placeholders})
                    """,
                    batch,
                ).fetchall()
                by_id = {str(row[3]): row for row in rows if row[3]}
                for arxiv_id in batch:
                    row = by_id.get(arxiv_id)
                    if row is not None:
                        output.append(self._row_to_result(row, len(output) + 1))
        return output

    def _require_source_files(self) -> None:
        missing = [
            str(path)
            for path in (self.paper_zip_path, self.id_map_path)
            if not path.is_file()
        ]
        if missing:
            raise PaSaCorpusConfigurationError(
                "Required PaSa corpus files are missing: " + ", ".join(missing)
            )

    def _require_ready(self) -> None:
        if not self.index_path.is_file():
            raise PaSaCorpusIndexError(
                f"BM25 index does not exist: {self.index_path}. Call build_index() first."
            )
        if not self.is_index_ready():
            raise PaSaCorpusIndexError(
                "BM25 index is incomplete, stale, or uses another schema. "
                "Rebuild it with build_index(force=True)."
            )

    @staticmethod
    def _require_fts5() -> None:
        with closing(sqlite3.connect(":memory:")) as connection:
            try:
                connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(text)")
            except sqlite3.OperationalError as exc:
                raise PaSaCorpusConfigurationError(
                    "This Python SQLite build does not provide FTS5/BM25 support."
                ) from exc

    def _connect(self, *, read_only: bool) -> sqlite3.Connection:
        if read_only:
            uri = self.index_path.as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
        else:
            connection = sqlite3.connect(self.index_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _configure_build_connection(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-131072")

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE papers (
                rowid INTEGER PRIMARY KEY,
                corpus_key TEXT NOT NULL UNIQUE,
                arxiv_id TEXT,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                year INTEGER,
                sources_json TEXT NOT NULL
            );
            CREATE INDEX papers_arxiv_id_idx ON papers(arxiv_id);
            CREATE TABLE index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE papers_fts USING fts5(
                title,
                abstract,
                content='papers',
                content_rowid='rowid',
                tokenize='porter unicode61 remove_diacritics 2'
            );
            """
        )

    @staticmethod
    def _insert_batch(
        connection: sqlite3.Connection,
        batch: list[tuple[str, str | None, str, str, int | None, str]],
    ) -> None:
        cursor = connection.cursor()
        for corpus_key, arxiv_id, title, abstract, year, sources_json in batch:
            cursor.execute(
                """
                INSERT INTO papers(
                    corpus_key, arxiv_id, title, abstract, year, sources_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (corpus_key, arxiv_id, title, abstract, year, sources_json),
            )
            rowid = cursor.lastrowid
            cursor.execute(
                "INSERT INTO papers_fts(rowid, title, abstract) VALUES (?, ?, ?)",
                (rowid, title, abstract),
            )
        connection.commit()

    def _load_title_to_arxiv(self) -> tuple[dict[str, str], int]:
        try:
            with self.id_map_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise PaSaCorpusConfigurationError(
                f"Unable to load PaSa ID mapping: {self.id_map_path}"
            ) from exc
        if not isinstance(raw, dict):
            raise PaSaCorpusConfigurationError("PaSa id2paper.json must be a JSON object")

        title_to_arxiv: dict[str, str] = {}
        for raw_id, raw_title in raw.items():
            arxiv_id = normalize_arxiv_id(raw_id)
            title_key = normalize_title(raw_title)
            if arxiv_id and title_key:
                # Keep the first deterministic ID if normalized-title collisions occur.
                title_to_arxiv.setdefault(title_key, arxiv_id)
        return title_to_arxiv, len(raw)

    def _source_signature(self) -> str:
        self._require_source_files()
        zip_stat = self.paper_zip_path.stat()
        id_stat = self.id_map_path.stat()
        return json.dumps(
            {
                "paper_zip_size": zip_stat.st_size,
                "paper_zip_mtime_ns": zip_stat.st_mtime_ns,
                "id_map_size": id_stat.st_size,
                "id_map_mtime_ns": id_stat.st_mtime_ns,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
        try:
            rows = connection.execute("SELECT key, value FROM index_metadata").fetchall()
        except sqlite3.Error as exc:
            raise PaSaCorpusIndexError("Index metadata cannot be read") from exc
        return {str(row[0]): str(row[1]) for row in rows}

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text or None

    @staticmethod
    def _row_to_result(row: sqlite3.Row, rank: int) -> PaSaBM25Result:
        try:
            sources = json.loads(row[5])
        except (TypeError, json.JSONDecodeError):
            sources = []
        if not isinstance(sources, list):
            sources = []
        # SQLite FTS5 bm25() returns smaller (normally negative) values for
        # better matches. Expose a positive, larger-is-better score externally.
        raw_score = float(row[6])
        return PaSaBM25Result(
            rank=rank,
            corpus_key=str(row[0]),
            title=str(row[1]),
            abstract=str(row[2]) or None,
            arxiv_id=str(row[3]) if row[3] else None,
            year=int(row[4]) if row[4] is not None else None,
            bm25_score=-raw_score,
            sources=[str(value) for value in sources],
        )


def normalize_title(value: Any) -> str:
    return "".join(character for character in str(value) if character.isalpha()).lower()


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
    return _ARXIV_VERSION_RE.sub("", text) or None


def infer_arxiv_year(arxiv_id: str | None) -> int | None:
    if not arxiv_id:
        return None
    match = _NEW_ARXIV_ID_RE.match(arxiv_id)
    if not match:
        return None
    year = 2000 + int(match.group(1))
    current_year = datetime.now().year
    return year if 2007 <= year <= current_year else None


def tokenize_query(query: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(query):
        token = match.group(0).casefold()
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def compile_fts_query(
    query: str,
    *,
    operator: Literal["or", "and"] = "or",
) -> str:
    if operator not in {"or", "and"}:
        raise ValueError("operator must be 'or' or 'and'")
    tokens = tokenize_query(query)
    if not tokens:
        return ""
    separator = " OR " if operator == "or" else " AND "
    escaped = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
    return separator.join(escaped)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local BM25 retrieval for the PaSa corpus.")
    parser.add_argument("--paper-zip", type=Path, default=None)
    parser.add_argument("--id-map", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build or reuse the full BM25 index.")
    build_parser.add_argument("--force", action="store_true", help="Rebuild an existing index.")
    build_parser.add_argument("--batch-size", type=int, default=2_000)
    build_parser.add_argument("--progress-every", type=int, default=10_000)

    search_parser = subparsers.add_parser("search", help="Search an existing BM25 index.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.add_argument("--operator", choices=("or", "and"), default="or")
    search_parser.add_argument(
        "--show-abstract",
        action="store_true",
        help="Print an abstract preview for every result.",
    )

    subparsers.add_parser("status", help="Show index readiness and metadata.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    corpus = PaSaCorpusBM25(
        paper_zip_path=args.paper_zip,
        id_map_path=args.id_map,
        index_path=args.index,
    )

    if args.command == "build":
        stats = corpus.build_index(
            force=args.force,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )
        print("\nBM25 index ready." if stats.complete else "\nBM25 index is incomplete.")
        return 0 if stats.complete else 1

    if args.command == "status":
        ready = corpus.is_index_ready()
        print(f"index_path = {corpus.index_path}")
        print(f"ready      = {ready}")
        if ready:
            print(json.dumps(corpus.index_metadata(), ensure_ascii=False, indent=2))
        return 0 if ready else 1

    if args.command == "search":
        results = corpus.search(args.query, limit=args.limit, operator=args.operator)
        print(f"query   = {args.query}")
        print(f"results = {len(results)}")
        for result in results:
            print(
                f"\n[{result.rank}] score={result.bm25_score:.6f} "
                f"arxiv={result.arxiv_id or '-'} year={result.year or '-'}"
            )
            print(result.title)
            if args.show_abstract and result.abstract:
                preview = result.abstract[:500]
                suffix = "..." if len(result.abstract) > len(preview) else ""
                print(f"{preview}{suffix}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


__all__ = [
    "PaSaCorpusBM25",
    "PaSaBM25Result",
    "PaSaIndexBuildStats",
    "PaSaCorpusError",
    "PaSaCorpusConfigurationError",
    "PaSaCorpusIndexError",
    "normalize_title",
    "normalize_arxiv_id",
    "infer_arxiv_year",
    "tokenize_query",
    "compile_fts_query",
]


if __name__ == "__main__":
    raise SystemExit(main())
