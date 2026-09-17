"""Frozen title-resolved citation graph for the released PaSa corpus."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import zipfile
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from .pasa_corpus import PROJECT_ROOT, normalize_arxiv_id, normalize_title


CITATION_GRAPH_SCHEMA_VERSION = "1"


class PaSaCitationGraphError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaSaCitationBuildStats:
    graph_path: str
    scanned_documents: int
    documents_with_references: int
    resolved_edges: int
    unresolved_reference_titles: int
    elapsed_seconds: float
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaSaCitationGraph:
    """One-hop backward/forward graph derived from ``sections`` titles.

    PaSa stores references as ``section heading -> referenced paper titles``.
    The graph resolves those titles against the frozen ``id2paper.json`` and
    persists only edges whose two endpoints have a local arXiv identifier.
    Forward citations are obtained by querying the reverse edge index.
    """

    def __init__(
        self,
        *,
        paper_zip_path: str | Path | None = None,
        id_map_path: str | Path | None = None,
        graph_path: str | Path | None = None,
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
        self.graph_path = self._resolve_path(
            graph_path,
            "PASA_CITATION_GRAPH_PATH",
            PROJECT_ROOT / "models" / "pasa_citations" / "pasa_citations.sqlite3",
        )

    @staticmethod
    def _resolve_path(
        configured: str | Path | None,
        env_name: str,
        default: Path,
    ) -> Path:
        if configured is not None:
            return Path(configured).expanduser().resolve()
        raw = os.getenv(env_name, "").strip()
        return Path(raw).expanduser().resolve() if raw else default.resolve()

    def is_ready(self) -> bool:
        if not self.graph_path.is_file():
            return False
        try:
            with closing(self._connect(read_only=True)) as connection:
                metadata = dict(
                    connection.execute("SELECT key, value FROM graph_metadata")
                )
            return (
                metadata.get("schema_version") == CITATION_GRAPH_SCHEMA_VERSION
                and metadata.get("complete") == "1"
                and metadata.get("source_signature") == self._source_signature()
            )
        except (OSError, sqlite3.Error):
            return False

    def metadata(self) -> dict[str, str]:
        self._require_ready()
        with closing(self._connect(read_only=True)) as connection:
            return dict(connection.execute("SELECT key, value FROM graph_metadata"))

    def build(
        self,
        *,
        force: bool = False,
        batch_size: int = 20_000,
        progress_every: int = 20_000,
        max_documents: int | None = None,
        progress: Callable[[str], None] | None = print,
    ) -> PaSaCitationBuildStats:
        if batch_size <= 0 or progress_every <= 0:
            raise ValueError("batch_size and progress_every must be positive")
        if max_documents is not None and max_documents <= 0:
            raise ValueError("max_documents must be positive")
        self._require_sources()
        if self.is_ready() and not force and max_documents is None:
            metadata = self.metadata()
            return PaSaCitationBuildStats(
                graph_path=str(self.graph_path),
                scanned_documents=int(metadata.get("scanned_documents", "0")),
                documents_with_references=int(
                    metadata.get("documents_with_references", "0")
                ),
                resolved_edges=int(metadata.get("resolved_edges", "0")),
                unresolved_reference_titles=int(
                    metadata.get("unresolved_reference_titles", "0")
                ),
                elapsed_seconds=0.0,
                complete=True,
            )

        started = time.perf_counter()
        title_to_arxiv = self._load_title_map()
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.graph_path.with_name(self.graph_path.name + ".building")
        if temporary.exists():
            temporary.unlink()

        scanned = 0
        docs_with_refs = 0
        resolved_edges = 0
        unresolved = 0
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=MEMORY")
            self._create_schema(connection)
            batch: list[tuple[str, str, int]] = []
            with zipfile.ZipFile(self.paper_zip_path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    if max_documents is not None and scanned >= max_documents:
                        break
                    try:
                        raw = json.loads(archive.read(info).decode("utf-8"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if not isinstance(raw, dict):
                        continue
                    scanned += 1
                    citing_id = title_to_arxiv.get(normalize_title(raw.get("title", "")))
                    references = self._reference_titles(raw.get("sections"))
                    if references:
                        docs_with_refs += 1
                    if not citing_id:
                        unresolved += len(references)
                        continue
                    seen_targets: set[str] = set()
                    for ordinal, title in enumerate(references, start=1):
                        cited_id = title_to_arxiv.get(normalize_title(title))
                        if not cited_id:
                            unresolved += 1
                            continue
                        if cited_id == citing_id or cited_id in seen_targets:
                            continue
                        seen_targets.add(cited_id)
                        batch.append((citing_id, cited_id, ordinal))
                    if len(batch) >= batch_size:
                        resolved_edges += self._insert_batch(connection, batch)
                        batch.clear()
                    if progress and scanned % progress_every == 0:
                        progress(
                            f"  scanned={scanned:,} resolved_edges={resolved_edges:,}"
                        )
            if batch:
                resolved_edges += self._insert_batch(connection, batch)
            complete = max_documents is None
            metadata = {
                "schema_version": CITATION_GRAPH_SCHEMA_VERSION,
                "complete": "1" if complete else "0",
                "built_at": datetime.now().astimezone().isoformat(),
                "source_signature": self._source_signature(),
                "scanned_documents": str(scanned),
                "documents_with_references": str(docs_with_refs),
                "resolved_edges": str(resolved_edges),
                "unresolved_reference_titles": str(unresolved),
            }
            connection.executemany(
                "INSERT INTO graph_metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            connection.commit()
            connection.execute("PRAGMA optimize")
        except Exception:
            connection.close()
            if temporary.exists():
                temporary.unlink()
            raise
        else:
            connection.close()
        os.replace(temporary, self.graph_path)
        stats = PaSaCitationBuildStats(
            graph_path=str(self.graph_path),
            scanned_documents=scanned,
            documents_with_references=docs_with_refs,
            resolved_edges=resolved_edges,
            unresolved_reference_titles=unresolved,
            elapsed_seconds=time.perf_counter() - started,
            complete=max_documents is None,
        )
        if progress:
            progress(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
        return stats

    def neighbors(
        self,
        arxiv_id: str,
        *,
        direction: Literal["forward", "backward"],
        limit: int,
    ) -> list[str]:
        if limit <= 0:
            return []
        self._require_ready()
        normalized = normalize_arxiv_id(arxiv_id)
        if not normalized:
            return []
        if direction == "backward":
            sql = (
                "SELECT cited_arxiv_id FROM citation_edges "
                "WHERE citing_arxiv_id=? ORDER BY ordinal, cited_arxiv_id LIMIT ?"
            )
        elif direction == "forward":
            sql = (
                "SELECT citing_arxiv_id FROM citation_edges "
                "WHERE cited_arxiv_id=? ORDER BY citing_arxiv_id LIMIT ?"
            )
        else:
            raise ValueError(f"Unsupported citation direction: {direction}")
        with closing(self._connect(read_only=True)) as connection:
            rows = connection.execute(sql, (normalized, int(limit))).fetchall()
        return [str(row[0]) for row in rows]

    @staticmethod
    def _reference_titles(sections: Any) -> list[str]:
        if not isinstance(sections, dict):
            return []
        output: list[str] = []
        seen: set[str] = set()
        for values in sections.values():
            items = values if isinstance(values, list) else [values]
            for value in items:
                title = str(value or "").strip()
                key = normalize_title(title)
                if key and key not in seen:
                    seen.add(key)
                    output.append(title)
        return output

    def _load_title_map(self) -> dict[str, str]:
        raw = json.loads(self.id_map_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise PaSaCitationGraphError("id2paper.json must contain an object")
        output: dict[str, str] = {}
        for raw_id, raw_title in raw.items():
            arxiv_id = normalize_arxiv_id(raw_id)
            title = normalize_title(raw_title)
            if arxiv_id and title:
                output.setdefault(title, arxiv_id)
        return output

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE graph_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE citation_edges (
                citing_arxiv_id TEXT NOT NULL,
                cited_arxiv_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (citing_arxiv_id, cited_arxiv_id)
            ) WITHOUT ROWID;
            CREATE INDEX idx_citation_edges_cited
                ON citation_edges(cited_arxiv_id, citing_arxiv_id);
            """
        )

    @staticmethod
    def _insert_batch(
        connection: sqlite3.Connection,
        batch: list[tuple[str, str, int]],
    ) -> int:
        before = connection.total_changes
        connection.executemany(
            "INSERT OR IGNORE INTO citation_edges "
            "(citing_arxiv_id, cited_arxiv_id, ordinal) VALUES (?, ?, ?)",
            batch,
        )
        connection.commit()
        return connection.total_changes - before

    def _source_signature(self) -> str:
        self._require_sources()
        zip_stat = self.paper_zip_path.stat()
        map_stat = self.id_map_path.stat()
        return json.dumps(
            {
                "paper_zip_size": zip_stat.st_size,
                "paper_zip_mtime_ns": zip_stat.st_mtime_ns,
                "id_map_size": map_stat.st_size,
                "id_map_mtime_ns": map_stat.st_mtime_ns,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _require_sources(self) -> None:
        missing = [
            str(path)
            for path in (self.paper_zip_path, self.id_map_path)
            if not path.is_file()
        ]
        if missing:
            raise PaSaCitationGraphError(
                "Missing PaSa citation source files: " + ", ".join(missing)
            )

    def _require_ready(self) -> None:
        if not self.is_ready():
            raise PaSaCitationGraphError(
                f"PaSa citation graph is missing or stale: {self.graph_path}"
            )

    def _connect(self, *, read_only: bool) -> sqlite3.Connection:
        if read_only:
            return sqlite3.connect(
                f"file:{self.graph_path.as_posix()}?mode=ro",
                uri=True,
            )
        return sqlite3.connect(self.graph_path)


__all__ = [
    "PaSaCitationGraph",
    "PaSaCitationBuildStats",
    "PaSaCitationGraphError",
]
