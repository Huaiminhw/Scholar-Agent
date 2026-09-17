"""Build the frozen PaSa backward/forward citation graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent_v2.retrieval.sources.pasa_citations import PaSaCitationGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a frozen citation graph from PaSa sections."
    )
    parser.add_argument("--paper-zip", type=Path)
    parser.add_argument("--id-map", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--progress-every", type=int, default=20_000)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph = PaSaCitationGraph(
        paper_zip_path=args.paper_zip,
        id_map_path=args.id_map,
        graph_path=args.output,
    )
    graph.build(
        force=args.force,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        max_documents=args.max_documents,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
