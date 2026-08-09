from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "bge-m3"


@dataclass
class SemanticScorer:
    """BGE-M3 dense embedding wrapper used only for local semantic scoring.

    Notes:
    - No generative LLM call / no token cost.
    - Model is loaded once and reused.
    - Embeddings are normalized, so dot product == cosine similarity.
    """

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "BGE_M3_MODEL_PATH",
            str(_DEFAULT_MODEL_PATH),
        )
    )
    device: str | None = None
    max_seq_length: int = 512
    batch_size: int = 16
    _model: object | None = field(default=None, init=False, repr=False)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def _ensure_model(self):
        if self._model is None:
            model_path = Path(self.model_name).expanduser().resolve()
            if not model_path.is_dir():
                raise RuntimeError(
                    "Local BGE-M3 model directory was not found: "
                    f"{model_path}. Download the model there or set BGE_M3_MODEL_PATH."
                )

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "sentence-transformers is required for Adaptive Routing. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            model = SentenceTransformer(
                str(model_path),
                local_files_only=True,
                **kwargs,
            )
            # BGE-M3 supports longer input, but 512 is enough for title+abstract routing
            # and keeps routing latency bounded.
            model.max_seq_length = self.max_seq_length
            self._model = model
        return self._model

    def warmup(self) -> None:
        self.encode(["academic paper retrieval"])

    @staticmethod
    def paper_text(title: str | None, abstract: str | None) -> str:
        title = (title or "").strip()
        abstract = (abstract or "").strip()
        if title and abstract:
            return f"Title: {title}\nAbstract: {abstract}"
        return title or abstract

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        items = [str(x).strip() for x in texts]
        if not items:
            return np.empty((0, 1024), dtype=np.float32)

        missing = list(dict.fromkeys(x for x in items if x and x not in self._cache))
        if missing:
            model = self._ensure_model()
            vectors = model.encode(
                missing,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for text, vector in zip(missing, vectors):
                self._cache[text] = np.asarray(vector, dtype=np.float32)

        # Empty text gets a zero vector rather than causing a model call.
        dim = 1024
        if self._cache:
            dim = len(next(iter(self._cache.values())))
        return np.stack(
            [self._cache.get(x, np.zeros(dim, dtype=np.float32)) for x in items],
            axis=0,
        )

    def similarity(self, left: str, right: str) -> float:
        emb = self.encode([left, right])
        if len(emb) != 2:
            return 0.0
        return float(np.dot(emb[0], emb[1]))


__all__ = ["SemanticScorer"]
