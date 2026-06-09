"""Multilingual E5 Large embedding model wrapper.

E5 task prefix on every input, L2-normalized output, batched encode.
Supports precision variants: fp32 (default), fp16 (CUDA).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import DEFAULT_BATCH_SIZE, DEFAULT_PREFIX, MODEL_NAME

PRECISIONS = ("fp32", "fp16")


class E5Embedder:
    """sentence-transformers multilingual-e5-large wrapper (GPU auto-detect)."""

    def __init__(self, precision: str = "fp32") -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"Unknown precision {precision!r}, expected {PRECISIONS}")
        self.precision = precision

        print(f"[Embedding] Loading {MODEL_NAME} precision={precision}")
        start = time.time()
        self.model = SentenceTransformer(MODEL_NAME)

        if precision == "fp16":
            if not torch.cuda.is_available():
                raise ValueError("fp16 precision requires a CUDA device")
            self.model = self.model.half()

        self.device = str(self.model.device)
        print(f"[Embedding] Loaded precision={precision} on {self.device} "
              f"({time.time() - start:.1f}s)")

    def generate(
        self,
        texts: list[str],
        prefix: str = DEFAULT_PREFIX,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Batch-encode texts. Returns L2-normalized array of shape (N, dim)."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        prefixed = [f"{prefix}{t}" for t in texts]

        all_embeddings = []
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
