"""Multilingual E5 Large embedding model wrapper.

E5 task prefix on every input, L2-normalized output, batched encode.
"""
from __future__ import annotations

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import DEFAULT_BATCH_SIZE, DEFAULT_PREFIX, MODEL_NAME


class E5Embedder:
    """sentence-transformers multilingual-e5-large wrapper (fp32, GPU auto-detect)."""

    def __init__(self) -> None:
        print(f"[Embedding] Loading model: {MODEL_NAME}")
        start = time.time()
        self.model = SentenceTransformer(MODEL_NAME)
        self.device = str(self.model.device)
        print(f"[Embedding] Model loaded on {self.device} ({time.time() - start:.1f}s)")

    def generate(
        self,
        texts: list[str],
        prefix: str = DEFAULT_PREFIX,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Batch-encode texts. Returns L2-normalized array of shape (N, dim).

        E5 requires a task prefix ("query: " / "passage: ") on every input.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        prefixed = [f"{prefix}{t}" for t in texts]

        all_embeddings = []
        total = len(prefixed)
        for i in range(0, total, batch_size):
            batch = prefixed[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
