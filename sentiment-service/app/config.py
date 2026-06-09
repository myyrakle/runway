"""Configuration for the DeBERTa ABSA sentiment service."""
from __future__ import annotations

import os

import torch

VALID_PRECISIONS = ("fp32", "fp16", "int8")

# ABSA model (HuggingFace repo id).
MODEL_NAME = os.environ.get("MODEL_NAME", "yangheng/deberta-v3-base-absa-v1.1")

# Negative sentiment decision threshold (label==negative AND confidence >= this).
NEGATIVE_THRESHOLD = float(os.environ.get("NEGATIVE_THRESHOLD", "0.6"))

# Token truncation length (DeBERTa-v3-base max context).
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))

# Maximum number of texts accepted by the public batch API.
MAX_BATCH_ITEMS = int(os.environ.get("MAX_BATCH_ITEMS", "1024"))

# Maximum number of texts sent through one model forward pass.
INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "64"))

# Approximate upper bound for padded tokens per model forward pass. A value <= 0
# disables token-budget chunking and uses only INFERENCE_BATCH_SIZE.
MAX_BATCH_TOKENS = int(os.environ.get("MAX_BATCH_TOKENS", "0"))

# Sort large batches by approximate sequence length before chunking. This keeps
# dynamic padding lower inside each model forward pass while preserving response order.
SORT_BATCH_BY_LENGTH = os.environ.get("SORT_BATCH_BY_LENGTH", "1").lower() not in {
    "0",
    "false",
    "no",
}

# Optional tokenizer padding alignment. Leave unset/0 to auto-use 8 for CUDA fp16.
PAD_TO_MULTIPLE_OF = int(os.environ.get("PAD_TO_MULTIPLE_OF", "0"))

# Precision loaded at startup. Docker images should set this explicitly.
DEFAULT_PRECISION = os.environ.get("DEFAULT_PRECISION", "fp32")
if DEFAULT_PRECISION not in VALID_PRECISIONS:
    raise ValueError(
        f"DEFAULT_PRECISION must be one of {VALID_PRECISIONS}, got {DEFAULT_PRECISION!r}"
    )

# Precisions this process may load. Defaults to only the startup precision to avoid
# accidentally keeping multiple model copies resident in one container.
ALLOWED_PRECISIONS = tuple(
    precision.strip()
    for precision in os.environ.get("ALLOWED_PRECISIONS", DEFAULT_PRECISION).split(",")
    if precision.strip()
)
if not ALLOWED_PRECISIONS:
    raise ValueError("ALLOWED_PRECISIONS must contain at least one precision")
unknown_precisions = set(ALLOWED_PRECISIONS) - set(VALID_PRECISIONS)
if unknown_precisions:
    raise ValueError(
        f"ALLOWED_PRECISIONS contains unknown values: {sorted(unknown_precisions)}"
    )
if DEFAULT_PRECISION not in ALLOWED_PRECISIONS:
    raise ValueError("DEFAULT_PRECISION must be included in ALLOWED_PRECISIONS")


def is_precision_allowed(precision: str) -> bool:
    return precision in ALLOWED_PRECISIONS


# Device auto-detect: CUDA if available else CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
