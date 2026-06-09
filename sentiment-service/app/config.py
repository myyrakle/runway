"""Configuration for the DeBERTa ABSA sentiment service."""
from __future__ import annotations

import os

import torch

# ABSA model (HuggingFace repo id).
MODEL_NAME = os.environ.get("MODEL_NAME", "yangheng/deberta-v3-base-absa-v1.1")

# Negative sentiment decision threshold (label==negative AND confidence >= this).
NEGATIVE_THRESHOLD = float(os.environ.get("NEGATIVE_THRESHOLD", "0.6"))

# Token truncation length (DeBERTa-v3-base max context).
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))

# Device auto-detect: CUDA if available else CPU. fp32 (no quantization).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
