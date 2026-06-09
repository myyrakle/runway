"""Configuration for the multilingual-e5-large embedding service."""
from __future__ import annotations

import os

# Embedding model (HuggingFace repo id).
MODEL_NAME = os.environ.get("MODEL_NAME", "intfloat/multilingual-e5-large")

# Output dimension (L2-normalized).
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))

# Default encode batch size.
DEFAULT_BATCH_SIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", "256"))

# E5 models require a task prefix on every input:
#   "query: "   for short queries / labels
#   "passage: " for long documents / quotes
# default_prefix is applied when a request does not specify one.
DEFAULT_PREFIX = os.environ.get("DEFAULT_PREFIX", "query: ")
