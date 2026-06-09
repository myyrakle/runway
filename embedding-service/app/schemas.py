"""Pydantic request/response models for the embedding service."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import DEFAULT_BATCH_SIZE, DEFAULT_PREFIX


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., description="Texts to embed")
    prefix: str = Field(
        DEFAULT_PREFIX,
        description='E5 task prefix. Use "query: " for short queries/labels, '
                    '"passage: " for long documents/quotes.',
    )
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1, description="Encode batch size")


class EmbedResponse(BaseModel):
    embeddings: list[list[float]] = Field(..., description="L2-normalized vectors")
    dim: int
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    dim: int


class BenchmarkRequest(BaseModel):
    texts: list[str] = Field(
        default_factory=lambda: ["The quick brown fox jumps over the lazy dog."],
        description="Sample texts encoded once per iteration",
    )
    prefix: str = Field(DEFAULT_PREFIX, description="E5 task prefix")
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1, description="Encode batch size")
    iterations: int = Field(20, ge=1, le=1000, description="Timed iterations")
    warmup: int = Field(2, ge=0, le=100, description="Untimed warmup iterations")


class BenchmarkResponse(BaseModel):
    model: str
    device: str
    iterations: int
    warmup: int
    texts_per_iteration: int
    avg_ms: float
    min_ms: float
    max_ms: float
    total_ms: float
