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
