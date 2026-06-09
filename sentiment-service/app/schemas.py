"""Pydantic request/response models for the sentiment service."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Precision = Literal["fp32", "fp16", "int8"]
BenchMode = Literal["single", "batch"]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    aspect: str = Field("overall", description="ABSA aspect / target")
    precision: Precision = Field("fp32", description="Model precision variant")


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., description="Texts to analyze")
    aspect: str = Field("overall", description="ABSA aspect / target")
    precision: Precision = Field("fp32", description="Model precision variant")


class Probs(BaseModel):
    negative: float
    neutral: float
    positive: float


class SentimentResult(BaseModel):
    sentiment: str = Field(..., description="negative | neutral | positive")
    confidence: float
    is_negative: bool
    probs: Probs


class BatchAnalyzeResponse(BaseModel):
    results: list[SentimentResult]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    loaded_precisions: list[str]


class BenchmarkRequest(BaseModel):
    text: str = Field(
        "The battery life is terrible and the screen is great.",
        description="Sample text to repeatedly analyze",
    )
    aspect: str = Field("overall", description="ABSA aspect / target")
    precision: Precision = Field("fp32", description="Model precision variant to bench")
    mode: BenchMode = Field("single", description="single = analyze, batch = analyze_batch")
    batch_size: int = Field(
        32, ge=1, le=4096,
        description="Texts per call when mode=batch (text is repeated)",
    )
    iterations: int = Field(20, ge=1, le=1000, description="Timed iterations")
    warmup: int = Field(2, ge=0, le=100, description="Untimed warmup iterations")


class BenchmarkResponse(BaseModel):
    model: str
    precision: str
    device: str
    mode: str
    batch_size: int
    iterations: int
    warmup: int
    avg_ms: float
    min_ms: float
    max_ms: float
    total_ms: float
    # Per-item latency when mode=batch (avg_ms / batch_size). Null in single mode.
    per_item_ms: float | None = None
