"""Pydantic request/response models for the sentiment service."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    aspect: str = Field("overall", description="ABSA aspect / target")


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., description="Texts to analyze")
    aspect: str = Field("overall", description="ABSA aspect / target")


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


class BenchmarkRequest(BaseModel):
    text: str = Field(
        "The battery life is terrible and the screen is great.",
        description="Sample text to repeatedly analyze",
    )
    aspect: str = Field("overall", description="ABSA aspect / target")
    iterations: int = Field(20, ge=1, le=1000, description="Timed iterations")
    warmup: int = Field(2, ge=0, le=100, description="Untimed warmup iterations")


class BenchmarkResponse(BaseModel):
    model: str
    device: str
    iterations: int
    warmup: int
    avg_ms: float
    min_ms: float
    max_ms: float
    total_ms: float
