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
