"""Pydantic request/response models for the sentiment service."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import DEFAULT_PRECISION, MAX_BATCH_ITEMS

Precision = Literal["fp32", "fp16", "int8"]
BenchMode = Literal["single", "batch"]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    aspect: str = Field("overall", description="ABSA aspect / target")
    precision: Precision = Field(DEFAULT_PRECISION, description="Model precision variant")


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_ITEMS,
        description="Texts to analyze",
    )
    aspect: str = Field("overall", description="ABSA aspect / target")
    precision: Precision = Field(DEFAULT_PRECISION, description="Model precision variant")


class InvocationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "texts": [
                        "The battery life is terrible",
                        "The screen is great",
                    ],
                    "aspect": "overall",
                }
            ]
        }
    )

    text: str | None = Field(None, description="Single text to analyze")
    texts: list[str] | None = Field(
        None,
        min_length=1,
        max_length=MAX_BATCH_ITEMS,
        description="Batch texts to analyze",
        examples=[
            [
                "The battery life is terrible",
                "The screen is great",
            ]
        ],
    )
    instances: list[str] | None = Field(
        None,
        min_length=1,
        max_length=MAX_BATCH_ITEMS,
        description="SageMaker-style batch texts to analyze",
    )
    aspect: str = Field("overall", description="ABSA aspect / target")
    precision: Precision = Field(DEFAULT_PRECISION, description="Model precision variant")

    @model_validator(mode="after")
    def validate_payload(self) -> "InvocationRequest":
        provided = [
            self.text is not None,
            self.texts is not None,
            self.instances is not None,
        ]
        if sum(provided) != 1:
            raise ValueError("Provide exactly one of text, texts, or instances")
        return self

    def as_single_request(self) -> AnalyzeRequest:
        if self.text is None:
            raise ValueError("InvocationRequest does not contain a single text")
        return AnalyzeRequest(
            text=self.text,
            aspect=self.aspect,
            precision=self.precision,
        )

    def as_batch_request(self) -> BatchAnalyzeRequest:
        texts = self.texts if self.texts is not None else self.instances
        if texts is None:
            raise ValueError("InvocationRequest does not contain batch texts")
        return BatchAnalyzeRequest(
            texts=texts,
            aspect=self.aspect,
            precision=self.precision,
        )


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
    precision: Precision = Field(DEFAULT_PRECISION, description="Model precision variant to bench")
    mode: BenchMode = Field("single", description="single = analyze, batch = analyze_batch")
    batch_size: int = Field(
        32, ge=1, le=MAX_BATCH_ITEMS,
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
