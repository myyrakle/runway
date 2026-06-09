"""FastAPI app exposing DeBERTa ABSA sentiment analysis.

Endpoints:
  GET  /health
  POST /analyze        single text
  POST /analyze/batch  list of texts
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.config import DEVICE, MODEL_NAME
from app.model import DeBERTaABSA
from app.schemas import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    HealthResponse,
    SentimentResult,
)

# Module-level singleton holder. Loaded once on startup.
_model: DeBERTaABSA | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = DeBERTaABSA()
    yield
    _model = None


app = FastAPI(
    title="Sentiment Service (DeBERTa ABSA)",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model() -> DeBERTaABSA:
    if _model is None:
        raise RuntimeError("Model not loaded")
    return _model


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model is not None else "loading",
        model=MODEL_NAME,
        device=DEVICE,
    )


@app.post("/analyze", response_model=SentimentResult)
async def analyze(req: AnalyzeRequest) -> SentimentResult:
    # torch inference is blocking → offload to threadpool.
    result = await run_in_threadpool(get_model().analyze, req.text, req.aspect)
    return SentimentResult(**result)


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(req: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    results = await run_in_threadpool(get_model().analyze_batch, req.texts, req.aspect)
    return BatchAnalyzeResponse(results=[SentimentResult(**r) for r in results])


@app.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    """Time repeated single-text inference. Warmup runs are excluded from stats."""
    model = get_model()

    def run() -> list[float]:
        for _ in range(req.warmup):
            model.analyze(req.text, req.aspect)
        durations = []
        for _ in range(req.iterations):
            start = time.perf_counter()
            model.analyze(req.text, req.aspect)
            durations.append((time.perf_counter() - start) * 1000.0)
        return durations

    durations = await run_in_threadpool(run)
    return BenchmarkResponse(
        model=MODEL_NAME,
        device=DEVICE,
        iterations=req.iterations,
        warmup=req.warmup,
        avg_ms=sum(durations) / len(durations),
        min_ms=min(durations),
        max_ms=max(durations),
        total_ms=sum(durations),
    )
