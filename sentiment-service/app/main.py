"""FastAPI app exposing DeBERTa ABSA sentiment analysis.

Endpoints:
  GET  /ping          SageMaker health check
  POST /invocations   SageMaker inference adapter (single or batch)
  GET  /health
  POST /analyze        single text   (precision variant selectable)
  POST /analyze/batch  list of texts (precision variant selectable)
  POST /benchmark      latency stats (precision / single|batch selectable)
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.config import DEFAULT_PRECISION, DEVICE, MODEL_NAME, is_precision_allowed
from app.model import DeBERTaABSA
from app.schemas import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    HealthResponse,
    InvocationRequest,
    SentimentResult,
)

# Lazily-loaded model variants, keyed by precision.
_models: dict[str, DeBERTaABSA] = {}
_load_lock = threading.Lock()


def _get_variant(precision: str) -> DeBERTaABSA:
    """Return the model for a precision, loading + caching on first use."""
    if not is_precision_allowed(precision):
        raise HTTPException(
            status_code=400,
            detail=f"Precision {precision!r} is not enabled for this container",
        )
    cached = _models.get(precision)
    if cached is not None:
        return cached
    with _load_lock:
        if precision not in _models:
            try:
                _models[precision] = DeBERTaABSA(precision=precision)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _models[precision]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _models[DEFAULT_PRECISION] = DeBERTaABSA(precision=DEFAULT_PRECISION)
    yield
    _models.clear()


app = FastAPI(
    title="Sentiment Service (DeBERTa ABSA)",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if DEFAULT_PRECISION in _models else "loading",
        model=MODEL_NAME,
        device=DEVICE,
        loaded_precisions=sorted(_models.keys()),
    )


@app.get("/ping")
async def ping() -> dict[str, str]:
    return {"status": "ok" if DEFAULT_PRECISION in _models else "loading"}


@app.post("/invocations", response_model=SentimentResult | BatchAnalyzeResponse)
async def invocations(
    req: InvocationRequest,
) -> SentimentResult | BatchAnalyzeResponse:
    if isinstance(req.text, str):
        return await _run_single_analysis(req.as_single_request())
    return await _run_batch_analysis(req.as_batch_request())


async def _run_single_analysis(req: AnalyzeRequest) -> SentimentResult:
    model = await run_in_threadpool(_get_variant, req.precision)
    result = await run_in_threadpool(model.analyze, req.text, req.aspect)
    return SentimentResult(**result)


async def _run_batch_analysis(req: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    model = await run_in_threadpool(_get_variant, req.precision)
    results = await run_in_threadpool(model.analyze_batch, req.texts, req.aspect)
    return BatchAnalyzeResponse(results=[SentimentResult(**r) for r in results])


@app.post("/analyze", response_model=SentimentResult)
async def analyze(req: AnalyzeRequest) -> SentimentResult:
    return await _run_single_analysis(req)


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(req: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    return await _run_batch_analysis(req)


@app.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    """Time repeated inference for a precision variant. Warmup excluded from stats."""
    model = await run_in_threadpool(_get_variant, req.precision)

    if req.mode == "batch":
        payload = [req.text] * req.batch_size
        call = lambda: model.analyze_batch(payload, req.aspect)
    else:
        call = lambda: model.analyze(req.text, req.aspect)

    def run() -> list[float]:
        for _ in range(req.warmup):
            call()
        durations = []
        for _ in range(req.iterations):
            start = time.perf_counter()
            call()
            durations.append((time.perf_counter() - start) * 1000.0)
        return durations

    durations = await run_in_threadpool(run)
    avg_ms = sum(durations) / len(durations)
    return BenchmarkResponse(
        model=MODEL_NAME,
        precision=req.precision,
        device=model.device,
        mode=req.mode,
        batch_size=req.batch_size if req.mode == "batch" else 1,
        iterations=req.iterations,
        warmup=req.warmup,
        avg_ms=avg_ms,
        min_ms=min(durations),
        max_ms=max(durations),
        total_ms=sum(durations),
        per_item_ms=avg_ms / req.batch_size if req.mode == "batch" else None,
    )
