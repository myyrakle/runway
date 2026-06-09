"""FastAPI app exposing multilingual-e5-large text embeddings.

Endpoints:
  GET  /ping        SageMaker health check
  POST /invocations SageMaker inference adapter
  GET  /health
  POST /embed      list of texts -> L2-normalized vectors (precision selectable)
  POST /benchmark  encode latency stats (precision selectable)
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.config import EMBEDDING_DIM, MODEL_NAME
from app.model import E5Embedder
from app.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
)

# Lazily-loaded model variants, keyed by precision. fp32 loaded on startup.
_models: dict[str, E5Embedder] = {}
_load_lock = threading.Lock()


def _get_variant(precision: str) -> E5Embedder:
    """Return the model for a precision, loading + caching on first use."""
    cached = _models.get(precision)
    if cached is not None:
        return cached
    with _load_lock:
        if precision not in _models:
            try:
                _models[precision] = E5Embedder(precision=precision)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _models[precision]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _models["fp32"] = E5Embedder(precision="fp32")
    yield
    _models.clear()


app = FastAPI(
    title="Embedding Service (multilingual-e5-large)",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    fp32 = _models.get("fp32")
    return HealthResponse(
        status="ok" if fp32 is not None else "loading",
        model=MODEL_NAME,
        device=fp32.device if fp32 is not None else "unknown",
        dim=EMBEDDING_DIM,
        loaded_precisions=sorted(_models.keys()),
    )


@app.get("/ping")
async def ping() -> dict[str, str]:
    return {"status": "ok" if "fp32" in _models else "loading"}


@app.post("/invocations", response_model=EmbedResponse)
async def invocations(req: EmbedRequest) -> EmbedResponse:
    return await embed(req)


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    model = await run_in_threadpool(_get_variant, req.precision)
    arr = await run_in_threadpool(model.generate, req.texts, req.prefix, req.batch_size)
    vectors = arr.tolist()
    dim = arr.shape[1] if arr.ndim == 2 and arr.shape[0] > 0 else EMBEDDING_DIM
    return EmbedResponse(embeddings=vectors, dim=dim, count=len(vectors))


@app.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    """Time repeated batch encoding for a precision variant. Warmup excluded."""
    model = await run_in_threadpool(_get_variant, req.precision)
    n = len(req.texts)

    def run() -> list[float]:
        for _ in range(req.warmup):
            model.generate(req.texts, req.prefix, req.batch_size)
        durations = []
        for _ in range(req.iterations):
            start = time.perf_counter()
            model.generate(req.texts, req.prefix, req.batch_size)
            durations.append((time.perf_counter() - start) * 1000.0)
        return durations

    durations = await run_in_threadpool(run)
    avg_ms = sum(durations) / len(durations)
    return BenchmarkResponse(
        model=MODEL_NAME,
        precision=req.precision,
        device=model.device,
        iterations=req.iterations,
        warmup=req.warmup,
        texts_per_iteration=n,
        avg_ms=avg_ms,
        min_ms=min(durations),
        max_ms=max(durations),
        total_ms=sum(durations),
        per_text_ms=avg_ms / n if n else 0.0,
    )
