"""FastAPI app exposing DeBERTa ABSA sentiment analysis.

Endpoints:
  GET  /ping          SageMaker health check
  POST /invocations   SageMaker inference adapter (batch: texts / instances)
  GET  /health
  POST /analyze        single text   (precision variant selectable)
  POST /analyze/batch  list of texts (precision variant selectable)
  POST /benchmark      latency stats (precision / single|batch selectable)

The inference endpoints (/invocations, /analyze, /analyze/batch) validate their
request bodies with Pydantic models (app.schemas) so the request schema shows up
in OpenAPI/Swagger, but they still return a JSONResponse of plain dicts rather
than a response_model. The response rebuild was the dominant host-side Pydantic
cost (the response_model reconstructed up to MAX_BATCH_ITEMS result objects per
call before serializing); the model already returns plain dicts, so we hand them
straight to JSONResponse and skip that step.
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.config import (
    DEFAULT_PRECISION,
    DEVICE,
    MODEL_NAME,
    is_precision_allowed,
)
from app.model import DeBERTaABSA
from app.schemas import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    BenchmarkRequest,
    BenchmarkResponse,
    HealthResponse,
    InvocationRequest,
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


# --- inference (Pydantic-validated request, plain-dict response) -------------

async def _resolve_model(precision: str) -> DeBERTaABSA:
    # Hot path: an already-loaded variant is a plain dict hit, so resolve it on the
    # event loop and skip the threadpool round trip. Only a cold (lazy-load) variant
    # is offloaded, since constructing the model blocks.
    cached = _models.get(precision)
    if cached is not None:
        return cached
    return await run_in_threadpool(_get_variant, precision)


async def _run_single(text: str, aspect: str, precision: str) -> dict:
    model = await _resolve_model(precision)
    return await run_in_threadpool(model.analyze, text, aspect)


async def _run_batch(texts: list[str], aspect: str, precision: str) -> dict:
    model = await _resolve_model(precision)
    results = await run_in_threadpool(model.analyze_batch, texts, aspect)
    return {"results": results}


@app.post("/invocations")
async def invocations(body: InvocationRequest) -> JSONResponse:
    """/invocations is batch-only: accept `texts` (or its `instances` alias)."""
    batch = body.texts if body.texts is not None else body.instances
    if batch is None:
        raise HTTPException(
            status_code=400,
            detail="Batch texts required: provide `texts` or `instances`",
        )
    return JSONResponse(await _run_batch(batch, body.aspect, body.precision))


@app.post("/analyze")
async def analyze(body: AnalyzeRequest) -> JSONResponse:
    return JSONResponse(await _run_single(body.text, body.aspect, body.precision))


@app.post("/analyze/batch")
async def analyze_batch(body: BatchAnalyzeRequest) -> JSONResponse:
    return JSONResponse(await _run_batch(body.texts, body.aspect, body.precision))


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
