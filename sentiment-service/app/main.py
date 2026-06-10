"""FastAPI app exposing DeBERTa ABSA sentiment analysis.

Endpoints:
  GET  /ping          SageMaker health check
  POST /invocations   SageMaker inference adapter (single or batch)
  GET  /health
  POST /analyze        single text   (precision variant selectable)
  POST /analyze/batch  list of texts (precision variant selectable)
  POST /benchmark      latency stats (precision / single|batch selectable)

The inference endpoints (/invocations, /analyze, /analyze/batch) parse the raw
JSON body and return a JSONResponse directly, deliberately bypassing Pydantic
request/response models. On the hot path Pydantic was the dominant host-side
cost: /invocations validated the texts list twice (once as InvocationRequest,
again when rebuilt as BatchAnalyzeRequest) and the response_model rebuilt up to
MAX_BATCH_ITEMS result objects per call before serializing. The model already
returns plain dicts, so we hand them straight to JSONResponse.
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.config import (
    DEFAULT_PRECISION,
    DEVICE,
    MAX_BATCH_ITEMS,
    MODEL_NAME,
    is_precision_allowed,
)
from app.model import DeBERTaABSA
from app.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    HealthResponse,
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


# --- inference hot path (no Pydantic) ----------------------------------------

def _aspect_precision(body: dict) -> tuple[str, str]:
    aspect = body.get("aspect", "overall")
    precision = body.get("precision", DEFAULT_PRECISION)
    if not isinstance(aspect, str) or not isinstance(precision, str):
        raise HTTPException(
            status_code=400, detail="aspect and precision must be strings"
        )
    return aspect, precision


def _require_batch(texts) -> list:
    if not isinstance(texts, list) or not texts:
        raise HTTPException(
            status_code=400, detail="Batch texts must be a non-empty list"
        )
    if len(texts) > MAX_BATCH_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(texts)} exceeds MAX_BATCH_ITEMS ({MAX_BATCH_ITEMS})",
        )
    return texts


async def _run_single(text: str, aspect: str, precision: str) -> dict:
    model = await run_in_threadpool(_get_variant, precision)
    return await run_in_threadpool(model.analyze, text, aspect)


async def _run_batch(texts: list[str], aspect: str, precision: str) -> dict:
    model = await run_in_threadpool(_get_variant, precision)
    results = await run_in_threadpool(model.analyze_batch, texts, aspect)
    return {"results": results}


async def _dispatch_invocation(body: dict) -> dict:
    """Route a raw /invocations payload to the single or batch inference path."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    text = body.get("text")
    texts = body.get("texts")
    instances = body.get("instances")
    if sum(value is not None for value in (text, texts, instances)) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of text, texts, or instances",
        )
    aspect, precision = _aspect_precision(body)
    if text is not None:
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="text must be a string")
        return await _run_single(text, aspect, precision)
    batch = _require_batch(texts if texts is not None else instances)
    return await _run_batch(batch, aspect, precision)


@app.post("/invocations")
async def invocations(request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(await _dispatch_invocation(body))


@app.post("/analyze")
async def analyze(request: Request) -> JSONResponse:
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise HTTPException(status_code=400, detail="text is required")
    aspect, precision = _aspect_precision(body)
    return JSONResponse(await _run_single(body["text"], aspect, precision))


@app.post("/analyze/batch")
async def analyze_batch(request: Request) -> JSONResponse:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    batch = _require_batch(body.get("texts"))
    aspect, precision = _aspect_precision(body)
    return JSONResponse(await _run_batch(batch, aspect, precision))


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
