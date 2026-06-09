"""FastAPI app exposing multilingual-e5-large text embeddings.

Endpoints:
  GET  /health
  POST /embed   list of texts -> L2-normalized vectors
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.config import EMBEDDING_DIM, MODEL_NAME
from app.model import E5Embedder
from app.schemas import EmbedRequest, EmbedResponse, HealthResponse

_model: E5Embedder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = E5Embedder()
    yield
    _model = None


app = FastAPI(
    title="Embedding Service (multilingual-e5-large)",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model() -> E5Embedder:
    if _model is None:
        raise RuntimeError("Model not loaded")
    return _model


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model is not None else "loading",
        model=MODEL_NAME,
        device=_model.device if _model is not None else "unknown",
        dim=EMBEDDING_DIM,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    # sentence-transformers encode is blocking → offload to threadpool.
    arr = await run_in_threadpool(
        get_model().generate, req.texts, req.prefix, req.batch_size
    )
    vectors = arr.tolist()
    dim = arr.shape[1] if arr.ndim == 2 and arr.shape[0] > 0 else EMBEDDING_DIM
    return EmbedResponse(embeddings=vectors, dim=dim, count=len(vectors))
