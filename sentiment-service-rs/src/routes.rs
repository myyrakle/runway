//! HTTP layer (axum). Endpoints mirror the Python service:
//!   GET  /ping           health check (SageMaker)
//!   GET  /health
//!   POST /invocations    batch-or-single (text | texts | instances)
//!   POST /analyze        single text
//!   POST /analyze/batch  list of texts
//!
//! Inference runs on a blocking threadpool (spawn_blocking) since the backend call is
//! synchronous/CPU-GPU bound, keeping the async runtime free for connection handling.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;

use crate::inference::Engine;
use crate::schemas::{
    AnalyzeRequest, BatchAnalyzeRequest, BatchAnalyzeResponse, HealthResponse, InvocationRequest,
};

pub type AppState = Arc<Engine>;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/ping", get(ping))
        .route("/health", get(health))
        .route("/invocations", post(invocations))
        .route("/analyze", post(analyze))
        .route("/analyze/batch", post(analyze_batch))
        .with_state(state)
}

/// Error helper producing a FastAPI-style `{"detail": "..."}` body.
fn err(status: StatusCode, detail: impl Into<String>) -> Response {
    (status, Json(json!({ "detail": detail.into() }))).into_response()
}

async fn ping(State(engine): State<AppState>) -> Json<serde_json::Value> {
    let _ = engine;
    Json(json!({ "status": "ok" }))
}

async fn health(State(engine): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: engine.config.model_name.clone(),
        device: engine.backend_name().to_string(),
        loaded_precisions: vec!["fp16".to_string()],
    })
}

fn validate_batch_len(engine: &Engine, n: usize) -> Result<(), Response> {
    if n == 0 {
        return Err(err(StatusCode::BAD_REQUEST, "texts must not be empty"));
    }
    if n > engine.config.max_batch_items {
        return Err(err(
            StatusCode::BAD_REQUEST,
            format!(
                "texts exceeds MAX_BATCH_ITEMS ({})",
                engine.config.max_batch_items
            ),
        ));
    }
    Ok(())
}

/// Run a single analysis off the async runtime.
async fn run_single(engine: AppState, text: String, aspect: String) -> Response {
    let result = tokio::task::spawn_blocking(move || engine.analyze(&text, &aspect)).await;
    match result {
        Ok(Ok(r)) => Json(r).into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")),
    }
}

async fn run_batch(engine: AppState, texts: Vec<String>, aspect: String) -> Response {
    let result =
        tokio::task::spawn_blocking(move || engine.analyze_batch(&texts, &aspect)).await;
    match result {
        Ok(Ok(results)) => Json(BatchAnalyzeResponse { results }).into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")),
    }
}

async fn analyze(State(engine): State<AppState>, Json(req): Json<AnalyzeRequest>) -> Response {
    run_single(engine, req.text, req.aspect).await
}

async fn analyze_batch(
    State(engine): State<AppState>,
    Json(req): Json<BatchAnalyzeRequest>,
) -> Response {
    if let Err(resp) = validate_batch_len(&engine, req.texts.len()) {
        return resp;
    }
    run_batch(engine, req.texts, req.aspect).await
}

/// /invocations is batch-or-single: accept exactly one of text / texts / instances.
async fn invocations(
    State(engine): State<AppState>,
    Json(req): Json<InvocationRequest>,
) -> Response {
    let provided = [
        req.text.is_some(),
        req.texts.is_some(),
        req.instances.is_some(),
    ]
    .iter()
    .filter(|&&b| b)
    .count();

    if provided != 1 {
        return err(
            StatusCode::BAD_REQUEST,
            "Provide exactly one of text, texts, or instances",
        );
    }

    if let Some(text) = req.text {
        return run_single(engine, text, req.aspect).await;
    }
    let batch = req.texts.or(req.instances).unwrap();
    if let Err(resp) = validate_batch_len(&engine, batch.len()) {
        return resp;
    }
    run_batch(engine, batch, req.aspect).await
}
