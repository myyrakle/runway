//! HTTP layer (axum). Endpoints mirror the Python service:
//!   GET  /ping           health check (SageMaker)
//!   GET  /health
//!   POST /invocations    batch-or-single (text | texts | instances | groups)
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

async fn run_pairs(engine: AppState, texts: Vec<String>, aspects: Vec<String>) -> Response {
    let result =
        tokio::task::spawn_blocking(move || engine.analyze_pairs(&texts, &aspects)).await;
    match result {
        Ok(Ok(results)) => Json(BatchAnalyzeResponse { results }).into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")),
    }
}

/// What an /invocations body resolves to, independent of the HTTP layer.
enum Dispatch {
    Single { text: String, aspect: String },
    Batch { texts: Vec<String>, aspect: String },
    Pairs { texts: Vec<String>, aspects: Vec<String> },
}

/// Validate the "exactly one payload" rule and extract the work to run. Returns the
/// FastAPI-style `detail` string on a validation failure. `aspect` (top-level) is used
/// for the single/batch forms; the `groups` form carries a per-item aspect instead.
fn plan_invocation(req: InvocationRequest) -> Result<Dispatch, String> {
    let provided = [
        req.text.is_some(),
        req.texts.is_some(),
        req.instances.is_some(),
        req.groups.is_some(),
    ]
    .iter()
    .filter(|&&b| b)
    .count();

    if provided != 1 {
        return Err("Provide exactly one of text, texts, instances, or groups".to_string());
    }

    if let Some(text) = req.text {
        return Ok(Dispatch::Single {
            text,
            aspect: req.aspect,
        });
    }
    if let Some(groups) = req.groups {
        let texts = groups.iter().map(|g| g.text.clone()).collect();
        let aspects = groups.iter().map(|g| g.aspect.clone()).collect();
        return Ok(Dispatch::Pairs { texts, aspects });
    }
    let texts = req.texts.or(req.instances).unwrap();
    Ok(Dispatch::Batch {
        texts,
        aspect: req.aspect,
    })
}

/// /invocations accepts exactly one of text / texts / instances / groups.
async fn invocations(
    State(engine): State<AppState>,
    Json(req): Json<InvocationRequest>,
) -> Response {
    match plan_invocation(req) {
        Err(detail) => err(StatusCode::BAD_REQUEST, detail),
        Ok(Dispatch::Single { text, aspect }) => run_single(engine, text, aspect).await,
        Ok(Dispatch::Batch { texts, aspect }) => {
            if let Err(resp) = validate_batch_len(&engine, texts.len()) {
                return resp;
            }
            run_batch(engine, texts, aspect).await
        }
        Ok(Dispatch::Pairs { texts, aspects }) => {
            if let Err(resp) = validate_batch_len(&engine, texts.len()) {
                return resp;
            }
            run_pairs(engine, texts, aspects).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> InvocationRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn plan_groups_flattens_to_parallel_texts_and_aspects() {
        let req = parse(
            r#"{"groups": [
                {"text": "battery is bad", "aspect": "battery"},
                {"text": "battery is bad", "aspect": "price"},
                {"text": "screen is great", "aspect": "screen"}
            ]}"#,
        );
        match plan_invocation(req).unwrap() {
            Dispatch::Pairs { texts, aspects } => {
                assert_eq!(
                    texts,
                    vec!["battery is bad", "battery is bad", "screen is great"]
                );
                assert_eq!(aspects, vec!["battery", "price", "screen"]);
            }
            _ => panic!("expected Pairs dispatch"),
        }
    }

    #[test]
    fn plan_rejects_groups_combined_with_texts() {
        let req = parse(r#"{"texts": ["a"], "groups": [{"text": "b", "aspect": "x"}]}"#);
        assert!(plan_invocation(req).is_err());
    }

    #[test]
    fn plan_routes_texts_and_instances_to_batch() {
        match plan_invocation(parse(r#"{"texts": ["a", "b"], "aspect": "battery"}"#)).unwrap() {
            Dispatch::Batch { texts, aspect } => {
                assert_eq!(texts, vec!["a", "b"]);
                assert_eq!(aspect, "battery");
            }
            _ => panic!("expected Batch"),
        }
        match plan_invocation(parse(r#"{"instances": ["a"]}"#)).unwrap() {
            Dispatch::Batch { texts, .. } => assert_eq!(texts, vec!["a"]),
            _ => panic!("expected Batch from instances"),
        }
    }

    #[test]
    fn plan_rejects_empty_payload() {
        assert!(plan_invocation(parse(r#"{}"#)).is_err());
    }
}
