//! Request/response shapes, mirroring the Python service's `app/schemas.py` so the
//! HTTP contract is identical. `precision` is accepted for wire-compatibility but the
//! binary always serves its single baked engine (fp16); an unsupported value is
//! rejected with 400 in the routes layer.

use serde::{Deserialize, Serialize};

fn default_aspect() -> String {
    "overall".to_string()
}

fn default_precision() -> String {
    "fp16".to_string()
}

#[derive(Debug, Deserialize)]
pub struct AnalyzeRequest {
    pub text: String,
    #[serde(default = "default_aspect")]
    pub aspect: String,
    // Accepted for wire-compatibility with the Python API; this binary serves one
    // baked fp16 engine, so the value is parsed but not acted on.
    #[serde(default = "default_precision")]
    #[allow(dead_code)]
    pub precision: String,
}

#[derive(Debug, Deserialize)]
pub struct BatchAnalyzeRequest {
    pub texts: Vec<String>,
    #[serde(default = "default_aspect")]
    pub aspect: String,
    // Accepted for wire-compatibility with the Python API; this binary serves one
    // baked fp16 engine, so the value is parsed but not acted on.
    #[serde(default = "default_precision")]
    #[allow(dead_code)]
    pub precision: String,
}

/// A single (text, aspect) pair, so one batch can mix aspects per item.
#[derive(Debug, Deserialize)]
pub struct Group {
    pub text: String,
    #[serde(default = "default_aspect")]
    pub aspect: String,
}

/// SageMaker `/invocations` body: exactly one of `text`, `texts`, `instances`, or `groups`.
#[derive(Debug, Deserialize)]
pub struct InvocationRequest {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub texts: Option<Vec<String>>,
    #[serde(default)]
    pub instances: Option<Vec<String>>,
    /// Per-item (text, aspect) pairs. When present, the top-level `aspect` is ignored
    /// and each group carries its own aspect.
    #[serde(default)]
    pub groups: Option<Vec<Group>>,
    #[serde(default = "default_aspect")]
    pub aspect: String,
    // Accepted for wire-compatibility with the Python API; this binary serves one
    // baked fp16 engine, so the value is parsed but not acted on.
    #[serde(default = "default_precision")]
    #[allow(dead_code)]
    pub precision: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Probs {
    pub negative: f32,
    pub neutral: f32,
    pub positive: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SentimentResult {
    pub sentiment: String,
    pub confidence: f32,
    pub is_negative: bool,
    pub probs: Probs,
}

#[derive(Debug, Serialize)]
pub struct BatchAnalyzeResponse {
    pub results: Vec<SentimentResult>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub device: String,
    pub loaded_precisions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_aspect_defaults_to_overall() {
        let g: Group = serde_json::from_str(r#"{"text": "battery is bad"}"#).unwrap();
        assert_eq!(g.text, "battery is bad");
        assert_eq!(g.aspect, "overall");
    }

    #[test]
    fn invocation_parses_groups_of_text_aspect_pairs() {
        let req: InvocationRequest = serde_json::from_str(
            r#"{"groups": [
                {"text": "battery is bad", "aspect": "battery"},
                {"text": "battery is bad", "aspect": "price"}
            ]}"#,
        )
        .unwrap();
        let groups = req.groups.expect("groups present");
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].aspect, "battery");
        assert_eq!(groups[1].aspect, "price");
        assert!(req.texts.is_none() && req.text.is_none());
    }
}
