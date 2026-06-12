//! Runtime configuration, read from environment variables. Mirrors the knobs in the
//! Python service's `app/config.py` that are relevant to the native-TensorRT path.
//! (PyTorch/ONNX-only knobs and the multi-precision machinery are intentionally
//! dropped: this binary serves exactly one prebuilt fp16 engine.)

use std::env;

fn env_string(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    match env::var(key) {
        Ok(v) => !matches!(v.to_ascii_lowercase().as_str(), "0" | "false" | "no" | ""),
        Err(_) => default,
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub model_name: String,
    /// Path to the HF `tokenizer.json` (fast tokenizer), baked into the image.
    pub tokenizer_path: String,
    /// Path to the serialized TensorRT engine (`.plan`).
    pub engine_path: String,

    pub negative_threshold: f32,
    pub max_length: usize,
    pub max_batch_items: usize,
    pub inference_batch_size: usize,
    /// Approximate padded-token budget per forward pass; 0 disables token-budget
    /// chunking (then only `inference_batch_size` bounds a chunk).
    pub max_batch_tokens: usize,
    pub sort_batch_by_length: bool,
    /// Pad sequence length up to a multiple of this. 0 => auto (8, matching the
    /// Python service's fp16 default) so TensorRT sees few distinct shapes.
    pub pad_to_multiple_of: usize,

    pub port: u16,
    /// Force the CPU stub backend even when the `trt` feature is compiled in.
    pub force_stub: bool,
}

impl Config {
    pub fn from_env() -> Self {
        let pad = env_usize("PAD_TO_MULTIPLE_OF", 0);
        Config {
            model_name: env_string("MODEL_NAME", "yangheng/deberta-v3-base-absa-v1.1"),
            tokenizer_path: env_string("TOKENIZER_PATH", "artifacts/tokenizer.json"),
            engine_path: env_string("TRT_ENGINE_PATH", "artifacts/model.plan"),
            negative_threshold: env_f32("NEGATIVE_THRESHOLD", 0.6),
            max_length: env_usize("MAX_LENGTH", 512),
            max_batch_items: env_usize("MAX_BATCH_ITEMS", 1024),
            inference_batch_size: env_usize("INFERENCE_BATCH_SIZE", 64),
            max_batch_tokens: env_usize("MAX_BATCH_TOKENS", 0),
            sort_batch_by_length: env_bool("SORT_BATCH_BY_LENGTH", true),
            pad_to_multiple_of: if pad > 0 { pad } else { 8 },
            port: env_usize("PORT", 8080) as u16,
            force_stub: env_bool("STUB", false),
        }
    }
}
