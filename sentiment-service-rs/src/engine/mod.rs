//! Inference backend abstraction. The HTTP/tokenizer/batching layers only ever see
//! the `Backend` trait, so the GPU-specific TensorRT path stays isolated behind the
//! `trt` cargo feature. Without that feature (or with STUB=1) the CPU `stub` backend
//! is used, which lets the whole service compile and run without CUDA/TensorRT.

use anyhow::Result;

pub mod stub;
#[cfg(feature = "trt")]
pub mod trt;

/// One forward pass. `input_ids`/`attention_mask` are row-major [batch, seq] int32.
/// Returns logits as a flat row-major [batch, num_labels] f32 vector.
pub trait Backend: Send + Sync {
    fn infer(
        &self,
        input_ids: &[i32],
        attention_mask: &[i32],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<f32>>;

    fn num_labels(&self) -> usize;

    /// Human-readable backend name, surfaced in /health as `device`.
    fn name(&self) -> &'static str;
}

/// Select + construct the backend. Prefers TensorRT when compiled in and not forced
/// off; otherwise the CPU stub.
pub fn build(engine_path: &str, force_stub: bool) -> Result<Box<dyn Backend>> {
    #[cfg(feature = "trt")]
    {
        if !force_stub {
            let be = trt::TrtBackend::load(engine_path)?;
            return Ok(Box::new(be));
        }
    }
    #[cfg(not(feature = "trt"))]
    {
        let _ = engine_path;
        if !force_stub {
            tracing::warn!(
                "built without the `trt` feature; using CPU stub backend \
                 (outputs are not real predictions)"
            );
        }
    }
    Ok(Box::new(stub::StubBackend::new(3)))
}
