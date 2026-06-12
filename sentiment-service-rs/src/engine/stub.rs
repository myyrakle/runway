//! CPU stub backend: no GPU, no TensorRT. Returns zero logits (=> uniform softmax),
//! so the HTTP/tokenizer/batching stack can be exercised end to end on any machine.
//! NOT a real model — predictions are meaningless; it exists for local dev and CI.

use anyhow::Result;

use super::Backend;

pub struct StubBackend {
    num_labels: usize,
}

impl StubBackend {
    pub fn new(num_labels: usize) -> Self {
        Self { num_labels }
    }
}

impl Backend for StubBackend {
    fn infer(
        &self,
        _input_ids: &[i32],
        _attention_mask: &[i32],
        batch: usize,
        _seq: usize,
    ) -> Result<Vec<f32>> {
        Ok(vec![0.0_f32; batch * self.num_labels])
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn name(&self) -> &'static str {
        "cpu-stub"
    }
}
