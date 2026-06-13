//! Inference orchestration: batching + chunking + softmax + result assembly.
//! Ports app/model.py's analyze_batch / _iter_inference_chunks / _analyze_batch_chunk.

use std::sync::OnceLock;
use std::time::Instant;

use anyhow::Result;

use crate::config::Config;
use crate::engine::Backend;
use crate::schemas::{Probs, SentimentResult};
use crate::tokenizer::AbsaTokenizer;

const LABELS: [&str; 3] = ["negative", "neutral", "positive"];

/// Set SENTIMENT_TIMING=1 to log a per-request tokenize-vs-forward breakdown. Off by
/// default (one env read, cached) so it costs nothing in production.
fn timing_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("SENTIMENT_TIMING")
            .map(|v| !matches!(v.as_str(), "" | "0" | "false" | "no"))
            .unwrap_or(false)
    })
}

pub struct Engine {
    pub config: Config,
    tokenizer: AbsaTokenizer,
    backend: Box<dyn Backend>,
}

impl Engine {
    pub fn new(config: Config, tokenizer: AbsaTokenizer, backend: Box<dyn Backend>) -> Self {
        Self {
            config,
            tokenizer,
            backend,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn analyze(&self, text: &str, aspect: &str) -> Result<SentimentResult> {
        let mut out = self.analyze_batch(std::slice::from_ref(&text.to_string()), aspect)?;
        Ok(out.remove(0))
    }

    /// Batch analysis preserving input order. Mirrors the Python sort-by-length →
    /// chunk → forward → reorder pipeline.
    pub fn analyze_batch(&self, texts: &[String], aspect: &str) -> Result<Vec<SentimentResult>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // (original_index, text). Optionally sort by approximate length to keep padding
        // low within each chunk; original order is restored at the end.
        let mut indexed: Vec<(usize, &String)> = texts.iter().enumerate().collect();
        if self.config.sort_batch_by_length {
            let alen = aspect.chars().count();
            indexed.sort_by_key(|(_, t)| t.chars().count() + alen);
        }

        let mut ordered: Vec<Option<SentimentResult>> = (0..texts.len()).map(|_| None).collect();

        let timing = timing_enabled();
        let (mut tok_ms, mut fwd_ms) = (0.0_f64, 0.0_f64);
        let mut chunks_run = 0usize;

        for chunk in self.iter_chunks(&indexed, aspect) {
            let chunk_texts: Vec<String> = chunk.iter().map(|(_, t)| (*t).clone()).collect();
            let (chunk_results, t_ms, f_ms) = self.run_chunk(&chunk_texts, aspect)?;
            tok_ms += t_ms;
            fwd_ms += f_ms;
            chunks_run += 1;
            for ((orig_idx, _), result) in chunk.iter().zip(chunk_results.into_iter()) {
                ordered[*orig_idx] = Some(result);
            }
        }

        if timing {
            tracing::info!(
                texts = texts.len(),
                chunks = chunks_run,
                tokenize_ms = format!("{tok_ms:.1}"),
                forward_ms = format!("{fwd_ms:.1}"),
                "analyze_batch timing"
            );
        }

        Ok(ordered.into_iter().flatten().collect())
    }

    /// Yield chunks bounded by INFERENCE_BATCH_SIZE and (optionally) MAX_BATCH_TOKENS.
    fn iter_chunks<'a>(
        &self,
        indexed: &'a [(usize, &'a String)],
        aspect: &str,
    ) -> Vec<Vec<(usize, &'a String)>> {
        let batch_size = self.config.inference_batch_size.max(1);
        let mut chunks: Vec<Vec<(usize, &String)>> = Vec::new();

        if self.config.max_batch_tokens == 0 {
            for window in indexed.chunks(batch_size) {
                chunks.push(window.to_vec());
            }
            return chunks;
        }

        let alen = aspect.chars().count();
        let max_tokens = self.config.max_batch_tokens;
        let mut current: Vec<(usize, &String)> = Vec::new();
        let mut current_max_tokens = 0usize;

        for item in indexed {
            let item_tokens = self.estimate_tokens(item.1, alen);
            let next_max = current_max_tokens.max(item_tokens);
            let next_size = current.len() + 1;
            let exceed_size = next_size > batch_size;
            let exceed_tokens = next_size * next_max > max_tokens;

            if !current.is_empty() && (exceed_size || exceed_tokens) {
                chunks.push(std::mem::take(&mut current));
                current_max_tokens = 0;
            }
            current.push(*item);
            current_max_tokens = current_max_tokens.max(item_tokens);
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        chunks
    }

    fn estimate_tokens(&self, text: &str, aspect_len: usize) -> usize {
        // Cheap upper-ish estimate before real tokenization, matching the Python heuristic
        // min(MAX_LENGTH, len(text) + len(aspect) + 3).
        (text.chars().count() + aspect_len + 3).min(self.config.max_length)
    }

    /// One bounded forward pass: tokenize → backend infer → softmax → results.
    /// Returns (results, tokenize_ms, forward_ms) for optional timing instrumentation.
    fn run_chunk(&self, texts: &[String], aspect: &str) -> Result<(Vec<SentimentResult>, f64, f64)> {
        let t0 = Instant::now();
        let enc = self.tokenizer.encode_pairs(texts, aspect)?;
        let tokenize_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let logits = self
            .backend
            .infer(&enc.input_ids, &enc.attention_mask, enc.batch, enc.seq)?;
        let forward_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let num_labels = self.backend.num_labels();
        let threshold = self.config.negative_threshold;
        let mut results = Vec::with_capacity(enc.batch);

        for row in logits.chunks(num_labels) {
            let probs = softmax(row);
            // First-max index, matching Python's `prob.index(max(prob))` (argmax that
            // returns the FIRST occurrence on ties).
            let mut label_idx = 0usize;
            let mut confidence = probs.first().copied().unwrap_or(0.0);
            for (i, &p) in probs.iter().enumerate().skip(1) {
                if p > confidence {
                    confidence = p;
                    label_idx = i;
                }
            }

            // Match the Python label map (index 0=negative, 1=neutral, 2=positive).
            let neg = probs.first().copied().unwrap_or(0.0);
            let neu = probs.get(1).copied().unwrap_or(0.0);
            let pos = probs.get(2).copied().unwrap_or(0.0);

            results.push(SentimentResult {
                sentiment: LABELS.get(label_idx).copied().unwrap_or("unknown").to_string(),
                confidence,
                is_negative: label_idx == 0 && confidence >= threshold,
                probs: Probs {
                    negative: neg,
                    neutral: neu,
                    positive: pos,
                },
            });
        }
        Ok((results, tokenize_ms, forward_ms))
    }
}

/// Numerically stable softmax over a small logit row.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}
