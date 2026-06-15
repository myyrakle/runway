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

    /// Batch analysis with a single aspect applied to every text. Thin wrapper over
    /// [`analyze_pairs`](Self::analyze_pairs).
    pub fn analyze_batch(&self, texts: &[String], aspect: &str) -> Result<Vec<SentimentResult>> {
        let aspects = vec![aspect.to_string(); texts.len()];
        self.analyze_pairs(texts, &aspects)
    }

    /// Per-row analysis preserving input order: each text is paired with its own aspect,
    /// so one batch can mix aspects. Mirrors the Python sort-by-length → chunk → forward
    /// → reorder pipeline. `aspects` must be parallel to `texts`.
    pub fn analyze_pairs(
        &self,
        texts: &[String],
        aspects: &[String],
    ) -> Result<Vec<SentimentResult>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        debug_assert_eq!(
            texts.len(),
            aspects.len(),
            "analyze_pairs requires one aspect per text"
        );

        // (original_index, text, aspect). Optionally sort by approximate length to keep
        // padding low within each chunk; the aspect travels with its text and original
        // order is restored at the end.
        let mut indexed: Vec<(usize, &String, &String)> = texts
            .iter()
            .zip(aspects.iter())
            .enumerate()
            .map(|(i, (t, a))| (i, t, a))
            .collect();
        if self.config.sort_batch_by_length {
            indexed.sort_by_key(|(_, t, a)| t.chars().count() + a.chars().count());
        }

        let mut ordered: Vec<Option<SentimentResult>> = (0..texts.len()).map(|_| None).collect();

        let timing = timing_enabled();
        let (mut tok_ms, mut fwd_ms) = (0.0_f64, 0.0_f64);
        let mut chunks_run = 0usize;

        for chunk in self.iter_chunks(&indexed) {
            let chunk_texts: Vec<String> = chunk.iter().map(|(_, t, _)| (*t).clone()).collect();
            let chunk_aspects: Vec<String> = chunk.iter().map(|(_, _, a)| (*a).clone()).collect();
            let (chunk_results, t_ms, f_ms) = self.run_chunk(&chunk_texts, &chunk_aspects)?;
            tok_ms += t_ms;
            fwd_ms += f_ms;
            chunks_run += 1;
            for ((orig_idx, _, _), result) in chunk.iter().zip(chunk_results) {
                ordered[*orig_idx] = Some(result);
            }
        }

        if timing {
            tracing::info!(
                texts = texts.len(),
                chunks = chunks_run,
                tokenize_ms = format!("{tok_ms:.1}"),
                forward_ms = format!("{fwd_ms:.1}"),
                "analyze_pairs timing"
            );
        }

        Ok(ordered.into_iter().flatten().collect())
    }

    /// Yield chunks bounded by INFERENCE_BATCH_SIZE and (optionally) MAX_BATCH_TOKENS.
    /// Each item carries its own aspect, used for the per-row token estimate.
    fn iter_chunks<'a>(
        &self,
        indexed: &'a [(usize, &'a String, &'a String)],
    ) -> Vec<Vec<(usize, &'a String, &'a String)>> {
        let batch_size = self.config.inference_batch_size.max(1);
        let mut chunks: Vec<Vec<(usize, &String, &String)>> = Vec::new();

        if self.config.max_batch_tokens == 0 {
            for window in indexed.chunks(batch_size) {
                chunks.push(window.to_vec());
            }
            return chunks;
        }

        let max_tokens = self.config.max_batch_tokens;
        let mut current: Vec<(usize, &String, &String)> = Vec::new();
        let mut current_max_tokens = 0usize;

        for item in indexed {
            let item_tokens = self.estimate_tokens(item.1, item.2.chars().count());
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
    /// `aspects` is parallel to `texts` within this chunk.
    fn run_chunk(
        &self,
        texts: &[String],
        aspects: &[String],
    ) -> Result<(Vec<SentimentResult>, f64, f64)> {
        let t0 = Instant::now();
        let enc = self.tokenizer.encode_pairs(texts, aspects)?;
        let tokenize_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Padding-waste audit: PAD tokens still cost GPU FLOPs (the mask only zeroes
        // attention contributions). fill = real / padded; low fill => GPU time burned on
        // padding => tighter token-budget chunking / outlier isolation pays off.
        if timing_enabled() {
            let real_tokens: i64 = enc.attention_mask.iter().map(|&m| m as i64).sum();
            let padded = enc.batch * enc.seq;
            let mean_real = real_tokens as f64 / enc.batch.max(1) as f64;
            let fill = if padded > 0 {
                real_tokens as f64 / padded as f64 * 100.0
            } else {
                0.0
            };
            tracing::info!(
                batch = enc.batch,
                seq = enc.seq,
                mean_real_tokens = format!("{mean_real:.1}"),
                fill_pct = format!("{fill:.1}"),
                "run_chunk padding"
            );
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Backend;
    use crate::test_support::{token_id, write_fixture_tokenizer};

    /// Backend whose verdict depends on a sentinel token in each row: "bad" -> negative,
    /// "good" -> positive. Lets a test assert which result row came from which input.
    struct SentinelBackend;

    impl Backend for SentinelBackend {
        fn infer(
            &self,
            input_ids: &[i32],
            _attention_mask: &[i32],
            batch: usize,
            seq: usize,
        ) -> Result<Vec<f32>> {
            let (good, bad) = (token_id("good"), token_id("bad"));
            let mut out = Vec::with_capacity(batch * 3);
            for b in 0..batch {
                let row = &input_ids[b * seq..(b + 1) * seq];
                if row.contains(&bad) {
                    out.extend_from_slice(&[5.0, 0.0, 0.0]); // negative
                } else if row.contains(&good) {
                    out.extend_from_slice(&[0.0, 0.0, 5.0]); // positive
                } else {
                    out.extend_from_slice(&[0.0, 5.0, 0.0]); // neutral
                }
            }
            Ok(out)
        }

        fn num_labels(&self) -> usize {
            3
        }

        fn name(&self) -> &'static str {
            "sentinel"
        }
    }

    fn test_config(sort: bool, batch_size: usize) -> Config {
        Config {
            model_name: "test".to_string(),
            tokenizer_path: "test".to_string(),
            engine_path: "test".to_string(),
            negative_threshold: 0.6,
            max_length: 512,
            max_batch_items: 1024,
            inference_batch_size: batch_size,
            max_batch_tokens: 0,
            sort_batch_by_length: sort,
            pad_to_multiple_of: 0,
            port: 8080,
            force_stub: true,
        }
    }

    fn engine(sort: bool, batch_size: usize) -> Engine {
        let path = write_fixture_tokenizer("sentiment_rs_inf_pairs.json");
        let tok = crate::tokenizer::load_or_fail(path.to_str().unwrap(), 512, 0).unwrap();
        Engine::new(test_config(sort, batch_size), tok, Box::new(SentinelBackend))
    }

    #[test]
    fn analyze_pairs_returns_results_in_input_order_despite_length_sort() {
        // Input order is [good, bad]; length-sort reorders to [bad, good] internally.
        // Results must come back mapped to the ORIGINAL order.
        let eng = engine(true, 10);
        let texts = vec!["good".to_string(), "bad".to_string()];
        let aspects = vec!["screen".to_string(), "battery".to_string()];

        let out = eng.analyze_pairs(&texts, &aspects).unwrap();

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].sentiment, "positive"); // "good"
        assert_eq!(out[1].sentiment, "negative"); // "bad"
    }

    #[test]
    fn analyze_pairs_maps_each_row_across_separate_chunks() {
        // batch_size 1 forces one chunk per row; reassembly by index must still hold.
        let eng = engine(false, 1);
        let texts = vec!["bad".to_string(), "good".to_string(), "bad".to_string()];
        let aspects = vec![
            "battery".to_string(),
            "screen".to_string(),
            "price".to_string(),
        ];

        let out = eng.analyze_pairs(&texts, &aspects).unwrap();

        let labels: Vec<&str> = out.iter().map(|r| r.sentiment.as_str()).collect();
        assert_eq!(labels, vec!["negative", "positive", "negative"]);
    }

    #[test]
    fn analyze_batch_applies_one_aspect_and_preserves_order() {
        let eng = engine(true, 10);
        let texts = vec!["good".to_string(), "bad".to_string()];

        let out = eng.analyze_batch(&texts, "overall").unwrap();

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].sentiment, "positive");
        assert_eq!(out[1].sentiment, "negative");
    }
}
