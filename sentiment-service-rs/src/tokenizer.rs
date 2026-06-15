//! Tokenization via the HF `tokenizers` crate (native Rust, no Python/transformers).
//! Loads the fast `tokenizer.json` baked into the image and encodes (text, aspect)
//! pairs exactly like the Python service: truncation to `max_length`, dynamic padding
//! to the longest sequence in the batch, optionally rounded up to a multiple.
//!
//! Note: the engine takes only {input_ids, attention_mask}; token_type_ids are
//! intentionally not produced (deberta-v3-base has type_vocab_size=0 and the ONNX
//! export drops them — see scripts/export_onnx.py in the Python service).

use anyhow::{anyhow, Context, Result};
use tokenizers::{
    EncodeInput, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy,
};

pub struct Encoded {
    pub input_ids: Vec<i32>,
    pub attention_mask: Vec<i32>,
    pub batch: usize,
    pub seq: usize,
}

pub struct AbsaTokenizer {
    inner: Tokenizer,
}


impl AbsaTokenizer {
    pub fn load(path: &str, max_length: usize, pad_to_multiple_of: usize) -> Result<Self> {
        let mut inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("failed to load tokenizer {path}: {e}"))?;

        inner
            .with_truncation(Some(TruncationParams {
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .map_err(|e| anyhow!("with_truncation failed: {e}"))?;

        // Start from the tokenizer's own padding config (pad id/token) if present, so
        // we use the model's real [PAD] id; only override the strategy + multiple.
        let mut padding = inner.get_padding().cloned().unwrap_or_default();
        padding.strategy = PaddingStrategy::BatchLongest;
        padding.pad_to_multiple_of = if pad_to_multiple_of > 0 {
            Some(pad_to_multiple_of)
        } else {
            None
        };
        inner.with_padding(Some(padding));

        Ok(Self { inner })
    }

    /// Encode a batch of (text, aspect) pairs into padded int32 tensors. `aspects` is
    /// parallel to `texts`, so each row is paired with its own aspect (one batch can
    /// mix aspects). Use [`encode_batch`](Self::encode_batch) for a shared aspect.
    pub fn encode_pairs(&self, texts: &[String], aspects: &[String]) -> Result<Encoded> {
        if texts.is_empty() {
            return Ok(Encoded {
                input_ids: Vec::new(),
                attention_mask: Vec::new(),
                batch: 0,
                seq: 0,
            });
        }
        debug_assert_eq!(
            texts.len(),
            aspects.len(),
            "encode_pairs requires one aspect per text"
        );

        let inputs: Vec<EncodeInput> = texts
            .iter()
            .zip(aspects.iter())
            .map(|(t, a)| EncodeInput::Dual(t.clone().into(), a.clone().into()))
            .collect();

        let encodings = self
            .inner
            .encode_batch(inputs, true)
            .map_err(|e| anyhow!("encode_batch failed: {e}"))?;

        let batch = encodings.len();
        let seq = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

        let mut input_ids = Vec::with_capacity(batch * seq);
        let mut attention_mask = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            // BatchLongest padding guarantees equal length across the batch.
            debug_assert_eq!(ids.len(), seq);
            debug_assert_eq!(mask.len(), seq);
            input_ids.extend(ids.iter().map(|&v| v as i32));
            attention_mask.extend(mask.iter().map(|&v| v as i32));
        }

        Ok(Encoded {
            input_ids,
            attention_mask,
            batch,
            seq,
        })
    }
}

/// Load + sanity-check the tokenizer at startup so a bad path fails fast.
pub fn load_or_fail(path: &str, max_length: usize, pad_to_multiple_of: usize) -> Result<AbsaTokenizer> {
    AbsaTokenizer::load(path, max_length, pad_to_multiple_of)
        .with_context(|| format!("tokenizer init failed for {path}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{token_id, write_fixture_tokenizer};

    fn fixture() -> AbsaTokenizer {
        let path = write_fixture_tokenizer("sentiment_rs_tok_encode_pairs.json");
        AbsaTokenizer::load(path.to_str().unwrap(), 512, 0).unwrap()
    }

    #[test]
    fn encode_pairs_applies_each_rows_own_aspect() {
        let tk = fixture();
        let texts = vec!["good".to_string(), "good".to_string()];
        let aspects = vec!["battery".to_string(), "screen".to_string()];

        let enc = tk.encode_pairs(&texts, &aspects).unwrap();

        assert_eq!(enc.batch, 2);
        // Layout per row: [CLS] good [SEP] <aspect> [SEP] -> aspect at column 3.
        let aspect_col = 3;
        let row0 = &enc.input_ids[0..enc.seq];
        let row1 = &enc.input_ids[enc.seq..2 * enc.seq];
        assert_eq!(row0[aspect_col], token_id("battery"));
        assert_eq!(row1[aspect_col], token_id("screen"));
    }

    #[test]
    fn encode_pairs_swapping_aspects_swaps_encoded_rows() {
        let tk = fixture();
        let texts = vec!["good".to_string(), "good".to_string()];

        let a = tk
            .encode_pairs(&texts, &["screen".to_string(), "battery".to_string()])
            .unwrap();
        let aspect_col = 3;
        assert_eq!(a.input_ids[aspect_col], token_id("screen"));
        assert_eq!(a.input_ids[a.seq + aspect_col], token_id("battery"));
    }
}
