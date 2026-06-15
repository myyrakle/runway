//! Test-only helpers. Builds a tiny real `tokenizer.json` on disk so the
//! tokenizer/inference layers can be exercised without the full DeBERTa tokenizer
//! that ships baked into the deployment image.

use std::collections::HashMap;
use std::path::PathBuf;

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::Tokenizer;

/// Vocabulary words the fixture knows; everything else maps to `[UNK]`.
const VOCAB: &[&str] = &[
    "[PAD]", "[CLS]", "[SEP]", "[UNK]", "good", "bad", "battery", "screen", "price",
];

/// Build a minimal WordLevel tokenizer that emits `[CLS] A [SEP] B [SEP]` for pairs,
/// save it to a temp path, and return that path. The token id for an aspect differs
/// per aspect, so a per-row aspect change is observable in the encoded ids.
pub fn write_fixture_tokenizer(name: &str) -> PathBuf {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, tok) in VOCAB.iter().enumerate() {
        vocab.insert((*tok).to_string(), i as u32);
    }

    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .expect("build wordlevel");

    let mut tk = Tokenizer::new(model);
    tk.with_pre_tokenizer(Some(Whitespace {}));

    let post = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .unwrap()
        .try_pair("[CLS] $A:0 [SEP]:0 $B:1 [SEP]:1")
        .unwrap()
        .special_tokens(vec![("[CLS]", 1u32), ("[SEP]", 2u32)])
        .build()
        .unwrap();
    tk.with_post_processor(Some(post));

    let path = std::env::temp_dir().join(name);
    tk.save(&path, false).expect("save tokenizer fixture");
    path
}

/// Token id for a vocab word, for asserting which aspect landed in a given row.
pub fn token_id(word: &str) -> i32 {
    VOCAB
        .iter()
        .position(|w| *w == word)
        .expect("word in fixture vocab") as i32
}
