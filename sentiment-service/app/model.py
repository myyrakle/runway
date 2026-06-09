"""DeBERTa ABSA model wrapper.

ABSA input format [CLS] text [SEP] aspect [SEP], 3-class label map.
Supports precision variants: fp32 (default), fp16 (CUDA), int8 (dynamic quant, CPU).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import (
    DEVICE,
    INFERENCE_BATCH_SIZE,
    MAX_LENGTH,
    MODEL_NAME,
    NEGATIVE_THRESHOLD,
    PAD_TO_MULTIPLE_OF,
    SORT_BATCH_BY_LENGTH,
)

PRECISIONS = ("fp32", "fp16", "int8")


class DeBERTaABSA:
    """DeBERTa-based Aspect-Based Sentiment Analysis model."""

    # ABSA labels: 0=negative, 1=neutral, 2=positive
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(self, precision: str = "fp32") -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"Unknown precision {precision!r}, expected {PRECISIONS}")
        self.precision = precision

        print(f"[Model] Loading DeBERTa ABSA ({MODEL_NAME}) precision={precision}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()

        if precision == "fp16":
            if DEVICE != "cuda":
                raise ValueError("fp16 precision requires a CUDA device")
            self.device = "cuda"
            model = model.half().to(self.device)
        elif precision == "int8":
            # Dynamic quantization targets CPU Linear layers.
            self.device = "cpu"
            model = model.to(self.device)
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:  # fp32
            self.device = DEVICE
            model = model.to(self.device)

        self.model = model
        self.pad_to_multiple_of = self._resolve_pad_to_multiple_of()
        print(f"[Model] Loaded precision={precision} on {self.device}")

    def _resolve_pad_to_multiple_of(self) -> int | None:
        if PAD_TO_MULTIPLE_OF > 0:
            return PAD_TO_MULTIPLE_OF
        if self.precision == "fp16" and self.device == "cuda":
            return 8
        return None

    def _tokenize(self, texts, aspects):
        kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": MAX_LENGTH,
            "padding": True,
        }
        pad_to_multiple_of = getattr(
            self, "pad_to_multiple_of", self._resolve_pad_to_multiple_of()
        )
        if pad_to_multiple_of is not None:
            kwargs["pad_to_multiple_of"] = pad_to_multiple_of

        inputs = self.tokenizer(texts, aspects, **kwargs)
        if self.device == "cuda":
            return {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        return {k: v.to(self.device) for k, v in inputs.items()}

    def analyze(self, text: str, aspect: str = "overall") -> dict:
        """Single-text sentiment analysis."""
        inputs = self._tokenize(text, aspect)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu()
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label].item()

        sentiment = self.label_map[pred_label]
        is_negative = pred_label == 0 and confidence >= NEGATIVE_THRESHOLD

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "is_negative": is_negative,
            "probs": {
                "negative": probs[0][0].item(),
                "neutral": probs[0][1].item(),
                "positive": probs[0][2].item(),
            },
        }

    def analyze_batch(self, texts: list[str], aspect: str = "overall") -> list[dict]:
        """Batch sentiment analysis."""
        if not texts:
            return []

        indexed_texts = list(enumerate(texts))
        if SORT_BATCH_BY_LENGTH:
            indexed_texts.sort(key=lambda item: len(item[1]) + len(aspect))

        ordered_results: list[dict | None] = [None] * len(texts)
        for start in range(0, len(indexed_texts), INFERENCE_BATCH_SIZE):
            chunk = indexed_texts[start:start + INFERENCE_BATCH_SIZE]
            chunk_indices = [index for index, _ in chunk]
            chunk_texts = [text for _, text in chunk]
            chunk_results = self._analyze_batch_chunk(chunk_texts, aspect)
            for index, result in zip(chunk_indices, chunk_results):
                ordered_results[index] = result
        return [result for result in ordered_results if result is not None]

    def _analyze_batch_chunk(self, texts: list[str], aspect: str) -> list[dict]:
        """Run one bounded model forward pass."""
        inputs = self._tokenize(texts, [aspect] * len(texts))

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu()
            pred_labels = torch.argmax(probs, dim=-1).detach().cpu()

        results = []
        for label, prob in zip(pred_labels, probs):
            label_idx = label.item()
            confidence = prob[label_idx].item()
            sentiment = self.label_map[label_idx]
            is_negative = label_idx == 0 and confidence >= NEGATIVE_THRESHOLD

            results.append({
                "sentiment": sentiment,
                "confidence": confidence,
                "is_negative": is_negative,
                "probs": {
                    "negative": prob[0].item(),
                    "neutral": prob[1].item(),
                    "positive": prob[2].item(),
                },
            })

        return results
