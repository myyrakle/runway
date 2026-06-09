"""DeBERTa ABSA model wrapper.

ABSA input format [CLS] text [SEP] aspect [SEP], 3-class label map.
Supports precision variants: fp32 (default), fp16 (CUDA), int8 (dynamic quant, CPU).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import DEVICE, MAX_LENGTH, MODEL_NAME, NEGATIVE_THRESHOLD

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
        print(f"[Model] Loaded precision={precision} on {self.device}")

    def _tokenize(self, texts, aspects):
        inputs = self.tokenizer(
            texts,
            aspects,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def analyze(self, text: str, aspect: str = "overall") -> dict:
        """Single-text sentiment analysis."""
        inputs = self._tokenize(text, aspect)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
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

        inputs = self._tokenize(texts, [aspect] * len(texts))

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1)

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
