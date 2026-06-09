from __future__ import annotations

import unittest

from pydantic import ValidationError

from tests.fakes import install_fake_ml_modules

fake_torch = install_fake_ml_modules()

import app.model as model_module
import app.schemas as schema_module
from app.model import DeBERTaABSA
from app.schemas import BatchAnalyzeRequest


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts, aspects, **kwargs):
        batch_texts = list(texts)
        self.calls.append(batch_texts)
        batch_size = len(batch_texts)
        return {
            "input_ids": fake_torch.ones((batch_size, 4), dtype=fake_torch.long),
            "attention_mask": fake_torch.ones((batch_size, 4), dtype=fake_torch.long),
        }


class _FakeModel:
    def __call__(self, **inputs):
        batch_size = inputs["input_ids"].shape[0]
        logits = fake_torch.tensor([[0.0, 0.1, 3.0]] * batch_size)
        return type("Output", (), {"logits": logits})()


class BatchInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_batch_size = getattr(model_module, "INFERENCE_BATCH_SIZE", None)
        model_module.INFERENCE_BATCH_SIZE = 2

    def tearDown(self) -> None:
        if self.original_batch_size is None:
            delattr(model_module, "INFERENCE_BATCH_SIZE")
        else:
            model_module.INFERENCE_BATCH_SIZE = self.original_batch_size

    def _fake_absa(self) -> DeBERTaABSA:
        absa = object.__new__(DeBERTaABSA)
        absa.precision = "fp32"
        absa.device = "cpu"
        absa.tokenizer = _FakeTokenizer()
        absa.model = _FakeModel()
        return absa

    def test_analyze_batch_splits_large_inputs_into_inference_chunks(self) -> None:
        absa = self._fake_absa()

        results = absa.analyze_batch(["a", "b", "c", "d", "e"], "overall")

        self.assertEqual(len(results), 5)
        self.assertEqual(absa.tokenizer.calls, [["a", "b"], ["c", "d"], ["e"]])
        self.assertTrue(all(result["sentiment"] == "positive" for result in results))

    def test_batch_request_rejects_more_than_max_batch_items(self) -> None:
        self.assertTrue(hasattr(schema_module, "MAX_BATCH_ITEMS"))
        with self.assertRaises(ValidationError):
            BatchAnalyzeRequest(texts=["x"] * (schema_module.MAX_BATCH_ITEMS + 1))
