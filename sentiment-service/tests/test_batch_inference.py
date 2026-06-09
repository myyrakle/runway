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
        self.kwargs: list[dict] = []

    def __call__(self, texts, aspects, **kwargs):
        batch_texts = list(texts)
        self.calls.append(batch_texts)
        self.kwargs.append(kwargs)
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
        self.original_sort = getattr(model_module, "SORT_BATCH_BY_LENGTH", None)
        model_module.INFERENCE_BATCH_SIZE = 2
        model_module.SORT_BATCH_BY_LENGTH = False

    def tearDown(self) -> None:
        if self.original_batch_size is None:
            delattr(model_module, "INFERENCE_BATCH_SIZE")
        else:
            model_module.INFERENCE_BATCH_SIZE = self.original_batch_size
        if self.original_sort is None:
            delattr(model_module, "SORT_BATCH_BY_LENGTH")
        else:
            model_module.SORT_BATCH_BY_LENGTH = self.original_sort

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

    def test_analyze_batch_can_sort_chunks_by_length_while_preserving_result_count(self) -> None:
        model_module.SORT_BATCH_BY_LENGTH = True
        absa = self._fake_absa()

        results = absa.analyze_batch(["aaaa", "b", "cccccc", "dd"], "overall")

        self.assertEqual(len(results), 4)
        self.assertEqual(absa.tokenizer.calls, [["b", "dd"], ["aaaa", "cccccc"]])

    def test_fp16_cuda_tokenization_pads_to_tensor_core_multiple(self) -> None:
        absa = self._fake_absa()
        absa.precision = "fp16"
        absa.device = "cuda"

        absa.analyze_batch(["a", "b"], "overall")

        self.assertEqual(absa.tokenizer.kwargs[0]["pad_to_multiple_of"], 8)

    def test_batch_request_rejects_more_than_max_batch_items(self) -> None:
        self.assertTrue(hasattr(schema_module, "MAX_BATCH_ITEMS"))
        with self.assertRaises(ValidationError):
            BatchAnalyzeRequest(texts=["x"] * (schema_module.MAX_BATCH_ITEMS + 1))
