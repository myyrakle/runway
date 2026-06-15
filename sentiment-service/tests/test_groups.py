"""Tests for the per-(text, aspect) `groups` batch form on /invocations."""
from __future__ import annotations

import asyncio
import unittest

from pydantic import ValidationError

from tests.fakes import install_fake_fastapi_modules, install_fake_ml_modules

install_fake_ml_modules()
install_fake_fastapi_modules()

import app.main as main_module
import app.model as model_module
import app.schemas as schema_module
from app.model import DeBERTaABSA
from app.schemas import InvocationRequest

fake_torch = install_fake_ml_modules()


class GroupSchemaTests(unittest.TestCase):
    def test_groups_accepts_aspect_text_pairs(self) -> None:
        req = InvocationRequest(
            groups=[
                {"aspect": "battery", "text": "battery life is terrible"},
                {"aspect": "screen", "text": "the screen is great"},
            ]
        )
        self.assertEqual(req.groups[0].aspect, "battery")
        self.assertEqual(req.groups[0].text, "battery life is terrible")
        self.assertEqual(req.groups[1].aspect, "screen")

    def test_group_aspect_defaults_to_overall(self) -> None:
        req = InvocationRequest(groups=[{"text": "no aspect given"}])
        self.assertEqual(req.groups[0].aspect, "overall")

    def test_groups_is_mutually_exclusive_with_texts(self) -> None:
        with self.assertRaises(ValidationError):
            InvocationRequest(
                texts=["a"], groups=[{"aspect": "x", "text": "b"}]
            )

    def test_groups_alone_satisfies_exactly_one_rule(self) -> None:
        # groups counts as the single provided payload; no text/texts/instances needed.
        req = InvocationRequest(groups=[{"aspect": "x", "text": "b"}])
        self.assertIsNotNone(req.groups)

    def test_groups_rejects_more_than_max_batch_items(self) -> None:
        too_many = [{"aspect": "x", "text": "t"}] * (schema_module.MAX_BATCH_ITEMS + 1)
        with self.assertRaises(ValidationError):
            InvocationRequest(groups=too_many)


class _PairRecordingTokenizer:
    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.aspect_calls: list[list[str]] = []

    def __call__(self, texts, aspects, **kwargs):
        batch_texts = list(texts)
        self.text_calls.append(batch_texts)
        self.aspect_calls.append(list(aspects))
        return {
            "input_ids": fake_torch.ones((len(batch_texts), 4), dtype=fake_torch.long),
            "attention_mask": fake_torch.ones(
                (len(batch_texts), 4), dtype=fake_torch.long
            ),
        }


class _PairFakeModel:
    def __call__(self, **inputs):
        batch_size = inputs["input_ids"].shape[0]
        logits = fake_torch.tensor([[0.0, 0.1, 3.0]] * batch_size)
        return type("Output", (), {"logits": logits})()


class AnalyzePairsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = (
            model_module.INFERENCE_BATCH_SIZE,
            model_module.SORT_BATCH_BY_LENGTH,
            model_module.MAX_BATCH_TOKENS,
        )
        model_module.INFERENCE_BATCH_SIZE = 10
        model_module.SORT_BATCH_BY_LENGTH = False
        model_module.MAX_BATCH_TOKENS = 0

    def tearDown(self) -> None:
        (
            model_module.INFERENCE_BATCH_SIZE,
            model_module.SORT_BATCH_BY_LENGTH,
            model_module.MAX_BATCH_TOKENS,
        ) = self._orig

    def _fake_absa(self) -> DeBERTaABSA:
        absa = object.__new__(DeBERTaABSA)
        absa.precision = "fp32"
        absa.device = "cpu"
        absa.tokenizer = _PairRecordingTokenizer()
        absa.model = _PairFakeModel()
        return absa

    def test_analyze_pairs_tokenizes_each_text_with_its_own_aspect(self) -> None:
        absa = self._fake_absa()

        results = absa.analyze_pairs(
            ["battery is bad", "battery is bad", "screen is great"],
            ["battery", "price", "screen"],
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(
            absa.tokenizer.text_calls,
            [["battery is bad", "battery is bad", "screen is great"]],
        )
        self.assertEqual(absa.tokenizer.aspect_calls, [["battery", "price", "screen"]])

    def test_analyze_pairs_keeps_aspect_paired_through_length_sort(self) -> None:
        model_module.SORT_BATCH_BY_LENGTH = True
        absa = self._fake_absa()

        # "b"+"B" (len 2) sorts before "aaaa"+"A" (len 5); the aspect must travel
        # with its text, and results must come back in the original order.
        results = absa.analyze_pairs(["aaaa", "b"], ["A", "B"])

        self.assertEqual(len(results), 2)
        self.assertEqual(absa.tokenizer.text_calls, [["b", "aaaa"]])
        self.assertEqual(absa.tokenizer.aspect_calls, [["B", "A"]])

    def test_analyze_batch_still_applies_single_aspect_to_all(self) -> None:
        absa = self._fake_absa()

        absa.analyze_batch(["a", "b", "c"], "overall")

        self.assertEqual(absa.tokenizer.aspect_calls, [["overall", "overall", "overall"]])


class _PairRecordingTokenizer:
    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.aspect_calls: list[list[str]] = []

    def __call__(self, texts, aspects, **kwargs):
        batch_texts = list(texts)
        self.text_calls.append(batch_texts)
        self.aspect_calls.append(list(aspects))
        return {
            "input_ids": fake_torch.ones((len(batch_texts), 4), dtype=fake_torch.long),
            "attention_mask": fake_torch.ones(
                (len(batch_texts), 4), dtype=fake_torch.long
            ),
        }


class _PairFakeModel:
    def __call__(self, **inputs):
        batch_size = inputs["input_ids"].shape[0]
        logits = fake_torch.tensor([[0.0, 0.1, 3.0]] * batch_size)
        return type("Output", (), {"logits": logits})()


class AnalyzePairsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = (
            model_module.INFERENCE_BATCH_SIZE,
            model_module.SORT_BATCH_BY_LENGTH,
            model_module.MAX_BATCH_TOKENS,
        )
        model_module.INFERENCE_BATCH_SIZE = 10
        model_module.SORT_BATCH_BY_LENGTH = False
        model_module.MAX_BATCH_TOKENS = 0

    def tearDown(self) -> None:
        (
            model_module.INFERENCE_BATCH_SIZE,
            model_module.SORT_BATCH_BY_LENGTH,
            model_module.MAX_BATCH_TOKENS,
        ) = self._orig

    def _fake_absa(self) -> DeBERTaABSA:
        absa = object.__new__(DeBERTaABSA)
        absa.precision = "fp32"
        absa.device = "cpu"
        absa.tokenizer = _PairRecordingTokenizer()
        absa.model = _PairFakeModel()
        return absa

    def test_analyze_pairs_tokenizes_each_text_with_its_own_aspect(self) -> None:
        absa = self._fake_absa()

        results = absa.analyze_pairs(
            ["battery is bad", "battery is bad", "screen is great"],
            ["battery", "price", "screen"],
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(
            absa.tokenizer.text_calls,
            [["battery is bad", "battery is bad", "screen is great"]],
        )
        self.assertEqual(absa.tokenizer.aspect_calls, [["battery", "price", "screen"]])

    def test_analyze_pairs_keeps_aspect_paired_through_length_sort(self) -> None:
        model_module.SORT_BATCH_BY_LENGTH = True
        absa = self._fake_absa()

        # "b"+"B" (len 2) sorts before "aaaa"+"A" (len 5); the aspect must travel
        # with its text, and results must come back in the original order.
        results = absa.analyze_pairs(["aaaa", "b"], ["A", "B"])

        self.assertEqual(len(results), 2)
        self.assertEqual(absa.tokenizer.text_calls, [["b", "aaaa"]])
        self.assertEqual(absa.tokenizer.aspect_calls, [["B", "A"]])

    def test_analyze_batch_still_applies_single_aspect_to_all(self) -> None:
        absa = self._fake_absa()

        absa.analyze_batch(["a", "b", "c"], "overall")

        self.assertEqual(
            absa.tokenizer.aspect_calls, [["overall", "overall", "overall"]]
        )


class _RecordingModel:
    device = "cpu"

    def __init__(self) -> None:
        self.pairs_calls: list[tuple[list[str], list[str]]] = []
        self.batch_calls: list[tuple[list[str], str]] = []

    def analyze_batch(self, texts: list[str], aspect: str):
        self.batch_calls.append((texts, aspect))
        return [self._result() for _ in texts]

    def analyze_pairs(self, texts: list[str], aspects: list[str]):
        self.pairs_calls.append((texts, aspects))
        return [self._result() for _ in texts]

    @staticmethod
    def _result() -> dict:
        return {
            "sentiment": "positive",
            "confidence": 0.85,
            "is_negative": False,
            "probs": {"negative": 0.05, "neutral": 0.1, "positive": 0.85},
        }


class GroupsInvocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _RecordingModel()
        self.original_get_variant = main_module._get_variant
        main_module._get_variant = lambda precision: self.model

    def tearDown(self) -> None:
        main_module._get_variant = self.original_get_variant

    def test_groups_invocation_routes_to_analyze_pairs_flattened(self) -> None:
        body = InvocationRequest(
            groups=[
                {"aspect": "battery", "text": "battery life is terrible"},
                {"aspect": "price", "text": "battery life is terrible"},
                {"aspect": "screen", "text": "the screen is great"},
            ]
        )
        response = asyncio.run(main_module.invocations(body))

        self.assertEqual(len(response.content["results"]), 3)
        self.assertEqual(
            self.model.pairs_calls,
            [
                (
                    [
                        "battery life is terrible",
                        "battery life is terrible",
                        "the screen is great",
                    ],
                    ["battery", "price", "screen"],
                )
            ],
        )
        self.assertEqual(self.model.batch_calls, [])
