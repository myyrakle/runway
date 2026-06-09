from __future__ import annotations

import asyncio
import unittest

from tests.fakes import install_fake_fastapi_modules, install_fake_ml_modules

install_fake_ml_modules()
install_fake_fastapi_modules()

import app.main as main_module
from app.schemas import BatchAnalyzeRequest


class _RecordingModel:
    device = "cpu"

    def __init__(self) -> None:
        self.single_calls: list[tuple[str, str]] = []
        self.batch_calls: list[tuple[list[str], str]] = []

    def analyze(self, text: str, aspect: str):
        self.single_calls.append((text, aspect))
        return {
            "sentiment": "positive",
            "confidence": 0.85,
            "is_negative": False,
            "probs": {"negative": 0.05, "neutral": 0.1, "positive": 0.85},
        }

    def analyze_batch(self, texts: list[str], aspect: str):
        self.batch_calls.append((texts, aspect))
        return [
            {
                "sentiment": "positive",
                "confidence": 0.85,
                "is_negative": False,
                "probs": {"negative": 0.05, "neutral": 0.1, "positive": 0.85},
            }
            for _ in texts
        ]


class InvocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _RecordingModel()
        self.original_get_variant = main_module._get_variant
        main_module._get_variant = lambda precision: self.model

    def tearDown(self) -> None:
        main_module._get_variant = self.original_get_variant

    def test_invocations_batch_uses_same_batch_inference_path(self) -> None:
        req = BatchAnalyzeRequest(texts=["a", "b"], aspect="battery")

        result = asyncio.run(main_module.invocations(req))

        self.assertEqual(len(result.results), 2)
        self.assertEqual(self.model.batch_calls, [(["a", "b"], "battery")])
        self.assertEqual(self.model.single_calls, [])
