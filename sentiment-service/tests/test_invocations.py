from __future__ import annotations

import asyncio
import unittest

from tests.fakes import install_fake_fastapi_modules, install_fake_ml_modules

install_fake_ml_modules()
install_fake_fastapi_modules()

import app.main as main_module
from app.schemas import InvocationRequest


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
        result = asyncio.run(
            main_module._dispatch_invocation({"texts": ["a", "b"], "aspect": "battery"})
        )

        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(self.model.batch_calls, [(["a", "b"], "battery")])
        self.assertEqual(self.model.single_calls, [])

    def test_invocations_single_text_uses_single_inference_path(self) -> None:
        result = asyncio.run(
            main_module._dispatch_invocation({"text": "a", "aspect": "battery"})
        )

        self.assertEqual(result["sentiment"], "positive")
        self.assertEqual(self.model.single_calls, [("a", "battery")])
        self.assertEqual(self.model.batch_calls, [])

    def test_invocations_instances_payload_uses_batch_inference_path(self) -> None:
        result = asyncio.run(
            main_module._dispatch_invocation(
                {"instances": ["a", "b", "c"], "aspect": "screen"}
            )
        )

        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(self.model.batch_calls, [(["a", "b", "c"], "screen")])
        self.assertEqual(self.model.single_calls, [])

    def test_invocations_requires_exactly_one_payload_field(self) -> None:
        from fastapi import HTTPException

        for body in ({}, {"text": "a", "texts": ["b"]}):
            with self.assertRaises(HTTPException):
                asyncio.run(main_module._dispatch_invocation(body))

    def test_invocations_rejects_empty_batch(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            asyncio.run(main_module._dispatch_invocation({"texts": []}))

    def test_invocation_schema_documents_texts_for_batch(self) -> None:
        schema = InvocationRequest.model_json_schema()
        properties = schema["properties"]

        self.assertIn("text", properties)
        self.assertIn("texts", properties)

    def test_invocation_schema_shows_batch_texts_example(self) -> None:
        schema = InvocationRequest.model_json_schema()
        texts_schema = schema["properties"]["texts"]

        self.assertEqual(texts_schema["anyOf"][0]["type"], "array")
        self.assertEqual(
            schema["examples"][0]["texts"],
            ["The battery life is terrible", "The screen is great"],
        )
