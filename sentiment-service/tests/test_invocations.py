from __future__ import annotations

import asyncio
import unittest

from pydantic import ValidationError

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
        body = InvocationRequest(texts=["a", "b"], aspect="battery")
        response = asyncio.run(main_module.invocations(body))

        self.assertEqual(len(response.content["results"]), 2)
        self.assertEqual(self.model.batch_calls, [(["a", "b"], "battery")])
        self.assertEqual(self.model.single_calls, [])

    def test_invocations_instances_payload_uses_batch_inference_path(self) -> None:
        body = InvocationRequest(instances=["a", "b", "c"], aspect="screen")
        response = asyncio.run(main_module.invocations(body))

        self.assertEqual(len(response.content["results"]), 3)
        self.assertEqual(self.model.batch_calls, [(["a", "b", "c"], "screen")])
        self.assertEqual(self.model.single_calls, [])

    def test_invocations_is_batch_only(self) -> None:
        # A bare single `text` is a valid payload shape but /invocations is batch-only,
        # so the route rejects it; empty/missing payloads fail Pydantic validation.
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            asyncio.run(main_module.invocations(InvocationRequest(text="a")))
        for kwargs in ({}, {"texts": []}):
            with self.assertRaises(ValidationError):
                InvocationRequest(**kwargs)
        self.assertEqual(self.model.batch_calls, [])
        self.assertEqual(self.model.single_calls, [])

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
