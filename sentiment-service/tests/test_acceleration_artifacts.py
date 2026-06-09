from __future__ import annotations

from pathlib import Path
import unittest


class AccelerationArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_makefile_exposes_acceleration_targets(self) -> None:
        makefile = (self.root / "Makefile").read_text()

        for target in (
            "onnx-export:",
            "onnx-cuda-sample:",
            "ort-trt-sample:",
            "trt-build:",
            "trt-sample:",
        ):
            self.assertIn(target, makefile)

    def test_acceleration_scripts_exist(self) -> None:
        for script in (
            "export_onnx.py",
            "build_trt_engine.py",
            "run_accelerated.py",
        ):
            self.assertTrue((self.root / "scripts" / script).exists())

    def test_dedicated_acceleration_dockerfiles_exist(self) -> None:
        for dockerfile, backend in (
            ("Dockerfile.sagemaker-cu128-onnx-cuda", "INFERENCE_BACKEND=onnx-cuda"),
            ("Dockerfile.sagemaker-cu128-ort-trt", "INFERENCE_BACKEND=ort-trt"),
            ("Dockerfile.sagemaker-cu128-tensorrt", "INFERENCE_BACKEND=tensorrt"),
        ):
            text = (self.root / "docker" / dockerfile).read_text()
            self.assertIn(backend, text)

    def test_pyproject_declares_acceleration_optional_dependencies(self) -> None:
        pyproject = (self.root / "pyproject.toml").read_text()

        self.assertIn("[project.optional-dependencies]", pyproject)
        self.assertIn("onnxruntime-gpu", pyproject)
        self.assertIn("tensorrt", pyproject)

