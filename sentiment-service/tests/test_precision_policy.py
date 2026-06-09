from __future__ import annotations

import importlib
import os
from pathlib import Path
import unittest

from tests.fakes import install_fake_ml_modules

install_fake_ml_modules()


class PrecisionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_default = os.environ.get("DEFAULT_PRECISION")
        self.old_allowed = os.environ.get("ALLOWED_PRECISIONS")

    def tearDown(self) -> None:
        self._restore_env("DEFAULT_PRECISION", self.old_default)
        self._restore_env("ALLOWED_PRECISIONS", self.old_allowed)
        import app.config as config

        importlib.reload(config)

    def _restore_env(self, name: str, value: str | None) -> None:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def test_config_restricts_precision_to_allowed_set(self) -> None:
        os.environ["DEFAULT_PRECISION"] = "fp16"
        os.environ["ALLOWED_PRECISIONS"] = "fp16"

        import app.config as config

        config = importlib.reload(config)

        self.assertEqual(config.DEFAULT_PRECISION, "fp16")
        self.assertEqual(config.ALLOWED_PRECISIONS, ("fp16",))
        self.assertTrue(config.is_precision_allowed("fp16"))
        self.assertFalse(config.is_precision_allowed("fp32"))

    def test_cuda_dockerfiles_are_split_by_default_precision(self) -> None:
        root = Path(__file__).resolve().parents[1]

        self.assertFalse((root / "Dockerfile.sagemaker-cu128").exists())
        self.assertFalse((root / "Dockerfile.sagemaker-cu128-fp32").exists())
        self.assertFalse((root / "Dockerfile.sagemaker-cu128-fp16").exists())

        fp32 = (root / "docker" / "Dockerfile.sagemaker-cu128-fp32").read_text()
        fp16 = (root / "docker" / "Dockerfile.sagemaker-cu128-fp16").read_text()

        self.assertIn("DEFAULT_PRECISION=fp32", fp32)
        self.assertIn("ALLOWED_PRECISIONS=fp32", fp32)
        self.assertIn("DEFAULT_PRECISION=fp16", fp16)
        self.assertIn("ALLOWED_PRECISIONS=fp16", fp16)

    def test_readme_documents_docker_build_commands_with_docker_directory(self) -> None:
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text()
        docker_readme = (root / "docker" / "README.md").read_text()

        self.assertIn("docker/README.md", readme)

        self.assertIn(
            "docker build -f docker/Dockerfile.sagemaker-cu128-fp32 "
            "-t sentiment-service:cu128-fp32 .",
            docker_readme,
        )
        self.assertIn(
            "docker build -f docker/Dockerfile.sagemaker-cu128-fp16 "
            "-t sentiment-service:cu128-fp16 .",
            docker_readme,
        )
        self.assertIn(
            "docker build -f docker/Dockerfile.sagemaker-cpu "
            "-t sentiment-service:cpu .",
            docker_readme,
        )
        self.assertIn(
            "docker build -f docker/Dockerfile.sagemaker-cpu-int8 "
            "-t sentiment-service:cpu-int8 .",
            docker_readme,
        )
        self.assertIn("DEFAULT_PRECISION=fp16", docker_readme)
        self.assertIn("ALLOWED_PRECISIONS=fp16", docker_readme)

    def test_cpu_int8_dockerfile_sets_int8_and_thread_defaults(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "docker" / "Dockerfile.sagemaker-cpu-int8").read_text()

        self.assertIn("DEFAULT_PRECISION=int8", dockerfile)
        self.assertIn("ALLOWED_PRECISIONS=int8", dockerfile)
        self.assertIn("OMP_NUM_THREADS=4", dockerfile)
        self.assertIn("MKL_NUM_THREADS=4", dockerfile)
