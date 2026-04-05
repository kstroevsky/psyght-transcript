"""Tests for phase1 runtime-device and CPU-thread selection helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from phase1.config import runtime
from phase1.config.asr import resolve_asr_threads
from phase1.config.models import resolve_lang_model_map


class RuntimeConfigTest(unittest.TestCase):
    """Verify Apple Silicon fallbacks and CPU thread sizing helpers."""

    def test_asr_runtime_falls_back_from_mps_to_cpu(self) -> None:
        with (
            patch.object(runtime, "requested_asr_device", return_value="mps"),
            patch.object(runtime, "_mps_is_available", return_value=True),
            patch.object(runtime, "_cuda_is_available", return_value=False),
        ):
            config = runtime.get_asr_runtime_config()

        self.assertEqual("cpu", config["device"])
        self.assertIn("faster-whisper/CTranslate2", config["fallback_reason"])

    def test_resolve_asr_threads_uses_all_cpu_cores_without_override(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(runtime, "cpu_core_count", return_value=10),
            patch("phase1.config.asr.cpu_core_count", return_value=10),
        ):
            self.assertEqual(10, resolve_asr_threads("cpu"))

    def test_resolve_asr_threads_respects_env_override(self) -> None:
        with patch.dict(os.environ, {"WHISPERX_ASR_THREADS": "3"}, clear=False):
            self.assertEqual(3, resolve_asr_threads("cpu"))

    def test_runtime_overrides_apply_per_run(self) -> None:
        with (
            patch.object(runtime, "_cuda_is_available", return_value=False),
            patch.object(runtime, "_mps_is_available", return_value=False),
        ):
            config = runtime.get_asr_runtime_config({"device": "cpu", "compute_type": "float32"})

        self.assertEqual("cpu", config["device"])
        self.assertEqual("float32", config["compute_type"])

    def test_diarization_runtime_prefers_mps_on_apple_silicon_when_available(self) -> None:
        with (
            patch.object(runtime, "_cuda_is_available", return_value=False),
            patch.object(runtime, "_mps_is_available", return_value=True),
        ):
            config = runtime.get_diarization_runtime_config()

        self.assertEqual("mps", config["requested_device"])
        self.assertEqual("mps", config["device"])

    def test_diarization_runtime_respects_generic_device_override(self) -> None:
        with (
            patch.object(runtime, "_cuda_is_available", return_value=False),
            patch.object(runtime, "_mps_is_available", return_value=True),
        ):
            config = runtime.get_diarization_runtime_config({"device": "cpu"})

        self.assertEqual("cpu", config["requested_device"])
        self.assertEqual("cpu", config["device"])

    def test_alignment_runtime_prefers_mps_on_apple_silicon_when_available(self) -> None:
        with (
            patch.object(runtime, "_cuda_is_available", return_value=False),
            patch.object(runtime, "_mps_is_available", return_value=True),
        ):
            config = runtime.get_alignment_runtime_config()

        self.assertEqual("mps", config["requested_device"])
        self.assertEqual("mps", config["device"])

    def test_alignment_runtime_respects_generic_device_override(self) -> None:
        with (
            patch.object(runtime, "_cuda_is_available", return_value=False),
            patch.object(runtime, "_mps_is_available", return_value=True),
        ):
            config = runtime.get_alignment_runtime_config({"device": "cpu"})

        self.assertEqual("cpu", config["requested_device"])
        self.assertEqual("cpu", config["device"])

    def test_language_model_map_respects_preset_override(self) -> None:
        mapping = resolve_lang_model_map(
            {
                "model_name": "global-model",
                "language_models": {"ru": "ru-model", "default": "fallback-model"},
            }
        )

        self.assertEqual("ru-model", mapping["ru"])
        self.assertEqual("global-model", mapping["en"])
        self.assertEqual("fallback-model", mapping[None])


if __name__ == "__main__":
    unittest.main()
