"""Tests for the dedicated therapy pipeline CLI wrapper."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from phase1.tools import run_therapy_pipeline as therapy_cli


class RunTherapyPipelineCliTest(unittest.TestCase):
    def test_live_canary_is_enabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "audio.wav"
            output_dir = Path(tmp_dir) / "out"
            audio_path.write_bytes(b"")

            with (
                patch.object(therapy_cli, "execute", return_value=types.SimpleNamespace(wall_clock_sec=1.0)) as execute_mock,
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_therapy_pipeline.py",
                        str(audio_path),
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
            ):
                therapy_cli.main()

        options = execute_mock.call_args.args[0]
        self.assertTrue(options.backend.options["use_live_canary"])

    def test_canary_json_disables_default_live_canary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "audio.wav"
            output_dir = Path(tmp_dir) / "out"
            canary_path = Path(tmp_dir) / "canary.json"
            audio_path.write_bytes(b"")
            canary_path.write_text("[]", encoding="utf-8")

            with (
                patch.object(therapy_cli, "execute", return_value=types.SimpleNamespace(wall_clock_sec=1.0)) as execute_mock,
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_therapy_pipeline.py",
                        str(audio_path),
                        "--output-dir",
                        str(output_dir),
                        "--canary-json",
                        str(canary_path),
                    ],
                ),
            ):
                therapy_cli.main()

        options = execute_mock.call_args.args[0]
        self.assertNotIn("use_live_canary", options.backend.options)
        self.assertEqual(str(canary_path.resolve()), options.backend.options["canary_transcript_path"])

    def test_disable_live_canary_requires_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "audio.wav"
            output_dir = Path(tmp_dir) / "out"
            audio_path.write_bytes(b"")

            with patch.object(
                sys,
                "argv",
                [
                    "run_therapy_pipeline.py",
                    str(audio_path),
                    "--output-dir",
                    str(output_dir),
                    "--disable-live-canary",
                ],
            ):
                with self.assertRaisesRegex(ValueError, "requires Canary input"):
                    therapy_cli.main()


if __name__ == "__main__":
    unittest.main()
