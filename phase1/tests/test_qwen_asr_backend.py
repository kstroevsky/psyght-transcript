"""Tests for the Qwen3-ASR backend helper integration."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from phase1.backends.qwen_asr_backend import QwenASRBackend


class QwenAsrBackendTest(unittest.TestCase):
    def test_transcribe_invokes_helper_and_returns_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            helper_python = Path(tmp_dir) / "python"
            helper_script = Path(tmp_dir) / "qwen_asr_transcribe.py"
            helper_python.write_text("", encoding="utf-8")
            helper_script.write_text("", encoding="utf-8")

            def _run(command, check, capture_output, text):
                output_json = Path(command[command.index("--output-json") + 1])
                output_json.write_text(
                    json.dumps(
                        {
                            "language": "Russian",
                            "segments": [{"start": 0.0, "end": 1.0, "text": "тест", "words": []}],
                            "meta": {"used_forced_aligner": True},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                patch("phase1.backends.qwen_asr_backend.qwen_helper_python", return_value=helper_python),
                patch("phase1.backends.qwen_asr_backend.qwen_helper_transcribe_script", return_value=helper_script),
                patch("phase1.backends.qwen_asr_backend.qwen_setup_script", return_value=Path(tmp_dir) / "setup.py"),
                patch("phase1.backends.qwen_asr_backend.subprocess.run", side_effect=_run) as run_mock,
            ):
                segments, language, meta = QwenASRBackend().transcribe(
                    [0.0, 0.0, 0.0, 0.0],
                    "ru",
                    backend_options={
                        "model_path": "/tmp/model",
                        "forced_aligner_model_path": "/tmp/aligner",
                    },
                    return_meta=True,
                )

        self.assertEqual("Russian", language)
        self.assertEqual("тест", segments[0]["text"])
        self.assertTrue(meta["helper_meta"]["used_forced_aligner"])
        command = run_mock.call_args.args[0]
        self.assertIn("--use-forced-aligner", command)
        self.assertIn("/tmp/model", command)

    def test_align_is_noop_when_forced_aligner_enabled(self) -> None:
        backend = QwenASRBackend()
        segments = [{"start": 0.0, "end": 1.0, "text": "тест"}]

        aligned, meta = backend.align(
            segments,
            "ru",
            [0.0, 0.0],
            backend_options={"forced_aligner_model_path": "/tmp/aligner"},
            return_meta=True,
        )

        self.assertEqual(segments, aligned)
        self.assertIn("forced aligner", meta["skip_reason"].lower())

    def test_align_delegates_to_whisperx_when_forced_aligner_disabled(self) -> None:
        backend = QwenASRBackend()
        backend._aligner.align = MagicMock(return_value=[{"text": "aligned"}])  # type: ignore[method-assign]

        result = backend.align(
            [{"text": "raw"}],
            "ru",
            [0.0, 0.0],
            backend_options={"use_forced_aligner": False},
        )

        self.assertEqual([{"text": "aligned"}], result)
        backend._aligner.align.assert_called_once()


if __name__ == "__main__":
    unittest.main()

