"""Tests for the live Canary backend wrapper."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phase1.backends.canary_backend import CanaryBackend, resolve_canary_runtime
from phase1.tools import canary_transcribe as canary_cli


class CanaryBackendTest(unittest.TestCase):
    def test_runtime_falls_back_from_mps_to_cpu(self) -> None:
        runtime = resolve_canary_runtime({"device": "mps"}, "ru")

        self.assertEqual("cpu", runtime["device"])
        self.assertIn("falling back to CPU", runtime["fallback_reason"])

    def test_transcribe_uses_helper_output_and_preserves_words(self) -> None:
        backend = CanaryBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            helper_python = Path(tmp_dir) / "python"
            helper_script = Path(tmp_dir) / "script.py"
            helper_python.write_text("", encoding="utf-8")
            helper_script.write_text("", encoding="utf-8")

            def fake_run(command, check, capture_output, text):
                output_json = Path(command[command.index("--output-json") + 1])
                self.assertIn("--chunk-seconds", command)
                self.assertEqual("30.0", command[command.index("--chunk-seconds") + 1])
                output_json.write_text(
                    json.dumps(
                        {
                            "meta": {
                                "model_id": "nvidia/canary-1b-v2",
                                "chunk_seconds": 30.0,
                                "effective_prompt": {"source_lang": "ru", "target_lang": "ru", "taskname": "asr"},
                            },
                            "segments": [
                                {
                                    "start": 0.0,
                                    "end": 1.0,
                                    "text": "hello",
                                    "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                class Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Result()

            with (
                patch("phase1.backends.canary_backend.canary_helper_python", return_value=helper_python),
                patch("phase1.backends.canary_backend.canary_helper_script", return_value=helper_script),
                patch("phase1.backends.canary_backend.write_wav_mono"),
                patch("phase1.backends.canary_backend.subprocess.run", side_effect=fake_run),
            ):
                segments, language, meta = backend.transcribe([0.0, 0.0], "ru", return_meta=True)

        self.assertEqual("ru", language)
        self.assertEqual("hello", segments[0]["text"])
        self.assertEqual("hello", segments[0]["words"][0]["word"])
        self.assertIn("01a_canary_segments.json", meta["artifact_payloads"])
        self.assertEqual(30.0, meta["canary_chunk_seconds"])
        self.assertEqual("ru", meta["canary_effective_prompt"]["source_lang"])

    def test_helper_forwards_prompt_kwargs_through_var_kwargs_signature(self) -> None:
        class _FakeModel:
            def transcribe(self, audio, batch_size=1, timestamps=False, **prompt):
                del audio, batch_size, timestamps
                return prompt

        args = SimpleNamespace(
            batch_size=1,
            source_lang="ru",
            target_lang="ru",
            taskname="asr",
            pnc="yes",
            timestamps=True,
        )

        kwargs, prompt = canary_cli._build_transcribe_kwargs(_FakeModel(), args)

        self.assertEqual("ru", kwargs["source_lang"])
        self.assertEqual("ru", kwargs["target_lang"])
        self.assertEqual("asr", kwargs["taskname"])
        self.assertEqual("yes", kwargs["pnc"])
        self.assertTrue(kwargs["timestamps"])
        self.assertEqual(kwargs, prompt)

    def test_helper_chunked_decode_offsets_segment_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_wav = Path(tmp_dir) / "input.wav"
            canary_cli._write_wav_mono(input_wav, [0.0] * (2 * canary_cli.SAMPLE_RATE))

            class _FakeModel:
                def transcribe(self, audio, batch_size=1, timestamps=False, **prompt):
                    del batch_size, timestamps, prompt
                    chunk_name = Path(audio[0]).name
                    if chunk_name == "chunk_000.wav":
                        text = "первый"
                    else:
                        text = "второй"
                    return [
                        SimpleNamespace(
                            text=text,
                            timestamp={
                                "word": [{"word": text, "start": 0.0, "end": 0.5}],
                                "segment": [{"start": 0.0, "end": 0.5, "text": text}],
                            },
                        )
                    ]

            args = SimpleNamespace(
                input_wav=str(input_wav),
                model_id="nvidia/canary-1b-v2",
                device="cpu",
                batch_size=1,
                source_lang="ru",
                target_lang="ru",
                taskname="asr",
                pnc="yes",
                timestamps=True,
                chunk_seconds=1.0,
            )

            payload, inference_sec, effective_prompt = canary_cli._transcribe_chunks(_FakeModel(), args)

        self.assertGreaterEqual(inference_sec, 0.0)
        self.assertEqual({"batch_size": 1, "source_lang": "ru", "target_lang": "ru", "taskname": "asr", "pnc": "yes", "timestamps": True}, effective_prompt)
        self.assertEqual(2, payload["meta"]["chunk_count"])
        self.assertEqual(["первый", "второй"], [segment["text"] for segment in payload["segments"]])
        self.assertEqual(0.0, payload["segments"][0]["start"])
        self.assertEqual(1.0, payload["segments"][1]["start"])


if __name__ == "__main__":
    unittest.main()
