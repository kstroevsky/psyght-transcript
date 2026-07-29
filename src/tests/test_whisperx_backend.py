"""Focused tests for WhisperX backend timestamp sanitization."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from phase1.backends.whisperx_backend import WhisperXBackend, _sanitize_segments


class WhisperXBackendTest(unittest.TestCase):
    def test_sanitize_segments_clamps_invalid_segment_and_word_intervals(self) -> None:
        segments = [
            {
                "start": 4.591,
                "end": 4.271,
                "text": "example",
                "words": [
                    {"word": "bad", "start": 4.7, "end": 4.5},
                    {"word": "plain"},
                ],
            }
        ]

        sanitized = _sanitize_segments(segments)

        self.assertEqual(1, len(sanitized))
        self.assertEqual(4.591, sanitized[0]["start"])
        self.assertEqual(4.7, sanitized[0]["end"])
        self.assertEqual(4.7, sanitized[0]["words"][0]["start"])
        self.assertEqual(4.7, sanitized[0]["words"][0]["end"])

    def test_align_sanitizes_invalid_whisperx_intervals(self) -> None:
        fake_whisperx = SimpleNamespace(
            load_align_model=lambda language_code, device: ("align-model", {"language": language_code}),
            align=lambda segments, align_model, metadata, audio, device, return_char_alignments: {
                "segments": [{"start": 4.591, "end": 4.271, "text": "example", "words": []}]
            },
        )

        with patch.dict(sys.modules, {"whisperx": fake_whisperx}):
            aligned = WhisperXBackend().align(
                [{"start": 4.591, "end": 4.271, "text": "example"}],
                "ru",
                [0.0, 0.0],
            )

        self.assertEqual([{"start": 4.591, "end": 4.591, "text": "example", "words": []}], aligned)

    def test_transcribe_forced_language_passes_language_to_whisperx_and_keeps_requested_language(self) -> None:
        backend = WhisperXBackend()

        class _FakeModel:
            def __init__(self) -> None:
                self.calls = []

            def transcribe(self, audio, **kwargs):
                self.calls.append({"audio": audio, "kwargs": kwargs})
                return {"segments": [{"start": 0.0, "end": 1.0, "text": "пример", "words": []}], "language": "en"}

        fake_model = _FakeModel()

        with (
            patch.object(
                backend,
                "_build_asr_runtime",
                return_value={
                    "batch_size": 16,
                    "task": "transcribe",
                    "language_models": {None: "base", "ru": "ru-model"},
                },
            ),
            patch.object(backend, "_load_model", return_value=fake_model),
        ):
            segments, language, meta = backend.transcribe([0.0, 0.0], "ru", return_meta=True)

        self.assertEqual("ru", language)
        self.assertEqual("ru-model", meta["model_name"])
        self.assertEqual("ru", meta["requested_language"])
        self.assertEqual("en", meta["reported_language"])
        self.assertEqual("ru", fake_model.calls[0]["kwargs"]["language"])
        self.assertEqual("transcribe", fake_model.calls[0]["kwargs"]["task"])
        self.assertEqual([{"start": 0.0, "end": 1.0, "text": "пример", "words": []}], segments)


if __name__ == "__main__":
    unittest.main()
