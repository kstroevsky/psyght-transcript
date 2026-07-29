"""Tests for therapy fine-tuning data preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase1.pipeline.audio import AudioBundle
from phase1.therapy.finetune import (
    FixedWindowTrainingPairBuilder,
    PrimarySegmentTrainingPairBuilder,
    build_review_chunks,
    load_reviewed_segments,
    write_training_pairs_jsonl,
)


class TherapyFinetuneTest(unittest.TestCase):
    def test_build_review_chunks_writes_audio_manifest_and_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "review"
            with patch(
                "phase1.therapy.finetune.load_audio_bundle",
                return_value=AudioBundle(audio=[0.0] * 32000, duration=2.0),
            ):
                manifest = build_review_chunks(
                    audio_path="/tmp/source.wav",
                    output_dir=output_dir,
                    chunk_seconds=1.0,
                )

            self.assertEqual(2, len(manifest["entries"]))
            manifest_path = output_dir / "manifest.json"
            template_path = output_dir / "review_templates" / "chunk_000.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(template_path.exists())
            template_payload = json.loads(template_path.read_text(encoding="utf-8"))
            self.assertEqual([], template_payload["segments"])

    def test_fixed_window_pair_builder_emits_jsonl_rows(self) -> None:
        whisper_segments = [
            {"start": 0.0, "end": 30.0, "text": "whisper one"},
            {"start": 30.0, "end": 60.0, "text": "whisper two"},
        ]
        ctc_segments = [
            {"start": 0.0, "end": 30.0, "text": "ctc one"},
            {"start": 30.0, "end": 60.0, "text": "ctc two"},
        ]
        clean_segments = [
            {"start": 0.0, "end": 30.0, "text": "clean one"},
            {"start": 30.0, "end": 60.0, "text": "clean two"},
        ]

        pairs = FixedWindowTrainingPairBuilder(window_seconds=30.0).build_pairs(
            whisper_segments=whisper_segments,
            ctc_segments=ctc_segments,
            clean_segments=clean_segments,
        )

        self.assertEqual(2, len(pairs))
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pairs.jsonl"
            write_training_pairs_jsonl(pairs, output_path)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual("ctc one", rows[0]["ctc"])
        self.assertEqual("whisper two", rows[1]["whisper"])
        self.assertEqual("clean two", rows[1]["clean"])

    def test_load_reviewed_segments_infers_missing_end_times_from_next_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            review_path = Path(tmp_dir) / "gemini.json"
            review_path.write_text(
                json.dumps(
                    [
                        {"speaker": "A", "start_sec": 0.0, "text": "раз"},
                        {"speaker": "B", "start_sec": 2.5, "text": "два"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            segments = load_reviewed_segments(review_path)

        self.assertEqual(2.5, segments[0]["end"])
        self.assertGreater(segments[1]["end"], segments[1]["start"])

    def test_primary_segment_pair_builder_matches_runtime_segment_granularity(self) -> None:
        pairs = PrimarySegmentTrainingPairBuilder().build_pairs(
            whisper_segments=[{"start": 0.0, "end": 5.0, "text": "whisper text"}],
            ctc_segments=[{"start": 0.0, "end": 5.0, "text": "ctc text"}],
            clean_segments=[{"start": 0.0, "end": 5.0, "text": "clean text"}],
        )

        self.assertEqual(1, len(pairs))
        self.assertEqual("segment_0000", pairs[0].chunk_id)
        self.assertEqual("clean text", pairs[0].clean)


if __name__ == "__main__":
    unittest.main()
