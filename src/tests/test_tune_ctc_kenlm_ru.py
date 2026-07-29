"""Tests for the RU CTC + KenLM tuning workflow helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phase1.tuning.ctc_kenlm_ru import load_tuning_manifest, rank_tuning_results


class TuneCTCKenLMRUTest(unittest.TestCase):
    """Verify manifest validation and production ranking rules."""

    def test_load_tuning_manifest_rejects_missing_reference_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest_path.write_text(json.dumps([{"audio_path": "/tmp/audio.mp3"}]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing `reference_path`"):
                load_tuning_manifest(manifest_path)

    def test_load_tuning_manifest_accepts_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps({"audio_path": "/tmp/a.mp3", "reference_path": "/tmp/a.txt", "duration_limit": 10}),
                        json.dumps({"audio_path": "/tmp/b.mp3", "reference_path": "/tmp/b.txt"}),
                    ]
                ),
                encoding="utf-8",
            )

            entries = load_tuning_manifest(manifest_path)

        self.assertEqual(2, len(entries))
        self.assertEqual(10.0, entries[0].duration_limit)
        self.assertIsNone(entries[1].duration_limit)

    def test_rank_tuning_results_orders_by_wer_then_cer_then_rtf(self) -> None:
        ranked = rank_tuning_results(
            [
                {
                    "candidate_id": "slower",
                    "status": "completed",
                    "metrics": {"wer": 0.1, "cer": 0.2},
                    "performance": {"realtime_factor": 0.6},
                },
                {
                    "candidate_id": "faster",
                    "status": "completed",
                    "metrics": {"wer": 0.1, "cer": 0.2},
                    "performance": {"realtime_factor": 0.4},
                },
                {
                    "candidate_id": "worse_wer",
                    "status": "completed",
                    "metrics": {"wer": 0.2, "cer": 0.1},
                    "performance": {"realtime_factor": 0.1},
                },
            ]
        )

        self.assertEqual(["faster", "slower", "worse_wer"], [row["candidate_id"] for row in ranked])


if __name__ == "__main__":
    unittest.main()
