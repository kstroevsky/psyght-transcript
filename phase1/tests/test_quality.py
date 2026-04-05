"""Tests for the dedicated phase1 quality meta-layer."""

from __future__ import annotations

import unittest

from contracts.transcript import Segment, Transcript
from phase1.quality import build_run_quality_report, rank_run_summaries


class QualityLayerTest(unittest.TestCase):
    """Verify the standalone quality meta-layer contracts."""

    def _transcript(self, text: str) -> Transcript:
        return Transcript(
            language="en",
            duration=2.0,
            segments=[Segment(speaker="SPEAKER_00", start=0.0, end=1.0, text=text)],
        )

    def test_build_run_quality_report_combines_sections(self) -> None:
        report = build_run_quality_report(
            transcript=self._transcript("hello world"),
            run_meta={
                "duration_sec": 2.0,
                "wall_clock_sec": 1.0,
                "stage_timings_sec": {"asr": 0.5},
                "runtime": {
                    "resolved_asr": {"fallback_reason": None},
                    "resolved_alignment": {"fallback_reason": None},
                    "resolved_diarization": {"fallback_reason": None},
                },
                "skip_alignment": False,
                "skip_diarization": False,
            },
            reference_text="hello world",
        )

        self.assertEqual(0.5, report["performance"]["realtime_factor"])
        self.assertEqual(0.0, report["metrics"]["wer"])
        self.assertEqual([], report["stability"]["fallback_reasons"])

    def test_rank_run_summaries_prefers_lower_error_then_lower_rtf(self) -> None:
        ranked = rank_run_summaries(
            [
                {
                    "preset_id": "slower",
                    "status": "completed",
                    "metrics": {"wer": 0.1, "cer": 0.2},
                    "stability": {"fallback_count": 0, "skipped_stage_count": 0},
                    "performance": {"wall_clock_sec": 2.0, "realtime_factor": 0.8},
                },
                {
                    "preset_id": "faster",
                    "status": "completed",
                    "metrics": {"wer": 0.1, "cer": 0.2},
                    "stability": {"fallback_count": 0, "skipped_stage_count": 0},
                    "performance": {"wall_clock_sec": 1.0, "realtime_factor": 0.4},
                },
            ],
            has_reference=True,
        )

        self.assertEqual(["faster", "slower"], [row["preset_id"] for row in ranked])
