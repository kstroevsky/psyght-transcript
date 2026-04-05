"""Tests for backend-independent speaker assignment."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from phase1.pipeline.assign import assign_and_build, assigned_segments_for_dump


class AssignTest(unittest.TestCase):
    """Verify overlap-based speaker assignment without WhisperX helpers."""

    def test_assigns_word_and_segment_speakers_from_diarization_overlap(self) -> None:
        aligned_segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello there",
                "words": [
                    {"word": "hello", "start": 0.1, "end": 0.8, "score": 0.9},
                    {"word": "there", "start": 1.1, "end": 1.9, "score": 0.8},
                    {"word": "!"},
                ],
            }
        ]
        diarization = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
        ]

        assigned = assigned_segments_for_dump(aligned_segments, diarization)
        transcript = assign_and_build(aligned_segments, diarization, "en", 2.0)

        self.assertEqual("SPEAKER_01", assigned[0]["words"][1]["speaker"])
        self.assertEqual("SPEAKER_00", assigned[0]["words"][0]["speaker"])
        self.assertNotIn("speaker", assigned[0]["words"][2])
        self.assertEqual("SPEAKER_01", assigned[0]["speaker"])
        self.assertEqual("SPEAKER_01", transcript.segments[0].speaker)
        self.assertEqual(2, len(transcript.segments[0].words))

    def test_sorts_and_optionally_merges_same_speaker_segments(self) -> None:
        aligned_segments = [
            {
                "start": 2.0,
                "end": 3.0,
                "text": "second",
                "words": [{"word": "second", "start": 2.1, "end": 2.8, "score": 0.9}],
            },
            {
                "start": 0.0,
                "end": 1.0,
                "text": "first",
                "words": [{"word": "first", "start": 0.1, "end": 0.8, "score": 0.9}],
            },
            {
                "start": 1.05,
                "end": 1.9,
                "text": "middle",
                "words": [{"word": "middle", "start": 1.1, "end": 1.8, "score": 0.9}],
            },
        ]
        diarization = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0},
        ]

        assigned = assigned_segments_for_dump(
            aligned_segments,
            diarization,
            merge_consecutive_same_speaker=True,
            merge_gap_sec=0.2,
        )
        transcript = assign_and_build(
            aligned_segments,
            diarization,
            "en",
            3.0,
            merge_consecutive_same_speaker=True,
            merge_gap_sec=0.2,
        )

        self.assertEqual(["first middle", "second"], [segment["text"] for segment in assigned])
        self.assertEqual([0.0, 2.0], [segment["start"] for segment in assigned])
        self.assertEqual(["first middle", "second"], [segment.text for segment in transcript.segments])

    def test_whisperx_assignment_strategy_can_normalize_speakers_and_keep_segment_metadata(self) -> None:
        aligned_segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "confidence": "low",
                "source": "ctc_fallback",
                "words": [{"word": "hello", "start": 0.1, "end": 0.6, "score": 0.9}],
            }
        ]
        diarization = [{"speaker": "SPEAKER_09", "start": 0.0, "end": 1.0}]

        with patch(
            "whisperx.assign_word_speakers",
            return_value={
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello",
                        "speaker": "SPEAKER_09",
                        "words": [
                            {
                                "word": "hello",
                                "start": 0.1,
                                "end": 0.6,
                                "score": 0.9,
                                "speaker": "SPEAKER_09",
                            }
                        ],
                    }
                ]
            },
        ):
            assigned = assigned_segments_for_dump(
                aligned_segments,
                diarization,
                strategy="whisperx",
                speaker_label_style="alpha",
            )
            transcript = assign_and_build(
                aligned_segments,
                diarization,
                "en",
                1.0,
                strategy="whisperx",
                speaker_label_style="alpha",
            )

        self.assertEqual("A", assigned[0]["speaker"])
        self.assertEqual("A", assigned[0]["words"][0]["speaker"])
        self.assertEqual("low", assigned[0]["confidence"])
        self.assertEqual("ctc_fallback", assigned[0]["source"])
        self.assertEqual("A", transcript.segments[0].speaker)
        self.assertEqual("A", transcript.segments[0].words[0].speaker)
        self.assertEqual("low", transcript.segments[0].confidence)
        self.assertEqual("ctc_fallback", transcript.segments[0].source)


if __name__ == "__main__":
    unittest.main()
