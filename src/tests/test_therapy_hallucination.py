"""Tests for the timestamp-entropy hallucination guard."""

from __future__ import annotations

import unittest

from phase1.therapy.hallucination import CTCFallbackHallucinationResolver, TimestampEntropyHallucinationDetector


class TherapyHallucinationTest(unittest.TestCase):
    def test_detector_flags_uniform_word_duration_pattern(self) -> None:
        detector = TimestampEntropyHallucinationDetector()
        segment = {
            "start": 0.0,
            "end": 3.0,
            "text": "one two three four",
            "words": [
                {"word": "one", "start": 0.0, "end": 0.6},
                {"word": "two", "start": 0.7, "end": 1.3},
                {"word": "three", "start": 1.4, "end": 2.0},
                {"word": "four", "start": 2.1, "end": 2.7},
            ],
        }

        self.assertTrue(detector.is_hallucinated(segment))

    def test_resolver_swaps_in_ctc_text_for_flagged_segments(self) -> None:
        resolver = CTCFallbackHallucinationResolver()
        whisper_segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "bad bad bad bad",
                "words": [
                    {"word": "bad", "start": 0.0, "end": 0.6},
                    {"word": "bad", "start": 0.7, "end": 1.3},
                    {"word": "bad", "start": 1.4, "end": 2.0},
                    {"word": "bad", "start": 2.1, "end": 2.7},
                ],
            },
            {
                "start": 3.0,
                "end": 4.0,
                "text": "good segment",
                "words": [{"word": "good", "start": 3.1, "end": 3.3}],
            },
        ]
        ctc_segments = [
            {"start": 0.0, "end": 3.0, "text": "clean anchor text"},
            {"start": 3.0, "end": 4.0, "text": "good segment"},
        ]

        resolved, meta = resolver.resolve(whisper_segments, ctc_segments)

        self.assertEqual("clean anchor text", resolved[0]["text"])
        self.assertEqual("low", resolved[0]["confidence"])
        self.assertEqual("ctc_fallback", resolved[0]["source"])
        self.assertEqual("high", resolved[1]["confidence"])
        self.assertEqual("merged", resolved[1]["source"])
        self.assertEqual(1, meta["flagged_segment_count"])


if __name__ == "__main__":
    unittest.main()
