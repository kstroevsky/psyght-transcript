"""Contract tests for the shared transcript serializer/deserializer."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from contracts.transcript import transcript_from_dict, transcript_to_dict


class TranscriptContractTest(unittest.TestCase):
    """Verify that the shared contract preserves the persisted transcript JSON shape."""

    def test_round_trip_preserves_fixture_shape(self) -> None:
        fixture_path = Path(__file__).resolve().parents[2] / "phase1" / "test_first2min.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))

        transcript = transcript_from_dict(payload)
        round_tripped = transcript_to_dict(transcript)

        self.assertEqual(payload, round_tripped)
        self.assertEqual(["duration", "language", "segments"], sorted(round_tripped.keys()))
        self.assertEqual(
            ["end", "speaker", "start", "text", "words"],
            sorted(round_tripped["segments"][0].keys()),
        )
        self.assertEqual(
            ["end", "score", "start", "word"],
            sorted(round_tripped["segments"][0]["words"][0].keys()),
        )

    def test_round_trip_preserves_optional_segment_and_word_fields(self) -> None:
        payload = {
            "language": "ru",
            "duration": 1.0,
            "segments": [
                {
                    "speaker": "A",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Привет",
                    "confidence": "low",
                    "source": "ctc_fallback",
                    "words": [
                        {
                            "word": "Привет",
                            "start": 0.0,
                            "end": 0.5,
                            "score": 0.9,
                            "speaker": "A",
                        }
                    ],
                }
            ],
        }

        round_tripped = transcript_to_dict(transcript_from_dict(payload))

        self.assertEqual(payload, round_tripped)


if __name__ == "__main__":
    unittest.main()
