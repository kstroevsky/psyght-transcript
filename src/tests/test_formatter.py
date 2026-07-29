"""Tests for the phase1 transcript formatter helpers."""

from __future__ import annotations

import unittest

from contracts.transcript import Segment, Transcript, Word
from phase1.output.formatter import to_json, to_txt


class FormatterTest(unittest.TestCase):
    """Verify that human-readable and JSON transcript rendering stay stable."""

    def test_to_txt_renders_expected_timestamp_format(self) -> None:
        transcript = Transcript(
            language="en",
            duration=4.5,
            segments=[
                Segment(
                    speaker="SPEAKER_00",
                    start=1.5,
                    end=2.75,
                    text="Hello there",
                    words=[Word(word="Hello", start=1.5, end=1.9, score=0.9)],
                )
            ],
        )

        self.assertEqual(
            "[00:00:01.50 -> 00:00:02.75] SPEAKER_00: Hello there",
            to_txt(transcript),
        )

    def test_to_json_uses_shared_contract_shape(self) -> None:
        transcript = Transcript(
            language="en",
            duration=4.5,
            segments=[
                Segment(
                    speaker="SPEAKER_00",
                    start=1.5,
                    end=2.75,
                    text="Hello there",
                    words=[Word(word="Hello", start=1.5, end=1.9, score=0.9)],
                )
            ],
        )

        self.assertEqual(
            {
                "language": "en",
                "duration": 4.5,
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start": 1.5,
                        "end": 2.75,
                        "text": "Hello there",
                        "words": [
                            {
                                "word": "Hello",
                                "start": 1.5,
                                "end": 1.9,
                                "score": 0.9,
                            }
                        ],
                    }
                ],
            },
            to_json(transcript),
        )


if __name__ == "__main__":
    unittest.main()
