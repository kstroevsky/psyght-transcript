"""Tests for pure phase2 row-building helpers."""

from __future__ import annotations

import json
import unittest

from contracts.transcript import Segment, Transcript, Word
from phase2.db.rows import build_segment_rows


class RowBuilderTest(unittest.TestCase):
    """Verify that segment rows preserve ordering and serialized word payloads."""

    def test_build_segment_rows_preserves_segment_order_and_words(self) -> None:
        transcript = Transcript(
            language="en",
            duration=2.0,
            segments=[
                Segment(
                    speaker="SPEAKER_00",
                    start=0.0,
                    end=1.0,
                    text="first",
                    words=[Word(word="first", start=0.0, end=0.4, score=0.8, speaker="SPEAKER_00")],
                ),
                Segment(speaker="SPEAKER_01", start=1.0, end=2.0, text="second"),
            ],
        )
        rows = build_segment_rows("meeting-1", transcript, [[0.1, 0.2], [0.3, 0.4]])

        self.assertEqual("first", rows[0][4])
        self.assertEqual("second", rows[1][4])
        self.assertEqual([0.1, 0.2], rows[0][6])
        self.assertEqual([], json.loads(rows[1][5]))
        self.assertEqual("first", json.loads(rows[0][5])[0]["word"])
        self.assertEqual("SPEAKER_00", json.loads(rows[0][5])[0]["speaker"])


if __name__ == "__main__":
    unittest.main()
