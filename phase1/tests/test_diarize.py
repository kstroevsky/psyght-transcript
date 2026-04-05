"""Focused tests for diarization artifact normalization."""

from __future__ import annotations

import unittest

from phase1.pipeline.diarize import diarization_to_records


class _FakeSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"[{self.start}, {self.end}]"


class _FakeDiarization:
    def to_dict(self, orient: str = "records"):
        if orient != "records":
            raise AssertionError(f"unexpected orient: {orient}")
        return [
            {
                "segment": _FakeSegment(1.25, 2.5),
                "speaker": "SPEAKER_00",
                "label": "A",
            }
        ]


class DiarizeTest(unittest.TestCase):
    def test_diarization_to_records_strips_non_json_segment_objects(self) -> None:
        records = diarization_to_records(_FakeDiarization())

        self.assertEqual(
            [{"speaker": "SPEAKER_00", "label": "A", "start": 1.25, "end": 2.5}],
            records,
        )


if __name__ == "__main__":
    unittest.main()
