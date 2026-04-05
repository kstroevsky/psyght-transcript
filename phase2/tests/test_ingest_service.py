"""Tests for the refactored phase2 ingest service."""

from __future__ import annotations

import unittest
from pathlib import Path

from phase2.ingest_service import load_transcript


class IngestServiceTest(unittest.TestCase):
    """Verify that phase2 loads the persisted phase1 transcript contract unchanged."""

    def test_load_transcript_uses_shared_contract(self) -> None:
        fixture_path = Path(__file__).resolve().parents[2] / "phase1" / "test_first2min.json"
        transcript = load_transcript(fixture_path)

        self.assertEqual("en", transcript.language)
        self.assertEqual(119.999, transcript.duration)
        self.assertEqual("SPEAKER_UNKNOWN", transcript.segments[0].speaker)
        self.assertEqual("Hello,", transcript.segments[0].words[0].word)


if __name__ == "__main__":
    unittest.main()
