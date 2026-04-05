"""Tests for the therapy ECLM correction stage helpers."""

from __future__ import annotations

import unittest

from phase1.therapy.eclm import build_eclm_input, validate_eclm_candidate
from phase1.therapy.models import MergeRequest


class TherapyECLMTest(unittest.TestCase):
    def test_build_eclm_input_matches_training_format(self) -> None:
        request = MergeRequest(
            language="ru",
            start=0.0,
            end=1.0,
            ctc_text="ctc text",
            whisper_text="whisper text",
            canary_text="",
        )

        self.assertEqual("ctc: ctc text whisper: whisper text", build_eclm_input(request))

    def test_validate_eclm_candidate_rejects_wrong_script_for_russian(self) -> None:
        request = MergeRequest(
            language="ru",
            start=0.0,
            end=1.0,
            ctc_text="у меня пропали",
            whisper_text="мне пропали",
            canary_text="",
        )

        validation = validate_eclm_candidate(request, "you disappeared")

        self.assertFalse(validation["accepted"])
        self.assertIn("unexpected_script", validation["reasons"])


if __name__ == "__main__":
    unittest.main()
