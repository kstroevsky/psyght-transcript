"""Tests for the phase1 evaluation layer (phase1/eval).

These assert the canonical WER/CER *rate* (deterministic and used for ranking)
and unambiguous S/D/I cases, but avoid asserting the S/D/I split on inputs with
multiple optimal alignments, since that attribution legitimately depends on
whether rapidfuzz is installed.
"""

from __future__ import annotations

import unittest

from contracts.transcript import Segment, Transcript, Word
from phase1.eval import (
    RU_FOLD_PROFILE,
    ReferenceTranscript,
    RefSegment,
    char_error_rate,
    cp_word_error_rate,
    diarization_error_rate,
    normalize_text,
    reference_error_rates,
    reference_from_payload,
    render_word_diff,
    score_transcript,
    word_error_rate,
)


class NormalizeTest(unittest.TestCase):
    def test_default_profile_matches_legacy_behavior(self) -> None:
        self.assertEqual("привет как дела", normalize_text("Привет, как дела?!"))
        self.assertEqual("a b c", normalize_text("  a   b\nc "))

    def test_ru_fold_profile_folds_yo(self) -> None:
        self.assertEqual("все ежик", normalize_text("Всё ёжик", RU_FOLD_PROFILE))

    def test_collapse_repeats_profile(self) -> None:
        self.assertEqual("да нет", normalize_text("да да да нет", "ru_fold_repeats"))


class WerTest(unittest.TestCase):
    def test_identical_is_zero(self) -> None:
        result = word_error_rate("раз два три", "раз два три")
        self.assertEqual(0.0, result.rate)
        self.assertEqual(3, result.hits)

    def test_pure_deletion(self) -> None:
        result = word_error_rate("раз два три четыре", "раз два три")
        self.assertEqual(0.25, result.rate)
        self.assertEqual(1, result.deletions)
        self.assertEqual(0, result.insertions)

    def test_pure_insertion(self) -> None:
        result = word_error_rate("раз два", "раз два три")
        self.assertEqual(0.5, result.rate)
        self.assertEqual(1, result.insertions)

    def test_empty_reference_with_hypothesis_is_full_error(self) -> None:
        self.assertEqual(1.0, word_error_rate("", "что-то").rate)
        self.assertEqual(0.0, word_error_rate("", "").rate)

    def test_cer_matches_known_value(self) -> None:
        self.assertEqual(0.4, char_error_rate("раз два три четыре", "раз два три").rate)

    def test_reference_error_rates_mapping(self) -> None:
        rates = reference_error_rates("привет как дела", "привет дела")
        self.assertEqual({"wer", "cer"}, set(rates))
        self.assertAlmostEqual(0.333333, rates["wer"], places=5)


class DiffTest(unittest.TestCase):
    def test_inline_markers(self) -> None:
        rendered = render_word_diff("раз два три", "раз две")
        self.assertIn("{два|две}", rendered)  # substitution
        self.assertIn("[-три-]", rendered)  # deletion (in ref, missing from hyp)

    def test_only_errors_elides_correct_runs(self) -> None:
        ref = "a b c d e f g h"
        hyp = "a b c d X f g h"
        rendered = render_word_diff(ref, hyp, only_errors=True, context=1)
        self.assertIn("{e|x}", rendered.lower())
        self.assertNotIn(" a ", f" {rendered} ")  # far-away correct tokens elided


class ReferenceTest(unittest.TestCase):
    def test_gemini_segment_list_infers_ends(self) -> None:
        payload = [
            {"speaker": "A", "start_sec": 0.0, "text": "привет"},
            {"speaker": "B", "start_sec": 2.0, "text": "как дела"},
        ]
        ref = reference_from_payload(payload)
        self.assertTrue(ref.has_speakers)
        self.assertTrue(ref.has_times)
        self.assertTrue(ref.inferred_ends)
        self.assertEqual(2.0, ref.segments[0].end)

    def test_phase1_transcript_payload(self) -> None:
        payload = {
            "language": "ru",
            "duration": 3.0,
            "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "привет"}],
        }
        ref = reference_from_payload(payload)
        self.assertEqual("ru", ref.language)
        self.assertEqual("привет", ref.segments[0].text)


class SpeakerTest(unittest.TestCase):
    def test_cp_wer_is_speaker_permutation_invariant(self) -> None:
        reference = {"A": "привет мир", "B": "как дела"}
        hypothesis = {"X": "как дела", "Y": "привет мир"}
        result = cp_word_error_rate(reference, hypothesis)
        self.assertEqual(0.0, result.cp_wer)
        self.assertEqual("A", result.mapping["Y"])

    def test_cp_wer_counts_content_errors(self) -> None:
        result = cp_word_error_rate({"A": "раз два три четыре"}, {"A": "раз два три"})
        self.assertEqual(0.25, result.cp_wer)

    def test_der_zero_for_matching_diarization(self) -> None:
        ref = [(0.0, 2.0, "A"), (2.0, 4.0, "B")]
        self.assertEqual(0.0, diarization_error_rate(ref, ref)["der"])

    def test_der_penalizes_speaker_confusion(self) -> None:
        ref = [(0.0, 2.0, "A"), (2.0, 4.0, "B")]
        hyp = [(0.0, 4.0, "A")]  # one speaker swallows both turns
        self.assertGreater(diarization_error_rate(ref, hyp)["der"], 0.4)


class ScoreTranscriptTest(unittest.TestCase):
    def _hypothesis(self) -> Transcript:
        return Transcript(
            language="ru",
            duration=4.0,
            segments=[
                Segment(speaker="A", start=0.0, end=2.0, text="привет мир", words=[Word("привет", 0.0, 1.0), Word("мир", 1.0, 2.0)]),
                Segment(speaker="B", start=2.0, end=4.0, text="как дела", words=[Word("как", 2.0, 3.0), Word("дела", 3.0, 4.0)]),
            ],
        )

    def test_full_score_with_speakers_and_times(self) -> None:
        reference = ReferenceTranscript(
            segments=(
                RefSegment(speaker="A", start=0.0, end=2.0, text="привет мир"),
                RefSegment(speaker="B", start=2.0, end=4.0, text="как дела"),
            ),
            language="ru",
        )
        result = score_transcript(reference, self._hypothesis())
        self.assertEqual(0.0, result["wer"])
        self.assertEqual(0.0, result["cp_wer"])
        self.assertEqual(0.0, result["der"])
        self.assertEqual({}, result["skipped"])

    def test_skips_speaker_metrics_without_reference_speakers(self) -> None:
        reference = ReferenceTranscript(
            segments=(RefSegment(speaker=None, start=None, end=None, text="привет мир как дела"),)
        )
        result = score_transcript(reference, self._hypothesis())
        self.assertIn("cp_wer", result["skipped"])
        self.assertIn("der", result["skipped"])
        self.assertNotIn("cp_wer", result)


if __name__ == "__main__":
    unittest.main()
