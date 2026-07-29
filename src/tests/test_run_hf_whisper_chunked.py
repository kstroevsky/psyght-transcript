"""Focused tests for the ad hoc HF Whisper chunk runner helpers."""

from __future__ import annotations

import unittest

from phase1.tools.run_hf_whisper_chunked import build_next_prompt, extract_last_sentence


class RunHFWhisperChunkedTest(unittest.TestCase):
    def test_extract_last_sentence_prefers_last_sentence_boundary(self) -> None:
        text = "First sentence. Second sentence! Third sentence?"
        self.assertEqual("Third sentence?", extract_last_sentence(text))

    def test_extract_last_sentence_strips_leading_punctuation_noise(self) -> None:
        text = "!Но вы хотите, чтобы я сейчас начал их больше вписывать..."
        self.assertEqual("Но вы хотите, чтобы я сейчас начал их больше вписывать...", extract_last_sentence(text))

    def test_build_next_prompt_trims_to_max_chars_from_end(self) -> None:
        text = "Sentence one. " + ("x" * 40) + "!"
        prompt = build_next_prompt(text, max_chars=12)
        self.assertEqual(("x" * 11) + "!", prompt)


if __name__ == "__main__":
    unittest.main()
