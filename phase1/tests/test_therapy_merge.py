"""Tests for therapy merge-stage selection rules."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from phase1.therapy.merge import OllamaQwenSegmentMerger, RuleBasedSegmentMerger
from phase1.therapy.models import MergeRequest


class TherapyMergeTest(unittest.TestCase):
    def test_rule_based_merger_prefers_ctc_anchor_when_both_inputs_exist(self) -> None:
        merger = RuleBasedSegmentMerger()

        decision = merger.merge(
            MergeRequest(
                language="ru",
                start=0.0,
                end=1.0,
                ctc_text="немножко куда-то на секунду пропали",
                whisper_text="you disappeared for a second",
                canary_text="you disappeared for a second",
            )
        )

        self.assertEqual("немножко куда-то на секунду пропали", decision.text)
        self.assertEqual(("rule_based_ctc_anchor",), decision.notes)

    def test_rule_based_merger_still_uses_whisper_when_ctc_is_empty(self) -> None:
        merger = RuleBasedSegmentMerger()

        decision = merger.merge(
            MergeRequest(
                language="ru",
                start=0.0,
                end=1.0,
                ctc_text="",
                whisper_text="текст из whisper",
                canary_text="",
            )
        )

        self.assertEqual("текст из whisper", decision.text)

    def test_rule_based_merger_uses_whisper_when_ctc_is_repetitive_low_value_text(self) -> None:
        merger = RuleBasedSegmentMerger()

        decision = merger.merge(
            MergeRequest(
                language="ru",
                start=0.0,
                end=4.0,
                ctc_text="знаете, я встал и зани, что у меня даже волнуюсь волнуюсь",
                whisper_text="Знаете, я встал и заметил, что у меня даже потело тело.",
                canary_text="",
            )
        )

        self.assertEqual("Знаете, я встал и заметил, что у меня даже потело тело.", decision.text)
        self.assertEqual("whisper_semantic_override", decision.source)
        self.assertEqual(("whisper_semantic_override", "rule_based_ctc_anchor"), decision.notes)

    def test_ollama_merger_uses_non_thinking_deterministic_payload(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b")
        request = MergeRequest(
            language="ru",
            start=0.0,
            end=1.0,
            ctc_text="ctc",
            whisper_text="whisper",
            canary_text="canary",
        )

        with patch.object(merger, "_request_json", return_value={"response": "итог"}) as request_mock:
            decision = merger.merge(request)

        self.assertEqual("итог", decision.text)
        payload = request_mock.call_args.args[0]
        self.assertEqual("qwen3:8b", payload["model"])
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(0, payload["options"]["temperature"])
        self.assertEqual(1, payload["options"]["top_k"])
        self.assertEqual(96, payload["options"]["num_predict"])

    def test_ollama_merger_anchors_to_ctc_when_competing_text_is_foreign_for_russian(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json") as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="немножко куда-то на секунду пропали",
                    whisper_text="you disappeared for a second",
                    canary_text="i am going to the next one",
                )
            )

        self.assertEqual("немножко куда-то на секунду пропали", decision.text)
        self.assertEqual(("language_script_guard_ctc_anchor", "rule_based_ctc_anchor"), decision.notes)
        request_mock.assert_not_called()

    def test_ollama_merger_drops_foreign_segment_without_russian_anchor(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json") as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="",
                    whisper_text="yes now everything is turned off",
                    canary_text="",
                )
            )

        self.assertEqual("", decision.text)
        self.assertEqual(("language_script_guard_drop",), decision.notes)
        request_mock.assert_not_called()

    def test_ollama_merger_rejects_foreign_ollama_output_when_ctc_has_russian_anchor(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json", return_value={"response": "yes everything is turned off"}) as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="немножко куда-то на секунду пропали",
                    whisper_text="кажется связь на секунду отвалилась",
                    canary_text="",
                )
            )

        self.assertEqual("немножко куда-то на секунду пропали", decision.text)
        self.assertEqual(("missing_canary_ctc_anchor",), decision.notes)
        request_mock.assert_not_called()

    def test_ollama_merger_skips_model_when_canary_is_missing(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json") as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="немножко куда-то на секунду пропали",
                    whisper_text="немножко куда-то на секунду пропали сейчас",
                    canary_text="",
                )
            )

        self.assertEqual("немножко куда-то на секунду пропали", decision.text)
        self.assertEqual(("missing_canary_ctc_anchor",), decision.notes)
        request_mock.assert_not_called()

    def test_ollama_merger_uses_whisper_for_short_ctc_boundary_leak(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json") as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="волнуюсь почему",
                    whisper_text="А почему?",
                    canary_text="неважный сегмент canary",
                )
            )

        self.assertEqual("А почему?", decision.text)
        self.assertEqual("whisper_semantic_override", decision.source)
        self.assertEqual(("whisper_semantic_override", "short_ctc_anchor"), decision.notes)
        request_mock.assert_not_called()

    def test_ollama_merger_uses_whisper_when_high_overlap_ctc_repeats_tokens(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json") as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=2.0,
                    ctc_text="сейчас правда тревожно тревожно очень сильно",
                    whisper_text="сейчас правда тревожно очень сильно",
                    canary_text="сейчас правда тревожно очень сильно",
                )
            )

        self.assertEqual("сейчас правда тревожно очень сильно", decision.text)
        self.assertEqual("whisper_semantic_override", decision.source)
        self.assertEqual(("whisper_semantic_override", "high_overlap_ctc_anchor"), decision.notes)
        request_mock.assert_not_called()

    def test_ollama_merger_keeps_ctc_for_valid_short_anchor(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json") as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="Отлично.",
                    whisper_text="Отлично!",
                    canary_text="отлично",
                )
            )

        self.assertEqual("Отлично.", decision.text)
        self.assertEqual(("short_ctc_anchor",), decision.notes)
        request_mock.assert_not_called()

    def test_ollama_fallback_preserves_whisper_override_source(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json", side_effect=RuntimeError("offline")):
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=4.0,
                    ctc_text="знаете, я встал и зани, что у меня даже волнуюсь волнуюсь",
                    whisper_text="Знаете, я встал и заметил, что у меня даже потело тело.",
                    canary_text="я встал и звонил то что у меня даже нет поделать и",
                )
            )

        self.assertEqual("Знаете, я встал и заметил, что у меня даже потело тело.", decision.text)
        self.assertEqual("whisper_semantic_override", decision.source)
        self.assertEqual(("ollama_fallback", "whisper_semantic_override", "rule_based_ctc_anchor"), decision.notes)

    def test_ollama_merger_honors_call_budget(self) -> None:
        merger = OllamaQwenSegmentMerger(model="qwen3:8b", max_ollama_calls=0)

        with patch.object(merger, "_request_json") as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="точный акустический якорь слова",
                    whisper_text="альтернативный вариант фразы",
                    canary_text="другой вариант фразы",
                )
            )

        self.assertEqual("точный акустический якорь слова", decision.text)
        self.assertEqual(("ollama_call_budget_ctc_anchor",), decision.notes)
        request_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
