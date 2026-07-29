"""Tests for therapy merge-stage selection rules."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from phase1.therapy.merge import (
    OllamaSegmentMerger,
    RuleBasedSegmentMerger,
    build_ollama_merge_payload,
    resolve_merge_instruction_variant,
)
from phase1.therapy.models import MergeRequest


class TherapyMergeTest(unittest.TestCase):
    def test_rule_based_merger_keeps_canary_as_base_when_inputs_disagree(self) -> None:
        merger = RuleBasedSegmentMerger()

        decision = merger.merge(
            MergeRequest(
                language="ru",
                start=0.0,
                end=1.0,
                ctc_text="немножко куда-то на секунду пропали",
                whisper_text="кажется связь на секунду отвалилась",
                canary_text="связь на секунду пропала",
            )
        )

        self.assertEqual("связь на секунду пропала", decision.text)
        self.assertEqual("canary_base", decision.source)
        self.assertEqual(("rule_based_canary_base",), decision.notes)

    def test_rule_based_merger_keeps_canary_when_ctc_is_empty(self) -> None:
        merger = RuleBasedSegmentMerger()

        decision = merger.merge(
            MergeRequest(
                language="ru",
                start=0.0,
                end=1.0,
                ctc_text="",
                whisper_text="текст из whisper",
                canary_text="текст из canary",
            )
        )

        self.assertEqual("текст из canary", decision.text)
        self.assertEqual(("missing_ctc_canary_base",), decision.notes)

    def test_rule_based_merger_uses_ctc_when_canary_is_foreign(self) -> None:
        merger = RuleBasedSegmentMerger()

        decision = merger.merge(
            MergeRequest(
                language="ru",
                start=0.0,
                end=4.0,
                ctc_text="немножко куда-то на секунду пропали",
                whisper_text="you disappeared for a second",
                canary_text="you disappeared for a second",
            )
        )

        self.assertEqual("немножко куда-то на секунду пропали", decision.text)
        self.assertEqual("ctc_anchor", decision.source)
        self.assertEqual(("canary_script_guard_ctc_anchor",), decision.notes)

    def test_rule_based_merger_uses_ctc_when_ctc_and_whisper_agree(self) -> None:
        merger = RuleBasedSegmentMerger()

        decision = merger.merge(
            MergeRequest(
                language="ru",
                start=0.0,
                end=1.0,
                ctc_text="правильная версия",
                whisper_text="правильная версия",
                canary_text="сомнительная версия",
            )
        )

        self.assertEqual("правильная версия", decision.text)
        self.assertEqual("ctc_consensus_override", decision.source)
        self.assertEqual(("ctc_whisper_consensus_override",), decision.notes)

    def test_ollama_merger_uses_non_thinking_deterministic_payload(self) -> None:
        merger = OllamaSegmentMerger(model="gemma4-27b")
        request = MergeRequest(
            language="ru",
            start=0.0,
            end=1.0,
            ctc_text="точный текст",
            whisper_text="альтернативный текст",
            canary_text="базовый текст",
        )

        with patch.object(merger, "_request_json", return_value={"response": "итог"}) as request_mock:
            decision = merger.merge(request)

        self.assertEqual("итог", decision.text)
        payload = request_mock.call_args.args[0]
        self.assertEqual("gemma4-27b", payload["model"])
        self.assertIn("Base transcript [Canary]", payload["prompt"])
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(0, payload["options"]["temperature"])
        self.assertEqual(1, payload["options"]["top_k"])
        self.assertEqual(96, payload["options"]["num_predict"])

    def test_payload_can_use_named_instruction_variant(self) -> None:
        payload = build_ollama_merge_payload(
            "gemma4-27b",
            MergeRequest(
                language="ru",
                start=0.0,
                end=1.0,
                ctc_text="точный текст",
                whisper_text="альтернативный текст",
                canary_text="базовый текст",
            ),
            instruction_variant="consensus_gate_v1",
        )

        self.assertIn("Instruction profile [consensus_gate_v1]", payload["prompt"])
        self.assertIn("Change Canary only if at least one gate passes", payload["prompt"])

    def test_unknown_instruction_variant_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported merge instruction variant"):
            resolve_merge_instruction_variant("not_a_real_variant")

    def test_ollama_merger_keeps_canary_when_ctc_is_missing(self) -> None:
        merger = OllamaSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json") as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="",
                    whisper_text="альтернативный вариант",
                    canary_text="базовый canary текст",
                )
            )

        self.assertEqual("базовый canary текст", decision.text)
        self.assertEqual(("missing_ctc_canary_base",), decision.notes)
        request_mock.assert_not_called()

    def test_ollama_merger_anchors_to_ctc_when_canary_is_foreign_for_russian(self) -> None:
        merger = OllamaSegmentMerger(model="qwen3:8b")

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
        self.assertEqual(("canary_script_guard_ctc_anchor",), decision.notes)
        request_mock.assert_not_called()

    def test_ollama_merger_rejects_unanchored_ollama_output(self) -> None:
        merger = OllamaSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json", return_value={"response": "полностью новая формулировка без опоры"}) as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="точная фраза из ctc",
                    whisper_text="вариант из whisper",
                    canary_text="базовая canary фраза",
                )
            )

        self.assertEqual("базовая canary фраза", decision.text)
        self.assertIn("low_anchor_overlap_ollama", decision.notes)
        request_mock.assert_called_once()

    def test_ollama_fallback_preserves_canary_base(self) -> None:
        merger = OllamaSegmentMerger(model="qwen3:8b")

        with patch.object(merger, "_request_json", side_effect=RuntimeError("offline")):
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=4.0,
                    ctc_text="знаете, точная версия из ctc",
                    whisper_text="вариант из whisper",
                    canary_text="базовая версия из canary",
                )
            )

        self.assertEqual("базовая версия из canary", decision.text)
        self.assertEqual("canary_base", decision.source)
        self.assertEqual(("ollama_fallback", "rule_based_canary_base"), decision.notes)

    def test_ollama_merger_honors_call_budget(self) -> None:
        merger = OllamaSegmentMerger(model="qwen3:8b", max_ollama_calls=0)

        with patch.object(merger, "_request_json") as request_mock:
            decision = merger.merge(
                MergeRequest(
                    language="ru",
                    start=0.0,
                    end=1.0,
                    ctc_text="точный акустический якорь слова",
                    whisper_text="альтернативный вариант фразы",
                    canary_text="базовая canary фраза",
                )
            )

        self.assertEqual("базовая canary фраза", decision.text)
        self.assertEqual(("ollama_call_budget_canary_base", "rule_based_canary_base"), decision.notes)
        request_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
