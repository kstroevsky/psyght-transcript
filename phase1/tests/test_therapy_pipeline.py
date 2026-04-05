"""Tests for the additive therapy backend orchestration."""

from __future__ import annotations

import unittest

from phase1.therapy.models import MergeDecision
from phase1.therapy.pipeline import TherapyHybridTranscriber, resolve_therapy_backend_options, validate_primary_segments


class _StaticStage:
    def __init__(self, segments, meta):
        self._segments = segments
        self._meta = meta

    def run(self, audio, *, language):
        del audio, language
        return self._segments, self._meta


class _StaticResolver:
    def __init__(self, resolved_segments, meta):
        self._segments = resolved_segments
        self._meta = meta

    def resolve(self, whisper_segments, ctc_segments):
        del whisper_segments, ctc_segments
        return self._segments, self._meta


class _PassthroughResolver:
    def resolve(self, whisper_segments, ctc_segments):
        del ctc_segments
        return whisper_segments, {"flagged_segment_count": 0}


class _StaticMerger:
    def merge(self, request):
        return MergeDecision(text=f"{request.ctc_text} || {request.whisper_text}".strip(" |"))


class _DropMerger:
    def merge(self, request):
        del request
        return MergeDecision(text="", confidence="low", notes=("language_script_guard_drop",))


class _StaticCorrector:
    def __init__(self, text):
        self._text = text

    def correct(self, request):
        del request
        return MergeDecision(text=self._text, source="eclm")

    def runtime_summary(self):
        return {"provider": "mt5", "device": "cpu"}


class TherapyPipelineTest(unittest.TestCase):
    @staticmethod
    def _transcriber():
        return TherapyHybridTranscriber({"merge_provider": {"id": "rule_based"}, "eclm": {"enabled": False}})

    def test_resolve_therapy_backend_options_sets_expected_defaults(self) -> None:
        resolved = resolve_therapy_backend_options({})

        self.assertEqual("whisperx", resolved["speaker_assignment_strategy"])
        self.assertEqual("alpha", resolved["speaker_label_style"])
        self.assertEqual("bzikst/faster-whisper-large-v3-russian", resolved["whisper"]["model_name"])
        self.assertEqual("beam", resolved["ctc"]["decoder"]["strategy"])
        self.assertEqual(32, resolved["ctc"]["decoder"]["beam_width"])

    def test_resolve_therapy_backend_options_prefers_artifact_canary_over_live_backend(self) -> None:
        resolved = resolve_therapy_backend_options(
            {
                "use_live_canary": True,
                "canary_transcript_path": "/tmp/canary.json",
            }
        )

        self.assertEqual("artifact", resolved["canary"]["provider"])
        self.assertEqual("/tmp/canary.json", resolved["canary"]["artifact_path"])

    def test_validate_primary_segments_rejects_latin_only_forced_ru_output(self) -> None:
        validation = validate_primary_segments(
            [{"start": 0.0, "end": 1.0, "text": "hello there", "words": [{"word": "hello"}]}],
            language="ru",
        )

        self.assertFalse(validation["accepted"])
        self.assertEqual("ctc_primary_replacement", validation["runtime_decision"])
        self.assertIn("latin_only_transcript", validation["fallback_reason"])

    def test_transcriber_emits_merge_debug_and_keeps_ctc_fallback_segments(self) -> None:
        transcriber = self._transcriber()
        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "whisper one"}]
        ctc_segments = [{"start": 0.0, "end": 1.0, "text": "ctc one"}]
        canary_segments = [{"start": 0.0, "end": 1.0, "text": "canary one"}]
        guarded_segments = [
            {"start": 0.0, "end": 1.0, "text": "anchor only", "source": "ctc_fallback", "confidence": "low"},
            {"start": 1.0, "end": 2.0, "text": "whisper two"},
        ]
        transcriber._primary_stage = _StaticStage(whisper_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._ctc_stage = _StaticStage(
            ctc_segments + [{"start": 1.0, "end": 2.0, "text": "ctc two"}],
            {"detected_language": "ru", "asr": {"device": "cpu"}},
        )
        transcriber._canary_stage = _StaticStage(canary_segments + [{"start": 1.0, "end": 2.0, "text": "canary two"}], {})
        transcriber._hallucination_resolver = _StaticResolver(guarded_segments, {"flagged_segment_count": 1})
        transcriber._merger = _StaticMerger()

        segments, meta = transcriber.run(audio=[], language="ru")

        self.assertEqual("anchor only", segments[0]["text"])
        self.assertEqual("ctc two || whisper two", segments[1]["text"])
        self.assertEqual("ctc_fallback", segments[0]["source"])
        self.assertEqual("merged", segments[1]["source"])
        self.assertIn("01e_therapy_merge_debug.json", meta["artifact_payloads"])
        self.assertEqual(2, len(meta["artifact_payloads"]["01e_therapy_merge_debug.json"]))
        self.assertIn("01f_therapy_eclm_debug.json", meta["artifact_payloads"])

    def test_transcriber_replaces_invalid_primary_whisper_segments_with_ctc(self) -> None:
        transcriber = self._transcriber()
        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "hello there", "words": [{"word": "hello"}]}]
        ctc_segments = [{"start": 0.0, "end": 1.0, "text": "привет"}]
        transcriber._primary_stage = _StaticStage(
            whisper_segments,
            {
                "detected_language": "ru",
                "asr": {
                    "device": "cpu",
                    "model_name": "bzikst/faster-whisper-large-v3-russian",
                    "task": "transcribe",
                },
            },
        )
        transcriber._ctc_stage = _StaticStage(
            ctc_segments,
            {"detected_language": "ru", "asr": {"device": "cpu"}},
        )
        transcriber._canary_stage = _StaticStage([], {})
        transcriber._hallucination_resolver = _PassthroughResolver()

        segments, meta = transcriber.run(audio=[], language="ru")

        self.assertEqual("ctc_primary_replacement", meta["whisper_primary_runtime_decision"])
        self.assertEqual("привет", meta["artifact_payloads"]["01a_therapy_whisper_segments.json"][0]["text"])
        self.assertEqual("привет", segments[0]["text"])

    def test_transcriber_skips_segments_when_merger_requests_language_guard_drop(self) -> None:
        transcriber = self._transcriber()
        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "english stray"}]
        ctc_segments = [{"start": 0.0, "end": 1.0, "text": ""}]
        guarded_segments = [{"start": 0.0, "end": 1.0, "text": "english stray"}]
        transcriber._primary_stage = _StaticStage(whisper_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._ctc_stage = _StaticStage(ctc_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._canary_stage = _StaticStage([], {})
        transcriber._hallucination_resolver = _StaticResolver(guarded_segments, {"flagged_segment_count": 0})
        transcriber._merger = _DropMerger()

        segments, meta = transcriber.run(audio=[], language="ru")

        self.assertEqual([], segments)
        self.assertEqual([], meta["artifact_payloads"]["01e_therapy_merge_debug.json"])

    def test_transcriber_prefers_valid_eclm_output_over_merge_candidate(self) -> None:
        transcriber = self._transcriber()
        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "да теперь все будет хорошо"}]
        ctc_segments = [{"start": 0.0, "end": 1.0, "text": "да теперь всё будет хорошо"}]
        transcriber._primary_stage = _StaticStage(whisper_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._ctc_stage = _StaticStage(ctc_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._canary_stage = _StaticStage([], {})
        transcriber._hallucination_resolver = _PassthroughResolver()
        transcriber._corrector = _StaticCorrector("да теперь всё будет хорошо")

        segments, meta = transcriber.run(audio=[], language="ru")

        self.assertEqual("да теперь всё будет хорошо", segments[0]["text"])
        self.assertEqual("eclm", segments[0]["source"])
        self.assertTrue(meta["artifact_payloads"]["01f_therapy_eclm_debug.json"][0]["accepted"])


if __name__ == "__main__":
    unittest.main()
