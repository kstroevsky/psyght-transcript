"""Tests for the additive therapy backend orchestration."""

from __future__ import annotations

import unittest

from phase1.therapy.models import MergeDecision
from phase1.therapy.pipeline import TherapyHybridTranscriber, resolve_therapy_backend_options, validate_canary_segments


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
        return MergeDecision(
            text=" || ".join(part for part in (request.canary_text, request.ctc_text, request.whisper_text) if part)
        )


class _DropMerger:
    def merge(self, request):
        del request
        return MergeDecision(text="", confidence="low", notes=("language_script_guard_drop",))


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
        self.assertEqual("backend", resolved["canary"]["provider"])

    def test_resolve_therapy_backend_options_prefers_artifact_canary_over_live_backend(self) -> None:
        resolved = resolve_therapy_backend_options(
            {
                "use_live_canary": True,
                "canary_transcript_path": "/tmp/canary.json",
            }
        )

        self.assertEqual("artifact", resolved["canary"]["provider"])
        self.assertEqual("/tmp/canary.json", resolved["canary"]["artifact_path"])

    def test_validate_canary_segments_rejects_latin_only_forced_ru_output(self) -> None:
        validation = validate_canary_segments(
            [{"start": 0.0, "end": 1.0, "text": "hello there", "words": [{"word": "hello"}]}],
            language="ru",
        )

        self.assertFalse(validation["accepted"])
        self.assertEqual("accepted_with_warnings", validation["runtime_decision"])
        self.assertIn("latin_only_transcript", validation["fallback_reason"])

    def test_transcriber_uses_canary_segments_as_final_windows_and_guarded_whisper_as_aux(self) -> None:
        transcriber = self._transcriber()
        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "raw whisper"}]
        ctc_segments = [{"start": 0.0, "end": 1.0, "text": "ctc one"}]
        canary_segments = [{"start": 0.0, "end": 1.0, "text": "canary one"}]
        guarded_segments = [{"start": 0.0, "end": 1.0, "text": "guarded whisper", "source": "ctc_fallback", "confidence": "low"}]
        transcriber._whisper_stage = _StaticStage(whisper_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._ctc_stage = _StaticStage(ctc_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._canary_stage = _StaticStage(canary_segments, {"detected_language": "ru"})
        transcriber._hallucination_resolver = _StaticResolver(guarded_segments, {"flagged_segment_count": 1})
        transcriber._merger = _StaticMerger()

        segments, meta = transcriber.run(audio=[], language="ru")

        self.assertEqual("canary one || ctc one || guarded whisper", segments[0]["text"])
        self.assertEqual("merged", segments[0]["source"])
        self.assertIn("01e_therapy_merge_debug.json", meta["artifact_payloads"])
        self.assertEqual("guarded whisper", meta["artifact_payloads"]["01d_therapy_hallucination_segments.json"][0]["text"])
        self.assertEqual("canary one", meta["artifact_payloads"]["01e_therapy_merge_debug.json"][0]["canary_base_text"])
        self.assertIn("01f_therapy_eclm_debug.json", meta["artifact_payloads"])

    def test_transcriber_requires_canary_segments(self) -> None:
        transcriber = self._transcriber()
        transcriber._whisper_stage = _StaticStage([{"start": 0.0, "end": 1.0, "text": "whisper"}], {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._ctc_stage = _StaticStage([{"start": 0.0, "end": 1.0, "text": "ctc"}], {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._canary_stage = _StaticStage([], {})
        transcriber._hallucination_resolver = _PassthroughResolver()

        with self.assertRaisesRegex(RuntimeError, "requires a Canary transcript"):
            transcriber.run(audio=[], language="ru")

    def test_transcriber_skips_segments_when_merger_requests_language_guard_drop(self) -> None:
        transcriber = self._transcriber()
        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "english stray"}]
        ctc_segments = [{"start": 0.0, "end": 1.0, "text": ""}]
        canary_segments = [{"start": 0.0, "end": 1.0, "text": "canary stray"}]
        guarded_segments = [{"start": 0.0, "end": 1.0, "text": "english stray"}]
        transcriber._whisper_stage = _StaticStage(whisper_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._ctc_stage = _StaticStage(ctc_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._canary_stage = _StaticStage(canary_segments, {})
        transcriber._hallucination_resolver = _StaticResolver(guarded_segments, {"flagged_segment_count": 0})
        transcriber._merger = _DropMerger()

        segments, meta = transcriber.run(audio=[], language="ru")

        self.assertEqual([], segments)
        self.assertEqual([], meta["artifact_payloads"]["01e_therapy_merge_debug.json"])

    def test_transcriber_reports_eclm_runtime_disabled_even_when_requested(self) -> None:
        transcriber = TherapyHybridTranscriber({"merge_provider": {"id": "rule_based"}, "eclm": {"enabled": True, "model_path": "/tmp/eclm"}})
        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "да теперь все будет хорошо"}]
        ctc_segments = [{"start": 0.0, "end": 1.0, "text": "да теперь всё будет хорошо"}]
        canary_segments = [{"start": 0.0, "end": 1.0, "text": "да теперь все будет хорошо"}]
        transcriber._whisper_stage = _StaticStage(whisper_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._ctc_stage = _StaticStage(ctc_segments, {"detected_language": "ru", "asr": {"device": "cpu"}})
        transcriber._canary_stage = _StaticStage(canary_segments, {"detected_language": "ru"})
        transcriber._hallucination_resolver = _PassthroughResolver()

        segments, meta = transcriber.run(audio=[], language="ru")

        self.assertEqual("да теперь все будет хорошо", segments[0]["text"])
        self.assertTrue(meta["eclm"]["requested_enabled"])
        self.assertFalse(meta["eclm"]["enabled"])
        self.assertEqual("disabled_in_canary_primary_runtime", meta["eclm"]["skip_reason"])
        self.assertEqual([], meta["artifact_payloads"]["01f_therapy_eclm_debug.json"])


if __name__ == "__main__":
    unittest.main()
