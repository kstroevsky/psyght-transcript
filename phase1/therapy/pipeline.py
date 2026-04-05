"""Composable therapy-focused transcription stages built on top of phase1 backends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import Any

from phase1.backends.gigaam_ctc_backend import GigaAMCTCBackend
from phase1.backends.whisperx_backend import WhisperXBackend
from phase1.config.env import BASE_DIR
from phase1.therapy.canary import ArtifactCanaryProvider, BackendCanaryProvider, EmptyCanaryProvider
from phase1.therapy.eclm import build_segment_corrector, should_try_eclm, validate_eclm_candidate
from phase1.therapy.hallucination import CTCFallbackHallucinationResolver
from phase1.therapy.merge import merger_from_options
from phase1.therapy.models import MergeRequest
from phase1.therapy.windows import normalize_text, text_for_window

DEFAULT_THERAPY_WHISPER_MODEL = "bzikst/faster-whisper-large-v3-russian"
DEFAULT_THERAPY_WHISPER_SOURCE_MODEL = "antony66/whisper-large-v3-russian"
DEFAULT_THERAPY_ECLM_MODEL = str((BASE_DIR / "models" / "therapy_eclm" / "latest").resolve())
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def default_external_ru_lm_paths() -> tuple[str | None, str | None]:
    """Return the current best external RU KenLM paths when present locally."""

    binary = Path("/tmp/gigaam-ctc-with-lm/language_model/ru_3gram.bin")
    unigrams = Path("/tmp/gigaam-ctc-with-lm/language_model/unigrams.txt")
    return (str(binary) if binary.exists() else None, str(unigrams) if unigrams.exists() else None)


def _segment_text_stats(segment: dict[str, Any]) -> tuple[int, int, bool, bool]:
    text = normalize_text(str(segment.get("text") or ""))
    total_words = len(text.split()) if text else 0
    timed_words = sum(1 for word in segment.get("words", []) if "start" in word and "end" in word)
    has_cyrillic = bool(_CYRILLIC_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    return total_words, timed_words, has_cyrillic, has_latin


def validate_primary_segments(
    segments: list[dict[str, Any]],
    *,
    language: str,
) -> dict[str, Any]:
    """Validate forced-language therapy Whisper output before downstream stages."""

    language_code = str(language or "").lower()
    segment_count = len(segments)
    total_words = 0
    timed_words = 0
    cyrillic_segment_count = 0
    latin_only_segment_count = 0
    timed_segment_count = 0

    for segment in segments:
        segment_words, segment_timed_words, has_cyrillic, has_latin = _segment_text_stats(segment)
        total_words += segment_words
        timed_words += segment_timed_words
        if segment_timed_words:
            timed_segment_count += 1
        if has_cyrillic:
            cyrillic_segment_count += 1
        if has_latin and not has_cyrillic:
            latin_only_segment_count += 1

    timed_word_ratio = (timed_words / total_words) if total_words else 0.0
    reasons: list[str] = []
    if language_code in {"ru", "uk"}:
        if segment_count and latin_only_segment_count == segment_count:
            reasons.append("latin_only_transcript")
        elif segment_count and latin_only_segment_count >= max(2, segment_count // 2):
            reasons.append("predominantly_latin_transcript")
        if cyrillic_segment_count == 0 and latin_only_segment_count > 0 and timed_word_ratio < 0.2:
            reasons.append("alignment_coverage_incompatible_with_transcript")

    accepted = not reasons
    return {
        "segment_count": segment_count,
        "total_words": total_words,
        "timed_words": timed_words,
        "timed_word_ratio": round(timed_word_ratio, 3),
        "timed_segment_count": timed_segment_count,
        "cyrillic_segment_count": cyrillic_segment_count,
        "latin_only_segment_count": latin_only_segment_count,
        "accepted": accepted,
        "runtime_decision": "accepted_primary" if accepted else "ctc_primary_replacement",
        "fallback_reason": "; ".join(reasons) if reasons else None,
    }


def resolve_therapy_backend_options(backend_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize backend options for the additive therapy pipeline."""

    backend_options = dict(backend_options or {})

    whisper_options = dict(backend_options.get("whisper") or {})
    whisper_options.setdefault("model_name", str(backend_options.get("whisper_model_name") or DEFAULT_THERAPY_WHISPER_MODEL))
    whisper_options.setdefault("source_model_name", str(backend_options.get("whisper_source_model_name") or DEFAULT_THERAPY_WHISPER_SOURCE_MODEL))
    whisper_options.setdefault("vad_method", str(backend_options.get("vad_method") or whisper_options.get("vad_method") or "silero"))
    if backend_options.get("device") is not None and whisper_options.get("device") is None:
        whisper_options["device"] = backend_options["device"]
    if backend_options.get("compute_type") is not None and whisper_options.get("compute_type") is None:
        whisper_options["compute_type"] = backend_options["compute_type"]

    ctc_options = dict(backend_options.get("ctc") or {})
    ctc_options.setdefault("model_name", str(backend_options.get("ctc_model_name") or "ai-sage/GigaAM-v3"))
    ctc_options.setdefault("revision", str(backend_options.get("ctc_revision") or "e2e_ctc"))
    ctc_options.setdefault(
        "chunk_duration_sec",
        float(ctc_options["chunk_duration_sec"]) if "chunk_duration_sec" in ctc_options else 20.0,
    )
    if backend_options.get("device") is not None and ctc_options.get("device") is None:
        ctc_options["device"] = backend_options["device"]

    decoder = dict(ctc_options.get("decoder") or {})
    decoder.setdefault("strategy", "beam")
    decoder.setdefault("beam_width", int(decoder["beam_width"]) if "beam_width" in decoder else 32)
    lm_options = dict(decoder.get("lm") or {})
    lm_options.setdefault("enabled", True)
    lm_options.setdefault("alpha", float(lm_options["alpha"]) if "alpha" in lm_options else 0.1)
    lm_options.setdefault("beta", float(lm_options["beta"]) if "beta" in lm_options else 0.0)
    external_binary_path, external_unigrams_path = default_external_ru_lm_paths()
    if external_binary_path and lm_options.get("binary_path") is None:
        lm_options["binary_path"] = external_binary_path
    if external_unigrams_path and lm_options.get("unigrams_path") is None:
        lm_options["unigrams_path"] = external_unigrams_path
    decoder["lm"] = lm_options
    ctc_options["decoder"] = decoder

    merge_provider_options = dict(backend_options.get("merge_provider") or {})
    canary_options = dict(backend_options.get("canary") or {})
    if backend_options.get("use_live_canary"):
        canary_options.setdefault("provider", "backend")
    if backend_options.get("canary_transcript_path") is not None:
        canary_options["provider"] = "artifact"
        canary_options["artifact_path"] = str(backend_options["canary_transcript_path"])
    canary_options.setdefault("backend_id", "canary")

    eclm_options = dict(backend_options.get("eclm") or {})
    if backend_options.get("eclm_model_path") is not None and eclm_options.get("model_path") is None:
        eclm_options["model_path"] = str(backend_options["eclm_model_path"])
    if backend_options.get("eclm_device") is not None and eclm_options.get("device") is None:
        eclm_options["device"] = str(backend_options["eclm_device"])
    eclm_options.setdefault("model_path", DEFAULT_THERAPY_ECLM_MODEL)
    if "enabled" not in eclm_options:
        eclm_options["enabled"] = Path(str(eclm_options["model_path"])).expanduser().exists()
    eclm_options.setdefault("provider", "mt5" if eclm_options["enabled"] else "none")
    eclm_options.setdefault("max_source_length", 512)
    eclm_options.setdefault("max_new_tokens", 128)
    eclm_options.setdefault("num_beams", 1)
    eclm_options.setdefault("repetition_penalty", 1.05)

    return {
        "whisper": whisper_options,
        "ctc": ctc_options,
        "merge_provider": merge_provider_options,
        "canary": canary_options,
        "eclm": eclm_options,
        "speaker_assignment_strategy": str(backend_options.get("speaker_assignment_strategy") or "whisperx"),
        "speaker_label_style": str(backend_options.get("speaker_label_style") or "alpha"),
        "merge_consecutive_speakers": bool(backend_options.get("merge_consecutive_speakers", False)),
        "speaker_merge_gap_sec": float(backend_options.get("speaker_merge_gap_sec", 0.35)),
    }


class WhisperXPrimaryStage:
    """Primary therapy pass: WhisperX VAD transcription followed by word alignment."""

    def __init__(self, backend: WhisperXBackend | None = None, options: dict[str, Any] | None = None) -> None:
        self._backend = backend or WhisperXBackend()
        self._options = dict(options or {})

    def run(self, audio: Any, *, language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        asr_segments, detected_language, asr_meta = self._backend.transcribe(
            audio,
            language,
            backend_options=self._options,
            asr_options={"condition_on_previous_text": False},
            return_meta=True,
        )
        aligned_segments, alignment_meta = self._backend.align(
            asr_segments,
            detected_language,
            audio,
            backend_options=self._options,
            return_meta=True,
        )
        return aligned_segments, {
            "detected_language": detected_language,
            "asr": asr_meta,
            "alignment": alignment_meta,
        }


class CTCAnchorStage:
    """Secondary therapy pass: best current Russian CTC beam+LM anchor."""

    def __init__(self, backend: GigaAMCTCBackend | None = None, options: dict[str, Any] | None = None) -> None:
        self._backend = backend or GigaAMCTCBackend()
        self._options = dict(options or {})

    def run(self, audio: Any, *, language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del language
        ctc_segments, detected_language, asr_meta = self._backend.transcribe(
            audio,
            "ru",
            backend_options=self._options,
            return_meta=True,
        )
        aligned_segments, alignment_meta = self._backend.align(
            ctc_segments,
            detected_language,
            audio,
            backend_options=self._options,
            return_meta=True,
        )
        return aligned_segments, {
            "detected_language": detected_language,
            "asr": asr_meta,
            "alignment": alignment_meta,
        }


class TherapyHybridTranscriber:
    """Compose the therapy pipeline stages into one backend-facing transcriber."""

    def __init__(self, backend_options: dict[str, Any] | None = None) -> None:
        resolved = resolve_therapy_backend_options(backend_options)
        self._resolved_options = resolved
        self._primary_stage = WhisperXPrimaryStage(options=resolved["whisper"])
        self._ctc_stage = CTCAnchorStage(options=resolved["ctc"])
        canary_options = dict(resolved.get("canary") or {})
        provider = str(canary_options.get("provider") or "none").lower()
        if provider == "artifact":
            self._canary_stage = ArtifactCanaryProvider(str(canary_options["artifact_path"]))
        elif provider == "backend":
            self._canary_stage = BackendCanaryProvider(dict(canary_options.get("backend_options") or {}))
        else:
            self._canary_stage = EmptyCanaryProvider()
        self._hallucination_resolver = CTCFallbackHallucinationResolver()
        self._merger = merger_from_options(resolved["merge_provider"])
        self._corrector = build_segment_corrector(resolved["eclm"])
        self._eclm_options = dict(resolved.get("eclm") or {})

    def _resolve_primary_segments(
        self,
        whisper_segments: list[dict[str, Any]],
        ctc_segments: list[dict[str, Any]],
        *,
        language: str,
        whisper_meta: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        validation = validate_primary_segments(whisper_segments, language=language)
        validation["model_name"] = ((whisper_meta.get("asr") or {}).get("model_name"))
        validation["task"] = ((whisper_meta.get("asr") or {}).get("task"))
        if validation["accepted"]:
            return whisper_segments, validation
        return [dict(segment) for segment in ctc_segments], validation

    def _merge_segments(
        self,
        *,
        whisper_segments: list[dict[str, Any]],
        ctc_segments: list[dict[str, Any]],
        canary_segments: list[dict[str, Any]],
        language: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        final_segments: list[dict[str, Any]] = []
        merge_debug: list[dict[str, Any]] = []
        eclm_debug: list[dict[str, Any]] = []
        for segment in whisper_segments:
            payload = dict(segment)
            start = float(payload.get("start", 0.0))
            end = float(payload.get("end", start))
            whisper_text = normalize_text(str(payload.get("text") or ""))
            ctc_text = text_for_window(ctc_segments, start, end)
            canary_text = text_for_window(canary_segments, start, end)
            request = MergeRequest(
                language=language,
                start=start,
                end=end,
                ctc_text=ctc_text,
                whisper_text=whisper_text,
                canary_text=canary_text,
            )

            if payload.get("source") == "ctc_fallback":
                selected_text = normalize_text(str(payload.get("text") or ctc_text))
                selected_confidence = str(payload.get("confidence") or "low")
                selected_source = str(payload.get("source") or "ctc_fallback")
                notes: tuple[str, ...] = ("hallucination_guard",)
                merged_text = selected_text
            else:
                decision = self._merger.merge(request)
                decision_text = normalize_text(decision.text)
                if decision_text:
                    merged_text = decision_text
                elif "language_script_guard_drop" in decision.notes:
                    merged_text = ""
                else:
                    merged_text = normalize_text(whisper_text or ctc_text or canary_text)
                selected_text = merged_text
                selected_confidence = str(decision.confidence or "high")
                selected_source = str(decision.source or "merged")
                notes = tuple(decision.notes)

                if self._corrector is not None and merged_text:
                    correction_row: dict[str, Any] = {
                        "start": start,
                        "end": end,
                        "ctc_text": ctc_text,
                        "whisper_text": whisper_text,
                        "merged_text": merged_text,
                    }
                    if should_try_eclm(request):
                        try:
                            correction = self._corrector.correct(request)
                            validation = validate_eclm_candidate(request, correction.text)
                            correction_row["eclm_text"] = validation["candidate_text"]
                            correction_row["token_overlap_ratio"] = validation["token_overlap_ratio"]
                            correction_row["validation_reasons"] = list(validation["reasons"])
                            correction_row["accepted"] = bool(validation["accepted"])
                            if validation["accepted"] and validation["candidate_text"]:
                                selected_text = validation["candidate_text"]
                                selected_confidence = str(correction.confidence or selected_confidence)
                                selected_source = str(correction.source or "eclm")
                        except Exception as exc:  # pragma: no cover - defensive runtime fallback
                            correction_row["eclm_text"] = ""
                            correction_row["token_overlap_ratio"] = 0.0
                            correction_row["validation_reasons"] = [f"runtime_error:{type(exc).__name__}"]
                            correction_row["accepted"] = False
                    else:
                        correction_row["eclm_text"] = ""
                        correction_row["token_overlap_ratio"] = 0.0
                        correction_row["validation_reasons"] = ["skipped_by_gate"]
                        correction_row["accepted"] = False
                    correction_row["selected_text"] = selected_text
                    correction_row["selected_source"] = selected_source
                    eclm_debug.append(correction_row)

            if not selected_text:
                continue

            final_payload = {
                "start": start,
                "end": end,
                "text": selected_text,
                "confidence": selected_confidence,
                "source": selected_source,
            }
            final_segments.append(final_payload)
            merge_debug.append(
                {
                    "start": start,
                    "end": end,
                    "whisper_text": whisper_text,
                    "ctc_text": ctc_text,
                    "canary_text": canary_text,
                    "final_text": merged_text,
                    "selected_text": selected_text,
                    "confidence": final_payload["confidence"],
                    "source": final_payload["source"],
                    "notes": list(notes),
                }
            )
        return final_segments, merge_debug, eclm_debug

    def run(
        self,
        audio: Any,
        *,
        language: str = "ru",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run the full additive therapy pipeline and return merged ASR segments."""

        if language not in {"ru", None}:
            raise ValueError("therapy_hybrid backend currently supports only Russian recordings.")
        effective_language = str(language or "ru")

        with ThreadPoolExecutor(max_workers=2) as pool:
            whisper_future = pool.submit(self._primary_stage.run, audio, language=effective_language)
            ctc_future = pool.submit(self._ctc_stage.run, audio, language=effective_language)
            whisper_segments, whisper_meta = whisper_future.result()
            ctc_segments, ctc_meta = ctc_future.result()

        effective_primary_segments, whisper_primary_validation = self._resolve_primary_segments(
            whisper_segments,
            ctc_segments,
            language=effective_language,
            whisper_meta=whisper_meta,
        )
        canary_segments, canary_meta = self._canary_stage.run(audio, language=effective_language)
        guarded_segments, hallucination_meta = self._hallucination_resolver.resolve(effective_primary_segments, ctc_segments)
        final_segments, merge_debug, eclm_debug = self._merge_segments(
            whisper_segments=guarded_segments,
            ctc_segments=ctc_segments,
            canary_segments=canary_segments,
            language=effective_language,
        )

        return final_segments, {
            "device": (whisper_meta.get("asr") or {}).get("device"),
            "compute_type": (whisper_meta.get("asr") or {}).get("compute_type"),
            "detected_language": whisper_meta.get("detected_language", effective_language),
            "whisper": whisper_meta,
            "ctc": ctc_meta,
            "canary": canary_meta,
            "hallucination_guard": hallucination_meta,
            "eclm": {
                "enabled": self._corrector is not None,
                **self._eclm_options,
                **(self._corrector.runtime_summary() if self._corrector is not None else {}),
            },
            "whisper_primary_validation": whisper_primary_validation,
            "whisper_primary_runtime_decision": whisper_primary_validation["runtime_decision"],
            "whisper_primary_fallback_reason": whisper_primary_validation.get("fallback_reason"),
            "merge_provider": dict(self._resolved_options.get("merge_provider") or {}),
            "artifact_payloads": {
                "01a_therapy_whisper_segments.json": effective_primary_segments,
                "01b_therapy_ctc_segments.json": ctc_segments,
                "01c_therapy_canary_segments.json": canary_segments,
                "01d_therapy_hallucination_segments.json": guarded_segments,
                "01e_therapy_merge_debug.json": merge_debug,
                "01f_therapy_eclm_debug.json": eclm_debug,
            },
        }
