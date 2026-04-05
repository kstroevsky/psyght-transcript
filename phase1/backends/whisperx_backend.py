"""WhisperX backend adapter used by the phase1 runtime and compare wrapper."""

from __future__ import annotations

import gc
from threading import Lock
from typing import Any

from phase1.config import (
    get_alignment_runtime_config,
    get_asr_runtime_config,
    resolve_asr_settings,
    resolve_detect_model,
    resolve_lang_model_map,
)


def _sanitize_words(words: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None, float | None]:
    sanitized: list[dict[str, Any]] = []
    min_start: float | None = None
    max_end: float | None = None
    for word in words:
        payload = dict(word)
        if "start" in payload:
            start = float(payload["start"])
            end = float(payload.get("end", start))
            if end < start:
                end = start
            payload["start"] = start
            payload["end"] = end
            min_start = start if min_start is None else min(min_start, start)
            max_end = end if max_end is None else max(max_end, end)
        sanitized.append(payload)
    return sanitized, min_start, max_end


def _sanitize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized_segments: list[dict[str, Any]] = []
    for segment in segments:
        payload = dict(segment)
        words, min_word_start, max_word_end = _sanitize_words(list(payload.get("words", [])))
        payload["words"] = words

        start = float(payload.get("start", min_word_start or 0.0))
        end = float(payload.get("end", max_word_end if max_word_end is not None else start))
        if min_word_start is not None:
            start = min(start, min_word_start)
        if max_word_end is not None:
            end = max(end, max_word_end)
        if end < start:
            end = start

        payload["start"] = start
        payload["end"] = end
        sanitized_segments.append(payload)
    return sanitized_segments


class WhisperXBackend:
    """Current transcription backend implemented on top of WhisperX."""

    backend_id = "whisperx"
    _model_load_lock = Lock()
    _align_lock = Lock()

    def runtime_summary(self, backend_options: dict[str, Any] | None = None) -> dict[str, Any]:
        backend_options = backend_options or {}
        return {
            "id": self.backend_id,
            "asr": get_asr_runtime_config(backend_options),
            "alignment": get_alignment_runtime_config(backend_options),
        }

    def _build_asr_runtime(self, backend_options: dict[str, Any] | None = None) -> dict[str, Any]:
        backend_options = backend_options or {}
        runtime = get_asr_runtime_config(backend_options)
        runtime.update(resolve_asr_settings(runtime["device"], backend_options))
        vad_options = dict(backend_options.get("vad_options", {}))
        vad_options.setdefault("chunk_size", runtime["vad_chunk_size"])
        runtime["vad_options"] = vad_options
        runtime["detect_model"] = resolve_detect_model(backend_options)
        runtime["language_models"] = resolve_lang_model_map(backend_options)
        runtime["task"] = str(backend_options.get("task") or "transcribe")
        return runtime

    def _normalize_asr_options(self, asr_options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Accept faster-whisper style aliases while still targeting WhisperX's wrapper API."""

        resolved = dict(asr_options or {})
        if "temperature" in resolved and "temperatures" not in resolved:
            temperature = resolved.pop("temperature")
            resolved["temperatures"] = [float(temperature)]
        return resolved

    def _load_model(
        self,
        model_name: str,
        runtime: dict[str, Any],
        *,
        language: str | None = None,
        asr_options: dict[str, Any] | None = None,
    ):
        import whisperx

        if runtime.get("fallback_reason"):
            print(f"[ASR] WARNING: {runtime['fallback_reason']}")
        print(
            f"[ASR] Loading model={model_name} device={runtime['device']} "
            f"batch_size={runtime['batch_size']} vad_method={runtime['vad_method']} "
            f"vad_chunk_size={runtime['vad_chunk_size']} threads={runtime['threads']}"
        )
        base_options = {"max_new_tokens": 128, "repetition_penalty": 1.1}
        if asr_options:
            base_options.update(self._normalize_asr_options(asr_options))

        with self._model_load_lock:
            return whisperx.load_model(
                model_name,
                runtime["device"],
                compute_type=runtime["compute_type"],
                language=language,
                task=runtime["task"],
                vad_method=runtime["vad_method"],
                vad_options=runtime["vad_options"],
                threads=runtime["threads"],
                asr_options=base_options,
            )

    def transcribe(
        self,
        audio: Any,
        language: str | None = None,
        *,
        backend_options: dict[str, Any] | None = None,
        asr_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], str, dict[str, Any]]:
        runtime = self._build_asr_runtime(backend_options)
        transcribe_kwargs = {"batch_size": runtime["batch_size"]}
        language_models = runtime["language_models"]

        if language is not None:
            forced_language = str(language)
            model_name = language_models.get(forced_language, language_models[None])
            runtime["model_name"] = model_name
            runtime["requested_language"] = forced_language
            model = self._load_model(model_name, runtime, language=forced_language, asr_options=asr_options)
            result = model.transcribe(audio, language=forced_language, task=runtime["task"], **transcribe_kwargs)
            del model
            gc.collect()
            sanitized_segments = _sanitize_segments(result["segments"])
            runtime["reported_language"] = result.get("language", forced_language)
            if return_meta:
                return sanitized_segments, forced_language, runtime
            return sanitized_segments, forced_language

        detect_model = runtime["detect_model"]
        runtime["model_name"] = detect_model
        detect_runner = self._load_model(detect_model, runtime, language=None, asr_options=asr_options)
        detect_result = detect_runner.transcribe(audio, batch_size=min(runtime["batch_size"], 4), task=runtime["task"])
        detected = detect_result.get("language", "en")
        runtime["reported_language"] = detected
        model_name = language_models.get(detected, language_models[None])

        if model_name == detect_model:
            print(f"[ASR] Detected language={detected}; reusing detection pass result.")
            del detect_runner
            gc.collect()
            sanitized_segments = _sanitize_segments(detect_result["segments"])
            if return_meta:
                return sanitized_segments, detected, runtime
            return sanitized_segments, detected

        print(f"[ASR] Detected language={detected}; switching to model={model_name}")
        del detect_runner
        gc.collect()
        runtime["model_name"] = model_name
        model = self._load_model(model_name, runtime, language=detected, asr_options=asr_options)
        result = model.transcribe(audio, task=runtime["task"], **transcribe_kwargs)
        del model
        gc.collect()
        sanitized_segments = _sanitize_segments(result["segments"])
        if return_meta:
            return sanitized_segments, detected, runtime
        return sanitized_segments, detected

    def align(
        self,
        segments: list[dict[str, Any]],
        language: str,
        audio: Any,
        *,
        backend_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        import whisperx

        runtime = get_alignment_runtime_config(backend_options)
        if runtime.get("fallback_reason"):
            print(f"[ALIGN] WARNING: {runtime['fallback_reason']}")
        print(f"[ALIGN] Loading alignment model for language={language} device={runtime['device']} ...")
        with self._align_lock:
            try:
                align_model, metadata = whisperx.load_align_model(language_code=language, device=runtime["device"])
            except ValueError as exc:
                print(
                    f"[ALIGN] WARNING: No alignment model available for language={language!r} "
                    f"({exc}). Skipping alignment - word timestamps will be absent."
                )
                runtime["skip_reason"] = str(exc)
                if return_meta:
                    return segments, runtime
                return segments
            except Exception as exc:
                if runtime["device"] != "mps":
                    raise
                runtime["device"] = "cpu"
                runtime["fallback_reason"] = f"MPS alignment init failed ({exc}); falling back to CPU."
                print(f"[ALIGN] WARNING: {runtime['fallback_reason']}")
                align_model, metadata = whisperx.load_align_model(language_code=language, device=runtime["device"])

            try:
                result = whisperx.align(
                    segments,
                    align_model,
                    metadata,
                    audio,
                    runtime["device"],
                    return_char_alignments=False,
                )
            except Exception as exc:
                if runtime["device"] != "mps":
                    raise
                runtime["device"] = "cpu"
                runtime["fallback_reason"] = f"MPS alignment inference failed ({exc}); falling back to CPU."
                print(f"[ALIGN] WARNING: {runtime['fallback_reason']}")
                del align_model
                gc.collect()
                align_model, metadata = whisperx.load_align_model(language_code=language, device=runtime["device"])
                result = whisperx.align(
                    segments,
                    align_model,
                    metadata,
                    audio,
                    runtime["device"],
                    return_char_alignments=False,
                )
        try:
            sanitized_segments = _sanitize_segments(result["segments"])
            if return_meta:
                return sanitized_segments, runtime
            return sanitized_segments
        finally:
            del align_model
            gc.collect()
