"""Additive Russian therapy-transcription backend built from modular stage objects."""

from __future__ import annotations

from typing import Any

from phase1.backends.whisperx_backend import WhisperXBackend
from phase1.therapy.pipeline import TherapyHybridTranscriber, resolve_therapy_backend_options


class TherapyHybridBackend:
    """WhisperX VAD + CTC anchor + hallucination guard + LLM merge."""

    backend_id = "therapy_hybrid"

    def __init__(self) -> None:
        self._aligner = WhisperXBackend()

    def runtime_summary(self, backend_options: dict[str, Any] | None = None) -> dict[str, Any]:
        resolved = resolve_therapy_backend_options(backend_options)
        return {
            "id": self.backend_id,
            "therapy": resolved,
            "alignment": self._aligner.runtime_summary(resolved["whisper"]).get("alignment"),
        }

    def transcribe(
        self,
        audio: Any,
        language: str | None = None,
        *,
        backend_options: dict[str, Any] | None = None,
        asr_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], str, dict[str, Any]]:
        del asr_options
        effective_language = str(language or "ru")
        transcriber = TherapyHybridTranscriber(backend_options)
        segments, meta = transcriber.run(audio, language=effective_language)
        if return_meta:
            return segments, effective_language, meta
        return segments, effective_language

    def align(
        self,
        segments: list[dict[str, Any]],
        language: str,
        audio: Any,
        *,
        backend_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        resolved = resolve_therapy_backend_options(backend_options)
        aligned_segments, align_meta = self._aligner.align(
            segments,
            language,
            audio,
            backend_options=resolved["whisper"],
            return_meta=True,
        )
        for input_segment, aligned_segment in zip(segments, aligned_segments):
            for key in ("confidence", "source"):
                if input_segment.get(key) is not None:
                    aligned_segment[key] = input_segment[key]
        if return_meta:
            return aligned_segments, align_meta
        return aligned_segments
