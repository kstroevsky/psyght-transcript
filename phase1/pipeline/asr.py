"""ASR stage facade delegated to the configured transcription backend."""

from __future__ import annotations

from typing import Any

from phase1.backends import get_backend


def transcribe(
    audio: Any,
    language: str | None = None,
    asr_options: dict[str, Any] | None = None,
    return_meta: bool = False,
    *,
    backend_id: str = "gigaam_ctc",
    backend_options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Transcribe audio through the selected backend."""

    return get_backend(backend_id).transcribe(
        audio,
        language,
        backend_options=backend_options,
        asr_options=asr_options,
        return_meta=return_meta,
    )
