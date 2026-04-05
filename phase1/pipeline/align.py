"""Alignment stage facade delegated to the configured transcription backend."""

from __future__ import annotations

from typing import Any

from phase1.backends import get_backend


def align(
    segments: list[dict[str, Any]],
    language: str,
    audio: Any,
    return_meta: bool = False,
    *,
    backend_id: str = "gigaam_ctc",
    backend_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Align segments to word timestamps through the selected backend."""

    return get_backend(backend_id).align(
        segments,
        language,
        audio,
        backend_options=backend_options,
        return_meta=return_meta,
    )
