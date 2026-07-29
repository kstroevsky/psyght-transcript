"""Backend contracts for pluggable phase1 transcription engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendSpec:
    """Backend selection embedded into a run or preset."""

    id: str = "gigaam_ctc"
    options: dict[str, Any] = field(default_factory=dict)


class TranscriptionBackend(Protocol):
    """Minimal backend interface needed by the phase1 runtime."""

    backend_id: str

    def transcribe(
        self,
        audio: Any,
        language: str | None = None,
        *,
        backend_options: dict[str, Any] | None = None,
        asr_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], str, dict[str, Any]]: ...

    def align(
        self,
        segments: list[dict[str, Any]],
        language: str,
        audio: Any,
        *,
        backend_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]: ...

    def runtime_summary(self, backend_options: dict[str, Any] | None = None) -> dict[str, Any]: ...
