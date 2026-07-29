"""Registry of available transcription backends."""

from __future__ import annotations

from phase1.backends.base import TranscriptionBackend
from phase1.backends.canary_backend import CanaryBackend
from phase1.backends.gigaam_ctc_backend import GigaAMCTCBackend
from phase1.backends.therapy_hybrid_backend import TherapyHybridBackend
from phase1.backends.whisperx_backend import WhisperXBackend

_BACKENDS: dict[str, TranscriptionBackend] = {
    CanaryBackend.backend_id: CanaryBackend(),
    GigaAMCTCBackend.backend_id: GigaAMCTCBackend(),
    TherapyHybridBackend.backend_id: TherapyHybridBackend(),
    WhisperXBackend.backend_id: WhisperXBackend(),
}


def get_backend(backend_id: str) -> TranscriptionBackend:
    """Return a registered backend or raise a clear error."""

    try:
        return _BACKENDS[backend_id]
    except KeyError as exc:
        raise ValueError(f"Unknown transcription backend: {backend_id}") from exc


def list_backend_ids() -> list[str]:
    """Expose backend identifiers for CLI validation and docs."""

    return sorted(_BACKENDS)
