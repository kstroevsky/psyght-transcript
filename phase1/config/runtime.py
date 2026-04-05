"""Runtime device selection helpers for WhisperX and pyannote in this deployment model."""

from __future__ import annotations

import os
import platform
from typing import Any


def _torch_module() -> Any | None:
    """Keep runtime config importable even when torch is absent in the active interpreter."""

    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only outside the phase1 venv
        return None
    return torch


def _cuda_is_available() -> bool:
    """Return whether CUDA is available in the active torch install."""

    torch = _torch_module()
    return bool(torch and torch.cuda.is_available())


def _mps_is_available() -> bool:
    """Return whether Apple MPS is available in the active torch install."""

    torch = _torch_module()
    if torch is None:
        return False
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None:
        return False
    return bool(mps.is_built() and mps.is_available())


def is_apple_silicon() -> bool:
    """Return whether the current runtime is macOS on Apple Silicon."""

    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def cpu_core_count() -> int:
    """Return the local CPU core count used to size CPU-bound inference."""

    return max(1, os.cpu_count() or 1)


def _requested_device(override: str | None, env_key: str) -> str:
    if override:
        return override
    env_override = os.getenv(env_key)
    if env_override:
        return env_override
    if _cuda_is_available():
        return "cuda"
    return "cpu"


def requested_asr_device(override: str | None = None) -> str:
    """Return the requested ASR device before any safety fallback."""

    return _requested_device(override, "WHISPERX_DEVICE")


def requested_alignment_device(override: str | None = None) -> str:
    """Return the requested alignment device before any safety fallback."""

    if override:
        return override
    env_override = os.getenv("WHISPERX_ALIGNMENT_DEVICE")
    if env_override:
        return env_override
    if _cuda_is_available():
        return "cuda"
    if _mps_is_available():
        return "mps"
    return "cpu"


def requested_diarization_device(override: str | None = None) -> str:
    """Return the requested diarization device before any safety fallback."""

    if override:
        return override
    env_override = os.getenv("WHISPERX_DIARIZE_DEVICE")
    if env_override:
        return env_override
    if _cuda_is_available():
        return "cuda"
    if _mps_is_available():
        return "mps"
    return "cpu"


def _compute_type_for_device(device: str, override: str | None = None) -> str:
    """Choose the default compute type for a resolved device."""

    compute_override = override or os.getenv("WHISPERX_COMPUTE_TYPE")
    if compute_override:
        return compute_override
    if device == "cuda":
        return "float16"
    if device == "mps":
        return "float32"
    return "int8"


def get_asr_runtime_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the safe ASR runtime for faster-whisper on the current machine."""

    overrides = overrides or {}
    requested = requested_asr_device(overrides.get("device"))
    resolved = requested
    fallback_reason = None

    if requested == "cuda" and not _cuda_is_available():
        resolved = "cpu"
        fallback_reason = "CUDA requested for ASR but not available; falling back to CPU."
    elif requested == "mps":
        if not _mps_is_available():
            resolved = "cpu"
            fallback_reason = "MPS requested for ASR but not available; falling back to CPU."
        else:
            resolved = "cpu"
            fallback_reason = (
                "MPS requested for ASR but faster-whisper/CTranslate2 prebuilt binaries only support "
                "cpu/cuda; falling back to CPU."
            )

    return {
        "requested_device": requested,
        "device": resolved,
        "compute_type": _compute_type_for_device(resolved, overrides.get("compute_type")),
        "fallback_reason": fallback_reason,
        "apple_silicon": is_apple_silicon(),
        "cuda_available": _cuda_is_available(),
        "mps_available": _mps_is_available(),
    }


def get_alignment_runtime_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the preferred alignment runtime before model loading."""

    overrides = overrides or {}
    requested = requested_alignment_device(overrides.get("alignment_device") or overrides.get("device"))
    resolved = requested
    fallback_reason = None

    if requested == "cuda" and not _cuda_is_available():
        resolved = "cpu"
        fallback_reason = "CUDA requested for alignment but not available; falling back to CPU."
    elif requested == "mps" and not _mps_is_available():
        resolved = "cpu"
        fallback_reason = "MPS requested for alignment but not available; falling back to CPU."

    return {
        "requested_device": requested,
        "device": resolved,
        "fallback_reason": fallback_reason,
        "apple_silicon": is_apple_silicon(),
        "cuda_available": _cuda_is_available(),
        "mps_available": _mps_is_available(),
    }


def get_diarization_runtime_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the preferred diarization runtime before pipeline preflight."""

    overrides = overrides or {}
    requested = requested_diarization_device(overrides.get("diarization_device") or overrides.get("device"))
    resolved = requested
    fallback_reason = None

    if requested == "cuda" and not _cuda_is_available():
        resolved = "cpu"
        fallback_reason = "CUDA requested for diarization but not available; falling back to CPU."
    elif requested == "mps" and not _mps_is_available():
        resolved = "cpu"
        fallback_reason = "MPS requested for diarization but not available; falling back to CPU."

    return {
        "requested_device": requested,
        "device": resolved,
        "compute_type": _compute_type_for_device(
            resolved,
            overrides.get("diarization_compute_type") or overrides.get("compute_type"),
        ),
        "fallback_reason": fallback_reason,
        "apple_silicon": is_apple_silicon(),
        "cuda_available": _cuda_is_available(),
        "mps_available": _mps_is_available(),
    }


_asr_runtime = get_asr_runtime_config()
_diarization_runtime = get_diarization_runtime_config()

DEVICE = _asr_runtime["device"]
COMPUTE_TYPE = _asr_runtime["compute_type"]
DIARIZE_DEVICE = _diarization_runtime["device"]
DIARIZE_COMPUTE_TYPE = _diarization_runtime["compute_type"]
