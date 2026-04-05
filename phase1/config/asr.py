"""ASR decoding defaults tuned for the current self-hosted deployment targets."""

from __future__ import annotations

import os
from typing import Any

from phase1.config.env import env_int, env_str
from phase1.config.runtime import DEVICE, cpu_core_count


def _override_int(overrides: dict[str, Any] | None, key: str) -> int | None:
    value = None if overrides is None else overrides.get(key)
    return int(value) if value is not None else None


def _override_str(overrides: dict[str, Any] | None, key: str) -> str | None:
    value = None if overrides is None else overrides.get(key)
    return str(value) if value is not None else None


def resolve_asr_threads(device: str, overrides: dict[str, Any] | None = None) -> int:
    """Resolve ASR CPU thread count with preset override, env, then default precedence."""

    override = _override_int(overrides, "threads")
    if override is not None:
        return override

    raw = os.getenv("WHISPERX_ASR_THREADS")
    if raw:
        return int(raw)
    return 4 if device == "cuda" else cpu_core_count()


def resolve_asr_batch_size(device: str, overrides: dict[str, Any] | None = None) -> int:
    """Resolve ASR batch size with preset override precedence."""

    override = _override_int(overrides, "batch_size")
    if override is not None:
        return override
    return env_int("WHISPERX_ASR_BATCH_SIZE", 8 if device == "cuda" else 2)


def resolve_asr_vad_chunk_size(device: str, overrides: dict[str, Any] | None = None) -> int:
    """Resolve ASR VAD chunk size with preset override precedence."""

    override = _override_int(overrides, "vad_chunk_size")
    if override is not None:
        return override
    return env_int("WHISPERX_ASR_VAD_CHUNK_SIZE", 30 if device == "cuda" else 10)


def resolve_asr_vad_method(device: str, overrides: dict[str, Any] | None = None) -> str:
    """Resolve ASR VAD method with preset override precedence."""

    override = _override_str(overrides, "vad_method")
    if override is not None:
        return override
    return env_str("WHISPERX_ASR_VAD_METHOD", "pyannote" if device == "cuda" else "silero")


def resolve_asr_settings(device: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the resolved ASR decoding settings for a specific run."""

    return {
        "batch_size": resolve_asr_batch_size(device, overrides),
        "threads": resolve_asr_threads(device, overrides),
        "vad_method": resolve_asr_vad_method(device, overrides),
        "vad_chunk_size": resolve_asr_vad_chunk_size(device, overrides),
    }


_DEFAULT_SETTINGS = resolve_asr_settings(DEVICE)

ASR_BATCH_SIZE = _DEFAULT_SETTINGS["batch_size"]
ASR_THREADS = _DEFAULT_SETTINGS["threads"]
ASR_VAD_CHUNK_SIZE = _DEFAULT_SETTINGS["vad_chunk_size"]
ASR_VAD_METHOD = _DEFAULT_SETTINGS["vad_method"]
