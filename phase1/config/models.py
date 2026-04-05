"""Model identifiers and offline-mode toggles for the transcription pipeline."""

from __future__ import annotations

import os
from typing import Any

from phase1.config.env import env_str

HF_TOKEN = os.getenv("HF_TOKEN")
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0")
TRANSFORMERS_OFFLINE = os.getenv("TRANSFORMERS_OFFLINE", "0")
WHISPER_MODEL_OVERRIDE = os.getenv("WHISPERX_MODEL_NAME")

PYANNOTE_DIARIZATION_MODEL = env_str(
    "WHISPERX_PYANNOTE_DIARIZATION_MODEL",
    "pyannote/speaker-diarization-3.1",
)
PYANNOTE_SEGMENTATION_MODEL = env_str(
    "WHISPERX_PYANNOTE_SEGMENTATION_MODEL",
    "pyannote/segmentation-3.0",
)

SUPPORTED_LANGS = {"ru", "uk", "en"}
DEFAULT_DETECT_MODEL = "large-v3"


def resolve_detect_model(overrides: dict[str, Any] | None = None) -> str:
    """Resolve the multilingual detection model for the current run."""

    if overrides and overrides.get("detect_model"):
        return str(overrides["detect_model"])
    return DEFAULT_DETECT_MODEL


def resolve_whisper_model_override(overrides: dict[str, Any] | None = None) -> str | None:
    """Resolve a global Whisper model override for the current run."""

    if overrides and overrides.get("model_name"):
        return str(overrides["model_name"])
    return WHISPER_MODEL_OVERRIDE


def resolve_lang_model_map(overrides: dict[str, Any] | None = None) -> dict[str | None, str]:
    """Resolve language routing for a specific run without mutating module globals."""

    model_override = resolve_whisper_model_override(overrides)
    explicit_map = dict(overrides.get("language_models", {})) if overrides else {}
    default_model = explicit_map.get("default") or model_override or DEFAULT_DETECT_MODEL

    # Routing stays explicit because language selection controls both accuracy and offline
    # cache requirements. The values remain compatible with faster-whisper/CTranslate2 loading.
    return {
        "ru": explicit_map.get("ru") or model_override or "bzikst/faster-whisper-large-v3-russian",
        "uk": explicit_map.get("uk") or model_override or DEFAULT_DETECT_MODEL,
        "en": explicit_map.get("en") or model_override or DEFAULT_DETECT_MODEL,
        None: default_model,
    }


LANG_MODEL_MAP = resolve_lang_model_map()
