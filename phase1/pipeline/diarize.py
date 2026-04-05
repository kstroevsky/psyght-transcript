"""Diarization stage wrapper that caches and reuses the pyannote pipeline."""

from __future__ import annotations

import threading
from typing import Any

from phase1.config import (
    HF_TOKEN,
    PYANNOTE_DIARIZATION_MODEL,
    PYANNOTE_SEGMENTATION_MODEL,
    get_diarization_runtime_config as resolve_diarization_runtime_config,
)

_PIPELINE_CACHE: dict[tuple[str, str, str], tuple[Any, dict[str, Any]]] = {}
_PIPELINE_LOCK = threading.Lock()


def _json_safe_diarization_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    segment = payload.pop("segment", None)
    if segment is not None:
        if "start" not in payload and hasattr(segment, "start"):
            payload["start"] = float(segment.start)
        if "end" not in payload and hasattr(segment, "end"):
            payload["end"] = float(segment.end)
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = value
        elif hasattr(value, "start") and hasattr(value, "end"):
            normalized[str(key)] = {"start": float(value.start), "end": float(value.end)}
        else:
            normalized[str(key)] = str(value)
    return normalized


def _resolved_diarization_runtime() -> dict[str, Any]:
    """Expose the resolved diarization runtime config for run metadata and diagnostics."""

    runtime = resolve_diarization_runtime_config()
    runtime.update(
        {
            "diarization_model": PYANNOTE_DIARIZATION_MODEL,
            "segmentation_model": PYANNOTE_SEGMENTATION_MODEL,
        }
    )
    return runtime


def get_diarization_runtime_config() -> dict[str, Any]:
    """Expose the initial diarization runtime config for run metadata and diagnostics."""

    return _resolved_diarization_runtime()


def diarization_to_records(diarized: Any) -> list[dict[str, Any]]:
    """Normalize pyannote output into the artifact format written by phase1."""

    if isinstance(diarized, list) and all(isinstance(item, dict) for item in diarized):
        return [_json_safe_diarization_record(item) for item in diarized]
    if hasattr(diarized, "to_dict"):
        return [_json_safe_diarization_record(item) for item in diarized.to_dict(orient="records")]
    return [{"value": str(diarized)}]


def _pipeline_cache_key(runtime: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(runtime["diarization_model"]),
        str(runtime["segmentation_model"]),
        str(runtime["device"]),
    )


def _load_diarization_pipeline(overrides: dict[str, Any] | None = None):
    runtime = _resolved_diarization_runtime()
    if overrides:
        runtime.update(resolve_diarization_runtime_config(overrides))
        runtime["diarization_model"] = str(overrides.get("diarization_model", runtime["diarization_model"]))
        runtime["segmentation_model"] = str(overrides.get("segmentation_model", runtime["segmentation_model"]))

    cache_key = _pipeline_cache_key(runtime)
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _PIPELINE_LOCK:
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN is required for pyannote diarization models.")

        if runtime.get("fallback_reason"):
            print(f"[DIARIZE] WARNING: {runtime['fallback_reason']}")
        print(
            "[DIARIZE] "
            f"device={runtime['device']} "
            f"compute_type={runtime['compute_type']} "
            f"diarization_model={runtime['diarization_model']} "
            f"segmentation_model={runtime['segmentation_model']}"
        )

        from huggingface_hub import snapshot_download
        from whisperx.diarize import DiarizationPipeline

        # Pre-download both gated repos so failures happen before ASR runs for a long time.
        snapshot_download(repo_id=runtime["diarization_model"], token=HF_TOKEN)
        snapshot_download(repo_id=runtime["segmentation_model"], token=HF_TOKEN)
        try:
            pipeline = DiarizationPipeline(
                model_name=runtime["diarization_model"],
                token=HF_TOKEN,
                device=runtime["device"],
            )
        except Exception as exc:
            if runtime["device"] != "mps":
                raise RuntimeError(
                    "Diarization pipeline init failed with "
                    f"diarization_model={runtime['diarization_model']} "
                    f"segmentation_model={runtime['segmentation_model']} "
                    f"device={runtime['device']}. Original error: {exc}"
                ) from exc
            runtime["device"] = "cpu"
            runtime["compute_type"] = "int8"
            runtime["fallback_reason"] = f"MPS diarization init failed ({exc}); falling back to CPU."
            print(f"[DIARIZE] WARNING: {runtime['fallback_reason']}")
            pipeline = DiarizationPipeline(
                model_name=runtime["diarization_model"],
                token=HF_TOKEN,
                device=runtime["device"],
            )

        cached = (pipeline, dict(runtime))
        _PIPELINE_CACHE[_pipeline_cache_key(runtime)] = cached
        return cached


def preflight_diarization(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Eagerly initialize diarization so auth/model access failures happen first."""

    _, runtime = _load_diarization_pipeline(overrides)
    return runtime


def diarize(
    audio_input: Any,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    return_meta: bool = False,
    overrides: dict[str, Any] | None = None,
):
    """Run speaker diarization with the current runtime config and speaker-count hints."""

    pipeline, runtime = _load_diarization_pipeline(overrides)
    kwargs: dict[str, int] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    try:
        diarized = pipeline(audio_input, **kwargs)
    except Exception as exc:
        if runtime["device"] == "mps":
            runtime["device"] = "cpu"
            runtime["compute_type"] = "int8"
            runtime["fallback_reason"] = f"MPS diarization inference failed ({exc}); falling back to CPU."
            print(f"[DIARIZE] WARNING: {runtime['fallback_reason']}")
            from whisperx.diarize import DiarizationPipeline

            pipeline = DiarizationPipeline(
                model_name=runtime["diarization_model"],
                token=HF_TOKEN,
                device=runtime["device"],
            )
            _PIPELINE_CACHE[_pipeline_cache_key(runtime)] = (pipeline, dict(runtime))
            diarized = pipeline(audio_input, **kwargs)
        else:
            raise RuntimeError(
                "Diarization failed with "
                f"diarization_model={runtime['diarization_model']} "
                f"segmentation_model={runtime['segmentation_model']} "
                f"device={runtime['device']}. Original error: {exc}"
            ) from exc

    if return_meta:
        return diarized, runtime
    return diarized
