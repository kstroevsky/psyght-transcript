"""Artifact persistence helpers for phase1 progress, debugging, and failure recovery."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts.transcript import Transcript, transcript_to_dict
from phase1.runtime.options import PipelineArtifacts, RunPaths


def jsonable(value: Any) -> Any:
    """Convert runtime objects into the JSON-safe structures stored as artifacts."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]

    start = getattr(value, "start", None)
    end = getattr(value, "end", None)
    if start is not None and end is not None:
        try:
            return {"start": float(start), "end": float(end), "repr": str(value)}
        except Exception:  # pragma: no cover - defensive fallback
            return {"repr": str(value)}

    if hasattr(value, "__dict__"):
        try:
            return {key: jsonable(item) for key, item in vars(value).items()}
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)

    return str(value)


def dump_json(path: Path, payload: Any) -> None:
    """Write JSON artifacts using the same formatting as the previous single-file runner."""

    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def utc_now() -> str:
    """Return an ISO8601 UTC timestamp for progress and error artifacts."""

    return datetime.now(timezone.utc).isoformat()


def write_run_meta(path: Path | None, payload: dict[str, Any]) -> None:
    """Persist the evolving run metadata contract when intermediate artifacts are enabled."""

    if path is not None:
        dump_json(path, payload)


def write_intermediate_artifact(paths: RunPaths, name: str, payload: Any) -> None:
    """Persist a named intermediate artifact only when an artifact directory is configured."""

    if paths.intermediate_dir is not None:
        dump_json(paths.intermediate_dir / name, payload)


def write_structured_transcript(paths: RunPaths, transcript: Transcript) -> None:
    """Persist the final structured transcript artifact using the shared contract serializer."""

    write_intermediate_artifact(paths, "05_transcript_structured.json", transcript_to_dict(transcript))


def persist_error_artifact(intermediate_dir: Path | None, stage: str, exc: BaseException) -> None:
    """Write the stable error artifact expected by the existing debugging workflow."""

    if intermediate_dir is None:
        return
    dump_json(
        intermediate_dir / "99_error.json",
        {
            "timestamp_utc": utc_now(),
            "stage": stage,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def persist_partial_artifacts(paths: RunPaths, state: PipelineArtifacts) -> None:
    """Preserve in-memory artifacts on failure so interrupted runs stay inspectable."""

    if paths.intermediate_dir is None:
        return

    from phase1.pipeline.diarize import diarization_to_records

    raw_path = paths.intermediate_dir / "01_asr_raw_segments.json"
    if state.raw_segments is not None and not raw_path.exists():
        dump_json(raw_path, state.raw_segments)

    aligned_path = paths.intermediate_dir / "02_aligned_segments.json"
    if state.aligned_segments is not None and not aligned_path.exists():
        dump_json(aligned_path, state.aligned_segments)

    diarization_path = paths.intermediate_dir / "03_diarization_segments.json"
    if state.diarized is not None and not diarization_path.exists():
        dump_json(diarization_path, diarization_to_records(state.diarized))

    meta_path = paths.intermediate_dir / "03_diarization_meta.json"
    if state.diarization_meta is not None and not meta_path.exists():
        dump_json(meta_path, state.diarization_meta)
