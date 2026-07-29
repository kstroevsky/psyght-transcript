"""Preset loading and validation for phase1 compare mode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phase1.backends import BackendSpec, get_backend, list_backend_ids
from phase1.runtime.options import RunOptions

_PIPELINE_KEYS = {
    "language",
    "min_speakers",
    "max_speakers",
    "content_mode",
    "duration_limit",
    "parallel_post_asr",
    "skip_alignment",
    "skip_diarization",
}


@dataclass(frozen=True)
class ComparePipelineSpec:
    """Typed pipeline policy for one compare preset."""

    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    content_mode: str = "multi-speaker"
    duration_limit: float | None = None
    parallel_post_asr: bool = True
    skip_alignment: bool = False
    skip_diarization: bool = False

    @property
    def runs_alignment(self) -> bool:
        """Whether the preset should execute the alignment stage."""

        return not self.skip_alignment

    @property
    def runs_diarization(self) -> bool:
        """Whether the preset should execute diarization instead of synthetic speaker output."""

        return not self.skip_diarization and self.content_mode != "single-speaker"

    def manifest(self) -> dict[str, Any]:
        """Serialize the normalized pipeline policy for experiment artifacts."""

        return {
            "language": self.language,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "content_mode": self.content_mode,
            "duration_limit": self.duration_limit,
            "parallel_post_asr": self.parallel_post_asr,
            "skip_alignment": self.skip_alignment,
            "skip_diarization": self.skip_diarization,
        }


@dataclass(frozen=True)
class ComparePreset:
    """Validated compare preset with enough information to build a run."""

    path: Path
    preset_id: str
    description: str | None
    pipeline: ComparePipelineSpec
    backend: BackendSpec

    def to_run_options(
        self,
        *,
        audio_path: str,
        run_dir: Path,
        experiment_id: str,
    ) -> RunOptions:
        """Convert the preset into the standard phase1 run options."""

        return RunOptions(
            audio_path=audio_path,
            language=self.pipeline.language,
            min_speakers=self.pipeline.min_speakers,
            max_speakers=self.pipeline.max_speakers,
            content_mode=self.pipeline.content_mode,
            save_intermediate_dir=str(run_dir),
            progress_file=None,
            parallel_post_asr=self.pipeline.parallel_post_asr,
            skip_alignment=self.pipeline.skip_alignment,
            skip_diarization=self.pipeline.skip_diarization,
            output_dir=str(run_dir),
            duration_limit=self.pipeline.duration_limit,
            asr_options=dict(self.backend.options.get("asr_options", {})) or None,
            backend=self.backend,
            experiment_id=experiment_id,
            preset_id=self.preset_id,
        )

    def manifest(self) -> dict[str, Any]:
        """Serialize the preset for experiment-level metadata."""

        return {
            "id": self.preset_id,
            "description": self.description,
            "path": str(self.path),
            "pipeline": self.pipeline.manifest(),
            "backend": {
                "id": self.backend.id,
                "options": self.backend.options,
            },
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid preset JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Preset must be a JSON object: {path}")
    return payload


def _validate_pipeline(payload: dict[str, Any], path: Path) -> ComparePipelineSpec:
    unknown = sorted(set(payload) - _PIPELINE_KEYS)
    if unknown:
        raise ValueError(f"Unknown pipeline preset keys in {path}: {', '.join(unknown)}")
    content_mode = str(payload.get("content_mode") or "multi-speaker")
    if content_mode not in {"single-speaker", "multi-speaker"}:
        raise ValueError(f"Unsupported preset pipeline content_mode in {path}: {content_mode}")
    duration_limit = payload.get("duration_limit")
    return ComparePipelineSpec(
        language=payload.get("language"),
        min_speakers=payload.get("min_speakers"),
        max_speakers=payload.get("max_speakers"),
        content_mode=content_mode,
        duration_limit=float(duration_limit) if duration_limit is not None else None,
        parallel_post_asr=bool(payload.get("parallel_post_asr", True)),
        skip_alignment=bool(payload.get("skip_alignment", False)),
        skip_diarization=bool(payload.get("skip_diarization", False)),
    )


def _validate_backend(payload: dict[str, Any] | None, path: Path) -> BackendSpec:
    payload = payload or {"id": "gigaam_ctc", "options": {"revision": "e2e_ctc"}}
    if not isinstance(payload, dict):
        raise ValueError(f"Preset backend must be an object: {path}")
    backend_id = payload.get("id")
    if not backend_id:
        raise ValueError(f"Preset backend.id is required in {path}")
    get_backend(str(backend_id))
    options = payload.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError(f"Preset backend.options must be an object: {path}")
    return BackendSpec(id=str(backend_id), options=dict(options))


def build_compare_preset(payload: dict[str, Any], *, source: str | Path) -> ComparePreset:
    """Validate an in-memory preset payload into a :class:`ComparePreset`.

    Shared by file loading (``load_preset``) and the experiment-manifest layer so
    both go through identical pipeline/backend validation.
    """

    source_path = Path(source)
    preset_id = payload.get("id")
    if not preset_id or not isinstance(preset_id, str):
        raise ValueError(f"Preset id is required in {source_path}")
    pipeline = _validate_pipeline(payload.get("pipeline") or {}, source_path)
    return ComparePreset(
        path=source_path,
        preset_id=preset_id,
        description=payload.get("description"),
        pipeline=pipeline,
        backend=_validate_backend(payload.get("backend"), source_path),
    )


def load_preset(path: str | Path) -> ComparePreset:
    """Load and validate one compare preset."""

    preset_path = Path(path)
    return build_compare_preset(_load_json(preset_path), source=preset_path)


def discover_preset_paths(preset_paths: list[str] | None = None, preset_dirs: list[str] | None = None) -> list[Path]:
    """Resolve explicit preset files and JSON files from directories."""

    resolved: list[Path] = []
    for raw_path in preset_paths or []:
        resolved.append(Path(raw_path))
    for raw_dir in preset_dirs or []:
        directory = Path(raw_dir)
        if not directory.exists():
            raise ValueError(f"Preset directory does not exist: {directory}")
        resolved.extend(sorted(path for path in directory.iterdir() if path.suffix.lower() == ".json"))
    if not resolved:
        raise ValueError("At least one --preset or --preset-dir is required.")
    return resolved


def load_presets(preset_paths: list[str] | None = None, preset_dirs: list[str] | None = None) -> list[ComparePreset]:
    """Load compare presets and reject duplicate identifiers."""

    presets = [load_preset(path) for path in discover_preset_paths(preset_paths, preset_dirs)]
    seen: dict[str, Path] = {}
    for preset in presets:
        previous = seen.get(preset.preset_id)
        if previous is not None:
            raise ValueError(f"Duplicate preset id {preset.preset_id!r}: {previous} and {preset.path}")
        seen[preset.preset_id] = preset.path
    return presets


def preset_help() -> str:
    """Short backend help text for docs and CLI errors."""

    return ", ".join(list_backend_ids())
