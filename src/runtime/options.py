"""Normalized run options, filesystem paths, and mutable pipeline state for phase1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts.transcript import Transcript
from phase1.backends import BackendSpec
from phase1.config import (
    COMPUTE_TYPE,
    DEVICE,
    DIARIZE_DEVICE,
    HF_HUB_OFFLINE,
    PYANNOTE_DIARIZATION_MODEL,
    PYANNOTE_SEGMENTATION_MODEL,
    TRANSFORMERS_OFFLINE,
    requested_alignment_device,
    requested_asr_device,
    requested_diarization_device,
)
from phase1.pipeline.audio import AudioBundle


@dataclass(frozen=True)
class RunOptions:
    """Behavior-preserving wrapper around the public `run()` parameters."""

    audio_path: str
    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    content_mode: str = "multi-speaker"
    save_intermediate_dir: str | None = None
    progress_file: str | None = None
    parallel_post_asr: bool = True
    skip_alignment: bool = False
    skip_diarization: bool = False
    output_dir: str | None = None
    duration_limit: float | None = None
    asr_options: dict[str, Any] | None = None
    backend: BackendSpec = field(default_factory=BackendSpec)
    experiment_id: str | None = None
    preset_id: str | None = None

    @property
    def effective_skip_diarization(self) -> bool:
        """Single-speaker mode synthesizes diarization instead of running pyannote."""

        return self.skip_diarization or self.content_mode == "single-speaker"

    @property
    def effective_min_speakers(self) -> int | None:
        """Apply backend-specific diarization defaults without changing the public CLI."""

        if self.effective_skip_diarization:
            return self.min_speakers
        if self.backend.id == "therapy_hybrid" and self.content_mode == "multi-speaker":
            return 2 if self.min_speakers is None else self.min_speakers
        return self.min_speakers

    @property
    def effective_max_speakers(self) -> int | None:
        """Apply backend-specific diarization defaults without changing the public CLI."""

        if self.effective_skip_diarization:
            return self.max_speakers
        if self.backend.id == "therapy_hybrid" and self.content_mode == "multi-speaker":
            return 2 if self.max_speakers is None else self.max_speakers
        return self.max_speakers


@dataclass(frozen=True)
class RunPaths:
    """Resolved output locations derived from the public run options."""

    intermediate_dir: Path | None
    progress_path: Path | None
    run_meta_path: Path | None
    output_stem: str


@dataclass
class PipelineArtifacts:
    """In-memory artifacts tracked so failures can still persist partial work."""

    raw_segments: list[dict[str, Any]] | None = None
    aligned_segments: list[dict[str, Any]] | None = None
    diarized: Any = None
    asr_meta: dict[str, Any] | None = None
    alignment_meta: dict[str, Any] | None = None
    diarization_meta: dict[str, Any] | None = None
    detected_language: str | None = None
    duration: float | None = None


@dataclass
class RunExecutionResult:
    """Rich execution result used by compare mode while preserving the legacy `run()` return."""

    transcript: Transcript
    options: RunOptions
    paths: RunPaths
    run_meta: dict[str, Any]
    stage_timings_sec: dict[str, float]
    wall_clock_sec: float
    audio_bundle: AudioBundle


@dataclass
class ProgressState:
    """Current progress marker mirrored into the on-disk progress contract."""

    stage: str = "started"
    percent: int = 0


def build_run_paths(options: RunOptions) -> RunPaths:
    """Resolve artifact and output paths without changing the public CLI surface."""

    intermediate_dir = Path(options.save_intermediate_dir) if options.save_intermediate_dir else None
    progress_path = Path(options.progress_file) if options.progress_file else None
    if intermediate_dir:
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        if progress_path is None:
            progress_path = intermediate_dir / "progress.json"

    if options.output_dir:
        output_dir = Path(options.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = str(output_dir / Path(options.audio_path).stem)
    else:
        output_stem = str(Path(options.audio_path).with_suffix(""))

    run_meta_path = intermediate_dir / "00_run_meta.json" if intermediate_dir else None
    return RunPaths(
        intermediate_dir=intermediate_dir,
        progress_path=progress_path,
        run_meta_path=run_meta_path,
        output_stem=output_stem,
    )


def build_run_meta(options: RunOptions) -> dict[str, Any]:
    """Capture the stable run metadata contract written beside intermediate artifacts."""

    return {
        "experiment_id": options.experiment_id,
        "preset_id": options.preset_id,
        "backend": {
            "id": options.backend.id,
            "options": options.backend.options,
        },
        "audio_path": str(Path(options.audio_path).resolve()),
        "requested_language": options.language,
        "detected_language": None,
        "duration_sec": None,
        "requested_duration_limit_sec": options.duration_limit,
        "effective_duration_limit_sec": None,
        "content_mode": options.content_mode,
        "min_speakers": options.min_speakers,
        "max_speakers": options.max_speakers,
        "effective_min_speakers": options.effective_min_speakers,
        "effective_max_speakers": options.effective_max_speakers,
        "skip_alignment": options.skip_alignment,
        "skip_diarization": options.skip_diarization,
        "parallel_post_asr": options.parallel_post_asr,
        "runtime": {
            "device": DEVICE,
            "diarize_device": DIARIZE_DEVICE,
            "compute_type": COMPUTE_TYPE,
            "hf_hub_offline": HF_HUB_OFFLINE,
            "transformers_offline": TRANSFORMERS_OFFLINE,
            "pyannote_diarization_model": PYANNOTE_DIARIZATION_MODEL,
            "pyannote_segmentation_model": PYANNOTE_SEGMENTATION_MODEL,
            "requested_asr_device": requested_asr_device(),
            "requested_alignment_device": requested_alignment_device(),
            "requested_diarize_device": requested_diarization_device(),
            "asr_threads": None,
            "asr_batch_size": None,
            "asr_vad_method": None,
            "asr_vad_chunk_size": None,
            "asr_vad_options": None,
            "resolved_asr": None,
            "resolved_alignment": None,
            "resolved_diarization": None,
            "audio_preprocessing": {
                "requested": options.backend.options.get("audio_preprocessing") or [],
                "applied": [],
                "artifact_path": None,
            },
        },
        "stage_timings_sec": {},
        "wall_clock_sec": None,
    }
