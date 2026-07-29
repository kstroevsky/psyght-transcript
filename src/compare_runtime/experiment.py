"""Experiment orchestration for phase1 compare mode."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from phase1.backends import get_backend
from phase1.compare_runtime.presets import ComparePreset
from phase1.config.env import BASE_DIR
from phase1.config.runtime import get_diarization_runtime_config
from phase1.pipeline.audio import AudioBundle, load_audio_bundle, read_audio_cache, write_audio_cache
from phase1.quality import build_run_quality_report, rank_run_summaries
from phase1.runtime.artifacts import dump_json
from phase1.runtime.runner import execute


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cpu_only_for_preset(preset: ComparePreset) -> bool:
    backend_summary = get_backend(preset.backend.id).runtime_summary(preset.backend.options)
    devices = {str((backend_summary.get("asr") or {}).get("device", "cpu"))}
    if preset.pipeline.runs_alignment:
        devices.add(str((backend_summary.get("alignment") or {}).get("device", "cpu")))
    if preset.pipeline.runs_diarization:
        devices.add(str(get_diarization_runtime_config(preset.backend.options).get("device", "cpu")))
    return devices <= {"cpu"}


def resolve_effective_worker_count(presets: list[ComparePreset], requested_max_workers: int) -> tuple[int, str | None]:
    """Clamp compare-mode concurrency to CPU-only runs."""

    requested = max(1, requested_max_workers)
    if requested == 1:
        return 1, None
    if all(_cpu_only_for_preset(preset) for preset in presets):
        return requested, None
    return 1, "Non-CPU preset detected; compare mode was serialized for stability."


def _experiment_root(audio_path: str, output_dir: str | None) -> tuple[str, Path]:
    base_dir = Path(output_dir) if output_dir else BASE_DIR / "runs"
    experiment_id = f"compare_{Path(audio_path).stem}_{_timestamp()}"
    root = base_dir / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    return experiment_id, root


def _shared_audio_bundle(shared_audio_path: Path, duration: float) -> AudioBundle:
    return AudioBundle(audio=np.array(read_audio_cache(shared_audio_path), dtype=np.float32, copy=True), duration=duration)


def _write_saved_presets(root: Path, presets: list[ComparePreset]) -> Path:
    presets_dir = root / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    for preset in presets:
        dump_json(
            presets_dir / f"{preset.preset_id}.json",
            {
                "id": preset.preset_id,
                "description": preset.description,
                "pipeline": preset.pipeline.manifest(),
                "backend": {
                    "id": preset.backend.id,
                    "options": preset.backend.options,
                },
                "source_path": str(preset.path),
            },
        )
    return presets_dir


def _run_summary(
    preset: ComparePreset,
    run_dir: Path,
    audio_path: str,
    reference_text: str | None,
    transcript,
    *,
    error: BaseException | None = None,
) -> dict[str, Any]:
    run_meta = _read_json(run_dir / "00_run_meta.json")
    progress = _read_json(run_dir / "progress.json")
    error_payload = _read_json(run_dir / "99_error.json")
    quality_report = build_run_quality_report(
        transcript=transcript,
        run_meta=run_meta,
        reference_text=reference_text,
        error=error,
    )
    quality_report["stability"]["failure_stage"] = error_payload.get("stage") or progress.get("stage")
    if error_payload.get("error"):
        quality_report["stability"]["error"] = error_payload.get("error")

    stem = Path(audio_path).stem
    return {
        "preset_id": preset.preset_id,
        "description": preset.description,
        "status": "completed" if error is None else "failed",
        "artifacts_dir": str(run_dir),
        "output_audio": str(run_dir / "00_processed_audio.wav") if (run_dir / "00_processed_audio.wav").exists() else None,
        "output_txt": str(run_dir / f"{stem}.txt"),
        "output_json": str(run_dir / f"{stem}.json"),
        "performance": quality_report["performance"],
        "stability": quality_report["stability"],
        "metrics": quality_report["metrics"],
        "backend": {
            "id": preset.backend.id,
            "options": preset.backend.options,
        },
        "pipeline": preset.pipeline.manifest(),
        "run_meta": run_meta,
    }


def run_experiment(
    *,
    audio_path: str,
    presets: list[ComparePreset],
    output_dir: str | None,
    reference_text: str | None,
    max_workers: int,
) -> dict[str, Any]:
    """Run one audio file against multiple presets and persist experiment reports."""

    experiment_id, root = _experiment_root(audio_path, output_dir)
    effective_workers, worker_reason = resolve_effective_worker_count(presets, max_workers)
    saved_presets_dir = _write_saved_presets(root, presets)
    experiment_payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "audio_path": str(Path(audio_path).resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_max_workers": max(1, max_workers),
        "effective_max_workers": effective_workers,
        "worker_clamp_reason": worker_reason,
        "reference_enabled": reference_text is not None,
        "presets": [preset.manifest() for preset in presets],
        "saved_presets_dir": str(saved_presets_dir),
        "shared_audio_cache": None,
    }

    shared_audio_path: Path | None = None
    shared_audio_duration: float | None = None
    if effective_workers > 1:
        shared_audio = load_audio_bundle(audio_path)
        shared_audio_path = root / "shared_audio.npy"
        write_audio_cache(shared_audio_path, shared_audio.audio)
        shared_audio_duration = shared_audio.duration
        experiment_payload["shared_audio_cache"] = str(shared_audio_path)
        experiment_payload["shared_audio_duration_sec"] = shared_audio_duration

    dump_json(root / "experiment.json", experiment_payload)

    def execute_preset(preset: ComparePreset) -> dict[str, Any]:
        run_dir = root / preset.preset_id
        run_options = preset.to_run_options(audio_path=audio_path, run_dir=run_dir, experiment_id=experiment_id)
        audio_bundle = None
        if shared_audio_path is not None and shared_audio_duration is not None:
            audio_bundle = _shared_audio_bundle(shared_audio_path, shared_audio_duration)
        try:
            result = execute(run_options, audio_bundle=audio_bundle)
            return _run_summary(
                preset,
                run_dir,
                audio_path,
                reference_text,
                result.transcript,
                error=None,
            )
        except Exception as exc:
            return _run_summary(
                preset,
                run_dir,
                audio_path,
                reference_text,
                transcript=None,
                error=exc,
            )

    if effective_workers == 1:
        runs = [execute_preset(preset) for preset in presets]
    else:
        futures = {}
        runs = []
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            for preset in presets:
                futures[pool.submit(execute_preset, preset)] = preset.preset_id
            for future in as_completed(futures):
                runs.append(future.result())

    ranked_runs = rank_run_summaries(runs, has_reference=reference_text is not None)
    summary = {
        "experiment_id": experiment_id,
        "audio_path": str(Path(audio_path).resolve()),
        "reference_enabled": reference_text is not None,
        "requested_max_workers": experiment_payload["requested_max_workers"],
        "effective_max_workers": effective_workers,
        "worker_clamp_reason": worker_reason,
        "saved_presets_dir": str(saved_presets_dir),
        "best_preset_id": ranked_runs[0]["preset_id"] if ranked_runs else None,
        "runs": ranked_runs,
    }
    dump_json(root / "summary.json", summary)

    experiment_payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(root / "experiment.json", experiment_payload)
    return summary
