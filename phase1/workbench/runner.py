"""Run an experiment manifest and build its leaderboard.

Glue layer: for one audio file or each corpus item, run the variants through the
existing ``compare_runtime.run_experiment`` (no new run engine), then score each
produced transcript against the structured reference with the eval layer
(WER/cpWER/DER), and aggregate into a ranked leaderboard.

The heavy ``run_experiment`` call is injectable so the orchestration is testable
without running any model.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from contracts.transcript import transcript_from_dict
from phase1.compare_runtime.experiment import run_experiment as _run_experiment
from phase1.config.env import BASE_DIR
from phase1.eval.corpus import load_corpus
from phase1.eval.reference import flatten_text, load_reference
from phase1.eval.score import score_transcript
from phase1.runtime.artifacts import dump_json
from phase1.workbench.leaderboard import build_leaderboard, render_leaderboard
from phase1.workbench.manifest import ExperimentManifest

RunExperiment = Callable[..., dict[str, Any]]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_items(manifest: ExperimentManifest) -> list[tuple[str, Path, Path | None, float | None]]:
    if manifest.corpus is not None:
        corpus = load_corpus(manifest.corpus)
        items: list[tuple[str, Path, Path | None, float | None]] = []
        for item in corpus.items:
            if item.audio is None:
                raise ValueError(f"Corpus item {item.id!r} has no 'audio'; cannot run the pipeline for it.")
            items.append((item.id, item.audio, item.reference, item.duration_limit))
        return items
    if manifest.audio is None:
        raise ValueError("Manifest has neither 'audio' nor 'corpus'.")
    return [(manifest.id, manifest.audio, manifest.reference, None)]


def _apply_duration_limit(presets, duration_limit: float | None):
    if duration_limit is None:
        return presets
    return [replace(preset, pipeline=replace(preset.pipeline, duration_limit=duration_limit)) for preset in presets]


def _score_row(run: dict[str, Any], reference, profile: str | None) -> dict[str, Any]:
    performance = run.get("performance", {})
    metrics = run.get("metrics", {})
    stability = run.get("stability", {})
    row: dict[str, Any] = {
        "variant": run.get("preset_id"),
        "status": run.get("status"),
        "run_dir": run.get("artifacts_dir"),
        "realtime_factor": performance.get("realtime_factor"),
        "wall_clock_sec": performance.get("wall_clock_sec"),
        "fallback_count": stability.get("fallback_count", 0),
        "proxy_quality_score": metrics.get("proxy_quality_score"),
    }
    if reference is None or run.get("status") != "completed":
        return row
    transcript_path = Path(run.get("output_json", ""))
    if not transcript_path.exists():
        return row
    hypothesis = transcript_from_dict(json.loads(transcript_path.read_text(encoding="utf-8")))
    score = score_transcript(reference, hypothesis, profile=profile)
    row.update(wer=score["wer"], cer=score["cer"], cp_wer=score.get("cp_wer"), der=score.get("der"), eval=score)
    return row


def run_manifest(
    manifest: ExperimentManifest,
    *,
    dry_run: bool = False,
    run_experiment: RunExperiment = _run_experiment,
) -> dict[str, Any]:
    """Execute a manifest and return the experiment result with a leaderboard.

    With ``dry_run=True`` no model runs: the expanded variant presets are returned
    for inspection. Otherwise each item is run and scored, and the result plus a
    ``leaderboard.md`` are written under the output directory.
    """

    if dry_run:
        return {"dry_run": True, "manifest": manifest.manifest_summary()}

    output_root = manifest.output_dir or (BASE_DIR / "runs" / f"workbench_{manifest.id}_{_timestamp()}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    item_results: list[dict[str, Any]] = []
    for item_id, audio, reference_path, duration_limit in _resolve_items(manifest):
        reference = load_reference(reference_path) if reference_path and Path(reference_path).exists() else None
        reference_text = flatten_text(reference) if reference is not None else None
        presets = _apply_duration_limit(list(manifest.presets), duration_limit)
        summary = run_experiment(
            audio_path=str(audio),
            presets=presets,
            output_dir=str(output_root / item_id),
            reference_text=reference_text,
            max_workers=manifest.max_workers,
        )
        item_rows = [_score_row(run, reference, manifest.profile) for run in summary.get("runs", [])]
        for row in item_rows:
            row["item_id"] = item_id
            rows.append(row)
        item_results.append({"item_id": item_id, "audio": str(audio), "experiment": summary.get("experiment_id"), "variants": item_rows})

    leaderboard = build_leaderboard(rows)
    result = {
        "experiment_id": f"workbench_{manifest.id}_{_timestamp()}",
        "manifest": manifest.manifest_summary(),
        "output_dir": str(output_root),
        "items": item_results,
        "leaderboard": leaderboard,
    }
    dump_json(output_root / "workbench_result.json", result)
    (output_root / "leaderboard.md").write_text(render_leaderboard(leaderboard, title=f"{manifest.id} leaderboard") + "\n", encoding="utf-8")
    return result


__all__ = ["run_manifest"]
