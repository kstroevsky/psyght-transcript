"""Reference-driven tuning workflow for Russian CTC + KenLM decoding."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

from phase1.backends import BackendSpec
from phase1.backends.ctc_kenlm import language_model_catalog
from phase1.compare_runtime.reference import load_reference_text
from phase1.config.env import BASE_DIR
from phase1.pipeline.audio import AudioBundle, load_audio_bundle
from phase1.quality import proxy_quality_metrics, realtime_factor, reference_error_rates, transcript_text
from phase1.runtime.options import RunOptions
from phase1.runtime.runner import execute

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class TuneManifestEntry:
    """One reference-backed evaluation case for decoder tuning."""

    case_id: str
    audio_path: str
    reference_path: str
    duration_limit: float | None


@dataclass(frozen=True)
class TuneCandidate:
    """One decoder configuration under evaluation."""

    candidate_id: str
    family: str
    description: str
    backend_options: dict[str, Any]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", value).strip("._-")
    return slug or "case"


def parse_int_grid(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",")]
    parsed = [int(item) for item in values if item]
    if not parsed:
        raise ValueError("Expected at least one integer grid value.")
    return parsed


def parse_float_grid(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",")]
    parsed = [float(item) for item in values if item]
    if not parsed:
        raise ValueError("Expected at least one float grid value.")
    return parsed


def load_tuning_manifest(path: str | Path) -> list[TuneManifestEntry]:
    """Load a JSON or JSONL tuning manifest with audio/reference pairs."""

    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    raw_entries: list[dict[str, Any]] = []
    if manifest_path.suffix.lower() == ".jsonl":
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid manifest row at line {line_number}: expected an object.")
            raw_entries.append(payload)
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("entries")
        if not isinstance(payload, list):
            raise ValueError("Manifest JSON must be a list or an object with an `entries` list.")
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("Manifest entries must be objects.")
            raw_entries.append(row)

    entries: list[TuneManifestEntry] = []
    for index, row in enumerate(raw_entries, start=1):
        audio_path = str(row.get("audio_path") or "").strip()
        reference_path = str(row.get("reference_path") or "").strip()
        if not audio_path:
            raise ValueError(f"Manifest entry {index} is missing `audio_path`.")
        if not reference_path:
            raise ValueError(f"Manifest entry {index} is missing `reference_path`.")
        case_id = str(row.get("case_id") or _slugify(Path(audio_path).stem))
        duration_limit = row.get("duration_limit")
        entries.append(
            TuneManifestEntry(
                case_id=case_id,
                audio_path=audio_path,
                reference_path=reference_path,
                duration_limit=float(duration_limit) if duration_limit is not None else None,
            )
        )
    if not entries:
        raise ValueError("Manifest must contain at least one evaluation case.")
    return entries


def default_external_lm_paths() -> tuple[str | None, str | None]:
    """Return the current off-the-shelf RU baseline LM paths when present locally."""

    binary = Path("/tmp/gigaam-ctc-with-lm/language_model/ru_3gram.bin")
    unigrams = Path("/tmp/gigaam-ctc-with-lm/language_model/unigrams.txt")
    return (str(binary) if binary.exists() else None, str(unigrams) if unigrams.exists() else None)


def build_tuning_candidates(
    *,
    beam_widths: list[int],
    alphas: list[float],
    betas: list[float],
    external_binary_path: str | None,
    external_unigrams_path: str | None,
    self_binary_path: str,
    self_unigrams_path: str,
    chunk_duration_sec: float,
    model_name: str,
    revision: str,
) -> list[TuneCandidate]:
    """Create the complete tuning candidate grid."""

    def base_options() -> dict[str, Any]:
        return {
            "device": "cpu",
            "model_name": model_name,
            "revision": revision,
            "chunk_duration_sec": chunk_duration_sec,
        }

    candidates = [
        TuneCandidate(
            candidate_id="greedy",
            family="greedy",
            description="Greedy punctuated CTC baseline.",
            backend_options={**base_options(), "decoder": {"strategy": "greedy"}},
        )
    ]
    for beam_width in beam_widths:
        candidates.append(
            TuneCandidate(
                candidate_id=f"beam_no_lm_bw{beam_width}",
                family="beam_no_lm",
                description=f"Beam search without LM (beam_width={beam_width}).",
                backend_options={
                    **base_options(),
                    "decoder": {
                        "strategy": "beam",
                        "beam_width": beam_width,
                        "lm": {"enabled": False},
                    },
                },
            )
        )

    def lm_candidates(
        family: str,
        prefix: str,
        binary_path: str,
        unigrams_path: str,
    ) -> list[TuneCandidate]:
        family_candidates: list[TuneCandidate] = []
        for beam_width, alpha, beta in product(beam_widths, alphas, betas):
            family_candidates.append(
                TuneCandidate(
                    candidate_id=f"{prefix}_bw{beam_width}_a{alpha:.2f}_b{beta:.2f}",
                    family=family,
                    description=f"{family} beam search (beam_width={beam_width}, alpha={alpha:.2f}, beta={beta:.2f}).",
                    backend_options={
                        **base_options(),
                        "decoder": {
                            "strategy": "beam",
                            "beam_width": beam_width,
                            "lm": {
                                "enabled": True,
                                "binary_path": binary_path,
                                "unigrams_path": unigrams_path,
                                "alpha": alpha,
                                "beta": beta,
                            },
                        },
                    },
                )
            )
        return family_candidates

    if external_binary_path and external_unigrams_path:
        candidates.extend(
            lm_candidates(
                family="beam_external_lm",
                prefix="beam_external",
                binary_path=external_binary_path,
                unigrams_path=external_unigrams_path,
            )
        )
    candidates.extend(
        lm_candidates(
            family="beam_self_lm",
            prefix="beam_self",
            binary_path=self_binary_path,
            unigrams_path=self_unigrams_path,
        )
    )
    return candidates


def rank_tuning_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort aggregated tuning results by WER, CER, then realtime factor."""

    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row.get("status") == "completed" else 1,
            float((row.get("metrics") or {}).get("wer", 1.0)),
            float((row.get("metrics") or {}).get("cer", 1.0)),
            float((row.get("performance") or {}).get("realtime_factor", 1e9)),
            str(row.get("candidate_id")),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _aggregate_case_results(candidate: TuneCandidate, case_results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in case_results if row.get("status") == "completed"]
    if len(completed) != len(case_results):
        return {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "description": candidate.description,
            "status": "failed",
            "metrics": {"wer": 1.0, "cer": 1.0},
            "performance": {"realtime_factor": 1e9, "wall_clock_sec": sum(float((row.get("performance") or {}).get("wall_clock_sec", 0.0)) for row in case_results)},
            "cases": case_results,
        }
    count = float(len(completed))
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "description": candidate.description,
        "status": "completed",
        "metrics": {
            "wer": round(sum(float(row["metrics"]["wer"]) for row in completed) / count, 6),
            "cer": round(sum(float(row["metrics"]["cer"]) for row in completed) / count, 6),
            "proxy_quality_score": round(sum(float(row["metrics"]["proxy_quality_score"]) for row in completed) / count, 6),
        },
        "performance": {
            "realtime_factor": round(sum(float(row["performance"]["realtime_factor"]) for row in completed) / count, 6),
            "wall_clock_sec": round(sum(float(row["performance"]["wall_clock_sec"]) for row in completed), 6),
        },
        "cases": case_results,
    }


def run_tuning_manifest(
    *,
    manifest_entries: list[TuneManifestEntry],
    output_dir: str | Path | None,
    beam_widths: list[int],
    alphas: list[float],
    betas: list[float],
    chunk_duration_sec: float = 20.0,
    model_name: str = "ai-sage/GigaAM-v3",
    revision: str = "e2e_ctc",
    external_binary_path: str | None = None,
    external_unigrams_path: str | None = None,
    self_binary_path: str | None = None,
    self_unigrams_path: str | None = None,
) -> dict[str, Any]:
    """Run the full tuning grid against a reference-backed manifest."""

    catalog = language_model_catalog()["ru"]
    self_binary = str(self_binary_path or catalog["binary_path"])
    self_unigrams = str(self_unigrams_path or catalog["unigrams_path"])
    if not Path(self_binary).exists():
        raise FileNotFoundError(f"Self-trained RU LM binary was not found: {self_binary}")
    if not Path(self_unigrams).exists():
        raise FileNotFoundError(f"Self-trained RU LM vocab was not found: {self_unigrams}")

    root = Path(output_dir) if output_dir else BASE_DIR / "runs" / f"tune_ctc_kenlm_ru_{_timestamp()}"
    root.mkdir(parents=True, exist_ok=True)

    candidates = build_tuning_candidates(
        beam_widths=beam_widths,
        alphas=alphas,
        betas=betas,
        external_binary_path=external_binary_path,
        external_unigrams_path=external_unigrams_path,
        self_binary_path=self_binary,
        self_unigrams_path=self_unigrams,
        chunk_duration_sec=chunk_duration_sec,
        model_name=model_name,
        revision=revision,
    )
    audio_cache: dict[str, AudioBundle] = {}
    reference_cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        case_results: list[dict[str, Any]] = []
        for entry in manifest_entries:
            audio_bundle = audio_cache.setdefault(entry.audio_path, load_audio_bundle(entry.audio_path))
            reference_text = reference_cache.setdefault(entry.reference_path, load_reference_text(entry.reference_path) or "")
            case_root = root / candidate.candidate_id / entry.case_id
            options = RunOptions(
                audio_path=entry.audio_path,
                language="ru",
                content_mode="single-speaker",
                save_intermediate_dir=str(case_root),
                output_dir=str(case_root),
                parallel_post_asr=True,
                skip_alignment=True,
                skip_diarization=True,
                duration_limit=entry.duration_limit,
                backend=BackendSpec(id="gigaam_ctc", options=candidate.backend_options),
                preset_id=candidate.candidate_id,
                experiment_id=root.name,
            )
            try:
                result = execute(options, audio_bundle=audio_bundle)
                candidate_text = transcript_text(result.transcript)
                metrics = {
                    **reference_error_rates(reference_text, candidate_text),
                    **proxy_quality_metrics(result.transcript),
                }
                case_results.append(
                    {
                        "case_id": entry.case_id,
                        "status": "completed",
                        "audio_path": entry.audio_path,
                        "reference_path": entry.reference_path,
                        "artifacts_dir": str(case_root),
                        "metrics": metrics,
                        "performance": {
                            "wall_clock_sec": float(result.wall_clock_sec),
                            "realtime_factor": realtime_factor(result.run_meta),
                        },
                    }
                )
            except Exception as exc:
                case_results.append(
                    {
                        "case_id": entry.case_id,
                        "status": "failed",
                        "audio_path": entry.audio_path,
                        "reference_path": entry.reference_path,
                        "artifacts_dir": str(case_root),
                        "error": str(exc),
                        "metrics": {"wer": 1.0, "cer": 1.0, "proxy_quality_score": 0.0},
                        "performance": {"wall_clock_sec": 0.0, "realtime_factor": 1e9},
                    }
                )
        rows.append(_aggregate_case_results(candidate, case_results))

    ranked = rank_tuning_results(rows)
    summary = {
        "tuning_id": root.name,
        "output_dir": str(root),
        "manifest": [
            {
                "case_id": entry.case_id,
                "audio_path": entry.audio_path,
                "reference_path": entry.reference_path,
                "duration_limit": entry.duration_limit,
            }
            for entry in manifest_entries
        ],
        "beam_widths": beam_widths,
        "alphas": alphas,
        "betas": betas,
        "best_candidate_id": ranked[0]["candidate_id"] if ranked else None,
        "results": ranked,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
