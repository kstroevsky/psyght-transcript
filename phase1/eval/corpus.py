"""Evaluation corpus: a set of (audio, gold-reference) pairs and aggregate scoring.

You cannot choose a pipeline from one file. This is the abstraction that turns
"score one transcript" into "score a pipeline across the whole held-out set and
aggregate", which is what actually drives accuracy decisions.

A corpus is a small YAML/JSON manifest. Paths are resolved relative to the
manifest file, so dropping in a new gold transcript is a two-line edit. Adding
items never touches code.

This module is deliberately model-free: it scores *already-produced* hypothesis
transcripts against the references (cheap, no GPU). Producing the hypotheses by
running a preset across the corpus is orchestration and lives in the workbench.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from contracts.transcript import Transcript, transcript_from_dict
from phase1.eval.normalize import NormProfile, get_profile
from phase1.eval.reference import ReferenceTranscript, load_reference
from phase1.eval.score import score_transcript

_ITEM_KEYS = {"id", "audio", "reference", "hypothesis", "language", "duration_limit", "notes"}
_CORPUS_KEYS = {"name", "profile", "defaults", "items"}


@dataclass(frozen=True)
class CorpusItem:
    """One evaluation unit: an audio file paired with its gold reference."""

    id: str
    reference: Path
    audio: Path | None = None
    hypothesis: Path | None = None
    language: str | None = None
    duration_limit: float | None = None
    notes: str | None = None

    def load_reference(self) -> ReferenceTranscript:
        return load_reference(self.reference)


@dataclass(frozen=True)
class Corpus:
    """A named, validated set of evaluation items with a default norm profile."""

    name: str
    items: tuple[CorpusItem, ...]
    profile: NormProfile
    source_path: Path | None = None


def load_corpus(path: str | Path) -> Corpus:
    """Load and validate a corpus manifest (YAML or JSON)."""

    manifest_path = Path(path)
    payload = _read_manifest(manifest_path)
    unknown = sorted(set(payload) - _CORPUS_KEYS)
    if unknown:
        raise ValueError(f"Unknown corpus keys in {manifest_path}: {', '.join(unknown)}")
    if not isinstance(payload.get("items"), list) or not payload["items"]:
        raise ValueError(f"Corpus {manifest_path} must define a non-empty 'items' list.")

    base = manifest_path.parent
    defaults = dict(payload.get("defaults") or {})
    profile = get_profile(payload.get("profile"))
    seen: set[str] = set()
    items: list[CorpusItem] = []
    for raw in payload["items"]:
        item = _build_item(raw, defaults, base, manifest_path)
        if item.id in seen:
            raise ValueError(f"Duplicate corpus item id {item.id!r} in {manifest_path}")
        seen.add(item.id)
        items.append(item)
    return Corpus(name=str(payload.get("name") or manifest_path.stem), items=tuple(items), profile=profile, source_path=manifest_path)


def score_corpus(
    corpus: Corpus,
    hypotheses: dict[str, Transcript],
    *,
    profile: NormProfile | str | None = None,
) -> dict[str, Any]:
    """Score a mapping of ``item_id -> hypothesis transcript`` against the corpus.

    Items without a supplied hypothesis are reported as ``missing`` rather than
    skipped silently. Returns per-item scores plus macro (mean of per-file) and
    micro (pooled edits / pooled units) aggregates.
    """

    resolved = get_profile(profile) if profile is not None else corpus.profile
    per_item: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in corpus.items:
        hypothesis = hypotheses.get(item.id)
        if hypothesis is None:
            missing.append(item.id)
            continue
        score = score_transcript(item.load_reference(), hypothesis, profile=resolved)
        per_item.append({"id": item.id, "notes": item.notes, **score})
    return {
        "corpus": corpus.name,
        "profile": resolved.name,
        "n_items": len(corpus.items),
        "n_scored": len(per_item),
        "missing": missing,
        "items": per_item,
        "aggregate": _aggregate(per_item),
    }


def load_hypotheses_from_dir(corpus: Corpus, transcripts_dir: str | Path) -> dict[str, Transcript]:
    """Resolve ``item_id -> Transcript`` by looking for ``<dir>/<id>.json``.

    Falls back to a per-item ``hypothesis:`` path declared in the manifest.
    """

    directory = Path(transcripts_dir) if transcripts_dir is not None else None
    resolved: dict[str, Transcript] = {}
    for item in corpus.items:
        candidate = None
        if directory is not None and (directory / f"{item.id}.json").exists():
            candidate = directory / f"{item.id}.json"
        elif item.hypothesis is not None and item.hypothesis.exists():
            candidate = item.hypothesis
        if candidate is not None:
            resolved[item.id] = transcript_from_dict(json.loads(candidate.read_text(encoding="utf-8")))
    return resolved


def _aggregate(per_item: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_item:
        return {}
    wers = [row["wer"] for row in per_item]
    cers = [row["cer"] for row in per_item]
    cp_wers = [row["cp_wer"] for row in per_item if "cp_wer" in row]
    ders = [row["der"] for row in per_item if "der" in row]
    pooled_wer = _micro(per_item, "wer_detail")
    pooled_cer = _micro(per_item, "cer_detail")
    aggregate = {
        "macro_wer": round(mean(wers), 6),
        "macro_cer": round(mean(cers), 6),
        "micro_wer": pooled_wer,
        "micro_cer": pooled_cer,
    }
    if cp_wers:
        aggregate["macro_cp_wer"] = round(mean(cp_wers), 6)
    if ders:
        aggregate["mean_der"] = round(mean(ders), 6)
    return aggregate


def _micro(per_item: list[dict[str, Any]], detail_key: str) -> float | None:
    total_edits = 0
    total_units = 0
    for row in per_item:
        detail = row.get(detail_key) or {}
        substitutions = detail.get("substitutions")
        if substitutions is None:  # S/D/I unavailable (huge input without rapidfuzz)
            return None
        total_edits += substitutions + detail.get("deletions", 0) + detail.get("insertions", 0)
        total_units += detail.get("reference_length", 0)
    return round(total_edits / total_units, 6) if total_units else None


def _read_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # PyYAML is a runtime dependency of the project env.

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Corpus manifest must be a mapping: {path}")
    return payload


def _build_item(raw: dict[str, Any], defaults: dict[str, Any], base: Path, manifest_path: Path) -> CorpusItem:
    if not isinstance(raw, dict):
        raise ValueError(f"Corpus item must be a mapping in {manifest_path}: {raw!r}")
    unknown = sorted(set(raw) - _ITEM_KEYS)
    if unknown:
        raise ValueError(f"Unknown corpus item keys in {manifest_path}: {', '.join(unknown)}")
    merged = {**defaults, **raw}
    item_id = merged.get("id")
    if not item_id:
        raise ValueError(f"Corpus item requires an 'id' in {manifest_path}: {raw!r}")
    if not merged.get("reference"):
        raise ValueError(f"Corpus item {item_id!r} requires a 'reference' in {manifest_path}")
    duration_limit = merged.get("duration_limit")
    return CorpusItem(
        id=str(item_id),
        reference=_resolve(base, merged["reference"]),
        audio=_resolve(base, merged.get("audio")) if merged.get("audio") else None,
        hypothesis=_resolve(base, merged.get("hypothesis")) if merged.get("hypothesis") else None,
        language=_opt_str(merged.get("language")),
        duration_limit=float(duration_limit) if duration_limit is not None else None,
        notes=_opt_str(merged.get("notes")),
    )


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path)


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None
