"""Experiment manifests: declare many pipeline variants in one file.

A manifest is the unit of "an experiment": a set of pipeline *variants* run
against one audio file or a whole eval corpus, plus the reference binding and
normalization profile used to score them.

Design choice: a variant expands into the *exact same* validated
``ComparePreset`` the compare runtime already consumes — there is no second
pipeline schema to keep in sync. The manifest only adds what a folder of preset
JSONs cannot: shared ``defaults``, variant-to-variant ``extends`` inheritance,
and binding to an audio file / eval corpus + reference for scoring. Every variant
still goes through ``build_compare_preset``, so unknown pipeline/backend keys and
bad backend ids fail validation here, before any model runs.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phase1.compare_runtime.presets import ComparePreset, build_compare_preset

_MANIFEST_KEYS = {"id", "description", "audio", "reference", "corpus", "profile", "max_workers", "output_dir", "defaults", "variants"}
_VARIANT_KEYS = {"id", "description", "extends", "pipeline", "backend"}
_DEFAULTS_KEYS = {"pipeline", "backend"}


@dataclass(frozen=True)
class ExperimentManifest:
    """A validated experiment: variants + what to run and score them against."""

    id: str
    presets: tuple[ComparePreset, ...]
    description: str | None = None
    audio: Path | None = None
    reference: Path | None = None
    corpus: Path | None = None
    profile: str | None = None
    max_workers: int = 1
    output_dir: Path | None = None
    source_path: Path | None = None

    def manifest_summary(self) -> dict[str, Any]:
        """JSON-safe summary recorded alongside experiment artifacts."""

        return {
            "id": self.id,
            "description": self.description,
            "audio": str(self.audio) if self.audio else None,
            "reference": str(self.reference) if self.reference else None,
            "corpus": str(self.corpus) if self.corpus else None,
            "profile": self.profile,
            "max_workers": self.max_workers,
            "variants": [preset.manifest() for preset in self.presets],
        }


def load_manifest(path: str | Path) -> ExperimentManifest:
    """Load and fully validate an experiment manifest (YAML or JSON)."""

    manifest_path = Path(path)
    payload = _read_manifest(manifest_path)
    unknown = sorted(set(payload) - _MANIFEST_KEYS)
    if unknown:
        raise ValueError(f"Unknown manifest keys in {manifest_path}: {', '.join(unknown)}")
    manifest_id = payload.get("id")
    if not manifest_id or not isinstance(manifest_id, str):
        raise ValueError(f"Manifest 'id' is required in {manifest_path}")

    if bool(payload.get("audio")) == bool(payload.get("corpus")):
        raise ValueError(f"Manifest {manifest_path} must set exactly one of 'audio' or 'corpus'.")

    presets = _expand_variants(payload, manifest_path)
    base = manifest_path.parent
    return ExperimentManifest(
        id=str(manifest_id),
        presets=presets,
        description=payload.get("description"),
        audio=_resolve(base, payload.get("audio")),
        reference=_resolve(base, payload.get("reference")),
        corpus=_resolve(base, payload.get("corpus")),
        profile=_opt_str(payload.get("profile")),
        max_workers=int(payload.get("max_workers", 1)),
        output_dir=_resolve(base, payload.get("output_dir")),
        source_path=manifest_path,
    )


def _expand_variants(payload: dict[str, Any], manifest_path: Path) -> tuple[ComparePreset, ...]:
    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"Manifest {manifest_path} must define a non-empty 'variants' list.")
    defaults = payload.get("defaults") or {}
    _reject_unknown(defaults, _DEFAULTS_KEYS, f"defaults in {manifest_path}")

    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in variants:
        if not isinstance(raw, dict):
            raise ValueError(f"Each variant must be a mapping in {manifest_path}: {raw!r}")
        _reject_unknown(raw, _VARIANT_KEYS, f"variant in {manifest_path}")
        variant_id = raw.get("id")
        if not variant_id or not isinstance(variant_id, str):
            raise ValueError(f"Each variant requires a string 'id' in {manifest_path}")
        if variant_id in by_id:
            raise ValueError(f"Duplicate variant id {variant_id!r} in {manifest_path}")
        by_id[variant_id] = raw
        order.append(variant_id)

    cache: dict[str, dict[str, Any]] = {}
    presets: list[ComparePreset] = []
    for variant_id in order:
        merged = _deep_merge(copy.deepcopy(defaults), _resolve_chain(variant_id, by_id, cache, set(), manifest_path))
        merged["id"] = variant_id  # the variant's own id always wins over inherited
        presets.append(build_compare_preset(merged, source=f"{manifest_path}#{variant_id}"))
    return tuple(presets)


def _resolve_chain(variant_id: str, by_id: dict[str, dict], cache: dict[str, dict], stack: set[str], manifest_path: Path) -> dict[str, Any]:
    if variant_id in cache:
        return cache[variant_id]
    if variant_id in stack:
        raise ValueError(f"Cyclic 'extends' involving {variant_id!r} in {manifest_path}")
    raw = copy.deepcopy(by_id[variant_id])
    extends = raw.pop("extends", None)
    if extends is not None and extends not in by_id:
        raise ValueError(f"Variant {variant_id!r} extends unknown variant {extends!r} in {manifest_path}")
    stack.add(variant_id)
    base = _resolve_chain(extends, by_id, cache, stack, manifest_path) if extends else {}
    stack.discard(variant_id)
    merged = _deep_merge(copy.deepcopy(base), raw)
    cache[variant_id] = merged
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _read_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return payload


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {', '.join(unknown)}")


def _resolve(base: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path)


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None
