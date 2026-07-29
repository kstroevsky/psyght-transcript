"""Structured gold-reference loading for accuracy evaluation.

A reference can arrive in three shapes seen in this repo:

1. **phase1 transcript JSON** -- ``{"language", "duration", "segments": [...]}``
   with per-segment ``speaker``/``start``/``end``/``text`` (e.g.
   ``phase1/test_first2min.json``).
2. **Gemini review JSON** -- a bare list of ``{"speaker", "start_sec", "text"}``
   with no end times and no words (e.g. the first-10-min Bogomolov gold). End
   times are inferred from the next segment's start so speaker-aware metrics
   (cpWER, DER) still work, flagged as ``inferred_ends``.
3. **Plain text** -- either ``[ts -> ts] SPEAKER: line`` transcript lines or raw
   prose; only flat text is recovered (no reliable speakers/times).

The loader normalizes all three into a :class:`ReferenceTranscript` so the
metric code never has to branch on input format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LINE_RE = re.compile(r"^\[[^\]]*\]\s+([^:]+):\s*(.*)$")


@dataclass(frozen=True)
class RefSegment:
    """One reference segment; ``start``/``end`` are ``None`` when unknown."""

    speaker: str | None
    start: float | None
    end: float | None
    text: str


@dataclass(frozen=True)
class ReferenceTranscript:
    """A format-normalized gold reference plus provenance flags."""

    segments: tuple[RefSegment, ...]
    language: str | None = None
    source_path: str | None = None
    inferred_ends: bool = False

    @property
    def has_speakers(self) -> bool:
        return any(seg.speaker for seg in self.segments)

    @property
    def has_times(self) -> bool:
        return all(seg.start is not None for seg in self.segments) and bool(self.segments)


def flatten_text(reference: ReferenceTranscript) -> str:
    """Join all segment texts into one space-separated string."""

    return " ".join(seg.text.strip() for seg in reference.segments if seg.text.strip())


def text_by_speaker(reference: ReferenceTranscript) -> dict[str, str]:
    """Concatenate text per speaker label (chronological order preserved)."""

    grouped: dict[str, list[str]] = {}
    for seg in reference.segments:
        if not seg.text.strip():
            continue
        grouped.setdefault(seg.speaker or "UNKNOWN", []).append(seg.text.strip())
    return {speaker: " ".join(parts) for speaker, parts in grouped.items()}


def load_reference(path: str | Path) -> ReferenceTranscript:
    """Load and normalize a gold reference from JSON or text on disk."""

    reference_path = Path(path)
    raw = reference_path.read_text(encoding="utf-8")
    if reference_path.suffix.lower() == ".json":
        return _from_json(json.loads(raw), str(reference_path))
    return _from_text(raw, str(reference_path))


def reference_from_payload(payload: Any, source_path: str | None = None) -> ReferenceTranscript:
    """Build a reference from an already-parsed JSON payload (dict or list)."""

    return _from_json(payload, source_path)


def _from_json(payload: Any, source_path: str | None) -> ReferenceTranscript:
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        segments = tuple(
            RefSegment(
                speaker=_opt_str(seg.get("speaker")),
                start=_opt_float(seg.get("start", seg.get("start_sec"))),
                end=_opt_float(seg.get("end", seg.get("end_sec"))),
                text=str(seg.get("text", "")),
            )
            for seg in payload["segments"]
        )
        return ReferenceTranscript(segments=segments, language=_opt_str(payload.get("language")), source_path=source_path)
    if isinstance(payload, list):
        return _from_segment_list(payload, source_path)
    raise ValueError("Unsupported reference JSON: expected a transcript object or a segment list.")


def _from_segment_list(items: list[dict[str, Any]], source_path: str | None) -> ReferenceTranscript:
    starts = [_opt_float(item.get("start", item.get("start_sec"))) for item in items]
    ends = [_opt_float(item.get("end", item.get("end_sec"))) for item in items]
    inferred = False
    durations = [e - s for s, e in zip(starts, ends) if s is not None and e is not None and e > s]
    typical = (sum(durations) / len(durations)) if durations else 2.0
    for index in range(len(items)):
        if ends[index] is not None:
            continue
        next_start = starts[index + 1] if index + 1 < len(items) else None
        if starts[index] is not None and next_start is not None and next_start > starts[index]:
            ends[index] = next_start
        elif starts[index] is not None:
            ends[index] = starts[index] + typical
        inferred = True
    segments = tuple(
        RefSegment(
            speaker=_opt_str(item.get("speaker")),
            start=starts[index],
            end=ends[index],
            text=str(item.get("text", "")),
        )
        for index, item in enumerate(items)
    )
    return ReferenceTranscript(segments=segments, source_path=source_path, inferred_ends=inferred)


def _from_text(raw: str, source_path: str | None) -> ReferenceTranscript:
    segments: list[RefSegment] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _LINE_RE.match(stripped)
        if match:
            segments.append(RefSegment(speaker=match.group(1).strip(), start=None, end=None, text=match.group(2).strip()))
        else:
            segments.append(RefSegment(speaker=None, start=None, end=None, text=stripped))
    return ReferenceTranscript(segments=tuple(segments), source_path=source_path)


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
