"""Time-window helpers shared by therapy-pipeline stages and fine-tuning prep."""

from __future__ import annotations

import re
from typing import Any, Iterator

_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace while preserving punctuation and casing."""

    return _SPACE_RE.sub(" ", str(text or "").replace("\n", " ")).strip()


def overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    """Return the overlap duration between two time windows."""

    return max(0.0, min(end, other_end) - max(start, other_start))


def sort_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort time-stamped segments deterministically."""

    return sorted(
        segments,
        key=lambda segment: (
            float(segment.get("start", 0.0)),
            float(segment.get("end", segment.get("start", 0.0))),
            str(segment.get("text", "")),
        ),
    )


def segment_words_text(segment: dict[str, Any], start: float, end: float) -> str:
    """Extract word-level text overlapping one time window when timings are present."""

    pieces: list[str] = []
    for word in segment.get("words", []):
        if "start" not in word or "end" not in word:
            continue
        if overlap(start, end, float(word["start"]), float(word["end"])) <= 0.0:
            continue
        token = normalize_text(str(word.get("word", "")))
        if token:
            pieces.append(token)
    return normalize_text(" ".join(pieces))


def text_for_window(segments: list[dict[str, Any]], start: float, end: float) -> str:
    """Collect transcript text overlapping one time window."""

    pieces: list[str] = []
    for segment in sort_segments(segments):
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        if overlap(start, end, segment_start, segment_end) <= 0.0:
            continue
        words_text = segment_words_text(segment, start, end)
        piece = words_text or normalize_text(str(segment.get("text", "")))
        if piece:
            pieces.append(piece)
    return normalize_text(" ".join(pieces))


def max_segment_end(segments: list[dict[str, Any]]) -> float:
    """Return the maximum segment end timestamp or zero for empty input."""

    if not segments:
        return 0.0
    return max(float(segment.get("end", segment.get("start", 0.0))) for segment in segments)


def iter_fixed_windows(duration: float, window_seconds: float) -> Iterator[tuple[float, float]]:
    """Yield contiguous fixed-size windows from zero to the provided duration."""

    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be positive, got {window_seconds}.")
    cursor = 0.0
    while cursor < duration:
        next_cursor = min(duration, cursor + window_seconds)
        yield round(cursor, 3), round(next_cursor, 3)
        cursor = next_cursor
