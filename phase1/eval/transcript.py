"""Adapters from the shared ``contracts.Transcript`` to evaluation inputs.

Keeps the metric modules ignorant of the transcript dataclass shape: they take
plain text / per-speaker dicts / timed tuples, and this module is the only place
that knows how to extract those from a hypothesis ``Transcript``.
"""

from __future__ import annotations

from contracts.transcript import Transcript


def transcript_text(transcript: Transcript) -> str:
    """Flatten transcript segments into one space-separated string."""

    return " ".join(segment.text.strip() for segment in transcript.segments if segment.text.strip())


def transcript_by_speaker(transcript: Transcript) -> dict[str, str]:
    """Concatenate text per speaker label, preserving chronological order."""

    grouped: dict[str, list[str]] = {}
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        grouped.setdefault(segment.speaker or "UNKNOWN", []).append(segment.text.strip())
    return {speaker: " ".join(parts) for speaker, parts in grouped.items()}


def transcript_timed_segments(transcript: Transcript) -> list[tuple[float, float, str]]:
    """Return ``(start, end, speaker)`` tuples for diarization scoring."""

    return [
        (float(segment.start), float(segment.end), segment.speaker or "UNKNOWN")
        for segment in transcript.segments
        if segment.end is not None and segment.start is not None
    ]
