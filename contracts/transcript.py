"""Stable transcript data contract shared between transcription and ingestion phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Word:
    """Word-level timing emitted by alignment and preserved through storage."""

    word: str
    start: float
    end: float
    score: float = 0.0
    speaker: str | None = None


@dataclass
class Segment:
    """Speaker-attributed transcript segment with optional aligned words."""

    speaker: str
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    confidence: str | None = None
    source: str | None = None


@dataclass
class Transcript:
    """Top-level transcript payload written by phase1 and consumed by phase2."""

    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)


def _word_from_dict(payload: dict[str, Any]) -> Word:
    return Word(
        word=str(payload["word"]),
        start=float(payload["start"]),
        end=float(payload["end"]),
        score=float(payload.get("score", 0.0)),
        speaker=str(payload["speaker"]) if payload.get("speaker") is not None else None,
    )


def _segment_from_dict(payload: dict[str, Any]) -> Segment:
    return Segment(
        speaker=str(payload["speaker"]),
        start=float(payload["start"]),
        end=float(payload["end"]),
        text=str(payload["text"]),
        words=[_word_from_dict(word) for word in payload.get("words", [])],
        confidence=str(payload["confidence"]) if payload.get("confidence") is not None else None,
        source=str(payload["source"]) if payload.get("source") is not None else None,
    )


def transcript_from_dict(payload: dict[str, Any]) -> Transcript:
    """Build the shared transcript model from the persisted JSON contract."""

    return Transcript(
        language=str(payload["language"]),
        duration=float(payload["duration"]),
        segments=[_segment_from_dict(segment) for segment in payload.get("segments", [])],
    )


def _word_to_dict(word: Word) -> dict[str, float | str]:
    payload: dict[str, float | str] = {
        "word": word.word,
        "start": word.start,
        "end": word.end,
        "score": word.score,
    }
    if word.speaker is not None:
        payload["speaker"] = word.speaker
    return payload


def _segment_to_dict(segment: Segment) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "speaker": segment.speaker,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": [_word_to_dict(word) for word in segment.words],
    }
    if segment.confidence is not None:
        payload["confidence"] = segment.confidence
    if segment.source is not None:
        payload["source"] = segment.source
    return payload


def transcript_to_dict(transcript: Transcript) -> dict[str, Any]:
    """Convert the shared transcript model back into the persisted JSON contract."""

    return {
        "language": transcript.language,
        "duration": transcript.duration,
        "segments": [_segment_to_dict(segment) for segment in transcript.segments],
    }
