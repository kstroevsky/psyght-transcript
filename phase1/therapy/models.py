"""Lightweight contracts for the additive therapy transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MergeRequest:
    """One time-aligned merge request across CTC, Whisper, and Canary transcripts."""

    language: str
    start: float
    end: float
    ctc_text: str
    whisper_text: str
    canary_text: str = ""


@dataclass(frozen=True)
class MergeDecision:
    """Normalized output produced by the merge stage."""

    text: str
    confidence: str = "high"
    source: str = "merged"
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrainingPair:
    """One seq2seq fine-tuning example for the ECLM stage."""

    ctc: str
    whisper: str
    clean: str
    start_sec: float
    end_sec: float
    chunk_id: str | None = None
