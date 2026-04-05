"""Pure row-building helpers for transcript ingestion into PostgreSQL."""

from __future__ import annotations

import json

from contracts.transcript import Segment, Transcript


Row = tuple[str, str, float, float, str, str, list[float]]


def serialize_words(segment: Segment) -> str:
    """Serialize word-level timings into the JSONB payload stored per transcript segment."""

    return json.dumps(
        [
            {
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "score": word.score,
                **({"speaker": word.speaker} if word.speaker is not None else {}),
            }
            for word in segment.words
        ],
        ensure_ascii=False,
    )


def build_segment_rows(
    meeting_id: str,
    transcript: Transcript,
    embeddings: list[list[float]],
) -> list[Row]:
    """Build `segments` rows while preserving transcript ordering and word payloads."""

    rows: list[Row] = []
    for segment, embedding in zip(transcript.segments, embeddings):
        rows.append(
            (
                meeting_id,
                segment.speaker,
                segment.start,
                segment.end,
                segment.text,
                serialize_words(segment),
                embedding,
            )
        )
    return rows
