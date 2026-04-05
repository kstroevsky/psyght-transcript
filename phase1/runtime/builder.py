"""Helpers that build structured transcript objects from stage-level pipeline outputs."""

from __future__ import annotations

from contracts.transcript import Segment, Transcript, Word, transcript_to_dict


def _word_from_payload(payload: dict[str, object]) -> Word | None:
    if "start" not in payload or "end" not in payload:
        return None
    return Word(
        word=str(payload.get("word", "")).strip(),
        start=float(payload["start"]),
        end=float(payload["end"]),
        score=float(payload.get("score", 0.0)),
        speaker=str(payload["speaker"]) if payload.get("speaker") is not None else None,
    )


def build_single_speaker_transcript(
    aligned_segments: list[dict[str, object]],
    language: str,
    duration: float,
    content_mode: str,
) -> Transcript:
    """Build the synthetic single-speaker transcript used when diarization is skipped."""

    speaker = "SPEAKER_00" if content_mode == "single-speaker" else "SPEAKER_UNKNOWN"
    segments: list[Segment] = []
    for segment in aligned_segments:
        words = [
            word
            for word in (_word_from_payload(payload) for payload in segment.get("words", []))
            if word is not None
        ]
        segments.append(
            Segment(
                speaker=speaker,
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=str(segment.get("text", "")).strip(),
                words=words,
                confidence=str(segment["confidence"]) if segment.get("confidence") is not None else None,
                source=str(segment["source"]) if segment.get("source") is not None else None,
            )
        )
    return Transcript(language=language, duration=duration, segments=segments)


def synthetic_diarization_segments(duration: float) -> list[dict[str, float | str]]:
    """Mirror the old synthetic diarization artifact used for single-speaker runs."""

    return [{"speaker": "SPEAKER_00", "start": 0.0, "end": duration}]


def structured_segments(segments: list[Segment]) -> list[dict[str, object]]:
    """Serialize only the segment list for intermediate assignment artifacts."""

    return transcript_to_dict(Transcript(language="", duration=0.0, segments=segments))["segments"]
