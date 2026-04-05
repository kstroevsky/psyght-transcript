"""Speaker-assignment helpers that convert canonical stage output into the shared transcript contract."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from contracts.transcript import Segment, Transcript, Word
from phase1.pipeline.diarize import diarization_to_records

UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"
ASSIGNMENT_STRATEGY_OVERLAP = "overlap"
ASSIGNMENT_STRATEGY_WHISPERX = "whisperx"
SPEAKER_LABEL_STYLE_CANONICAL = "canonical"
SPEAKER_LABEL_STYLE_ALPHA = "alpha"


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _point_speaker(timestamp: float, diarization: list[dict[str, Any]]) -> str | None:
    for record in diarization:
        start = float(record.get("start", 0.0))
        end = float(record.get("end", start))
        if start <= timestamp <= end:
            return str(record.get("speaker", UNKNOWN_SPEAKER))
    return None


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            float(record["start"]) if "start" in record else float("inf"),
            float(record["end"]) if "end" in record else (float(record["start"]) if "start" in record else float("inf")),
            str(record.get("speaker", UNKNOWN_SPEAKER)),
        ),
    )


def _require_supported_strategy(strategy: str) -> str:
    normalized = str(strategy or ASSIGNMENT_STRATEGY_OVERLAP).strip().lower()
    if normalized not in {ASSIGNMENT_STRATEGY_OVERLAP, ASSIGNMENT_STRATEGY_WHISPERX}:
        raise ValueError(f"Unsupported speaker assignment strategy: {strategy!r}")
    return normalized


def _require_supported_label_style(label_style: str) -> str:
    normalized = str(label_style or SPEAKER_LABEL_STYLE_CANONICAL).strip().lower()
    if normalized not in {SPEAKER_LABEL_STYLE_CANONICAL, SPEAKER_LABEL_STYLE_ALPHA}:
        raise ValueError(f"Unsupported speaker label style: {label_style!r}")
    return normalized


def _speaker_for_window(start: float, end: float, diarization: list[dict[str, Any]]) -> str:
    totals: dict[str, float] = defaultdict(float)
    if end <= start:
        return _point_speaker(start, diarization) or UNKNOWN_SPEAKER

    for record in diarization:
        speaker = str(record.get("speaker", UNKNOWN_SPEAKER))
        totals[speaker] += _overlap(start, end, float(record.get("start", 0.0)), float(record.get("end", 0.0)))

    if not totals:
        return UNKNOWN_SPEAKER
    speaker, duration = max(totals.items(), key=lambda item: (item[1], item[0]))
    return speaker if duration > 0 else UNKNOWN_SPEAKER


def _assign_words(
    words: list[dict[str, Any]],
    diarization: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    assigned: list[dict[str, Any]] = []
    speaker_durations: dict[str, float] = defaultdict(float)

    for word in words:
        payload = dict(word)
        if "start" in payload and "end" in payload:
            speaker = _speaker_for_window(float(payload["start"]), float(payload["end"]), diarization)
            payload["speaker"] = speaker
            speaker_durations[speaker] += max(0.0, float(payload["end"]) - float(payload["start"]))
        assigned.append(payload)
    return assigned, speaker_durations


def _segment_speaker(
    segment: dict[str, Any],
    word_speaker_durations: dict[str, float],
    diarization: list[dict[str, Any]],
) -> str:
    if word_speaker_durations:
        speaker, duration = max(word_speaker_durations.items(), key=lambda item: (item[1], item[0]))
        if duration > 0:
            return speaker
    return _speaker_for_window(float(segment["start"]), float(segment["end"]), diarization)


def _assign_segments(
    aligned_segments: list[dict[str, Any]],
    diarize_segments: Any,
) -> list[dict[str, Any]]:
    diarization = _sort_records(diarization_to_records(diarize_segments))
    assigned_segments: list[dict[str, Any]] = []
    for segment in _sort_records(aligned_segments):
        payload = dict(segment)
        assigned_words, word_speaker_durations = _assign_words(_sort_records(list(payload.get("words", []))), diarization)
        payload["words"] = assigned_words
        payload["speaker"] = _segment_speaker(payload, word_speaker_durations, diarization)
        assigned_segments.append(payload)
    return assigned_segments


def _assign_segments_with_whisperx(
    aligned_segments: list[dict[str, Any]],
    diarize_segments: Any,
) -> list[dict[str, Any]]:
    import whisperx

    original_segments = [
        {**dict(segment), "words": [dict(word) for word in list(segment.get("words", []))]}
        for segment in _sort_records(aligned_segments)
    ]
    result = whisperx.assign_word_speakers(diarize_segments, {"segments": original_segments})
    assigned_segments = _sort_records(list(result.get("segments", [])))
    for original, assigned in zip(original_segments, assigned_segments):
        for key in ("confidence", "source"):
            if original.get(key) is not None and assigned.get(key) is None:
                assigned[key] = original[key]
    return assigned_segments


def _join_segment_text(left: str, right: str) -> str:
    left_clean = left.strip()
    right_clean = right.strip().lstrip("—").strip()
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean
    return f"{left_clean} {right_clean}".strip()


def _merge_consecutive_same_speaker_segments(
    segments: list[dict[str, Any]],
    *,
    merge_gap_sec: float,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    merged: list[dict[str, Any]] = []
    for segment in _sort_records(segments):
        payload = dict(segment)
        payload["words"] = _sort_records(list(payload.get("words", [])))
        if not merged:
            merged.append(payload)
            continue

        previous = merged[-1]
        gap = float(payload.get("start", 0.0)) - float(previous.get("end", 0.0))
        same_speaker = str(previous.get("speaker", UNKNOWN_SPEAKER)) == str(payload.get("speaker", UNKNOWN_SPEAKER))
        if same_speaker and gap <= merge_gap_sec:
            previous["end"] = max(float(previous.get("end", 0.0)), float(payload.get("end", 0.0)))
            previous["text"] = _join_segment_text(str(previous.get("text", "")), str(payload.get("text", "")))
            previous["words"] = _sort_records(list(previous.get("words", [])) + payload["words"])
            if previous.get("confidence") == "low" or payload.get("confidence") == "low":
                previous["confidence"] = "low"
            elif previous.get("confidence") is not None or payload.get("confidence") is not None:
                previous["confidence"] = previous.get("confidence") or payload.get("confidence")
            if previous.get("source") == "ctc_fallback" or payload.get("source") == "ctc_fallback":
                previous["source"] = "ctc_fallback"
            elif previous.get("source") is not None or payload.get("source") is not None:
                previous["source"] = previous.get("source") or payload.get("source")
            continue
        merged.append(payload)
    return merged


def _alpha_speaker_name(index: int) -> str:
    quotient = index
    label = ""
    while True:
        quotient, remainder = divmod(quotient, 26)
        label = chr(ord("A") + remainder) + label
        if quotient == 0:
            return label
        quotient -= 1


def _normalize_speaker_labels(
    segments: list[dict[str, Any]],
    *,
    label_style: str,
) -> list[dict[str, Any]]:
    normalized_style = _require_supported_label_style(label_style)
    if normalized_style == SPEAKER_LABEL_STYLE_CANONICAL:
        return segments

    speaker_map: dict[str, str] = {}

    def map_speaker(value: Any) -> str:
        speaker = str(value or UNKNOWN_SPEAKER)
        if speaker == UNKNOWN_SPEAKER:
            return speaker
        if speaker not in speaker_map:
            speaker_map[speaker] = _alpha_speaker_name(len(speaker_map))
        return speaker_map[speaker]

    normalized_segments: list[dict[str, Any]] = []
    for segment in segments:
        payload = dict(segment)
        payload["speaker"] = map_speaker(payload.get("speaker"))
        normalized_words: list[dict[str, Any]] = []
        for word in list(payload.get("words", [])):
            word_payload = dict(word)
            if word_payload.get("speaker") is not None:
                word_payload["speaker"] = map_speaker(word_payload.get("speaker"))
            normalized_words.append(word_payload)
        payload["words"] = normalized_words
        normalized_segments.append(payload)
    return normalized_segments


def assigned_segments_for_dump(
    aligned_segments: list[dict[str, Any]],
    diarize_segments: Any,
    *,
    merge_consecutive_same_speaker: bool = False,
    merge_gap_sec: float = 0.35,
    strategy: str = ASSIGNMENT_STRATEGY_OVERLAP,
    speaker_label_style: str = SPEAKER_LABEL_STYLE_CANONICAL,
) -> list[dict[str, Any]]:
    """Return the merged speaker-assigned segments used for intermediate artifacts."""

    normalized_strategy = _require_supported_strategy(strategy)
    if normalized_strategy == ASSIGNMENT_STRATEGY_WHISPERX:
        assigned = _assign_segments_with_whisperx(aligned_segments, diarize_segments)
    else:
        assigned = _assign_segments(aligned_segments, diarize_segments)
    assigned = _normalize_speaker_labels(assigned, label_style=speaker_label_style)
    if merge_consecutive_same_speaker:
        return _merge_consecutive_same_speaker_segments(assigned, merge_gap_sec=merge_gap_sec)
    return assigned


def assign_and_build(
    aligned_segments: list[dict[str, Any]],
    diarize_segments: Any,
    language: str,
    duration: float,
    *,
    merge_consecutive_same_speaker: bool = False,
    merge_gap_sec: float = 0.35,
    strategy: str = ASSIGNMENT_STRATEGY_OVERLAP,
    speaker_label_style: str = SPEAKER_LABEL_STYLE_CANONICAL,
) -> Transcript:
    """Assign speakers to aligned segments and build the shared transcript model."""

    merged = assigned_segments_for_dump(
        aligned_segments,
        diarize_segments,
        merge_consecutive_same_speaker=merge_consecutive_same_speaker,
        merge_gap_sec=merge_gap_sec,
        strategy=strategy,
        speaker_label_style=speaker_label_style,
    )
    segments: list[Segment] = []
    for segment in merged:
        words = [
            Word(
                word=str(word.get("word", "")).strip(),
                start=float(word["start"]),
                end=float(word["end"]),
                score=float(word.get("score", 0.0)),
                speaker=str(word["speaker"]) if word.get("speaker") is not None else None,
            )
            for word in segment.get("words", [])
            if "start" in word and "end" in word
        ]
        segments.append(
            Segment(
                speaker=str(segment.get("speaker", UNKNOWN_SPEAKER)),
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=str(segment.get("text", "")).strip(),
                words=words,
                confidence=str(segment["confidence"]) if segment.get("confidence") is not None else None,
                source=str(segment["source"]) if segment.get("source") is not None else None,
            )
        )
    return Transcript(language=language, duration=duration, segments=segments)
