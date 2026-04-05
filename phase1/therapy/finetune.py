"""Fine-tuning data prep for the therapy ECLM path."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from phase1.pipeline.audio import SAMPLE_RATE, load_audio_bundle, write_wav_mono
from phase1.therapy.models import TrainingPair
from phase1.therapy.windows import iter_fixed_windows, max_segment_end, normalize_text, text_for_window

_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def _normalize_review_segment(segment: dict[str, Any], *, default_end: float | None = None) -> dict[str, Any]:
    start = segment.get("start_sec", segment.get("start"))
    end = segment.get("end_sec", segment.get("end", default_end))
    if start is None:
        raise ValueError("Reviewed transcript segments must include start or start_sec.")
    text = normalize_text(str(segment.get("text") or ""))
    if not text:
        raise ValueError("Reviewed transcript segments must include non-empty text.")
    if end is None:
        raise ValueError("Reviewed transcript segments must include end/end_sec or be inferable from neighbors.")
    payload = {
        "speaker": str(segment.get("speaker") or "SPEAKER_UNKNOWN"),
        "start": float(start),
        "end": float(end),
        "text": text,
    }
    return payload


def load_segment_payload(path: str | Path) -> list[dict[str, Any]]:
    """Load either a raw segment list or a transcript artifact with a top-level `segments` key."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("segments", payload.get("chunks", []))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a segment list in {path}.")
    return [dict(item) for item in payload if isinstance(item, dict)]


def load_reviewed_segments(path: str | Path) -> list[dict[str, Any]]:
    """Load the normalized Gemini-reviewed transcript template."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(segments, list):
        raise ValueError(f"Expected a `segments` list in reviewed transcript {path}.")
    top_level_end = None
    if isinstance(payload, dict):
        end_value = payload.get("end_sec") or payload.get("duration_sec") or payload.get("duration")
        if end_value is not None:
            top_level_end = float(end_value)

    sortable: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = segment.get("start_sec", segment.get("start"))
        if start is None:
            raise ValueError("Reviewed transcript segments must include start or start_sec.")
        sortable.append(dict(segment, __start=float(start)))
    sortable.sort(key=lambda segment: float(segment["__start"]))

    gap_candidates = [
        float(sortable[index + 1]["__start"]) - float(segment["__start"])
        for index, segment in enumerate(sortable[:-1])
        if float(sortable[index + 1]["__start"]) > float(segment["__start"])
    ]
    fallback_span = 5.0
    if gap_candidates:
        fallback_span = min(15.0, max(2.0, statistics.median(gap_candidates)))

    normalized_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(sortable):
        next_start = float(sortable[index + 1]["__start"]) if index + 1 < len(sortable) else None
        default_end = next_start
        if default_end is None and top_level_end is not None and top_level_end > float(segment["__start"]):
            default_end = top_level_end
        if default_end is None:
            default_end = float(segment["__start"]) + fallback_span
        payload_segment = dict(segment)
        payload_segment.pop("__start", None)
        normalized_segments.append(_normalize_review_segment(payload_segment, default_end=default_end))
    return normalized_segments


def build_review_chunks(
    *,
    audio_path: str | Path,
    output_dir: str | Path,
    chunk_seconds: float = 600.0,
    duration_limit: float | None = None,
) -> dict[str, Any]:
    """Prepare fixed-duration WAV chunks and blank review templates for Gemini."""

    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be positive, got {chunk_seconds}.")

    bundle = load_audio_bundle(str(audio_path))
    duration = bundle.duration if duration_limit is None else min(bundle.duration, float(duration_limit))
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)

    root = Path(output_dir).expanduser().resolve()
    audio_dir = root / "audio_chunks"
    review_dir = root / "review_templates"
    audio_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    total_samples = int(duration * SAMPLE_RATE)
    for index, start_sample in enumerate(range(0, total_samples, chunk_samples)):
        end_sample = min(total_samples, start_sample + chunk_samples)
        start_sec = round(start_sample / SAMPLE_RATE, 3)
        end_sec = round(end_sample / SAMPLE_RATE, 3)
        chunk_id = f"chunk_{index:03d}"
        wav_path = audio_dir / f"{chunk_id}.wav"
        review_path = review_dir / f"{chunk_id}.json"
        write_wav_mono(wav_path, bundle.audio[start_sample:end_sample])
        review_path.write_text(
            json.dumps(
                {
                    "chunk_id": chunk_id,
                    "audio_path": str(wav_path),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "instructions": (
                        "Upload the WAV file to Gemini, then fill this JSON template's `segments` list "
                        "with the timestamped speaker-attributed transcript."
                    ),
                    "segments": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_entries.append(
            {
                "chunk_id": chunk_id,
                "audio_path": str(wav_path),
                "review_template_path": str(review_path),
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )

    manifest = {
        "audio_path": str(Path(audio_path).expanduser().resolve()),
        "chunk_seconds": float(chunk_seconds),
        "duration_sec": duration,
        "entries": manifest_entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class FixedWindowTrainingPairBuilder:
    """Build fixed 30-second ECLM pairs from aligned transcript artifacts."""

    def __init__(self, *, window_seconds: float = 30.0) -> None:
        self._window_seconds = float(window_seconds)

    def build_pairs(
        self,
        *,
        whisper_segments: list[dict[str, Any]],
        ctc_segments: list[dict[str, Any]],
        clean_segments: list[dict[str, Any]],
    ) -> list[TrainingPair]:
        duration = max(
            max_segment_end(whisper_segments),
            max_segment_end(ctc_segments),
            max_segment_end(clean_segments),
        )
        pairs: list[TrainingPair] = []
        for start_sec, end_sec in iter_fixed_windows(duration, self._window_seconds):
            whisper_text = text_for_window(whisper_segments, start_sec, end_sec)
            ctc_text = text_for_window(ctc_segments, start_sec, end_sec)
            clean_text = text_for_window(clean_segments, start_sec, end_sec)
            if not whisper_text or not ctc_text or not clean_text:
                continue
            pairs.append(
                TrainingPair(
                    ctc=ctc_text,
                    whisper=whisper_text,
                    clean=clean_text,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            )
        return pairs


class PrimarySegmentTrainingPairBuilder:
    """Build ECLM pairs on the same segment boundaries used at runtime."""

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(normalize_text(text)) if len(token) > 1]

    def _best_clean_slice(self, clean_text: str, *, ctc_text: str, whisper_text: str) -> str:
        sentences = [normalize_text(part) for part in _SENTENCE_SPLIT_RE.split(normalize_text(clean_text)) if normalize_text(part)]
        if len(sentences) <= 1:
            return normalize_text(clean_text)

        source_tokens = set(self._tokens(ctc_text)) | set(self._tokens(whisper_text))
        if not source_tokens:
            return normalize_text(clean_text)

        best_text = normalize_text(clean_text)
        best_score = -1.0
        for start_index in range(len(sentences)):
            combined = ""
            for end_index in range(start_index, len(sentences)):
                combined = f"{combined} {sentences[end_index]}".strip()
                candidate_tokens = self._tokens(combined)
                if not candidate_tokens:
                    continue
                overlap = len(set(candidate_tokens) & source_tokens) / len(set(candidate_tokens))
                length_penalty = abs(len(candidate_tokens) - max(len(self._tokens(ctc_text)), len(self._tokens(whisper_text)), 1)) / max(
                    len(candidate_tokens), 1
                )
                score = overlap - (0.25 * length_penalty)
                if score > best_score:
                    best_score = score
                    best_text = combined
        return best_text

    def build_pairs(
        self,
        *,
        whisper_segments: list[dict[str, Any]],
        ctc_segments: list[dict[str, Any]],
        clean_segments: list[dict[str, Any]],
    ) -> list[TrainingPair]:
        pairs: list[TrainingPair] = []
        for index, whisper_segment in enumerate(
            sorted(
                whisper_segments,
                key=lambda segment: (
                    float(segment.get("start", 0.0)),
                    float(segment.get("end", segment.get("start", 0.0))),
                ),
            )
        ):
            start_sec = float(whisper_segment.get("start", 0.0))
            end_sec = float(whisper_segment.get("end", start_sec))
            whisper_text = normalize_text(str(whisper_segment.get("text") or ""))
            ctc_text = text_for_window(ctc_segments, start_sec, end_sec)
            clean_text = text_for_window(clean_segments, start_sec, end_sec)
            if not whisper_text or not ctc_text or not clean_text:
                continue
            clean_text = self._best_clean_slice(clean_text, ctc_text=ctc_text, whisper_text=whisper_text)
            whisper_tokens = self._tokens(whisper_text)
            ctc_tokens = self._tokens(ctc_text)
            clean_tokens = self._tokens(clean_text)
            if not whisper_tokens or not ctc_tokens or not clean_tokens:
                continue
            source_tokens = set(whisper_tokens) | set(ctc_tokens)
            overlap_ratio = len(set(clean_tokens) & source_tokens) / len(set(clean_tokens))
            max_source_tokens = max(len(whisper_tokens), len(ctc_tokens), 1)
            if len(clean_tokens) > max_source_tokens + 6:
                continue
            if overlap_ratio < 0.35:
                continue
            pairs.append(
                TrainingPair(
                    ctc=ctc_text,
                    whisper=whisper_text,
                    clean=clean_text,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    chunk_id=f"segment_{index:04d}",
                )
            )
        return pairs


def write_training_pairs_jsonl(
    pairs: list[TrainingPair],
    output_path: str | Path,
) -> None:
    """Persist ECLM examples in the requested JSONL format."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "ctc": pair.ctc,
                "whisper": pair.whisper,
                "clean": pair.clean,
                "start_sec": pair.start_sec,
                "end_sec": pair.end_sec,
                **({"chunk_id": pair.chunk_id} if pair.chunk_id is not None else {}),
            },
            ensure_ascii=False,
        )
        for pair in pairs
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
