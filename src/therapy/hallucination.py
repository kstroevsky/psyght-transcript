"""Timestamp-pattern hallucination guard for therapy recordings."""

from __future__ import annotations

from statistics import pvariance
from typing import Any

from phase1.therapy.windows import text_for_window


class TimestampEntropyHallucinationDetector:
    """Detect equal-spaced Whisper repetition loops from aligned word timings."""

    def __init__(
        self,
        *,
        min_consecutive_words: int = 4,
        duration_min_sec: float = 0.4,
        duration_max_sec: float = 1.2,
        variance_threshold: float = 0.05,
    ) -> None:
        self._min_consecutive_words = int(min_consecutive_words)
        self._duration_min_sec = float(duration_min_sec)
        self._duration_max_sec = float(duration_max_sec)
        self._variance_threshold = float(variance_threshold)

    def is_hallucinated(self, segment: dict[str, Any]) -> bool:
        """Return whether one aligned segment matches the repetition-loop heuristic."""

        durations: list[float] = []
        for word in segment.get("words", []):
            if "start" not in word or "end" not in word:
                durations.clear()
                continue
            duration = float(word["end"]) - float(word["start"])
            if self._duration_min_sec <= duration <= self._duration_max_sec:
                durations.append(duration)
            else:
                durations.clear()
            if len(durations) < self._min_consecutive_words:
                continue
            window = durations[-self._min_consecutive_words :]
            if pvariance(window) < self._variance_threshold:
                return True
        return False


class CTCFallbackHallucinationResolver:
    """Replace hallucinated Whisper segments with the aligned CTC anchor text."""

    def __init__(self, detector: TimestampEntropyHallucinationDetector | None = None) -> None:
        self._detector = detector or TimestampEntropyHallucinationDetector()

    def resolve(
        self,
        whisper_segments: list[dict[str, Any]],
        ctc_segments: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply the timestamp-entropy rule and swap in CTC text when it triggers."""

        flagged_ranges: list[dict[str, float | str]] = []
        resolved_segments: list[dict[str, Any]] = []
        for segment in whisper_segments:
            payload = dict(segment)
            start = float(payload.get("start", 0.0))
            end = float(payload.get("end", start))
            if not self._detector.is_hallucinated(payload):
                payload.setdefault("confidence", "high")
                payload.setdefault("source", "merged")
                resolved_segments.append(payload)
                continue

            ctc_text = text_for_window(ctc_segments, start, end)
            if ctc_text:
                payload["text"] = ctc_text
                payload["confidence"] = "low"
                payload["source"] = "ctc_fallback"
                flagged_ranges.append({"start": start, "end": end, "reason": "uniform_word_durations"})
            else:
                payload.setdefault("confidence", "low")
                payload.setdefault("source", "merged")
                flagged_ranges.append({"start": start, "end": end, "reason": "uniform_word_durations_no_ctc_text"})
            resolved_segments.append(payload)

        return resolved_segments, {
            "flagged_segment_count": len(flagged_ranges),
            "flagged_ranges": flagged_ranges,
            "rule": {
                "min_consecutive_words": self._detector._min_consecutive_words,
                "duration_min_sec": self._detector._duration_min_sec,
                "duration_max_sec": self._detector._duration_max_sec,
                "variance_threshold": self._detector._variance_threshold,
            },
        }
