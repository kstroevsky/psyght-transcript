"""Canary transcript providers for the additive therapy merge stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase1.therapy.windows import normalize_text


class EmptyCanaryProvider:
    """Default provider used when no Canary transcript is available."""

    def run(
        self,
        audio: Any,
        *,
        language: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del audio, language
        return [], {"provider": "none", "available": False}


class BackendCanaryProvider:
    """Run the live Canary backend on the current audio instead of loading an artifact."""

    def __init__(self, backend_options: dict[str, Any] | None = None) -> None:
        self._backend_options = dict(backend_options or {})

    def run(
        self,
        audio: Any,
        *,
        language: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from phase1.backends.canary_backend import CanaryBackend

        segments, detected_language, meta = CanaryBackend().transcribe(
            audio,
            language,
            backend_options=self._backend_options,
            return_meta=True,
        )
        return segments, {
            "provider": "backend",
            "available": True,
            "detected_language": detected_language,
            **meta,
        }


def _chunk_timestamp_bounds(
    timestamp_payloads: list[dict[str, Any]],
    chunk_name: str,
) -> tuple[float | None, float | None]:
    for timestamp_payload in timestamp_payloads:
        if str(timestamp_payload.get("chunk")) != chunk_name:
            continue
        word_entries = ((timestamp_payload.get("timestamp") or {}).get("word")) or []
        timed_words = [
            (float(word["start"]), float(word["end"]))
            for word in word_entries
            if "start" in word and "end" in word
        ]
        if not timed_words:
            return None, None
        starts, ends = zip(*timed_words)
        return min(starts), max(ends)
    return None, None


class ArtifactCanaryProvider:
    """Load a precomputed Canary artifact captured by the local experiments."""

    def __init__(self, artifact_path: str | Path) -> None:
        self._artifact_path = Path(artifact_path).expanduser().resolve()

    def run(
        self,
        audio: Any,
        *,
        language: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del audio, language
        payload = json.loads(self._artifact_path.read_text(encoding="utf-8"))
        meta = dict(payload.get("meta") or {})
        chunk_seconds = float(meta.get("chunk_seconds") or 30.0)
        timestamp_payloads = list(payload.get("timestamps") or [])

        segments: list[dict[str, Any]] = []
        for index, chunk in enumerate(payload.get("chunks") or []):
            chunk_name = str(chunk.get("chunk") or f"chunk_{index:03d}.wav")
            start = round(index * chunk_seconds, 3)
            end = round((index + 1) * chunk_seconds, 3)
            timestamp_start, timestamp_end = _chunk_timestamp_bounds(timestamp_payloads, chunk_name)
            if timestamp_start is not None:
                start = round((index * chunk_seconds) + timestamp_start, 3)
            if timestamp_end is not None:
                end = round((index * chunk_seconds) + timestamp_end, 3)
            text = normalize_text(str(chunk.get("text") or ""))
            if not text:
                continue
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

        return segments, {
            "provider": "artifact",
            "available": True,
            "artifact_path": str(self._artifact_path),
            "chunk_seconds": chunk_seconds,
            "segment_count": len(segments),
            "meta": meta,
        }
