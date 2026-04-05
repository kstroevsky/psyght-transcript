"""Reference loading and normalization for compare-mode quality metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from contracts import transcript_from_dict

_TRANSCRIPT_LINE_RE = re.compile(r"^\[[^\]]+\]\s+[^:]+:\s*(.*)$")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def flatten_transcript_text(payload: Any) -> str:
    """Flatten a phase1 transcript dict or dataclass into plain text."""

    if hasattr(payload, "segments"):
        return " ".join(str(segment.text).strip() for segment in payload.segments if str(segment.text).strip())
    if isinstance(payload, dict) and "segments" in payload:
        transcript = transcript_from_dict(payload)
        return flatten_transcript_text(transcript)
    raise ValueError("Unsupported transcript payload for reference loading.")


def load_reference_text(path: str | None) -> str | None:
    """Load optional reference text from plain text, phase1 txt, or phase1 json."""

    if not path:
        return None
    reference_path = Path(path)
    text = reference_path.read_text(encoding="utf-8")
    if reference_path.suffix.lower() == ".json":
        payload = json.loads(text)
        return flatten_transcript_text(payload)

    lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        match = _TRANSCRIPT_LINE_RE.match(stripped)
        lines.append(match.group(1) if match else stripped)
    return " ".join(lines)


def normalize_text(text: str) -> str:
    """Normalize multilingual transcript text for WER/CER comparisons."""

    lowered = text.casefold()
    without_punct = _PUNCT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", without_punct).strip()
