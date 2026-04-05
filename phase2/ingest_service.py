"""Transcript-loading and ingestion helpers used by the public phase2 CLI."""

from __future__ import annotations

import json
from pathlib import Path

from contracts.transcript import Transcript, transcript_from_dict
from phase2.db.writer import insert_meeting


def load_transcript(json_path: str | Path) -> Transcript:
    """Load a phase1 transcript JSON file through the shared contract serializer."""

    transcript_path = Path(json_path)
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    return transcript_from_dict(payload)


def ingest_transcript_file(json_path: str | Path, audio_path: str = "") -> str:
    """Load a transcript JSON file and insert it into the phase2 database schema."""

    transcript = load_transcript(json_path)
    return insert_meeting(transcript, audio_path)
