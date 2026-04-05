"""TXT and JSON serialization helpers for the final phase1 transcript outputs."""

from __future__ import annotations

import json
from pathlib import Path

from contracts.transcript import Transcript, transcript_to_dict


def _fmt_time(seconds: float) -> str:
    """Format timestamps using the established `[HH:MM:SS.ss]` transcript style."""

    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{sec:05.2f}"


def to_txt(transcript: Transcript) -> str:
    """Render the human-readable transcript text file consumed by operators."""

    return "\n".join(
        f"[{_fmt_time(segment.start)} -> {_fmt_time(segment.end)}] {segment.speaker}: {segment.text}"
        for segment in transcript.segments
    )


def to_json(transcript: Transcript) -> dict:
    """Return the stable JSON payload emitted by the phase1 CLI."""

    return transcript_to_dict(transcript)


def save(transcript: Transcript, output_stem: str) -> None:
    """Write the final `.txt` and `.json` outputs beside the source audio or in `--output-dir`."""

    txt_path = Path(f"{output_stem}.txt")
    json_path = Path(f"{output_stem}.json")
    txt_path.write_text(to_txt(transcript), encoding="utf-8")
    json_path.write_text(
        json.dumps(to_json(transcript), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OUT] Saved {txt_path} and {json_path}")
