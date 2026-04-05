"""Public phase2 CLI entrypoint preserved while ingestion logic moved into smaller modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    """Allow the phase2 CLI to import the shared `contracts` package when run in-place."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase2.ingest_service import ingest_transcript_file, load_transcript  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the public phase2 CLI parser without changing flags or defaults."""

    parser = argparse.ArgumentParser(description="Ingest transcript JSON into Postgres")
    parser.add_argument("json_file")
    parser.add_argument("--audio-path", default="")
    return parser


def main() -> None:
    """Parse CLI args and ingest the requested transcript file into Postgres."""

    args = build_parser().parse_args()
    meeting_id = ingest_transcript_file(args.json_file, args.audio_path)
    print(f"Meeting ID: {meeting_id}")


if __name__ == "__main__":
    main()
