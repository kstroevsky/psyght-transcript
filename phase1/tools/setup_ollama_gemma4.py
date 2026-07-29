"""Bootstrap a reusable Dockerized Ollama instance with local Gemma4-27B loaded."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.tools import setup_ollama_qwen as base_cli  # noqa: E402


def build_parser():
    return base_cli.build_parser(
        description="Start Dockerized Ollama and preload local Gemma4-27B.",
        default_container_name="phase1-ollama-gemma4",
        default_model="gemma4-27b",
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        base_cli.run_setup(args)
    except Exception as exc:
        print(f"[OLLAMA] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
