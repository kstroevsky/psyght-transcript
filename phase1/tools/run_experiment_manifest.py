"""Run a phase1 experiment manifest: many pipeline variants, one ranked board.

Validate + preview without running anything:
    python -m phase1.tools.run_experiment_manifest --manifest m.yaml --dry-run

Run against the manifest's audio file or eval corpus and write a leaderboard:
    python -m phase1.tools.run_experiment_manifest --manifest m.yaml

This is the agent-facing entrypoint; the marimo workbench drives the same
``phase1.workbench`` API interactively.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.workbench import load_manifest, render_leaderboard, run_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a phase1 experiment manifest and rank its variants.")
    parser.add_argument("--manifest", required=True, help="Path to an experiment manifest (YAML/JSON).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print expanded variant presets without running.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    result = run_manifest(manifest, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[workbench] manifest '{manifest.id}' OK: {len(manifest.presets)} variant(s).")
        print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))
        return
    print(render_leaderboard(result["leaderboard"], title=f"{manifest.id} leaderboard"))
    print(f"\n[workbench] wrote {result['output_dir']}/leaderboard.md")


if __name__ == "__main__":
    main()
