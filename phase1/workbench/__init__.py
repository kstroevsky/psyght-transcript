"""phase1 experiment workbench: manifest-driven pipeline experiments.

The top experiment layer. It composes the compare runtime (to run pipeline
variants) and the eval layer (to score them) into one declarative flow:

    manifest (variants + corpus/reference)  ->  run_manifest  ->  leaderboard

Public surface:
- :func:`load_manifest`, :class:`ExperimentManifest`
- :func:`run_manifest`
- :func:`build_leaderboard`, :func:`render_leaderboard`

Both the agent-facing CLI (``phase1/tools/run_experiment_manifest.py``) and the
human-facing marimo notebook call this same API.
"""

from __future__ import annotations

from phase1.workbench.leaderboard import build_leaderboard, rank_leaderboard, render_leaderboard
from phase1.workbench.manifest import ExperimentManifest, load_manifest
from phase1.workbench.runner import run_manifest

__all__ = [
    "ExperimentManifest",
    "load_manifest",
    "run_manifest",
    "build_leaderboard",
    "rank_leaderboard",
    "render_leaderboard",
]
