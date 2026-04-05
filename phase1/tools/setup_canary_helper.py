"""Create the isolated helper venv used for NVIDIA Canary transcription."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _phase1_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _requirements_path() -> Path:
    return _phase1_dir() / "requirements-canary.txt"


def _default_venv_dir() -> Path:
    return _phase1_dir() / ".canary-venv"


def _default_python() -> str:
    for candidate in ("python3.11", "python3.12"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return sys.executable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the phase1 Canary helper environment.")
    parser.add_argument("--python", default=_default_python(), help="Python 3.11/3.12 interpreter for the helper venv")
    parser.add_argument("--venv-dir", default=str(_default_venv_dir()))
    parser.add_argument("--requirements", default=str(_requirements_path()))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    venv_dir = Path(args.venv_dir)
    requirements = Path(args.requirements)
    if not requirements.exists():
        raise FileNotFoundError(f"Missing Canary helper requirements file: {requirements}")

    subprocess.run([args.python, "-m", "venv", str(venv_dir)], check=True)
    helper_pip = venv_dir / "bin" / "pip"
    subprocess.run([str(helper_pip), "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(helper_pip), "install", "-r", str(requirements)], check=True)
    print(f"[CANARY] helper ready at {venv_dir / 'bin' / 'python'}")


if __name__ == "__main__":
    main()
