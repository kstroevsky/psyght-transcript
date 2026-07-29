"""Environment loading and path constants for the standalone transcription phase."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only outside the phase1 venv
    def load_dotenv() -> bool:
        """No-op fallback that keeps config importable when dotenv is absent."""

        return False


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def env_str(key: str, default: str) -> str:
    """Read a string env var while keeping config defaults in one place."""

    return os.getenv(key) or default


def env_int(key: str, default: int) -> int:
    """Read an integer env var with a project-defined fallback."""

    raw = os.getenv(key)
    return int(raw) if raw else default
