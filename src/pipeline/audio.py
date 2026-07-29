"""Audio-loading and optional preprocessing helpers used by phase1 runs."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
SAMPLE_RATE = 16000
_EPSILON = 1e-8


@dataclass(frozen=True)
class AudioBundle:
    """Loaded audio reused across one or more phase1 runs."""

    audio: Any
    duration: float


def load_audio(path: str) -> tuple[Any, float]:
    """Load audio into WhisperX's 16 kHz array format and return its duration."""

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported format: {audio_path.suffix}")

    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    duration = len(audio) / SAMPLE_RATE
    return audio, duration


def load_audio_bundle(path: str) -> AudioBundle:
    """Load audio and wrap it in a reusable bundle object."""

    audio, duration = load_audio(path)
    return AudioBundle(audio=audio, duration=duration)


def write_audio_cache(path: str | Path, audio: Any) -> None:
    """Persist audio into an on-disk NumPy array for compare-mode reuse."""

    np.save(Path(path), np.asarray(audio, dtype=np.float32), allow_pickle=False)


def read_audio_cache(path: str | Path) -> Any:
    """Read a memory-mapped audio array written by `write_audio_cache`."""

    return np.load(Path(path), mmap_mode="r")


def _audio_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32)) + _EPSILON))


def _ensure_audio_array(audio: Any) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = np.squeeze(array)
    if array.ndim != 1:
        raise ValueError("Phase1 audio preprocessing expects mono 16 kHz audio.")
    return np.array(array, dtype=np.float32, copy=True)


def _deepfilter_python() -> Path:
    override = os.getenv("PHASE1_DEEPFILTER_PYTHON")
    if override:
        return Path(override)
    return Path(sys.executable)


def _deepfilter_site_packages() -> Path | None:
    override = os.getenv("PHASE1_DEEPFILTER_SITE_PACKAGES")
    if override:
        return Path(override)
    candidates = sorted((Path(__file__).resolve().parents[1] / ".deepfilter-venv" / "lib").glob("python*/site-packages"))
    return candidates[0] if candidates else None


def _deepfilter_script() -> Path:
    return Path(__file__).resolve().parents[1] / "tools" / "deepfilter_enhance.py"


def resolve_audio_preprocessing(overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Normalize audio-preprocessing specs declared in backend options."""

    raw = [] if overrides is None else overrides.get("audio_preprocessing") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("backend.options.audio_preprocessing must be a list or string.")

    resolved: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            spec = {"id": item}
        elif isinstance(item, dict):
            spec = dict(item)
        else:
            raise ValueError("audio_preprocessing items must be strings or objects.")

        transform_id = str(spec.get("id", "")).strip().lower()
        if transform_id != "deepfilternet":
            raise ValueError(f"Unsupported audio_preprocessing transform: {transform_id!r}")
        spec["id"] = transform_id
        resolved.append(spec)
    return resolved


def read_wav_mono(path: str | Path) -> np.ndarray:
    """Read a mono PCM16 WAV file into the same float32 format used by phase1."""

    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        payload = handle.readframes(frames)

    if channels != 1:
        raise ValueError(f"Expected mono WAV from {wav_path}, got {channels} channels.")
    if sample_width != 2:
        raise ValueError(f"Expected PCM16 WAV from {wav_path}, got sample width {sample_width}.")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz WAV from {wav_path}, got {sample_rate} Hz.")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def write_wav_mono(path: str | Path, audio: Any) -> None:
    """Persist a mono float32 waveform as a PCM16 WAV for inspection or subprocess use."""

    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(_ensure_audio_array(audio), -1.0, 1.0)
    pcm16 = np.round(array * 32767.0).astype("<i2")
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())


def _run_deepfilternet(audio: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    helper_python = _deepfilter_python()
    helper_script = _deepfilter_script()
    helper_site_packages = _deepfilter_site_packages()
    if not helper_python.exists():
        raise RuntimeError(
            "DeepFilterNet helper environment was not found. "
            f"Expected python at {helper_python}. Set PHASE1_DEEPFILTER_PYTHON to override."
        )
    if not helper_script.exists():
        raise RuntimeError(f"DeepFilterNet helper script is missing: {helper_script}")
    if helper_site_packages is None or not helper_site_packages.exists():
        raise RuntimeError(
            "DeepFilterNet helper site-packages were not found. "
            "Create phase1/.deepfilter-venv and install requirements-deepfilter.txt, "
            "or set PHASE1_DEEPFILTER_SITE_PACKAGES."
        )

    with tempfile.TemporaryDirectory(prefix="phase1_deepfilter_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        write_wav_mono(input_path, audio)

        command = [
            str(helper_python),
            str(helper_script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        if spec.get("post_filter"):
            command.append("--post-filter")
        model_base_dir = spec.get("model_base_dir")
        if model_base_dir:
            command.extend(["--model-base-dir", str(model_base_dir)])

        env = dict(os.environ)
        env["PHASE1_DEEPFILTER_SITE_PACKAGES"] = str(helper_site_packages)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip() or "Unknown DeepFilterNet failure."
            raise RuntimeError(f"DeepFilterNet preprocessing failed: {details}")

        enhanced = read_wav_mono(output_path)

    return enhanced, {
        "input_rms": round(_audio_rms(audio), 6),
        "output_rms": round(_audio_rms(enhanced), 6),
        "post_filter": bool(spec.get("post_filter", False)),
        "helper_python": str(helper_python),
        "helper_site_packages": str(helper_site_packages),
        "helper_script": str(helper_script),
    }


def apply_audio_preprocessing(
    audio: Any,
    overrides: dict[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Apply backend-selected preprocessing transforms and capture per-step metadata."""

    requested = resolve_audio_preprocessing(overrides)
    if not requested:
        return audio, []

    processed = _ensure_audio_array(audio)
    applied: list[dict[str, Any]] = []
    for spec in requested:
        if spec["id"] == "deepfilternet":
            processed, stats = _run_deepfilternet(processed, spec)
        else:  # pragma: no cover - guarded by resolve_audio_preprocessing
            raise ValueError(f"Unsupported audio_preprocessing transform: {spec['id']!r}")
        processed = np.clip(np.asarray(processed, dtype=np.float32), -1.0, 1.0)
        applied.append(
            {
                "id": spec["id"],
                "params": {key: value for key, value in spec.items() if key != "id"},
                "stats": stats,
            }
        )
    return processed, applied
