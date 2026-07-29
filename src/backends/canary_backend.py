"""NVIDIA Canary backend implemented through an isolated NeMo helper environment."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from phase1.config.env import BASE_DIR
from phase1.config.runtime import _cuda_is_available
from phase1.pipeline.audio import write_wav_mono


def canary_helper_python() -> Path:
    override = os.getenv("PHASE1_CANARY_PYTHON")
    if override:
        return Path(override).expanduser()
    return BASE_DIR / ".canary-venv" / "bin" / "python"


def canary_helper_script() -> Path:
    override = os.getenv("PHASE1_CANARY_HELPER_SCRIPT")
    if override:
        return Path(override).expanduser()
    return BASE_DIR / "tools" / "canary_transcribe.py"


def canary_setup_script() -> Path:
    override = os.getenv("PHASE1_CANARY_SETUP_SCRIPT")
    if override:
        return Path(override).expanduser()
    return BASE_DIR / "tools" / "setup_canary_helper.py"


def resolve_canary_runtime(backend_options: dict[str, Any] | None = None, language: str | None = None) -> dict[str, Any]:
    """Resolve runtime options for the Canary backend without importing NeMo."""

    backend_options = backend_options or {}
    requested_device = str(backend_options.get("device") or "cpu")
    resolved_device = requested_device
    fallback_reason = None
    if requested_device == "cuda" and not _cuda_is_available():
        resolved_device = "cpu"
        fallback_reason = "CUDA requested for Canary but not available; falling back to CPU."
    elif requested_device == "mps":
        resolved_device = "cpu"
        fallback_reason = "MPS requested for Canary but NeMo Canary runs on CPU/CUDA here; falling back to CPU."

    effective_language = str(language or backend_options.get("language") or "ru")
    chunk_value = None if backend_options is None else backend_options.get("chunk_seconds")
    chunk_seconds = 30.0 if chunk_value is None else float(chunk_value)
    return {
        "requested_device": requested_device,
        "device": resolved_device,
        "fallback_reason": fallback_reason,
        "model_id": str(backend_options.get("model_id") or "nvidia/canary-1b-v2"),
        "batch_size": int(backend_options.get("batch_size") or 1),
        "source_lang": str(backend_options.get("source_lang") or effective_language),
        "target_lang": str(backend_options.get("target_lang") or effective_language),
        "taskname": str(backend_options.get("taskname") or "asr"),
        "pnc": str(backend_options.get("pnc") or "yes"),
        "timestamps": bool(backend_options.get("timestamps", True)),
        "chunk_seconds": chunk_seconds,
        "helper_python": str(canary_helper_python()),
        "helper_script": str(canary_helper_script()),
        "helper_setup_script": str(canary_setup_script()),
    }


class CanaryBackend:
    """Chunkless long-form ASR backend powered by NVIDIA Canary 1B v2."""

    backend_id = "canary"

    def runtime_summary(self, backend_options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "id": self.backend_id,
            "asr": resolve_canary_runtime(backend_options),
            "alignment": {
                "device": None,
                "skip_reason": "Canary returns native timestamps; separate alignment is a no-op.",
            },
        }

    def transcribe(
        self,
        audio: Any,
        language: str | None = None,
        *,
        backend_options: dict[str, Any] | None = None,
        asr_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], str, dict[str, Any]]:
        del asr_options
        runtime = resolve_canary_runtime(backend_options, language)
        helper_python = Path(runtime["helper_python"])
        helper_script = Path(runtime["helper_script"])
        if not helper_python.exists():
            raise RuntimeError(
                "Canary helper Python was not found. "
                f"Expected {helper_python}. Run {runtime['helper_setup_script']} to create phase1/.canary-venv."
            )
        if not helper_script.exists():
            raise RuntimeError(f"Canary helper script is missing: {helper_script}")

        with tempfile.TemporaryDirectory(prefix="phase1_canary_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_wav = tmp_path / "input.wav"
            output_json = tmp_path / "output.json"
            write_wav_mono(input_wav, audio)
            command = [
                str(helper_python),
                str(helper_script),
                "--input-wav",
                str(input_wav),
                "--output-json",
                str(output_json),
                "--model-id",
                str(runtime["model_id"]),
                "--device",
                str(runtime["device"]),
                "--batch-size",
                str(runtime["batch_size"]),
                "--source-lang",
                str(runtime["source_lang"]),
                "--target-lang",
                str(runtime["target_lang"]),
                "--taskname",
                str(runtime["taskname"]),
                "--pnc",
                str(runtime["pnc"]),
                "--chunk-seconds",
                str(runtime["chunk_seconds"]),
            ]
            if runtime["timestamps"]:
                command.append("--timestamps")
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                details = completed.stderr.strip() or completed.stdout.strip() or "Canary helper failed."
                raise RuntimeError(f"Canary transcription failed: {details}")
            payload = json.loads(output_json.read_text(encoding="utf-8"))

        segments = list(payload.get("segments") or [])
        runtime["helper_meta"] = payload.get("meta") or {}
        runtime["canary_effective_prompt"] = dict((payload.get("meta") or {}).get("effective_prompt") or {})
        runtime["canary_chunk_seconds"] = (payload.get("meta") or {}).get("chunk_seconds")
        runtime["artifact_payloads"] = {"01a_canary_segments.json": segments}
        if return_meta:
            return segments, str(runtime["target_lang"]), runtime
        return segments, str(runtime["target_lang"])

    def align(
        self,
        segments: list[dict[str, Any]],
        language: str,
        audio: Any,
        *,
        backend_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        del language, audio, backend_options
        runtime = {
            "device": None,
            "skip_reason": "Canary returns native segment/word timestamps; alignment is a no-op.",
        }
        if return_meta:
            return segments, runtime
        return segments
