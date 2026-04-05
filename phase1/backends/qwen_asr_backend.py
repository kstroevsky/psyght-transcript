"""Qwen3-ASR backend implemented through the isolated helper environment."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from phase1.backends.whisperx_backend import WhisperXBackend
from phase1.config.runtime import _cuda_is_available
from phase1.pipeline.audio import write_wav_mono
from phase1.qwen_asr.workflow import (
    qwen_helper_python,
    qwen_helper_transcribe_script,
    qwen_setup_script,
    resolve_qwen_forced_aligner_path,
    resolve_qwen_model_path,
)


def resolve_qwen_backend_runtime(
    backend_options: dict[str, Any] | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Resolve helper/runtime settings for Qwen3-ASR inference."""

    backend_options = backend_options or {}
    requested_device = str(backend_options.get("device") or "cpu")
    resolved_device = requested_device
    fallback_reason = None
    if requested_device == "cuda" and not _cuda_is_available():
        resolved_device = "cpu"
        fallback_reason = "CUDA requested for Qwen3-ASR but not available; falling back to CPU."
    elif requested_device == "mps":
        resolved_device = "cpu"
        fallback_reason = "MPS requested for Qwen3-ASR but helper runs on CPU/CUDA here; falling back to CPU."

    aligner_path = resolve_qwen_forced_aligner_path(backend_options)
    use_forced_aligner = bool(backend_options.get("use_forced_aligner", aligner_path is not None))
    return {
        "requested_device": requested_device,
        "device": resolved_device,
        "fallback_reason": fallback_reason,
        "model_path": resolve_qwen_model_path(backend_options),
        "forced_aligner_model_path": aligner_path,
        "use_forced_aligner": use_forced_aligner and aligner_path is not None,
        "context": str(backend_options.get("context") or ""),
        "max_inference_batch_size": int(backend_options.get("max_inference_batch_size") or 8),
        "max_new_tokens": int(backend_options.get("max_new_tokens") or 512),
        "helper_python": str(qwen_helper_python()),
        "helper_script": str(qwen_helper_transcribe_script()),
        "helper_setup_script": str(qwen_setup_script()),
        "language": str(language or backend_options.get("language") or "ru"),
    }


class QwenASRBackend:
    """Phase1 backend that delegates transcription to the Qwen helper env."""

    backend_id = "qwen_asr"

    def __init__(self) -> None:
        self._aligner = WhisperXBackend()

    def runtime_summary(self, backend_options: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime = resolve_qwen_backend_runtime(backend_options)
        alignment = {
            "device": None,
            "skip_reason": "Qwen forced aligner already provided timestamps.",
        }
        if not runtime["use_forced_aligner"]:
            alignment = self._aligner.runtime_summary(backend_options).get("alignment")
        return {
            "id": self.backend_id,
            "asr": runtime,
            "alignment": alignment,
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
        runtime = resolve_qwen_backend_runtime(backend_options, language)
        helper_python = Path(runtime["helper_python"])
        helper_script = Path(runtime["helper_script"])
        if not helper_python.exists():
            raise RuntimeError(
                "Qwen3-ASR helper Python was not found. "
                f"Expected {helper_python}. Run {runtime['helper_setup_script']} first."
            )
        if not helper_script.exists():
            raise RuntimeError(f"Qwen3-ASR helper script is missing: {helper_script}")

        with tempfile.TemporaryDirectory(prefix="phase1_qwen_asr_") as tmp_dir:
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
                "--model-path",
                str(runtime["model_path"]),
                "--device",
                str(runtime["device"]),
                "--language",
                str(runtime["language"]),
                "--context",
                str(runtime["context"]),
                "--max-inference-batch-size",
                str(runtime["max_inference_batch_size"]),
                "--max-new-tokens",
                str(runtime["max_new_tokens"]),
            ]
            if runtime["use_forced_aligner"] and runtime["forced_aligner_model_path"]:
                command.extend(
                    [
                        "--use-forced-aligner",
                        "--forced-aligner-path",
                        str(runtime["forced_aligner_model_path"]),
                    ]
                )
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                details = completed.stderr.strip() or completed.stdout.strip() or "Qwen3-ASR helper failed."
                raise RuntimeError(f"Qwen3-ASR transcription failed: {details}")
            payload = json.loads(output_json.read_text(encoding="utf-8"))

        segments = list(payload.get("segments") or [])
        runtime["helper_meta"] = dict(payload.get("meta") or {})
        runtime["artifact_payloads"] = {"01a_qwen_asr_segments.json": segments}
        runtime["detected_language"] = payload.get("language") or runtime["language"]
        if return_meta:
            return segments, str(runtime["detected_language"]), runtime
        return segments, str(runtime["detected_language"])

    def align(
        self,
        segments: list[dict[str, Any]],
        language: str,
        audio: Any,
        *,
        backend_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        runtime = resolve_qwen_backend_runtime(backend_options, language)
        if runtime["use_forced_aligner"]:
            meta = {
                "device": None,
                "skip_reason": "Qwen forced aligner already provided timestamps.",
            }
            if return_meta:
                return segments, meta
            return segments
        return self._aligner.align(
            segments,
            language,
            audio,
            backend_options=backend_options,
            return_meta=return_meta,
        )
