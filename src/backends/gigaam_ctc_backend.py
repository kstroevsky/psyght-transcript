"""Russian CTC backend powered by GigaAM-v3 with WhisperX alignment on top."""

from __future__ import annotations

import copy
import gc
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from phase1.backends.ctc_kenlm import decode_logits_batch_with_helper, resolve_ctc_decoder_runtime
from phase1.backends.whisperx_backend import WhisperXBackend
from phase1.config.runtime import _cuda_is_available, _mps_is_available, is_apple_silicon
from phase1.pipeline.audio import SAMPLE_RATE, write_wav_mono


def _extract_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("text", "prediction", "transcription"):
            value = payload.get(key)
            if value:
                return _extract_text(value)
        if "chunks" in payload and isinstance(payload["chunks"], list):
            return " ".join(filter(None, (_extract_text(item) for item in payload["chunks"]))).strip()
    if isinstance(payload, list):
        return " ".join(filter(None, (_extract_text(item) for item in payload))).strip()
    text = getattr(payload, "text", None)
    if text:
        return str(text).strip()
    return str(payload).strip()


class GigaAMCTCBackend:
    """Chunked Russian CTC transcription with WhisperX alignment and diarization."""

    backend_id = "gigaam_ctc"

    def __init__(self) -> None:
        self._aligner = WhisperXBackend()

    def runtime_summary(self, backend_options: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime = self._runtime_config(backend_options, language="ru")
        return {
            "id": self.backend_id,
            "asr": runtime,
            "alignment": self._aligner.runtime_summary(backend_options).get("alignment"),
        }

    def _runtime_config(
        self,
        backend_options: dict[str, Any] | None = None,
        *,
        language: str = "ru",
    ) -> dict[str, Any]:
        backend_options = backend_options or {}
        requested = str(backend_options.get("device") or "cpu")
        resolved = requested
        fallback_reason = None

        if requested == "cuda" and not _cuda_is_available():
            resolved = "cpu"
            fallback_reason = "CUDA requested for GigaAM CTC but not available; falling back to CPU."
        elif requested == "mps":
            if not _mps_is_available():
                resolved = "cpu"
                fallback_reason = "MPS requested for GigaAM CTC but not available; falling back to CPU."
            else:
                resolved = "cpu"
                fallback_reason = "GigaAM CTC currently runs on CPU in phase1 for stability."

        return {
            "requested_device": requested,
            "device": resolved,
            "compute_type": "float32",
            "fallback_reason": fallback_reason,
            "apple_silicon": is_apple_silicon(),
            "cuda_available": _cuda_is_available(),
            "mps_available": _mps_is_available(),
            "model_name": str(backend_options.get("model_name") or "ai-sage/GigaAM-v3"),
            "revision": str(backend_options.get("revision") or "e2e_ctc"),
            "chunk_duration_sec": float(backend_options.get("chunk_duration_sec") or 20.0),
            "decoder": resolve_ctc_decoder_runtime(backend_options, language),
        }

    @lru_cache(maxsize=4)
    def _load_model(self, model_name: str, revision: str):
        from transformers import AutoModel

        print(f"[ASR] Loading GigaAM model={model_name} revision={revision} device=cpu")
        model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )
        return model

    def _chunk_ranges(self, audio: np.ndarray, chunk_duration_sec: float) -> list[tuple[float, float, np.ndarray]]:
        chunk_samples = max(1, int(chunk_duration_sec * SAMPLE_RATE))
        chunks: list[tuple[float, float, np.ndarray]] = []
        for start in range(0, len(audio), chunk_samples):
            end = min(len(audio), start + chunk_samples)
            chunk = audio[start:end]
            if not np.any(np.abs(chunk) > 1e-5):
                continue
            chunks.append((start / SAMPLE_RATE, end / SAMPLE_RATE, chunk))
        return chunks

    def _transcribe_chunk(self, model: Any, chunk_audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            write_wav_mono(tmp_path, chunk_audio)
            result = model.transcribe(str(tmp_path))
            return _extract_text(result)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _tokenizer_model_path(self, model: Any) -> str:
        config_path = model.config.cfg["model"]["cfg"]["decoding"].get("model_path")
        if config_path:
            return str(config_path)
        raise RuntimeError("GigaAM tokenizer model path was not found.")

    def _chunk_logits(self, model: Any, chunk_audio: np.ndarray) -> np.ndarray:
        import torch

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            write_wav_mono(tmp_path, chunk_audio)
            with torch.inference_mode():
                wav, length = model.model.prepare_wav(str(tmp_path))
                encoded, encoded_len = model.model.forward(wav, length)
                logits = model.model.head.decoder_layers(encoded).transpose(1, 2)
            valid_length = int(encoded_len[0].item())
            chunk_logits = logits[0, :valid_length].detach().cpu().float().numpy()
            del wav, length, encoded, encoded_len, logits
            return chunk_logits
        finally:
            tmp_path.unlink(missing_ok=True)

    def transcribe(
        self,
        audio: Any,
        language: str | None = None,
        *,
        backend_options: dict[str, Any] | None = None,
        asr_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], str, dict[str, Any]]:
        if language not in {None, "ru"}:
            fallback = self._aligner.transcribe(
                audio,
                language,
                backend_options=backend_options,
                asr_options=asr_options,
                return_meta=True,
            )
            segments, detected_language, fallback_meta = fallback
            fallback_meta = dict(fallback_meta)
            fallback_meta["fallback_reason"] = (
                f"GigaAM CTC default backend supports only Russian; falling back to whisperx for {language!r}."
            )
            if return_meta:
                return segments, detected_language, fallback_meta
            return segments, detected_language

        del asr_options
        runtime = self._runtime_config(backend_options, language="ru")
        decoder_runtime = runtime.get("decoder") or {}
        if decoder_runtime.get("fallback_reason"):
            print(f"[ASR] WARNING: {decoder_runtime['fallback_reason']}")

        model = self._load_model(runtime["model_name"], runtime["revision"])
        audio_array = np.asarray(audio, dtype=np.float32)
        chunk_ranges = self._chunk_ranges(audio_array, runtime["chunk_duration_sec"])
        segments: list[dict[str, Any]] = []
        if decoder_runtime.get("strategy") == "beam" and chunk_ranges:
            try:
                logits_batches = [self._chunk_logits(model, chunk_audio) for _, _, chunk_audio in chunk_ranges]
                texts = decode_logits_batch_with_helper(
                    logits_batches,
                    self._tokenizer_model_path(model),
                    decoder_runtime,
                )
                for (start_sec, end_sec, _), text in zip(chunk_ranges, texts):
                    if text:
                        segments.append(
                            {
                                "start": round(start_sec, 3),
                                "end": round(end_sec, 3),
                                "text": text,
                            }
                        )
            except Exception as exc:
                runtime["decoder"] = copy.deepcopy(decoder_runtime)
                runtime["decoder"]["strategy"] = "greedy"
                runtime["decoder"]["fallback_reason"] = (
                    f"Beam search with KenLM failed ({exc}); falling back to greedy decoding."
                )
                print(f"[ASR] WARNING: {runtime['decoder']['fallback_reason']}")
                segments.clear()
        if not segments:
            for start_sec, end_sec, chunk_audio in chunk_ranges:
                text = self._transcribe_chunk(model, chunk_audio)
                if text:
                    segments.append(
                        {
                            "start": round(start_sec, 3),
                            "end": round(end_sec, 3),
                            "text": text,
                        }
                    )
        gc.collect()
        if return_meta:
            return segments, "ru", runtime
        return segments, "ru"

    def align(
        self,
        segments: list[dict[str, Any]],
        language: str,
        audio: Any,
        *,
        backend_options: dict[str, Any] | None = None,
        return_meta: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._aligner.align(
            segments,
            language,
            audio,
            backend_options=backend_options,
            return_meta=return_meta,
        )
