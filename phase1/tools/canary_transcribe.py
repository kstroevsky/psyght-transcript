"""Helper script that runs NVIDIA Canary in an isolated helper environment."""

from __future__ import annotations

import argparse
import inspect
import json
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _json_safe(item) for key, item in vars(value).items()}
    return str(value)


def _record_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {str(key): _json_safe(item) for key, item in vars(value).items()}
    return {"value": _json_safe(value)}


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "pred_text", "prediction", "transcription"):
            if value.get(key):
                return _extract_text(value[key])
    text = getattr(value, "text", None)
    if text:
        return str(text).strip()
    return ""


def _extract_timestamp_map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        timestamp = value.get("timestamp") or value.get("timestamps") or {}
        return _record_dict(timestamp)
    timestamp = getattr(value, "timestamp", None) or getattr(value, "timestamps", None)
    return _record_dict(timestamp) if timestamp is not None else {}


def _normalized_word_entries(timestamp_map: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = timestamp_map.get("word") or []
    words: list[dict[str, Any]] = []
    for entry in raw_entries:
        payload = _record_dict(entry)
        if "start" not in payload or "end" not in payload:
            continue
        token = str(payload.get("word") or payload.get("text") or "").strip()
        if not token:
            continue
        words.append(
            {
                "word": token,
                "start": float(payload["start"]),
                "end": float(payload["end"]),
                **({"score": float(payload["score"])} if payload.get("score") is not None else {}),
            }
        )
    return words


def _segment_words(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [
        dict(word)
        for word in words
        if float(word.get("start", 0.0)) >= start and float(word.get("end", 0.0)) <= end
    ]


def _normalized_segments(text: str, timestamp_map: dict[str, Any]) -> list[dict[str, Any]]:
    words = _normalized_word_entries(timestamp_map)
    raw_segments = timestamp_map.get("segment") or []
    segments: list[dict[str, Any]] = []
    for entry in raw_segments:
        payload = _record_dict(entry)
        if "start" not in payload or "end" not in payload:
            continue
        start = float(payload["start"])
        end = float(payload["end"])
        segment_words = _segment_words(words, start, end)
        segment_text = str(payload.get("text") or "").strip()
        if not segment_text and segment_words:
            segment_text = " ".join(str(word["word"]).strip() for word in segment_words)
        if not segment_text:
            continue
        segments.append({"start": start, "end": end, "text": segment_text, "words": segment_words})
    if segments:
        return segments
    if words:
        return [
            {
                "start": float(words[0]["start"]),
                "end": float(words[-1]["end"]),
                "text": text or " ".join(str(word["word"]).strip() for word in words),
                "words": words,
            }
        ]
    return [{"start": 0.0, "end": 0.0, "text": text, "words": []}] if text else []


def _read_wav_mono(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        payload = handle.readframes(frames)

    if channels != 1:
        raise ValueError(f"Expected mono WAV, got {channels} channels from {path}.")
    if sample_width != 2:
        raise ValueError(f"Expected PCM16 WAV, got sample width {sample_width} from {path}.")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz WAV, got {sample_rate} Hz from {path}.")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def _write_wav_mono(path: str | Path, audio: np.ndarray) -> None:
    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = np.round(payload * 32767.0).astype("<i2")
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())


def _split_wav_chunks(input_wav: str | Path, chunk_dir: str | Path, chunk_seconds: float) -> list[tuple[Path, float]]:
    audio = _read_wav_mono(input_wav)
    if chunk_seconds <= 0:
        path = Path(chunk_dir) / "chunk_000.wav"
        _write_wav_mono(path, audio)
        return [(path, 0.0)]

    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_samples = max(1, int(round(chunk_seconds * SAMPLE_RATE)))
    chunks: list[tuple[Path, float]] = []
    for index, start in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[start : start + chunk_samples]
        chunk_path = chunk_dir / f"chunk_{index:03d}.wav"
        _write_wav_mono(chunk_path, chunk)
        chunks.append((chunk_path, round(start / SAMPLE_RATE, 3)))
    return chunks


def _offset_words(words: list[dict[str, Any]], offset_sec: float) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for word in words:
        payload = dict(word)
        if "start" in payload:
            payload["start"] = round(float(payload["start"]) + offset_sec, 3)
        if "end" in payload:
            payload["end"] = round(float(payload["end"]) + offset_sec, 3)
        shifted.append(payload)
    return shifted


def _offset_segments(segments: list[dict[str, Any]], offset_sec: float) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for segment in segments:
        payload = dict(segment)
        payload["start"] = round(float(payload.get("start", 0.0)) + offset_sec, 3)
        payload["end"] = round(float(payload.get("end", payload["start"])) + offset_sec, 3)
        payload["words"] = _offset_words(list(payload.get("words") or []), offset_sec)
        shifted.append(payload)
    return shifted


def _has_timestamps(timestamp_map: dict[str, Any]) -> bool:
    return bool((timestamp_map.get("word") or []) or (timestamp_map.get("segment") or []))


def _supports_kwargs(callable_obj: Any) -> bool:
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in inspect.signature(callable_obj).parameters.values())


def _build_transcribe_kwargs(model: Any, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    effective_prompt: dict[str, Any] = {}
    parameters = inspect.signature(model.transcribe).parameters
    accepts_kwargs = _supports_kwargs(model.transcribe)
    values = {
        "batch_size": args.batch_size,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "taskname": args.taskname,
        "pnc": args.pnc,
        "timestamps": args.timestamps,
    }
    for key, value in values.items():
        if key in parameters or accepts_kwargs:
            kwargs[key] = value
            effective_prompt[key] = value
    return kwargs, effective_prompt


def _load_model(model_id: str, device: str):
    from nemo.collections.asr.models import ASRModel

    load_started = time.perf_counter()
    kwargs: dict[str, Any] = {}
    if "map_location" in inspect.signature(ASRModel.from_pretrained).parameters:
        kwargs["map_location"] = device
    model = ASRModel.from_pretrained(model_id, **kwargs)
    if device != "cpu" and hasattr(model, "to"):
        model = model.to(device)
    return model, round(time.perf_counter() - load_started, 3)


def _transcribe(model: Any, args: argparse.Namespace, input_wav: str | Path) -> tuple[Any, float, dict[str, Any]]:
    transcribe_started = time.perf_counter()
    kwargs, effective_prompt = _build_transcribe_kwargs(model, args)
    result = model.transcribe([str(input_wav)], **kwargs)
    return (
        result[0] if isinstance(result, list) else result,
        round(time.perf_counter() - transcribe_started, 3),
        effective_prompt,
    )


def _transcribe_chunks(model: Any, args: argparse.Namespace) -> tuple[dict[str, Any], float, dict[str, Any]]:
    chunk_seconds = float(args.chunk_seconds)
    with tempfile.TemporaryDirectory(prefix="phase1_canary_chunks_") as chunk_dir:
        chunk_paths = _split_wav_chunks(args.input_wav, chunk_dir, chunk_seconds)
        chunk_records: list[dict[str, Any]] = []
        timestamp_payloads: list[dict[str, Any]] = []
        segment_records: list[dict[str, Any]] = []
        raw_records: list[dict[str, Any]] = []
        joined_texts: list[str] = []
        inference_sec = 0.0
        effective_prompt: dict[str, Any] = {}
        used_timestamps_any = False

        for chunk_path, offset_sec in chunk_paths:
            hypothesis, elapsed_sec, prompt = _transcribe(model, args, chunk_path)
            inference_sec += elapsed_sec
            if prompt:
                effective_prompt = dict(prompt)

            text = _extract_text(hypothesis)
            timestamp_map = _extract_timestamp_map(hypothesis)
            local_segments = _normalized_segments(text, timestamp_map)
            used_timestamps = _has_timestamps(timestamp_map)
            used_timestamps_any = used_timestamps_any or used_timestamps

            if text:
                joined_texts.append(text)
            chunk_records.append(
                {
                    "chunk": chunk_path.name,
                    "elapsed_sec": round(elapsed_sec, 3),
                    "text": text,
                    "used_timestamps": used_timestamps,
                }
            )
            timestamp_payloads.append({"chunk": chunk_path.name, "timestamp": timestamp_map})
            segment_records.extend(_offset_segments(local_segments, offset_sec))
            raw_records.append({"chunk": chunk_path.name, "raw": _json_safe(hypothesis)})

    return (
        {
            "meta": {
                "model_id": args.model_id,
                "device": args.device,
                "batch_size": args.batch_size,
                "source_lang": args.source_lang,
                "target_lang": args.target_lang,
                "taskname": args.taskname,
                "pnc": args.pnc,
                "timestamps": bool(args.timestamps),
                "chunk_seconds": chunk_seconds,
                "chunk_count": len(chunk_records),
                "used_timestamps_any": used_timestamps_any,
                "effective_prompt": effective_prompt,
            },
            "text": " ".join(piece for piece in joined_texts if piece).strip(),
            "segments": segment_records,
            "chunks": chunk_records,
            "timestamps": timestamp_payloads,
            "raw": raw_records,
        },
        round(inference_sec, 3),
        effective_prompt,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NVIDIA Canary on one WAV file.")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-id", default="nvidia/canary-1b-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-lang", default="ru")
    parser.add_argument("--target-lang", default="ru")
    parser.add_argument("--taskname", default="asr")
    parser.add_argument("--pnc", default="yes")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    model, model_load_sec = _load_model(args.model_id, args.device)
    output, inference_sec, effective_prompt = _transcribe_chunks(model, args)
    output["meta"]["model_load_sec"] = model_load_sec
    output["meta"]["inference_sec"] = inference_sec
    output["meta"]["effective_prompt"] = effective_prompt
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
