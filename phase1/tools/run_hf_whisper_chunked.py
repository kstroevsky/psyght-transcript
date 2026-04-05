"""Chunked Hugging Face Whisper runner for one-off transcription experiments."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.config.env import BASE_DIR  # noqa: E402


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_SPACE_RE = re.compile(r"\s+")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text.replace("\n", " ")).strip()


def extract_last_sentence(text: str) -> str:
    """Return the last sentence-like fragment from one chunk for prompt carryover."""

    normalized = normalize_text(text)
    if not normalized:
        return ""
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    if not parts:
        return ""
    sentence = parts[-1].lstrip("!?.:,;-'\" ").strip()
    return sentence


def build_next_prompt(text: str, max_chars: int) -> str:
    sentence = extract_last_sentence(text)
    if not sentence:
        return ""
    if len(sentence) <= max_chars:
        return sentence
    return sentence[-max_chars:].lstrip()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run_ffmpeg_extract(input_path: Path, output_wav: Path, duration_limit: float | None) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if duration_limit is not None:
        command.extend(["-t", str(duration_limit)])
    command.append(str(output_wav))
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg failed"
        raise RuntimeError(details)


def split_wav_chunks(wav_path: Path, chunk_dir: Path, chunk_seconds: int) -> list[Path]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    chunk_samples = int(sample_rate * chunk_seconds)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, start in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) == 0:
            continue
        path = chunk_dir / f"chunk_{index:03d}.wav"
        sf.write(str(path), chunk, sample_rate)
        paths.append(path)
    return paths


def load_pipeline(model_id: str, device: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

    dtype = torch.float16 if device == "mps" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = WhisperProcessor.from_pretrained(model_id)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    return processor, asr, str(dtype)


def infer_chunk(
    *,
    asr,
    processor,
    chunk_path: Path,
    language: str,
    task: str,
    max_new_tokens: int,
    repetition_penalty: float,
    num_beams: int,
    prompt_text: str,
) -> dict[str, Any]:
    generate_kwargs: dict[str, Any] = {
        "language": language,
        "task": task,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "num_beams": num_beams,
    }
    if prompt_text:
        generate_kwargs["prompt_ids"] = processor.get_prompt_ids(prompt_text, return_tensors="pt")
    output = asr(
        str(chunk_path),
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
    )
    return {
        "text": normalize_text(str(output.get("text") or "")),
        "chunks": json_safe(output.get("chunks") or []),
        "raw": json_safe(output),
        "prompt_text": prompt_text,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Hugging Face Whisper model on manually chunked audio.")
    parser.add_argument("audio", help="Path to source audio/video file")
    parser.add_argument("--model-id", default="antony66/whisper-large-v3-russian")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--duration-limit", type=float, default=600.0)
    parser.add_argument("--chunk-seconds", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    parser.add_argument("--language", default="russian")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-prompt-chars", type=int, default=240)
    parser.add_argument("--disable-carryover-prompt", action="store_true")
    return parser


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.audio).expanduser().resolve()
    run_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else BASE_DIR / "runs" / f"whisper_hf_{input_path.stem}_{timestamp_slug()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    wav_path = run_dir / "input_16k.wav"
    chunk_dir = run_dir / f"chunks_{args.chunk_seconds}s"

    extract_started = time.perf_counter()
    run_ffmpeg_extract(input_path, wav_path, args.duration_limit)
    extract_sec = round(time.perf_counter() - extract_started, 3)
    chunk_paths = split_wav_chunks(wav_path, chunk_dir, args.chunk_seconds)

    device = resolve_device(args.device)
    load_started = time.perf_counter()
    processor, asr, torch_dtype = load_pipeline(args.model_id, device)
    model_load_sec = round(time.perf_counter() - load_started, 3)

    chunk_records: list[dict[str, Any]] = []
    final_lines: list[str] = []
    prompt_text = ""
    inference_started = time.perf_counter()
    for index, chunk_path in enumerate(chunk_paths, start=1):
        chunk_started = time.perf_counter()
        result = infer_chunk(
            asr=asr,
            processor=processor,
            chunk_path=chunk_path,
            language=args.language,
            task=args.task,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            prompt_text="" if args.disable_carryover_prompt else prompt_text,
        )
        elapsed_sec = round(time.perf_counter() - chunk_started, 3)
        final_lines.append(result["text"])
        next_prompt = "" if args.disable_carryover_prompt else build_next_prompt(result["text"], args.max_prompt_chars)
        prompt_text = next_prompt
        chunk_records.append(
            {
                "chunk": chunk_path.name,
                "elapsed_sec": elapsed_sec,
                "text": result["text"],
                "prompt_text": result["prompt_text"],
                "next_prompt_text": next_prompt,
                "segments": result["chunks"],
                "raw": result["raw"],
            }
        )
        print(
            f"[whisper-hf] {index}/{len(chunk_paths)} {chunk_path.name} "
            f"text_len={len(result['text'])} prompt_len={len(result['prompt_text'])}",
            flush=True,
        )

    inference_sec = round(time.perf_counter() - inference_started, 3)
    final_text = "\n".join(line for line in final_lines if line)

    model_slug = args.model_id.replace("/", "_").replace("-", "_")
    txt_path = run_dir / f"{model_slug}.txt"
    json_path = run_dir / f"{model_slug}.json"
    txt_path.write_text(final_text + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "meta": {
                    "model_id": args.model_id,
                    "device": device,
                    "torch_dtype": torch_dtype,
                    "language": args.language,
                    "task": args.task,
                    "duration_limit_sec": args.duration_limit,
                    "chunk_seconds": args.chunk_seconds,
                    "chunk_count": len(chunk_paths),
                    "max_new_tokens": args.max_new_tokens,
                    "repetition_penalty": args.repetition_penalty,
                    "num_beams": args.num_beams,
                    "return_timestamps": True,
                    "carryover_prompt": not args.disable_carryover_prompt,
                    "max_prompt_chars": args.max_prompt_chars,
                    "extract_audio_sec": extract_sec,
                    "model_load_sec": model_load_sec,
                    "inference_sec": inference_sec,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                "text": final_text,
                "chunks": chunk_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[whisper-hf] wrote {txt_path}", flush=True)
    print(f"[whisper-hf] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
