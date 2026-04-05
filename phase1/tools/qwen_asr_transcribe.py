"""Helper script that runs Qwen3-ASR inference in the isolated helper environment."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()


def _read_wav_duration_sec(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        sample_rate = handle.getframerate()
    return round(float(frames) / float(sample_rate), 3)


def _resolve_torch_runtime(device: str) -> tuple[str, str]:
    import torch

    requested = str(device or "cpu").strip().lower()
    if requested in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for Qwen3-ASR helper but not available.")
        if torch.cuda.get_device_capability(0)[0] >= 8:
            return "cuda:0", "bfloat16"
        return "cuda:0", "float16"
    if requested == "mps":
        return "cpu", "float32"
    return "cpu", "float32"


def _dtype_object(dtype_name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def _word_entries(result: Any) -> list[dict[str, Any]]:
    timestamps = getattr(result, "time_stamps", None)
    if timestamps is None:
        return []
    words: list[dict[str, Any]] = []
    for item in list(timestamps):
        words.append(
            {
                "word": str(getattr(item, "text", "")).strip(),
                "start": round(float(getattr(item, "start_time", 0.0)), 3),
                "end": round(float(getattr(item, "end_time", 0.0)), 3),
            }
        )
    return [word for word in words if word["word"]]


def _group_words_to_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not words:
        return []
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [words[0]]
    for word in words[1:]:
        gap = float(word["start"]) - float(current[-1]["end"])
        duration = float(word["end"]) - float(current[0]["start"])
        token = str(current[-1]["word"])
        if gap > 0.65 or duration > 12.0 or token.endswith((".", "!", "?", "…")):
            segments.append(current)
            current = [word]
            continue
        current.append(word)
    if current:
        segments.append(current)
    return [
        {
            "start": round(float(group[0]["start"]), 3),
            "end": round(float(group[-1]["end"]), 3),
            "text": " ".join(str(item["word"]).strip() for item in group).strip(),
            "words": [dict(item) for item in group],
        }
        for group in segments
        if group
    ]


def _single_segment(text: str, duration_sec: float) -> list[dict[str, Any]]:
    stripped = str(text or "").strip()
    if not stripped:
        return []
    return [{"start": 0.0, "end": duration_sec, "text": stripped, "words": []}]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR in the helper environment.")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--context", default="")
    parser.add_argument("--max-inference-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--use-forced-aligner", action="store_true")
    parser.add_argument("--forced-aligner-path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from qwen_asr import Qwen3ASRModel
    from phase1.qwen_asr.finetune import resolve_language_name

    device_map, dtype_name = _resolve_torch_runtime(args.device)
    dtype = _dtype_object(dtype_name)
    forced_aligner_kwargs: dict[str, object] | None = None
    if args.use_forced_aligner and args.forced_aligner_path:
        forced_aligner_kwargs = {
            "dtype": dtype,
            "device_map": device_map,
        }

    asr = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=device_map,
        forced_aligner=args.forced_aligner_path if args.use_forced_aligner and args.forced_aligner_path else None,
        forced_aligner_kwargs=forced_aligner_kwargs,
        max_inference_batch_size=int(args.max_inference_batch_size),
        max_new_tokens=int(args.max_new_tokens),
    )

    effective_language = resolve_language_name(args.language)
    results = asr.transcribe(
        audio=str(Path(args.input_wav).expanduser().resolve()),
        context=str(args.context or ""),
        language=effective_language,
        return_time_stamps=bool(args.use_forced_aligner and args.forced_aligner_path),
    )
    result = results[0]
    duration_sec = _read_wav_duration_sec(args.input_wav)
    words = _word_entries(result)
    segments = _group_words_to_segments(words) if words else _single_segment(result.text, duration_sec)
    payload = {
        "language": str(getattr(result, "language", effective_language) or effective_language),
        "text": str(getattr(result, "text", "") or "").strip(),
        "segments": segments,
        "meta": {
            "model_path": str(args.model_path),
            "device_map": device_map,
            "dtype": dtype_name,
            "context": str(args.context or ""),
            "forced_aligner_path": str(args.forced_aligner_path or ""),
            "used_forced_aligner": bool(args.use_forced_aligner and args.forced_aligner_path),
            "max_inference_batch_size": int(args.max_inference_batch_size),
            "max_new_tokens": int(args.max_new_tokens),
            "duration_sec": duration_sec,
        },
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
