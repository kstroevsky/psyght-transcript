"""Dataset prep and supervised fine-tuning helpers for Qwen3-ASR."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phase1.pipeline.audio import SAMPLE_RATE, load_audio_bundle, write_wav_mono

_WHITESPACE_RE = re.compile(r"\s+")
_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(?:SPEAKER[_ -]?\d+|SPEAKER_[A-Z0-9]+|[A-ZА-ЯЁ]\d*|СПИКЕР\s*\d+)\s*[:：-]\s*",
    re.IGNORECASE,
)
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_LANGUAGE_NAME_BY_CODE = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}


@dataclass(frozen=True)
class TranscriptSegment:
    """One normalized Gemini transcript segment."""

    start_sec: float
    end_sec: float
    text: str
    speaker: str = "SPEAKER_UNKNOWN"


@dataclass(frozen=True)
class QwenTrainingExample:
    """One supervised Qwen3-ASR training example."""

    clip_id: str
    audio_path: str
    text: str
    training_text: str
    start_sec: float
    end_sec: float
    duration_sec: float
    prompt: str = ""
    language: str = "Russian"
    speaker_count: int = 0
    source_segment_count: int = 0

    def to_row(self) -> dict[str, object]:
        return {
            "audio": self.audio_path,
            "text": self.training_text,
            "prompt": self.prompt,
            "transcript_text": self.text,
            "language": self.language,
            "clip_id": self.clip_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration_sec,
            "speaker_count": self.speaker_count,
            "source_segment_count": self.source_segment_count,
        }


@dataclass(frozen=True)
class QwenTrainingRuntime:
    """Resolved hardware and precision settings for Qwen3-ASR SFT."""

    device: str
    device_map: str
    dtype_name: str
    use_bf16: bool
    use_fp16: bool


def trainer_device_kwargs(runtime: QwenTrainingRuntime) -> dict[str, bool]:
    """Return TrainingArguments device flags that match the resolved runtime."""

    if runtime.device == "cpu":
        return {
            "no_cuda": True,
            "use_cpu": True,
            "use_mps_device": False,
        }
    if runtime.device == "mps":
        return {
            "no_cuda": False,
            "use_cpu": False,
            "use_mps_device": True,
        }
    return {
        "no_cuda": False,
        "use_cpu": False,
        "use_mps_device": False,
    }


def write_training_runtime_manifest(
    output_dir: str | Path,
    *,
    model_path: str | Path,
    runtime: QwenTrainingRuntime,
) -> Path:
    """Persist the resolved runtime manifest for a completed Qwen SFT run."""

    manifest_path = Path(output_dir).expanduser().resolve() / "training_runtime.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "model_path": str(Path(model_path).expanduser().resolve()),
                "device": runtime.device,
                "device_map": runtime.device_map,
                "dtype": runtime.dtype_name,
                "use_bf16": runtime.use_bf16,
                "use_fp16": runtime.use_fp16,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def normalize_transcript_text(text: str) -> str:
    """Collapse whitespace and drop obvious speaker prefixes from transcript text."""

    cleaned = _WHITESPACE_RE.sub(" ", str(text or "").replace("\u00a0", " ")).strip()
    cleaned = _SPEAKER_PREFIX_RE.sub("", cleaned)
    return cleaned.strip()


def resolve_language_name(language: str | None = None, *, fallback_code: str | None = None) -> str:
    """Map a short code to the Qwen3-ASR language string expected in training text."""

    raw = str(language or "").strip()
    if raw:
        lowered = raw.lower()
        return _LANGUAGE_NAME_BY_CODE.get(lowered, raw)
    if fallback_code:
        lowered = str(fallback_code).strip().lower()
        if lowered:
            return _LANGUAGE_NAME_BY_CODE.get(lowered, lowered)
    return "Russian"


def format_training_text(text: str, *, language: str) -> str:
    """Render the official Qwen3-ASR SFT label format."""

    normalized = normalize_transcript_text(text)
    if not normalized:
        raise ValueError("Training transcript text must be non-empty.")
    return f"language {resolve_language_name(language)}<asr_text>{normalized}"


def _normalize_segment_record(
    segment: dict[str, Any],
    *,
    default_end: float | None = None,
    time_offset: float = 0.0,
) -> TranscriptSegment:
    start = segment.get("start_sec", segment.get("start"))
    end = segment.get("end_sec", segment.get("end", default_end))
    if start is None:
        raise ValueError("Transcript segments must include start or start_sec.")
    if end is None:
        raise ValueError("Transcript segments must include end/end_sec or be inferable from neighbors.")

    start_sec = float(start) + float(time_offset)
    end_sec = float(end) + float(time_offset)
    if end_sec <= start_sec:
        raise ValueError(f"Transcript segment end must be greater than start, got {start_sec}->{end_sec}.")

    text = normalize_transcript_text(str(segment.get("text") or ""))
    if not text:
        raise ValueError("Transcript segments must include non-empty text.")

    return TranscriptSegment(
        start_sec=start_sec,
        end_sec=end_sec,
        text=text,
        speaker=str(segment.get("speaker") or "SPEAKER_UNKNOWN"),
    )


def _normalize_transcript_payload(path: str | Path) -> list[TranscriptSegment]:
    transcript_path = Path(path).expanduser().resolve()
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(segments, list):
        raise ValueError(f"Expected a `segments` list in transcript {transcript_path}.")

    chunk_start = None
    chunk_end = None
    if isinstance(payload, dict):
        if payload.get("start_sec") is not None:
            chunk_start = float(payload["start_sec"])
        if payload.get("end_sec") is not None:
            chunk_end = float(payload["end_sec"])
        elif payload.get("duration_sec") is not None and chunk_start is not None:
            chunk_end = chunk_start + float(payload["duration_sec"])

    sortable: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        raw_start = segment.get("start_sec", segment.get("start"))
        if raw_start is None:
            raise ValueError("Transcript segments must include start or start_sec.")
        sortable.append(dict(segment, __start=float(raw_start)))
    sortable.sort(key=lambda item: float(item["__start"]))
    if not sortable:
        return []

    time_offset = 0.0
    if chunk_start is not None and chunk_start > 0:
        chunk_duration = None
        if chunk_end is not None and chunk_end > chunk_start:
            chunk_duration = chunk_end - chunk_start
        max_known_end = max(float(item.get("end_sec", item.get("end", item["__start"]))) for item in sortable)
        max_start = max(float(item["__start"]) for item in sortable)
        if chunk_duration is not None and max_known_end <= chunk_duration + 1.0 and max_start <= chunk_duration + 1.0:
            time_offset = chunk_start

    gap_candidates = [
        float(sortable[index + 1]["__start"]) - float(item["__start"])
        for index, item in enumerate(sortable[:-1])
        if float(sortable[index + 1]["__start"]) > float(item["__start"])
    ]
    fallback_span = 5.0
    if gap_candidates:
        midpoint = sorted(gap_candidates)[len(gap_candidates) // 2]
        fallback_span = min(15.0, max(1.0, float(midpoint)))

    normalized: list[TranscriptSegment] = []
    for index, item in enumerate(sortable):
        next_start = float(sortable[index + 1]["__start"]) if index + 1 < len(sortable) else None
        default_end = next_start
        if default_end is None and chunk_end is not None:
            default_end = chunk_end - time_offset
        if default_end is None:
            default_end = float(item["__start"]) + fallback_span
        payload_segment = dict(item)
        payload_segment.pop("__start", None)
        normalized.append(
            _normalize_segment_record(payload_segment, default_end=default_end, time_offset=time_offset)
        )
    return normalized


def load_gemini_transcript(source: str | Path) -> list[TranscriptSegment]:
    """Load one transcript JSON or a directory of review-template JSON files."""

    root = Path(source).expanduser().resolve()
    if root.is_file():
        return _normalize_transcript_payload(root)

    segments: list[TranscriptSegment] = []
    for path in sorted(root.glob("*.json")):
        segments.extend(_normalize_transcript_payload(path))
    return sorted(segments, key=lambda item: (item.start_sec, item.end_sec, item.speaker, item.text))


def normalize_gemini_transcript_payload(source: str | Path) -> dict[str, object]:
    """Convert a supported transcript source into canonical JSON for build-jsonl."""

    root = Path(source).expanduser().resolve()
    source_paths = [root] if root.is_file() else sorted(root.glob("*.json"))
    if not source_paths:
        raise ValueError(f"No transcript JSON files found under {root}.")

    normalized_segments = load_gemini_transcript(root)
    if not normalized_segments:
        raise ValueError(f"No usable transcript segments found in {root}.")

    source_segment_count = 0
    missing_explicit_end_count = 0
    integer_start_count = 0
    list_payload_count = 0
    warnings: list[str] = []

    for path in source_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        segments = payload.get("segments") if isinstance(payload, dict) else payload
        if not isinstance(segments, list):
            raise ValueError(f"Expected a `segments` list in transcript {path}.")
        if isinstance(payload, list):
            list_payload_count += 1
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            source_segment_count += 1
            raw_start = segment.get("start_sec", segment.get("start"))
            if raw_start is not None and abs(float(raw_start) - round(float(raw_start))) < 1e-9:
                integer_start_count += 1
            if segment.get("end_sec") is None and segment.get("end") is None:
                missing_explicit_end_count += 1

    if missing_explicit_end_count:
        warnings.append(
            f"{missing_explicit_end_count} source segments did not include explicit end times; end_sec was inferred."
        )
    if source_segment_count and integer_start_count == source_segment_count:
        warnings.append("All source start timestamps are integer-second only; sub-second timing would improve clip quality.")
    if list_payload_count:
        warnings.append("Top-level transcript list payloads were normalized into canonical {\"segments\": [...]} JSON.")

    segments_payload = [
        {
            "speaker": segment.speaker,
            "start_sec": round(segment.start_sec, 3),
            "end_sec": round(segment.end_sec, 3),
            "text": segment.text,
        }
        for segment in normalized_segments
    ]
    first_start = segments_payload[0]["start_sec"]
    last_end = segments_payload[-1]["end_sec"]
    return {
        "source_path": str(root),
        "source_file_count": len(source_paths),
        "source_segment_count": source_segment_count,
        "start_sec": first_start,
        "end_sec": last_end,
        "warnings": warnings,
        "segments": segments_payload,
    }


def _group_segments(
    segments: list[TranscriptSegment],
    *,
    min_clip_seconds: float,
    max_clip_seconds: float,
    max_gap_seconds: float,
) -> list[list[TranscriptSegment]]:
    groups: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    natural_pause_seconds = min(float(max_gap_seconds), 0.35)

    for segment in segments:
        if not current:
            current = [segment]
            continue

        current_duration = current[-1].end_sec - current[0].start_sec
        gap = segment.start_sec - current[-1].end_sec
        proposed_duration = max(segment.end_sec, current[-1].end_sec) - current[0].start_sec

        if gap > max_gap_seconds or proposed_duration > max_clip_seconds:
            groups.append(current)
            current = [segment]
            continue

        if current_duration >= min_clip_seconds and gap >= natural_pause_seconds:
            groups.append(current)
            current = [segment]
            continue

        current.append(segment)

    if current:
        groups.append(current)
    return groups


def build_training_examples(
    *,
    audio_path: str | Path,
    transcript_source: str | Path,
    output_dir: str | Path,
    language: str = "Russian",
    prompt: str = "",
    min_clip_seconds: float = 6.0,
    max_clip_seconds: float = 24.0,
    max_gap_seconds: float = 0.75,
    min_keep_seconds: float = 1.0,
    eval_ratio: float = 0.1,
    duration_limit: float | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Build WAV clips, train/eval JSONL files, and a launcher for Qwen3-ASR SFT."""

    if min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be positive.")
    if max_clip_seconds < min_clip_seconds:
        raise ValueError("max_clip_seconds must be greater than or equal to min_clip_seconds.")
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be non-negative.")
    if min_keep_seconds <= 0:
        raise ValueError("min_keep_seconds must be positive.")
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be in [0, 1).")

    bundle = load_audio_bundle(str(audio_path))
    max_duration = bundle.duration if duration_limit is None else min(bundle.duration, float(duration_limit))
    transcript_segments = [
        segment for segment in load_gemini_transcript(transcript_source) if segment.start_sec < max_duration
    ]
    clipped_segments = [
        TranscriptSegment(
            start_sec=max(0.0, segment.start_sec),
            end_sec=min(max_duration, segment.end_sec),
            text=segment.text,
            speaker=segment.speaker,
        )
        for segment in transcript_segments
        if min(max_duration, segment.end_sec) > max(0.0, segment.start_sec)
    ]
    if not clipped_segments:
        raise ValueError("The transcript did not contain any usable timestamped segments for the requested audio window.")

    groups = _group_segments(
        clipped_segments,
        min_clip_seconds=float(min_clip_seconds),
        max_clip_seconds=float(max_clip_seconds),
        max_gap_seconds=float(max_gap_seconds),
    )

    root = Path(output_dir).expanduser().resolve()
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    examples: list[QwenTrainingExample] = []
    skipped_short = 0
    for index, group in enumerate(groups):
        start_sec = group[0].start_sec
        end_sec = max(segment.end_sec for segment in group)
        duration_sec = round(end_sec - start_sec, 3)
        if duration_sec < float(min_keep_seconds):
            skipped_short += 1
            continue

        clip_id = f"utt_{index:06d}"
        clip_path = clips_dir / f"{clip_id}.wav"
        start_sample = max(0, int(round(start_sec * SAMPLE_RATE)))
        end_sample = min(len(bundle.audio), int(round(end_sec * SAMPLE_RATE)))
        write_wav_mono(clip_path, bundle.audio[start_sample:end_sample])
        transcript_text = normalize_transcript_text(" ".join(segment.text for segment in group))
        speaker_count = len({segment.speaker for segment in group if segment.speaker})
        examples.append(
            QwenTrainingExample(
                clip_id=clip_id,
                audio_path=str(clip_path),
                text=transcript_text,
                training_text=format_training_text(transcript_text, language=language),
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                duration_sec=duration_sec,
                prompt=str(prompt or ""),
                language=resolve_language_name(language),
                speaker_count=speaker_count,
                source_segment_count=len(group),
            )
        )
    if not examples:
        raise ValueError("No training examples were produced. Lower min_keep_seconds or inspect the transcript timestamps.")

    examples.sort(key=lambda item: (item.start_sec, item.end_sec, item.clip_id))
    eval_count = 0
    if eval_ratio > 0 and len(examples) > 1:
        eval_count = max(1, int(round(len(examples) * eval_ratio)))
        eval_count = min(eval_count, len(examples) - 1)
    train_examples = examples[:-eval_count] if eval_count else examples
    eval_examples = examples[-eval_count:] if eval_count else []

    train_path = root / "train.jsonl"
    eval_path = root / "eval.jsonl"
    write_examples_jsonl(train_examples, train_path)
    if eval_examples:
        write_examples_jsonl(eval_examples, eval_path)
    elif eval_path.exists():
        eval_path.unlink()

    launcher_path = root / "run_qwen3_asr_sft.sh"
    output_model_dir = root / "artifacts" / "qwen3_asr_1_7b_sft"
    launcher_text = render_training_launcher(
        train_file=train_path,
        eval_file=eval_path if eval_examples else None,
        output_dir=output_model_dir,
        model_path=model_path or "Qwen/Qwen3-ASR-1.7B",
    )
    launcher_path.write_text(launcher_text, encoding="utf-8")
    launcher_path.chmod(0o755)

    all_duration = round(sum(example.duration_sec for example in examples), 3)
    train_duration = round(sum(example.duration_sec for example in train_examples), 3)
    eval_duration = round(sum(example.duration_sec for example in eval_examples), 3)
    clip_lengths = [example.duration_sec for example in examples]
    manifest = {
        "audio_path": str(Path(audio_path).expanduser().resolve()),
        "transcript_source": str(Path(transcript_source).expanduser().resolve()),
        "language": resolve_language_name(language),
        "prompt": str(prompt or ""),
        "duration_limit_sec": max_duration,
        "grouping": {
            "min_clip_seconds": float(min_clip_seconds),
            "max_clip_seconds": float(max_clip_seconds),
            "max_gap_seconds": float(max_gap_seconds),
            "min_keep_seconds": float(min_keep_seconds),
            "eval_ratio": float(eval_ratio),
        },
        "counts": {
            "examples": len(examples),
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "source_segments": len(clipped_segments),
            "skipped_short_groups": skipped_short,
        },
        "durations_sec": {
            "examples": all_duration,
            "train_examples": train_duration,
            "eval_examples": eval_duration,
            "min_clip": min(clip_lengths) if clip_lengths else 0.0,
            "max_clip": max(clip_lengths) if clip_lengths else 0.0,
            "avg_clip": round(all_duration / len(examples), 3) if examples else 0.0,
        },
        "artifacts": {
            "clips_dir": str(clips_dir),
            "train_jsonl": str(train_path),
            **({"eval_jsonl": str(eval_path)} if eval_examples else {}),
            "launcher": str(launcher_path),
            "suggested_output_dir": str(output_model_dir),
        },
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_examples_jsonl(examples: list[QwenTrainingExample], output_path: str | Path) -> None:
    """Persist Qwen3-ASR training rows in JSONL format."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(example.to_row(), ensure_ascii=False) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def render_training_launcher(
    *,
    train_file: str | Path,
    eval_file: str | Path | None,
    output_dir: str | Path,
    helper_python: str | Path | None = None,
    model_path: str = "Qwen/Qwen3-ASR-1.7B",
    batch_size: int = 8,
    grad_acc: int = 8,
    learning_rate: float = 2e-5,
    epochs: float = 1.0,
    save_steps: int = 100,
    num_workers: int = 2,
) -> str:
    """Render a shell launcher that can run the local Qwen3-ASR SFT command later."""

    repo_root = Path(__file__).resolve().parents[2]
    helper_python_path = (
        Path(helper_python).expanduser().resolve()
        if helper_python is not None
        else (repo_root / "phase1" / ".qwen-asr-venv" / "bin" / "python").resolve()
    )
    tool_path = (repo_root / "phase1" / "tools" / "qwen_asr_finetune.py").resolve()
    train_path = Path(train_file).expanduser().resolve()
    eval_path = Path(eval_file).expanduser().resolve() if eval_file is not None else None
    model_out = Path(output_dir).expanduser().resolve()
    eval_flag = f'  --eval-file "{eval_path}" \\\n' if eval_path is not None else ""
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f'HELPER_PYTHON="${{HELPER_PYTHON:-{helper_python_path}}}"\n'
        'TORCHRUN="${TORCHRUN:-$(dirname "$HELPER_PYTHON")/torchrun}"\n'
        'NPROC_PER_NODE="${NPROC_PER_NODE:-1}"\n'
        f'MODEL_PATH="${{MODEL_PATH:-{model_path}}}"\n'
        f'OUTPUT_DIR="${{OUTPUT_DIR:-{model_out}}}"\n'
        f'BATCH_SIZE="${{BATCH_SIZE:-{int(batch_size)}}}"\n'
        f'GRAD_ACC="${{GRAD_ACC:-{int(grad_acc)}}}"\n'
        f'LEARNING_RATE="${{LEARNING_RATE:-{learning_rate}}}"\n'
        f'EPOCHS="${{EPOCHS:-{float(epochs)}}}"\n'
        f'SAVE_STEPS="${{SAVE_STEPS:-{int(save_steps)}}}"\n'
        f'NUM_WORKERS="${{NUM_WORKERS:-{int(num_workers)}}}"\n\n'
        'if [ ! -x "$HELPER_PYTHON" ]; then\n'
        '  echo "Missing Qwen ASR helper python at $HELPER_PYTHON" >&2\n'
        "  exit 1\n"
        "fi\n\n"
        'if [ "$NPROC_PER_NODE" -gt 1 ]; then\n'
        '  exec "$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" '
        f'"{tool_path}" train-sft \\\n'
        '    --model-path "$MODEL_PATH" \\\n'
        f'    --train-file "{train_path}" \\\n'
        f"{eval_flag}"
        '    --output-dir "$OUTPUT_DIR" \\\n'
        '    --batch-size "$BATCH_SIZE" \\\n'
        '    --grad-acc "$GRAD_ACC" \\\n'
        '    --learning-rate "$LEARNING_RATE" \\\n'
        '    --epochs "$EPOCHS" \\\n'
        '    --save-steps "$SAVE_STEPS" \\\n'
        '    --num-workers "$NUM_WORKERS"\n'
        "fi\n\n"
        f'exec "$HELPER_PYTHON" "{tool_path}" train-sft \\\n'
        '  --model-path "$MODEL_PATH" \\\n'
        f'  --train-file "{train_path}" \\\n'
        f"{eval_flag}"
        '  --output-dir "$OUTPUT_DIR" \\\n'
        '  --batch-size "$BATCH_SIZE" \\\n'
        '  --grad-acc "$GRAD_ACC" \\\n'
        '  --learning-rate "$LEARNING_RATE" \\\n'
        '  --epochs "$EPOCHS" \\\n'
        '  --save-steps "$SAVE_STEPS" \\\n'
        '  --num-workers "$NUM_WORKERS"\n'
    )


def patch_outer_forward(model: Any) -> None:
    """Patch the wrapper module to delegate loss-producing forward calls to `.thinker`."""

    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch Qwen3-ASR training forward pass because `.thinker.forward` is missing."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def resolve_qwen_training_runtime(requested_device: str | None = None) -> QwenTrainingRuntime:
    """Resolve the safest available precision/device combination for SFT."""

    import torch

    requested = str(requested_device or "").strip().lower()
    if requested in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Qwen3-ASR training but is not available.")
        if torch.cuda.get_device_capability(0)[0] >= 8:
            return QwenTrainingRuntime("cuda", "cuda:0", "bfloat16", True, False)
        return QwenTrainingRuntime("cuda", "cuda:0", "float16", False, True)
    if requested == "cpu":
        return QwenTrainingRuntime("cpu", "cpu", "float32", False, False)
    if requested == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_built() or not mps.is_available():
            raise RuntimeError("MPS was requested for Qwen3-ASR training but is not available.")
        return QwenTrainingRuntime("mps", "mps", "float16", False, True)

    if torch.cuda.is_available():
        if torch.cuda.get_device_capability(0)[0] >= 8:
            return QwenTrainingRuntime("cuda", "cuda:0", "bfloat16", True, False)
        return QwenTrainingRuntime("cuda", "cuda:0", "float16", False, True)
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_built() and mps.is_available():
        return QwenTrainingRuntime("mps", "mps", "float16", False, True)
    return QwenTrainingRuntime("cpu", "cpu", "float32", False, False)


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    """Return the numerically latest `checkpoint-*` subdirectory under one output root."""

    root = Path(output_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return None

    best_step: int | None = None
    best_path: Path | None = None
    for child in root.iterdir():
        match = _CHECKPOINT_RE.match(child.name)
        if match is None or not child.is_dir():
            continue
        step = int(match.group(1))
        if best_step is None or step > best_step:
            best_step = step
            best_path = child
    return None if best_path is None else str(best_path)


def copy_required_hf_files_for_qwen_asr(src_dir: str | Path, dst_dir: str | Path) -> None:
    """Copy the tokenizer/processor assets each checkpoint needs for inference."""

    source_root = Path(src_dir).expanduser().resolve()
    target_root = Path(dst_dir).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    required = [
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    for filename in required:
        source = source_root / filename
        if source.exists():
            shutil.copy2(source, target_root / filename)


def run_qwen3_sft(args: Any) -> None:
    """Run the official Qwen3-ASR supervised fine-tuning loop against JSONL data."""

    import librosa
    import torch
    from datasets import load_dataset
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments

    def _load_audio(path: str, sr: int = 16000):
        waveform, _ = librosa.load(path, sr=sr, mono=True)
        return waveform

    def _build_prefix_messages(prompt: str, audio_array: Any) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
        ]

    def _make_preprocess_fn_prefix_only(processor: Any):
        def _preprocess(example: dict[str, Any]) -> dict[str, Any]:
            prompt = str(example.get("prompt") or "")
            prefix_messages = _build_prefix_messages(prompt, None)
            prefix_text = processor.apply_chat_template(
                [prefix_messages],
                add_generation_prompt=True,
                tokenize=False,
            )[0]
            return {
                "prompt": prompt,
                "audio": example["audio"],
                "target": example["text"],
                "prefix_text": prefix_text,
            }

        return _preprocess

    @dataclass
    class _DataCollatorForQwen3ASRFinetuning:
        processor: Any
        sampling_rate: int = 16000

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            audio_paths = [str(feature["audio"]) for feature in features]
            prefix_texts = [str(feature["prefix_text"]) for feature in features]
            targets = [str(feature["target"]) for feature in features]

            eos = self.processor.tokenizer.eos_token or ""
            full_texts = [prefix + target + eos for prefix, target in zip(prefix_texts, targets)]
            audios = [_load_audio(path, sr=self.sampling_rate) for path in audio_paths]

            full_inputs = self.processor(
                text=full_texts,
                audio=audios,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            prefix_inputs = self.processor(
                text=prefix_texts,
                audio=audios,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
            labels = full_inputs["input_ids"].clone()
            for index, prefix_len in enumerate(prefix_lens):
                labels[index, :prefix_len] = -100

            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100

            full_inputs["labels"] = labels
            return full_inputs

    class _CastFloatInputsTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                for key, value in list(inputs.items()):
                    if torch.is_tensor(value) and value.is_floating_point():
                        inputs[key] = value.to(dtype=model_dtype)
            return inputs

    class _MakeEveryCheckpointInferableCallback(TrainerCallback):
        def __init__(self, base_model_path: str) -> None:
            self._base_model_path = base_model_path

        def on_save(self, training_args: TrainingArguments, state, control, **kwargs):
            if training_args.process_index != 0:
                return control
            checkpoint_dir = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            if not checkpoint_dir.exists():
                checkpoint_dir = Path(str(kwargs.get("checkpoint", checkpoint_dir)))
            copy_required_hf_files_for_qwen_asr(self._base_model_path, checkpoint_dir)
            return control

    if not args.train_file:
        raise ValueError("train_file is required and must point to JSONL rows with `audio` and `text` fields.")

    runtime = resolve_qwen_training_runtime(getattr(args, "device", None))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[runtime.dtype_name]
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=runtime.device_map,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    raw_dataset = load_dataset(
        "json",
        data_files={
            "train": str(Path(args.train_file).expanduser().resolve()),
            **(
                {"validation": str(Path(args.eval_file).expanduser().resolve())}
                if str(args.eval_file or "").strip()
                else {}
            ),
        },
    )
    dataset = raw_dataset.map(_make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep_columns = {"prompt", "audio", "target", "prefix_text"}
    for split_name in dataset.keys():
        drop_columns = [column for column in dataset[split_name].column_names if column not in keep_columns]
        if drop_columns:
            dataset[split_name] = dataset[split_name].remove_columns(drop_columns)

    collator = _DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=int(args.sr))
    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_acc),
        learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.epochs),
        logging_steps=int(args.log_steps),
        lr_scheduler_type=str(args.lr_scheduler_type),
        warmup_ratio=float(args.warmup_ratio),
        dataloader_num_workers=int(args.num_workers),
        dataloader_pin_memory=bool(int(args.pin_memory)),
        dataloader_persistent_workers=bool(int(args.persistent_workers)),
        dataloader_prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
        save_strategy=str(args.save_strategy),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=int(args.save_steps),
        do_eval=bool(str(args.eval_file or "").strip()),
        bf16=runtime.use_bf16,
        fp16=runtime.use_fp16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        **trainer_device_kwargs(runtime),
    )

    trainer = _CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[_MakeEveryCheckpointInferableCallback(base_model_path=str(args.model_path))],
    )

    resume_from = str(args.resume_from or "").strip()
    if not resume_from and int(args.resume) == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    write_training_runtime_manifest(
        args.output_dir,
        model_path=args.model_path,
        runtime=runtime,
    )
