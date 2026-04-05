"""Fine-tuning utilities for the additive therapy transcription pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.therapy.finetune import (  # noqa: E402
    FixedWindowTrainingPairBuilder,
    PrimarySegmentTrainingPairBuilder,
    build_review_chunks,
    load_reviewed_segments,
    load_segment_payload,
    write_training_pairs_jsonl,
)


def _sort_review_segments(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        segments,
        key=lambda segment: (
            float(segment.get("start", 0.0)),
            float(segment.get("end", segment.get("start", 0.0))),
            str(segment.get("speaker", "")),
        ),
    )


def _load_review_source(review_source: str | Path) -> list[dict[str, object]]:
    root = Path(review_source).expanduser().resolve()
    if root.is_file():
        return _sort_review_segments(load_reviewed_segments(root))
    segments: list[dict[str, object]] = []
    for path in sorted(root.glob("*.json")):
        segments.extend(load_reviewed_segments(path))
    return _sort_review_segments(segments)


def _canonical_review_segment(segment: dict[str, object]) -> dict[str, object]:
    return {
        "speaker": str(segment.get("speaker") or "SPEAKER_UNKNOWN"),
        "start_sec": float(segment["start"]),
        "end_sec": float(segment["end"]),
        "text": str(segment.get("text") or "").strip(),
    }


def _try_load_template_payload(path: str | Path) -> dict[str, object] | None:
    template_path = Path(path).expanduser().resolve()
    if not template_path.exists():
        return None
    try:
        payload = json.loads(template_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _review_template_incomplete(path: str | Path) -> bool:
    payload = _try_load_template_payload(path)
    if payload is None:
        return True
    segments = payload.get("segments")
    return not isinstance(segments, list) or not segments


def select_review_template(review_dir: str | Path, chunk_id: str | None = None) -> Path:
    root = Path(review_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Review directory was not found: {root}")

    if chunk_id is not None:
        filename = chunk_id if str(chunk_id).endswith(".json") else f"{chunk_id}.json"
        template_path = root / filename
        if not template_path.exists():
            raise FileNotFoundError(f"Review template was not found: {template_path}")
        return template_path

    for template_path in sorted(root.glob("*.json")):
        if _review_template_incomplete(template_path):
            return template_path
    raise ValueError(f"No incomplete review templates found in {root}")


def validate_review_template(path: str | Path) -> tuple[dict[str, object] | None, str | None]:
    template_path = Path(path).expanduser().resolve()
    if not template_path.exists():
        return None, f"Review template not found: {template_path}"

    try:
        payload = json.loads(template_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"Review template JSON is invalid: {exc}"

    if not isinstance(payload, dict):
        return None, "Review template must be a JSON object with a `segments` list."
    segments_payload = payload.get("segments")
    if not isinstance(segments_payload, list):
        return None, "Review template must contain a `segments` list."
    if not segments_payload:
        return None, "Review template `segments` list is empty."

    try:
        normalized_segments = _sort_review_segments(load_reviewed_segments(template_path))
    except Exception as exc:
        return None, f"Review template segments are invalid: {exc}"

    canonical_payload = dict(payload)
    canonical_payload["segments"] = [_canonical_review_segment(segment) for segment in normalized_segments]
    return canonical_payload, None


def wait_for_review_template(
    review_dir: str | Path,
    *,
    chunk_id: str | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> Path:
    template_path = select_review_template(review_dir, chunk_id=chunk_id)
    payload = _try_load_template_payload(template_path) or {}
    print_fn(f"[therapy-finetune] chunk_id={payload.get('chunk_id') or template_path.stem}")
    print_fn(f"[therapy-finetune] review_template={template_path}")
    print_fn(f"[therapy-finetune] audio_chunk={payload.get('audio_path') or '<unknown>'}")
    print_fn("[therapy-finetune] Fill the existing JSON template with a non-empty `segments` list.")

    while True:
        input_fn("[therapy-finetune] Press Enter after you upload/edit the file: ")
        canonical_payload, error = validate_review_template(template_path)
        if error is not None:
            print_fn(f"[therapy-finetune] waiting: {error}")
            continue
        template_path.write_text(
            json.dumps(canonical_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print_fn(f"[therapy-finetune] review accepted: {template_path}")
        return template_path


def _load_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _format_mt5_input(row: dict[str, object]) -> str:
    return f"ctc: {str(row.get('ctc') or '').strip()} whisper: {str(row.get('whisper') or '').strip()}".strip()


def _resolve_training_device(requested: str | None = None) -> str:
    if requested:
        return str(requested)
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_built() and mps.is_available():
        return "mps"
    return "cpu"


def _configure_trainable_mt5_layers(model, *, train_last_n_blocks: int) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    train_last_n_blocks = max(1, int(train_last_n_blocks))
    if hasattr(model, "lm_head"):
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True

    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "block"):
        for block in list(encoder.block)[-train_last_n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        final_layer_norm = getattr(encoder, "final_layer_norm", None)
        if final_layer_norm is not None:
            for parameter in final_layer_norm.parameters():
                parameter.requires_grad = True

    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "block"):
        for block in list(decoder.block)[-train_last_n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        final_layer_norm = getattr(decoder, "final_layer_norm", None)
        if final_layer_norm is not None:
            for parameter in final_layer_norm.parameters():
                parameter.requires_grad = True

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
    }


def _train_eclm(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer

    rows = _load_jsonl(args.train_jsonl)
    if not rows:
        raise ValueError("Training JSONL did not contain any rows.")

    device = _resolve_training_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    parameter_stats = _configure_trainable_mt5_layers(model, train_last_n_blocks=args.train_last_n_blocks)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class _TrainingDataset(Dataset):
        def __init__(self, training_rows: list[dict[str, object]]) -> None:
            self._rows = training_rows

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, index: int) -> dict[str, str]:
            row = self._rows[index]
            return {
                "input_text": _format_mt5_input(row),
                "target_text": str(row.get("clean") or "").strip(),
            }

    def collate(batch: list[dict[str, str]]) -> dict[str, object]:
        model_inputs = tokenizer(
            [item["input_text"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
        )
        labels = tokenizer(
            text_target=[item["target_text"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_target_length,
        )
        label_ids = labels["input_ids"]
        label_ids[label_ids == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = label_ids
        return model_inputs

    loader = DataLoader(
        _TrainingDataset(rows),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        collate_fn=collate,
    )
    model.to(torch.device(device))
    model.train()
    optimizer = Adafactor(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    total_epochs = max(1, int(math.ceil(float(args.epochs))))
    global_step = 0
    for epoch_index in range(total_epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            global_step += 1
            if global_step % max(1, int(args.logging_steps)) == 0:
                print(
                    f"[therapy-finetune] epoch={epoch_index + 1}/{total_epochs} "
                    f"step={global_step} loss={float(loss.detach().cpu()):.4f}"
                )

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "device": device,
                "rows": len(rows),
                "epochs": total_epochs,
                "batch_size": int(args.batch_size),
                "train_last_n_blocks": int(args.train_last_n_blocks),
                **parameter_stats,
                "learning_rate": float(args.learning_rate),
                "max_source_length": int(args.max_source_length),
                "max_target_length": int(args.max_target_length),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and train the therapy ECLM path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-review", help="Cut 10-minute WAV chunks and blank Gemini templates.")
    prepare.add_argument("audio")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--chunk-seconds", type=float, default=600.0)
    prepare.add_argument("--duration-limit", type=float, default=None)

    wait = subparsers.add_parser("wait-review", help="Wait until one review template is completed and valid.")
    wait.add_argument("--review-dir", required=True, help="Directory with JSON review templates")
    wait.add_argument("--chunk-id", default=None, help="Optional chunk id like chunk_000")

    build = subparsers.add_parser("build-jsonl", help="Create 30-second ECLM training pairs.")
    build.add_argument("--whisper-artifact", required=True, help="Usually 01a_therapy_whisper_segments.json")
    build.add_argument("--ctc-artifact", required=True, help="Usually 01b_therapy_ctc_segments.json")
    build.add_argument("--review-source", "--review-dir", dest="review_source", required=True, help="Completed Gemini review JSON file or directory")
    build.add_argument("--output", required=True)
    build.add_argument("--window-seconds", type=float, default=30.0)
    build.add_argument("--pair-mode", choices=["fixed_window", "primary_segments"], default="primary_segments")

    train = subparsers.add_parser("train-eclm", help="Fine-tune google/mt5-large on the generated JSONL.")
    train.add_argument("--train-jsonl", required=True)
    train.add_argument("--output-dir", default=str((Path(__file__).resolve().parents[1] / "models" / "therapy_eclm" / "latest").resolve()))
    train.add_argument("--model-id", default="google/mt5-large")
    train.add_argument("--epochs", type=float, default=5.0)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--max-source-length", type=int, default=512)
    train.add_argument("--max-target-length", type=int, default=256)
    train.add_argument("--logging-steps", type=int, default=10)
    train.add_argument("--device", default=None)
    train.add_argument("--train-last-n-blocks", type=int, default=1)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-review":
        manifest = build_review_chunks(
            audio_path=args.audio,
            output_dir=args.output_dir,
            chunk_seconds=args.chunk_seconds,
            duration_limit=args.duration_limit,
        )
        print(f"[therapy-finetune] prepared {len(manifest['entries'])} review chunks")
        return

    if args.command == "wait-review":
        wait_for_review_template(args.review_dir, chunk_id=args.chunk_id)
        return

    if args.command == "build-jsonl":
        whisper_segments = load_segment_payload(args.whisper_artifact)
        ctc_segments = load_segment_payload(args.ctc_artifact)
        reviewed_segments = _load_review_source(args.review_source)
        if args.pair_mode == "fixed_window":
            pair_builder = FixedWindowTrainingPairBuilder(window_seconds=args.window_seconds)
        else:
            pair_builder = PrimarySegmentTrainingPairBuilder()
        pairs = pair_builder.build_pairs(
            whisper_segments=whisper_segments,
            ctc_segments=ctc_segments,
            clean_segments=reviewed_segments,
        )
        write_training_pairs_jsonl(pairs, args.output)
        print(f"[therapy-finetune] wrote {len(pairs)} training pairs to {Path(args.output).expanduser().resolve()}")
        return

    if args.command == "train-eclm":
        _train_eclm(args)
        print(f"[therapy-finetune] saved model to {Path(args.output_dir).expanduser().resolve()}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
