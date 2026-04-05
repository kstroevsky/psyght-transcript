"""Qwen3-ASR workbench CLI for bootstrap, dataset prep, training, and pipeline runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.qwen_asr.finetune import (  # noqa: E402
    build_training_examples,
    normalize_gemini_transcript_payload,
    run_qwen3_sft,
)
from phase1.qwen_asr.workflow import (  # noqa: E402
    DEFAULT_QWEN_ASR_MODEL_ID,
    DEFAULT_QWEN_FORCED_ALIGNER_ID,
    download_qwen_assets,
    ensure_qwen_helper,
    format_command,
    qwen_helper_python,
    resolve_qwen_forced_aligner_path,
    resolve_qwen_model_path,
    run_qwen_pipeline,
    run_qwen_training_job,
)


def _print_json(prefix: str, payload: dict[str, object]) -> None:
    print(f"{prefix} {json.dumps(payload, ensure_ascii=False, indent=2)}")


def _require_success(result, *, label: str) -> None:
    if result.returncode != 0:
        details = result.combined_output() or f"{label} failed without output."
        raise RuntimeError(details)
    print(f"[qwen-asr-finetune] {label} command: {format_command(result.command)}")
    if result.combined_output():
        print(result.combined_output())


def _launch_ui(host: str, port: int, share: bool) -> None:
    from phase1.qwen_asr.ui import launch_qwen_asr_ui

    launch_qwen_asr_ui(server_name=host, server_port=port, share=share)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap and run the Qwen3-ASR workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap",
        help="Create the helper env and download the local Qwen model bundle.",
    )
    bootstrap.add_argument("--skip-helper-install", action="store_true")
    bootstrap.add_argument("--skip-model-download", action="store_true")
    bootstrap.add_argument("--skip-forced-aligner-download", action="store_true")
    bootstrap.add_argument("--python", default=None, help="Optional Python 3.11/3.12 interpreter for the helper env")
    bootstrap.add_argument("--venv-dir", default=None)
    bootstrap.add_argument("--requirements", default=None)
    bootstrap.add_argument("--model-id", default=DEFAULT_QWEN_ASR_MODEL_ID)
    bootstrap.add_argument("--model-dir", default=None)
    bootstrap.add_argument("--forced-aligner-id", default=DEFAULT_QWEN_FORCED_ALIGNER_ID)
    bootstrap.add_argument("--forced-aligner-dir", default=None)

    build = subparsers.add_parser(
        "build-jsonl",
        help="Slice audio with Gemini timestamps and emit train/eval JSONL plus a launcher.",
    )
    build.add_argument("audio")
    build.add_argument("--transcript", required=True, help="Gemini JSON transcript file or directory of review templates")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--lang", default="ru", help="Short language code or Qwen language name")
    build.add_argument("--prompt", default="", help="Optional system prompt stored in every training row")
    build.add_argument("--min-clip-seconds", type=float, default=6.0)
    build.add_argument("--max-clip-seconds", type=float, default=24.0)
    build.add_argument("--max-gap-seconds", type=float, default=0.75)
    build.add_argument("--min-keep-seconds", type=float, default=1.0)
    build.add_argument("--eval-ratio", type=float, default=0.1)
    build.add_argument("--duration-limit", type=float, default=None)
    build.add_argument("--model-path", default=None, help="Model path baked into the generated launcher")

    normalize = subparsers.add_parser(
        "normalize-transcript",
        help="Rewrite a supported transcript JSON into canonical `segments` JSON with inferred end_sec.",
    )
    normalize.add_argument("--transcript", required=True, help="Gemini JSON transcript file or directory of review templates")
    normalize.add_argument("--output", required=True, help="Path for the normalized JSON file")

    train = subparsers.add_parser(
        "train",
        help="Run supervised fine-tuning through the dedicated Qwen helper environment.",
    )
    train.add_argument("--model-path", default=None)
    train.add_argument("--train-file", required=True)
    train.add_argument("--eval-file", default="")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--sr", type=int, default=16000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--grad-acc", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--log-steps", type=int, default=10)
    train.add_argument("--lr-scheduler-type", default="linear")
    train.add_argument("--warmup-ratio", type=float, default=0.02)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--pin-memory", type=int, default=1)
    train.add_argument("--persistent-workers", type=int, default=1)
    train.add_argument("--prefetch-factor", type=int, default=2)
    train.add_argument("--save-strategy", default="steps")
    train.add_argument("--save-steps", type=int, default=200)
    train.add_argument("--save-total-limit", type=int, default=5)
    train.add_argument("--resume-from", default="")
    train.add_argument("--resume", type=int, default=0)
    train.add_argument("--device", default=None, help="Optional training device override: cpu | cuda | mps")

    train_internal = subparsers.add_parser(
        "train-sft",
        help="Internal helper-env SFT command. Use `train` from the main environment.",
    )
    train_internal.add_argument("--model-path", default=DEFAULT_QWEN_ASR_MODEL_ID)
    train_internal.add_argument("--train-file", required=True)
    train_internal.add_argument("--eval-file", default="")
    train_internal.add_argument("--output-dir", required=True)
    train_internal.add_argument("--sr", type=int, default=16000)
    train_internal.add_argument("--batch-size", type=int, default=32)
    train_internal.add_argument("--grad-acc", type=int, default=4)
    train_internal.add_argument("--learning-rate", type=float, default=2e-5)
    train_internal.add_argument("--epochs", type=float, default=1.0)
    train_internal.add_argument("--log-steps", type=int, default=10)
    train_internal.add_argument("--lr-scheduler-type", default="linear")
    train_internal.add_argument("--warmup-ratio", type=float, default=0.02)
    train_internal.add_argument("--num-workers", type=int, default=4)
    train_internal.add_argument("--pin-memory", type=int, default=1)
    train_internal.add_argument("--persistent-workers", type=int, default=1)
    train_internal.add_argument("--prefetch-factor", type=int, default=2)
    train_internal.add_argument("--save-strategy", default="steps")
    train_internal.add_argument("--save-steps", type=int, default=200)
    train_internal.add_argument("--save-total-limit", type=int, default=5)
    train_internal.add_argument("--resume-from", default="")
    train_internal.add_argument("--resume", type=int, default=0)
    train_internal.add_argument("--device", default=None)

    run_pipeline = subparsers.add_parser(
        "run-pipeline",
        help="Run the standard phase1 runtime through the Qwen ASR backend.",
    )
    run_pipeline.add_argument("audio")
    run_pipeline.add_argument("--output-dir", required=True)
    run_pipeline.add_argument("--lang", default="ru", choices=["ru", "uk", "en"])
    run_pipeline.add_argument("--model-path", default=None)
    run_pipeline.add_argument("--context", default="")
    run_pipeline.add_argument("--disable-forced-aligner", action="store_true")
    run_pipeline.add_argument("--forced-aligner-path", default=None)
    run_pipeline.add_argument("--min-speakers", type=int, default=None)
    run_pipeline.add_argument("--max-speakers", type=int, default=None)
    run_pipeline.add_argument("--duration-limit", type=float, default=None)
    run_pipeline.add_argument("--skip-diarization", action="store_true")
    run_pipeline.add_argument("--save-intermediate-dir", default=None)

    launch_ui = subparsers.add_parser(
        "launch-ui",
        help="Start a local Gradio UI for manual Qwen3-ASR bootstrap, training, and pipeline runs.",
    )
    launch_ui.add_argument("--host", default="127.0.0.1")
    launch_ui.add_argument("--port", type=int, default=7860)
    launch_ui.add_argument("--share", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "bootstrap":
        if not args.skip_helper_install:
            result = ensure_qwen_helper(
                python_executable=args.python,
                venv_dir=args.venv_dir,
                requirements=args.requirements,
            )
            _require_success(result, label="helper install")
        payload: dict[str, object] = {
            "helper_python": str(Path(args.venv_dir).expanduser().resolve() / "bin" / "python")
            if args.venv_dir
            else str(qwen_helper_python()),
        }
        if not args.skip_model_download:
            assets = download_qwen_assets(
                model_id=args.model_id,
                model_dir=args.model_dir,
                forced_aligner_id=args.forced_aligner_id,
                forced_aligner_dir=args.forced_aligner_dir,
                download_forced_aligner=not args.skip_forced_aligner_download,
            )
            payload.update(assets)
        _print_json("[qwen-asr-finetune] bootstrap", payload)
        return

    if args.command == "build-jsonl":
        manifest = build_training_examples(
            audio_path=args.audio,
            transcript_source=args.transcript,
            output_dir=args.output_dir,
            language=args.lang,
            prompt=args.prompt,
            min_clip_seconds=args.min_clip_seconds,
            max_clip_seconds=args.max_clip_seconds,
            max_gap_seconds=args.max_gap_seconds,
            min_keep_seconds=args.min_keep_seconds,
            eval_ratio=args.eval_ratio,
            duration_limit=args.duration_limit,
            model_path=args.model_path or resolve_qwen_model_path({}),
        )
        print(f"[qwen-asr-finetune] examples={manifest['counts']['examples']}")
        print(f"[qwen-asr-finetune] train_jsonl={manifest['artifacts']['train_jsonl']}")
        if manifest["counts"]["eval_examples"]:
            print(f"[qwen-asr-finetune] eval_jsonl={manifest['artifacts']['eval_jsonl']}")
        print(f"[qwen-asr-finetune] launcher={manifest['artifacts']['launcher']}")
        return

    if args.command == "normalize-transcript":
        normalized = normalize_gemini_transcript_payload(args.transcript)
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[qwen-asr-finetune] normalized_transcript={output_path}")
        print(f"[qwen-asr-finetune] segments={len(normalized['segments'])}")
        if normalized["warnings"]:
            print("[qwen-asr-finetune] warnings:")
            for warning in normalized["warnings"]:
                print(f"  - {warning}")
        return

    if args.command == "train":
        result = run_qwen_training_job(
            train_file=args.train_file,
            eval_file=args.eval_file or None,
            output_dir=args.output_dir,
            model_path=args.model_path or resolve_qwen_model_path({}),
            sr=args.sr,
            batch_size=args.batch_size,
            grad_acc=args.grad_acc,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            log_steps=args.log_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            resume_from=args.resume_from,
            resume=args.resume,
            device=args.device,
        )
        _require_success(result, label="training")
        print(f"[qwen-asr-finetune] output_dir={Path(args.output_dir).expanduser().resolve()}")
        return

    if args.command == "train-sft":
        run_qwen3_sft(args)
        print(f"[qwen-asr-finetune] output_dir={Path(args.output_dir).expanduser().resolve()}")
        return

    if args.command == "run-pipeline":
        result = run_qwen_pipeline(
            audio_path=args.audio,
            output_dir=args.output_dir,
            language=args.lang,
            model_path=args.model_path or resolve_qwen_model_path({}),
            context=args.context,
            forced_aligner_model_path=args.forced_aligner_path or resolve_qwen_forced_aligner_path({}),
            use_forced_aligner=not args.disable_forced_aligner,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            duration_limit=args.duration_limit,
            skip_diarization=args.skip_diarization,
            save_intermediate_dir=args.save_intermediate_dir,
        )
        _print_json("[qwen-asr-finetune] pipeline", result)
        return

    if args.command == "launch-ui":
        _launch_ui(args.host, args.port, args.share)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
