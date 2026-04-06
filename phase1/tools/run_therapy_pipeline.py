"""Additive CLI for the Russian therapy transcription pipeline."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.backends import BackendSpec  # noqa: E402
from phase1.config.env import BASE_DIR  # noqa: E402
from phase1.runtime.options import RunOptions  # noqa: E402
from phase1.runtime.runner import execute  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the additive Russian therapy transcription pipeline.")
    parser.add_argument("audio", help="Path to the source audio or video file")
    parser.add_argument("--output-dir", default=None, help="Root directory for transcript outputs and artifacts")
    parser.add_argument("--duration-limit", type=float, default=None)
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=2)
    parser.add_argument("--canary-json", default=None, help="Optional precomputed Canary artifact JSON")
    parser.add_argument(
        "--disable-live-canary",
        action="store_true",
        help="Disable the live Canary stage in the dedicated therapy runner.",
    )
    parser.add_argument("--merge-provider", choices=["ollama", "rule_based"], default="ollama")
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--eclm-model", default=None, help="Optional fine-tuned mT5 checkpoint for Step 5B")
    parser.add_argument("--disable-eclm", action="store_true", help="Disable the Step 5B seq2seq correction stage")
    parser.add_argument("--eclm-device", default=None, help="Optional device override for the Step 5B model")
    parser.add_argument("--device", default=None, help="Shared backend device override")
    parser.add_argument("--compute-type", default=None, help="WhisperX compute type override")
    parser.add_argument("--whisper-model", default=None, help="CTranslate2 Whisper model id used by WhisperX")
    parser.add_argument("--ctc-model", default=None, help="Optional GigaAM model override")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.disable_live_canary and not args.canary_json:
        raise ValueError("therapy_hybrid requires Canary input; pass --canary-json or keep live Canary enabled.")

    input_path = Path(args.audio).expanduser().resolve()
    run_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else BASE_DIR / "runs" / f"therapy_{input_path.stem}_{_timestamp()}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    backend_options: dict[str, object] = {
        "merge_provider": {
            "id": args.merge_provider,
            "model": args.ollama_model,
            "base_url": args.ollama_base_url,
        },
        "eclm": {
            "enabled": not args.disable_eclm,
        },
    }
    if not args.disable_live_canary and not args.canary_json:
        backend_options["use_live_canary"] = True
    if args.eclm_model:
        backend_options["eclm_model_path"] = str(Path(args.eclm_model).expanduser().resolve())
    if args.eclm_device:
        backend_options["eclm_device"] = args.eclm_device
    if args.canary_json:
        backend_options["canary_transcript_path"] = str(Path(args.canary_json).expanduser().resolve())
    if args.device:
        backend_options["device"] = args.device
    if args.compute_type:
        backend_options["compute_type"] = args.compute_type
    if args.whisper_model:
        backend_options["whisper_model_name"] = args.whisper_model
    if args.ctc_model:
        backend_options["ctc_model_name"] = args.ctc_model

    result = execute(
        RunOptions(
            audio_path=str(input_path),
            language="ru",
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            save_intermediate_dir=str(artifact_dir),
            output_dir=str(run_root),
            duration_limit=args.duration_limit,
            backend=BackendSpec(id="therapy_hybrid", options=backend_options),
        )
    )
    print(f"[therapy] transcript saved under {run_root}")
    print(f"[therapy] wall_clock_sec={result.wall_clock_sec}")


if __name__ == "__main__":
    main()
