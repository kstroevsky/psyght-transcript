"""Public phase1 CLI entrypoint preserved while the runtime moved into smaller modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    """Allow the phase1 CLI to import the shared `contracts` package when run in-place."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.backends import BackendSpec, list_backend_ids  # noqa: E402
from phase1.runtime.runner import run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without changing any public flags or defaults."""

    parser = argparse.ArgumentParser(description="Phase1 transcription pipeline")
    parser.add_argument("audio", help="Path to audio/video file")
    parser.add_argument("--lang", default=None, choices=["ru", "uk", "en"])
    parser.add_argument("--backend", default="gigaam_ctc", choices=list_backend_ids())
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--content-mode",
        default="multi-speaker",
        choices=["single-speaker", "multi-speaker"],
    )
    parser.add_argument("--save-intermediate-dir", default=None)
    parser.add_argument("--progress-file", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where .txt and .json outputs are written. Defaults to the audio directory.",
    )
    parser.add_argument(
        "--duration-limit",
        type=float,
        default=None,
        help="Limit processing to the first N seconds of the audio.",
    )
    parser.add_argument(
        "--no-condition",
        action="store_true",
        help="Disable condition_on_previous_text in ASR (prevents some repetition issues).",
    )
    parser.add_argument("--disable-parallel-post-asr", action="store_true")
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--device", default=None, help="Backend device override when supported.")
    parser.add_argument("--compute-type", default=None, help="Backend compute type override when supported.")
    parser.add_argument("--use-live-canary", action="store_true", help="Use the live Canary backend inside therapy_hybrid.")
    parser.add_argument("--canary-json", default=None, help="Precomputed Canary artifact used by therapy_hybrid.")
    parser.add_argument("--merge-provider", choices=["ollama", "rule_based"], default=None)
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--eclm-model", default=None, help="Optional fine-tuned mT5 checkpoint used by therapy_hybrid")
    parser.add_argument("--disable-eclm", action="store_true", help="Disable the therapy seq2seq correction stage")
    parser.add_argument("--eclm-device", default=None, help="Optional device override for the therapy seq2seq correction stage")
    return parser


def _backend_spec(args: argparse.Namespace) -> BackendSpec:
    options: dict[str, object] = {}
    if args.device:
        options["device"] = args.device
    if args.compute_type:
        options["compute_type"] = args.compute_type

    if args.backend == "therapy_hybrid":
        if args.use_live_canary:
            options["use_live_canary"] = True
        if args.canary_json:
            options["canary_transcript_path"] = args.canary_json
        if args.merge_provider:
            options["merge_provider"] = {
                "id": args.merge_provider,
                "model": args.ollama_model,
                "base_url": args.ollama_base_url,
            }
        if args.disable_eclm:
            options["eclm"] = {"enabled": False}
        if args.eclm_model:
            options["eclm_model_path"] = args.eclm_model
        if args.eclm_device:
            options["eclm_device"] = args.eclm_device
    return BackendSpec(id=args.backend, options=options)


def main() -> None:
    """Parse CLI args and dispatch into the refactored runtime runner."""

    args = build_parser().parse_args()
    run(
        audio_path=args.audio,
        language=args.lang,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        content_mode=args.content_mode,
        save_intermediate_dir=args.save_intermediate_dir,
        progress_file=args.progress_file,
        parallel_post_asr=not args.disable_parallel_post_asr,
        skip_alignment=args.skip_alignment,
        skip_diarization=args.skip_diarization,
        output_dir=args.output_dir,
        duration_limit=args.duration_limit,
        asr_options={"condition_on_previous_text": False} if args.no_condition else None,
        backend=_backend_spec(args),
    )


if __name__ == "__main__":
    main()
