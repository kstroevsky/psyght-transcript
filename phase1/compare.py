"""Compare CLI for running multiple phase1 presets against one audio file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.backends import BackendSpec, list_backend_ids  # noqa: E402
from phase1.compare_runtime.experiment import run_experiment  # noqa: E402
from phase1.compare_runtime.presets import ComparePipelineSpec, ComparePreset, load_presets  # noqa: E402
from phase1.compare_runtime.reference import load_reference_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run A/B transcription experiments for one audio file")
    parser.add_argument("audio", help="Path to audio/video file")
    parser.add_argument("--backend", action="append", default=[], choices=list_backend_ids(), help="Backend id to compare without a preset file")
    parser.add_argument("--preset", action="append", default=[], help="Path to a preset JSON file")
    parser.add_argument(
        "--preset-dir",
        action="append",
        default=[],
        help="Directory containing preset JSON files",
    )
    parser.add_argument("--lang", default=None, choices=["ru", "uk", "en"])
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--duration-limit", type=float, default=None)
    parser.add_argument("--use-live-canary", action="store_true")
    parser.add_argument("--canary-json", default=None)
    parser.add_argument("--merge-provider", choices=["ollama", "rule_based"], default=None)
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where compare experiment folders are written. Defaults to phase1/runs.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional TXT, phase1 JSON, or plain text reference transcript for WER/CER scoring.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum compare workers. Serialized automatically for non-CPU presets.",
    )
    return parser


def _backend_options(args: argparse.Namespace, backend_id: str) -> dict[str, object]:
    options: dict[str, object] = {}
    if backend_id == "therapy_hybrid":
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
    return options


def _simple_presets(args: argparse.Namespace) -> list[ComparePreset]:
    pipeline = ComparePipelineSpec(
        language=args.lang,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        duration_limit=args.duration_limit,
    )
    presets: list[ComparePreset] = []
    for backend_id in args.backend:
        presets.append(
            ComparePreset(
                path=Path(f"<cli:{backend_id}>"),
                preset_id=backend_id,
                description=f"CLI compare preset for backend {backend_id}",
                pipeline=pipeline,
                backend=BackendSpec(id=backend_id, options=_backend_options(args, backend_id)),
            )
        )
    return presets


def _print_summary(summary: dict) -> None:
    print("[COMPARE] Ranking")
    for run in summary.get("runs", []):
        performance = run.get("performance", {})
        metrics = run.get("metrics", {})
        stability = run.get("stability", {})
        quality_bits = [f"proxy={metrics.get('proxy_quality_score', 0.0):.3f}"]
        if "wer" in metrics:
            quality_bits.append(f"wer={metrics['wer']:.3f}")
        if "cer" in metrics:
            quality_bits.append(f"cer={metrics['cer']:.3f}")
        print(
            f"  {run['rank']:>2}. {run['preset_id']} status={run['status']} "
            f"rtf={performance.get('realtime_factor', 0.0):.3f} "
            f"wall={performance.get('wall_clock_sec', 0.0):.2f}s "
            f"fallbacks={stability.get('fallback_count', 0)} "
            + " ".join(quality_bits)
        )


def main() -> None:
    args = build_parser().parse_args()
    presets = load_presets(args.preset, args.preset_dir) if (args.preset or args.preset_dir) else []
    presets.extend(_simple_presets(args))
    if not presets:
        raise ValueError("Provide at least one --preset/--preset-dir or one --backend.")
    reference_text = load_reference_text(args.reference)
    summary = run_experiment(
        audio_path=args.audio,
        presets=presets,
        output_dir=args.output_dir,
        reference_text=reference_text,
        max_workers=args.max_workers,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
