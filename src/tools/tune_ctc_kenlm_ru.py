"""CLI for reference-driven RU CTC + KenLM tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.tuning.ctc_kenlm_ru import (  # noqa: E402
    default_external_lm_paths,
    load_tuning_manifest,
    parse_float_grid,
    parse_int_grid,
    run_tuning_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    external_binary, external_unigrams = default_external_lm_paths()
    parser = argparse.ArgumentParser(description="Tune RU CTC beam search + KenLM on a reference-backed manifest.")
    parser.add_argument("--manifest", required=True, help="Path to a JSON or JSONL manifest with audio/reference cases.")
    parser.add_argument("--output-dir", default=None, help="Directory where tuning artifacts and summary.json are written.")
    parser.add_argument("--beam-widths", default="32,64,128", help="Comma-separated beam widths.")
    parser.add_argument("--alphas", default="0.00,0.05,0.10,0.15,0.20,0.30", help="Comma-separated alpha values.")
    parser.add_argument("--betas", default="0.0,0.25,0.5,0.75,1.0", help="Comma-separated beta values.")
    parser.add_argument("--chunk-duration-sec", type=float, default=20.0, help="Audio chunk size for GigaAM CTC decoding.")
    parser.add_argument("--external-lm-binary", default=external_binary, help="Off-the-shelf RU LM binary for benchmarking.")
    parser.add_argument("--external-lm-unigrams", default=external_unigrams, help="Unigram vocab file for the off-the-shelf RU LM.")
    parser.add_argument("--self-lm-binary", default=None, help="Override the self-trained RU LM binary path.")
    parser.add_argument("--self-lm-unigrams", default=None, help="Override the self-trained RU LM unigram vocab path.")
    parser.add_argument("--model-name", default="ai-sage/GigaAM-v3", help="GigaAM model name.")
    parser.add_argument("--revision", default="e2e_ctc", help="GigaAM model revision.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = run_tuning_manifest(
        manifest_entries=load_tuning_manifest(args.manifest),
        output_dir=args.output_dir,
        beam_widths=parse_int_grid(args.beam_widths),
        alphas=parse_float_grid(args.alphas),
        betas=parse_float_grid(args.betas),
        chunk_duration_sec=args.chunk_duration_sec,
        model_name=args.model_name,
        revision=args.revision,
        external_binary_path=args.external_lm_binary,
        external_unigrams_path=args.external_lm_unigrams,
        self_binary_path=args.self_lm_binary,
        self_unigrams_path=args.self_lm_unigrams,
    )
    print("[TUNE] Ranking")
    for row in summary.get("results", []):
        metrics = row.get("metrics", {})
        performance = row.get("performance", {})
        print(
            f"  {row['rank']:>3}. {row['candidate_id']} status={row['status']} "
            f"wer={metrics.get('wer', 1.0):.3f} cer={metrics.get('cer', 1.0):.3f} "
            f"rtf={performance.get('realtime_factor', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
