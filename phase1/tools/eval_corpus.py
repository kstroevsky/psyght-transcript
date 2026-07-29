"""Score transcripts against gold references: one file, or a whole corpus.

This is the model-free measurement entrypoint. It does not run ASR; it scores
hypothesis transcripts you already produced (e.g. under ``phase1/runs/``) against
gold references, with WER/CER/cpWER/DER plus a readable diff.

Single file:
    python -m phase1.tools.eval_corpus --reference gold.json --hypothesis run.json --only-errors

Corpus (batch):
    python -m phase1.tools.eval_corpus --corpus phase1/eval/corpus.example.yaml \\
        --transcripts-dir phase1/runs/<experiment> --report-md report.md
"""

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

from contracts.transcript import transcript_from_dict  # noqa: E402
from phase1.eval import (  # noqa: E402
    load_corpus,
    load_hypotheses_from_dir,
    load_reference,
    render_corpus_report,
    render_score_report,
    render_word_diff,
    score_corpus,
    score_transcript,
)
from phase1.eval.reference import flatten_text  # noqa: E402
from phase1.eval.transcript import transcript_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score transcripts against gold references (no models run).")
    parser.add_argument("--corpus", default=None, help="Corpus manifest (YAML/JSON) for batch scoring.")
    parser.add_argument("--transcripts-dir", default=None, help="Directory of <item_id>.json hypotheses for the corpus.")
    parser.add_argument("--reference", default=None, help="Single-file gold reference (txt / phase1 json / gemini json).")
    parser.add_argument("--hypothesis", default=None, help="Single-file hypothesis transcript (phase1 json).")
    parser.add_argument("--profile", default=None, help="Normalization profile: default | ru_fold | ru_fold_repeats.")
    parser.add_argument("--only-errors", action="store_true", help="Show only error regions in the diff.")
    parser.add_argument("--report-md", default=None, help="Write the markdown report to this path.")
    return parser


def _run_single(args: argparse.Namespace) -> str:
    reference = load_reference(args.reference)
    hypothesis = transcript_from_dict(json.loads(Path(args.hypothesis).read_text(encoding="utf-8")))
    score = score_transcript(reference, hypothesis, profile=args.profile)
    diff = render_word_diff(
        flatten_text(reference),
        transcript_text(hypothesis),
        args.profile,
        only_errors=args.only_errors,
    )
    return render_score_report(score, title=f"{Path(args.hypothesis).name} vs {Path(args.reference).name}", diff=diff)


def _run_corpus(args: argparse.Namespace) -> str:
    corpus = load_corpus(args.corpus)
    hypotheses = load_hypotheses_from_dir(corpus, args.transcripts_dir)
    result = score_corpus(corpus, hypotheses, profile=args.profile)
    return render_corpus_report(result)


def main() -> None:
    args = build_parser().parse_args()
    if args.corpus:
        report = _run_corpus(args)
    elif args.reference and args.hypothesis:
        report = _run_single(args)
    else:
        raise SystemExit("Provide either --corpus (+ --transcripts-dir) or --reference and --hypothesis.")
    print(report)
    if args.report_md:
        Path(args.report_md).write_text(report + "\n", encoding="utf-8")
        print(f"\n[eval] wrote {args.report_md}")


if __name__ == "__main__":
    main()
