"""Decode one or more CTC logit matrices with pyctcdecode + optional KenLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyctcdecode import build_ctcdecoder
from sentencepiece import SentencePieceProcessor


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.backends.ctc_kenlm import build_ctc_decoder_labels  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode saved CTC logits with pyctcdecode.")
    parser.add_argument("--manifest", required=True, help="JSON file with `paths: []` of .npy logit matrices")
    parser.add_argument("--tokenizer-model", default=None, help="SentencePiece model used by the acoustic model")
    parser.add_argument("--labels-json", default=None, help="Optional JSON array of explicit decoder labels")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--beam-prune-logp", type=float, default=-10.0)
    parser.add_argument("--token-min-logp", type=float, default=-5.0)
    parser.add_argument("--hotword", action="append", default=[])
    parser.add_argument("--hotword-weight", type=float, default=10.0)
    parser.add_argument("--kenlm-model", default=None, help="Optional KenLM binary path")
    parser.add_argument("--unigrams", default=None, help="Optional newline-delimited unigram vocab file")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--unk-score-offset", type=float, default=-10.0)
    parser.add_argument("--no-lm-score-boundary", action="store_true")
    return parser


def _sentencepiece_pieces(model_path: str | Path) -> list[str]:
    tokenizer = SentencePieceProcessor()
    if not tokenizer.load(str(model_path)):
        raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")
    return [tokenizer.id_to_piece(index) for index in range(tokenizer.get_piece_size())]


def _load_labels(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("labels-json must contain a non-empty JSON array.")
    return [str(item) for item in payload]


def _load_logits(logits_paths: list[str]) -> tuple[list[np.ndarray], int]:
    logits_batches: list[np.ndarray] = []
    logits_dim: int | None = None
    for raw_path in logits_paths:
        logits = np.load(Path(raw_path), allow_pickle=False)
        if logits.ndim != 2:
            raise RuntimeError(f"Expected each logits matrix to be 2D, got shape {tuple(logits.shape)} from {raw_path}.")
        current_dim = int(logits.shape[-1])
        if logits_dim is None:
            logits_dim = current_dim
        elif logits_dim != current_dim:
            raise RuntimeError(f"All logits must share one decoder dimension, got {logits_dim} and {current_dim}.")
        logits_batches.append(logits)
    if logits_dim is None:
        raise RuntimeError("Manifest must contain a non-empty `paths` list.")
    return logits_batches, logits_dim


def _load_unigrams(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    unigrams: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token:
            unigrams.append(token)
    return unigrams or None


def main() -> None:
    args = _build_parser().parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    logits_paths = manifest.get("paths")
    if not isinstance(logits_paths, list) or not logits_paths:
        raise RuntimeError("Manifest must contain a non-empty `paths` list.")
    logits_batches, logits_dim = _load_logits(logits_paths)
    if args.tokenizer_model:
        labels = build_ctc_decoder_labels(_sentencepiece_pieces(args.tokenizer_model), logits_dim)
    elif args.labels_json:
        labels = build_ctc_decoder_labels(_load_labels(args.labels_json), logits_dim)
    else:
        raise RuntimeError("Either --tokenizer-model or --labels-json is required.")

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=args.kenlm_model,
        unigrams=_load_unigrams(args.unigrams),
        alpha=args.alpha,
        beta=args.beta,
        unk_score_offset=args.unk_score_offset,
        lm_score_boundary=not args.no_lm_score_boundary,
    )
    texts: list[str] = []
    for logits in logits_batches:
        texts.append(
            decoder.decode(
                logits,
                beam_width=args.beam_width,
                beam_prune_logp=args.beam_prune_logp,
                token_min_logp=args.token_min_logp,
                hotwords=args.hotword or None,
                hotword_weight=args.hotword_weight,
            )
        )

    Path(args.output).write_text(json.dumps({"texts": texts}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
