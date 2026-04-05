"""Helpers for CTC beam search + KenLM assets used by phase1 backends."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from phase1.config.env import BASE_DIR, MODELS_DIR

_DEFAULT_MODEL_ROOT = MODELS_DIR / "ctc_kenlm"
_DEFAULT_HELPER_DIR = BASE_DIR / ".ctc-kenlm-venv"
_SPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%)\]}»])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{«])\s+")
CTC_BLANK_LABEL = ""
SENTENCEPIECE_UNK_PIECE = "<unk>"
CTC_UNK_LABEL = "▁⁇▁"
SENTENCEPIECE_SPACE_PIECE = "▁"


def ctc_kenlm_model_root() -> Path:
    override = os.getenv("PHASE1_CTC_KENLM_MODEL_ROOT")
    return Path(override).expanduser() if override else _DEFAULT_MODEL_ROOT


def ctc_kenlm_helper_python() -> Path:
    override = os.getenv("PHASE1_CTC_KENLM_PYTHON")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_HELPER_DIR / "bin" / "python"


def ctc_kenlm_helper_script() -> Path:
    override = os.getenv("PHASE1_CTC_KENLM_HELPER_SCRIPT")
    if override:
        return Path(override).expanduser()
    return BASE_DIR / "tools" / "ctc_kenlm_decode.py"


def ctc_kenlm_setup_script() -> Path:
    override = os.getenv("PHASE1_CTC_KENLM_SETUP_SCRIPT")
    if override:
        return Path(override).expanduser()
    return BASE_DIR / "tools" / "setup_ctc_kenlm_helper.py"


def language_model_catalog() -> dict[str, dict[str, Any]]:
    root = ctc_kenlm_model_root()
    return {
        "ru": {
            "language": "ru",
            "source": "Russian Wikipedia dump plus optional extra domain corpora trained locally.",
            "binary_path": root / "ru" / "ruwiki_plus_4gram" / "lm.binary",
            "unigrams_path": root / "ru" / "ruwiki_plus_4gram" / "vocab.txt",
            "training_script": BASE_DIR / "tools" / "train_kenlm_ru.py",
        },
        "uk": {
            "language": "uk",
            "source": "Hugging Face: Yehor/kenlm-uk news/lm-4gram-100k.",
            "binary_path": root / "uk" / "yehor_lm_4gram_100k" / "lm.binary",
            "unigrams_path": root / "uk" / "yehor_lm_4gram_100k" / "vocab-100000.txt",
            "download_script": BASE_DIR / "tools" / "fetch_ctc_lm_assets.py",
            "repo_id": "Yehor/kenlm-uk",
            "repo_subdir": "news/lm-4gram-100k",
        },
        "en": {
            "language": "en",
            "source": "NVIDIA NGC: Riva ASR English(en-US) LM, model speechtotext_en_us_lm:deployable_v1.1.",
            "binary_path": root / "en" / "riva_en_us_lm" / "lm.binary",
            "unigrams_path": root / "en" / "riva_en_us_lm" / "flashlight_decoder_vocab.txt",
            "download_script": BASE_DIR / "tools" / "fetch_ctc_lm_assets.py",
            "ngc_model": "nvidia/riva/speechtotext_en_us_lm:deployable_v1.1",
        },
    }


def normalize_sentencepiece_label(piece: str) -> str:
    """Normalize special SentencePiece pieces into deterministic decoder labels."""

    normalized = str(piece)
    if normalized == SENTENCEPIECE_UNK_PIECE:
        return CTC_UNK_LABEL
    if normalized in {"<blank>", "<blk>"}:
        return CTC_BLANK_LABEL
    if normalized == " ":
        return SENTENCEPIECE_SPACE_PIECE
    return normalized


def build_ctc_decoder_labels(pieces: list[str], logits_dim: int) -> list[str]:
    """Build an explicit decoder label list that exactly matches the CTC head dimension."""

    if logits_dim <= 0:
        raise ValueError(f"CTC logits dimension must be positive, got {logits_dim}.")
    labels = [normalize_sentencepiece_label(piece) for piece in pieces]
    if len(labels) + 1 == logits_dim:
        if CTC_BLANK_LABEL in labels:
            raise ValueError("SentencePiece labels already contain a blank token before explicit blank insertion.")
        return [*labels, CTC_BLANK_LABEL]
    if len(labels) == logits_dim:
        blank_count = labels.count(CTC_BLANK_LABEL)
        if blank_count != 1:
            raise ValueError(
                f"Expected exactly one explicit blank token when label count matches logits dim {logits_dim}, got {blank_count}."
            )
        return labels
    raise ValueError(
        f"Cannot align {len(labels)} SentencePiece labels to CTC logits dim {logits_dim}. "
        "Expected either an exact match with one explicit blank token or one missing blank token."
    )


def cleanup_decoded_text(text: str) -> str:
    """Normalize decoder whitespace while preserving punctuation."""

    cleaned = (
        text.replace(CTC_UNK_LABEL, " ")
        .replace("⁇", " ")
        .replace(SENTENCEPIECE_UNK_PIECE, " ")
        .replace(SENTENCEPIECE_SPACE_PIECE, " ")
    )
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return ""
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_AFTER_OPEN_RE.sub(r"\1", cleaned)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _normalize_hotwords(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",")]
        return [item for item in items if item]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise ValueError("backend.options.decoder.hotwords must be a string or list.")


def resolve_ctc_decoder_runtime(
    backend_options: dict[str, Any] | None,
    language: str = "ru",
) -> dict[str, Any]:
    """Resolve beam-search runtime settings without importing helper dependencies."""

    backend_options = backend_options or {}
    raw_decoder = backend_options.get("decoder") or {}
    if not isinstance(raw_decoder, dict):
        raise ValueError("backend.options.decoder must be an object.")

    enabled = bool(raw_decoder.get("enabled", True))
    requested_strategy = str(raw_decoder.get("strategy") or "greedy").lower()
    if not enabled:
        requested_strategy = "greedy"
    if requested_strategy not in {"beam", "greedy"}:
        raise ValueError("backend.options.decoder.strategy must be 'beam' or 'greedy'.")

    lm_raw = raw_decoder.get("lm") or {}
    if not isinstance(lm_raw, dict):
        raise ValueError("backend.options.decoder.lm must be an object.")

    lm_enabled_requested = bool(lm_raw.get("enabled", True))
    lm_language = str(lm_raw.get("language") or language or "ru")
    catalog = language_model_catalog().get(lm_language)
    binary_path = _path_or_none(lm_raw.get("binary_path")) or (
        catalog["binary_path"] if catalog is not None else None
    )
    unigrams_path = _path_or_none(lm_raw.get("unigrams_path")) or (
        catalog["unigrams_path"] if catalog is not None else None
    )

    helper_python = ctc_kenlm_helper_python()
    helper_script = ctc_kenlm_helper_script()
    runtime = {
        "strategy_requested": requested_strategy,
        "strategy": "greedy",
        "beam_width": _int_or_default(raw_decoder.get("beam_width"), 128),
        "beam_prune_logp": _float_or_default(raw_decoder.get("beam_prune_logp"), -10.0),
        "token_min_logp": _float_or_default(raw_decoder.get("token_min_logp"), -5.0),
        "hotwords": _normalize_hotwords(raw_decoder.get("hotwords")),
        "hotword_weight": _float_or_default(raw_decoder.get("hotword_weight"), 10.0),
        "helper_python": str(helper_python),
        "helper_script": str(helper_script),
        "helper_setup_script": str(ctc_kenlm_setup_script()),
        "fallback_reason": None,
        "lm": {
            "enabled_requested": lm_enabled_requested,
            "enabled": False,
            "language": lm_language,
            "binary_path": str(binary_path) if binary_path is not None else None,
            "unigrams_path": str(unigrams_path) if unigrams_path is not None else None,
            "alpha": _float_or_default(lm_raw.get("alpha"), 0.6),
            "beta": _float_or_default(lm_raw.get("beta"), 1.2),
            "unk_score_offset": _float_or_default(lm_raw.get("unk_score_offset"), -10.0),
            "lm_score_boundary": bool(lm_raw.get("lm_score_boundary", True)),
            "source": catalog.get("source") if catalog is not None else None,
            "available": False,
        },
    }

    if requested_strategy != "beam":
        return runtime

    if not helper_python.exists():
        runtime["fallback_reason"] = (
            "CTC KenLM helper Python was not found. "
            f"Expected {helper_python}. Run {ctc_kenlm_setup_script()} to create phase1/.ctc-kenlm-venv."
        )
        return runtime
    if not helper_script.exists():
        runtime["fallback_reason"] = f"CTC KenLM helper script is missing: {helper_script}"
        return runtime

    if not lm_enabled_requested:
        runtime["strategy"] = "beam"
        return runtime

    if binary_path is None or not binary_path.exists():
        runtime["fallback_reason"] = (
            f"KenLM binary was not found for language {lm_language!r}. "
            f"Expected {binary_path}. Prepare it before using beam search."
        )
        return runtime

    runtime["strategy"] = "beam"
    runtime["lm"]["enabled"] = True
    runtime["lm"]["available"] = True
    return runtime


def decode_logits_batch_with_helper(
    logits_batches: list[np.ndarray],
    tokenizer_model_path: str | Path,
    runtime: dict[str, Any],
) -> list[str]:
    """Decode one batch of CTC logits through the isolated KenLM helper."""

    if runtime.get("strategy") != "beam":
        raise RuntimeError("decode_logits_batch_with_helper requires a resolved beam-search runtime.")

    helper_python = Path(str(runtime["helper_python"]))
    helper_script = Path(str(runtime["helper_script"]))
    if not helper_python.exists():
        raise RuntimeError(f"CTC KenLM helper Python does not exist: {helper_python}")
    if not helper_script.exists():
        raise RuntimeError(f"CTC KenLM helper script does not exist: {helper_script}")

    with tempfile.TemporaryDirectory(prefix="phase1_ctc_kenlm_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        manifest_path = tmp_path / "manifest.json"
        output_path = tmp_path / "decoded.json"

        logits_paths: list[str] = []
        for index, logits in enumerate(logits_batches):
            logits_path = tmp_path / f"logits_{index:04d}.npy"
            np.save(logits_path, np.asarray(logits, dtype=np.float32), allow_pickle=False)
            logits_paths.append(str(logits_path))

        manifest_path.write_text(json.dumps({"paths": logits_paths}), encoding="utf-8")
        command = [
            str(helper_python),
            str(helper_script),
            "--manifest",
            str(manifest_path),
            "--tokenizer-model",
            str(tokenizer_model_path),
            "--output",
            str(output_path),
            "--beam-width",
            str(runtime["beam_width"]),
            "--beam-prune-logp",
            str(runtime["beam_prune_logp"]),
            "--token-min-logp",
            str(runtime["token_min_logp"]),
            "--hotword-weight",
            str(runtime["hotword_weight"]),
        ]
        if runtime.get("lm", {}).get("enabled"):
            lm_runtime = runtime["lm"]
            command.extend(
                [
                    "--kenlm-model",
                    str(lm_runtime["binary_path"]),
                    "--alpha",
                    str(lm_runtime["alpha"]),
                    "--beta",
                    str(lm_runtime["beta"]),
                    "--unk-score-offset",
                    str(lm_runtime["unk_score_offset"]),
                ]
            )
            if lm_runtime.get("unigrams_path"):
                command.extend(["--unigrams", str(lm_runtime["unigrams_path"])])
            if not lm_runtime.get("lm_score_boundary", True):
                command.append("--no-lm-score-boundary")
        for hotword in runtime.get("hotwords") or []:
            command.extend(["--hotword", hotword])

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip() or "Unknown CTC KenLM helper failure."
            raise RuntimeError(details)

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        texts = payload.get("texts")
        if not isinstance(texts, list):
            raise RuntimeError("CTC KenLM helper did not return a `texts` list.")
        if len(texts) != len(logits_batches):
            raise RuntimeError(
                f"CTC KenLM helper returned {len(texts)} texts for {len(logits_batches)} logit batches."
            )
        return [cleanup_decoded_text(str(text)) for text in texts]
