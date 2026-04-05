"""Fetch packaged CTC KenLM assets into the repo's stable offline paths."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.backends.ctc_kenlm import language_model_catalog  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch packaged CTC KenLM assets.")
    parser.add_argument("--lang", action="append", choices=["uk", "en"], required=True)
    parser.add_argument(
        "--ngc-model",
        default=language_model_catalog()["en"]["ngc_model"],
        help="NGC model reference for the English Riva LM.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing local assets.")
    return parser


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _fetch_ukrainian(force: bool) -> None:
    catalog = language_model_catalog()["uk"]
    binary_path = Path(catalog["binary_path"])
    vocab_path = Path(catalog["unigrams_path"])
    if binary_path.exists() and vocab_path.exists() and not force:
        print(f"[LM] Ukrainian asset already exists at {binary_path.parent}")
        return

    subdir = str(catalog["repo_subdir"])
    with tempfile.TemporaryDirectory(prefix="phase1_ctc_lm_uk_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        snapshot_download(
            repo_id=str(catalog["repo_id"]),
            allow_patterns=[
                f"{subdir}/lm.binary",
                f"{subdir}/vocab-100000.txt",
            ],
            local_dir=str(tmp_path),
        )
        _copy_file(tmp_path / subdir / "lm.binary", binary_path)
        _copy_file(tmp_path / subdir / "vocab-100000.txt", vocab_path)
    print(f"[LM] Installed Ukrainian KenLM at {binary_path.parent}")


def _fetch_english(ngc_model: str, force: bool) -> None:
    catalog = language_model_catalog()["en"]
    binary_path = Path(catalog["binary_path"])
    vocab_path = Path(catalog["unigrams_path"])
    if binary_path.exists() and vocab_path.exists() and not force:
        print(f"[LM] English asset already exists at {binary_path.parent}")
        return

    ngc = shutil.which("ngc")
    if ngc is None:
        raise RuntimeError(
            "The `ngc` CLI was not found. Install NVIDIA NGC CLI and authenticate before fetching the Riva LM."
        )

    with tempfile.TemporaryDirectory(prefix="phase1_ctc_lm_en_") as tmp_dir:
        subprocess.run(
            [
                ngc,
                "registry",
                "model",
                "download-version",
                ngc_model,
                "--dest",
                tmp_dir,
            ],
            check=True,
        )
        tmp_path = Path(tmp_dir)
        binary_candidates = sorted(tmp_path.rglob("*.binary"))
        vocab_candidates = sorted(tmp_path.rglob("flashlight_decoder_vocab.txt"))
        if not vocab_candidates:
            vocab_candidates = sorted(tmp_path.rglob("*vocab*.txt"))
        if not binary_candidates:
            raise RuntimeError(f"No KenLM binary was found in the downloaded NGC package: {ngc_model}")
        if not vocab_candidates:
            raise RuntimeError(f"No decoder vocabulary file was found in the downloaded NGC package: {ngc_model}")
        _copy_file(binary_candidates[0], binary_path)
        _copy_file(vocab_candidates[0], vocab_path)
    print(f"[LM] Installed English Riva LM at {binary_path.parent}")


def main() -> None:
    args = _build_parser().parse_args()
    selected = sorted(set(args.lang))
    if "uk" in selected:
        _fetch_ukrainian(force=args.force)
    if "en" in selected:
        _fetch_english(args.ngc_model, force=args.force)


if __name__ == "__main__":
    main()
