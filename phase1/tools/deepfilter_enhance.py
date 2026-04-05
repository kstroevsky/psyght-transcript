"""Run DeepFilterNet in an isolated helper environment against one WAV file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _ensure_deepfilter_site_packages() -> None:
    site_packages = os.getenv("PHASE1_DEEPFILTER_SITE_PACKAGES")
    if site_packages and site_packages not in sys.path:
        sys.path.insert(0, site_packages)


_ensure_deepfilter_site_packages()
_ensure_repo_root_on_path()

import numpy as np
import torch
from df.enhance import enhance, init_df

from phase1.pipeline.audio import SAMPLE_RATE, read_wav_mono, write_wav_mono


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    duration = len(audio) / float(source_rate)
    source_times = np.linspace(0.0, duration, num=len(audio), endpoint=False, dtype=np.float64)
    target_length = max(1, int(round(duration * target_rate)))
    target_times = np.linspace(0.0, duration, num=target_length, endpoint=False, dtype=np.float64)
    return np.interp(target_times, source_times, audio).astype(np.float32)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance one WAV file with DeepFilterNet.")
    parser.add_argument("--input", required=True, help="Input mono 16 kHz WAV path")
    parser.add_argument("--output", required=True, help="Output mono 16 kHz WAV path")
    parser.add_argument("--model-base-dir", default=None, help="Optional DeepFilterNet model cache override")
    parser.add_argument("--post-filter", action="store_true", help="Enable DeepFilterNet post-filtering")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    waveform = torch.from_numpy(read_wav_mono(input_path)).unsqueeze(0)
    model, df_state, _ = init_df(model_base_dir=args.model_base_dir, post_filter=args.post_filter)
    df_sample_rate = int(df_state.sr())

    if df_sample_rate != SAMPLE_RATE:
        waveform = torch.from_numpy(_resample(waveform.squeeze(0).numpy(), SAMPLE_RATE, df_sample_rate)).unsqueeze(0)

    enhanced = enhance(model, df_state, waveform).detach().cpu()
    if df_sample_rate != SAMPLE_RATE:
        enhanced = torch.from_numpy(_resample(enhanced.squeeze(0).numpy(), df_sample_rate, SAMPLE_RATE)).unsqueeze(0)

    write_wav_mono(output_path, enhanced.squeeze(0).numpy())


if __name__ == "__main__":
    main()
