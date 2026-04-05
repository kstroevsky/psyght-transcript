"""Utility script that pre-downloads all gated and routed models for offline phase1 runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only outside the phase1 venv
    def load_dotenv() -> bool:
        """No-op fallback that keeps this script importable without dotenv."""

        return False


def _ensure_repo_root_on_path() -> None:
    """Allow this script to reuse the phase package layout when run from `phase1/`."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


load_dotenv()
_ensure_repo_root_on_path()

from phase1.compare_runtime.presets import load_presets  # noqa: E402
from phase1.config import (  # noqa: E402
    HF_TOKEN,
    LANG_MODEL_MAP,
    PYANNOTE_DIARIZATION_MODEL,
    PYANNOTE_SEGMENTATION_MODEL,
    resolve_detect_model,
    resolve_lang_model_map,
)
from phase1.qwen_asr.workflow import (  # noqa: E402
    DEFAULT_QWEN_ASR_MODEL_ID,
    DEFAULT_QWEN_FORCED_ALIGNER_ID,
    download_qwen_assets,
)
from phase1.therapy.pipeline import resolve_therapy_backend_options  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download phase1 models for offline use")
    parser.add_argument("--preset", action="append", default=[], help="Preset JSON file to scan for backend models")
    parser.add_argument(
        "--preset-dir",
        action="append",
        default=[],
        help="Directory of preset JSON files to scan for backend models",
    )
    parser.add_argument(
        "--install-ctc-kenlm-helper",
        action="store_true",
        help="Create phase1/.ctc-kenlm-venv for beam-search decoding with KenLM.",
    )
    parser.add_argument(
        "--install-canary-helper",
        action="store_true",
        help="Create phase1/.canary-venv for the live NVIDIA Canary backend.",
    )
    parser.add_argument(
        "--install-ollama-qwen",
        action="store_true",
        help="Start/reuse a Dockerized Ollama instance and preload qwen3:8b.",
    )
    parser.add_argument(
        "--install-qwen-asr-helper",
        action="store_true",
        help="Create phase1/.qwen-asr-venv for Qwen3-ASR training and inference.",
    )
    parser.add_argument(
        "--download-qwen-asr-model",
        action="store_true",
        help="Download Qwen/Qwen3-ASR-1.7B into phase1/models/qwen_asr.",
    )
    parser.add_argument(
        "--download-qwen-asr-forced-aligner",
        action="store_true",
        help="Also download Qwen/Qwen3-ForcedAligner-0.6B into phase1/models/qwen_asr.",
    )
    parser.add_argument("--qwen-asr-model-id", default=DEFAULT_QWEN_ASR_MODEL_ID)
    parser.add_argument("--qwen-asr-model-dir", default=None)
    parser.add_argument("--qwen-asr-forced-aligner-id", default=DEFAULT_QWEN_FORCED_ALIGNER_ID)
    parser.add_argument("--qwen-asr-forced-aligner-dir", default=None)
    parser.add_argument(
        "--skip-default-models",
        action="store_true",
        help="Skip default WhisperX/GigaAM/pyannote downloads and only run the explicitly requested helper/model actions.",
    )
    parser.add_argument(
        "--ctc-kenlm-lang",
        action="append",
        default=[],
        choices=["uk", "en"],
        help="Also fetch packaged CTC KenLM assets for the selected language.",
    )
    return parser


def _tool_script(name: str) -> Path:
    return Path(__file__).resolve().parent / "tools" / name


def _run_tool(script_name: str, *args: str) -> None:
    script_path = _tool_script(script_name)
    command = [sys.executable, str(script_path), *args]
    print(f"[DL] Running {' '.join(command)}")
    subprocess.run(command, check=True)


def _preset_whisper_models(preset_paths: list[str], preset_dirs: list[str]) -> list[str]:
    models: set[str] = set()
    if not preset_paths and not preset_dirs:
        return []
    for preset in load_presets(preset_paths, preset_dirs):
        if preset.backend.id == "whisperx":
            models.update(resolve_lang_model_map(preset.backend.options).values())
            models.add(resolve_detect_model(preset.backend.options))
            continue
        if preset.backend.id == "therapy_hybrid":
            models.add(resolve_therapy_backend_options(preset.backend.options)["whisper"]["model_name"])
    return sorted(models)


def _preset_gigaam_models(preset_paths: list[str], preset_dirs: list[str]) -> list[str]:
    models: set[str] = set()
    if not preset_paths and not preset_dirs:
        return []
    for preset in load_presets(preset_paths, preset_dirs):
        if preset.backend.id == "gigaam_ctc":
            model_name = str(preset.backend.options.get("model_name") or "ai-sage/GigaAM-v3")
            models.add(model_name)
            continue
        if preset.backend.id == "therapy_hybrid":
            models.add(resolve_therapy_backend_options(preset.backend.options)["ctc"]["model_name"])
    return sorted(models)


def _preset_canary_models(preset_paths: list[str], preset_dirs: list[str]) -> list[str]:
    models: set[str] = set()
    if not preset_paths and not preset_dirs:
        return []
    for preset in load_presets(preset_paths, preset_dirs):
        if preset.backend.id == "canary":
            models.add(str(preset.backend.options.get("model_id") or "nvidia/canary-1b-v2"))
            continue
        if preset.backend.id == "therapy_hybrid":
            canary_options = dict(resolve_therapy_backend_options(preset.backend.options).get("canary") or {})
            if str(canary_options.get("provider") or "").lower() == "backend":
                backend_options = dict(canary_options.get("backend_options") or {})
                models.add(str(backend_options.get("model_id") or "nvidia/canary-1b-v2"))
    return sorted(models)


def main() -> None:
    """Download every model needed by the current routing and diarization configuration."""

    args = build_parser().parse_args()
    need_default_models = not args.skip_default_models
    if need_default_models and not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set in .env for gated model downloads.")

    if need_default_models:
        from huggingface_hub import snapshot_download

        whisper_models = sorted(set(LANG_MODEL_MAP.values()) | {"large-v3"} | set(_preset_whisper_models(args.preset, args.preset_dir)))
        gigaam_models = sorted({"ai-sage/GigaAM-v3"} | set(_preset_gigaam_models(args.preset, args.preset_dir)))
        canary_models = sorted(set(_preset_canary_models(args.preset, args.preset_dir)))
        pyannote_models = [PYANNOTE_DIARIZATION_MODEL, PYANNOTE_SEGMENTATION_MODEL]
        for model_id in whisper_models + gigaam_models + canary_models + pyannote_models:
            print(f"[DL] {model_id}")
            snapshot_download(repo_id=model_id, token=HF_TOKEN)
    if args.install_ctc_kenlm_helper:
        _run_tool("setup_ctc_kenlm_helper.py")
    if args.install_canary_helper:
        _run_tool("setup_canary_helper.py")
    if args.install_ollama_qwen:
        _run_tool("setup_ollama_qwen.py")
    if args.install_qwen_asr_helper:
        _run_tool("setup_qwen_asr_helper.py")
    if args.download_qwen_asr_model:
        assets = download_qwen_assets(
            model_id=args.qwen_asr_model_id,
            model_dir=args.qwen_asr_model_dir,
            forced_aligner_id=args.qwen_asr_forced_aligner_id,
            forced_aligner_dir=args.qwen_asr_forced_aligner_dir,
            download_forced_aligner=args.download_qwen_asr_forced_aligner,
            token=HF_TOKEN or None,
        )
        print(f"[DL] Qwen ASR assets: {assets}")
    if args.ctc_kenlm_lang:
        tool_args: list[str] = []
        for lang in sorted(set(args.ctc_kenlm_lang)):
            tool_args.extend(["--lang", lang])
        _run_tool("fetch_ctc_lm_assets.py", *tool_args)
    print("[DL] All models downloaded.")


if __name__ == "__main__":
    main()
