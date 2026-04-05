"""High-level workflow helpers for Qwen3-ASR bootstrap, training, and pipeline runs."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from phase1.backends import BackendSpec
from phase1.config import HF_TOKEN
from phase1.config.env import BASE_DIR, MODELS_DIR
from phase1.runtime.options import RunOptions
from phase1.runtime.runner import execute

DEFAULT_QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_QWEN_FORCED_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess result reused by CLI and UI layers."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def combined_output(self) -> str:
        stdout = self.stdout.strip()
        stderr = self.stderr.strip()
        if stdout and stderr:
            return f"{stdout}\n{stderr}"
        return stdout or stderr


def _slug_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(":", "_")


def qwen_assets_root() -> Path:
    override = os.getenv("PHASE1_QWEN_ASR_MODEL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (MODELS_DIR / "qwen_asr").resolve()


def qwen_default_asset_dir(repo_id: str) -> Path:
    return qwen_assets_root() / _slug_repo_id(repo_id)


def qwen_download_workers() -> int:
    override = os.getenv("PHASE1_QWEN_ASR_DOWNLOAD_WORKERS", "1").strip()
    try:
        return max(1, int(override))
    except ValueError:
        return 1


def qwen_helper_python() -> Path:
    override = os.getenv("PHASE1_QWEN_ASR_PYTHON")
    if override:
        return Path(override).expanduser()
    return BASE_DIR / ".qwen-asr-venv" / "bin" / "python"


def qwen_setup_script() -> Path:
    override = os.getenv("PHASE1_QWEN_ASR_SETUP_SCRIPT")
    if override:
        return Path(override).expanduser().resolve()
    return (BASE_DIR / "tools" / "setup_qwen_asr_helper.py").resolve()


def qwen_helper_transcribe_script() -> Path:
    override = os.getenv("PHASE1_QWEN_ASR_HELPER_SCRIPT")
    if override:
        return Path(override).expanduser().resolve()
    return (BASE_DIR / "tools" / "qwen_asr_transcribe.py").resolve()


def _normalize_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(value)


def resolve_local_qwen_asset_path(repo_id: str) -> str | None:
    candidate = qwen_default_asset_dir(repo_id)
    if candidate.exists():
        return str(candidate)
    return None


def resolve_qwen_model_path(overrides: dict[str, Any] | None = None) -> str:
    """Resolve the preferred Qwen3-ASR model path for training/inference."""

    overrides = overrides or {}
    if overrides.get("model_path"):
        candidate = Path(str(overrides["model_path"])).expanduser()
        return str(candidate.resolve()) if candidate.exists() else str(overrides["model_path"])
    if overrides.get("model_id"):
        return str(overrides["model_id"])
    local = resolve_local_qwen_asset_path(DEFAULT_QWEN_ASR_MODEL_ID)
    return local or DEFAULT_QWEN_ASR_MODEL_ID


def resolve_qwen_forced_aligner_path(overrides: dict[str, Any] | None = None) -> str | None:
    """Resolve the optional local Qwen forced-aligner path."""

    overrides = overrides or {}
    if overrides.get("use_forced_aligner") is False:
        return None
    if overrides.get("forced_aligner_model_path"):
        candidate = Path(str(overrides["forced_aligner_model_path"])).expanduser()
        return str(candidate.resolve()) if candidate.exists() else str(overrides["forced_aligner_model_path"])
    if overrides.get("forced_aligner_model_id"):
        return str(overrides["forced_aligner_model_id"])
    return resolve_local_qwen_asset_path(DEFAULT_QWEN_FORCED_ALIGNER_ID)


def format_command(command: list[str] | tuple[str, ...]) -> str:
    return " ".join(str(part) for part in command)


def run_command(command: list[str] | tuple[str, ...], *, env: dict[str, str] | None = None) -> CommandResult:
    completed = subprocess.run(
        [str(part) for part in command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return CommandResult(
        command=tuple(str(part) for part in command),
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def ensure_qwen_helper(
    *,
    python_executable: str | None = None,
    venv_dir: str | None = None,
    requirements: str | None = None,
) -> CommandResult:
    """Create or refresh the isolated Qwen helper environment."""

    command: list[str] = [
        os.fspath(_normalize_executable(python_executable or os.getenv("PYTHON", os.sys.executable))),
        os.fspath(qwen_setup_script()),
    ]
    if python_executable:
        command.extend(["--python", python_executable])
    if venv_dir:
        command.extend(["--venv-dir", venv_dir])
    if requirements:
        command.extend(["--requirements", requirements])
    return run_command(command)


def download_qwen_assets(
    *,
    model_id: str = DEFAULT_QWEN_ASR_MODEL_ID,
    model_dir: str | None = None,
    forced_aligner_id: str = DEFAULT_QWEN_FORCED_ALIGNER_ID,
    forced_aligner_dir: str | None = None,
    download_forced_aligner: bool = False,
    token: str | None = HF_TOKEN,
) -> dict[str, str]:
    """Download the Qwen ASR model bundle into repo-local storage."""

    workers = qwen_download_workers()
    target_model_dir = Path(model_dir).expanduser().resolve() if model_dir else qwen_default_asset_dir(model_id)
    target_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(target_model_dir),
        token=token or None,
        max_workers=workers,
    )
    payload = {
        "model_id": model_id,
        "model_path": str(Path(model_path).expanduser().resolve()),
    }
    if download_forced_aligner:
        target_aligner_dir = (
            Path(forced_aligner_dir).expanduser().resolve()
            if forced_aligner_dir
            else qwen_default_asset_dir(forced_aligner_id)
        )
        target_aligner_dir.mkdir(parents=True, exist_ok=True)
        aligner_path = snapshot_download(
            repo_id=forced_aligner_id,
            local_dir=str(target_aligner_dir),
            token=token or None,
            max_workers=workers,
        )
        payload["forced_aligner_id"] = forced_aligner_id
        payload["forced_aligner_path"] = str(Path(aligner_path).expanduser().resolve())
    return payload


def run_qwen_training_job(
    *,
    train_file: str,
    output_dir: str,
    model_path: str | None = None,
    eval_file: str | None = None,
    sr: int = 16000,
    batch_size: int = 32,
    grad_acc: int = 4,
    learning_rate: float = 2e-5,
    epochs: float = 1.0,
    log_steps: int = 10,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.02,
    num_workers: int = 4,
    pin_memory: int = 1,
    persistent_workers: int = 1,
    prefetch_factor: int = 2,
    save_strategy: str = "steps",
    save_steps: int = 200,
    save_total_limit: int = 5,
    resume_from: str = "",
    resume: int = 0,
    device: str | None = None,
) -> CommandResult:
    """Run the low-level helper-env SFT command from the main environment."""

    command: list[str] = [
        os.fspath(qwen_helper_python()),
        os.fspath((BASE_DIR / "tools" / "qwen_asr_finetune.py").resolve()),
        "train-sft",
        "--train-file",
        str(Path(train_file).expanduser().resolve()),
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
        "--model-path",
        str(model_path or resolve_qwen_model_path()),
        "--sr",
        str(int(sr)),
        "--batch-size",
        str(int(batch_size)),
        "--grad-acc",
        str(int(grad_acc)),
        "--learning-rate",
        str(float(learning_rate)),
        "--epochs",
        str(float(epochs)),
        "--log-steps",
        str(int(log_steps)),
        "--lr-scheduler-type",
        str(lr_scheduler_type),
        "--warmup-ratio",
        str(float(warmup_ratio)),
        "--num-workers",
        str(int(num_workers)),
        "--pin-memory",
        str(int(pin_memory)),
        "--persistent-workers",
        str(int(persistent_workers)),
        "--prefetch-factor",
        str(int(prefetch_factor)),
        "--save-strategy",
        str(save_strategy),
        "--save-steps",
        str(int(save_steps)),
        "--save-total-limit",
        str(int(save_total_limit)),
        "--resume-from",
        str(resume_from or ""),
        "--resume",
        str(int(resume)),
    ]
    if eval_file:
        command.extend(["--eval-file", str(Path(eval_file).expanduser().resolve())])
    if device:
        command.extend(["--device", str(device)])
    return run_command(command)


def run_qwen_pipeline(
    *,
    audio_path: str,
    output_dir: str,
    language: str = "ru",
    model_path: str | None = None,
    context: str = "",
    forced_aligner_model_path: str | None = None,
    use_forced_aligner: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    duration_limit: float | None = None,
    skip_diarization: bool = False,
    save_intermediate_dir: str | None = None,
) -> dict[str, Any]:
    """Run the standard phase1 runtime through the Qwen ASR backend."""

    run_output_dir = Path(output_dir).expanduser().resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = (
        Path(save_intermediate_dir).expanduser().resolve()
        if save_intermediate_dir is not None
        else (run_output_dir / "artifacts").resolve()
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    backend_options: dict[str, Any] = {
        "model_path": str(model_path or resolve_qwen_model_path()),
        "use_forced_aligner": bool(use_forced_aligner),
    }
    if context:
        backend_options["context"] = str(context)
    if forced_aligner_model_path:
        aligner_candidate = Path(str(forced_aligner_model_path)).expanduser()
        backend_options["forced_aligner_model_path"] = (
            str(aligner_candidate.resolve()) if aligner_candidate.exists() else str(forced_aligner_model_path)
        )

    result = execute(
        RunOptions(
            audio_path=str(Path(audio_path).expanduser().resolve()),
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            skip_diarization=bool(skip_diarization),
            save_intermediate_dir=str(artifact_dir),
            output_dir=str(run_output_dir),
            duration_limit=duration_limit,
            backend=BackendSpec(id="qwen_asr", options=backend_options),
        )
    )
    return {
        "output_dir": str(run_output_dir),
        "artifacts_dir": str(artifact_dir),
        "wall_clock_sec": float(result.wall_clock_sec),
        "text_path": str(Path(result.txt_path).expanduser().resolve()),
        "json_path": str(Path(result.json_path).expanduser().resolve()),
    }
