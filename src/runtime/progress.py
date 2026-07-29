"""Progress stages and progress-file writer for the phase1 CLI contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from phase1.runtime.artifacts import dump_json, utc_now

STAGE_STARTED = "started"
STAGE_DIARIZATION_PREFLIGHT_STARTED = "diarization_preflight_started"
STAGE_DIARIZATION_PREFLIGHT_COMPLETED = "diarization_preflight_completed"
STAGE_AUDIO_LOADING = "audio_loading"
STAGE_ASR_STARTED = "asr_started"
STAGE_ASR_COMPLETED = "asr_completed"
STAGE_ALIGNMENT_AND_DIARIZATION_STARTED = "alignment_and_diarization_started"
STAGE_ALIGNMENT_AND_DIARIZATION_COMPLETED = "alignment_and_diarization_completed"
STAGE_ALIGNMENT_STARTED = "alignment_started"
STAGE_ALIGNMENT_COMPLETED = "alignment_completed"
STAGE_ALIGNMENT_SKIPPED = "alignment_skipped"
STAGE_DIARIZATION_STARTED = "diarization_started"
STAGE_DIARIZATION_COMPLETED = "diarization_completed"
STAGE_DIARIZATION_SKIPPED = "diarization_skipped"
STAGE_MERGE_COMPLETED = "merge_completed"
STAGE_COMPLETED = "completed"


def write_progress(
    progress_path: Path | None,
    status: str,
    percent: int,
    stage: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Persist the stable progress JSON contract used by later phases and operators."""

    if progress_path is None:
        return
    dump_json(
        progress_path,
        {
            "timestamp_utc": utc_now(),
            "status": status,
            "percent": int(percent),
            "stage": stage,
            "details": details or {},
        },
    )
