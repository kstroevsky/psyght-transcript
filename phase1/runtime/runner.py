"""Behavior-preserving orchestration for the standalone phase1 transcription CLI."""

from __future__ import annotations

import signal
import time
import traceback
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import current_thread, main_thread
from typing import Any, Iterator

from contracts.transcript import Transcript
from phase1.backends import BackendSpec
from phase1.output.formatter import save
from phase1.pipeline.align import align
from phase1.pipeline.assign import assign_and_build, assigned_segments_for_dump
from phase1.pipeline.asr import transcribe
from phase1.pipeline.audio import AudioBundle, SAMPLE_RATE, apply_audio_preprocessing, load_audio, write_wav_mono
from phase1.pipeline.diarize import diarization_to_records, diarize, get_diarization_runtime_config, preflight_diarization
from phase1.runtime.artifacts import (
    persist_error_artifact,
    persist_partial_artifacts,
    write_intermediate_artifact,
    write_run_meta,
    write_structured_transcript,
)
from phase1.runtime.builder import build_single_speaker_transcript, structured_segments, synthetic_diarization_segments
from phase1.runtime.options import (
    PipelineArtifacts,
    ProgressState,
    RunExecutionResult,
    RunOptions,
    build_run_meta,
    build_run_paths,
)
from phase1.runtime.progress import (
    STAGE_ALIGNMENT_AND_DIARIZATION_COMPLETED,
    STAGE_ALIGNMENT_AND_DIARIZATION_STARTED,
    STAGE_ALIGNMENT_COMPLETED,
    STAGE_ALIGNMENT_SKIPPED,
    STAGE_ALIGNMENT_STARTED,
    STAGE_ASR_COMPLETED,
    STAGE_ASR_STARTED,
    STAGE_AUDIO_LOADING,
    STAGE_COMPLETED,
    STAGE_DIARIZATION_COMPLETED,
    STAGE_DIARIZATION_PREFLIGHT_COMPLETED,
    STAGE_DIARIZATION_PREFLIGHT_STARTED,
    STAGE_DIARIZATION_SKIPPED,
    STAGE_DIARIZATION_STARTED,
    STAGE_MERGE_COMPLETED,
    STAGE_STARTED,
    write_progress,
)


class PipelineInterrupted(RuntimeError):
    """Raised when phase1 receives a catchable termination signal."""


def _update_progress(progress_path, progress: ProgressState, stage: str, percent: int) -> None:
    progress.stage = stage
    progress.percent = percent
    write_progress(progress_path, "running", progress.percent, progress.stage)


def _apply_duration_limit(audio_bundle: AudioBundle, duration_limit: float | None) -> AudioBundle:
    if duration_limit is None:
        return audio_bundle
    print(f"[AUDIO] Slicing to first {duration_limit} seconds.")
    limited_audio = audio_bundle.audio[: int(duration_limit * SAMPLE_RATE)]
    return AudioBundle(audio=limited_audio, duration=len(limited_audio) / SAMPLE_RATE)


def _apply_audio_preprocessing(
    paths,
    run_meta: dict[str, Any],
    options: RunOptions,
    audio_bundle: AudioBundle,
    stage_timings: dict[str, float],
) -> AudioBundle:
    requested = (run_meta.get("runtime", {}).get("audio_preprocessing") or {}).get("requested") or []
    artifact_path = None
    started = time.perf_counter()
    processed_audio, applied = apply_audio_preprocessing(audio_bundle.audio, options.backend.options)
    if paths.intermediate_dir is not None and (options.experiment_id is not None or requested or applied):
        artifact_path = paths.intermediate_dir / "00_processed_audio.wav"
        write_wav_mono(artifact_path, processed_audio)
    stage_timings["audio_preprocessing"] = round(time.perf_counter() - started, 6)
    runtime_audio = run_meta["runtime"]["audio_preprocessing"]
    runtime_audio["applied"] = applied
    runtime_audio["artifact_path"] = str(artifact_path) if artifact_path is not None else None
    return AudioBundle(audio=processed_audio, duration=len(processed_audio) / SAMPLE_RATE)


def _write_diarization_outputs(paths, diarized: Any, diarization_meta: dict[str, Any] | None) -> None:
    write_intermediate_artifact(paths, "03_diarization_segments.json", diarization_to_records(diarized))
    write_intermediate_artifact(
        paths,
        "03_diarization_meta.json",
        diarization_meta or get_diarization_runtime_config(),
    )


def _update_runtime_metadata(run_meta: dict[str, Any], state: PipelineArtifacts) -> None:
    """Merge the latest resolved stage runtimes into the persisted run metadata."""

    runtime = run_meta["runtime"]
    if state.asr_meta is not None:
        runtime["device"] = state.asr_meta.get("device", runtime["device"])
        runtime["compute_type"] = state.asr_meta.get("compute_type", runtime["compute_type"])
        runtime["asr_threads"] = state.asr_meta.get("threads")
        runtime["asr_batch_size"] = state.asr_meta.get("batch_size")
        runtime["asr_vad_method"] = state.asr_meta.get("vad_method")
        runtime["asr_vad_chunk_size"] = state.asr_meta.get("vad_chunk_size")
        runtime["asr_vad_options"] = state.asr_meta.get("vad_options")
        runtime["resolved_asr"] = state.asr_meta
    if state.alignment_meta is not None:
        runtime["resolved_alignment"] = state.alignment_meta
    if state.diarization_meta is not None:
        runtime["diarize_device"] = state.diarization_meta.get("device", runtime["diarize_device"])
        runtime["resolved_diarization"] = state.diarization_meta


def _unpack_transcribe_result(
    result: tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    if len(result) == 3:
        return result
    raw_segments, detected_language = result
    return raw_segments, detected_language, None


def _unpack_align_result(
    result: list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


def _raise_interruption(signum: int, _frame: Any) -> None:
    raise PipelineInterrupted(f"Received signal {signal.Signals(signum).name}.")


@contextmanager
def _installed_signal_handlers() -> Iterator[None]:
    previous: dict[int, Any] = {}
    if current_thread() is not main_thread():
        yield
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handler = signal.getsignal(sig)
            signal.signal(sig, _raise_interruption)
            previous[sig] = previous_handler
        except (ValueError, OSError, RuntimeError):  # pragma: no cover - depends on runtime thread state
            continue
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def _measure(stage_timings: dict[str, float], stage_name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        stage_timings[stage_name] = round(time.perf_counter() - started, 6)


def _run_parallel_post_asr(
    options: RunOptions,
    paths,
    progress: ProgressState,
    state: PipelineArtifacts,
    audio: Any,
    stage_timings: dict[str, float],
) -> None:
    _update_progress(paths.progress_path, progress, STAGE_ALIGNMENT_AND_DIARIZATION_STARTED, 50)

    def run_align() -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
        started = time.perf_counter()
        result = align(
            state.raw_segments,
            state.detected_language,
            audio,
            True,
            backend_id=options.backend.id,
            backend_options=options.backend.options,
        )
        aligned_segments, alignment_meta = _unpack_align_result(result)
        return aligned_segments, alignment_meta, round(time.perf_counter() - started, 6)

    def run_diarization() -> tuple[Any, dict[str, Any] | None, float]:
        started = time.perf_counter()
        diarized, diarization_meta = diarize(
            audio,
            min_speakers=options.effective_min_speakers,
            max_speakers=options.effective_max_speakers,
            return_meta=True,
            overrides=options.backend.options,
        )
        return diarized, diarization_meta, round(time.perf_counter() - started, 6)

    with ThreadPoolExecutor(max_workers=2) as pool:
        align_future = pool.submit(run_align)
        diarize_future = pool.submit(run_diarization)
        align_exc = None
        diarize_exc = None
        try:
            state.aligned_segments, state.alignment_meta, stage_timings["alignment"] = align_future.result()
            write_intermediate_artifact(paths, "02_aligned_segments.json", state.aligned_segments)
        except Exception as exc:  # pragma: no cover - defensive capture
            align_exc = exc
        try:
            state.diarized, state.diarization_meta, stage_timings["diarization"] = diarize_future.result()
            _write_diarization_outputs(paths, state.diarized, state.diarization_meta)
        except Exception as exc:  # pragma: no cover - defensive capture
            diarize_exc = exc
    if align_exc:
        raise align_exc
    if diarize_exc:
        raise diarize_exc
    _update_progress(paths.progress_path, progress, STAGE_ALIGNMENT_AND_DIARIZATION_COMPLETED, 85)


def _run_sequential_post_asr(
    options: RunOptions,
    paths,
    progress: ProgressState,
    state: PipelineArtifacts,
    audio: Any,
    stage_timings: dict[str, float],
) -> None:
    if options.skip_alignment:
        state.aligned_segments = state.raw_segments
        state.alignment_meta = {
            "device": None,
            "skip_reason": "alignment disabled by run option",
        }
        stage_timings["alignment"] = 0.0
        write_intermediate_artifact(paths, "02_aligned_segments.json", state.aligned_segments)
        _update_progress(paths.progress_path, progress, STAGE_ALIGNMENT_SKIPPED, 60)
    else:
        _update_progress(paths.progress_path, progress, STAGE_ALIGNMENT_STARTED, 50)
        with _measure(stage_timings, "alignment"):
            align_result = align(
                state.raw_segments,
                state.detected_language,
                audio,
                True,
                backend_id=options.backend.id,
                backend_options=options.backend.options,
            )
            state.aligned_segments, state.alignment_meta = _unpack_align_result(align_result)
        write_intermediate_artifact(paths, "02_aligned_segments.json", state.aligned_segments)
        _update_progress(paths.progress_path, progress, STAGE_ALIGNMENT_COMPLETED, 70)

    if options.effective_skip_diarization:
        state.diarization_meta = {
            "mode": "synthetic_single_speaker",
            "reason": "single-speaker mode or diarization disabled",
        }
        stage_timings["diarization"] = 0.0
        write_intermediate_artifact(paths, "03_diarization_meta.json", state.diarization_meta)
        _update_progress(paths.progress_path, progress, STAGE_DIARIZATION_SKIPPED, 85)
        return

    _update_progress(paths.progress_path, progress, STAGE_DIARIZATION_STARTED, 72)
    with _measure(stage_timings, "diarization"):
        state.diarized, state.diarization_meta = diarize(
            audio,
            min_speakers=options.effective_min_speakers,
            max_speakers=options.effective_max_speakers,
            return_meta=True,
            overrides=options.backend.options,
        )
    _write_diarization_outputs(paths, state.diarized, state.diarization_meta)
    _update_progress(paths.progress_path, progress, STAGE_DIARIZATION_COMPLETED, 85)


def _build_transcript(options: RunOptions, paths, state: PipelineArtifacts) -> Transcript:
    merge_same_speaker = bool(options.backend.options.get("merge_consecutive_speakers", False))
    merge_gap_sec = float(options.backend.options.get("speaker_merge_gap_sec", 0.35))
    speaker_assignment_strategy = str(
        options.backend.options.get("speaker_assignment_strategy")
        or ("whisperx" if options.backend.id == "therapy_hybrid" else "overlap")
    )
    speaker_label_style = str(
        options.backend.options.get("speaker_label_style")
        or ("alpha" if options.backend.id == "therapy_hybrid" else "canonical")
    )
    if options.effective_skip_diarization:
        transcript = build_single_speaker_transcript(
            state.aligned_segments,
            state.detected_language,
            state.duration,
            options.content_mode,
        )
        write_intermediate_artifact(paths, "03_diarization_segments.json", synthetic_diarization_segments(state.duration))
        write_intermediate_artifact(paths, "04_assigned_segments.json", structured_segments(transcript.segments))
        return transcript

    write_intermediate_artifact(
        paths,
        "04_assigned_segments.json",
        assigned_segments_for_dump(
            state.aligned_segments,
            state.diarized,
            merge_consecutive_same_speaker=merge_same_speaker,
            merge_gap_sec=merge_gap_sec,
            strategy=speaker_assignment_strategy,
            speaker_label_style=speaker_label_style,
        ),
    )
    return assign_and_build(
        state.aligned_segments,
        state.diarized,
        state.detected_language,
        state.duration,
        merge_consecutive_same_speaker=merge_same_speaker,
        merge_gap_sec=merge_gap_sec,
        strategy=speaker_assignment_strategy,
        speaker_label_style=speaker_label_style,
    )


def _finalize_run_meta(paths, run_meta: dict[str, Any], stage_timings: dict[str, float], started_at: float) -> float:
    wall_clock_sec = round(time.perf_counter() - started_at, 6)
    run_meta["stage_timings_sec"] = dict(stage_timings)
    run_meta["wall_clock_sec"] = wall_clock_sec
    write_run_meta(paths.run_meta_path, run_meta)
    return wall_clock_sec


def execute(
    options: RunOptions,
    *,
    audio_bundle: AudioBundle | None = None,
) -> RunExecutionResult:
    """Run the full phase1 pipeline and return rich execution details."""

    paths = build_run_paths(options)
    progress = ProgressState(stage=STAGE_STARTED, percent=0)
    state = PipelineArtifacts()
    run_meta = build_run_meta(options)
    stage_timings: dict[str, float] = {}
    wall_clock_started = time.perf_counter()

    write_run_meta(paths.run_meta_path, run_meta)
    write_progress(paths.progress_path, "running", progress.percent, progress.stage)

    try:
        with _installed_signal_handlers():
            if not options.effective_skip_diarization:
                _update_progress(paths.progress_path, progress, STAGE_DIARIZATION_PREFLIGHT_STARTED, 1)
                with _measure(stage_timings, "diarization_preflight"):
                    state.diarization_meta = preflight_diarization(options.backend.options)
                run_meta["runtime"]["diarization_preflight"] = "ok"
                _update_runtime_metadata(run_meta, state)
                write_run_meta(paths.run_meta_path, run_meta)
                _update_progress(paths.progress_path, progress, STAGE_DIARIZATION_PREFLIGHT_COMPLETED, 2)

            _update_progress(paths.progress_path, progress, STAGE_AUDIO_LOADING, 5)
            with _measure(stage_timings, "audio_loading"):
                effective_audio_bundle = audio_bundle
                if effective_audio_bundle is None:
                    audio, duration = load_audio(options.audio_path)
                    effective_audio_bundle = AudioBundle(audio=audio, duration=duration)
                effective_audio_bundle = _apply_duration_limit(effective_audio_bundle, options.duration_limit)
                effective_audio_bundle = _apply_audio_preprocessing(
                    paths,
                    run_meta,
                    options,
                    effective_audio_bundle,
                    stage_timings,
                )
            state.duration = effective_audio_bundle.duration
            run_meta["duration_sec"] = state.duration
            run_meta["effective_duration_limit_sec"] = state.duration
            write_run_meta(paths.run_meta_path, run_meta)

            _update_progress(paths.progress_path, progress, STAGE_ASR_STARTED, 10)
            with _measure(stage_timings, "asr"):
                transcribe_result = transcribe(
                    effective_audio_bundle.audio,
                    options.language,
                    asr_options=options.asr_options,
                    return_meta=True,
                    backend_id=options.backend.id,
                    backend_options=options.backend.options,
                )
            state.raw_segments, state.detected_language, state.asr_meta = _unpack_transcribe_result(transcribe_result)
            artifact_payloads = {}
            if state.asr_meta is not None:
                artifact_payloads = dict(state.asr_meta.pop("artifact_payloads", {}) or {})
            run_meta["detected_language"] = state.detected_language
            _update_runtime_metadata(run_meta, state)
            write_run_meta(paths.run_meta_path, run_meta)
            write_intermediate_artifact(paths, "01_asr_raw_segments.json", state.raw_segments)
            for artifact_name, artifact_payload in artifact_payloads.items():
                write_intermediate_artifact(paths, artifact_name, artifact_payload)
            _update_progress(paths.progress_path, progress, STAGE_ASR_COMPLETED, 45)

            if options.parallel_post_asr and not options.skip_alignment and not options.effective_skip_diarization:
                _run_parallel_post_asr(
                    options,
                    paths,
                    progress,
                    state,
                    effective_audio_bundle.audio,
                    stage_timings,
                )
            else:
                _run_sequential_post_asr(
                    options,
                    paths,
                    progress,
                    state,
                    effective_audio_bundle.audio,
                    stage_timings,
                )
            _update_runtime_metadata(run_meta, state)
            write_run_meta(paths.run_meta_path, run_meta)

            with _measure(stage_timings, "merge"):
                transcript = _build_transcript(options, paths, state)
            _update_progress(paths.progress_path, progress, STAGE_MERGE_COMPLETED, 95)
            with _measure(stage_timings, "output_save"):
                save(transcript, paths.output_stem)
                write_structured_transcript(paths, transcript)
            wall_clock_sec = _finalize_run_meta(paths, run_meta, stage_timings, wall_clock_started)
            write_progress(paths.progress_path, "completed", 100, STAGE_COMPLETED)
            return RunExecutionResult(
                transcript=transcript,
                options=options,
                paths=paths,
                run_meta=run_meta,
                stage_timings_sec=dict(stage_timings),
                wall_clock_sec=wall_clock_sec,
                audio_bundle=effective_audio_bundle,
            )
    except (Exception, KeyboardInterrupt) as exc:
        persist_partial_artifacts(paths, state)
        persist_error_artifact(paths.intermediate_dir, progress.stage, exc)
        _finalize_run_meta(paths, run_meta, stage_timings, wall_clock_started)
        write_progress(
            paths.progress_path,
            "failed",
            progress.percent,
            progress.stage,
            details={"error": str(exc), "traceback": traceback.format_exc()},
        )
        raise


def run(
    audio_path: str,
    language: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    content_mode: str = "multi-speaker",
    save_intermediate_dir: str | None = None,
    progress_file: str | None = None,
    parallel_post_asr: bool = True,
    skip_alignment: bool = False,
    skip_diarization: bool = False,
    output_dir: str | None = None,
    duration_limit: float | None = None,
    asr_options: dict[str, Any] | None = None,
    backend: BackendSpec | None = None,
) -> Transcript:
    """Run the full phase1 pipeline while preserving the established CLI contracts."""

    options = RunOptions(
        audio_path=audio_path,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        content_mode=content_mode,
        save_intermediate_dir=save_intermediate_dir,
        progress_file=progress_file,
        parallel_post_asr=parallel_post_asr,
        skip_alignment=skip_alignment,
        skip_diarization=skip_diarization,
        output_dir=output_dir,
        duration_limit=duration_limit,
        asr_options=asr_options,
        backend=backend or BackendSpec(),
    )
    return execute(options).transcript
