"""Tests for the refactored phase1 runtime orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contracts.transcript import Segment, Transcript
from phase1.backends import BackendSpec
from phase1.runtime import runner


class RunnerTest(unittest.TestCase):
    """Verify that the refactored runner preserves progress and artifact contracts."""

    def _audio_path(self, tmp_dir: str) -> str:
        audio_path = Path(tmp_dir) / "meeting.mp3"
        audio_path.write_bytes(b"")
        return str(audio_path)

    def test_full_multi_speaker_success_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            intermediate_dir = Path(tmp_dir) / "artifacts"
            output_dir = Path(tmp_dir) / "out"
            transcript = Transcript(
                language="en",
                duration=2.0,
                segments=[Segment(speaker="SPEAKER_01", start=0.0, end=1.0, text="hello")],
            )

            with (
                patch.object(runner, "preflight_diarization", return_value={"device": "cpu"}),
                patch.object(runner, "load_audio", return_value=(list(range(32000)), 2.0)),
                patch.object(
                    runner,
                    "transcribe",
                    return_value=(
                        [{"start": 0.0, "end": 1.0, "text": "hello"}],
                        "en",
                        {
                            "device": "cpu",
                            "compute_type": "int8",
                            "threads": 10,
                            "batch_size": 2,
                            "vad_method": "silero",
                            "vad_chunk_size": 10,
                            "fallback_reason": "MPS requested for ASR but unsupported; falling back to CPU.",
                        },
                    ),
                ),
                patch.object(
                    runner,
                    "align",
                    return_value=(
                        [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}],
                        {"device": "cpu", "fallback_reason": "MPS alignment init failed; falling back to CPU."},
                    ),
                ),
                patch.object(runner, "diarize", return_value=("diarized", {"device": "cpu"})),
                patch.object(runner, "assigned_segments_for_dump", return_value=[{"speaker": "SPEAKER_01"}]),
                patch.object(runner, "assign_and_build", return_value=transcript),
                patch.object(runner, "save") as save_mock,
            ):
                result = runner.run(
                    audio_path=self._audio_path(tmp_dir),
                    language="en",
                    parallel_post_asr=False,
                    save_intermediate_dir=str(intermediate_dir),
                    output_dir=str(output_dir),
                )

            self.assertIs(result, transcript)
            save_mock.assert_called_once()
            self.assertTrue((intermediate_dir / "01_asr_raw_segments.json").exists())
            self.assertTrue((intermediate_dir / "02_aligned_segments.json").exists())
            self.assertTrue((intermediate_dir / "03_diarization_segments.json").exists())
            self.assertTrue((intermediate_dir / "03_diarization_meta.json").exists())
            self.assertTrue((intermediate_dir / "04_assigned_segments.json").exists())
            self.assertTrue((intermediate_dir / "05_transcript_structured.json").exists())

            progress_payload = json.loads((intermediate_dir / "progress.json").read_text(encoding="utf-8"))
            run_meta_payload = json.loads((intermediate_dir / "00_run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", progress_payload["status"])
            self.assertEqual(100, progress_payload["percent"])
            self.assertEqual("completed", progress_payload["stage"])
            self.assertEqual("en", run_meta_payload["detected_language"])
            self.assertEqual({"device": "cpu"}, run_meta_payload["runtime"]["resolved_diarization"])
            self.assertEqual("cpu", run_meta_payload["runtime"]["resolved_asr"]["device"])
            self.assertEqual(
                "MPS requested for ASR but unsupported; falling back to CPU.",
                run_meta_payload["runtime"]["resolved_asr"]["fallback_reason"],
            )
            self.assertEqual(10, run_meta_payload["runtime"]["asr_threads"])
            self.assertEqual("cpu", run_meta_payload["runtime"]["resolved_alignment"]["device"])

    def test_single_speaker_path_synthesizes_diarization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            intermediate_dir = Path(tmp_dir) / "artifacts"

            with (
                patch.object(runner, "preflight_diarization") as preflight_mock,
                patch.object(runner, "load_audio", return_value=(list(range(32000)), 2.0)),
                patch.object(
                    runner,
                    "transcribe",
                    return_value=(
                        [{"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 0.5}]}],
                        "en",
                    ),
                ),
                patch.object(
                    runner,
                    "align",
                    return_value=[{"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 0.5}]}],
                ),
                patch.object(runner, "diarize") as diarize_mock,
                patch.object(runner, "save"),
            ):
                transcript = runner.run(
                    audio_path=self._audio_path(tmp_dir),
                    language="en",
                    content_mode="single-speaker",
                    parallel_post_asr=False,
                    save_intermediate_dir=str(intermediate_dir),
                )

            preflight_mock.assert_not_called()
            diarize_mock.assert_not_called()
            self.assertEqual("SPEAKER_00", transcript.segments[0].speaker)
            diarization_payload = json.loads(
                (intermediate_dir / "03_diarization_segments.json").read_text(encoding="utf-8")
            )
            assigned_payload = json.loads(
                (intermediate_dir / "04_assigned_segments.json").read_text(encoding="utf-8")
            )
            self.assertEqual([{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}], diarization_payload)
            self.assertEqual("SPEAKER_00", assigned_payload[0]["speaker"])

    def test_parallel_post_asr_branch_uses_combined_progress_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stages: list[tuple[str, int, str]] = []
            transcript = Transcript(
                language="en",
                duration=2.0,
                segments=[Segment(speaker="SPEAKER_00", start=0.0, end=1.0, text="hello")],
            )

            def record_progress(_path, status, percent, stage, details=None):
                stages.append((status, percent, stage))

            with (
                patch.object(runner, "write_progress", side_effect=record_progress),
                patch.object(runner, "preflight_diarization", return_value={"device": "cpu"}),
                patch.object(runner, "load_audio", return_value=(list(range(32000)), 2.0)),
                patch.object(runner, "transcribe", return_value=([{"start": 0.0, "end": 1.0, "text": "hello"}], "en")),
                patch.object(runner, "align", return_value=[{"start": 0.0, "end": 1.0, "text": "hello", "words": []}]),
                patch.object(runner, "diarize", return_value=("diarized", {"device": "cpu"})),
                patch.object(runner, "assigned_segments_for_dump", return_value=[]),
                patch.object(runner, "assign_and_build", return_value=transcript),
                patch.object(runner, "save"),
            ):
                runner.run(
                    audio_path=self._audio_path(tmp_dir),
                    language="en",
                    parallel_post_asr=True,
                    save_intermediate_dir=str(Path(tmp_dir) / "artifacts"),
                )

            self.assertIn(("running", 50, runner.STAGE_ALIGNMENT_AND_DIARIZATION_STARTED), stages)
            self.assertIn(("running", 85, runner.STAGE_ALIGNMENT_AND_DIARIZATION_COMPLETED), stages)
            self.assertNotIn(("running", 50, runner.STAGE_ALIGNMENT_STARTED), stages)
            self.assertNotIn(("running", 72, runner.STAGE_DIARIZATION_STARTED), stages)

    def test_duration_limit_clips_diarization_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            intermediate_dir = Path(tmp_dir) / "artifacts"
            transcript = Transcript(
                language="en",
                duration=1.0,
                segments=[Segment(speaker="SPEAKER_00", start=0.0, end=1.0, text="hello")],
            )
            diarize_inputs: list[list[int]] = []

            def diarize_side_effect(audio, **_kwargs):
                diarize_inputs.append(audio)
                return "diarized", {"device": "cpu"}

            with (
                patch.object(runner, "preflight_diarization", return_value={"device": "cpu"}),
                patch.object(runner, "load_audio", return_value=(list(range(64000)), 4.0)),
                patch.object(runner, "transcribe", return_value=([{"start": 0.0, "end": 1.0, "text": "hello"}], "en")),
                patch.object(runner, "align", return_value=[{"start": 0.0, "end": 1.0, "text": "hello", "words": []}]),
                patch.object(runner, "diarize", side_effect=diarize_side_effect),
                patch.object(runner, "assigned_segments_for_dump", return_value=[]),
                patch.object(runner, "assign_and_build", return_value=transcript),
                patch.object(runner, "save"),
            ):
                runner.run(
                    audio_path=self._audio_path(tmp_dir),
                    language="en",
                    parallel_post_asr=False,
                    duration_limit=1.0,
                    save_intermediate_dir=str(intermediate_dir),
                )

            self.assertEqual(1, len(diarize_inputs))
            self.assertEqual(16000, len(diarize_inputs[0]))
            run_meta_payload = json.loads((intermediate_dir / "00_run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(1.0, run_meta_payload["duration_sec"])
            self.assertEqual(1.0, run_meta_payload["effective_duration_limit_sec"])

    def test_therapy_backend_defaults_two_speaker_diarization_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            intermediate_dir = Path(tmp_dir) / "artifacts"
            transcript = Transcript(
                language="ru",
                duration=2.0,
                segments=[Segment(speaker="A", start=0.0, end=1.0, text="привет")],
            )
            diarize_kwargs: list[dict[str, object]] = []

            def diarize_side_effect(audio, **kwargs):
                del audio
                diarize_kwargs.append(dict(kwargs))
                return "diarized", {"device": "cpu"}

            with (
                patch.object(runner, "preflight_diarization", return_value={"device": "cpu"}),
                patch.object(runner, "load_audio", return_value=(list(range(32000)), 2.0)),
                patch.object(runner, "transcribe", return_value=([{"start": 0.0, "end": 1.0, "text": "привет"}], "ru")),
                patch.object(runner, "align", return_value=[{"start": 0.0, "end": 1.0, "text": "привет", "words": []}]),
                patch.object(runner, "diarize", side_effect=diarize_side_effect),
                patch.object(runner, "assigned_segments_for_dump", return_value=[]),
                patch.object(runner, "assign_and_build", return_value=transcript),
                patch.object(runner, "save"),
            ):
                runner.execute(
                    runner.RunOptions(
                        audio_path=self._audio_path(tmp_dir),
                        language="ru",
                        parallel_post_asr=False,
                        save_intermediate_dir=str(intermediate_dir),
                        backend=BackendSpec(id="therapy_hybrid", options={"use_live_canary": True}),
                    )
                )

            self.assertEqual(1, len(diarize_kwargs))
            self.assertEqual(2, diarize_kwargs[0]["min_speakers"])
            self.assertEqual(2, diarize_kwargs[0]["max_speakers"])
            run_meta_payload = json.loads((intermediate_dir / "00_run_meta.json").read_text(encoding="utf-8"))
            self.assertIsNone(run_meta_payload["min_speakers"])
            self.assertIsNone(run_meta_payload["max_speakers"])
            self.assertEqual(2, run_meta_payload["effective_min_speakers"])
            self.assertEqual(2, run_meta_payload["effective_max_speakers"])

    def test_audio_preprocessing_metadata_and_artifact_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            intermediate_dir = Path(tmp_dir) / "artifacts"
            transcript = Transcript(
                language="en",
                duration=2.0,
                segments=[Segment(speaker="SPEAKER_00", start=0.0, end=1.0, text="hello")],
            )

            with (
                patch.object(runner, "preflight_diarization", return_value={"device": "cpu"}),
                patch.object(runner, "load_audio", return_value=(list(range(32000)), 2.0)),
                patch.object(
                    runner,
                    "apply_audio_preprocessing",
                    return_value=(
                        list(range(32000)),
                        [{"id": "deepfilternet", "params": {}, "stats": {"input_rms": 0.1, "output_rms": 0.2}}],
                    ),
                ),
                patch.object(runner, "write_wav_mono") as write_wav_mock,
                patch.object(runner, "transcribe", return_value=([{"start": 0.0, "end": 1.0, "text": "hello"}], "en")),
                patch.object(runner, "align", return_value=[{"start": 0.0, "end": 1.0, "text": "hello", "words": []}]),
                patch.object(runner, "diarize", return_value=("diarized", {"device": "cpu"})),
                patch.object(runner, "assigned_segments_for_dump", return_value=[]),
                patch.object(runner, "assign_and_build", return_value=transcript),
                patch.object(runner, "save"),
            ):
                runner.execute(
                    runner.RunOptions(
                        audio_path=self._audio_path(tmp_dir),
                        language="en",
                        parallel_post_asr=False,
                        save_intermediate_dir=str(intermediate_dir),
                        output_dir=str(Path(tmp_dir) / "out"),
                        experiment_id="exp",
                        backend=BackendSpec(id="whisperx", options={"audio_preprocessing": [{"id": "deepfilternet"}]}),
                    )
                )

            run_meta_payload = json.loads((intermediate_dir / "00_run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [{"id": "deepfilternet", "params": {}, "stats": {"input_rms": 0.1, "output_rms": 0.2}}],
                run_meta_payload["runtime"]["audio_preprocessing"]["applied"],
            )
            self.assertTrue(
                str(intermediate_dir / "00_processed_audio.wav")
                == run_meta_payload["runtime"]["audio_preprocessing"]["artifact_path"]
            )
            write_wav_mock.assert_called_once()

    def test_failure_path_persists_partial_artifacts_and_failed_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            intermediate_dir = Path(tmp_dir) / "artifacts"

            with (
                patch.object(runner, "preflight_diarization", return_value={"device": "cpu"}),
                patch.object(runner, "load_audio", return_value=(list(range(32000)), 2.0)),
                patch.object(runner, "transcribe", return_value=([{"start": 0.0, "end": 1.0, "text": "hello"}], "en")),
                patch.object(runner, "align", side_effect=RuntimeError("alignment failed")),
                patch.object(runner, "save"),
            ):
                with self.assertRaisesRegex(RuntimeError, "alignment failed"):
                    runner.run(
                        audio_path=self._audio_path(tmp_dir),
                        language="en",
                        parallel_post_asr=False,
                        save_intermediate_dir=str(intermediate_dir),
                    )

            progress_payload = json.loads((intermediate_dir / "progress.json").read_text(encoding="utf-8"))
            error_payload = json.loads((intermediate_dir / "99_error.json").read_text(encoding="utf-8"))
            self.assertEqual("failed", progress_payload["status"])
            self.assertEqual(50, progress_payload["percent"])
            self.assertEqual(runner.STAGE_ALIGNMENT_STARTED, progress_payload["stage"])
            self.assertEqual(runner.STAGE_ALIGNMENT_STARTED, error_payload["stage"])
            self.assertTrue((intermediate_dir / "01_asr_raw_segments.json").exists())
            self.assertFalse((intermediate_dir / "02_aligned_segments.json").exists())
            self.assertTrue((intermediate_dir / "03_diarization_meta.json").exists())

    def test_keyboard_interrupt_persists_failed_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            intermediate_dir = Path(tmp_dir) / "artifacts"

            with (
                patch.object(runner, "preflight_diarization", return_value={"device": "cpu"}),
                patch.object(runner, "load_audio", side_effect=KeyboardInterrupt()),
                patch.object(runner, "save"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(
                        audio_path=self._audio_path(tmp_dir),
                        language="en",
                        parallel_post_asr=False,
                        save_intermediate_dir=str(intermediate_dir),
                    )

            progress_payload = json.loads((intermediate_dir / "progress.json").read_text(encoding="utf-8"))
            error_payload = json.loads((intermediate_dir / "99_error.json").read_text(encoding="utf-8"))
            self.assertEqual("failed", progress_payload["status"])
            self.assertEqual(5, progress_payload["percent"])
            self.assertEqual(runner.STAGE_AUDIO_LOADING, progress_payload["stage"])
            self.assertEqual(runner.STAGE_AUDIO_LOADING, error_payload["stage"])


if __name__ == "__main__":
    unittest.main()
