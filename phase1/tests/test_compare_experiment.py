"""Tests for compare-mode experiment orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from contracts.transcript import Segment, Transcript
from phase1.backends import BackendSpec
from phase1.compare_runtime.experiment import resolve_effective_worker_count, run_experiment
from phase1.compare_runtime.presets import ComparePipelineSpec, ComparePreset


class CompareExperimentTest(unittest.TestCase):
    """Verify experiment reporting, ranking, and concurrency guards."""

    def _preset(self, preset_id: str, *, backend_options=None) -> ComparePreset:
        return ComparePreset(
            path=Path(f"/tmp/{preset_id}.json"),
            preset_id=preset_id,
            description=None,
            pipeline=ComparePipelineSpec(),
            backend=BackendSpec(id="whisperx", options=backend_options or {}),
        )

    def test_resolve_effective_worker_count_clamps_non_cpu_presets(self) -> None:
        with patch("phase1.compare_runtime.experiment._cpu_only_for_preset", side_effect=[True, False]):
            workers, reason = resolve_effective_worker_count(
                [self._preset("cpu"), self._preset("gpu")],
                3,
            )

        self.assertEqual(1, workers)
        self.assertIn("serialized", reason)

    def test_run_experiment_reuses_audio_cache_once_for_parallel_cpu_runs(self) -> None:
        presets = [
            self._preset("a", backend_options={"device": "cpu", "alignment_device": "cpu", "diarization_device": "cpu"}),
            self._preset("b", backend_options={"device": "cpu", "alignment_device": "cpu", "diarization_device": "cpu"}),
        ]
        load_calls = 0

        def fake_load_audio_bundle(_audio_path):
            nonlocal load_calls
            load_calls += 1
            return SimpleNamespace(audio=[0, 1, 2], duration=3.0)

        def fake_execute(run_options, audio_bundle=None):
            run_dir = Path(run_options.save_intermediate_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "progress.json").write_text(
                json.dumps({"status": "completed", "stage": "completed"}),
                encoding="utf-8",
            )
            (run_dir / "00_run_meta.json").write_text(
                json.dumps(
                    {
                        "duration_sec": 3.0,
                        "wall_clock_sec": 1.5,
                        "stage_timings_sec": {"asr": 1.0},
                        "runtime": {
                            "resolved_asr": {"fallback_reason": None},
                            "resolved_alignment": {"fallback_reason": None},
                            "resolved_diarization": {"fallback_reason": None},
                        },
                        "skip_alignment": False,
                        "skip_diarization": False,
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNotNone(audio_bundle)
            return SimpleNamespace(
                transcript=Transcript(
                    language="en",
                    duration=3.0,
                    segments=[Segment(speaker="SPEAKER_00", start=0.0, end=1.0, text=run_options.preset_id or "x")],
                )
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch("phase1.compare_runtime.experiment.load_audio_bundle", side_effect=fake_load_audio_bundle),
                patch("phase1.compare_runtime.experiment.write_audio_cache"),
                patch("phase1.compare_runtime.experiment.read_audio_cache", return_value=[0, 1, 2]),
                patch("phase1.compare_runtime.experiment.execute", side_effect=fake_execute),
            ):
                summary = run_experiment(
                    audio_path="/tmp/audio.mp3",
                    presets=presets,
                    output_dir=tmp_dir,
                    reference_text=None,
                    max_workers=2,
                )
            self.assertTrue(Path(summary["saved_presets_dir"]).exists())

        self.assertEqual(1, load_calls)
        self.assertEqual(2, summary["effective_max_workers"])
        self.assertEqual({"a", "b"}, {run["preset_id"] for run in summary["runs"]})

    def test_run_experiment_ranks_with_reference_metrics(self) -> None:
        presets = [self._preset("better"), self._preset("worse")]

        def fake_execute(run_options, audio_bundle=None):
            del audio_bundle
            run_dir = Path(run_options.save_intermediate_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "progress.json").write_text(
                json.dumps({"status": "completed", "stage": "completed"}),
                encoding="utf-8",
            )
            (run_dir / "00_run_meta.json").write_text(
                json.dumps(
                    {
                        "duration_sec": 2.0,
                        "wall_clock_sec": 1.0,
                        "stage_timings_sec": {"asr": 0.5},
                        "runtime": {
                            "resolved_asr": {"fallback_reason": None},
                            "resolved_alignment": {"fallback_reason": None},
                            "resolved_diarization": {"fallback_reason": None},
                        },
                        "skip_alignment": False,
                        "skip_diarization": False,
                    }
                ),
                encoding="utf-8",
            )
            text = "hello world" if run_options.preset_id == "better" else "goodbye world"
            return SimpleNamespace(
                transcript=Transcript(
                    language="en",
                    duration=2.0,
                    segments=[Segment(speaker="SPEAKER_00", start=0.0, end=1.0, text=text)],
                )
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("phase1.compare_runtime.experiment.execute", side_effect=fake_execute):
                summary = run_experiment(
                    audio_path="/tmp/audio.mp3",
                    presets=presets,
                    output_dir=tmp_dir,
                    reference_text="hello world",
                    max_workers=1,
                )

        self.assertEqual("better", summary["best_preset_id"])
        self.assertLess(summary["runs"][0]["metrics"]["wer"], summary["runs"][1]["metrics"]["wer"])

    def test_run_experiment_uses_rtf_as_reference_tiebreaker(self) -> None:
        presets = [self._preset("faster"), self._preset("slower")]

        def fake_execute(run_options, audio_bundle=None):
            del audio_bundle
            run_dir = Path(run_options.save_intermediate_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            wall_clock = 0.8 if run_options.preset_id == "faster" else 1.4
            (run_dir / "progress.json").write_text(
                json.dumps({"status": "completed", "stage": "completed"}),
                encoding="utf-8",
            )
            (run_dir / "00_run_meta.json").write_text(
                json.dumps(
                    {
                        "duration_sec": 2.0,
                        "wall_clock_sec": wall_clock,
                        "stage_timings_sec": {"asr": wall_clock},
                        "runtime": {
                            "resolved_asr": {"fallback_reason": None},
                            "resolved_alignment": {"fallback_reason": None},
                            "resolved_diarization": {"fallback_reason": None},
                        },
                        "skip_alignment": False,
                        "skip_diarization": False,
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(
                transcript=Transcript(
                    language="en",
                    duration=2.0,
                    segments=[Segment(speaker="SPEAKER_00", start=0.0, end=1.0, text="hello world")],
                )
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("phase1.compare_runtime.experiment.execute", side_effect=fake_execute):
                summary = run_experiment(
                    audio_path="/tmp/audio.mp3",
                    presets=presets,
                    output_dir=tmp_dir,
                    reference_text="hello world",
                    max_workers=1,
                )

        self.assertEqual("faster", summary["best_preset_id"])

    def test_run_experiment_reports_processed_audio_when_present(self) -> None:
        presets = [self._preset("audio")]

        def fake_execute(run_options, audio_bundle=None):
            del audio_bundle
            run_dir = Path(run_options.save_intermediate_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "progress.json").write_text(json.dumps({"status": "completed", "stage": "completed"}), encoding="utf-8")
            (run_dir / "00_run_meta.json").write_text(
                json.dumps(
                    {
                        "duration_sec": 2.0,
                        "wall_clock_sec": 1.0,
                        "stage_timings_sec": {"asr": 0.5},
                        "runtime": {
                            "resolved_asr": {"fallback_reason": None},
                            "resolved_alignment": {"fallback_reason": None},
                            "resolved_diarization": {"fallback_reason": None},
                        },
                        "skip_alignment": False,
                        "skip_diarization": False,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "00_processed_audio.wav").write_bytes(b"wav")
            return SimpleNamespace(
                transcript=Transcript(
                    language="en",
                    duration=2.0,
                    segments=[Segment(speaker="SPEAKER_00", start=0.0, end=1.0, text="hello")],
                )
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("phase1.compare_runtime.experiment.execute", side_effect=fake_execute):
                summary = run_experiment(
                    audio_path="/tmp/audio.mp3",
                    presets=presets,
                    output_dir=tmp_dir,
                    reference_text=None,
                    max_workers=1,
                )

        self.assertTrue(summary["runs"][0]["output_audio"].endswith("00_processed_audio.wav"))


if __name__ == "__main__":
    unittest.main()
