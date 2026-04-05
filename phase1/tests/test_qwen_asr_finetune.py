"""Tests for Qwen3-ASR fine-tuning data preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase1.pipeline.audio import AudioBundle, SAMPLE_RATE
from phase1.qwen_asr.finetune import (
    build_training_examples,
    find_latest_checkpoint,
    format_training_text,
    load_gemini_transcript,
    normalize_gemini_transcript_payload,
    QwenTrainingRuntime,
    trainer_device_kwargs,
    write_training_runtime_manifest,
)


class QwenAsrFinetuneTest(unittest.TestCase):
    def test_load_gemini_transcript_shifts_chunk_relative_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            review_dir = Path(tmp_dir)
            (review_dir / "chunk_000.json").write_text(
                json.dumps(
                    {
                        "chunk_id": "chunk_000",
                        "start_sec": 0.0,
                        "end_sec": 10.0,
                        "segments": [
                            {"speaker": "A", "start_sec": 0.0, "end_sec": 2.0, "text": "раз"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (review_dir / "chunk_001.json").write_text(
                json.dumps(
                    {
                        "chunk_id": "chunk_001",
                        "start_sec": 10.0,
                        "end_sec": 20.0,
                        "segments": [
                            {"speaker": "B", "start_sec": 0.5, "text": "два"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            segments = load_gemini_transcript(review_dir)

        self.assertEqual(2, len(segments))
        self.assertEqual(0.0, segments[0].start_sec)
        self.assertEqual(10.5, segments[1].start_sec)
        self.assertEqual(20.0, segments[1].end_sec)

    def test_build_training_examples_writes_train_eval_and_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            transcript_path = root / "gemini.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "segments": [
                            {"speaker": "A", "start_sec": 0.0, "end_sec": 3.0, "text": "привет"},
                            {"speaker": "A", "start_sec": 3.1, "end_sec": 6.0, "text": "как дела"},
                            {"speaker": "B", "start_sec": 10.0, "end_sec": 13.0, "text": "хорошо"},
                            {"speaker": "B", "start_sec": 13.2, "end_sec": 16.0, "text": "спасибо"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            audio_bundle = AudioBundle(audio=[0.0] * (SAMPLE_RATE * 30), duration=30.0)
            with patch("phase1.qwen_asr.finetune.load_audio_bundle", return_value=audio_bundle):
                manifest = build_training_examples(
                    audio_path="/tmp/source.wav",
                    transcript_source=transcript_path,
                    output_dir=root / "dataset",
                    language="ru",
                    min_clip_seconds=5.0,
                    max_clip_seconds=8.0,
                    max_gap_seconds=0.5,
                    eval_ratio=0.5,
                )

            train_rows = [
                json.loads(line)
                for line in Path(manifest["artifacts"]["train_jsonl"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            eval_rows = [
                json.loads(line)
                for line in Path(manifest["artifacts"]["eval_jsonl"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            launcher = Path(manifest["artifacts"]["launcher"]).read_text(encoding="utf-8")
            clip_exists = Path(train_rows[0]["audio"]).exists()

        self.assertEqual(2, manifest["counts"]["examples"])
        self.assertEqual(1, manifest["counts"]["train_examples"])
        self.assertEqual(1, manifest["counts"]["eval_examples"])
        self.assertTrue(clip_exists)
        self.assertTrue(train_rows[0]["text"].startswith("language Russian<asr_text>"))
        self.assertIn("--train-file", launcher)
        self.assertIn("--eval-file", launcher)

    def test_normalize_gemini_transcript_payload_canonicalizes_list_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            transcript_path = Path(tmp_dir) / "bogomolov.json"
            transcript_path.write_text(
                json.dumps(
                    [
                        {"speaker": "B", "start_sec": 0.0, "text": "раз"},
                        {"speaker": "A", "start_sec": 2.0, "text": "два"},
                        {"speaker": "B", "start_sec": 5.0, "text": "три"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            normalized = normalize_gemini_transcript_payload(transcript_path)

        self.assertEqual(3, len(normalized["segments"]))
        self.assertEqual(0.0, normalized["segments"][0]["start_sec"])
        self.assertEqual(2.0, normalized["segments"][0]["end_sec"])
        self.assertEqual(5.0, normalized["segments"][1]["end_sec"])
        self.assertGreater(normalized["segments"][2]["end_sec"], normalized["segments"][2]["start_sec"])
        self.assertTrue(normalized["warnings"])
        self.assertIn("explicit end times", " ".join(normalized["warnings"]))
        self.assertIn("integer-second", " ".join(normalized["warnings"]))

    def test_format_training_text_maps_short_language_code(self) -> None:
        self.assertEqual(
            "language Russian<asr_text>тест",
            format_training_text("тест", language="ru"),
        )

    def test_trainer_device_kwargs_force_cpu_runtime(self) -> None:
        runtime = QwenTrainingRuntime(
            device="cpu",
            device_map="cpu",
            dtype_name="float32",
            use_bf16=False,
            use_fp16=False,
        )

        kwargs = trainer_device_kwargs(runtime)

        self.assertEqual(
            {
                "no_cuda": True,
                "use_cpu": True,
                "use_mps_device": False,
            },
            kwargs,
        )

    def test_find_latest_checkpoint_returns_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "checkpoint-10").mkdir()
            (root / "checkpoint-200").mkdir()
            (root / "checkpoint-3").mkdir()

            latest = find_latest_checkpoint(root)

        self.assertEqual(str((root / "checkpoint-200").resolve()), latest)

    def test_write_training_runtime_manifest_persists_resolved_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime = QwenTrainingRuntime(
                device="cpu",
                device_map="cpu",
                dtype_name="float32",
                use_bf16=False,
                use_fp16=False,
            )

            manifest_path = write_training_runtime_manifest(
                Path(tmp_dir),
                model_path="/tmp/model",
                runtime=runtime,
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(str(Path("/tmp/model").resolve()), payload["model_path"])
        self.assertEqual("cpu", payload["device"])
        self.assertEqual("cpu", payload["device_map"])
        self.assertEqual("float32", payload["dtype"])
        self.assertFalse(payload["use_bf16"])
        self.assertFalse(payload["use_fp16"])


if __name__ == "__main__":
    unittest.main()
