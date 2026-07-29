"""Tests for compare preset loading and default resolution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phase1.compare_runtime.presets import load_presets


class ComparePresetTest(unittest.TestCase):
    """Verify compare preset validation and run-option defaults."""

    def test_duplicate_preset_ids_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = Path(tmp_dir) / "a.json"
            second = Path(tmp_dir) / "b.json"
            payload = {"id": "same", "backend": {"id": "whisperx", "options": {}}}
            first.write_text(json.dumps(payload), encoding="utf-8")
            second.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate preset id"):
                load_presets([str(first), str(second)], None)

    def test_unknown_backend_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            preset = Path(tmp_dir) / "broken.json"
            preset.write_text(json.dumps({"id": "broken", "backend": {"id": "missing", "options": {}}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unknown transcription backend"):
                load_presets([str(preset)], None)

    def test_omitted_fields_fall_back_to_single_run_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            preset_path = Path(tmp_dir) / "default.json"
            preset_path.write_text(json.dumps({"id": "default"}), encoding="utf-8")

            preset = load_presets([str(preset_path)], None)[0]
            options = preset.to_run_options(
                audio_path="/tmp/audio.mp3",
                run_dir=Path(tmp_dir) / "run",
                experiment_id="exp",
            )

            self.assertEqual("gigaam_ctc", preset.backend.id)
            self.assertEqual("e2e_ctc", preset.backend.options["revision"])
            self.assertEqual("multi-speaker", preset.pipeline.content_mode)
            self.assertEqual("multi-speaker", options.content_mode)
            self.assertTrue(options.parallel_post_asr)
            self.assertFalse(options.skip_alignment)
            self.assertFalse(options.skip_diarization)

    def test_gigaam_backend_preset_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            preset_path = Path(tmp_dir) / "gigaam.json"
            preset_path.write_text(
                json.dumps(
                    {
                        "id": "gigaam",
                        "pipeline": {"language": "ru", "duration_limit": 600},
                        "backend": {"id": "gigaam_ctc", "options": {"model_name": "ai-sage/GigaAM-v3"}},
                    }
                ),
                encoding="utf-8",
            )

            preset = load_presets([str(preset_path)], None)[0]

            self.assertEqual("gigaam_ctc", preset.backend.id)
            self.assertEqual(600.0, preset.pipeline.duration_limit)

    def test_therapy_hybrid_backend_preset_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            preset_path = Path(tmp_dir) / "therapy.json"
            preset_path.write_text(
                json.dumps(
                    {
                        "id": "therapy",
                        "pipeline": {"language": "ru"},
                        "backend": {"id": "therapy_hybrid", "options": {"merge_provider": {"id": "rule_based"}}},
                    }
                ),
                encoding="utf-8",
            )

            preset = load_presets([str(preset_path)], None)[0]

            self.assertEqual("therapy_hybrid", preset.backend.id)
            self.assertEqual("rule_based", preset.backend.options["merge_provider"]["id"])


if __name__ == "__main__":
    unittest.main()
