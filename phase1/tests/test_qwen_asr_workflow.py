"""Tests for the high-level Qwen3-ASR workflow helpers."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase1.qwen_asr import workflow


class QwenAsrWorkflowTest(unittest.TestCase):
    def test_qwen_helper_python_preserves_venv_symlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            bin_dir = base_dir / ".qwen-asr-venv" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            target = base_dir / "python3.12"
            target.write_text("", encoding="utf-8")
            helper_link = bin_dir / "python"
            helper_link.symlink_to(target)
            with patch.object(workflow, "BASE_DIR", base_dir):
                resolved = workflow.qwen_helper_python()

        self.assertEqual(helper_link, resolved)

    def test_resolve_qwen_model_path_preserves_repo_id(self) -> None:
        resolved = workflow.resolve_qwen_model_path({"model_path": "Qwen/Qwen3-ASR-1.7B"})
        self.assertEqual("Qwen/Qwen3-ASR-1.7B", resolved)

    def test_download_qwen_assets_uses_requested_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "model"
            aligner_dir = Path(tmp_dir) / "aligner"
            with patch.object(workflow, "snapshot_download", side_effect=[str(model_dir), str(aligner_dir)]) as download_mock:
                payload = workflow.download_qwen_assets(
                    model_id="Qwen/Qwen3-ASR-1.7B",
                    model_dir=str(model_dir),
                    forced_aligner_id="Qwen/Qwen3-ForcedAligner-0.6B",
                    forced_aligner_dir=str(aligner_dir),
                    download_forced_aligner=True,
                    token=None,
                )

        self.assertEqual(str(model_dir.resolve()), payload["model_path"])
        self.assertEqual(str(aligner_dir.resolve()), payload["forced_aligner_path"])
        self.assertEqual(2, download_mock.call_count)
        self.assertEqual(1, download_mock.call_args_list[0].kwargs["max_workers"])
        self.assertEqual(1, download_mock.call_args_list[1].kwargs["max_workers"])

    def test_run_qwen_training_job_builds_helper_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            helper_python = Path(tmp_dir) / "python"
            helper_python.write_text("", encoding="utf-8")
            with (
                patch.object(workflow, "qwen_helper_python", return_value=helper_python),
                patch.object(
                    workflow,
                    "run_command",
                    return_value=workflow.CommandResult(("cmd",), 0, "ok", ""),
                ) as run_command_mock,
            ):
                result = workflow.run_qwen_training_job(
                    train_file="/tmp/train.jsonl",
                    eval_file="/tmp/eval.jsonl",
                    output_dir="/tmp/out",
                    model_path="/tmp/model",
                    device="cpu",
                )

        self.assertEqual(0, result.returncode)
        command = run_command_mock.call_args.args[0]
        self.assertEqual(str(helper_python), command[0])
        self.assertIn("train-sft", command)
        self.assertIn("--device", command)
        self.assertIn("/tmp/model", command)


if __name__ == "__main__":
    unittest.main()
