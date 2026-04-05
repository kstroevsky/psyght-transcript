"""Tests for the Qwen3-ASR helper environment bootstrap."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase1.tools import setup_qwen_asr_helper as cli


class SetupQwenAsrHelperTest(unittest.TestCase):
    def test_main_rejects_python_3_13(self) -> None:
        with (
            patch.object(cli, "_python_version", return_value=(3, 13)),
            patch.object(sys, "argv", ["setup_qwen_asr_helper.py"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "Python 3.11 or 3.12"):
                cli.main()

    def test_main_creates_venv_and_installs_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            requirements = Path(tmp_dir) / "requirements.txt"
            requirements.write_text("qwen-asr\n", encoding="utf-8")
            with (
                patch.object(cli, "_python_version", return_value=(3, 12)),
                patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess(["python"], 0, "", "")) as run_mock,
                patch.object(
                    sys,
                    "argv",
                    [
                        "setup_qwen_asr_helper.py",
                        "--python",
                        "/usr/bin/python3.12",
                        "--venv-dir",
                        str(Path(tmp_dir) / ".venv"),
                        "--requirements",
                        str(requirements),
                    ],
                ),
            ):
                cli.main()

        commands = [call.args[0] for call in run_mock.call_args_list]
        self.assertEqual(3, len(commands))
        self.assertEqual(["/usr/bin/python3.12", "-m", "venv", str(Path(tmp_dir) / ".venv")], commands[0])
        self.assertEqual([str(Path(tmp_dir) / ".venv" / "bin" / "pip"), "install", "--upgrade", "pip"], commands[1])
        self.assertEqual(
            [str(Path(tmp_dir) / ".venv" / "bin" / "pip"), "install", "-r", str(requirements)],
            commands[2],
        )


if __name__ == "__main__":
    unittest.main()
