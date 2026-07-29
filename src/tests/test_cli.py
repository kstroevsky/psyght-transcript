"""Tests for the public phase1 CLI entrypoint."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from phase1 import transcribe as cli


class CliTest(unittest.TestCase):
    """Verify that the CLI preserves flag semantics while dispatching into the runner."""

    def test_no_condition_flag_disables_previous_text_conditioning(self) -> None:
        with (
            patch.object(cli, "run") as run_mock,
            patch.object(sys, "argv", ["transcribe.py", "/tmp/audio.mp3", "--no-condition"]),
        ):
            cli.main()

        self.assertEqual(
            {"condition_on_previous_text": False},
            run_mock.call_args.kwargs["asr_options"],
        )

    def test_backend_flag_builds_canary_backend_spec(self) -> None:
        with (
            patch.object(cli, "run") as run_mock,
            patch.object(sys, "argv", ["transcribe.py", "/tmp/audio.mp3", "--backend", "canary", "--lang", "ru"]),
        ):
            cli.main()

        backend = run_mock.call_args.kwargs["backend"]
        self.assertEqual("canary", backend.id)
        self.assertEqual({}, backend.options)

    def test_therapy_flags_build_backend_options(self) -> None:
        with (
            patch.object(cli, "run") as run_mock,
            patch.object(
                sys,
                "argv",
                [
                    "transcribe.py",
                    "/tmp/audio.mp3",
                    "--backend",
                    "therapy_hybrid",
                    "--lang",
                    "ru",
                    "--use-live-canary",
                    "--merge-provider",
                    "rule_based",
                    "--merge-instruction-variant",
                    "consensus_gate_v1",
                    "--eclm-model",
                    "/tmp/eclm",
                    "--eclm-device",
                    "mps",
                ],
            ),
        ):
            cli.main()

        backend = run_mock.call_args.kwargs["backend"]
        self.assertEqual("therapy_hybrid", backend.id)
        self.assertTrue(backend.options["use_live_canary"])
        self.assertEqual("rule_based", backend.options["merge_provider"]["id"])
        self.assertEqual("consensus_gate_v1", backend.options["merge_provider"]["instruction_variant"])
        self.assertEqual("/tmp/eclm", backend.options["eclm_model_path"])
        self.assertEqual("mps", backend.options["eclm_device"])

    def test_qwen_backend_is_not_advertised(self) -> None:
        self.assertNotIn("qwen_asr", cli.list_backend_ids())


if __name__ == "__main__":
    unittest.main()
