"""Tests for the public compare CLI entrypoint."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from phase1 import compare as cli


class CompareCliTest(unittest.TestCase):
    def test_backend_flags_create_simple_presets_without_json_files(self) -> None:
        with (
            patch.object(cli, "run_experiment", return_value={"runs": []}) as run_experiment_mock,
            patch.object(sys, "argv", ["compare.py", "/tmp/audio.mp3", "--backend", "canary", "--backend", "therapy_hybrid"]),
        ):
            cli.main()

        presets = run_experiment_mock.call_args.kwargs["presets"]
        self.assertEqual(["canary", "therapy_hybrid"], [preset.backend.id for preset in presets])


if __name__ == "__main__":
    unittest.main()
