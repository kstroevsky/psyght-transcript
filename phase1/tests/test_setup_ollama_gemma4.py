"""Tests for the Gemma4-specific Ollama bootstrap wrapper."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from phase1.tools import setup_ollama_gemma4 as cli


class SetupOllamaGemma4Test(unittest.TestCase):
    def test_build_parser_uses_gemma4_defaults(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([])

        self.assertEqual("phase1-ollama-gemma4", args.container_name)
        self.assertEqual("gemma4-27b", args.model)

    def test_main_delegates_to_shared_setup(self) -> None:
        with (
            patch.object(cli.base_cli, "run_setup") as run_setup_mock,
            patch.object(sys, "argv", ["setup_ollama_gemma4.py"]),
        ):
            result = cli.main()

        self.assertEqual(0, result)
        self.assertEqual("gemma4-27b", run_setup_mock.call_args.args[0].model)


if __name__ == "__main__":
    unittest.main()
