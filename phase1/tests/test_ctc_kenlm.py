"""Tests for CTC beam-search runtime resolution and text cleanup."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from phase1.backends.gigaam_ctc_backend import GigaAMCTCBackend
from phase1.backends.ctc_kenlm import (
    CTC_BLANK_LABEL,
    CTC_UNK_LABEL,
    build_ctc_decoder_labels,
    cleanup_decoded_text,
    ctc_kenlm_helper_python,
    ctc_kenlm_helper_script,
    resolve_ctc_decoder_runtime,
)


class CTCKenLMRuntimeTest(unittest.TestCase):
    """Verify decoder defaults, fallbacks, and punctuation cleanup."""

    def test_cleanup_decoded_text_strips_extra_spaces_around_punctuation(self) -> None:
        self.assertEqual("Привет, мир!", cleanup_decoded_text("  Привет ,   мир !  "))

    def test_ru_defaults_to_greedy_even_when_helper_and_lm_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "models"
            helper_python = Path(tmp_dir) / "helper-python"
            helper_script = Path(tmp_dir) / "helper-script.py"
            helper_python.write_text("", encoding="utf-8")
            helper_script.write_text("", encoding="utf-8")
            lm_dir = root / "ru" / "ruwiki_plus_4gram"
            lm_dir.mkdir(parents=True)
            (lm_dir / "lm.binary").write_text("lm", encoding="utf-8")
            (lm_dir / "vocab.txt").write_text("слово\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "PHASE1_CTC_KENLM_MODEL_ROOT": str(root),
                    "PHASE1_CTC_KENLM_PYTHON": str(helper_python),
                    "PHASE1_CTC_KENLM_HELPER_SCRIPT": str(helper_script),
                },
                clear=False,
            ):
                runtime = resolve_ctc_decoder_runtime({}, "ru")

        self.assertEqual("greedy", runtime["strategy"])
        self.assertFalse(runtime["lm"]["enabled"])
        self.assertIsNone(runtime["fallback_reason"])

    def test_explicit_beam_with_missing_ru_lm_falls_back_to_greedy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "models"
            helper_python = Path(tmp_dir) / "helper-python"
            helper_script = Path(tmp_dir) / "helper-script.py"
            helper_python.write_text("", encoding="utf-8")
            helper_script.write_text("", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "PHASE1_CTC_KENLM_MODEL_ROOT": str(root),
                    "PHASE1_CTC_KENLM_PYTHON": str(helper_python),
                    "PHASE1_CTC_KENLM_HELPER_SCRIPT": str(helper_script),
                },
                clear=False,
            ):
                runtime = resolve_ctc_decoder_runtime({"decoder": {"strategy": "beam"}}, "ru")

        self.assertEqual("greedy", runtime["strategy"])
        self.assertIn("KenLM binary was not found", runtime["fallback_reason"])

    def test_explicit_greedy_disables_beam_path(self) -> None:
        runtime = resolve_ctc_decoder_runtime({"decoder": {"strategy": "greedy"}}, "ru")

        self.assertEqual("greedy", runtime["strategy"])
        self.assertEqual("greedy", runtime["strategy_requested"])
        self.assertFalse(runtime["lm"]["enabled"])

    def test_explicit_beam_without_lm_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            helper_python = Path(tmp_dir) / "helper-python"
            helper_script = Path(tmp_dir) / "helper-script.py"
            helper_python.write_text("", encoding="utf-8")
            helper_script.write_text("", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "PHASE1_CTC_KENLM_PYTHON": str(helper_python),
                    "PHASE1_CTC_KENLM_HELPER_SCRIPT": str(helper_script),
                },
                clear=False,
            ):
                runtime = resolve_ctc_decoder_runtime({"decoder": {"strategy": "beam", "lm": {"enabled": False}}}, "ru")

        self.assertEqual("beam", runtime["strategy"])
        self.assertFalse(runtime["lm"]["enabled"])
        self.assertIsNone(runtime["fallback_reason"])

    def test_explicit_zero_lm_alpha_and_beta_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "models"
            helper_python = Path(tmp_dir) / "helper-python"
            helper_script = Path(tmp_dir) / "helper-script.py"
            helper_python.write_text("", encoding="utf-8")
            helper_script.write_text("", encoding="utf-8")
            lm_dir = root / "ru" / "ruwiki_plus_4gram"
            lm_dir.mkdir(parents=True)
            (lm_dir / "lm.binary").write_text("lm", encoding="utf-8")
            (lm_dir / "vocab.txt").write_text("слово\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "PHASE1_CTC_KENLM_MODEL_ROOT": str(root),
                    "PHASE1_CTC_KENLM_PYTHON": str(helper_python),
                    "PHASE1_CTC_KENLM_HELPER_SCRIPT": str(helper_script),
                },
                clear=False,
            ):
                runtime = resolve_ctc_decoder_runtime(
                    {
                        "decoder": {
                            "strategy": "beam",
                            "beam_width": 32,
                            "lm": {
                                "alpha": 0.0,
                                "beta": 0.0,
                                "unk_score_offset": 0.0,
                            },
                        }
                    },
                    "ru",
                )

        self.assertEqual("beam", runtime["strategy"])
        self.assertEqual(32, runtime["beam_width"])
        self.assertEqual(0.0, runtime["lm"]["alpha"])
        self.assertEqual(0.0, runtime["lm"]["beta"])
        self.assertEqual(0.0, runtime["lm"]["unk_score_offset"])

    def test_build_ctc_decoder_labels_appends_explicit_blank(self) -> None:
        labels = build_ctc_decoder_labels(["<unk>", "▁", "а"], 4)

        self.assertEqual([CTC_UNK_LABEL, "▁", "а", CTC_BLANK_LABEL], labels)

    def test_build_ctc_decoder_labels_rejects_mismatched_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cannot align"):
            build_ctc_decoder_labels(["<unk>", "▁", "а"], 6)

    def test_cleanup_decoded_text_removes_sentencepiece_special_tokens(self) -> None:
        self.assertEqual("Привет, мир!", cleanup_decoded_text(f"▁ Привет , {CTC_UNK_LABEL} мир !"))

    @unittest.skipUnless(ctc_kenlm_helper_python().exists(), "CTC KenLM helper env is required for decode smoke tests.")
    @unittest.skipUnless(ctc_kenlm_helper_script().exists(), "CTC KenLM helper script is required for decode smoke tests.")
    def test_helper_beam_without_lm_matches_synthetic_greedy_decode(self) -> None:
        labels = ["<unk>", "▁", "п", "р", "и", "в", "е", "т"]
        blank_index = len(labels)
        token_ids = [1, 2, 2, blank_index, 3, 4, 5, 6, 7, blank_index]
        logits = np.full((len(token_ids), blank_index + 1), -20.0, dtype=np.float32)
        for frame_index, token_id in enumerate(token_ids):
            logits[frame_index, token_id] = 0.0

        greedy_ids: list[int] = []
        previous = None
        for token_id in token_ids:
            if token_id == previous:
                continue
            previous = token_id
            if token_id == blank_index:
                continue
            greedy_ids.append(token_id)
        expected = "".join(labels[token_id] for token_id in greedy_ids).replace("▁", " ").replace("<unk>", " ").strip()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            labels_path = tmp_path / "labels.json"
            logits_path = tmp_path / "logits.npy"
            manifest_path = tmp_path / "manifest.json"
            output_path = tmp_path / "output.json"
            labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
            np.save(logits_path, logits, allow_pickle=False)
            manifest_path.write_text(json.dumps({"paths": [str(logits_path)]}), encoding="utf-8")
            subprocess.run(
                [
                    str(ctc_kenlm_helper_python()),
                    str(ctc_kenlm_helper_script()),
                    "--manifest",
                    str(manifest_path),
                    "--labels-json",
                    str(labels_path),
                    "--output",
                    str(output_path),
                    "--beam-width",
                    "32",
                ],
                check=True,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(expected, payload["texts"][0].strip())

    def test_chunk_logits_runs_decoder_layers_inside_inference_mode(self) -> None:
        tracker = {"active": False}

        @contextmanager
        def inference_mode():
            tracker["active"] = True
            try:
                yield
            finally:
                tracker["active"] = False

        fake_torch = types.SimpleNamespace(inference_mode=inference_mode)

        class FakeTensor:
            def __init__(self, array: np.ndarray) -> None:
                self._array = np.asarray(array)

            def transpose(self, dim0: int, dim1: int) -> "FakeTensor":
                return FakeTensor(np.swapaxes(self._array, dim0, dim1))

            def __getitem__(self, key) -> "FakeTensor":
                return FakeTensor(self._array[key])

            def detach(self) -> "FakeTensor":
                return self

            def cpu(self) -> "FakeTensor":
                return self

            def float(self) -> "FakeTensor":
                return self

            def numpy(self) -> np.ndarray:
                return np.asarray(self._array, dtype=np.float32)

        class FakeScalar:
            def __init__(self, value: int) -> None:
                self._value = value

            def item(self) -> int:
                return self._value

        class FakeDecoder:
            def decoder_layers(self, encoded) -> FakeTensor:
                if not tracker["active"]:
                    raise AssertionError("decoder_layers must run inside torch.inference_mode()")
                return FakeTensor(np.arange(12, dtype=np.float32).reshape(1, 3, 4))

        class FakeInnerModel:
            def __init__(self) -> None:
                self.head = FakeDecoder()

            def prepare_wav(self, path: str) -> tuple[str, str]:
                return path, "length"

            def forward(self, wav: str, length: str) -> tuple[str, list[FakeScalar]]:
                return "encoded", [FakeScalar(4)]

        fake_model = types.SimpleNamespace(model=FakeInnerModel())
        backend = GigaAMCTCBackend()

        with patch.dict(sys.modules, {"torch": fake_torch}), patch(
            "phase1.backends.gigaam_ctc_backend.write_wav_mono"
        ):
            logits = backend._chunk_logits(fake_model, np.zeros(8, dtype=np.float32))

        self.assertEqual((4, 3), logits.shape)


if __name__ == "__main__":
    unittest.main()
