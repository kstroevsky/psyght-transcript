"""Tests for phase1 audio helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase1.pipeline.audio import apply_audio_preprocessing, read_wav_mono, resolve_audio_preprocessing, write_wav_mono


class AudioHelperTest(unittest.TestCase):
    """Verify additive audio-preprocessing plumbing and WAV helpers."""

    def test_resolve_audio_preprocessing_accepts_string_and_object_specs(self) -> None:
        specs = resolve_audio_preprocessing(
            {
                "audio_preprocessing": [
                    "deepfilternet",
                    {"id": "deepfilternet", "post_filter": True},
                ]
            }
        )

        self.assertEqual("deepfilternet", specs[0]["id"])
        self.assertTrue(specs[1]["post_filter"])

    def test_apply_audio_preprocessing_without_requested_transforms_is_noop(self) -> None:
        audio = np.array([0.0, 0.1, -0.1], dtype=np.float32)

        processed, applied = apply_audio_preprocessing(audio, {})

        np.testing.assert_allclose(audio, processed)
        self.assertEqual([], applied)

    def test_write_and_read_wav_round_trip(self) -> None:
        audio = np.linspace(-0.5, 0.5, 320, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "sample.wav"
            write_wav_mono(wav_path, audio)
            restored = read_wav_mono(wav_path)

        np.testing.assert_allclose(audio, restored, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
