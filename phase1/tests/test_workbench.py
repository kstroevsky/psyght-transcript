"""Tests for the experiment workbench: manifest expansion, leaderboard, runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phase1.workbench import build_leaderboard, load_manifest, run_manifest
from phase1.workbench.manifest import ExperimentManifest


def _manifest_in(directory: Path, body: str, *, with_ref: bool = True) -> Path:
    if with_ref:
        (directory / "ref.json").write_text(
            json.dumps([{"speaker": "A", "start_sec": 0.0, "text": "привет мир"}], ensure_ascii=False),
            encoding="utf-8",
        )
    path = directory / "m.yaml"
    path.write_text(body, encoding="utf-8")
    return path


_BASE = """\
id: exp
audio: dummy.wav
reference: ref.json
profile: default
defaults:
  pipeline: {language: ru, duration_limit: 600}
  backend: {id: gigaam_ctc, options: {revision: e2e_ctc}}
variants:
  - id: greedy
    backend: {options: {decoder: {strategy: greedy}}}
  - id: beam
    extends: greedy
    backend: {options: {decoder: {strategy: beam, beam_width: 32}}}
"""


class ManifestExpansionTest(unittest.TestCase):
    def test_defaults_and_extends_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(_manifest_in(Path(tmp), _BASE))
            self.assertEqual(2, len(manifest.presets))
            greedy, beam = manifest.presets
            self.assertEqual("greedy", greedy.preset_id)
            self.assertEqual("ru", greedy.pipeline.language)  # from defaults
            self.assertEqual(600, greedy.pipeline.duration_limit)
            # beam inherits greedy's backend options then overrides the strategy
            self.assertEqual("beam", beam.backend.options["decoder"]["strategy"])
            self.assertEqual(32, beam.backend.options["decoder"]["beam_width"])
            self.assertEqual("e2e_ctc", beam.backend.options["revision"])  # from defaults

    def test_requires_exactly_one_of_audio_or_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            body = _BASE.replace("audio: dummy.wav\n", "audio: dummy.wav\ncorpus: c.yaml\n")
            with self.assertRaises(ValueError):
                load_manifest(_manifest_in(Path(tmp), body))

    def test_rejects_cyclic_extends(self) -> None:
        body = (
            "id: exp\naudio: a.wav\nvariants:\n"
            "  - id: x\n    extends: y\n  - id: y\n    extends: x\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_manifest(_manifest_in(Path(tmp), body, with_ref=False))

    def test_rejects_unknown_variant_keys(self) -> None:
        body = "id: exp\naudio: a.wav\nvariants:\n  - id: x\n    bogus: 1\n"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_manifest(_manifest_in(Path(tmp), body, with_ref=False))


class LeaderboardTest(unittest.TestCase):
    def test_ranks_by_wer_then_aggregates_across_items(self) -> None:
        rows = [
            {"variant": "a", "item_id": "i1", "status": "completed", "wer": 0.2, "cer": 0.1, "realtime_factor": 1.0, "fallback_count": 0},
            {"variant": "a", "item_id": "i2", "status": "completed", "wer": 0.4, "cer": 0.3, "realtime_factor": 1.0, "fallback_count": 1},
            {"variant": "b", "item_id": "i1", "status": "completed", "wer": 0.1, "cer": 0.1, "realtime_factor": 2.0, "fallback_count": 0},
            {"variant": "b", "item_id": "i2", "status": "completed", "wer": 0.1, "cer": 0.1, "realtime_factor": 2.0, "fallback_count": 0},
        ]
        board = build_leaderboard(rows)
        self.assertEqual("b", board[0]["variant"])  # lower mean WER wins
        self.assertEqual(0.1, board[0]["macro_wer"])
        a_row = next(r for r in board if r["variant"] == "a")
        self.assertEqual(0.3, a_row["macro_wer"])  # mean(0.2, 0.4)
        self.assertEqual(1, a_row["fallbacks"])


class RunnerTest(unittest.TestCase):
    def _fake_run_experiment(self, **kwargs):
        runs = []
        for preset in kwargs["presets"]:
            run_dir = Path(kwargs["output_dir"]) / preset.preset_id
            run_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = run_dir / "out.json"
            # variant 'beam' transcribes perfectly; 'greedy' makes one error.
            text = "привет мир" if preset.preset_id == "beam" else "привет"
            transcript_path.write_text(
                json.dumps({"language": "ru", "duration": 2.0, "segments": [{"speaker": "A", "start": 0.0, "end": 2.0, "text": text}]}),
                encoding="utf-8",
            )
            runs.append(
                {
                    "preset_id": preset.preset_id,
                    "status": "completed",
                    "artifacts_dir": str(run_dir),
                    "output_json": str(transcript_path),
                    "performance": {"realtime_factor": 1.0, "wall_clock_sec": 3.0},
                    "metrics": {"proxy_quality_score": 0.8},
                    "stability": {"fallback_count": 0},
                }
            )
        return {"experiment_id": "exp_1", "runs": runs}

    def test_run_manifest_scores_and_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _manifest_in(root, _BASE)
            out = root / "out"
            body = manifest_path.read_text() + f"output_dir: {out}\n"
            manifest_path.write_text(body, encoding="utf-8")
            manifest = load_manifest(manifest_path)
            result = run_manifest(manifest, run_experiment=self._fake_run_experiment)
            board = {row["variant"]: row for row in result["leaderboard"]}
            self.assertEqual(0.0, board["beam"]["macro_wer"])  # perfect
            self.assertEqual(1, board["beam"]["rank"])
            self.assertGreater(board["greedy"]["macro_wer"], 0.0)
            self.assertTrue((out / "leaderboard.md").exists())
            self.assertTrue((out / "workbench_result.json").exists())

    def test_dry_run_does_not_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(_manifest_in(Path(tmp), _BASE))

            def explode(**kwargs):  # pragma: no cover - must not be called
                raise AssertionError("run_experiment must not be called on dry-run")

            result = run_manifest(manifest, dry_run=True, run_experiment=explode)
            self.assertTrue(result["dry_run"])
            self.assertEqual(2, len(result["manifest"]["variants"]))


if __name__ == "__main__":
    unittest.main()
