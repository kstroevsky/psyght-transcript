"""Tests for the phase1 evaluation corpus (manifest + aggregate scoring)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contracts.transcript import Segment, Transcript
from phase1.eval import load_corpus, load_hypotheses_from_dir, score_corpus
from phase1.eval.corpus import _aggregate


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class CorpusLoadTest(unittest.TestCase):
    def _manifest(self, directory: Path, body: str) -> Path:
        manifest = directory / "corpus.yaml"
        manifest.write_text(body, encoding="utf-8")
        return manifest

    def test_load_resolves_relative_paths_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "ref.json", [{"speaker": "A", "start_sec": 0.0, "text": "привет мир"}])
            manifest = self._manifest(
                root,
                "name: t\nprofile: ru_fold\ndefaults:\n  language: ru\nitems:\n  - id: one\n    reference: ref.json\n",
            )
            corpus = load_corpus(manifest)
            self.assertEqual("t", corpus.name)
            self.assertEqual("ru_fold", corpus.profile.name)
            self.assertEqual(1, len(corpus.items))
            self.assertEqual("ru", corpus.items[0].language)
            self.assertTrue(corpus.items[0].reference.is_absolute())

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "ref.json", [{"speaker": "A", "start_sec": 0.0, "text": "x"}])
            manifest = self._manifest(
                root,
                "name: t\nitems:\n  - id: dup\n    reference: ref.json\n  - id: dup\n    reference: ref.json\n",
            )
            with self.assertRaises(ValueError):
                load_corpus(manifest)

    def test_rejects_unknown_item_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "ref.json", [{"speaker": "A", "start_sec": 0.0, "text": "x"}])
            manifest = self._manifest(
                root,
                "name: t\nitems:\n  - id: one\n    reference: ref.json\n    bogus: 1\n",
            )
            with self.assertRaises(ValueError):
                load_corpus(manifest)


class CorpusScoreTest(unittest.TestCase):
    def _hyp(self, text: str) -> Transcript:
        return Transcript(language="ru", duration=2.0, segments=[Segment(speaker="A", start=0.0, end=2.0, text=text)])

    def test_score_corpus_reports_missing_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "ref.json", [{"speaker": "A", "start_sec": 0.0, "text": "привет мир"}])
            manifest = root / "corpus.yaml"
            manifest.write_text(
                "name: t\nprofile: default\nitems:\n  - id: one\n    reference: ref.json\n  - id: two\n    reference: ref.json\n",
                encoding="utf-8",
            )
            corpus = load_corpus(manifest)
            result = score_corpus(corpus, {"one": self._hyp("привет мир")})
            self.assertEqual(2, result["n_items"])
            self.assertEqual(1, result["n_scored"])
            self.assertEqual(["two"], result["missing"])
            self.assertEqual(0.0, result["aggregate"]["macro_wer"])

    def test_load_hypotheses_from_dir_matches_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "ref.json", [{"speaker": "A", "start_sec": 0.0, "text": "привет"}])
            manifest = root / "corpus.yaml"
            manifest.write_text("name: t\nitems:\n  - id: one\n    reference: ref.json\n", encoding="utf-8")
            hyp_dir = root / "runs"
            hyp_dir.mkdir()
            _write(
                hyp_dir / "one.json",
                {"language": "ru", "duration": 1.0, "segments": [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "привет"}]},
            )
            corpus = load_corpus(manifest)
            hypotheses = load_hypotheses_from_dir(corpus, hyp_dir)
            self.assertIn("one", hypotheses)
            self.assertEqual(0.0, score_corpus(corpus, hypotheses)["items"][0]["wer"])

    def test_micro_aggregate_pools_edits(self) -> None:
        per_item = [
            {"wer": 0.5, "cer": 0.0, "wer_detail": {"substitutions": 1, "deletions": 0, "insertions": 0, "reference_length": 2}, "cer_detail": {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_length": 4}},
            {"wer": 0.0, "cer": 0.0, "wer_detail": {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_length": 8}, "cer_detail": {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_length": 16}},
        ]
        aggregate = _aggregate(per_item)
        self.assertEqual(0.25, aggregate["macro_wer"])  # mean(0.5, 0.0)
        self.assertEqual(0.1, aggregate["micro_wer"])  # 1 edit / 10 ref words


if __name__ == "__main__":
    unittest.main()
