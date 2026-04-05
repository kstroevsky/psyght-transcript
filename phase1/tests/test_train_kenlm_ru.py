"""Tests for the RU KenLM trainer weighting helpers."""

from __future__ import annotations

import bz2
import tempfile
import unittest
from pathlib import Path

from phase1.tools.train_kenlm_ru import CorpusGroup, _build_corpus_groups, _write_corpus


class TrainKenLMRUTest(unittest.TestCase):
    """Verify weighted source-group metadata and legacy extras."""

    def test_build_corpus_groups_preserves_legacy_extra_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            extra_path = Path(tmp_dir) / "extra.txt"
            extra_path.write_text("hello", encoding="utf-8")

            groups = _build_corpus_groups([], [str(extra_path)])

        self.assertEqual(1, len(groups))
        self.assertEqual("extra", groups[0].label)
        self.assertEqual(1, groups[0].weight)

    def test_write_corpus_records_source_group_weights_and_counts(self) -> None:
        wiki_xml = """
        <mediawiki>
          <page>
            <title>Test</title>
            <ns>0</ns>
            <revision>
              <text>Привет мир. Пока.</text>
            </revision>
          </page>
        </mediawiki>
        """.strip()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dump_path = tmp_path / "wiki.xml.bz2"
            corpus_path = tmp_path / "corpus.txt"
            spoken_path = tmp_path / "spoken.txt"
            spoken_path.write_text("Ну привет. Ну пока.", encoding="utf-8")
            with bz2.open(dump_path, "wb") as handle:
                handle.write(wiki_xml.encode("utf-8"))
            dump_path.with_name(dump_path.name + ".complete").write_text("cached", encoding="utf-8")

            stats = _write_corpus(
                "https://example.invalid/wiki.xml.bz2",
                dump_path,
                1,
                [CorpusGroup(label="spoken", weight=3, source_paths=[spoken_path])],
                corpus_path,
                stream_wiki=False,
                max_pages=None,
                max_extra_lines=None,
            )

        source_groups = {row["label"]: row for row in stats["source_groups"]}
        self.assertEqual(2, source_groups["wiki"]["raw_sentences"])
        self.assertEqual(2, source_groups["wiki"]["weighted_sentences"])
        self.assertEqual(2, source_groups["spoken"]["raw_sentences"])
        self.assertEqual(6, source_groups["spoken"]["weighted_sentences"])


if __name__ == "__main__":
    unittest.main()
