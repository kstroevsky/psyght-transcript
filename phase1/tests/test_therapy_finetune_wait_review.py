"""Tests for blocking review intake in therapy_finetune CLI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phase1.tools import therapy_finetune as cli


def _write_template(path: Path, *, segments, chunk_id: str | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "chunk_id": chunk_id or path.stem,
                "audio_path": f"/tmp/{path.stem}.wav",
                "start_sec": 0.0,
                "end_sec": 600.0,
                "instructions": "fill segments",
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class TherapyFinetuneWaitReviewTest(unittest.TestCase):
    def test_select_review_template_returns_first_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            review_dir = Path(tmp_dir)
            _write_template(review_dir / "chunk_000.json", segments=[])
            _write_template(
                review_dir / "chunk_001.json",
                segments=[{"speaker": "A", "start_sec": 10.0, "end_sec": 20.0, "text": "done"}],
            )

            selected = cli.select_review_template(review_dir)

        self.assertEqual((review_dir / "chunk_000.json").resolve(), selected)

    def test_select_review_template_respects_explicit_chunk_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            review_dir = Path(tmp_dir)
            _write_template(review_dir / "chunk_000.json", segments=[])
            _write_template(review_dir / "chunk_001.json", segments=[])

            selected = cli.select_review_template(review_dir, chunk_id="chunk_001")

        self.assertEqual((review_dir / "chunk_001.json").resolve(), selected)

    def test_validate_review_template_reports_missing_file(self) -> None:
        payload, error = cli.validate_review_template("/tmp/does-not-exist-review.json")

        self.assertIsNone(payload)
        self.assertIn("not found", str(error))

    def test_validate_review_template_reports_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "chunk_000.json"
            template_path.write_text("{not json", encoding="utf-8")

            payload, error = cli.validate_review_template(template_path)

        self.assertIsNone(payload)
        self.assertIn("invalid", str(error))

    def test_validate_review_template_reports_empty_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "chunk_000.json"
            _write_template(template_path, segments=[])

            payload, error = cli.validate_review_template(template_path)

        self.assertIsNone(payload)
        self.assertIn("empty", str(error))

    def test_validate_review_template_reports_invalid_segment_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "chunk_000.json"
            _write_template(template_path, segments=[{"speaker": "A", "text": ""}])

            payload, error = cli.validate_review_template(template_path)

        self.assertIsNone(payload)
        self.assertIn("invalid", str(error))

    def test_wait_for_review_template_retries_until_valid_and_rewrites_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            review_dir = Path(tmp_dir)
            template_path = review_dir / "chunk_000.json"
            _write_template(template_path, segments=[])

            prompts: list[str] = []
            messages: list[str] = []
            attempt = {"count": 0}

            def fake_input(prompt: str) -> str:
                prompts.append(prompt)
                attempt["count"] += 1
                if attempt["count"] == 2:
                    template_path.write_text("{broken", encoding="utf-8")
                if attempt["count"] == 3:
                    _write_template(
                        template_path,
                        segments=[
                            {"speaker": "B", "start_sec": 30.0, "end_sec": 35.0, "text": "later"},
                            {"speaker": "A", "start_sec": 10.0, "end_sec": 20.0, "text": "earlier"},
                        ],
                    )
                return ""

            accepted_path = cli.wait_for_review_template(
                review_dir,
                input_fn=fake_input,
                print_fn=lambda *parts: messages.append(" ".join(str(part) for part in parts)),
            )

            payload = json.loads(template_path.read_text(encoding="utf-8"))

        self.assertEqual(template_path.resolve(), accepted_path)
        self.assertEqual(3, len(prompts))
        self.assertTrue(any("empty" in message for message in messages))
        self.assertTrue(any("invalid" in message for message in messages))
        self.assertEqual(
            [
                {"speaker": "A", "start_sec": 10.0, "end_sec": 20.0, "text": "earlier"},
                {"speaker": "B", "start_sec": 30.0, "end_sec": 35.0, "text": "later"},
            ],
            payload["segments"],
        )


if __name__ == "__main__":
    unittest.main()
