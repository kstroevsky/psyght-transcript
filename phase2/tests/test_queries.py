"""Tests for pure SQL builders used by phase2 search helpers."""

from __future__ import annotations

import unittest

from phase2.db.queries import build_keyword_search_query, build_semantic_search_query


class QueryBuilderTest(unittest.TestCase):
    """Verify SQL and params for keyword and semantic search with optional meeting scope."""

    def test_build_keyword_search_query_without_meeting_filter(self) -> None:
        sql, params = build_keyword_search_query("budget", limit=5)

        self.assertIn("WHERE s.text @@@ %s", sql)
        self.assertNotIn("s.meeting_id = %s", sql)
        self.assertEqual(["budget", 5], params)

    def test_build_keyword_search_query_with_meeting_filter(self) -> None:
        sql, params = build_keyword_search_query("budget", limit=5, meeting_id="meeting-1")

        self.assertIn("AND s.meeting_id = %s", sql)
        self.assertEqual(["budget", "meeting-1", 5], params)

    def test_build_semantic_search_query_without_meeting_filter(self) -> None:
        sql, params = build_semantic_search_query([0.1, 0.2], limit=7)

        self.assertIn("WHERE TRUE", sql)
        self.assertNotIn("meeting_id = %s", sql)
        self.assertEqual([[0.1, 0.2], 7], params)

    def test_build_semantic_search_query_with_meeting_filter(self) -> None:
        sql, params = build_semantic_search_query([0.1, 0.2], limit=7, meeting_id="meeting-1")

        self.assertIn("AND meeting_id = %s", sql)
        self.assertEqual([[0.1, 0.2], "meeting-1", 7], params)


if __name__ == "__main__":
    unittest.main()
