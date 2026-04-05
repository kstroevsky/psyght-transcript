"""Pure SQL builders for phase2 keyword and semantic search helpers."""

from __future__ import annotations


def build_keyword_search_query(
    query: str,
    limit: int = 20,
    meeting_id: str | None = None,
) -> tuple[str, list[object]]:
    """Build the ParadeDB BM25 query while keeping optional meeting scoping explicit."""

    where = "WHERE s.text @@@ %s"
    params: list[object] = [query]
    if meeting_id:
        where += " AND s.meeting_id = %s"
        params.append(meeting_id)
    params.append(limit)
    sql = f"""
        SELECT s.meeting_id, s.speaker, s.start_sec, s.end_sec, s.text,
               paradedb.score(s.id) AS score
        FROM segments s
        {where}
        ORDER BY score DESC
        LIMIT %s
    """
    return sql, params


def build_semantic_search_query(
    embedding: list[float],
    limit: int = 20,
    meeting_id: str | None = None,
) -> tuple[str, list[object]]:
    """Build the pgvector similarity query with the same optional meeting filter."""

    where = "WHERE TRUE"
    params: list[object] = [embedding]
    if meeting_id:
        where += " AND meeting_id = %s"
        params.append(meeting_id)
    params.append(limit)
    sql = f"""
        SELECT meeting_id, speaker, start_sec, end_sec, text,
               1 - (embedding <=> %s::vector) AS similarity
        FROM segments
        {where}
        ORDER BY similarity DESC
        LIMIT %s
    """
    return sql, params
