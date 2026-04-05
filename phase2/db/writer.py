"""Database writer for persisting meetings and segment embeddings into phase2 schema."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from contracts.transcript import Transcript
from phase2.db.connection import get_conn
from phase2.db.rows import build_segment_rows
from phase2.embeddings.encoder import encode_passages


def insert_meeting(
    transcript: Transcript,
    audio_path: str,
    recorded_at: datetime | None = None,
) -> str:
    """Insert one meeting row and all transcript segments in transcript order."""

    meeting_id = str(uuid.uuid4())
    texts = [segment.text for segment in transcript.segments]
    print(f"[DB] Encoding {len(texts)} segments")
    vectors = encode_passages(texts) if texts else []
    rows = build_segment_rows(meeting_id, transcript, vectors)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO meetings (id, recorded_at, language, duration, audio_path)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    meeting_id,
                    recorded_at or datetime.now(timezone.utc),
                    transcript.language,
                    transcript.duration,
                    audio_path,
                ),
            )
            if rows:
                cur.executemany(
                    """
                    INSERT INTO segments
                        (meeting_id, speaker, start_sec, end_sec, text, words, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
        conn.commit()

    print(f"[DB] Inserted meeting {meeting_id} with {len(texts)} segments")
    return meeting_id
