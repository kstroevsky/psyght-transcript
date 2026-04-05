-- BM25 supports keyword search over transcript text while keeping meeting/speaker fields filterable.
CREATE INDEX IF NOT EXISTS segments_text_bm25
    ON segments
    USING bm25 (id, text, speaker, meeting_id)
    WITH (key_field='id', text_fields='{"text": {"tokenizer": {"type": "default"}}}');

-- HNSW accelerates cosine-similarity search over multilingual-e5 embeddings.
CREATE INDEX IF NOT EXISTS segments_embedding_hnsw
    ON segments
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS segments_meeting_id_idx ON segments (meeting_id);
CREATE INDEX IF NOT EXISTS segments_speaker_idx ON segments (speaker);
CREATE INDEX IF NOT EXISTS segments_start_sec_idx ON segments (start_sec);
