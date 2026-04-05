-- Core storage for completed meetings and their searchable transcript segments.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meetings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recorded_at TIMESTAMPTZ,
    language    VARCHAR(5) NOT NULL,
    duration    DOUBLE PRECISION NOT NULL,
    audio_path  TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Each transcript segment keeps speaker attribution, raw word timing JSON, and one embedding.
CREATE TABLE IF NOT EXISTS segments (
    id          BIGSERIAL PRIMARY KEY,
    meeting_id  UUID NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker     VARCHAR(30) NOT NULL,
    start_sec   DOUBLE PRECISION NOT NULL,
    end_sec     DOUBLE PRECISION NOT NULL,
    text        TEXT NOT NULL,
    words       JSONB,
    embedding   VECTOR(1024)
);
