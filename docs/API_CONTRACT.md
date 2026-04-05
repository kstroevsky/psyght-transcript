# API Contract (Planned)

This file mirrors the current `PROJECT_BRIEF.md` API surface and is the source for Phase 4 implementation.

## REST Endpoints
- `POST /transcriptions`
  - Input: `audioPath`, optional `language`, optional `minSpeakers`, optional `maxSpeakers`
  - Output: `jobId`
- `GET /transcriptions/:jobId/status`
  - Output: `state`, `progress` (0-100), optional `meetingId`, optional `error`
- `GET /meetings/:id/segments`
  - Query: `limit`, `offset`, optional `speaker`
  - Output: paginated transcript segments with timestamps and speaker labels
- `GET /search?q=`
  - BM25 keyword search across all meetings, with optional meeting/speaker filters
- `GET /search/semantic?q=`
  - Vector similarity search across all meetings, with optional meeting/speaker filters

## WebSocket Events
- `transcription.progress`
  - Payload: `jobId`, `progress`, `state`
- `transcription.completed`
  - Payload: `jobId`, `meetingId`
- `transcription.failed`
  - Payload: `jobId`, `error`
