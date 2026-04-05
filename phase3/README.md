# Phase 3 (Planned): Redis + BullMQ Python Worker

## Goal
Run transcription jobs asynchronously from Redis/BullMQ, report progress, and persist outputs to PostgreSQL.

## Planned responsibilities
- Consume job payloads from BullMQ queue
- Execute pipeline (transcribe -> align -> diarize -> assign -> ingest)
- Emit progress updates back to BullMQ/NestJS
- Store final output in PostgreSQL via `phase2`

## Status
Not started.

## Initial queue payload contract
```json
{
  "jobId": "string",
  "audioPath": "/abs/path/file.mp3",
  "language": "ru|uk|en|null",
  "minSpeakers": null,
  "maxSpeakers": 4
}
```
