# Architecture (Target)

```text
Client
  | HTTP / WebSocket
  v
NestJS Backend (TypeScript)
  | BullMQ enqueue
  v
Redis 7
  | dequeue
  v
Python worker (WhisperX -> Pyannote -> e5)
  | persist
  v
PostgreSQL 17 + ParadeDB
```

## Core Components
- `contracts`: the only cross-phase shared surface; owns transcript dataclasses and JSON serializers
- `phase1`: local transcription primitives, compare-mode orchestration, backend adapters, progress/artifact persistence, and final output formatting
- `phase2`: DB schema, ingestion, embedding generation, and search helpers
- `phase3` (planned): Redis/BullMQ-connected worker runtime
- `phase4` (planned): NestJS API surface
- `phase5` (planned): WebSocket progress broadcast

## Internal boundaries
- `phase1/backends/*` owns pluggable transcription backend adapters. Implemented backends today are `whisperx`, `gigaam_ctc`, `canary`, and `therapy_hybrid`.
- `phase1/pipeline/*` owns stage wrappers for audio loading, diarization, and backend-independent speaker assignment.
- `phase1/therapy/*` owns the additive Russian therapy pipeline: Canary-first transcript structure, CTC anchoring, Whisper guardrails, Qwen merge, and ECLM data-prep helpers.
- `phase1/runtime/*` owns single-run orchestration, progress state, artifact persistence, and synthetic single-speaker handling.
- `phase1/quality/*` owns the quality meta-layer: independent quality checks, report assembly, and ranking shared by compare mode and decoder tuning.
- `phase1/compare_runtime/*` owns preset loading plus compare experiment orchestration only; it consumes `runtime/*` and `quality/*` but does not own scoring logic.
- `phase2/db/*` owns DB connections, row/query helpers, and repository-style operations.
- `phase2/embeddings/*` owns multilingual-e5 encoding only.

## Phase1 layering
- Backend layer: `phase1/backends/*`
- Stage layer: `phase1/pipeline/*`
- Therapy stage layer: `phase1/therapy/*`
- Application/runtime layer: `phase1/runtime/*`
- Meta/quality layer: `phase1/quality/*`
- Experiment layer: `phase1/compare_runtime/*`, `phase1/tuning/*`

Dependency direction is intentionally one-way:
`compare_runtime` / `tuning` -> `quality` -> `runtime` / `pipeline` / `backends`

Additional boundary notes:
- Compare preset JSON is normalized into typed pipeline policy objects before runtime options are built.
- The quality layer uses polymorphic check objects so new scoring methods can be added without modifying compare orchestration.
- The additive `therapy_hybrid` backend composes replaceable stage objects from `phase1/therapy/*` and still reuses the shared runtime/output contracts.

## Phase1 runtime notes
- Audio is decoded once, optionally clipped once, and the same waveform is reused across ASR, alignment, and diarization.
- `--duration-limit` therefore constrains diarization as well as ASR/alignment.
- Compare mode can reuse one decoded audio cache across multiple CPU-only presets.
- Device resolution is stage-specific on Apple Silicon:
  - ASR treats `mps` as unsupported for faster-whisper/CTranslate2 and falls back to CPU.
  - Alignment can try `mps`, but falls back to CPU if model load or inference fails.
  - Diarization defaults to CPU for stability; explicit `mps` is best-effort with CPU fallback.
- `phase1/runtime/options.py` writes additive run diagnostics into `00_run_meta.json` without changing transcript output formats.
- Compare experiments add top-level `experiment.json` and `summary.json`, while each preset sub-run keeps the same transcript and intermediate artifact contracts.

## Compatibility invariants
- Primary operator entrypoints are `phase1/download_models.py`, `phase1/tools/setup_ollama_qwen.py`, `phase1/transcribe.py`, `phase1/compare.py`, `phase1/tools/therapy_finetune.py`, and `phase2/ingest.py`.
- `phase1/compare.py` is additive and does not change `phase1/transcribe.py`.
- Transcript JSON/TXT output shape, progress-file stages, and DB schema remain unchanged.
- Cross-phase sharing is intentionally limited to `contracts` so later worker/API agents can operate with minimal context.
