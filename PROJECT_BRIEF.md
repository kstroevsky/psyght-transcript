# PROJECT_BRIEF

## Final Goal

Build a private, self-hosted meeting transcription and search service integrated into a larger NestJS backend. The system ingests audio recordings of meetings (each ~1 hour, in Russian, Ukrainian, or English), produces structured transcripts with per-speaker phrase separation and word-level timestamps, stores them in a searchable database, and exposes the results via a REST + WebSocket API.

The service must be fully operational offline after initial model download, GDPR-compliant (self-hosted in EU), and horizontally scalable at the transcription worker level.

---

## What the System Does

### Input
- Audio files: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.mp4`
- Optional hints: known language (`ru` / `uk` / `en`), expected speaker count
- Submitted on-demand via API call from the NestJS backend

### Processing Pipeline (Python worker)
1. Audio loading - validate format, resample to 16kHz
2. Language detection - auto-detect if not provided, route to best fine-tuned model
3. ASR - transcribe audio to text with phrase-level segments (WhisperX)
4. Alignment - refine to word-level timestamps (Wav2Vec2)
5. Diarization - identify and separate speakers (Pyannote 3.1 + CAM++)
6. Merge - assign each phrase and word to a speaker
7. Embedding - encode each segment to 1024-d vector (multilingual-e5-large)
8. Persist - write structured results to PostgreSQL

### Output (per meeting)
- Human-readable `.txt` transcript:
  ```
  [00:00:03.20 -> 00:00:07.80]  SPEAKER_00: Давай почнемо, є багато питань.
  [00:00:08.10 -> 00:00:12.40]  SPEAKER_01: Sure, let me share my screen.
  ```
- Structured `.json` with full word-level timestamps and speaker labels
- Persisted DB rows: one `meetings` record + N `segments` records with embeddings

### Storage & Search
- PostgreSQL 17 + ParadeDB (single self-hosted Docker container)
- BM25 keyword search (`pg_search`) across all transcript text, filterable by meeting or speaker
- Semantic / meaning-based search (`pgvector`, cosine similarity) for concept-level queries like "find all discussions about budget planning"
- Both search modes operate across all meetings simultaneously or scoped to one

### API Surface (NestJS)
- `POST /transcriptions` - submit audio file path, receive `jobId`
- `GET /transcriptions/:jobId/status` - poll job state + progress (0-100%)
- `GET /meetings/:id/segments` - paginated list of all phrases for a meeting
- `GET /search?q=` - BM25 keyword search
- `GET /search/semantic?q=` - vector similarity search
- WebSocket: real-time progress events pushed to client during processing

---

## Architecture

```text
Client
  | HTTP / WebSocket
  v
NestJS Backend (TypeScript)
  | BullMQ job enqueue
  v
Redis 7          <-------------------------.
  | job consumed                           |
  v                                        | progress updates
Python Transcription Worker                |
  (WhisperX -> Pyannote -> e5 embeddings)--'
  | writes results
  v
PostgreSQL 17 + ParadeDB
  (meetings + segments + BM25 + pgvector)
```

---

## Current Status
- Experiment workbench + evaluation layer landed on 2026-06-11:
  - new `phase1/eval/*` metric layer: scalable WER/CER with substitution/deletion/insertion breakdown (rapidfuzz-backed, guarded pure-Python fallback), speaker-attributed cpWER, diarization error rate (DER via `pyannote.metrics`), gold-reference loading for phase1-JSON / Gemini-list / plain-text formats, transcript-vs-reference diffs, and an evaluation-corpus abstraction; depends only on `contracts`
  - WER/CER are byte-identical to the previous quality-layer numbers for any non-empty reference (an empty reference now scores `1.0` instead of `0.0`, which cannot occur for a whole-transcript reference); `phase1/quality/checks.py` now reuses `phase1/eval`, removing the duplicated normalizer and the O(n*m) pure-Python Levenshtein
  - new `phase1/workbench/*` experiment layer: experiment manifests declare many pipeline variants with shared `defaults` and `extends` inheritance, each expanding into the same validated `ComparePreset` the compare runtime already consumes; a runner scores each variant against gold references and ranks them into a leaderboard
  - operator surface: `phase1/tools/eval_corpus.py` (model-free scoring of one transcript or a corpus), `phase1/tools/run_experiment_manifest.py` (`--dry-run` validate/preview, or run to a ranked leaderboard), and a `marimo` notebook `phase1/workbench/experiment_workbench.py` (runs gated behind explicit buttons) as the human interface
  - new optional deps `phase1/requirements-eval.txt` (rapidfuzz, pyannote.metrics) and `phase1/requirements-workbench.txt` (marimo); both layers import and run without them via fallbacks; starter corpus `phase1/eval/corpus.example.yaml` and example manifest `phase1/workbench/manifests/therapy_merge_compare.example.yaml`
  - normalization is profile-driven (`default`; `ru_fold` folds ё→е) and the profile is recorded with every score for reproducibility
  - verification: full `phase1/tests` suite passes (150 tests, up from 118) plus `py_compile` on all new and changed modules
  - leftover from the removed standalone Qwen-ASR backend remains on disk and is safe to delete: `phase1/.qwen-asr-venv` (~1.5 GB, git-ignored) and stale bytecode under `phase1/qwen_asr/`
  - next recommended step is to assemble a real held-out Russian gold corpus and run one manifest end-to-end to produce the first accuracy leaderboard, then validate whether `proxy_quality_score` tracks real WER on the gold
- Gemma4-27B merge support landed on 2026-04-09:
  - the therapy merge implementation is now named and documented as a generic local Ollama merge stage instead of a Qwen-only path; Qwen remains the default model, but local `gemma4-27b` is now a first-class supported alternative
  - added operator support via `phase1/tools/setup_ollama_gemma4.py`, `phase1/download_models.py --install-ollama-gemma4`, and `phase1/presets/therapy_hybrid_ru_gemma4_27b.json`
  - merge requests remain non-thinking and token-capped regardless of whether the selected local model is Qwen or Gemma4-27B
  - next recommended step is to run the new Gemma4-27B preset on reviewed Russian therapy windows and compare `01e_therapy_merge_debug.json` against the current Qwen baseline before changing defaults
- Bogomolov 10-minute Canary-first rerun completed on 2026-04-07 under `phase1/runs/bogomolov_canary_first_10min_20260407_retry2/`:
  - input was `/Users/kstroevsky/Downloads/2021-03-30 10.21.41 Zoom Meeting Vladimir Bogomolov.mp4` with `--duration-limit 600`; the run finished successfully with `wall_clock_sec=847.999`
  - live `therapy_hybrid` used WhisperX + GigaAM CTC + live Canary + diarization; Canary processed `20` helper chunks with `model_load_sec=61.271` and `inference_sec=494.368`
  - alignment again attempted `mps` first and fell back to CPU with `Output channels > 65536 not supported at the MPS device.`
  - the final transcript is still Canary-first and shows lexical rough edges in early long windows, so prompt wording alone is unlikely to resolve the remaining quality issues without broader evaluation or model-side changes
- Merge-instruction prompt comparison landed on 2026-04-07:
  - added `phase1/tools/compare_merge_instructions.py` plus named merge prompt variants `baseline_v1`, `priority_ladder_v1`, `consensus_gate_v1`, `token_preservation_v1`, and `verification_checklist_v1`
  - on the Bogomolov 10-minute rerun, all five instruction variants were executed on `5` divergent merge windows and wrote `artifacts/06_merge_instruction_compare_20260407_221605.json`
  - baseline, priority-ladder, token-preservation, and verification-checklist produced the same final accepted texts on the sampled windows; `consensus_gate_v1` was only more conservative about accepting LLM outputs and did not improve the final text
  - Ollama was unavailable in this session, so the comparison tool auto-fell back to local HF inference with `Qwen/Qwen2.5-0.5B-Instruct`; on this Apple Silicon host, that comparison path required CPU because `mps` hit `MPSTemporaryNDArray` size assertions
  - next recommended step is to keep `baseline_v1` (or `verification_checklist_v1`, which tied on the sampled windows) and expand the evaluation set with reviewed therapy segments before changing the production merge prompt
- Repo-sample Canary-first therapy run completed on 2026-04-07 under `phase1/runs/canary_first_repo_sample_20260407/`:
  - input was `phase1/test_first2min.mp3`; the run finished successfully with `wall_clock_sec=167.547`
  - live `therapy_hybrid` used WhisperX + GigaAM CTC + live Canary + diarization; Canary processed `4` helper chunks with `model_load_sec=81.896` and `inference_sec=40.877`
  - `01e_therapy_merge_debug.json` recorded `15/15` decisions as `source=canary_base`; `01f_therapy_eclm_debug.json` was present but empty, matching the current Canary-first Step 5B skip behavior
  - alignment first tried `mps` and fell back to CPU with `Output channels > 65536 not supported at the MPS device.`
  - final sample transcript wrote `15` speaker-tagged lines; next recommended step remains a representative long Russian therapy session plus merge-debug review because the short sample still shows lexical rough edges in the final text
- Canary-first `therapy_hybrid` landed on 2026-04-06:
  - live or artifact Canary is now the canonical transcript structure and base text for therapy runs; CTC remains the lexical truth anchor and Whisper is retained only as an auxiliary post-guard signal
  - the local Qwen merge prompt and deterministic fallback now preserve Canary by default and only apply conservative CTC-guided corrections instead of using the previous Whisper-primary merge behavior
  - the runtime still records `01f_therapy_eclm_debug.json`, but Step 5B is intentionally skipped in the Canary-first path until the ECLM model is retrained for the new inputs
  - next recommended step is to run the revised therapy pipeline on a representative long Russian session and review `01e_therapy_merge_debug.json` for real correction quality and duplicate-window cleanup
- Standalone Qwen3-ASR backend/workbench removal landed on 2026-04-06:
  - removed the dedicated `qwen_asr` backend, helper-env bootstrap, fine-tuning/UI tooling, download hooks, and focused tests
  - `phase1/transcribe.py` and `phase1/download_models.py` no longer expose standalone Qwen-ASR flags or bootstrap actions
  - the standalone Qwen-ASR path is no longer part of the supported operator surface; only the local text Qwen merge used inside `therapy_hybrid` remains
- Initial history cleanup landed on 2026-04-05:
  - the repo now has a curated commit history instead of one mixed uncommitted snapshot
  - dead internal modules `phase1/compare_runtime/metrics.py`, `phase1/output/schema.py`, `phase1/therapy/interfaces.py`, and `phase2/db/search.py` were removed
  - repo fixtures and operator docs are committed explicitly; ignore rules now target helper envs, local env files, model caches, run artifacts, and review chunks instead of broad `*.json` / `*.txt` patterns
  - removed unneeded local artifacts from the tree: `.DS_Store`, source `__pycache__`, `phase1/test_first2min.webm`, and `2021-03-30_09-39-42_Bogomolov_first_10min_0.txt`
  - next recommended step is to keep future runtime-generated assets out of git by extending the explicit path-based ignore policy rather than reintroducing broad extension ignores
- Fresh phase1 runtime bootstrap was repaired on 2026-03-24:
  - `phase1/requirements.txt` now includes `hydra-core`, which GigaAM trusted remote code requires during `AutoModel.from_pretrained`; without it, a clean `phase1/.venv` failed before CTC startup on the Bogomolov therapy rerun
- First-10-minute Bogomolov therapy full-stack rerun completed on 2026-03-24 under `phase1/runs/therapy_bogomolov_2021-03-30_10.21.41_first10min_fullstack_rerun2_20260324/`
  - input was `/Users/kstroevsky/Downloads/2021-03-30 10.21.41 Zoom Meeting Vladimir Bogomolov.mp4` with `--duration-limit 600`
  - runtime used live Ollama Qwen (`qwen3:8b`), live Canary (`nvidia/canary-1b-v2` on CPU, `20` chunks, `269.217s` inference), and the fine-tuned ECLM checkpoint at `phase1/models/therapy_eclm/segment56_mt5_large_v2` on CPU
  - run finished successfully with `wall_clock_sec=1341.956`; stage timings were `asr=1288.449s`, `alignment=46.150s`, `diarization=31.621s`
  - merge debug now contains `3` `whisper_semantic_override` selections; the previously bad `9.408-14.052` and `14.072-16.134` windows both resolved to Whisper in the final transcript
  - ECLM still evaluated `101` rows and accepted `0` corrections on this run; the merge-stage semantic override fixed the observed bad anchors without relying on Step 5B
  - next recommended step is to review the `3` override-tagged rows in `01e_therapy_merge_debug.json` and then decide whether to widen the heuristic slightly or keep it narrow and gather more therapy examples for ECLM retraining
- Additional therapy merge hardening landed on 2026-03-24:
  - short/high-overlap CTC anchor rules now allow a narrow Whisper semantic override when CTC is low-information or repetitively degraded while Whisper remains same-script
  - the same override is applied to the deterministic rule-based fallback, so Ollama failures no longer always force obviously bad CTC anchors
  - timestamp-pattern Whisper hallucination handling remains upstream in `phase1/therapy/hallucination.py`; the merge-stage override is intentionally text-only and conservative
  - focused verification passed: `python3 -m unittest phase1.tests.test_therapy_merge` and `python3 -m py_compile phase1/therapy/merge.py phase1/tests/test_therapy_merge.py`
  - broader follow-up is to rerun the Bogomolov first-10-minute therapy stack and inspect `01e_therapy_merge_debug.json` rows tagged with `whisper_semantic_override`
- `phase1` and `phase2` were internally refactored into smaller modules with a shared repo-root `contracts` package for transcript types/serialization.
- `phase1` now also includes:
  - `phase1/compare.py` for A/B-style multi-preset experiments against one audio file
  - `phase1/backends/*` for pluggable transcription backends
  - `phase1/compare_runtime/*` for preset loading, experiment orchestration, ranking, and optional reference scoring
- Additional phase1 quality-work landed on 2026-03-19:
  - preset-driven ASR hallucination-guard options for WhisperX/faster-whisper
  - preset-driven audio preprocessing via DeepFilterNet through an isolated helper environment at `phase1/.deepfilter-venv`
  - chronological sort + optional same-speaker merge in speaker assignment output
  - additive compare artifacts: saved preset manifests and `00_processed_audio.wav`
  - additive Russian `gigaam_ctc` backend for idea-5 style CTC experiments, still reusing WhisperX alignment
  - default phase1 backend switched to `gigaam_ctc` with `revision=e2e_ctc` (punctuated CTC output) and automatic WhisperX fallback for explicit non-Russian runs
  - `phase1/download_models.py` now prefetches default GigaAM model weights in addition to WhisperX/pyannote models
- Additional phase1 decoding work landed on 2026-03-21:
  - `gigaam_ctc` can now batch raw chunk logits through beam search + KenLM in an isolated `phase1/.ctc-kenlm-venv` helper while preserving the existing alignment/diarization/output contracts
  - RU runtime now defaults back to greedy decoding; beam search is opt-in and stays gated behind reference-backed tuning results
  - default self-trained Russian LM asset path remains `phase1/models/ctc_kenlm/ru/ruwiki_plus_4gram/`, but runtime no longer auto-enables it
  - added explicit SentencePiece-to-CTC label construction so the helper pins blank-token placement instead of relying on implicit `pyctcdecode` label expansion
  - added `phase1/tools/train_kenlm_ru.py` weighted corpus groups (`--corpus-group LABEL WEIGHT PATH`) for spoken-mix RU LM training in addition to the legacy flat `--extra-text` path
  - added `phase1/tools/tune_ctc_kenlm_ru.py` plus `phase1/tuning/ctc_kenlm_ru.py` for reference-backed RU decoder tuning across greedy, beam-without-LM, beam+external LM, and beam+self LM
  - added `phase1/tools/fetch_ctc_lm_assets.py` for packaged Ukrainian (`Yehor/kenlm-uk` 4-gram 100k) and English (NVIDIA Riva `speechtotext_en_us_lm:deployable_v1.1`) LM assets
  - added `phase1/tools/setup_ctc_kenlm_helper.py` + `phase1/requirements-ctc-kenlm.txt` for reproducible helper-env bootstrap on Python 3.11/3.12
  - fixed a beam-path regression in `gigaam_ctc` by running raw-logit extraction under `torch.inference_mode()`, and added a focused unit test for that path
  - smoke tuning run completed under `phase1/runs/tune_ctc_kenlm_ru_smoke_20260321/`; on the local short reference sample, `beam_no_lm_bw32` ranked ahead of greedy and all LM-backed candidates
  - bounded spoken-mix RU LM rebuild completed under `phase1/models/ctc_kenlm/ru/ruwiki_spoken_mix_4gram/` with wiki weight `1` and conversational transcript weights `3`
- Public CLI/progress behavior is intentionally unchanged; transcript JSON changed only additively:
  - `phase1/transcribe.py` and `phase2/ingest.py` keep the same CLI commands and flags
  - transcript `.txt` output is unchanged; structured `.json` may now include optional per-word `speaker` and per-segment `confidence` / `source`
  - progress stages/percentages and intermediate artifact filenames are unchanged
  - DB schema and search semantics are unchanged
- Added focused `unittest` coverage for:
  - shared transcript contract round-tripping
  - phase1 CLI flag handling, runtime orchestration branches, and Apple Silicon device fallback paths
  - compare preset validation, worker clamping, experiment ranking, and backend-independent speaker assignment
  - phase2 transcript loading, row building, and search query builders
- Structural verification completed:
  - `python3 -m unittest discover -s contracts/tests`
  - `python3 -m unittest discover -s phase1/tests`
  - `python3 -m unittest discover -s phase2/tests`
  - `python3 -m py_compile` on refactored modules
- Focused quality-work verification completed on 2026-03-19:
  - `python -m unittest phase1.tests.test_audio phase1.tests.test_assign phase1.tests.test_compare_presets phase1.tests.test_compare_experiment phase1.tests.test_runner`
  - `python -m unittest discover -s phase1/tests`
  - `python -m py_compile` on the modified phase1 modules and helper script
- Compare benchmark outputs captured on 2026-03-19 for `/Users/kstroevsky/Downloads/2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov.mp4` with 600-second limits:
  - separate presets (`baseline`, `1`, `2`, `3`, `5`) completed under `phase1/runs/compare_2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov_20260319_211237/`
  - combined combo (`1+2+3`) completed under `phase1/runs/compare_2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov_20260319_221051/`
  - each completed preset has `.txt`, `.json`, `00_run_meta.json`, `progress.json`, and `00_processed_audio.wav`
- Short RU decoder-only compare captured on 2026-03-21 for `phase1/test_first2min.mp3` with reference scoring:
  - run completed under `phase1/runs/compare_test_first2min_20260321_175047/`
  - presets: greedy punctuated CTC, beam + off-the-shelf RU 3-gram KenLM, beam + bounded self-trained RU 4-gram KenLM
  - ranking on this short sample was `greedy` > `self-trained` > `external`, but all three WER/CER values remained poor enough that this run should be treated as a smoke benchmark, not the final LM decision
- RU decoder-only compare captured on 2026-03-21 for the first 10 minutes of `/Users/kstroevsky/Downloads/2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov.mp4`:
  - run completed under `phase1/runs/compare_2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov_20260321_181447/` with `duration_limit=600`
  - presets: greedy punctuated CTC, beam + off-the-shelf RU 3-gram KenLM, beam + bounded self-trained RU 4-gram KenLM
  - ranking on this RU-only slice was `external` > `self-trained` > `greedy` by proxy score, with all runs completed and no beam-search fallback
- Corrected five-way RU decoder-only compare captured on 2026-03-21 for the first 10 minutes of `/Users/kstroevsky/Downloads/2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov.mp4`:
  - corrected rerun completed under `phase1/runs/compare_2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov_20260321_200349/` after fixing zero-valued decoder option preservation for `alpha` / `beta`
  - presets: greedy punctuated CTC, beam without LM, beam + off-the-shelf RU 3-gram KenLM, beam + bounded self-trained wiki-heavy RU 4-gram KenLM, beam + bounded self-trained spoken-mix RU 4-gram KenLM
  - proxy-score ranking on this RU-only slice was `external` > `greedy` > `self-trained spoken-mix` > `beam without LM` > `self-trained wiki-heavy`; all runs completed, all beam presets stayed on beam decoding, and none fell back to greedy
- NVIDIA Canary 1B v2 RU decode run captured on 2026-03-21 for the first 10 minutes of `/Users/kstroevsky/Downloads/2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov.mp4`:
  - run completed under `phase1/runs/canary_1b_v2_bogomolov_10min_20260321_214133/` with transcript artifacts `canary_1b_v2_ru_10min.txt` and `canary_1b_v2_ru_10min.json`
  - direct full-file decode was too slow on CPU and failed on MPS due a float64 tensor move limitation in NeMo restore; chunked 30-second decode (20 chunks) completed successfully
  - chunked metadata recorded `model_load_sec=24.126`, `inference_sec=245.163`, and `chunks_written=20`
- Whisper `antony66/whisper-large-v3-russian` RU decode run captured on 2026-03-21 for the first 10 minutes of `/Users/kstroevsky/Downloads/2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov.mp4`:
  - run completed under `phase1/runs/whisper_antony66_large_v3_russian_10min_20260321_224310/` with artifacts `antony66_whisper_large_v3_russian_10min.txt` and `antony66_whisper_large_v3_russian_10min.json`
  - the first chunked MPS pass used the wrong HF generation settings and produced a punctuation-only transcript; the run was rerun successfully with `language=russian`, `max_new_tokens=256`, and `return_timestamps=false`
  - corrected final path was chunked MPS decode (20 chunks) using the exact model id `antony66/whisper-large-v3-russian`; final metadata recorded `model_load_sec=6.646`, `inference_sec=132.773`, `text_len=2111`, and `nonempty_chunks=20`
  - corrected output is substantive Russian text but still noisy on this host/runtime path (for example leading `!` artifacts and weak segmentation), so it is suitable for manual inspection but is not currently a promotion candidate
  - additive HF Whisper tool `phase1/tools/run_hf_whisper_chunked.py` landed on 2026-03-22 with manual chunking, `return_timestamps=True`, and previous-sentence carryover via Whisper `prompt_ids`; focused helper tests live in `phase1/tests/test_run_hf_whisper_chunked.py`
  - tuned rerun completed under `phase1/runs/whisper_antony66_large_v3_russian_10min_prompted_tuned_20260322_0204/` using `num_beams=1`, `max_new_tokens=96`, and `repetition_penalty=1.1`; final metadata recorded `inference_sec=488.963`, `text_len=7824`, `words=1291`, and `repeat_phrase_count=0`
  - the tuned CPU path materially reduced both failure modes seen earlier: it no longer collapses to punctuation-only output like MPS, and it no longer runs away into repeated filler text like the first CPU prompt-carryover attempt
- Additional phase1 architecture hardening landed on 2026-03-22:
  - compare preset pipeline flags now normalize into typed `ComparePipelineSpec` objects before runtime options are built, so compare/runtime boundaries no longer pass loose pipeline dictionaries
  - added `phase1/quality/*` as a dedicated meta-layer for concurrent run-quality checks, report assembly, and ranking shared by compare mode and RU decoder tuning
- Additive therapy-pipeline work landed on 2026-03-22:
  - added `phase1/backends/therapy_hybrid_backend.py` plus `phase1/therapy/*` for a replaceable Russian therapy pipeline built from explicit stage contracts
  - the therapy path now composes WhisperX VAD transcription, parallel GigaAM CTC anchoring, timestamp-entropy hallucination fallback, and a Qwen/Ollama merge stage
  - added a live `phase1/backends/canary_backend.py` built on an isolated NeMo helper env (`phase1/.canary-venv`) plus `phase1/tools/canary_transcribe.py` and `phase1/tools/setup_canary_helper.py`
  - added `phase1/tools/setup_ollama_qwen.py` for one-shot Dockerized Ollama bootstrap with `qwen3:8b`, and integrated it into `phase1/download_models.py --install-ollama-qwen`
  - `therapy_hybrid` can now consume live Canary directly through a backend provider instead of requiring a precomputed artifact JSON
  - `phase1/transcribe.py` now exposes direct backend selection for `canary` and `therapy_hybrid`, and `phase1/compare.py` now supports simple `--backend ...` compare runs without preset JSON files
  - therapy runs write additive Whisper/CTC/Canary/hallucination/merge-debug artifacts for later inspection and fine-tuning
  - speaker assignment can now switch per backend between local overlap assignment and `whisperx.assign_word_speakers`; the therapy backend defaults to WhisperX assignment and `A`/`B` speaker labels
  - shared transcript JSON now supports optional word-level `speaker` plus segment-level `confidence` / `source` fields without breaking existing loaders
  - added `phase1/tools/run_therapy_pipeline.py`, `phase1/tools/therapy_finetune.py`, `phase1/requirements-eclm.txt`, and `phase1/presets/therapy_hybrid_ru.json`
  - `phase1/tools/therapy_finetune.py` now supports a blocking `wait-review` step that selects one review template, waits until the same JSON file is filled with a valid non-empty `segments` list, then rewrites it into canonical sorted form
  - live smoke runs completed on 2026-03-22 for:
    - `phase1/transcribe.py ... --backend canary` on `phase1/test_first2min.mp3`
    - `phase1/transcribe.py ... --backend therapy_hybrid --use-live-canary` on the first 60 seconds of `/Users/kstroevsky/Downloads/2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov.mp4`
    - `phase1/compare.py ... --backend canary --backend gigaam_ctc` on a 30-second smoke slice
  - after Dockerized Ollama bootstrap was verified, a full end-to-end 60-second therapy run with live WhisperX + CTC + Canary + Ollama Qwen merge completed under `phase1/runs/therapy_live_ru_ollama_60s_20260322/` and wrote the full additive artifact set through `05_transcript_structured.json`
  - follow-up hardening fixed two runtime issues discovered during those live runs:
    - WhisperX alignment outputs are now sanitized at the backend boundary so `end < start` intervals do not leak into transcript artifacts
    - the deterministic therapy merge fallback now prefers the CTC anchor when Ollama is unavailable instead of silently preferring Whisper text
    - Qwen merge requests and Ollama bootstrap verification now force non-thinking deterministic generation (`think=false`, low temperature, capped tokens) so `qwen3:8b` on CPU does not fall back into long reasoning responses or client timeouts
  - follow-up language hardening on 2026-03-23 made explicit requested language authoritative for the therapy path:
    - `WhisperXBackend` now passes explicit `language=` through to WhisperX transcription, records both requested and reported language, and preserves the requested language when the runtime is explicitly forced
    - the therapy merge stage now rejects or anchors Latin-only foreign-looking text for `ru` / `uk` segments instead of letting English Canary/Whisper text leak into the final export
    - the merge-to-export handoff now preserves an intentional empty decision for `language_script_guard_drop` instead of falling back to the original Whisper text
  - a full first-10-minute live therapy run with forced Russian (`--lang ru`) completed under `phase1/runs/therapy_live_ru_10min_forcedlang_guarded_20260323/`
    - final export is Russian-only and includes the full additive artifact chain through `05_transcript_structured.json`
    - runtime metadata confirms `requested_language=ru`, `detected_language=ru`, live Canary `source_lang=ru`, and live Canary `target_lang=ru`
    - residual quality issue is now duplicate/overlapping short segments, not English leakage in the final transcript
  - follow-up therapy runtime repair on 2026-03-23 fixed the upstream English leakage rather than only hiding it in merge:
    - `phase1/tools/canary_transcribe.py` now forwards explicit RU Canary prompt kwargs even when NeMo exposes them through `**prompt`, and long audio is decoded in 30-second helper chunks before timestamps are stitched back into one global segment list
    - forced-`ru` / `task=transcribe` smoke checks on this host showed the previously selected Russian CT2 runtime still emitted English under both raw `faster-whisper` and the WhisperX wrapper, while `bzikst/faster-whisper-large-v3-russian` returned Russian in the same conditions
    - the therapy default Whisper CT2 runtime was therefore switched to `bzikst/faster-whisper-large-v3-russian`, and the therapy pipeline now validates forced-`ru` Whisper output before writing `01a`; invalid Latin-script primary output is replaced with aligned CTC segments as a fail-closed safeguard
    - a repaired 60-second live therapy run completed under `phase1/runs/therapy_live_ru_60s_fixed_20260323/`, and its `01a`, `01c`, `01d`, and `05` artifacts are all Russian-only
    - therapy diarization now defaults to `min_speakers=2` and `max_speakers=2` when `therapy_hybrid` runs in multi-speaker mode without explicit overrides, matching the therapy spec and avoiding unconstrained speaker-count search
    - on Apple Silicon, diarization now prefers `mps` when available and also respects the generic backend `device` override unless a more specific `diarization_device` is set; a direct 60-second pyannote check on this host completed in `8.533s` on `mps`
    - `phase1/pipeline/diarize.py` now strips pyannote `segment` objects out of artifact records so `03_diarization_segments.json` stays JSON-safe across CPU and `mps` runs
    - the missing `03` / `04` / `05` stages for the earlier clean 10-minute therapy run were rebuilt under `phase1/runs/therapy_live_ru_10min_recovered_mps_20260323/`; recovered diarization completed in `26.495s` on `mps`, and the recovered `03`, `04`, `05`, and final transcript JSON artifacts are all Russian-only
  - follow-up Step 5A / 5B work on 2026-03-23 made the therapy merge/correction stack real rather than leaving fine-tuning as an offline side utility:
    - `phase1/tools/therapy_finetune.py` now accepts a direct Gemini JSON file as review input, infers missing review segment end-times from neighboring starts, supports both fixed-window and runtime-aligned segment pair building, and trains `google/mt5-large` without the `datasets` dependency by using a dedicated low-memory partial fine-tune loop in the phase1 venv
    - the provided Gemini transcript at repo root (`2021-03-30_09-39-42_Bogomolov_first_10min.json`) produced `20` fixed-window pairs and `56` aligned segment pairs from the repaired first-10-minute therapy artifacts under `phase1/models/therapy_eclm/`
    - trained ECLM checkpoints were materialized under `phase1/models/therapy_eclm/latest/` (window experiment) and `phase1/models/therapy_eclm/segment56_mt5_large_v2/` (runtime-aligned segment experiment); the runtime-integrated checkpoint is the latter
    - the therapy runtime now includes an additive ECLM stage under `phase1/therapy/eclm.py`, records `01f_therapy_eclm_debug.json`, validates seq2seq outputs against script and lexical-anchor rules, and only applies the correction stage to short near-consensus segments so Step 5B does not produce ungrounded or excessively expensive edits
    - the Ollama/Qwen Step 5A stage now fast-paths exact or high-overlap CTC anchors and only calls Ollama on genuinely ambiguous segments, so the merge stage is materially less wasteful on long recordings
    - attempted end-to-end smoke runs on `/Users/kstroevsky/Downloads/2021-03-30 10.21.41 Zoom Meeting Vladimir Bogomolov.mp4` confirmed the new 5A/5B code paths are wired through the normal therapy CLI, but the full-stack CPU runtime remains too expensive to complete the entire 38.8-minute recording comfortably in one interactive turn on this host
  - follow-up runtime hardening on 2026-03-23 made the first 10 minutes of the second Bogomolov recording complete through the dedicated therapy CLI with the intended full stack:
    - alignment device resolution now prefers `mps` on Apple Silicon when available (`WHISPERX_ALIGNMENT_DEVICE` override supported) and falls back to CPU during inference if the local MPS backend rejects the graph; on this host, long-form RU alignment still falls back with `Output channels > 65536 not supported at the MPS device`
    - `phase1/tools/run_therapy_pipeline.py` now enables live Canary by default for the dedicated therapy runner unless `--canary-json` is provided, and `resolve_therapy_backend_options` now makes an explicit Canary artifact override the live backend instead of being shadowed by it
    - the Ollama/Qwen merge stage is now bounded by default for long-form local runs: `timeout_sec=30`, `num_predict=96`, `max_ollama_calls=8`, `max_segment_tokens=12`, and no-Qwen fallback when Canary is absent; this keeps Step 5A meaningful without allowing runaway latency
    - full-stack first-10-minute run on `/Users/kstroevsky/Downloads/2021-03-30 10.21.41 Zoom Meeting Vladimir Bogomolov.mp4` completed under `phase1/runs/therapy_bogomolov_2021-03-30_10.21.41_first10min_fullstack_20260323/` with `wall_clock_sec=1436.851`; additive artifacts `01a`, `01c`, `01d`, and `05` are all Russian-only, live Canary produced `93` non-empty segments, and the final structured transcript contains `144` segments
    - on that full-stack validation run, Step 5A stayed safe but was only marginally helpful: merge-note counts were `65` high-overlap CTC anchors, `10` CTC/Whisper consensus, `10` short CTC anchors, `8` Ollama fallbacks, `6` Qwen-budgeted CTC anchors, `1` Canary-assisted CTC anchor, and `1` Ollama call-budget anchor
    - on the same run, Step 5B executed but accepted `0` corrections out of `101` evaluated rows; the current gate correctly rejected low-quality candidates (`candidate_too_long`, `low_anchor_overlap`, prompt leakage) rather than silently degrading transcript quality
- Reference-backed RU tuning smoke run captured on 2026-03-21 for `phase1/test_first2min.mp3`:
  - run completed under `phase1/runs/tune_ctc_kenlm_ru_smoke_20260321/`
  - reduced grid: `beam_width=32`, `alpha in {0.0, 0.1}`, `beta in {0.0, 0.5}`
  - `beam_no_lm_bw32` ranked first by WER/CER/RTF, which is the current evidence for keeping greedy as the runtime default until LM tuning improves beyond the no-LM beam baseline
  - follow-up smoke run with the new spoken-mix LM completed under `phase1/runs/tune_ctc_kenlm_ru_spoken_mix_smoke_20260321/`; the spoken-mix self LM still ranked behind `beam_no_lm_bw32` and greedy on the short reference sample
- End-to-end phase1 runtime was revalidated on 2026-03-18 against the first 600 seconds of `/Users/kstroevsky/Downloads/2021-03-30 09.39.42 Zoom Meeting Vladimir Bogomolov.mp4`.
- Acceptance result:
  - run completed successfully with terminal progress artifacts
  - `duration_limit=600` now clipped ASR, alignment, and diarization consistently
  - ASR fell back from requested `mps` to CPU without crashing
  - alignment retried on CPU after an MPS inference failure and completed successfully
- Compare-mode hardening findings from 2026-03-19 runs:
  - shared-audio memmap reuse needed writable copies for Silero VAD stability
  - signal-handler cleanup in worker threads needed to no-op outside the main interpreter thread
  - WhisperX alignment/model loading needed serialization when compare workers run concurrently
  - diarization-heavy presets are significantly slower and more reliable when run serialized (`--max-workers 1`)

---

## Decisions Made
- Two-phase structure (`phase1`, `phase2`) with explicit module boundaries.
- Added repo-root `contracts/` as the only shared cross-phase surface. Runtime logic, ML wrappers, DB access, and worker code remain phase-local.
- Split phase1 internals into:
  - `pipeline/*` for stage owners
  - `runtime/*` for orchestration, progress, artifacts, and synthetic single-speaker handling
  - `quality/*` for the compare/tuning meta-layer: polymorphic quality checks, report assembly, and ranking
  - `config/*` for env/runtime/ASR/model settings with the original `config` import surface preserved
- Split phase2 internals into:
  - `ingest_service.py` for transcript loading/orchestration
  - `db/rows.py` and `db/queries.py` for pure helpers
  - lazy dependency loading for embeddings and DB pool setup to keep imports lightweight
- Language -> model routing:
  - `ru` -> `bzikst/faster-whisper-large-v3-russian`
  - `uk` -> `large-v3`
  - `en` -> `large-v3`
  - unknown/mixed -> `large-v3`
- Device strategy:
  - ASR default: `cuda` -> `cpu`
  - On Apple Silicon, requested ASR `mps` is treated as unsupported for faster-whisper/CTranslate2 and is downgraded to CPU with a recorded fallback reason
  - Alignment resolves its own device separately; `mps` is allowed only if preflight and inference both succeed, otherwise it falls back to CPU with a recorded reason
  - Diarization resolves its own device separately; default remains `cuda` -> `cpu`, and explicit `mps` is best-effort with CPU fallback for reliability
  - Compute type: `float16` on CUDA, `int8` otherwise
- Phase1 runtime metadata (`00_run_meta.json`) now records:
  - requested vs effective duration limit
  - resolved ASR/alignment/diarization devices
  - ASR thread count, batch size, and VAD configuration
  - any device fallback reason used during the run
- Diarization model pinning:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`
  - models are pre-downloaded/cached before diarization stage
- Execution strategy:
  - ASR first
  - alignment + diarization can run in parallel after ASR when enabled
  - speaker assignment/merge after both stages complete
- Therapy pipeline invariants:
  - the dedicated `run_therapy_pipeline.py` entrypoint is expected to run live Canary by default unless a precomputed `--canary-json` artifact is explicitly supplied
  - Step 5A/5B are currently allowed to fail closed to anchored CTC output; bounded Qwen or rejected mT5 candidates are considered preferable to ungrounded fluent rewrites on local CPU-only runs
- Compare/benchmark strategy:
  - presets are JSON files with pipeline flags plus backend id/options
  - preset pipeline flags normalize into typed compare policies before `RunOptions` are constructed
  - compare mode writes one sub-run per preset plus top-level `experiment.json` / `summary.json`
  - concurrency is allowed only when all selected presets resolve to CPU-only execution
  - optional reference transcripts add WER/CER on top of proxy quality metrics
- Quality experiment strategy:
  - quality scoring is a separate meta-layer (`phase1/quality/*`), not part of runtime orchestration
  - compare mode and RU decoder tuning both use the same polymorphic quality-check surface
  - default run-quality checks can execute concurrently because they are independent
  - ASR-focused presets may legitimately run in single-speaker mode with diarization skipped when the change under test does not affect speaker labeling
  - compare experiments now also snapshot normalized preset JSON files under `presets/`
  - compare preset runs can write `00_processed_audio.wav` to expose the exact waveform fed into ASR
- HF Whisper one-off strategy:
  - when testing plain Hugging Face Whisper checkpoints outside the main phase1 backends, do not stack pipeline `chunk_length_s` on top of already pre-sliced 30-second WAV chunks
  - for `antony66/whisper-large-v3-russian` on this host, the current best manual settings are manual 30-second chunking, `return_timestamps=True`, carryover prompt via Whisper `prompt_ids`, `num_beams=1`, `max_new_tokens=96`, and `repetition_penalty=1.1`
- Phase1 audio handling:
  - decode once, clip once, and reuse the waveform for ASR, alignment, and diarization
  - `--duration-limit` now applies to diarization as well as ASR/alignment
  - compare mode can persist one decoded audio array for reuse across CPU-only parallel presets
- Phase1 backend boundary:
  - implemented transcription backends are `whisperx`, `gigaam_ctc`, `canary`, and `therapy_hybrid`
  - runtime default backend is now `gigaam_ctc` (`ai-sage/GigaAM-v3`, `revision=e2e_ctc`)
  - Russian `gigaam_ctc` now defaults to greedy decoding; beam search is an explicit decoder setting and is not the runtime default until reference-backed tuning beats greedy
  - the repo-default self-trained Russian KenLM asset is still `phase1/models/ctc_kenlm/ru/ruwiki_plus_4gram/`; external RU KenLMs remain compare/tuning baselines only
  - RU decoder tuning uses WER first, CER second, and realtime factor third when a reference transcript is present
  - explicit `uk`/`en` language requests fall back from default `gigaam_ctc` to WhisperX in backend code
  - primary phase1 operator entrypoints are `phase1/download_models.py`, `phase1/tools/setup_ollama_qwen.py`, `phase1/transcribe.py`, `phase1/compare.py`, and `phase1/tools/therapy_finetune.py`
  - speaker assignment remains local-overlap by default, but backends can now opt into `whisperx.assign_word_speakers` plus speaker-label normalization when needed
  - the additive `therapy_hybrid` backend composes explicit therapy-stage contracts under `phase1/therapy/*` instead of embedding merge/fallback logic in the generic runtime
- CLI correctness:
  - `--no-condition` now correctly sets `condition_on_previous_text=False`
- Embeddings: `intfloat/multilingual-e5-large` (1024-d), `passage:` prefix for storage, `query:` prefix for search.
- Job queue: BullMQ (NestJS) + `bullmq` Python package (worker) - shared Redis key schema, native progress tracking.
- Hugging Face is required only for initial model download. Runtime is fully offline via `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`.

---

## Constraints / Invariants
- `contracts/` is the only allowed cross-phase shared code surface.
- Primary operator entrypoints must stay explicit and stable: `phase1/download_models.py`, `phase1/tools/setup_ollama_qwen.py`, `phase1/transcribe.py`, `phase1/compare.py`, `phase1/tools/therapy_finetune.py`, and `phase2/ingest.py`.
- `phase1/compare.py` is additive and must not change the single-run output/progress contract of `phase1/transcribe.py`.
- `HF_TOKEN` required for first-run model download (pyannote gated models). Not required after.
- `DATABASE_URL` must be set for all phase2 operations.
- Output contract: `phase1/transcribe.py` writes `<stem>.txt` and `<stem>.json` (including word-level `words[]`, and optionally word-level `speaker` plus segment-level `confidence` / `source`).
- Progress contract: if `--progress-file` (or `--save-intermediate-dir`) is set, a progress JSON is maintained with stage/percent/status updates.
- Error contract: failed runs persist `99_error.json` and preserve all completed intermediate artifacts.
- Compare contract: each preset sub-run keeps the same phase1 transcript/intermediate artifacts; experiment-level tracking lives only in top-level `experiment.json` / `summary.json`.
- Quality-layer invariant: compare and tuning quality logic lives under `phase1/quality/*`; orchestration modules may compose checks but should not reimplement scoring/ranking locally.
- Interrupt contract: `SIGINT`, `SIGTERM`, and `KeyboardInterrupt` are converted into terminal failed artifacts; `SIGKILL` remains unrecoverable.
- DB schema invariant: `segments.embedding` is 1024-d vector (matches multilingual-e5-large).
- All infrastructure self-hosted (no paid APIs); target deployment: Hetzner GEX44 (RTX 4000, EU).
- GDPR compliance required - no audio or transcripts leave self-hosted infrastructure.
- Phase1 dependency constraint: keep `torch==2.8.0`, `torchaudio==2.8.0`, `torchvision==0.23.0`, and `torchcodec==0.7.0` pinned together for Apple Silicon reproducibility.
- Residual environment note: `pyannote.audio` may still emit a non-fatal `torchcodec` FFmpeg warning on hosts that only expose FFmpeg 8 shared libraries; phase1 no longer relies on torchcodec-based diarization input decoding.
- Current blocker (validated on 2026-03-15 and unchanged): with `whisperx 3.8.2` + `pyannote-audio 4.0.4`, initializing `pyannote/speaker-diarization-3.1` still attempts to download PLDA from `pyannote/speaker-diarization-community-1`; access to that gated repo is required unless diarization stack is changed.
- DeepFilterNet helper constraint:
  - keep DeepFilterNet out of the main `phase1/venv`; it currently needs `numpy<2`, which conflicts with `whisperx`/`pyannote`
  - the helper flow assumes `phase1/.deepfilter-venv/lib/python3.13/site-packages` is present, or that `PHASE1_DEEPFILTER_SITE_PACKAGES` / `PHASE1_DEEPFILTER_PYTHON` are set explicitly
- CTC KenLM helper constraint:
  - keep `pyctcdecode` + `kenlm` out of the main `phase1/venv`; `kenlm` fails to build in the current Python 3.13 runtime but installs cleanly in a dedicated Python 3.11/3.12 helper env
  - the beam-search flow assumes `phase1/.ctc-kenlm-venv/bin/python` is present, or that `PHASE1_CTC_KENLM_PYTHON` is set explicitly
  - the helper now assumes explicit decoder-label construction that matches the CTC head dimension exactly; implicit blank-token insertion is no longer trusted
  - English Riva LM fetch currently depends on NVIDIA `ngc` CLI availability and authentication
  - the bounded RU 4-gram model trained with `--max-pages 1000` is suitable as a working smoke asset, but it is not yet the strongest RU LM variant this repo can realistically produce
  - current local reference coverage is still limited to `phase1/test_first2min.txt`; long-form RU decoder promotion still requires at least one corrected long Russian meeting transcript
- Canary helper constraint:
  - keep NeMo/Canary dependencies in the dedicated `phase1/.canary-venv`; the live Canary flow assumes `phase1/.canary-venv/bin/python` is present, or that `PHASE1_CANARY_PYTHON` / `PHASE1_CANARY_HELPER_SCRIPT` are set explicitly
  - the helper bootstrap should use Python 3.11 or 3.12 when available; `phase1/tools/setup_canary_helper.py` auto-selects that before falling back to the current interpreter
- Ollama bootstrap constraint:
  - the repo-owned Ollama bootstrap is Docker-only; it reuses a named container (`phase1-ollama-qwen`) plus a persistent volume dir under `phase1/.ollama`
  - the intended local Qwen model tag is `qwen3:8b`, aligned with the therapy merge defaults and verified through the Ollama HTTP API after pull
  - therapy merge requests to Ollama must stay non-thinking and token-capped (`think=false`, deterministic options, bounded `num_predict`) on this host; uncapped default `qwen3:8b` requests were observed to stall or time out on CPU
- HF Whisper antony66 constraint:
  - `antony66/whisper-large-v3-russian` remains unreliable on `mps` on this host even after fixing prompt/timestamp handling; the corrected path is CPU-only for now
  - higher `max_new_tokens` values on the CPU prompt-carryover path can trigger repetition loops, so the tuned runner defaults now intentionally cap generation lower
- Therapy backend constraint:
  - the live therapy WhisperX path now uses `bzikst/faster-whisper-large-v3-russian` as the host-validated CTranslate2 runtime for Russian therapy work; the previously selected Russian CT2 runtime is no longer used on this host because it still emitted English under explicit `language=ru` / `task=transcribe` smoke tests
  - the live therapy Canary helper must forward explicit `source_lang` / `target_lang` / `taskname` prompt kwargs and chunk long audio into 30-second windows before reassembling global timestamps; helper metadata is the source of truth for the effective prompt actually used
  - `therapy_hybrid` is now Canary-first: live or artifact Canary must provide the canonical segment windows and base text, while CTC is the lexical truth anchor and Whisper is only an auxiliary post-guard signal
  - when `therapy_hybrid` runs with diarization enabled and no explicit speaker-count override, the effective diarization hints must be `min_speakers=2` and `max_speakers=2`
  - on Apple Silicon, pyannote diarization should prefer `mps` when available; the generic backend `device` override should also flow into diarization unless `diarization_device` is set explicitly
  - diarization artifacts must be JSON-safe; pyannote `segment` objects should never leak into `03_diarization_segments.json`
  - the current Step 5A merge path assumes a local Ollama-served Qwen model or falls back to a deterministic Canary-base merge; no-Ollama behavior is intentionally conservative rather than fluent
  - the current Step 5B runtime is intentionally disabled in the Canary-first path until the ECLM model is retrained for Canary+CTC inputs; on this host `mt5-large` inference is CPU-only in production because Apple `mps` hits a Metal NDArray size limit during generation
  - the ECLM input format now uses plain textual prefixes (`ctc:` / `whisper:`) instead of angle-bracket sentinel-like tags; the earlier tag style caused T5-family generation to leak `<extra_id_*>`-style garbage into outputs
  - the practical repo-owned ECLM training loop now unfreezes only the last encoder/decoder block plus `lm_head` by default; a full-rank optimizer state for all `mt5-large` weights exceeded local Apple `mps` memory during training
  - the manual Gemini review handoff now assumes the reviewed artifact is the existing chunk JSON template under `review_templates/`, not raw Gemini free-form text; one `wait-review` session validates exactly one chunk
  - when `phase1/transcribe.py` is run with an explicit therapy language like `--lang ru`, that requested language should be treated as authoritative through WhisperX transcription/alignment and therapy merge filtering rather than trusting downstream model language guesses
  - live `nvidia/canary-1b-v2` is now invoked with explicit `source_lang` / `target_lang` plus chunked helper decode; the therapy merge language-script guard remains enabled as defense-in-depth, and the repaired 60-second plus recovered 10-minute therapy runs are Russian-only in upstream Canary artifacts

---

## Implementation Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Python transcription pipeline (standalone CLI) | Scaffolded, internally modularized |
| 2 | Database layer: schema, ingestion, BM25 + semantic search | Scaffolded, internally modularized |
| 3 | Docker Compose + Python BullMQ worker (Redis-driven) | Not started |
| 4 | NestJS API: job submission, status, meetings, search endpoints | Not started |
| 5 | WebSocket real-time progress | Not started |
| 6 | Hardening: retries, parallel workers, Bull Board UI, integration tests | Not started |

## Next Recommended Step
- Run the revised Canary-first `therapy_hybrid` preset on at least one long corrected Russian therapy recording and inspect `01e_therapy_merge_debug.json` for duplicate-window cleanup plus real CTC-guided meaning corrections over Canary base text.
- If the Canary-first merge looks stable, collect the first 5 reviewed 10-minute therapy chunks and decide whether the next ECLM iteration should be retrained on Canary+CTC inputs before re-enabling Step 5B at runtime.
- Compare the live Qwen merge path from the Dockerized Ollama bootstrap against the current deterministic Canary-base fallback on the same corrected Russian therapy slices.
- If new compare or tuning heuristics are needed, add them as additional `phase1/quality/*` checks instead of embedding more scoring logic into `compare_runtime` or tuning orchestration.
- Run the new tuner against at least one corrected long Russian meeting transcript and promote beam only if a tuned candidate beats both greedy and `beam_no_lm` on WER/CER.
- Rebuild the spoken-mix RU LM without `--max-pages` once more conversational transcripts are available, then retune against the same held-out manifest before considering any runtime default change.
- Install/authenticate NVIDIA `ngc` CLI on the target host, then fetch the Riva English LM into `phase1/models/ctc_kenlm/en/riva_en_us_lm/` for offline reuse.
- If one unified ranking file is required for all six quality presets (`baseline`, `1`, `2`, `3`, `5`, `combo`), rerun compare in a single serialized batch (`--max-workers 1`) and let long diarization presets finish without interruption.
- If `antony66/whisper-large-v3-russian` should become a first-class compare/backend option, port the tuned CPU-only settings from `phase1/tools/run_hf_whisper_chunked.py` into a dedicated backend instead of relying on ad hoc scripts.
