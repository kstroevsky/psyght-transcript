# psyght-decoder

## Goal
Private, self-hosted meeting transcription and search service for a NestJS backend.

## Project brief
- Canonical project state and decisions: [PROJECT_BRIEF.md](/Users/kstroevsky/Desktop/dev/psyght-decoder/PROJECT_BRIEF.md)
- Architecture reference: [docs/ARCHITECTURE.md](/Users/kstroevsky/Desktop/dev/psyght-decoder/docs/ARCHITECTURE.md)
- API contract reference: [docs/API_CONTRACT.md](/Users/kstroevsky/Desktop/dev/psyght-decoder/docs/API_CONTRACT.md)

## Internal layout
- `contracts`: shared transcript dataclasses and JSON serializers used by `phase1` and `phase2`
- `phase1`: transcription pipeline, backend adapters, runtime orchestration, compare wrapper, and a separate quality meta-layer for scoring/ranking
- `phase2`: ParadeDB/Postgres ingestion and search scaffold
- `phase3`: Redis + BullMQ worker (planned scaffold)
- `phase4`: NestJS API (planned)
- `phase5`: WebSocket progress (planned)

Primary operator entrypoints:
- `phase1/download_models.py` for one-time model/helper bootstrap
- `phase1/tools/setup_ollama_qwen.py` for one-shot Dockerized Ollama + `qwen3:8b` bootstrap
- `phase1/tools/qwen_asr_finetune.py` for Qwen3-ASR bootstrap, dataset prep, supervised fine-tuning, manual UI, and Qwen-backed pipeline runs
- `phase1/transcribe.py` for any single-run transcription backend, including `canary` and `therapy_hybrid`
- `phase1/compare.py` for backend-to-backend compare runs without preset JSON files
- `phase1/tools/therapy_finetune.py` for review-bundle prep, blocking Gemini review intake, JSONL generation, and ECLM training
- `phase2/ingest.py` for database ingestion

## Phase 1 quickstart
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
python3 -m venv phase1/venv
phase1/venv/bin/pip install -r phase1/requirements.txt
cp phase1/.env.example phase1/.env
phase1/venv/bin/python phase1/download_models.py --install-ctc-kenlm-helper --install-canary-helper --install-ollama-qwen
phase1/venv/bin/python phase1/transcribe.py /absolute/path/meeting.mp3 --backend gigaam_ctc --lang ru --max-speakers 4
```

Phase1 notes:
- Default phase1 backend is `gigaam_ctc` (`ai-sage/GigaAM-v3`, revision `e2e_ctc`) for Russian CTC transcription with punctuation.
- The Russian `gigaam_ctc` runtime now defaults to greedy CTC decoding. Beam search stays opt-in through nested `backend.options.decoder` settings and should only replace greedy after reference-backed tuning.
- Create the helper env with `python phase1/tools/setup_ctc_kenlm_helper.py`. It intentionally uses Python 3.11/3.12 because the `kenlm` package does not build cleanly on the repo's Python 3.13 runtime.
- Russian LM training lives in `phase1/tools/train_kenlm_ru.py` and now supports weighted corpus groups via repeated `--corpus-group LABEL WEIGHT PATH` entries. The old `--extra-text` flag is still accepted as a weight-1 legacy input.
- Reference-backed RU decoder tuning lives in `phase1/tools/tune_ctc_kenlm_ru.py`. It benchmarks greedy, beam-without-LM, beam+external LM, and beam+self LM on the same manifest and ranks them by WER, CER, then realtime factor.
- Packaged CTC LMs can be fetched with `python phase1/tools/fetch_ctc_lm_assets.py --lang uk` for `Yehor/kenlm-uk` and `--lang en` for NVIDIA Riva English (`ngc` CLI + auth required).
- When `--lang` is explicitly non-Russian (`uk`/`en`), the default backend falls back to WhisperX automatically for ASR/alignment compatibility.
- Keep the pinned `torch` / `torchaudio` / `torchvision` / `torchcodec` versions from `phase1/requirements.txt` together on Apple Silicon.
- Qwen3-ASR supervised fine-tuning and inference live behind an isolated helper env at `phase1/.qwen-asr-venv`; use `phase1/tools/qwen_asr_finetune.py bootstrap` to install the helper and download repo-local Qwen model assets in one step.
- The same Qwen workbench also exposes `train`, `run-pipeline`, and `launch-ui`, so you do not need to compose low-level helper commands manually.
- `--duration-limit` now clips diarization as well as ASR/alignment.
- Requested ASR `mps` is downgraded to CPU automatically on Apple Silicon; alignment and diarization use best-effort MPS with CPU fallback when runtime checks fail.
- `00_run_meta.json` records resolved stage devices, ASR thread count, VAD settings, duration-limit diagnostics, and fallback reasons.

## Common phase1 commands
Single-run transcription:
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
phase1/venv/bin/python phase1/transcribe.py /absolute/path/meeting.mp3 --backend gigaam_ctc --lang ru --max-speakers 4
```

Live NVIDIA Canary 1B v2:
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
phase1/venv/bin/python phase1/transcribe.py /absolute/path/meeting.mp3 --backend canary --lang ru
```

Therapy pipeline with live Canary:
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
phase1/venv/bin/python phase1/transcribe.py /absolute/path/session.mp3 \
  --backend therapy_hybrid \
  --lang ru \
  --use-live-canary \
  --min-speakers 2 \
  --max-speakers 2
```

If Ollama is not running locally, the therapy merge stage falls back to a deterministic CTC-first merge instead of failing the run.

Dockerized Qwen bootstrap:
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
phase1/venv/bin/python phase1/tools/setup_ollama_qwen.py
```

Qwen3-ASR backend run:
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
phase1/venv/bin/python phase1/transcribe.py /absolute/path/meeting.mp3 \
  --backend qwen_asr \
  --lang ru \
  --qwen-model-path /absolute/path/phase1/models/qwen_asr/Qwen__Qwen3-ASR-1.7B \
  --qwen-forced-aligner-path /absolute/path/phase1/models/qwen_asr/Qwen__Qwen3-ForcedAligner-0.6B \
  --max-speakers 4
```

## Phase 1 compare mode
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
phase1/venv/bin/python phase1/compare.py /absolute/path/meeting.mp3 \
  --backend gigaam_ctc \
  --backend canary \
  --lang ru \
  --reference /absolute/path/reference.txt \
  --max-workers 1
```

Compare mode notes:
- Presets live under [phase1/presets](/Users/kstroevsky/Desktop/dev/psyght-decoder/phase1/presets) and are plain JSON files with `id`, optional `description`, `pipeline`, and `backend`.
- Preset `pipeline` payloads are normalized into typed compare policies before they are converted into runtime options, so compare orchestration does not pass loose pipeline dictionaries into the runtime layer.
- If a preset omits `backend`, compare now defaults it to `gigaam_ctc` with `revision=e2e_ctc`.
- `backend.options` can override backend runtime/model settings per run, including `model_name`, `device`, `batch_size`, `threads`, `vad_method`, `vad_chunk_size`, nested `vad_options`, nested `asr_options`, nested `decoder` settings for beam search + KenLM, additive `audio_preprocessing`, and diarization post-processing flags like `merge_consecutive_speakers`.
- Implemented backends today are `whisperx`, `gigaam_ctc`, `canary`, and `therapy_hybrid`.
- `phase1/quality/*` is the canonical meta-layer for compare and tuning quality checks. It owns concurrent run-quality checks, report assembly, and ranking so scoring stays outside the runtime and backend layers.
- Each preset run still writes the same `.txt` and `.json` transcript outputs plus the usual intermediate artifacts.
- Compare experiments write `experiment.json`, `summary.json`, a saved `presets/` snapshot, and one subdirectory per preset under `phase1/runs/compare_<audio-stem>_<timestamp>/`.
- `summary.json` stores ranking inputs for performance, stability, proxy quality, and optional WER/CER. When a reference is present, ranking uses WER first, CER second, and realtime factor third.
- When intermediate artifacts are enabled, additive `00_processed_audio.wav` files let you inspect or listen to the exact audio fed into ASR for that preset.
- DeepFilterNet preprocessing runs through a dedicated helper environment at `phase1/.deepfilter-venv`; install it with `python -m venv .deepfilter-venv && source .deepfilter-venv/bin/activate && pip install -r requirements-deepfilter.txt`.

## Therapy pipeline
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
phase1/venv/bin/python phase1/transcribe.py /absolute/path/session.mp3 \
  --backend therapy_hybrid \
  --lang ru \
  --use-live-canary \
  --output-dir /absolute/path/out/therapy_run \
  --save-intermediate-dir /absolute/path/out/therapy_run
```

Therapy pipeline notes:
- The additive `therapy_hybrid` backend runs WhisperX VAD transcription, parallel GigaAM CTC anchoring, a timestamp-entropy hallucination guard, and a Qwen/Ollama merge stage.
- `phase1/tools/run_therapy_pipeline.py` remains available as a thin convenience wrapper, but `phase1/transcribe.py --backend therapy_hybrid ...` is the primary entrypoint.
- Its speaker merge path switches to `whisperx.assign_word_speakers` and normalizes diarized speakers to `A`, `B`, ... for the final structured output.
- Additive artifacts now include `01a_therapy_whisper_segments.json`, `01b_therapy_ctc_segments.json`, `01c_therapy_canary_segments.json`, `01d_therapy_hallucination_segments.json`, and `01e_therapy_merge_debug.json`.
- Structured transcript JSON can now optionally include per-word `speaker` plus per-segment `confidence` and `source` fields without breaking existing consumers.

## Therapy fine-tuning
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
phase1/venv/bin/pip install -r phase1/requirements-eclm.txt
phase1/venv/bin/python phase1/tools/therapy_finetune.py prepare-review /absolute/path/session.mp3 --output-dir /absolute/path/review_bundle
phase1/venv/bin/python phase1/tools/therapy_finetune.py wait-review --review-dir /absolute/path/review_bundle/review_templates
phase1/venv/bin/python phase1/tools/therapy_finetune.py build-jsonl \
  --whisper-artifact /absolute/path/artifacts/01a_therapy_whisper_segments.json \
  --ctc-artifact /absolute/path/artifacts/01b_therapy_ctc_segments.json \
  --review-dir /absolute/path/review_bundle/review_templates \
  --output /absolute/path/eclm/train.jsonl
phase1/venv/bin/python phase1/tools/therapy_finetune.py train-eclm --train-jsonl /absolute/path/eclm/train.jsonl --output-dir /absolute/path/eclm/mt5-large
```

Fine-tuning notes:
- `prepare-review` cuts fixed 10-minute WAV chunks and writes blank JSON templates for Gemini-reviewed timestamped transcripts.
- `wait-review` selects one incomplete chunk template, prints the exact JSON path and audio chunk path, then blocks until that same template file is filled with a valid non-empty `segments` list and you press Enter.
- `build-jsonl` converts reviewed transcripts plus Whisper/CTC artifacts into 30-second JSONL pairs in the format `<ctc> ... <whisper> ... -> clean`.
- `train-eclm` fine-tunes `google/mt5-large` with `transformers` `Seq2SeqTrainer` using the requested batch-size / epoch / learning-rate defaults unless overridden.

## Qwen3-ASR fine-tuning
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder
phase1/venv/bin/python phase1/tools/qwen_asr_finetune.py bootstrap
phase1/venv/bin/python phase1/tools/qwen_asr_finetune.py normalize-transcript \
  --transcript /absolute/path/gemini_or_review.json \
  --output /absolute/path/qwen_asr_dataset/transcript.normalized.json
phase1/venv/bin/python phase1/tools/qwen_asr_finetune.py build-jsonl \
  /absolute/path/session.wav \
  --transcript /absolute/path/qwen_asr_dataset/transcript.normalized.json \
  --output-dir /absolute/path/qwen_asr_dataset \
  --lang ru \
  --model-path /absolute/path/phase1/models/qwen_asr/Qwen__Qwen3-ASR-1.7B
phase1/venv/bin/python phase1/tools/qwen_asr_finetune.py train \
  --train-file /absolute/path/qwen_asr_dataset/train.jsonl \
  --eval-file /absolute/path/qwen_asr_dataset/eval.jsonl \
  --output-dir /absolute/path/qwen_asr_runs/ru_hq_gemini_sft
```

Qwen3-ASR notes:
- `bootstrap` is the easy-run entrypoint: it creates `phase1/.qwen-asr-venv`, downloads `Qwen/Qwen3-ASR-1.7B`, and can also pull `Qwen/Qwen3-ForcedAligner-0.6B` into repo-local storage.
- Repo-local Qwen downloads now use serial Hugging Face snapshot workers by default for reliability; if a large shard still hangs on your host, rerun with `HF_HUB_DISABLE_XET=1`.
- `normalize-transcript` rewrites supported Gemini/review JSON into canonical `{"segments": [...]}` form with inferred `end_sec` plus warnings about coarse or incomplete timing. This is the safest way to ingest ad hoc HQ transcript files before `build-jsonl`.
- `build-jsonl` accepts one Gemini JSON transcript or a directory of review-template JSON files, normalizes timestamps, slices local WAV clips, and emits `train.jsonl`, optional `eval.jsonl`, `dataset_manifest.json`, `training_runtime.json`, and `run_qwen3_asr_sft.sh`.
- Training rows follow the upstream supervised format `language Russian<asr_text>...`, so the generated dataset can be consumed directly by the bundled `train-sft` command.
- `train` is the main-environment wrapper; it delegates into the isolated helper env and records the resolved training runtime so CUDA, MPS, and CPU hosts choose sane dtypes automatically.
- `run-pipeline` reuses the normal phase1 runtime with the `qwen_asr` backend, and falls back to WhisperX alignment if you disable or omit the Qwen forced aligner.
- `launch-ui` starts a local Gradio workbench with tabs for bootstrap, dataset prep, training, and Qwen pipeline runs.

## Phase 2 quickstart
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder/phase2
cp .env.example .env
docker compose up -d
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL="postgresql://app:secret@localhost:5432/meetings"
python ingest.py /absolute/path/meeting.json --audio-path /absolute/path/meeting.mp3
```

## Verification
Recent structural verification covers:
- `python3 -m unittest discover -s contracts/tests`
- `python3 -m unittest discover -s phase1/tests`
- `python3 -m unittest discover -s phase2/tests`
- `python3 -m py_compile ...` on the refactored Python modules

## Phase 3 prep (Redis)
```bash
cd /Users/kstroevsky/Desktop/dev/psyght-decoder/phase3
docker compose up -d
```
