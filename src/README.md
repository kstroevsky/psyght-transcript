# phase1 — Transcription Pipeline

Local transcription, compare, and therapy pipeline. The only phase with active runtime code today.

## Layer topology

```
        workbench                      (experiment manifests + leaderboard + marimo)
        |        \
compare_runtime / tuning
        |
      quality
        |
       eval                            (WER/CER/cpWER/DER, gold refs, corpus, diffs)
        |
    runtime / pipeline / backends
        |
     therapy (additive)
```

Dependency is intentionally one-way: experiment layers consume quality; quality consumes `eval` and runtime/pipeline/backends; `eval` depends only on the shared `contracts` package (plus optional `rapidfuzz`/`pyannote.metrics`). `workbench/*` composes `compare_runtime` + `eval` and is the top experiment layer. `therapy/*` composes backend and pipeline objects but does not import experiment layers.

## Backends

| ID | Model | Primary use |
|----|-------|-------------|
| `gigaam_ctc` | ai-sage/GigaAM-v3 e2e_ctc | Default Russian CTC transcription |
| `canary` | nvidia/canary-1b-v2 | Long-form Russian; live or artifact |
| `whisperx` | faster-whisper large-v3 | Non-Russian (uk/en) or auxiliary |
| `therapy_hybrid` | Canary + GigaAM CTC + WhisperX + local Ollama merge | Russian therapy sessions |

`therapy_hybrid` treats Canary as canonical structure, CTC as lexical truth anchor, and WhisperX as a guarded auxiliary signal after hallucination filtering. A local Ollama LLM handles divergent-window merge decisions; `qwen3:8b` remains the default and local `gemma4-27b` is now supported as an alternative. The pipeline falls back to a deterministic CTC-guided merge when Ollama is unavailable.

## Module map

| Path | Responsibility |
|------|----------------|
| `backends/` | Pluggable backend adapters; each implements `BaseBackend` |
| `pipeline/` | Stage wrappers: audio loading, diarization, speaker assignment |
| `therapy/` | Canary-first merge pipeline: hallucination guard, windows, merge, ECLM data prep |
| `runtime/` | Single-run orchestration, progress state, artifact persistence (`00_run_meta.json`, etc.) |
| `eval/` | Reference-backed accuracy metrics (WER/CER + S/D/I, cpWER, DER), gold-reference loading, corpus scoring, transcript diffs |
| `quality/` | Concurrent quality checks, report assembly, ranking; shared by compare and tuning; reuses `eval` for metrics |
| `compare_runtime/` | Preset loading, compare experiment orchestration; consumes `runtime/` and `quality/` |
| `tuning/` | Decoder tuning tools; consumes `quality/` |
| `workbench/` | Experiment manifests (pipeline variants over presets), leaderboard, and the marimo notebook |
| `tools/` | Operator scripts: bootstrap, fine-tune, fetch assets, LM training |
| `output/` | Transcript serialization contracts |
| `config/` | Shared config loading |
| `presets/` | JSON preset files for compare-mode runs |

## Primary entry points

- `transcribe.py` — single-run transcription for any backend, including `therapy_hybrid`
- `compare.py` — multi-backend compare run; reads presets from `presets/`
- `download_models.py` — one-time model + helper bootstrap
- `tools/setup_ollama_qwen.py` — Dockerized Ollama + `qwen3:8b` bootstrap
- `tools/setup_ollama_gemma4.py` — Dockerized Ollama + local `gemma4-27b` bootstrap
- `tools/therapy_finetune.py` — review-bundle prep, JSONL generation, ECLM training
- `tools/eval_corpus.py` — score transcripts against gold references (one file, or a corpus); no models run
- `tools/run_experiment_manifest.py` — run a manifest of pipeline variants → ranked accuracy leaderboard
- `workbench/experiment_workbench.py` — marimo notebook for the same flow (`marimo edit <path>`)

## Experiment workbench

The accuracy-driven experiment loop. Built metrics-first so every pipeline
comparison rests on trustworthy, scalable, speaker-aware numbers.

1. **Score** a hypothesis transcript against a gold reference (WER/CER/cpWER/DER + a
   readable S/D/I diff), no models run:

   ```bash
   python -m phase1.tools.eval_corpus --reference gold.json --hypothesis run.json --only-errors
   ```

2. **Corpus** — a YAML/JSON list of `(audio, reference)` pairs (`eval/corpus.example.yaml`).
   Adding a session is one entry; no code changes. Score a run folder against it:

   ```bash
   python -m phase1.tools.eval_corpus --corpus phase1/eval/corpus.example.yaml --transcripts-dir runs/<exp>
   ```

3. **Manifest** — declare many pipeline variants in one file with shared `defaults`
   and `extends` inheritance (`workbench/manifests/*.yaml`). Each variant expands into
   the same validated `ComparePreset` the compare runtime already consumes. Validate,
   then run to a leaderboard:

   ```bash
   python -m phase1.tools.run_experiment_manifest --manifest <m>.yaml --dry-run   # validate + preview
   python -m phase1.tools.run_experiment_manifest --manifest <m>.yaml             # run + rank
   ```

4. **Notebook** — `marimo edit phase1/workbench/experiment_workbench.py` exposes the
   same `phase1.workbench` / `phase1.eval` API interactively (runs are gated behind
   explicit buttons). Optional deps: `pip install -r phase1/requirements-workbench.txt`.

Reference-free proxy metrics still rank runs when no gold transcript exists; gold
references unlock WER/CER/cpWER/DER. The normalization profile (`default`,
`ru_fold` = fold ё→е) is recorded with every score so a number is reproducible.

## Environments

`phase1` uses multiple helper virtual environments for packages that conflict with the Python 3.13 main env:

| Path | Python | Purpose |
|------|--------|---------|
| `venv/` | 3.13 | Main runtime |
| `.canary-venv/` | 3.11 | Canary NeMo helper |
| `.ctc-kenlm-venv/` | 3.11 | KenLM beam-search helper |
| `.deepfilter-venv/` | 3.9/3.13 | DeepFilterNet audio enhancement |

The eval/workbench layers install into the main `venv/`:
`pip install -r requirements-eval.txt` (rapidfuzz, pyannote.metrics — scalable, speaker-aware metrics) and
`pip install -r requirements-workbench.txt` (adds marimo). Both layers import and run without these (pure-Python
fallbacks), but full-session scoring and the notebook need them.

See root `README.md` for setup commands and `docs/ARCHITECTURE.md` for system-level boundaries.
