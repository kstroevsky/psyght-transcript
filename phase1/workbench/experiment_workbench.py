"""Marimo workbench for phase1 transcription experiments.

A marimo app is a plain Python file: an AI agent edits it as code, and you open
it reactively with `marimo edit phase1/workbench/experiment_workbench.py`. It is a
thin UI over the `phase1.workbench` and `phase1.eval` APIs — no logic lives here
that isn't also reachable from the CLI, so the notebook and the agent stay in
sync.

Three sections:
  1. Experiment runner — validate a manifest (dry-run) and run it to a leaderboard.
  2. Transcript scorer — score one hypothesis vs a gold reference, with a diff.
  3. Corpus scorer — score a directory of transcripts against a corpus.

Heavy work (running models) is gated behind explicit run buttons via `mo.stop`,
so opening or importing this file never launches a transcription.

Run:  marimo edit phase1/workbench/experiment_workbench.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import json
    import sys
    from pathlib import Path

    # Make the repo importable when marimo runs this file in place.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import marimo as mo

    from contracts.transcript import transcript_from_dict
    from phase1.eval import (
        load_corpus,
        load_hypotheses_from_dir,
        load_reference,
        render_corpus_report,
        render_score_report,
        render_word_diff,
        score_corpus,
        score_transcript,
    )
    from phase1.eval.reference import flatten_text
    from phase1.eval.transcript import transcript_text
    from phase1.workbench import load_manifest, render_leaderboard, run_manifest

    return (
        Path,
        flatten_text,
        json,
        load_corpus,
        load_hypotheses_from_dir,
        load_manifest,
        load_reference,
        mo,
        render_corpus_report,
        render_leaderboard,
        render_score_report,
        render_word_diff,
        run_manifest,
        score_corpus,
        score_transcript,
        transcript_from_dict,
        transcript_text,
    )


@app.cell
def _intro(mo):
    mo.md(
        """
        # phase1 experiment workbench

        Compose transcription pipelines, run them, and rank them by accuracy.
        Nothing runs until you click a **Run** button.
        """
    )
    return


@app.cell
def _runner_controls(mo):
    manifest_path = mo.ui.text(
        value="phase1/workbench/manifests/therapy_merge_compare.example.yaml",
        label="Manifest path",
        full_width=True,
    )
    validate_button = mo.ui.run_button(label="Validate (dry-run)")
    run_button = mo.ui.run_button(label="Run experiment")
    mo.md("## 1. Experiment runner")
    return manifest_path, run_button, validate_button


@app.cell
def _runner_controls_view(manifest_path, mo, run_button, validate_button):
    mo.hstack([manifest_path, validate_button, run_button], justify="start")
    return


@app.cell
def _runner_dryrun(load_manifest, manifest_path, mo, validate_button):
    mo.stop(not validate_button.value, mo.md("_Click **Validate** to expand and check the manifest._"))
    manifest = load_manifest(manifest_path.value)
    variant_lines = "\n".join(f"- `{preset.preset_id}` → backend `{preset.backend.id}`" for preset in manifest.presets)
    mo.md(f"**{manifest.id}** — {len(manifest.presets)} variant(s):\n\n{variant_lines}")
    return


@app.cell
def _runner_run(load_manifest, manifest_path, mo, render_leaderboard, run_button, run_manifest):
    mo.stop(not run_button.value, mo.md("_Click **Run experiment** to execute and rank variants (runs models)._"))
    result = run_manifest(load_manifest(manifest_path.value))
    mo.md(render_leaderboard(result["leaderboard"], title="Leaderboard") + f"\n\nArtifacts: `{result['output_dir']}`")
    return


@app.cell
def _scorer_controls(mo):
    mo.md("## 2. Transcript scorer")
    reference_path = mo.ui.text(label="Reference (gold)", full_width=True)
    hypothesis_path = mo.ui.text(label="Hypothesis transcript (.json)", full_width=True)
    profile = mo.ui.dropdown(["default", "ru_fold", "ru_fold_repeats"], value="ru_fold", label="Norm profile")
    only_errors = mo.ui.checkbox(label="Diff: only error regions")
    score_button = mo.ui.run_button(label="Score")
    mo.vstack([reference_path, hypothesis_path, mo.hstack([profile, only_errors, score_button], justify="start")])
    return hypothesis_path, only_errors, profile, reference_path, score_button


@app.cell
def _scorer_run(
    Path,
    flatten_text,
    hypothesis_path,
    json,
    load_reference,
    mo,
    only_errors,
    profile,
    reference_path,
    render_score_report,
    render_word_diff,
    score_button,
    score_transcript,
    transcript_from_dict,
    transcript_text,
):
    mo.stop(not score_button.value, mo.md("_Provide a reference + hypothesis and click **Score**._"))
    reference = load_reference(reference_path.value)
    hypothesis = transcript_from_dict(json.loads(Path(hypothesis_path.value).read_text(encoding="utf-8")))
    score = score_transcript(reference, hypothesis, profile=profile.value)
    diff = render_word_diff(flatten_text(reference), transcript_text(hypothesis), profile.value, only_errors=only_errors.value)
    mo.md(render_score_report(score, title="Score", diff=diff))
    return


@app.cell
def _corpus_controls(mo):
    mo.md("## 3. Corpus scorer")
    corpus_path = mo.ui.text(value="phase1/eval/corpus.example.yaml", label="Corpus manifest", full_width=True)
    transcripts_dir = mo.ui.text(label="Transcripts dir (<item_id>.json)", full_width=True)
    corpus_button = mo.ui.run_button(label="Score corpus")
    mo.vstack([corpus_path, transcripts_dir, corpus_button])
    return corpus_button, corpus_path, transcripts_dir


@app.cell
def _corpus_run(
    corpus_button,
    corpus_path,
    load_corpus,
    load_hypotheses_from_dir,
    mo,
    render_corpus_report,
    score_corpus,
    transcripts_dir,
):
    mo.stop(not corpus_button.value, mo.md("_Point at a corpus + a transcripts dir and click **Score corpus**._"))
    corpus = load_corpus(corpus_path.value)
    hypotheses = load_hypotheses_from_dir(corpus, transcripts_dir.value)
    mo.md(render_corpus_report(score_corpus(corpus, hypotheses)))
    return


if __name__ == "__main__":
    app.run()
