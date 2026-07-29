"""Markdown/terminal rendering for evaluation scores.

One renderer used by the CLI and the marimo workbench so a single score and a
corpus run look the same everywhere. Output is GitHub-flavored markdown that
also reads fine in a terminal.
"""

from __future__ import annotations

from typing import Any


def render_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple markdown table; empty rows yield an italic placeholder."""

    if not rows:
        return "_(no rows)_"
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    head = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body])


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_score_report(score: dict[str, Any], *, title: str | None = None, diff: str | None = None) -> str:
    """Render one transcript score (and optional diff) as markdown."""

    lines: list[str] = []
    if title:
        lines.append(f"## {title}")
    metrics = [("WER", score.get("wer")), ("CER", score.get("cer"))]
    if "cp_wer" in score:
        metrics.append(("cpWER", score.get("cp_wer")))
    if "der" in score:
        metrics.append(("DER", score.get("der")))
    lines.append(render_markdown_table([name for name, _ in metrics], [[_fmt(value) for _, value in metrics]]))
    detail = score.get("wer_detail") or {}
    if detail.get("substitutions") is not None:
        lines.append(
            f"\nWER breakdown: {detail.get('substitutions')} sub, {detail.get('deletions')} del, "
            f"{detail.get('insertions')} ins over {detail.get('reference_length')} ref words "
            f"(profile `{score.get('profile')}`)."
        )
    if score.get("skipped"):
        lines.append("\nSkipped: " + ", ".join(f"{key} ({reason})" for key, reason in score["skipped"].items()))
    reference = score.get("reference") or {}
    if reference.get("inferred_ends"):
        lines.append("\n> Note: reference segment end-times were inferred from neighboring starts (DER is approximate).")
    if diff:
        lines.append("\n### Diff (reference | hypothesis)\n")
        lines.append("```\n" + diff + "\n```")
    return "\n".join(lines)


def render_corpus_report(result: dict[str, Any]) -> str:
    """Render a corpus scoring result: per-item rows plus aggregates."""

    lines = [
        f"# Corpus `{result.get('corpus')}` (profile `{result.get('profile')}`)",
        f"\nScored {result.get('n_scored')}/{result.get('n_items')} items.",
    ]
    if result.get("missing"):
        lines.append("Missing hypotheses: " + ", ".join(result["missing"]))

    headers = ["item", "WER", "CER", "cpWER", "DER", "ref_words"]
    rows = []
    for row in result.get("items", []):
        detail = row.get("wer_detail") or {}
        rows.append(
            [
                row["id"],
                _fmt(row.get("wer")),
                _fmt(row.get("cer")),
                _fmt(row.get("cp_wer")),
                _fmt(row.get("der")),
                detail.get("reference_length", "-"),
            ]
        )
    lines.append("\n" + render_markdown_table(headers, rows))

    aggregate = result.get("aggregate") or {}
    if aggregate:
        agg_rows = [[_fmt(value), key] for key, value in aggregate.items()]
        lines.append("\n## Aggregate\n")
        lines.append(render_markdown_table(["value", "metric"], agg_rows))
    return "\n".join(lines)
