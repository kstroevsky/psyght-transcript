"""Aggregate per-(variant, item) results into a ranked leaderboard.

Pure functions: take the flat result rows the runner produced and turn them into
one ranked table per variant. Kept separate from the runner so ranking/rendering
is unit-testable without running any model.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from phase1.eval.report import render_markdown_table


def _mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(mean(values), 6) if values else None


def build_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate flat ``(variant, item)`` rows into one ranked row per variant.

    Each input row is one variant run on one corpus item. Output rows carry macro
    means across items plus completion and stability counters, ranked best-first.
    """

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    aggregated: list[dict[str, Any]] = []
    for variant, variant_rows in by_variant.items():
        completed = [row for row in variant_rows if row.get("status") == "completed"]
        aggregated.append(
            {
                "variant": variant,
                "n_items": len(variant_rows),
                "n_completed": len(completed),
                "macro_wer": _mean_of(completed, "wer"),
                "macro_cer": _mean_of(completed, "cer"),
                "macro_cp_wer": _mean_of(completed, "cp_wer"),
                "mean_der": _mean_of(completed, "der"),
                "mean_rtf": _mean_of(completed, "realtime_factor"),
                "proxy": _mean_of(completed, "proxy_quality_score"),
                "fallbacks": sum(int(row.get("fallback_count") or 0) for row in variant_rows),
            }
        )
    return rank_leaderboard(aggregated)


def rank_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank aggregated rows: completeness, then WER (if any), else proxy quality."""

    has_reference = any(row.get("macro_wer") is not None for row in rows)

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        incomplete = 0 if row["n_completed"] == row["n_items"] and row["n_items"] else 1
        if has_reference:
            return (
                incomplete,
                row["macro_wer"] if row["macro_wer"] is not None else 1e9,
                row["macro_cer"] if row["macro_cer"] is not None else 1e9,
                row["mean_rtf"] if row["mean_rtf"] is not None else 1e9,
                row["variant"],
            )
        return (
            incomplete,
            -(row["proxy"] if row["proxy"] is not None else -1.0),
            row["fallbacks"],
            row["variant"],
        )

    ranked = sorted(rows, key=sort_key)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def render_leaderboard(rows: list[dict[str, Any]], *, title: str = "Leaderboard") -> str:
    """Render the ranked leaderboard as a markdown table."""

    headers = ["#", "variant", "done", "WER", "CER", "cpWER", "DER", "RTF", "proxy", "fallbacks"]
    table_rows = [
        [
            row.get("rank", "-"),
            row["variant"],
            f"{row['n_completed']}/{row['n_items']}",
            _fmt(row.get("macro_wer")),
            _fmt(row.get("macro_cer")),
            _fmt(row.get("macro_cp_wer")),
            _fmt(row.get("mean_der")),
            _fmt(row.get("mean_rtf")),
            _fmt(row.get("proxy")),
            row.get("fallbacks", 0),
        ]
        for row in rows
    ]
    return f"## {title}\n\n" + render_markdown_table(headers, table_rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
