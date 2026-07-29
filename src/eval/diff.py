"""Human- and agent-readable token diffs between reference and hypothesis.

WER as a single number tells you *how much* is wrong; this tells you *what* and
*where*, which is what actually drives the next heuristic. The output is plain
markup so it reads well in a terminal, a markdown report, or a marimo cell, and
the structured op list lets an agent reason about specific spans.

Markup: substitutions ``{ref|hyp}``, deletions ``[-ref-]`` (in ref, missing from
hyp), insertions ``[+hyp+]`` (in hyp, not in ref). Correct tokens are plain.
"""

from __future__ import annotations

from dataclasses import dataclass

from phase1.eval.normalize import NormProfile, tokenize
from phase1.eval.wer import DELETE, EQUAL, INSERT, REPLACE, align_sequences


@dataclass(frozen=True)
class DiffOp:
    """One aligned diff step with the actual tokens involved."""

    tag: str  # equal | replace | delete | insert
    reference: str | None
    hypothesis: str | None


def word_diff_ops(
    reference_text: str,
    hypothesis_text: str,
    profile: NormProfile | str | None = None,
) -> list[DiffOp]:
    """Return the aligned diff as structured ops over normalized tokens."""

    reference = tokenize(reference_text, profile)
    hypothesis = tokenize(hypothesis_text, profile)
    ops: list[DiffOp] = []
    for tag, ref_index, hyp_index in align_sequences(reference, hypothesis):
        ops.append(
            DiffOp(
                tag=tag,
                reference=reference[ref_index] if ref_index is not None else None,
                hypothesis=hypothesis[hyp_index] if hyp_index is not None else None,
            )
        )
    return ops


def render_word_diff(
    reference_text: str,
    hypothesis_text: str,
    profile: NormProfile | str | None = None,
    *,
    only_errors: bool = False,
    context: int = 4,
) -> str:
    """Render an inline token diff string.

    With ``only_errors=True`` the output keeps ``context`` correct tokens around
    each error region and elides long correct runs as ``…``, which keeps diffs of
    long transcripts readable.
    """

    ops = word_diff_ops(reference_text, hypothesis_text, profile)
    if not only_errors:
        return " ".join(_marker(op) for op in ops)
    return _render_error_focused(ops, context)


def _marker(op: DiffOp) -> str:
    if op.tag == EQUAL:
        return op.reference or ""
    if op.tag == REPLACE:
        return "{%s|%s}" % (op.reference, op.hypothesis)
    if op.tag == DELETE:
        return "[-%s-]" % op.reference
    if op.tag == INSERT:
        return "[+%s+]" % op.hypothesis
    return ""


def _render_error_focused(ops: list[DiffOp], context: int) -> str:
    keep = [op.tag != EQUAL for op in ops]
    window = [False] * len(ops)
    for index, is_error in enumerate(keep):
        if not is_error:
            continue
        for offset in range(-context, context + 1):
            neighbor = index + offset
            if 0 <= neighbor < len(ops):
                window[neighbor] = True
    pieces: list[str] = []
    elided = False
    for index, op in enumerate(ops):
        if window[index]:
            pieces.append(_marker(op))
            elided = False
        elif not elided:
            pieces.append("…")
            elided = True
    return " ".join(piece for piece in pieces if piece)
