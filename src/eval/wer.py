"""Word- and character-error-rate computation with an S/D/I breakdown.

Why this exists: the legacy ``quality.checks`` WER used a pure-Python
Wagner-Fischer matrix that is O(ref*hyp) in both time and memory. That is fine
for a 2-minute clip but degrades badly at the project's real target of ~1-hour
Russian sessions (CER alone is ~360k chars -> 1.3e11 cells). This module uses
``rapidfuzz`` (bit-parallel C Levenshtein with edit operations) when available
and keeps a guarded pure-Python fallback so the package imports and the test
suite run without the optional dependency.

The metric definitions are intentionally identical to the legacy ones so scores
and rankings do not move: ``WER = edits / len(reference_tokens)`` and
``CER = edits / len(reference_chars_without_spaces)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from phase1.eval.normalize import NormProfile, char_sequence, tokenize

# Above this many DP cells we refuse the pure-Python fallback rather than hang.
_MAX_FALLBACK_CELLS = 5_000_000

# Alignment op tags, shared with the diff renderer.
EQUAL = "equal"
REPLACE = "replace"
DELETE = "delete"
INSERT = "insert"

#: One aligned step: ``(tag, ref_index_or_None, hyp_index_or_None)``.
AlignOp = tuple[str, int | None, int | None]


@dataclass(frozen=True)
class ErrorRate:
    """An error-rate result with the substitution/deletion/insertion split.

    ``rate`` is ``(substitutions + deletions + insertions) / reference_length``.
    ``hits`` is the number of correctly matched units. The S/D/I fields are
    ``None`` only when the inputs were too large for the pure-Python fallback and
    ``rapidfuzz`` was unavailable (``rate`` is still exact in that case).
    """

    rate: float
    reference_length: int
    hypothesis_length: int
    hits: int | None = None
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "rate": self.rate,
            "reference_length": self.reference_length,
            "hypothesis_length": self.hypothesis_length,
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
        }


def _rapidfuzz_editops(reference: list, hypothesis: list):
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore
    except Exception:  # pragma: no cover - exercised only without rapidfuzz
        return None
    return Levenshtein.editops(reference, hypothesis).as_list()


def _alignment_from_editops(reference: list, hypothesis: list, editops: list[tuple]) -> list[AlignOp]:
    aligned: list[AlignOp] = []
    i = j = 0
    for tag, si, dj in editops:
        while i < si and j < dj:
            aligned.append((EQUAL, i, j))
            i += 1
            j += 1
        if tag == "replace":
            aligned.append((REPLACE, i, j))
            i += 1
            j += 1
        elif tag == "delete":
            aligned.append((DELETE, i, None))
            i += 1
        elif tag == "insert":
            aligned.append((INSERT, None, j))
            j += 1
    while i < len(reference) and j < len(hypothesis):
        aligned.append((EQUAL, i, j))
        i += 1
        j += 1
    return aligned


def _alignment_pure(reference: list, hypothesis: list) -> list[AlignOp]:
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    if (rows - 1) * (cols - 1) > _MAX_FALLBACK_CELLS:
        raise RuntimeError(
            "Input too large for the pure-Python WER fallback "
            f"({rows - 1} x {cols - 1} cells). Install 'rapidfuzz' (see "
            "phase1/requirements-eval.txt) for scalable alignment."
        )
    cost = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = i
    for j in range(1, cols):
        cost[0][j] = j
    for i in range(1, rows):
        ref_item = reference[i - 1]
        row, prev = cost[i], cost[i - 1]
        for j in range(1, cols):
            sub = prev[j - 1] + (ref_item != hypothesis[j - 1])
            row[j] = min(sub, prev[j] + 1, row[j - 1] + 1)
    aligned: list[AlignOp] = []
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            tag = EQUAL if reference[i - 1] == hypothesis[j - 1] else REPLACE
            aligned.append((tag, i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            aligned.append((DELETE, i - 1, None))
            i -= 1
        else:
            aligned.append((INSERT, None, j - 1))
            j -= 1
    aligned.reverse()
    return aligned


def align_sequences(reference: list, hypothesis: list) -> list[AlignOp]:
    """Return the aligned S/D/I/equal op list between two unit sequences."""

    if not reference and not hypothesis:
        return []
    editops = _rapidfuzz_editops(reference, hypothesis)
    if editops is not None:
        return _alignment_from_editops(reference, hypothesis, editops)
    return _alignment_pure(reference, hypothesis)


def _error_rate_from_units(reference: list, hypothesis: list) -> ErrorRate:
    reference_length = len(reference)
    hypothesis_length = len(hypothesis)
    if reference_length == 0:
        # No reference units: rate is 0 when hypothesis is also empty, else 1.
        rate = 0.0 if hypothesis_length == 0 else 1.0
        return ErrorRate(
            rate=round(rate, 6),
            reference_length=0,
            hypothesis_length=hypothesis_length,
            hits=0,
            substitutions=0,
            deletions=0,
            insertions=hypothesis_length,
        )
    try:
        aligned = align_sequences(reference, hypothesis)
    except RuntimeError:
        distance = _distance_two_row(reference, hypothesis)
        return ErrorRate(
            rate=round(distance / reference_length, 6),
            reference_length=reference_length,
            hypothesis_length=hypothesis_length,
        )
    counts = {EQUAL: 0, REPLACE: 0, DELETE: 0, INSERT: 0}
    for tag, _, _ in aligned:
        counts[tag] += 1
    edits = counts[REPLACE] + counts[DELETE] + counts[INSERT]
    return ErrorRate(
        rate=round(edits / reference_length, 6),
        reference_length=reference_length,
        hypothesis_length=hypothesis_length,
        hits=counts[EQUAL],
        substitutions=counts[REPLACE],
        deletions=counts[DELETE],
        insertions=counts[INSERT],
    )


def _distance_two_row(reference: list, hypothesis: list) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_item != hyp_item)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str, profile: NormProfile | str | None = None) -> ErrorRate:
    """Compute WER with an S/D/I breakdown under the given normalization profile."""

    return _error_rate_from_units(tokenize(reference, profile), tokenize(hypothesis, profile))


def char_error_rate(reference: str, hypothesis: str, profile: NormProfile | str | None = None) -> ErrorRate:
    """Compute CER (spaces removed) with an S/D/I breakdown."""

    return _error_rate_from_units(char_sequence(reference, profile), char_sequence(hypothesis, profile))


def reference_error_rates(reference_text: str, candidate_text: str, profile: NormProfile | str | None = None) -> dict[str, float]:
    """Legacy-compatible ``{"wer", "cer"}`` mapping for the quality layer."""

    return {
        "wer": word_error_rate(reference_text, candidate_text, profile).rate,
        "cer": char_error_rate(reference_text, candidate_text, profile).rate,
    }
