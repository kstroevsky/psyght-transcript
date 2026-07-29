"""Speaker-attributed accuracy metrics: cpWER and diarization error rate.

These matter because the project's residual quality problems are speaker/ordering
errors on 2-speaker therapy audio, which plain WER cannot see.

- **cpWER** (concatenated minimal-permutation WER): concatenate text per speaker
  on both sides, then choose the speaker mapping that minimizes total word edits.
  It penalizes content errors *and* speaker confusion in one number, and needs
  only text + speaker labels (works on the Gemini gold).
- **DER** (diarization error rate): time-based miss/false-alarm/confusion via
  ``pyannote.metrics`` (already installed). Needs timed segments on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

from phase1.eval.normalize import NormProfile, tokenize
from phase1.eval.wer import align_sequences

_MAX_EXACT_SPEAKERS = 6


@dataclass(frozen=True)
class CpWerResult:
    """Concatenated minimal-permutation WER and the chosen speaker mapping."""

    cp_wer: float
    reference_words: int
    edits: int
    mapping: dict[str, str]  # hypothesis speaker -> reference speaker (or "")
    exact: bool  # True if the optimal permutation was searched, False if greedy


def _edit_count(reference_text: str, hypothesis_text: str, profile: NormProfile | str | None) -> int:
    reference = tokenize(reference_text, profile)
    hypothesis = tokenize(hypothesis_text, profile)
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    counts = {"replace": 0, "delete": 0, "insert": 0}
    for tag, _, _ in align_sequences(reference, hypothesis):
        if tag in counts:
            counts[tag] += 1
    return counts["replace"] + counts["delete"] + counts["insert"]


def cp_word_error_rate(
    reference_by_speaker: dict[str, str],
    hypothesis_by_speaker: dict[str, str],
    profile: NormProfile | str | None = None,
) -> CpWerResult:
    """Compute cpWER between per-speaker reference and hypothesis text."""

    reference_words = sum(len(tokenize(text, profile)) for text in reference_by_speaker.values())
    ref_speakers = list(reference_by_speaker)
    hyp_speakers = list(hypothesis_by_speaker)
    size = max(len(ref_speakers), len(hyp_speakers))
    if size == 0:
        return CpWerResult(cp_wer=0.0, reference_words=0, edits=0, mapping={}, exact=True)

    ref_padded = ref_speakers + [None] * (size - len(ref_speakers))
    hyp_padded = hyp_speakers + [None] * (size - len(hyp_speakers))

    def pair_cost(ref_label: str | None, hyp_label: str | None) -> int:
        ref_text = reference_by_speaker.get(ref_label, "") if ref_label is not None else ""
        hyp_text = hypothesis_by_speaker.get(hyp_label, "") if hyp_label is not None else ""
        return _edit_count(ref_text, hyp_text, profile)

    exact = size <= _MAX_EXACT_SPEAKERS
    if exact:
        best_cost = None
        best_perm: tuple[Any, ...] = ()
        for perm in permutations(hyp_padded):
            cost = sum(pair_cost(ref_padded[i], perm[i]) for i in range(size))
            if best_cost is None or cost < best_cost:
                best_cost, best_perm = cost, perm
        edits, assignment = int(best_cost or 0), list(best_perm)
    else:
        edits, assignment = _greedy_assignment(ref_padded, hyp_padded, pair_cost)

    mapping: dict[str, str] = {}
    for ref_label, hyp_label in zip(ref_padded, assignment):
        if hyp_label is not None:
            mapping[hyp_label] = ref_label or ""
    denominator = reference_words if reference_words else 1
    return CpWerResult(
        cp_wer=round(edits / denominator, 6) if reference_words else (0.0 if edits == 0 else 1.0),
        reference_words=reference_words,
        edits=edits,
        mapping=mapping,
        exact=exact,
    )


def _greedy_assignment(ref_padded, hyp_padded, pair_cost):
    remaining = list(range(len(hyp_padded)))
    assignment: list[Any] = [None] * len(ref_padded)
    total = 0
    for i, ref_label in enumerate(ref_padded):
        best_j, best_cost = None, None
        for j in remaining:
            cost = pair_cost(ref_label, hyp_padded[j])
            if best_cost is None or cost < best_cost:
                best_j, best_cost = j, cost
        assignment[i] = hyp_padded[best_j]
        remaining.remove(best_j)
        total += int(best_cost or 0)
    return total, assignment


def diarization_error_rate(
    reference_segments: list[tuple[float, float, str]],
    hypothesis_segments: list[tuple[float, float, str]],
) -> dict[str, float] | None:
    """Compute DER and its components, or ``None`` if pyannote is unavailable.

    Each segment is ``(start, end, speaker)``. Returns ``der`` plus the
    ``miss``/``false_alarm``/``confusion`` seconds and ``total`` reference time.
    """

    try:
        from pyannote.core import Annotation, Segment  # type: ignore
        from pyannote.metrics.diarization import DiarizationErrorRate  # type: ignore
    except Exception:  # pragma: no cover - exercised only without pyannote
        return None

    def build(segments: list[tuple[float, float, str]]) -> "Annotation":
        annotation = Annotation()
        for index, (start, end, speaker) in enumerate(segments):
            if end is None or start is None or end <= start:
                continue
            annotation[Segment(float(start), float(end)), index] = str(speaker)
        return annotation

    reference = build(reference_segments)
    hypothesis = build(hypothesis_segments)
    if not reference.labels():
        return None
    metric = DiarizationErrorRate()
    components = metric(reference, hypothesis, detailed=True)
    miss = float(components.get("missed detection", 0.0))
    false_alarm = float(components.get("false alarm", 0.0))
    confusion = float(components.get("confusion", 0.0))
    total = float(components.get("total", 0.0))
    return {
        "der": round((miss + false_alarm + confusion) / total, 6) if total else 0.0,
        "miss_sec": round(miss, 3),
        "false_alarm_sec": round(false_alarm, 3),
        "confusion_sec": round(confusion, 3),
        "total_sec": round(total, 3),
    }
