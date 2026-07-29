"""End-to-end scoring of a hypothesis transcript against a gold reference.

This is the single entry point the corpus scorer, the quality layer, and the
marimo workbench call. It computes every metric that the available data supports
and explains (in ``skipped``) why any metric was not computed, so a report never
silently drops a number.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from contracts.transcript import Transcript
from phase1.eval.normalize import NormProfile, get_profile
from phase1.eval.reference import ReferenceTranscript, flatten_text, text_by_speaker
from phase1.eval.speaker import cp_word_error_rate, diarization_error_rate
from phase1.eval.transcript import transcript_by_speaker, transcript_text, transcript_timed_segments
from phase1.eval.wer import char_error_rate, word_error_rate


def score_transcript(
    reference: ReferenceTranscript,
    hypothesis: Transcript,
    *,
    profile: NormProfile | str | None = None,
) -> dict[str, Any]:
    """Score one hypothesis transcript against one reference.

    Returns a JSON-safe dict with ``wer``/``cer`` always present, ``cp_wer`` and
    ``der`` when the inputs support them, plus ``skipped`` reasons and the
    normalization profile used.
    """

    resolved = get_profile(profile)
    reference_text = flatten_text(reference)
    hypothesis_text = transcript_text(hypothesis)

    wer = word_error_rate(reference_text, hypothesis_text, resolved)
    cer = char_error_rate(reference_text, hypothesis_text, resolved)

    result: dict[str, Any] = {
        "profile": resolved.name,
        "wer": wer.rate,
        "cer": cer.rate,
        "wer_detail": wer.as_dict(),
        "cer_detail": cer.as_dict(),
        "reference": {
            "has_speakers": reference.has_speakers,
            "has_times": reference.has_times,
            "inferred_ends": reference.inferred_ends,
            "source_path": reference.source_path,
        },
        "skipped": {},
    }

    if reference.has_speakers and hypothesis.segments:
        cp = cp_word_error_rate(text_by_speaker(reference), transcript_by_speaker(hypothesis), resolved)
        result["cp_wer"] = cp.cp_wer
        result["cp_wer_detail"] = asdict(cp)
    else:
        result["skipped"]["cp_wer"] = "reference has no speaker labels" if not reference.has_speakers else "empty hypothesis"

    if reference.has_times:
        ref_segments = [(seg.start, seg.end, seg.speaker or "UNKNOWN") for seg in reference.segments]
        hyp_segments = transcript_timed_segments(hypothesis)
        der = diarization_error_rate(ref_segments, hyp_segments)
        if der is not None:
            result["der"] = der["der"]
            result["der_detail"] = der
        else:
            result["skipped"]["der"] = "pyannote.metrics unavailable or empty reference"
    else:
        result["skipped"]["der"] = "reference has no segment timings"

    return result
