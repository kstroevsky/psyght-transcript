"""phase1 evaluation layer: reference-backed accuracy metrics.

The lowest meta-layer in the experiment stack. It depends only on ``contracts``
plus optional accelerators (``rapidfuzz``, ``pyannote.metrics``) and is consumed
by ``phase1/quality`` and the experiment workbench. Keeping it dependency-light
means an agent can compute and reason about accuracy without importing the
runtime or any heavy ASR backend.

Public surface:
- normalization: :func:`normalize_text`, :class:`NormProfile`, :func:`get_profile`
- error rates: :func:`word_error_rate`, :func:`char_error_rate`, :class:`ErrorRate`
- speaker metrics: :func:`cp_word_error_rate`, :func:`diarization_error_rate`
- references: :func:`load_reference`, :class:`ReferenceTranscript`
- diffs: :func:`render_word_diff`, :func:`word_diff_ops`
- scoring: :func:`score_transcript`
"""

from __future__ import annotations

from phase1.eval.corpus import (
    Corpus,
    CorpusItem,
    load_corpus,
    load_hypotheses_from_dir,
    score_corpus,
)
from phase1.eval.diff import DiffOp, render_word_diff, word_diff_ops
from phase1.eval.normalize import (
    DEFAULT_PROFILE,
    RU_FOLD_PROFILE,
    NormProfile,
    get_profile,
    normalize_text,
    tokenize,
)
from phase1.eval.reference import (
    ReferenceTranscript,
    RefSegment,
    flatten_text,
    load_reference,
    reference_from_payload,
    text_by_speaker,
)
from phase1.eval.report import render_corpus_report, render_score_report
from phase1.eval.score import score_transcript
from phase1.eval.speaker import CpWerResult, cp_word_error_rate, diarization_error_rate
from phase1.eval.transcript import transcript_by_speaker, transcript_text, transcript_timed_segments
from phase1.eval.wer import (
    ErrorRate,
    align_sequences,
    char_error_rate,
    reference_error_rates,
    word_error_rate,
)

__all__ = [
    "DEFAULT_PROFILE",
    "RU_FOLD_PROFILE",
    "NormProfile",
    "get_profile",
    "normalize_text",
    "tokenize",
    "Corpus",
    "CorpusItem",
    "load_corpus",
    "load_hypotheses_from_dir",
    "score_corpus",
    "render_corpus_report",
    "render_score_report",
    "ErrorRate",
    "align_sequences",
    "char_error_rate",
    "reference_error_rates",
    "word_error_rate",
    "CpWerResult",
    "cp_word_error_rate",
    "diarization_error_rate",
    "ReferenceTranscript",
    "RefSegment",
    "flatten_text",
    "load_reference",
    "reference_from_payload",
    "text_by_speaker",
    "transcript_by_speaker",
    "transcript_text",
    "transcript_timed_segments",
    "DiffOp",
    "render_word_diff",
    "word_diff_ops",
    "score_transcript",
]
