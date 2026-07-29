"""Quality checks and ranking helpers for compare and tuning workflows."""

from __future__ import annotations

from collections import Counter
from typing import Any

from contracts.transcript import Transcript
from phase1.eval import normalize_text, reference_error_rates, tokenize, transcript_text
from phase1.quality.contracts import QualityCheckContext, QualityCheckResult

# WER/CER, normalization, and transcript flattening now live in the dedicated
# phase1.eval layer (scalable, speaker-aware, single source of truth). They are
# re-imported here so the quality layer's public surface is unchanged while the
# implementation is shared. `reference_error_rates` is byte-identical to the
# previous one for non-empty references (an empty reference now scores 1.0
# instead of 0.0, which cannot occur for a whole-transcript reference).


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _tokenize(text: str) -> list[str]:
    return tokenize(text)


def proxy_quality_metrics(transcript: Transcript) -> dict[str, float]:
    """Estimate transcript quality without requiring a gold reference."""

    total_words = 0
    timed_words = 0
    scored_words = 0
    score_sum = 0.0
    labeled_segments = 0
    empty_or_short_segments = 0
    all_tokens: list[str] = []

    for segment in transcript.segments:
        tokens = _tokenize(segment.text)
        all_tokens.extend(tokens)
        if segment.speaker and segment.speaker != "SPEAKER_UNKNOWN":
            labeled_segments += 1
        duration = max(0.0, float(segment.end) - float(segment.start))
        if not tokens or duration < 0.35:
            empty_or_short_segments += 1

        for word in segment.words:
            total_words += 1
            if word.start is not None and word.end is not None:
                timed_words += 1
            scored_words += 1
            score_sum += float(word.score)

    repeated_token_count = sum(count - 1 for count in Counter(all_tokens).values() if count > 1)
    metrics = {
        "timed_word_ratio": _safe_div(timed_words, total_words),
        "mean_word_score": _safe_div(score_sum, scored_words),
        "labeled_speaker_ratio": _safe_div(labeled_segments, len(transcript.segments)),
        "empty_or_short_segment_ratio": _safe_div(empty_or_short_segments, len(transcript.segments)),
        "repetition_ratio": _safe_div(repeated_token_count, len(all_tokens)),
    }
    metrics["proxy_quality_score"] = round(
        metrics["timed_word_ratio"] * 0.35
        + metrics["mean_word_score"] * 0.25
        + metrics["labeled_speaker_ratio"] * 0.20
        + (1 - metrics["empty_or_short_segment_ratio"]) * 0.10
        + (1 - metrics["repetition_ratio"]) * 0.10,
        6,
    )
    return metrics


def fallback_reasons(run_meta: dict[str, Any]) -> list[str]:
    """Extract all non-empty fallback reasons from run metadata."""

    runtime = run_meta.get("runtime", {})
    reasons: list[str] = []
    for key in ("resolved_asr", "resolved_alignment", "resolved_diarization"):
        value = runtime.get(key) or {}
        reason = value.get("fallback_reason")
        if reason:
            reasons.append(str(reason))
    return reasons


def skipped_stages(run_meta: dict[str, Any]) -> list[str]:
    """Extract stage skips from run metadata."""

    skipped: list[str] = []
    if run_meta.get("skip_alignment") or (run_meta.get("runtime", {}).get("resolved_alignment") or {}).get("skip_reason"):
        skipped.append("alignment")
    if run_meta.get("skip_diarization") or (run_meta.get("runtime", {}).get("resolved_diarization") or {}).get("mode") == "synthetic_single_speaker":
        skipped.append("diarization")
    return skipped


def realtime_factor(run_meta: dict[str, Any]) -> float:
    """Compute wall-clock seconds divided by processed audio duration."""

    duration = float(run_meta.get("duration_sec") or 0.0)
    return _safe_div(float(run_meta.get("wall_clock_sec") or 0.0), duration)


def rank_run_summaries(runs: list[dict[str, Any]], has_reference: bool) -> list[dict[str, Any]]:
    """Return compare summaries sorted by the default ranking rules."""

    def sort_key(run: dict[str, Any]) -> tuple[Any, ...]:
        metrics = run.get("metrics", {})
        stability = run.get("stability", {})
        success_rank = 0 if run.get("status") == "completed" else 1
        proxy_quality = -float(metrics.get("proxy_quality_score", 0.0))
        fallback_count = int(stability.get("fallback_count", 0))
        skipped_count = int(stability.get("skipped_stage_count", 0))
        wall_clock = float(run.get("performance", {}).get("wall_clock_sec", 0.0))
        rtf = float(run.get("performance", {}).get("realtime_factor", 0.0))
        if has_reference:
            return (
                success_rank,
                float(metrics.get("wer", 1.0)),
                float(metrics.get("cer", 1.0)),
                rtf,
                fallback_count,
                skipped_count,
                wall_clock,
                proxy_quality,
                str(run.get("preset_id")),
            )
        return (
            success_rank,
            proxy_quality,
            fallback_count,
            skipped_count,
            wall_clock,
            str(run.get("preset_id")),
        )

    ranked = sorted(runs, key=sort_key)
    for index, run in enumerate(ranked, start=1):
        run["rank"] = index
    return ranked


class PerformanceQualityCheck:
    """Performance-oriented check for one compare or tuning run."""

    name = "performance"
    section = "performance"

    def evaluate(self, context: QualityCheckContext) -> QualityCheckResult:
        return QualityCheckResult(
            section=self.section,
            values={
                "wall_clock_sec": float(context.run_meta.get("wall_clock_sec") or 0.0),
                "stage_timings_sec": context.run_meta.get("stage_timings_sec") or {},
                "realtime_factor": realtime_factor(context.run_meta),
            },
        )


class StabilityQualityCheck:
    """Stability-oriented check for runtime fallbacks, skips, and failures."""

    name = "stability"
    section = "stability"

    def evaluate(self, context: QualityCheckContext) -> QualityCheckResult:
        reasons = fallback_reasons(context.run_meta)
        skipped = skipped_stages(context.run_meta)
        return QualityCheckResult(
            section=self.section,
            values={
                "failure_stage": None,
                "fallback_reasons": reasons,
                "fallback_count": len(reasons),
                "skipped_stages": skipped,
                "skipped_stage_count": len(skipped),
                "error": str(context.error) if context.error else None,
            },
        )


class ProxyTranscriptQualityCheck:
    """Reference-free transcript quality check."""

    name = "proxy_transcript_quality"
    section = "metrics"

    def evaluate(self, context: QualityCheckContext) -> QualityCheckResult:
        values = {}
        if context.transcript is not None:
            values = proxy_quality_metrics(context.transcript)
        return QualityCheckResult(section=self.section, values=values)


class ReferenceTranscriptQualityCheck:
    """Reference-backed transcript quality check."""

    name = "reference_transcript_quality"
    section = "metrics"

    def evaluate(self, context: QualityCheckContext) -> QualityCheckResult:
        values = {}
        if context.transcript is not None and context.reference_text:
            values = reference_error_rates(context.reference_text, transcript_text(context.transcript))
        return QualityCheckResult(section=self.section, values=values)
