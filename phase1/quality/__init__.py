"""Canonical quality meta-layer for phase1 compare and tuning workflows."""

from phase1.quality.checks import (
    fallback_reasons,
    proxy_quality_metrics,
    rank_run_summaries,
    realtime_factor,
    reference_error_rates,
    skipped_stages,
    transcript_text,
)
from phase1.quality.contracts import QualityCheck, QualityCheckContext, QualityCheckResult
from phase1.quality.service import build_run_quality_report, default_run_quality_checks

__all__ = [
    "QualityCheck",
    "QualityCheckContext",
    "QualityCheckResult",
    "build_run_quality_report",
    "default_run_quality_checks",
    "fallback_reasons",
    "proxy_quality_metrics",
    "rank_run_summaries",
    "realtime_factor",
    "reference_error_rates",
    "skipped_stages",
    "transcript_text",
]
