"""Quality-report orchestration for compare and tuning workflows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from contracts.transcript import Transcript
from phase1.quality.checks import (
    PerformanceQualityCheck,
    ProxyTranscriptQualityCheck,
    ReferenceTranscriptQualityCheck,
    StabilityQualityCheck,
    rank_run_summaries,
)
from phase1.quality.contracts import QualityCheck, QualityCheckContext


def default_run_quality_checks(reference_text: str | None = None) -> tuple[QualityCheck, ...]:
    """Build the default independent checks for one run summary."""

    checks: list[QualityCheck] = [
        PerformanceQualityCheck(),
        StabilityQualityCheck(),
        ProxyTranscriptQualityCheck(),
    ]
    if reference_text:
        checks.append(ReferenceTranscriptQualityCheck())
    return tuple(checks)


def build_run_quality_report(
    *,
    transcript: Transcript | None,
    run_meta: dict[str, Any],
    reference_text: str | None,
    error: BaseException | None = None,
    checks: tuple[QualityCheck, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run one or more independent quality checks and merge their sections."""

    context = QualityCheckContext(
        transcript=transcript,
        run_meta=run_meta,
        reference_text=reference_text,
        error=error,
    )
    resolved_checks = checks or default_run_quality_checks(reference_text)
    report: dict[str, dict[str, Any]] = {
        "performance": {},
        "stability": {},
        "metrics": {},
    }
    if not resolved_checks:
        return report
    if len(resolved_checks) == 1:
        result = resolved_checks[0].evaluate(context)
        report.setdefault(result.section, {}).update(result.values)
        return report

    with ThreadPoolExecutor(max_workers=len(resolved_checks)) as pool:
        futures = [pool.submit(check.evaluate, context) for check in resolved_checks]
        for future in futures:
            result = future.result()
            report.setdefault(result.section, {}).update(result.values)
    return report


__all__ = ["build_run_quality_report", "default_run_quality_checks", "rank_run_summaries"]
