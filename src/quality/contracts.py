"""Contracts for the phase1 quality meta-layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from contracts.transcript import Transcript


@dataclass(frozen=True)
class QualityCheckContext:
    """Immutable input passed to each quality check implementation."""

    transcript: Transcript | None
    run_meta: dict[str, Any]
    reference_text: str | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class QualityCheckResult:
    """One quality-check contribution destined for a report section."""

    section: str
    values: dict[str, Any] = field(default_factory=dict)


class QualityCheck(Protocol):
    """Polymorphic contract for independent quality checks."""

    name: str
    section: str

    def evaluate(self, context: QualityCheckContext) -> QualityCheckResult: ...
