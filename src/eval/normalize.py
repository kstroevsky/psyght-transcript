"""Canonical text normalization for transcript evaluation metrics.

This is the single source of truth for how transcript text is normalized before
WER/CER/cpWER comparison. It exists to kill the duplicated ``normalize_text``
copies that previously lived in ``phase1/quality/checks.py`` and
``phase1/compare_runtime/reference.py``.

The default profile (``casefold`` + strip non-word punctuation + collapse
whitespace) is byte-for-byte equivalent to the legacy quality-layer normalizer,
so existing WER/CER numbers and ranking do not move. Extra, Russian-aware
options (``yo_to_ye``, ``collapse_word_repeats``) are opt-in and selected via a
named :class:`NormProfile` so an experiment can choose a stricter or looser
scoring convention without changing the metric code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormProfile:
    """A named bundle of normalization choices used by metrics and reports.

    Attributes:
        name: Stable identifier recorded in reports so a score is reproducible.
        casefold: Lowercase via :meth:`str.casefold` (Unicode-aware).
        strip_punct: Replace every non-word, non-space character with a space.
        yo_to_ye: Fold Russian ``ё``/``Ё`` to ``е``/``Е`` before comparison.
            ASR models and human references disagree on ``ё`` constantly; folding
            removes a systematic, content-irrelevant error source for RU/UK.
        collapse_word_repeats: Collapse immediate duplicate tokens (``"да да"``
            -> ``"да"``). Useful to stop stutter/repeat artifacts from dominating
            WER; off by default because it can hide real disfluency differences.
    """

    name: str = "default"
    casefold: bool = True
    strip_punct: bool = True
    yo_to_ye: bool = False
    collapse_word_repeats: bool = False


#: Legacy-equivalent profile: identical output to the old quality normalizer.
DEFAULT_PROFILE = NormProfile(name="default")

#: Russian-aware profile recommended for RU/UK accuracy comparison.
RU_FOLD_PROFILE = NormProfile(name="ru_fold", yo_to_ye=True)

_PROFILES: dict[str, NormProfile] = {
    DEFAULT_PROFILE.name: DEFAULT_PROFILE,
    RU_FOLD_PROFILE.name: RU_FOLD_PROFILE,
    "ru_fold_repeats": NormProfile(name="ru_fold_repeats", yo_to_ye=True, collapse_word_repeats=True),
}


def get_profile(profile: NormProfile | str | None) -> NormProfile:
    """Resolve a profile object, a registered profile name, or ``None``."""

    if profile is None:
        return DEFAULT_PROFILE
    if isinstance(profile, NormProfile):
        return profile
    try:
        return _PROFILES[str(profile)]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown normalization profile {profile!r}. Expected one of: {supported}") from exc


def normalize_text(text: str, profile: NormProfile | str | None = None) -> str:
    """Normalize transcript text for comparison under the given profile.

    With the default profile this matches the legacy behavior exactly:
    casefold, replace ``[^\\w\\s]`` with spaces, then collapse whitespace.
    """

    resolved = get_profile(profile)
    value = str(text or "")
    if resolved.casefold:
        value = value.casefold()
    if resolved.yo_to_ye:
        value = value.replace("ё", "е").replace("Ё", "Е").casefold() if resolved.casefold else value.replace("ё", "е").replace("Ё", "Е")
    if resolved.strip_punct:
        value = _PUNCT_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    if resolved.collapse_word_repeats and value:
        value = _collapse_repeats(value)
    return value


def tokenize(text: str, profile: NormProfile | str | None = None) -> list[str]:
    """Normalize then split into word tokens (empty input yields ``[]``)."""

    normalized = normalize_text(text, profile)
    return normalized.split() if normalized else []


def char_sequence(text: str, profile: NormProfile | str | None = None) -> list[str]:
    """Normalize then return the character sequence with spaces removed.

    Spaces are dropped so CER measures character edits, not tokenization, which
    matches the long-standing phase1 CER definition.
    """

    normalized = normalize_text(text, profile)
    return list(normalized.replace(" ", "")) if normalized else []


def _collapse_repeats(normalized: str) -> str:
    tokens = normalized.split()
    collapsed: list[str] = []
    for token in tokens:
        if not collapsed or collapsed[-1] != token:
            collapsed.append(token)
    return " ".join(collapsed)
