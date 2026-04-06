"""LLM-backed merge helpers for the therapy transcription pipeline."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from phase1.therapy.models import MergeDecision, MergeRequest
from phase1.therapy.windows import normalize_text

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+")
_LOW_INFORMATION_TOKENS = {
    "а",
    "але",
    "бы",
    "в",
    "вас",
    "вже",
    "вот",
    "вы",
    "да",
    "же",
    "и",
    "или",
    "их",
    "к",
    "как",
    "ли",
    "мне",
    "мы",
    "на",
    "не",
    "нет",
    "но",
    "ну",
    "он",
    "она",
    "они",
    "ось",
    "по",
    "та",
    "так",
    "то",
    "ты",
    "у",
    "це",
    "что",
    "що",
    "я",
    "і",
}


def _language_name(language: str) -> str:
    return {
        "ru": "Russian",
        "uk": "Ukrainian",
        "en": "English",
    }.get(str(language or "").lower(), str(language or "LANGUAGE"))


def build_qwen_merge_prompt(request: MergeRequest) -> str:
    """Render the merge prompt used for the Qwen/Ollama prototype stage."""

    language_name = _language_name(request.language)
    return (
        f"You are correcting one {language_name} transcript segment.\n"
        f"Base transcript [Canary]: {request.canary_text or '[empty]'}\n"
        f"Truth anchor [CTC]: {request.ctc_text or '[empty]'}\n"
        f"Auxiliary transcript [Whisper]: {request.whisper_text or '[empty]'}\n"
        "Rules:\n"
        "- Start from Canary and keep its structure unless a concrete correction is justified.\n"
        "- Use CTC as the main truth source for specific word and meaning corrections.\n"
        "- Use Whisper only as an auxiliary hint when Canary and CTC remain ambiguous.\n"
        "- Make the smallest possible edit. Do not rephrase, summarize, or add unsupported words.\n"
        f"- Output corrected {language_name} only. No commentary."
    )


def _has_expected_script(text: str, language: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    language_code = str(language or "").lower()
    if language_code in {"ru", "uk"}:
        return bool(_CYRILLIC_RE.search(normalized))
    if language_code == "en":
        return bool(_LATIN_RE.search(normalized))
    return False


def _looks_foreign_for_language(text: str, language: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    language_code = str(language or "").lower()
    if language_code not in {"ru", "uk"}:
        return False
    if _CYRILLIC_RE.search(normalized):
        return False
    return len(_LATIN_RE.findall(normalized)) >= 6


def _content_tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(normalize_text(text)) if len(token) > 1}


def _ordered_content_tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(normalize_text(text)) if len(token) > 1]


def _meaningful_tokens(text: str) -> list[str]:
    return [token for token in _ordered_content_tokens(text) if token not in _LOW_INFORMATION_TOKENS]


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def _has_adjacent_token_repeat(text: str) -> bool:
    tokens = _meaningful_tokens(text)
    return any(left == right for left, right in zip(tokens, tokens[1:]))


def _has_short_ctc_boundary_leak(ctc_text: str, whisper_text: str) -> bool:
    ctc_tokens = set(_meaningful_tokens(ctc_text))
    whisper_tokens = set(_meaningful_tokens(whisper_text))
    if not ctc_tokens or not whisper_tokens:
        return False
    if max(len(ctc_tokens), len(whisper_tokens)) > 3:
        return False
    shared_tokens = ctc_tokens & whisper_tokens
    if not shared_tokens:
        return False
    ctc_only_tokens = ctc_tokens - whisper_tokens
    whisper_only_tokens = whisper_tokens - ctc_tokens
    return bool(ctc_only_tokens) and len(ctc_only_tokens) > len(whisper_only_tokens)


def _source_overlap_ratio(candidate: str, *sources: str) -> float:
    candidate_tokens = _content_tokens(candidate)
    if not candidate_tokens:
        return 0.0
    source_tokens: set[str] = set()
    for source in sources:
        source_tokens |= _content_tokens(source)
    if not source_tokens:
        return 0.0
    return len(candidate_tokens & source_tokens) / len(candidate_tokens)


def _should_use_whisper_semantic_override(
    ctc_text: str,
    whisper_text: str,
    *,
    language: str,
    allow_short_divergence: bool = False,
) -> bool:
    # Timestamp-pattern Whisper hallucinations are filtered before merge; this gate only catches
    # low-value CTC anchors when Whisper still looks like valid same-script transcript text.
    if not whisper_text or not _has_expected_script(whisper_text, language):
        return False
    if _looks_foreign_for_language(whisper_text, language):
        return False
    whisper_tokens = _meaningful_tokens(whisper_text)
    if not whisper_tokens:
        return False
    if not _meaningful_tokens(ctc_text):
        return True
    if _has_adjacent_token_repeat(ctc_text) and not _has_adjacent_token_repeat(whisper_text):
        return True
    if allow_short_divergence and _has_short_ctc_boundary_leak(ctc_text, whisper_text):
        return True
    return False


def build_qwen_merge_payload(model: str, request: MergeRequest) -> dict[str, Any]:
    """Render a deterministic non-thinking Ollama request for transcript merge."""

    return {
        "model": model,
        "prompt": build_qwen_merge_prompt(request),
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "top_k": 1,
            "num_predict": 96,
        },
    }


class RuleBasedSegmentMerger:
    """Deterministic fallback merger used when no local LLM is configured."""

    def merge(self, request: MergeRequest) -> MergeDecision:
        ctc_text = normalize_text(request.ctc_text)
        whisper_text = normalize_text(request.whisper_text)
        canary_text = normalize_text(request.canary_text)
        ctc_has_expected_script = _has_expected_script(ctc_text, request.language)
        canary_has_expected_script = _has_expected_script(canary_text, request.language)
        whisper_has_expected_script = _has_expected_script(whisper_text, request.language)

        if not canary_text:
            if ctc_text and ctc_has_expected_script:
                return MergeDecision(text=ctc_text, source="ctc_anchor", notes=("missing_canary_ctc_anchor",))
            if whisper_text and whisper_has_expected_script:
                return MergeDecision(text=whisper_text, source="whisper_fallback", notes=("missing_canary_whisper_fallback",))
            return MergeDecision(text="", confidence="low", notes=("missing_canary_transcript",))
        if not canary_has_expected_script:
            if ctc_text and ctc_has_expected_script:
                return MergeDecision(text=ctc_text, source="ctc_anchor", notes=("canary_script_guard_ctc_anchor",))
            if whisper_text and whisper_has_expected_script:
                return MergeDecision(
                    text=whisper_text,
                    source="whisper_fallback",
                    notes=("canary_script_guard_whisper_fallback",),
                )
            return MergeDecision(text="", confidence="low", notes=("language_script_guard_drop",))
        if not ctc_text:
            return MergeDecision(text=canary_text, source="canary_base", notes=("missing_ctc_canary_base",))
        if not ctc_has_expected_script:
            return MergeDecision(text=canary_text, source="canary_base", notes=("unexpected_ctc_script_canary_base",))
        if ctc_text == canary_text:
            return MergeDecision(text=canary_text, source="canary_base", notes=("canary_ctc_consensus",))
        if whisper_text and whisper_has_expected_script and ctc_text == whisper_text:
            return MergeDecision(text=ctc_text, source="ctc_consensus_override", notes=("ctc_whisper_consensus_override",))
        return MergeDecision(text=canary_text, source="canary_base", notes=("rule_based_canary_base",))


class OllamaQwenSegmentMerger:
    """Prototype merge stage backed by a local Ollama-served Qwen model."""

    def __init__(
        self,
        *,
        model: str = "qwen3:8b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_sec: float = 30.0,
        max_ollama_calls: int = 8,
        max_segment_tokens: int = 12,
        fallback: RuleBasedSegmentMerger | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = float(timeout_sec)
        self._max_ollama_calls = max(0, int(max_ollama_calls))
        self._max_segment_tokens = max(1, int(max_segment_tokens))
        self._ollama_calls = 0
        self._fallback = fallback or RuleBasedSegmentMerger()

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self._base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - network failure path
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
        except URLError as exc:  # pragma: no cover - network failure path
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    def merge(self, request: MergeRequest) -> MergeDecision:
        normalized = MergeRequest(
            language=request.language,
            start=request.start,
            end=request.end,
            ctc_text=normalize_text(request.ctc_text),
            whisper_text=normalize_text(request.whisper_text),
            canary_text=normalize_text(request.canary_text),
        )
        if not normalized.canary_text and not normalized.ctc_text and not normalized.whisper_text:
            return MergeDecision(text="", confidence="low", notes=("empty_inputs",))

        fallback = self._fallback.merge(normalized)
        ctc_has_expected_script = _has_expected_script(normalized.ctc_text, normalized.language)
        canary_has_expected_script = _has_expected_script(normalized.canary_text, normalized.language)
        ctc_tokens = _content_tokens(normalized.ctc_text)
        whisper_tokens = _content_tokens(normalized.whisper_text)
        canary_tokens = _content_tokens(normalized.canary_text)
        max_token_count = max(len(ctc_tokens), len(whisper_tokens), len(canary_tokens))

        if not normalized.canary_text:
            return fallback
        if not canary_has_expected_script:
            return fallback
        if not normalized.ctc_text:
            return MergeDecision(text=normalized.canary_text, source="canary_base", notes=("missing_ctc_canary_base",))
        if not ctc_has_expected_script:
            return MergeDecision(text=normalized.canary_text, source="canary_base", notes=("unexpected_ctc_script_canary_base",))
        if normalized.ctc_text == normalized.canary_text:
            return MergeDecision(text=normalized.canary_text, source="canary_base", notes=("canary_ctc_consensus",))
        if normalized.whisper_text and _has_expected_script(normalized.whisper_text, normalized.language) and normalized.ctc_text == normalized.whisper_text:
            return MergeDecision(
                text=normalized.ctc_text,
                source="ctc_consensus_override",
                notes=("ctc_whisper_consensus_override",),
            )
        if max_token_count > self._max_segment_tokens:
            return MergeDecision(
                text=fallback.text,
                confidence=fallback.confidence,
                source=fallback.source,
                notes=("ollama_segment_budget_canary_base", *fallback.notes),
            )
        if self._ollama_calls >= self._max_ollama_calls:
            return MergeDecision(
                text=fallback.text,
                confidence=fallback.confidence,
                source=fallback.source,
                notes=("ollama_call_budget_canary_base", *fallback.notes),
            )

        try:
            self._ollama_calls += 1
            payload = self._request_json(build_qwen_merge_payload(self._model, normalized))
            text = normalize_text(str(payload.get("response") or ""))
            if text:
                if _looks_foreign_for_language(text, normalized.language):
                    return MergeDecision(
                        text=fallback.text,
                        confidence=fallback.confidence,
                        source=fallback.source,
                        notes=("language_script_guard_ollama", *fallback.notes),
                    )
                if len(_content_tokens(text)) >= 3 and _source_overlap_ratio(
                    text,
                    normalized.canary_text,
                    normalized.ctc_text,
                    normalized.whisper_text,
                ) < 0.5:
                    return MergeDecision(
                        text=fallback.text,
                        confidence=fallback.confidence,
                        source=fallback.source,
                        notes=("low_anchor_overlap_ollama", *fallback.notes),
                    )
                if len(_content_tokens(text)) > max_token_count + 3:
                    return MergeDecision(
                        text=fallback.text,
                        confidence=fallback.confidence,
                        source=fallback.source,
                        notes=("candidate_too_long_ollama", *fallback.notes),
                    )
                return MergeDecision(text=text)
            return MergeDecision(
                text=fallback.text,
                confidence=fallback.confidence,
                source=fallback.source,
                notes=("empty_ollama_response", *fallback.notes),
            )
        except Exception:
            return MergeDecision(
                text=fallback.text,
                confidence=fallback.confidence,
                source=fallback.source,
                notes=("ollama_fallback", *fallback.notes),
            )


def merger_from_options(options: dict[str, Any] | None = None):
    """Construct the merge stage from backend options."""

    options = options or {}
    provider = str(options.get("id") or "ollama").strip().lower()
    if provider == "rule_based":
        return RuleBasedSegmentMerger()
    if provider != "ollama":
        raise ValueError(f"Unsupported therapy merge provider: {provider!r}")
    return OllamaQwenSegmentMerger(
        model=str(options.get("model") or "qwen3:8b"),
        base_url=str(options.get("base_url") or "http://127.0.0.1:11434"),
        timeout_sec=float(options.get("timeout_sec") or 30.0),
        max_ollama_calls=int(options.get("max_ollama_calls") or 8),
        max_segment_tokens=int(options.get("max_segment_tokens") or 12),
    )
