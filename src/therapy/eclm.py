"""Seq2seq correction helpers for the therapy pipeline's ECLM stage."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from phase1.therapy.models import MergeDecision, MergeRequest
from phase1.therapy.windows import normalize_text

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+")
_SENTINEL_RE = re.compile(r"<extra_id_\d+>|<ct>|</?s>", re.IGNORECASE)
_PROMPT_LEAK_RE = re.compile(r"\b(ctc|whisper)\b", re.IGNORECASE)


def build_eclm_input(request: MergeRequest) -> str:
    """Render the seq2seq input format used for training and inference."""

    return f"ctc: {normalize_text(request.ctc_text)} whisper: {normalize_text(request.whisper_text)}".strip()


def _content_tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(normalize_text(text)) if len(token) > 1]


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_content_tokens(left))
    right_tokens = set(_content_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def _has_expected_script(text: str, language: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if str(language or "").lower() in {"ru", "uk"}:
        return bool(_CYRILLIC_RE.search(normalized))
    return True


def validate_eclm_candidate(request: MergeRequest, candidate_text: str) -> dict[str, Any]:
    """Reject correction candidates that are clearly unanchored or in the wrong script."""

    normalized = normalize_text(candidate_text)
    reasons: list[str] = []
    candidate_tokens = _content_tokens(normalized)
    if not normalized:
        reasons.append("empty_candidate")

    if normalized and not _has_expected_script(normalized, request.language):
        reasons.append("unexpected_script")
    if normalized and _PROMPT_LEAK_RE.search(normalized):
        reasons.append("prompt_leak")

    if normalized in {
        normalize_text(request.ctc_text),
        normalize_text(request.whisper_text),
        normalize_text(request.canary_text),
    }:
        return {
            "accepted": not reasons,
            "candidate_text": normalized,
            "token_overlap_ratio": 1.0,
            "reasons": reasons,
        }

    source_tokens = set(_content_tokens(request.ctc_text)) | set(_content_tokens(request.whisper_text))
    overlap_ratio = 0.0
    if candidate_tokens:
        overlap_ratio = len(set(candidate_tokens) & source_tokens) / len(set(candidate_tokens))
    if candidate_tokens and len(candidate_tokens) >= 3 and overlap_ratio < 0.5:
        reasons.append("low_anchor_overlap")

    max_input_tokens = max(
        len(_content_tokens(request.ctc_text)),
        len(_content_tokens(request.whisper_text)),
        len(_content_tokens(request.canary_text)),
        1,
    )
    if candidate_tokens and len(candidate_tokens) > max_input_tokens + 3:
        reasons.append("candidate_too_long")

    return {
        "accepted": not reasons,
        "candidate_text": normalized,
        "token_overlap_ratio": round(overlap_ratio, 3),
        "reasons": reasons,
    }


def resolve_eclm_device(requested_device: str | None = None) -> str:
    """Choose a local device for seq2seq inference/training."""

    if requested_device:
        return str(requested_device)
    return "cpu"


def should_try_eclm(request: MergeRequest) -> bool:
    """Restrict the expensive correction stage to short, anchored disagreements."""

    ctc_text = normalize_text(request.ctc_text)
    whisper_text = normalize_text(request.whisper_text)
    if not ctc_text or not whisper_text:
        return False
    if ctc_text == whisper_text:
        return False
    if str(request.language or "").lower() in {"ru", "uk"} and (
        not _has_expected_script(ctc_text, request.language) or not _has_expected_script(whisper_text, request.language)
    ):
        return False
    ctc_tokens = _content_tokens(ctc_text)
    whisper_tokens = _content_tokens(whisper_text)
    if _token_overlap_ratio(ctc_text, whisper_text) < 0.8:
        return False
    if abs(len(ctc_tokens) - len(whisper_tokens)) > 2:
        return False
    return max(len(ctc_tokens), len(whisper_tokens)) <= 5


class MT5SegmentCorrector:
    """Production ECLM stage backed by a local or HF-hosted mT5 checkpoint."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str | None = None,
        max_source_length: int = 512,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        repetition_penalty: float = 1.05,
    ) -> None:
        self._model_path = str(Path(model_path).expanduser().resolve())
        self._requested_device = device
        self._device = resolve_eclm_device(device)
        self._max_source_length = int(max_source_length)
        self._max_new_tokens = int(max_new_tokens)
        self._num_beams = int(num_beams)
        self._repetition_penalty = float(repetition_penalty)
        self._tokenizer = None
        self._model = None

    def runtime_summary(self) -> dict[str, Any]:
        return {
            "provider": "mt5",
            "model_path": self._model_path,
            "requested_device": self._requested_device,
            "device": self._device,
            "max_source_length": self._max_source_length,
            "max_new_tokens": self._max_new_tokens,
            "num_beams": self._num_beams,
            "repetition_penalty": self._repetition_penalty,
        }

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path)
        self._model.to(torch.device(self._device))
        self._model.eval()

    def correct(self, request: MergeRequest) -> MergeDecision:
        self._load()

        import torch

        payload = build_eclm_input(request)
        if not payload:
            return MergeDecision(text="", confidence="low", source="eclm", notes=("empty_eclm_input",))

        encoded = self._tokenizer(
            payload,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_source_length,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                num_beams=self._num_beams,
                do_sample=False,
                repetition_penalty=self._repetition_penalty,
                no_repeat_ngram_size=3,
            )
        text = normalize_text(_SENTINEL_RE.sub(" ", self._tokenizer.decode(output_ids[0], skip_special_tokens=True)))
        return MergeDecision(text=text, source="eclm")


def build_segment_corrector(options: dict[str, Any] | None = None):
    """Construct the optional ECLM correction stage from runtime options."""

    options = dict(options or {})
    if not bool(options.get("enabled")):
        return None
    provider = str(options.get("provider") or "mt5").strip().lower()
    if provider != "mt5":
        raise ValueError(f"Unsupported therapy ECLM provider: {provider!r}")
    model_path = str(options.get("model_path") or "").strip()
    if not model_path:
        raise ValueError("ECLM is enabled but no model_path was provided.")
    return MT5SegmentCorrector(
        model_path=model_path,
        device=options.get("device"),
        max_source_length=int(options.get("max_source_length") or 512),
        max_new_tokens=int(options.get("max_new_tokens") or 256),
        num_beams=int(options.get("num_beams") or 1),
        repetition_penalty=float(options.get("repetition_penalty") or 1.05),
    )
