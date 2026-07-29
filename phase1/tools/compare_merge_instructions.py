"""Compare prompt instruction variants for therapy merge windows from one saved run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.therapy.merge import DEFAULT_OLLAMA_MERGE_MODEL, OllamaSegmentMerger, list_merge_instruction_variants  # noqa: E402
from phase1.therapy.models import MergeDecision, MergeRequest  # noqa: E402
from phase1.therapy.windows import normalize_text, sort_segments, text_for_window  # noqa: E402

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _content_tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(normalize_text(text)) if len(token) > 1}


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


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


def _unsupported_token_count(candidate: str, *sources: str) -> int:
    candidate_tokens = _content_tokens(candidate)
    source_tokens: set[str] = set()
    for source in sources:
        source_tokens |= _content_tokens(source)
    return len(candidate_tokens - source_tokens)


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


def _ollama_available(base_url: str) -> bool:
    request = Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    try:
        with urlopen(request, timeout=3.0) as response:
            return response.status == 200
    except URLError:
        return False
    except OSError:
        return False


class _TracingMergerMixin:
    _last_prompt: str | None
    _last_response: str | None

    def merge_with_trace(self, request: MergeRequest) -> tuple[MergeDecision, str | None, str | None]:
        self._last_prompt = None
        self._last_response = None
        decision = self.merge(request)
        return decision, self._last_prompt, self._last_response


class _TracingOllamaMerger(_TracingMergerMixin, OllamaSegmentMerger):
    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._last_prompt = str(payload.get("prompt") or "")
        result = super()._request_json(payload)
        self._last_response = normalize_text(str(result.get("response") or ""))
        return result


class _LocalTransformersGenerator:
    def __init__(self, model_id: str, device: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = device
        self._max_new_tokens = int(max_new_tokens)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        model_kwargs: dict[str, Any] = {}
        if device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self._model.to(device)
        self._model.eval()

    def generate(self, prompt: str) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = prompt
        inputs = self._tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self._max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated = outputs[0][prompt_length:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class _TracingTransformersMerger(_TracingMergerMixin, OllamaSegmentMerger):
    def __init__(self, *, generator: _LocalTransformersGenerator, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._generator = generator

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._last_prompt = str(payload.get("prompt") or "")
        self._last_response = normalize_text(self._generator.generate(self._last_prompt))
        return {"response": self._last_response}


def _auto_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare therapy merge prompt instructions on saved artifacts.")
    parser.add_argument("run_dir", help="Existing therapy run directory containing artifacts/")
    parser.add_argument("--variant", action="append", choices=list_merge_instruction_variants(), default=None)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--max-segment-tokens", type=int, default=18)
    parser.add_argument("--min-content-tokens", type=int, default=3)
    parser.add_argument("--engine", choices=["auto", "ollama", "hf"], default="auto")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MERGE_MODEL)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--hf-device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-json", default=None)
    return parser


def _candidate_windows(
    *,
    canary_segments: list[dict[str, Any]],
    ctc_segments: list[dict[str, Any]],
    whisper_segments: list[dict[str, Any]],
    language: str,
    limit: int,
    max_segment_tokens: int,
    min_content_tokens: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, segment in enumerate(sort_segments(canary_segments)):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        canary_text = normalize_text(str(segment.get("text") or ""))
        ctc_text = text_for_window(ctc_segments, start, end)
        whisper_text = text_for_window(whisper_segments, start, end)
        if not canary_text or not ctc_text:
            continue
        canary_token_count = len(_content_tokens(canary_text))
        ctc_token_count = len(_content_tokens(ctc_text))
        if canary_token_count < min_content_tokens or ctc_token_count < min_content_tokens:
            continue
        max_token_count = max(len(_content_tokens(canary_text)), len(_content_tokens(ctc_text)), len(_content_tokens(whisper_text)))
        if max_token_count > max_segment_tokens:
            continue
        divergence = 1.0 - max(_overlap_ratio(canary_text, ctc_text), _overlap_ratio(canary_text, whisper_text))
        if divergence <= 0.0:
            continue
        candidates.append(
            {
                "candidate_id": f"seg_{index:03d}",
                "language": language,
                "start": start,
                "end": end,
                "canary_text": canary_text,
                "ctc_text": ctc_text,
                "whisper_text": whisper_text,
                "divergence": round(divergence, 3),
                "max_token_count": max_token_count,
            }
        )
    candidates.sort(key=lambda item: (-float(item["divergence"]), float(item["start"])))
    return candidates[: max(1, limit)]


def _score_decision(candidate: dict[str, Any], text: str, language: str) -> dict[str, Any]:
    canary_text = str(candidate["canary_text"])
    ctc_text = str(candidate["ctc_text"])
    whisper_text = str(candidate["whisper_text"])
    anchor_overlap = _source_overlap_ratio(text, canary_text, ctc_text, whisper_text)
    canary_overlap = _overlap_ratio(text, canary_text)
    ctc_overlap = _overlap_ratio(text, ctc_text)
    whisper_overlap = _overlap_ratio(text, whisper_text)
    unsupported_token_count = _unsupported_token_count(text, canary_text, ctc_text, whisper_text)
    same_script = _has_expected_script(text, language)
    strong_ctc_whisper_consensus = _overlap_ratio(ctc_text, whisper_text) >= 0.6
    consensus_gain = max(0.0, ctc_overlap - canary_overlap) if strong_ctc_whisper_consensus else 0.0
    score = (
        2.0 * anchor_overlap
        + 0.75 * canary_overlap
        + 0.75 * ctc_overlap
        + 0.5 * whisper_overlap
        + (0.5 if same_script else -1.0)
        + 0.75 * consensus_gain
        - 0.5 * unsupported_token_count
    )
    return {
        "score": round(score, 4),
        "anchor_overlap": round(anchor_overlap, 4),
        "canary_overlap": round(canary_overlap, 4),
        "ctc_overlap": round(ctc_overlap, 4),
        "whisper_overlap": round(whisper_overlap, 4),
        "unsupported_token_count": unsupported_token_count,
        "same_script": same_script,
        "changed_from_canary": normalize_text(text) != canary_text,
        "strong_ctc_whisper_consensus": strong_ctc_whisper_consensus,
    }


def _hf_runtime(args: argparse.Namespace) -> tuple[_LocalTransformersGenerator, dict[str, Any]]:
    device = _auto_device() if args.hf_device == "auto" else args.hf_device
    generator = _LocalTransformersGenerator(
        model_id=args.hf_model,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    return generator, {"engine": "hf", "model_id": args.hf_model, "device": device}


def _make_merger(args: argparse.Namespace, variant: str, *, hf_generator: _LocalTransformersGenerator | None = None):
    if args.engine == "ollama":
        return _TracingOllamaMerger(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            max_ollama_calls=max(64, args.candidate_limit + 4),
            max_segment_tokens=args.max_segment_tokens,
            instruction_variant=variant,
        )

    if hf_generator is None:
        raise ValueError("hf_generator is required when engine=hf")
    merger = _TracingTransformersMerger(
        generator=hf_generator,
        model=args.hf_model,
        base_url="http://localhost/unused",
        max_ollama_calls=max(64, args.candidate_limit + 4),
        max_segment_tokens=args.max_segment_tokens,
        instruction_variant=variant,
    )
    return merger


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    artifacts_dir = run_dir / "artifacts"
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else artifacts_dir / f"06_merge_instruction_compare_{_timestamp()}.json"
    )

    requested_variants = args.variant or list(list_merge_instruction_variants())
    effective_engine = args.engine
    if effective_engine == "auto":
        effective_engine = "ollama" if _ollama_available(args.ollama_base_url) else "hf"
    args.engine = effective_engine

    run_meta = _read_json(artifacts_dir / "00_run_meta.json")
    language = str(run_meta.get("requested_language") or run_meta.get("detected_language") or "ru")
    canary_segments = _read_json(artifacts_dir / "01c_therapy_canary_segments.json")
    ctc_segments = _read_json(artifacts_dir / "01b_therapy_ctc_segments.json")
    whisper_segments = _read_json(artifacts_dir / "01d_therapy_hallucination_segments.json")
    candidates = _candidate_windows(
        canary_segments=canary_segments,
        ctc_segments=ctc_segments,
        whisper_segments=whisper_segments,
        language=language,
        limit=args.candidate_limit,
        max_segment_tokens=args.max_segment_tokens,
        min_content_tokens=args.min_content_tokens,
    )
    if not candidates:
        raise RuntimeError("No merge candidates matched the current selection filters.")

    variant_results: list[dict[str, Any]] = []
    runtime_info: dict[str, Any] | None = None
    hf_generator: _LocalTransformersGenerator | None = None
    if args.engine == "hf":
        hf_generator, runtime_info = _hf_runtime(args)
    else:
        runtime_info = {"engine": "ollama", "model_id": args.ollama_model}
    for variant in requested_variants:
        merger = _make_merger(args, variant, hf_generator=hf_generator)
        decisions: list[dict[str, Any]] = []
        total_score = 0.0
        total_anchor_overlap = 0.0
        total_unsupported_tokens = 0
        llm_accepted_count = 0
        changed_from_canary_count = 0
        for candidate in candidates:
            request = MergeRequest(
                language=language,
                start=float(candidate["start"]),
                end=float(candidate["end"]),
                ctc_text=str(candidate["ctc_text"]),
                whisper_text=str(candidate["whisper_text"]),
                canary_text=str(candidate["canary_text"]),
            )
            decision, prompt, raw_response = merger.merge_with_trace(request)
            final_text = normalize_text(decision.text or candidate["canary_text"])
            metrics = _score_decision(candidate, final_text, language)
            total_score += float(metrics["score"])
            total_anchor_overlap += float(metrics["anchor_overlap"])
            total_unsupported_tokens += int(metrics["unsupported_token_count"])
            if decision.source == "merged":
                llm_accepted_count += 1
            if metrics["changed_from_canary"]:
                changed_from_canary_count += 1
            decisions.append(
                {
                    **candidate,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "decision_text": final_text,
                    "decision_source": decision.source,
                    "decision_confidence": decision.confidence,
                    "decision_notes": list(decision.notes),
                    **metrics,
                }
            )
        candidate_count = len(decisions)
        variant_results.append(
            {
                "variant": variant,
                "summary": {
                    "candidate_count": candidate_count,
                    "avg_score": round(total_score / candidate_count, 4),
                    "avg_anchor_overlap": round(total_anchor_overlap / candidate_count, 4),
                    "unsupported_token_total": total_unsupported_tokens,
                    "llm_accepted_count": llm_accepted_count,
                    "changed_from_canary_count": changed_from_canary_count,
                },
                "decisions": decisions,
            }
        )

    ranking = sorted(
        variant_results,
        key=lambda item: (
            -float(item["summary"]["avg_score"]),
            int(item["summary"]["unsupported_token_total"]),
            -int(item["summary"]["llm_accepted_count"]),
        ),
    )
    report = {
        "run_dir": str(run_dir),
        "audio_path": str(run_meta.get("audio_path") or ""),
        "requested_language": language,
        "engine": runtime_info["engine"] if runtime_info else args.engine,
        "engine_runtime": runtime_info or {},
        "candidate_selection": {
            "candidate_limit": args.candidate_limit,
            "max_segment_tokens": args.max_segment_tokens,
            "min_content_tokens": args.min_content_tokens,
            "candidate_count": len(candidates),
        },
        "candidates": candidates,
        "variants": variant_results,
        "ranking": [item["variant"] for item in ranking],
        "winner": ranking[0]["variant"],
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[merge-compare] output_json={output_json}")
    print(f"[merge-compare] engine={report['engine']} model={report['engine_runtime'].get('model_id')}")
    print(f"[merge-compare] winner={report['winner']}")
    for item in ranking:
        summary = item["summary"]
        print(
            "[merge-compare] "
            f"{item['variant']}: avg_score={summary['avg_score']} "
            f"accepted={summary['llm_accepted_count']}/{summary['candidate_count']} "
            f"changed={summary['changed_from_canary_count']} "
            f"unsupported_tokens={summary['unsupported_token_total']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
