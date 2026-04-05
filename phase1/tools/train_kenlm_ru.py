"""Train a Russian KenLM model from Russian Wikipedia plus weighted extra corpus groups."""

from __future__ import annotations

import argparse
import bz2
import collections
from contextlib import contextmanager
from dataclasses import dataclass
import html
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, BinaryIO, Iterator


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from phase1.backends.ctc_kenlm import language_model_catalog  # noqa: E402

_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_REF_RE = re.compile(r"<ref[^>/]*?>.*?</ref>|<ref[^>]*/>", flags=re.DOTALL | re.IGNORECASE)
_TAGGED_BLOCK_RE = re.compile(
    r"<(math|code|nowiki|gallery|timeline|syntaxhighlight|table)[^>]*?>.*?</\1>",
    flags=re.DOTALL | re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BOLD_ITALIC_RE = re.compile(r"''+")
_HEADING_RE = re.compile(r"=+\s*(.*?)\s*=+")
_EXTERNAL_LINK_RE = re.compile(r"\[(https?://[^\s\]]+)(?:\s+([^\]]+))?\]")
_INTERNAL_LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PUNCT_TOKEN_RE = re.compile(r"([,.;:!?()«»\"“”„])")
_WORD_RE = re.compile(r"[\w\-]+|[.,;:!?()«»\"“”„]", flags=re.UNICODE)


@dataclass(frozen=True)
class CorpusGroup:
    """One weighted non-Wikipedia corpus group used for LM training."""

    label: str
    weight: int
    source_paths: list[Path]


def _build_parser() -> argparse.ArgumentParser:
    catalog = language_model_catalog()["ru"]
    parser = argparse.ArgumentParser(description="Train Russian KenLM from Russian Wikipedia and extra corpora.")
    parser.add_argument(
        "--wiki-url",
        default="https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles-multistream.xml.bz2",
        help="Source dump URL for Russian Wikipedia.",
    )
    parser.add_argument(
        "--dump-path",
        default=str(Path(catalog["binary_path"]).parent / "sources" / "ruwiki-latest-pages-articles-multistream.xml.bz2"),
        help="Local cache path for the compressed Wikipedia dump.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(catalog["binary_path"]).parent),
        help="Directory where corpus, vocab, ARPA, binary, and metadata will be written.",
    )
    parser.add_argument("--extra-text", action="append", default=[], help="Extra text file or directory to append")
    parser.add_argument("--wiki-weight", type=int, default=1, help="Relative weight applied to Wikipedia sentences.")
    parser.add_argument(
        "--corpus-group",
        action="append",
        nargs=3,
        metavar=("LABEL", "WEIGHT", "PATH"),
        default=[],
        help="Add a weighted conversational/news corpus group. Example: --corpus-group spoken 3 /path/to/transcripts",
    )
    parser.add_argument("--order", type=int, default=4, help="KenLM n-gram order")
    parser.add_argument("--memory", default="50%", help="Memory limit passed to lmplz")
    parser.add_argument(
        "--prune",
        nargs="*",
        type=int,
        default=[0, 0, 1],
        help="KenLM prune thresholds. For 4-gram, default is: 0 0 1",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page limit for smoke tests or partial runs")
    parser.add_argument("--max-extra-lines", type=int, default=None, help="Optional cap for extra text lines")
    parser.add_argument("--vocab-size", type=int, default=200000, help="How many most common tokens to keep in vocab.txt")
    parser.add_argument("--keep-arpa", action="store_true", help="Keep lm.arpa after building lm.binary")
    parser.add_argument(
        "--stream-wiki",
        action="store_true",
        help="Stream the compressed Wikipedia dump directly from the source URL instead of caching it first.",
    )
    return parser


def _download_if_missing(url: str, destination: Path) -> None:
    complete_marker = destination.with_name(destination.name + ".complete")
    if destination.exists() and complete_marker.exists():
        print(f"[RU-LM] Reusing cached dump: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    part_path.unlink(missing_ok=True)
    if destination.exists():
        destination.unlink()
    complete_marker.unlink(missing_ok=True)
    print(f"[RU-LM] Downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, part_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    part_path.replace(destination)
    complete_marker.write_text(url, encoding="utf-8")


@contextmanager
def _open_wiki_dump(
    dump_path: Path,
    wiki_url: str,
    *,
    stream_wiki: bool,
) -> Iterator[BinaryIO]:
    if stream_wiki:
        print(f"[RU-LM] Streaming dump directly from {wiki_url}")
        response = urllib.request.urlopen(wiki_url)
        try:
            with bz2.BZ2File(response) as stream:
                yield stream
        finally:
            response.close()
        return

    _download_if_missing(wiki_url, dump_path)
    with bz2.open(dump_path, "rb") as stream:
        yield stream


def _strip_nested(text: str, opener: str, closer: str) -> str:
    result: list[str] = []
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith(opener, index):
            depth += 1
            index += len(opener)
            continue
        if depth > 0 and text.startswith(closer, index):
            depth -= 1
            index += len(closer)
            continue
        if depth == 0:
            result.append(text[index])
        index += 1
    return "".join(result)


def _replace_internal_link(match: re.Match[str]) -> str:
    value = match.group(1)
    lowered = value.lower()
    if lowered.startswith(("file:", "image:", "category:", "файл:", "изображение:", "категория:")):
        return " "
    parts = [part.strip() for part in value.split("|") if part.strip()]
    if not parts:
        return " "
    return parts[-1]


def _normalize_text(text: str) -> list[str]:
    text = html.unescape(text)
    text = _COMMENT_RE.sub(" ", text)
    text = _REF_RE.sub(" ", text)
    text = _TAGGED_BLOCK_RE.sub(" ", text)
    text = _strip_nested(text, "{{", "}}")
    text = _strip_nested(text, "{|", "|}")
    text = _EXTERNAL_LINK_RE.sub(lambda match: f" {match.group(2) or ''} ", text)
    text = _INTERNAL_LINK_RE.sub(_replace_internal_link, text)
    text = _HEADING_RE.sub(lambda match: f" {match.group(1)}. ", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    text = _PUNCT_TOKEN_RE.sub(r" \1 ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return []
    sentences = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        normalized = _WHITESPACE_RE.sub(" ", sentence).strip()
        if len(normalized) < 3:
            continue
        sentences.append(normalized)
    return sentences


def _iter_wiki_pages(stream: BinaryIO, max_pages: int | None) -> Iterator[tuple[str, str]]:
    processed = 0
    for _event, elem in ET.iterparse(stream, events=("end",)):
        if not elem.tag.endswith("page"):
            continue
        title = ""
        namespace = ""
        redirect = False
        body = ""
        for child in elem:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = child.text or ""
            elif tag == "ns":
                namespace = child.text or ""
            elif tag == "redirect":
                redirect = True
            elif tag == "revision":
                for revision_child in child:
                    if revision_child.tag.rsplit("}", 1)[-1] == "text":
                        body = revision_child.text or ""
                        break
        elem.clear()
        if namespace != "0" or redirect or not body:
            continue
        yield title, body
        processed += 1
        if max_pages is not None and processed >= max_pages:
            break


def _expand_extra_paths(raw_paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Extra text path does not exist: {path}")
        if path.is_dir():
            resolved.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        else:
            resolved.append(path)
    return resolved


def _build_corpus_groups(raw_groups: list[list[str]], legacy_extra_paths: list[str]) -> list[CorpusGroup]:
    groups: list[CorpusGroup] = []
    for index, raw_group in enumerate(raw_groups, start=1):
        label, raw_weight, raw_path = raw_group
        weight = int(raw_weight)
        if weight <= 0:
            raise ValueError(f"Corpus group {index} must have a positive integer weight.")
        groups.append(
            CorpusGroup(
                label=label.strip() or f"group_{index}",
                weight=weight,
                source_paths=_expand_extra_paths([raw_path]),
            )
        )
    if legacy_extra_paths:
        groups.append(
            CorpusGroup(
                label="extra",
                weight=1,
                source_paths=_expand_extra_paths(legacy_extra_paths),
            )
        )
    return groups


def _iter_group_sentences(groups: list[CorpusGroup], max_extra_lines: int | None) -> Iterator[tuple[CorpusGroup, str]]:
    emitted = 0
    for group in groups:
        for path in group.source_paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for sentence in _normalize_text(text):
                yield group, sentence
                emitted += 1
                if max_extra_lines is not None and emitted >= max_extra_lines:
                    return


def _ensure_kenlm_binaries() -> tuple[Path, Path]:
    lmplz = shutil.which("lmplz")
    build_binary = shutil.which("build_binary")
    if lmplz and build_binary:
        return Path(lmplz), Path(build_binary)

    vendor_root = language_model_catalog()["ru"]["training_script"].resolve().parents[1] / "models" / "ctc_kenlm" / "_vendor" / "kenlm"
    build_dir = vendor_root / "build"
    built_lmplz = build_dir / "bin" / "lmplz"
    built_build_binary = build_dir / "bin" / "build_binary"
    if built_lmplz.exists() and built_build_binary.exists():
        return built_lmplz, built_build_binary

    vendor_root.parent.mkdir(parents=True, exist_ok=True)
    if not vendor_root.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/kpu/kenlm.git",
                str(vendor_root),
            ],
            check=True,
        )
    subprocess.run(
        ["cmake", "-S", str(vendor_root), "-B", str(build_dir), "-DKENLM_MAX_ORDER=6"],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "lmplz",
            "build_binary",
            "-j",
            str(max(1, os.cpu_count() or 1)),
        ],
        check=True,
    )
    return built_lmplz, built_build_binary


def _write_corpus(
    wiki_url: str,
    dump_path: Path,
    wiki_weight: int,
    corpus_groups: list[CorpusGroup],
    corpus_path: Path,
    *,
    stream_wiki: bool,
    max_pages: int | None,
    max_extra_lines: int | None,
) -> dict[str, Any]:
    if wiki_weight <= 0:
        raise ValueError("wiki_weight must be a positive integer.")
    pages = 0
    wiki_sentences = 0
    wiki_weighted_sentences = 0
    extra_sentences = 0
    extra_weighted_sentences = 0
    vocab_counter: collections.Counter[str] = collections.Counter()
    group_stats: dict[str, dict[str, Any]] = {
        group.label: {
            "label": group.label,
            "weight": group.weight,
            "source_paths": [str(path) for path in group.source_paths],
            "raw_sentences": 0,
            "weighted_sentences": 0,
        }
        for group in corpus_groups
    }
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as handle:
        with _open_wiki_dump(dump_path, wiki_url, stream_wiki=stream_wiki) as wiki_stream:
            for _title, raw_text in _iter_wiki_pages(wiki_stream, max_pages=max_pages):
                pages += 1
                for sentence in _normalize_text(raw_text):
                    for _ in range(wiki_weight):
                        handle.write(sentence)
                        handle.write("\n")
                    wiki_sentences += 1
                    wiki_weighted_sentences += wiki_weight
                    tokens = _WORD_RE.findall(sentence)
                    for _ in range(wiki_weight):
                        vocab_counter.update(tokens)
        for group, sentence in _iter_group_sentences(corpus_groups, max_extra_lines=max_extra_lines):
            for _ in range(group.weight):
                handle.write(sentence)
                handle.write("\n")
            extra_sentences += 1
            extra_weighted_sentences += group.weight
            group_stats[group.label]["raw_sentences"] += 1
            group_stats[group.label]["weighted_sentences"] += group.weight
            tokens = _WORD_RE.findall(sentence)
            for _ in range(group.weight):
                vocab_counter.update(tokens)
    return {
        "wiki_pages": pages,
        "wiki_sentences": wiki_sentences,
        "wiki_weight": wiki_weight,
        "wiki_weighted_sentences": wiki_weighted_sentences,
        "extra_sentences": extra_sentences,
        "extra_weighted_sentences": extra_weighted_sentences,
        "source_groups": [
            {
                "label": "wiki",
                "weight": wiki_weight,
                "source_paths": [str(dump_path)],
                "raw_sentences": wiki_sentences,
                "weighted_sentences": wiki_weighted_sentences,
            },
            *group_stats.values(),
        ],
        "vocab_counter": vocab_counter,
    }


def main() -> None:
    args = _build_parser().parse_args()
    dump_path = Path(args.dump_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.txt"
    vocab_path = output_dir / "vocab.txt"
    arpa_path = output_dir / "lm.arpa"
    binary_path = output_dir / "lm.binary"
    meta_path = output_dir / "training_meta.json"

    corpus_groups = _build_corpus_groups(args.corpus_group, args.extra_text)
    lmplz, build_binary = _ensure_kenlm_binaries()

    stats = _write_corpus(
        args.wiki_url,
        dump_path,
        args.wiki_weight,
        corpus_groups,
        corpus_path,
        stream_wiki=args.stream_wiki,
        max_pages=args.max_pages,
        max_extra_lines=args.max_extra_lines,
    )
    vocab_counter = stats.pop("vocab_counter")
    with vocab_path.open("w", encoding="utf-8") as handle:
        for token, _count in vocab_counter.most_common(args.vocab_size):
            handle.write(token)
            handle.write("\n")

    lmplz_command = [
        str(lmplz),
        "-o",
        str(args.order),
        "--text",
        str(corpus_path),
        "--arpa",
        str(arpa_path),
        "--memory",
        args.memory,
        "--discount_fallback",
    ]
    if args.prune:
        lmplz_command.extend(["--prune", *[str(value) for value in args.prune]])
    subprocess.run(lmplz_command, check=True)
    subprocess.run(
        [
            str(build_binary),
            "trie",
            str(arpa_path),
            str(binary_path),
        ],
        check=True,
    )

    meta = {
        "wiki_url": args.wiki_url,
        "dump_path": str(dump_path),
        "output_dir": str(output_dir),
        "order": args.order,
        "memory": args.memory,
        "prune": args.prune,
        "max_pages": args.max_pages,
        "max_extra_lines": args.max_extra_lines,
        "stream_wiki": bool(args.stream_wiki),
        "extra_text": list(args.extra_text),
        "corpus_groups": [
            {
                "label": group.label,
                "weight": group.weight,
                "source_paths": [str(path) for path in group.source_paths],
            }
            for group in corpus_groups
        ],
        "corpus_path": str(corpus_path),
        "arpa_path": str(arpa_path),
        "binary_path": str(binary_path),
        "vocab_path": str(vocab_path),
        **stats,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RU-LM] Trained KenLM binary: {binary_path}")
    print(f"[RU-LM] Metadata: {meta_path}")

    if not args.keep_arpa and arpa_path.exists():
        arpa_path.unlink()
        print(f"[RU-LM] Removed intermediate ARPA: {arpa_path}")


if __name__ == "__main__":
    main()
