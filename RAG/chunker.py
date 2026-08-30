"""EEGAgent parent/child chunk construction over Docling structure blocks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from .docling_parser import parse_document


SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+")


def _token_ids(text: str, tokenizer=None) -> list[Any]:
    if tokenizer is not None:
        return list(tokenizer.encode(text, add_special_tokens=False))
    return re.findall(r"[A-Za-z0-9_+.-]+|[^\W\s]", text, re.UNICODE)


def token_count(text: str, tokenizer=None) -> int:
    return len(_token_ids(text, tokenizer))


def _split_oversized_text(text: str, max_tokens: int, tokenizer=None) -> list[str]:
    if token_count(text, tokenizer) <= max_tokens:
        return [text.strip()]
    if tokenizer is not None:
        ids = _token_ids(text, tokenizer)
        return [
            tokenizer.decode(ids[start:start + max_tokens], skip_special_tokens=True).strip()
            for start in range(0, len(ids), max_tokens)
        ]
    words = text.split()
    if len(words) > 1:
        return [" ".join(words[start:start + max_tokens]) for start in range(0, len(words), max_tokens)]
    return [text[start:start + max_tokens] for start in range(0, len(text), max_tokens)]


def _stable_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _block_units(block: dict[str, Any], max_tokens: int, tokenizer=None) -> list[str]:
    text = block["text"].strip()
    if block["type"] in {"table", "list"}:
        raw_units = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        raw_units = [unit.strip() for unit in SENTENCE_BOUNDARY.split(text) if unit.strip()]
    units: list[str] = []
    for unit in raw_units or [text]:
        units.extend(_split_oversized_text(unit, max_tokens, tokenizer))
    return units


def build_parent_chunks(
    blocks: list[dict[str, Any]], source: str, tokenizer=None,
    target_tokens: int = 900, max_tokens: int = 1400,
) -> list[dict[str, Any]]:
    """Build section parents, splitting oversized sections at block boundaries."""
    parents: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_path: tuple[str, ...] = ()

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = "\n\n".join(block["text"] for block in current if block["type"] != "heading").strip()
        if not text:
            current = []
            return
        pages = [block["page"] for block in current if block.get("page") is not None]
        title_path = list(current_path)
        parent_id = _stable_id(source, " / ".join(title_path), text[:240], str(len(parents)))
        parents.append({
            "parent_id": parent_id, "source": source,
            "title": title_path[-1] if title_path else Path(source).stem,
            "title_path": title_path,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "text": text, "token_count": token_count(text, tokenizer),
        })
        current = []

    for block in blocks:
        if block["type"] == "heading":
            flush()
            current_path = tuple(block.get("title_path", []))
            continue
        block_path = tuple(block.get("title_path", []))
        if current and block_path != current_path:
            flush()
        current_path = block_path
        projected = "\n\n".join(item["text"] for item in [*current, block])
        if current and token_count(projected, tokenizer) > max_tokens:
            flush()
        if token_count(block["text"], tokenizer) > max_tokens:
            for part in _split_oversized_text(block["text"], target_tokens, tokenizer):
                current = [{**block, "text": part}]
                flush()
        else:
            current.append(block)
            projected = "\n\n".join(item["text"] for item in current)
            if token_count(projected, tokenizer) >= target_tokens:
                flush()
    flush()
    return parents


def _tail_for_overlap(units: list[str], overlap_tokens: int, tokenizer=None) -> list[str]:
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        size = token_count(unit, tokenizer)
        if tail and total + size > overlap_tokens:
            break
        tail.insert(0, unit)
        total += size
        if total >= overlap_tokens:
            break
    return tail


def build_child_chunks(
    parent: dict[str, Any], tokenizer=None, target_tokens: int = 280,
    max_tokens: int = 400, overlap_tokens: int = 50,
) -> list[dict[str, Any]]:
    units = _block_units({"type": "paragraph", "text": parent["text"]}, max_tokens, tokenizer)
    groups: list[list[str]] = []
    current: list[str] = []
    overlap_only = False

    def append_group(group: list[str]) -> None:
        if group and (not groups or group != groups[-1]):
            groups.append(list(group))

    for unit in units:
        projected = " ".join([*current, unit])
        if current and token_count(projected, tokenizer) > max_tokens:
            if not overlap_only:
                append_group(current)
            current = _tail_for_overlap(current, overlap_tokens, tokenizer)
            if current and token_count(" ".join([*current, unit]), tokenizer) > max_tokens:
                current = []
        current.append(unit)
        overlap_only = False
        if token_count(" ".join(current), tokenizer) >= target_tokens:
            append_group(current)
            current = _tail_for_overlap(current, overlap_tokens, tokenizer)
            overlap_only = True
    if current and (not overlap_only or not groups):
        append_group(current)

    title_path = parent.get("title_path", [])
    prefix = " > ".join(title_path)
    children: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        text = " ".join(group).strip()
        retrieval_text = f"{prefix}\n{text}" if prefix else text
        children.append({
            "chunk_id": f"{parent['parent_id']}:child:{index:04d}",
            "parent_id": parent["parent_id"], "child_index": index,
            "source": parent["source"], "title": parent["title"],
            "title_path": title_path,
            "page_start": parent.get("page_start"), "page_end": parent.get("page_end"),
            "text": text, "retrieval_text": retrieval_text,
            "token_count": token_count(text, tokenizer),
        })
    for child in children:
        child["child_count"] = len(children)
    return children


def load_and_chunk(filepath: str, tokenizer=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = Path(filepath).name
    blocks = parse_document(filepath)
    parents = build_parent_chunks(blocks, source, tokenizer=tokenizer)
    children = [child for parent in parents for child in build_child_chunks(parent, tokenizer=tokenizer)]
    return parents, children


def merge_child_texts(children: Iterable[dict[str, Any]]) -> str:
    """Join overlapping children without repeating identical sentences or rows."""
    units: list[str] = []
    seen: set[str] = set()
    for child in children:
        for unit in re.split(r"(?<=[.!?。！？])\s+|\n+", child["text"]):
            clean = unit.strip()
            key = re.sub(r"\s+", " ", clean).casefold()
            if clean and key not in seen:
                units.append(clean)
                seen.add(key)
    return "\n\n".join(units)


def expand_child_window(
    matched: dict[str, Any], children: Iterable[dict[str, Any]], window: int = 1,
) -> str:
    """Return the matched child together with adjacent siblings."""
    siblings = sorted(
        (child for child in children if child["parent_id"] == matched["parent_id"]),
        key=lambda child: int(child["child_index"]),
    )
    position = next(
        (index for index, child in enumerate(siblings) if child["chunk_id"] == matched["chunk_id"]),
        None,
    )
    if position is None:
        return matched["text"]
    selected = siblings[max(0, position - window):position + window + 1]
    return merge_child_texts(selected)
