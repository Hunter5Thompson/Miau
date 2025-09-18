"""Markdown chunking utilities."""

from __future__ import annotations

import re
from typing import Sequence

from .models import Chunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _estimate_tokens(text: str) -> int:
    """Return a rough token estimate using whitespace separated words."""

    text = text.strip()
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _split_words(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    return words


def _join_words(words: Sequence[str]) -> str:
    return " ".join(words).strip()


def chunk_markdown(
    markdown: str,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Split Markdown text into semantic-aware chunks."""

    if chunk_tokens <= overlap_tokens:
        raise ValueError("chunk_tokens must be greater than overlap_tokens")

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    last_index = 0
    order = 0

    def flush_segment(segment_text: str) -> None:
        nonlocal order
        content = segment_text.strip()
        if not content:
            return
        words = _split_words(content)
        start = 0
        current_heading = heading_stack[-1][1] if heading_stack else None
        heading_path = [heading for _, heading in heading_stack]
        while start < len(words):
            end = min(start + chunk_tokens, len(words))
            chunk_words = words[start:end]
            chunk_text = _join_words(chunk_words)
            token_count = len(chunk_words) or _estimate_tokens(chunk_text)
            metadata = {
                "heading_path": heading_path,
                "token_start": start,
                "token_end": start + token_count,
            }
            chunks.append(
                Chunk(
                    text=chunk_text,
                    heading=current_heading,
                    tokens=token_count,
                    order=order,
                    metadata=metadata,
                )
            )
            order += 1
            if end >= len(words):
                break
            start = max(0, end - overlap_tokens)

    for match in _HEADING_RE.finditer(markdown):
        # Content before heading
        if match.start() > last_index:
            flush_segment(markdown[last_index : match.start()])
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        heading_stack = [entry for entry in heading_stack if entry[0] < level]
        heading_stack.append((level, heading_text))
        last_index = match.end()

    # Tail content
    if last_index < len(markdown):
        flush_segment(markdown[last_index:])

    return chunks


__all__ = ["chunk_markdown"]
