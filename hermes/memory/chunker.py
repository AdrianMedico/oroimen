"""chunker.py — Markdown header primary, Recursive Token fallback.

Reference: docs/TDD_VAULT_EMBEDDINGS.md §"Chunker"

Strategy:
1. Parse `text` for Markdown headers (`^#{1,6}\\s`).
2. For each section (between headers), check its token count.
3. If section <= max_tokens, emit as one chunk.
4. If section > max_tokens, apply Recursive Token chunking:
   - Split on paragraphs (`\\n\\n`).
   - Recombine paragraphs into chunks of max_tokens with overlap_tokens
     from the end of the previous chunk.
5. Concatenate all chunks in document order.

Tokens are estimated with a 4-chars-per-token heuristic
(conservative for English prose). For non-English content, this
underestimates; the chunker errs on the side of slightly larger
chunks (acceptable for embeddings).

WHY Markdown primary: Slice 1.5's TIER 1.5 (LAN worker) extracts
text to Markdown format. The header hierarchy is already
semantically meaningful. Splitting by header preserves document
structure. The Recursive Token fallback only kicks in when a
single section is too large (> max_tokens), which is rare for
well-structured Markdown.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

# Approximate token-to-character ratio for English prose.
# 1 token ≈ 4 chars. Conservative: slightly under-estimates tokens,
# so chunks end up slightly LARGER than max_tokens in characters.
CHARS_PER_TOKEN: Final[int] = 4

# Markdown header regex: `^#{1,6}\s` (anchored to line start).
_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Paragraph separator: one or more blank lines.
_PARAGRAPH_RE: Final[re.Pattern[str]] = re.compile(r"\n\s*\n")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of the source text, ready for embedding.

    `paragraph_count` is the number of paragraphs in this chunk
    (after the Recursive Token fallback). Useful for overlap checks
    in tests.
    """

    chunk_index: int
    text: str
    paragraph_count: int
    is_fallback_split: bool  # True if produced by Recursive Token fallback


class Chunker:
    """Markdown-header-aware text chunker with Recursive Token fallback.

    Usage:
        chunker = Chunker(max_tokens=1000, overlap_tokens=100)
        chunks = chunker.chunk(text)
    """

    def __init__(
        self,
        max_tokens: int = 1000,
        overlap_tokens: int = 100,
        encoding_chars_per_token: int = CHARS_PER_TOKEN,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError(
                f"overlap_tokens must be in [0, max_tokens); "
                f"got {overlap_tokens} with max_tokens={max_tokens}"
            )
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._chars_per_token = encoding_chars_per_token
        self._max_chars = max_tokens * encoding_chars_per_token
        self._overlap_chars = overlap_tokens * encoding_chars_per_token

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def overlap_tokens(self) -> int:
        return self._overlap_tokens

    def chunk(self, text: str) -> list[Chunk]:
        """Split `text` into chunks. Order = document order.

        Empty text returns an empty list. No exception.
        """
        if not text or not text.strip():
            return []

        sections = self._split_into_sections(text)
        chunks: list[Chunk] = []
        for section in sections:
            if self._char_len(section) <= self._max_chars:
                # Section fits in one chunk.
                chunks.append(self._make_chunk(section, is_fallback=False))
            else:
                # Recursive Token fallback: split section by paragraphs.
                chunks.extend(self._recursive_token_fallback(section))
        # Re-index.
        return [
            Chunk(
                chunk_index=i,
                text=c.text,
                paragraph_count=c.paragraph_count,
                is_fallback_split=c.is_fallback_split,
            )
            for i, c in enumerate(chunks)
        ]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _char_len(self, text: str) -> int:
        return len(text)

    def _split_into_sections(self, text: str) -> list[str]:
        """Split text at Markdown headers, returning a list of sections.

        Each section INCLUDES the header line. The first section may
        have no header (prose before any header).
        """
        matches = list(_HEADER_RE.finditer(text))
        if not matches:
            return [text.strip()]

        sections: list[str] = []
        # Preamble (text before the first header).
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(preamble)

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)
        return sections

    def _recursive_token_fallback(self, section: str) -> list[Chunk]:
        """Split an oversized section by paragraphs, recombining into
        max_tokens-sized chunks with overlap.

        Algorithm:
        1. Split section into paragraphs (split on `\\n\\n`).
        2. Iterate paragraphs; accumulate into a buffer until adding the
           next paragraph would exceed max_chars.
        3. Emit a chunk; carry the last `overlap_chars` characters into
           the next buffer.
        4. If a single paragraph is itself > max_chars, emit it as one
           oversized chunk (better than splitting mid-sentence).
        """
        paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(section) if p.strip()]
        if not paragraphs:
            return [self._make_chunk(section, is_fallback=True)]

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_chars = 0

        def emit() -> None:
            nonlocal buffer, buffer_chars
            if not buffer:
                return
            chunks.append(
                Chunk(
                    chunk_index=0,  # re-indexed by caller
                    text="\n\n".join(buffer),
                    paragraph_count=len(buffer),
                    is_fallback_split=True,
                )
            )
            # Carry the last overlap_chars from the buffer into next.
            joined = "\n\n".join(buffer)
            if len(joined) > self._overlap_chars:
                carry_text = joined[-self._overlap_chars :]
                # Reset buffer to just the carry text.
                buffer = [carry_text]
                buffer_chars = len(carry_text)
            else:
                buffer = []
                buffer_chars = 0

        for p in paragraphs:
            p_len = len(p)
            if p_len > self._max_chars:
                # Single paragraph > max_chars: emit the buffer first
                # (so we keep ordering), then emit the oversized paragraph
                # itself as a single chunk.
                emit()
                chunks.append(
                    Chunk(
                        chunk_index=0,  # re-indexed by caller
                        text=p,
                        paragraph_count=1,
                        is_fallback_split=True,
                    )
                )
                continue
            if buffer_chars + p_len > self._max_chars and buffer:
                emit()
            buffer.append(p)
            buffer_chars += p_len

        # Emit any remaining.
        if buffer:
            chunks.append(
                Chunk(
                    chunk_index=0,  # re-indexed by caller
                    text="\n\n".join(buffer),
                    paragraph_count=len(buffer),
                    is_fallback_split=True,
                )
            )

        return chunks

    def _make_chunk(self, text: str, *, is_fallback: bool) -> Chunk:
        paragraph_count = len([p for p in _PARAGRAPH_RE.split(text) if p.strip()])
        return Chunk(
            chunk_index=0,  # re-indexed by caller
            text=text,
            paragraph_count=paragraph_count,
            is_fallback_split=is_fallback,
        )
