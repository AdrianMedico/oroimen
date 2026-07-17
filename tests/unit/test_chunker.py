"""RED tests for Chunker (Slice 2.5 part B).

Strategy: Markdown headers (##, ###) primary, Recursive Token fallback
if section > max_tokens. Overlap 10-15% (~100 tokens).

Reference: docs/TDD_VAULT_EMBEDDINGS.md §"Chunker"
"""

from __future__ import annotations

from hermes.memory.chunker import Chunker

# ----------------------------------------------------------------------------
# Chunker behaviour (TDD §"Chunker")
# ----------------------------------------------------------------------------


def test_chunker_splits_on_h2_header() -> None:
    """A document with multiple H2 sections produces one chunk per section.

    Markdown header primary strategy. Text under each `## header` becomes
    one chunk.
    """
    # NOTE: explicit `+` between string literals. Without it, Python
    # compile-time string concat + the `* 20` operator produce 20 copies
    # of the WHOLE block (including the header), so the regex finds 60
    # headers and 60 chunks instead of 3 sections / 3 chunks.
    text = (
        "## Section A\n"
        + "Content of section A.\n\n" * 20
        + "\n"
        + "## Section B\n"
        + "Content of section B.\n\n" * 20
        + "\n"
        + "## Section C\n"
        + "Content of section C.\n\n" * 20
    )
    chunker = Chunker(max_tokens=1000)
    chunks = chunker.chunk(text)

    assert len(chunks) == 3
    assert chunks[0].text.startswith("## Section A")
    assert chunks[1].text.startswith("## Section B")
    assert chunks[2].text.startswith("## Section C")


def test_chunker_emits_whole_section_if_under_max_tokens() -> None:
    """A section shorter than max_tokens is NOT split further."""
    text = "## Short Section\nThis is a short section."
    chunker = Chunker(max_tokens=1000)
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunker_falls_back_to_token_recursive_if_section_too_long() -> None:
    """A section > max_tokens triggers Recursive Token Chunking fallback.

    The fallback splits on paragraphs (`\\n\\n`) and recombines into
    chunks of max_tokens.
    """
    long_section = (
        "## Big Section\n"
        + ("This is paragraph one of a very long section.\n\n" * 5)
        + ("This is paragraph two of a very long section.\n\n" * 5)
        + ("This is paragraph three of a very long section.\n\n" * 5)
        + ("This is paragraph four of a very long section.\n\n" * 5)
    )
    chunker = Chunker(max_tokens=200, overlap_tokens=20)
    chunks = chunker.chunk(long_section)

    # Recursive fallback should produce multiple chunks (>= 2).
    assert len(chunks) >= 2
    # Each chunk's text should not exceed the max_tokens limit
    # (approximate by character count, since token counting is heuristic).
    # Allow up to 1.25x max_chars for the last chunk that has the carry-over.
    for chunk in chunks:
        # ~4 chars per token. max_tokens=200 ~ 800 chars. Allow 25% headroom.
        assert len(chunk.text) <= 1100, f"chunk too long: {len(chunk.text)} chars"


def test_chunker_preserves_overlap_window() -> None:
    """Chunks in token-recursive fallback have a ~10% overlap from previous chunk.

    Overlap is ~100 tokens for max_tokens=1000. The last N tokens of
    chunk[i] should appear at the start of chunk[i+1].
    """
    paragraphs = [f"This is paragraph {i} with content." * 5 for i in range(20)]
    text = "## Long\n" + "\n\n".join(paragraphs)
    chunker = Chunker(max_tokens=100, overlap_tokens=20)  # 20% overlap
    chunks = chunker.chunk(text)

    # Recursive fallback should produce >= 2 chunks.
    assert len(chunks) >= 2
    # Overlap check: the LAST overlap_chars chars of chunks[0] should be
    # at the START of chunks[1]. (The chunker carries the last
    # overlap_chars from the previous chunk's joined text.)
    overlap_chars = 20 * 4  # overlap_tokens=20, ~4 chars/token
    last_chars = chunks[0].text[-overlap_chars:]
    assert chunks[1].text.startswith(last_chars), (
        f"expected chunks[1] to start with the last {overlap_chars} chars "
        f"of chunks[0]; got chunks[1][:80]={chunks[1].text[:80]!r}, "
        f"chunks[0][-overlap:]={last_chars!r}"
    )


def test_chunker_returns_empty_list_for_empty_text() -> None:
    """Empty text → empty chunks list (no exception)."""
    chunker = Chunker()
    assert chunker.chunk("") == []


def test_chunker_does_not_split_mid_paragraph() -> None:
    """In Recursive Token fallback, paragraphs are atomic — never split mid-paragraph.

    Even if a paragraph alone is > max_tokens, it is emitted as one chunk
    (better to have one oversized chunk than a broken sentence).
    """
    text = "## Section\n" + "This is one very long paragraph that is larger than max_tokens. " * 100
    chunker = Chunker(max_tokens=50, overlap_tokens=10)  # 10 < 50
    chunks = chunker.chunk(text)

    # The single paragraph is the whole section; it should be one chunk.
    assert len(chunks) == 1
    assert chunks[0].text.startswith("## Section")


def test_chunker_deterministic_for_same_input() -> None:
    """Two calls with same input produce identical output (no random tie-breaking)."""
    text = "## A\n" + "Content. " * 100 + "\n\n## B\n" + "More content. " * 100
    chunker = Chunker(max_tokens=500)
    chunks_a = chunker.chunk(text)
    chunks_b = chunker.chunk(text)

    assert len(chunks_a) == len(chunks_b)
    for a, b in zip(chunks_a, chunks_b, strict=True):
        assert a.text == b.text
        assert a.chunk_index == b.chunk_index
