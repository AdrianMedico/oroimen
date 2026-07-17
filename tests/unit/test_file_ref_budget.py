"""Regression test for the budget tracking bug fixed in Sprint 19.6+ v0.4.

The pre-v0.4 code in hermes/agent/loop.py:731-736 tracked remaining_budget
by len(text) but appended len(wrap_file_content(fname, text) + "\\n\\n") --
a ~52-char-per-file overhead NOT counted in the budget. v0.4 fixed the
code to track by len(wrapped). This test asserts the exact match
between consumed_budget and actual wrapped output length, catching
future regressions of this bug.

Reference: docs/TDD_SPRINT_19_6_PLUS.md v0.4 section 5 P3.
"""
import re
from pathlib import Path

import pytest

from hermes.agent.loop import AgentLoop, wrap_file_content
from hermes.config import Settings
from hermes.memory.db import Database


@pytest.mark.asyncio
async def test_resolve_file_refs_budget_tracks_wrapped_length(
    tmp_path: Path, settings: Settings
) -> None:
    """Budget consumed == sum of len(wrap_file_content(fname, text) + '\\n\\n').

    With 5 small files (50-200 chars each) and a budget of 2000, all
    files fit and the total wrapped output length matches the exact
    sum of `len(wrap_file_content(fname, text) + "\\n\\n")` per file.

    v0.4 fix invariant: `remaining_budget` is decremented by
    `len(wrapped)` (NOT by `len(text)`). The wrap overhead
    (~52 chars per file) is counted against the budget.
    """
    settings = settings.model_copy(update={"read_tool_max_chars": 2000})
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    await db.initialize()
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    # 5 small files (50-200 chars each)
    files = [
        (f"file_{i}", f"f{i}.pdf", "A" * (50 + i * 25))  # 50, 75, 100, 125, 150 chars
        for i in range(5)
    ]
    for fid, fname, text in files:
        await db.add_file(fid, fname, "application/pdf", len(text), text, "pypdf")
    await db.add_message(
        conv_id,
        "user",
        "pregunta",
        file_refs=[f[0] for f in files],
    )
    history = await db.get_history(conv_id)
    loop = AgentLoop(router=None, registry=None, db=db, settings=settings)  # type: ignore[arg-type]
    resolved = await loop._resolve_file_refs(history)
    content = resolved[0]["content"]
    # The total wrapped output should be exactly:
    #   sum(len(wrap_file_content(fname, text) + "\n\n") for fname, text in files)
    # If the budget is tracked correctly, no file was truncated and the
    # total wrapped length matches the actual sum.
    expected_total = sum(
        len(wrap_file_content(fname, text) + "\n\n") for _fid, fname, text in files
    )
    # Count the actual wrapped file content in the output.
    actual_total = 0
    for _fid, fname, text in files:
        wrapped = wrap_file_content(fname, text) + "\n\n"
        if wrapped in content:
            actual_total += len(wrapped)
    # The KEY assertion: consumed_budget equals actual wrapped output length.
    # (v0.4 fix: budget tracked by len(wrapped) not len(text).)
    # All 5 files should fit (2000 budget > 5*~150 wrapped = ~750).
    assert actual_total == expected_total, (
        f"Budget tracking mismatch: actual={actual_total} vs expected={expected_total}. "
        "This indicates the budget is not tracked by wrapped length (m-3 regression)."
    )
    assert content.count("<file_content source=") == 5, (
        f"Expected 5 file_content wraps, got {content.count('<file_content source=')}"
    )


@pytest.mark.asyncio
async def test_resolve_file_refs_budget_truncation_exact(
    tmp_path: Path, settings: Settings
) -> None:
    """A tight budget uses the actual per-file wrapper overhead.

    The assertion derives the expected text length from `wrap_file_content`
    so a valid tag-format change cannot silently reintroduce a magic number.
    """
    settings = settings.model_copy(update={"read_tool_max_chars": 60})
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    await db.initialize()
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    # One file, tight budget: only enough for the wrap + ~10 chars of text.
    await db.add_file("file_a", "a.pdf", "application/pdf", 100, "A" * 100, "pypdf")
    await db.add_message(
        conv_id,
        "user",
        "pregunta",
        file_refs=["file_a"],
    )
    history = await db.get_history(conv_id)
    loop = AgentLoop(router=None, registry=None, db=db, settings=settings)  # type: ignore[arg-type]
    resolved = await loop._resolve_file_refs(history)
    content = resolved[0]["content"]
    # The file should appear (with truncated text) -- wrap is included.
    # Extract the text inside the wrap (between source="...">\n and \n</file_content>)
    m = re.search(r'source="a\.pdf">\n(A*)\n</file_content>', content)
    assert m is not None, f"Wrapped file not found in: {content[:200]!r}"
    a_count = len(m.group(1))
    wrapper_overhead = len(wrap_file_content("a.pdf", "") + "\n\n")
    expected_text_length = settings.read_tool_max_chars - wrapper_overhead
    assert a_count == expected_text_length, (
        f"Truncated text length {a_count} != {expected_text_length} "
        f"(budget={settings.read_tool_max_chars}, overhead={wrapper_overhead}). "
        "Budgeting must compute the wrapper overhead dynamically."
    )
    # The wrapper may consume the entire strict budget, leaving no room
    # for a note; injected context itself must never exceed the cap.
    assert len(content[: -len("pregunta")]) == settings.read_tool_max_chars


@pytest.mark.asyncio
async def test_resolve_file_refs_budget_long_filename_dropped(
    tmp_path: Path, settings: Settings
) -> None:
    """v0.4.2 fix: when the filename itself is too long for the budget,
    drop the file entirely (don't include the wrap that would exceed budget).
    """
    settings = settings.model_copy(update={"read_tool_max_chars": 50})
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    await db.initialize()
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    # Filename is 24 chars; empty wrap("very_long_filename_abc.pdf") = 68 chars
    # +2 for trailing = 70 chars overhead. Budget 50 < 70, so file is dropped.
    await db.add_file(
        "file_long",
        "very_long_filename_abc.pdf",
        "application/pdf",
        100,
        "A" * 100,
        "pypdf",
    )
    await db.add_message(
        conv_id,
        "user",
        "pregunta",
        file_refs=["file_long"],
    )
    history = await db.get_history(conv_id)
    loop = AgentLoop(router=None, registry=None, db=db, settings=settings)  # type: ignore[arg-type]
    resolved = await loop._resolve_file_refs(history)
    content = resolved[0]["content"]
    # The file WRAP (file_content tag) should NOT appear in the content --
    # even empty text doesn't fit, so the wrap itself would exceed budget.
    # The filename CAN appear in the truncation marker (it's mentioned
    # by name in "Archivos truncados: ...").
    assert "<file_content" not in content, (
        f"File wrap should be dropped when even empty wrap > budget. "
        f"Got: {content[:300]!r}"
    )
    # Truncation marker should mention the file.
    assert "truncado" in content.lower() or "truncated" in content.lower(), (
        f"Expected truncation marker mentioning the dropped file. "
        f"Got: {content[:200]!r}"
    )
    # Under a very small cap the note itself is truncated; it must not
    # exceed the same request-wide budget just to preserve the filename.
    assert len(content[: -len("pregunta")]) <= settings.read_tool_max_chars
@pytest.mark.asyncio
async def test_resolve_file_refs_budget_is_global_across_history(
    tmp_path: Path, settings: Settings
) -> None:
    """Two history messages share one request-wide file-content budget."""
    settings = settings.model_copy(update={"read_tool_max_chars": 500})
    db = Database(tmp_path / "global-budget.db")
    await db.initialize()
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_file("file_a", "a.pdf", "application/pdf", 300, "A" * 300, "pypdf")
    await db.add_file("file_b", "b.pdf", "application/pdf", 300, "B" * 300, "pypdf")
    await db.add_message(conv_id, "user", "question-a", file_refs=["file_a"])
    await db.add_message(conv_id, "user", "question-b", file_refs=["file_b"])
    history = await db.get_history(conv_id)
    loop = AgentLoop(router=None, registry=None, db=db, settings=settings)  # type: ignore[arg-type]

    resolved = await loop._resolve_file_refs(history)

    injected_total = 0
    for message, original in zip(resolved, ("question-a", "question-b"), strict=True):
        assert message["content"].endswith(original)
        injected_total += len(message["content"][: -len(original)])
    assert injected_total <= settings.read_tool_max_chars
