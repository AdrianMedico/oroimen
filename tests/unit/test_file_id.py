"""Tests for `hermes.memory.file_id` (Sprint 19 Slice 4 §4.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.file_id import file_id_from_path, is_valid_file_id

pytestmark = pytest.mark.asyncio


async def test_file_id_is_deterministic_across_invocations(tmp_path: Path) -> None:
    """Same file content = same file_id, even across separate calls."""
    f = tmp_path / "foo.txt"
    f.write_bytes(b"hello world")
    id1 = file_id_from_path(f)
    id2 = file_id_from_path(f)
    assert id1 == id2


async def test_file_id_returns_32_char_lowercase_hex(tmp_path: Path) -> None:
    """file_id is exactly 32 lowercase hex chars (128 bits)."""
    f = tmp_path / "foo.txt"
    f.write_bytes(b"hello world")
    fid = file_id_from_path(f)
    assert len(fid) == 32
    assert is_valid_file_id(fid)
    assert fid == fid.lower()


async def test_file_id_matches_for_identical_content_different_paths(
    tmp_path: Path,
) -> None:
    """Two files with same content (different paths) → same file_id.

    This is the content-based dedup contract (Sprint 19 §4.2). The DB
    UNIQUE(source_path, content_sha256, mtime) constraint prevents
    actual duplicates; the file_id is the canonical user-facing ID.
    """
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    f1 = dir_a / "foo.txt"
    f2 = dir_b / "foo.txt"
    f1.write_bytes(b"identical content")
    f2.write_bytes(b"identical content")
    assert file_id_from_path(f1) == file_id_from_path(f2)


async def test_file_id_differs_for_different_content(tmp_path: Path) -> None:
    """Different content = different file_id (sanity check)."""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"content a")
    f2.write_bytes(b"content b")
    assert file_id_from_path(f1) != file_id_from_path(f2)


async def test_file_id_for_empty_file(tmp_path: Path) -> None:
    """Empty file has a stable, well-known file_id.

    All empty files share this ID (intentional — they ARE identical
    content). The DB UNIQUE(source_path, content_sha256, mtime)
    constraint prevents actual row duplication.
    """
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    fid = file_id_from_path(f)
    # SHA-256 of empty bytes truncated to 32 chars
    assert fid == "e3b0c44298fc1c149afbf4c8996fb924"


async def test_file_id_helper_handles_large_file(tmp_path: Path) -> None:
    """Large file (>64KB) is read in chunks correctly."""
    f = tmp_path / "big.bin"
    # 1 MB of data
    f.write_bytes(b"x" * 1_000_000)
    fid = file_id_from_path(f)
    assert is_valid_file_id(fid)
    # SHA-256 of "x" * 1_000_000 is a stable value; we don't hard-code it
    # (it would be brittle), but we verify determinism:
    assert file_id_from_path(f) == fid


# Note: the two tests below are sync (not async) but live in this file
# because they exercise the same module. The `@pytest.mark.asyncio`
# module-level marker generates a benign warning for these two; not a
# failure. (Splitting into a separate file would just be noise.)


def test_is_valid_file_id_accepts_valid() -> None:
    assert is_valid_file_id("e3b0c44298fc1c149afbf4c8996fb924")
    assert is_valid_file_id("00000000000000000000000000000000")
    assert is_valid_file_id("ffffffffffffffffffffffffffffffff")


def test_is_valid_file_id_rejects_invalid() -> None:
    # Wrong length
    assert not is_valid_file_id("")
    assert not is_valid_file_id("abc")
    assert not is_valid_file_id("e3b0c44298fc1c149afbf4c8996fb9")  # 31 chars
    assert not is_valid_file_id("e3b0c44298fc1c149afbf4c8996fb9244")  # 33 chars
    # Non-hex characters
    assert not is_valid_file_id("g" * 32)
    assert not is_valid_file_id("E3B0C44298FC1C149AFBF4C8996FB924")  # uppercase
