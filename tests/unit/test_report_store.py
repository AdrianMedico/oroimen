"""Unit tests for ``hermes.jobs.report_store`` — LocalReportStore read path.

Slice 1C2: ``LocalReportStore`` is the sole reader of report content.
Tests cover the full surface: ``exists``, ``stat_size``, ``read``,
plus construction validation. Adversarial cases (oversize, invalid
UTF-8, symlink escape) are exercised per the brief §11.

All tests are offline: no network, no DB, no async I/O. ``tmp_path``
is used for filesystem setup. Symlink tests skip on Windows.
"""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path

import pytest

from hermes.jobs.exceptions import (
    InvalidJobIdError,
    InvalidUTF8Error,
    PathEscapeError,
    ReportTooLargeError,
    SymlinkEscapeError,
)
from hermes.jobs.report_store import LocalReportStore

_HELLO_BYTES = b"hello"  # UP012: bytes literal, not "hello".encode("utf-8")


# =====================================================================
# Construction validation
# =====================================================================


def test_construct_rejects_none_root(tmp_path: Path) -> None:
    """root=None → ValueError at construction."""
    with pytest.raises(ValueError):
        LocalReportStore(root=None, max_bytes=10_240)  # type: ignore[arg-type]


def test_construct_rejects_max_bytes_below_floor(tmp_path: Path) -> None:
    """max_bytes < 10_240 → ValueError at construction."""
    with pytest.raises(ValueError):
        LocalReportStore(root=tmp_path, max_bytes=10_239)


def test_construct_accepts_minimum_max_bytes(tmp_path: Path) -> None:
    """max_bytes == 10_240 (the floor) is accepted."""
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    assert store.max_bytes == 10_240
    assert store.root == tmp_path


# =====================================================================
# exists
# =====================================================================


def test_exists_returns_true_for_written_file(tmp_path: Path) -> None:
    """A real .md file in the root → exists() returns True."""
    job_id = "0123456789ab"
    (tmp_path / f"{job_id}.md").write_bytes(_HELLO_BYTES)
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    assert store.exists(job_id) is True


def test_exists_returns_false_for_missing(tmp_path: Path) -> None:
    """No file → exists() returns False (no exception)."""
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    assert store.exists("0123456789ab") is False


def test_exists_returns_false_for_invalid_job_id(tmp_path: Path) -> None:
    """Invalid job_id → exists() returns False (not raised)."""
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    assert store.exists("not-a-uuid") is False
    assert store.exists("") is False
    assert store.exists("0123..56789a") is False


# =====================================================================
# stat_size
# =====================================================================


def test_stat_size_within_limit(tmp_path: Path) -> None:
    """A small file → integer size returned."""
    job_id = "0123456789ab"
    (tmp_path / f"{job_id}.md").write_bytes(_HELLO_BYTES)
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    size = store.stat_size(job_id)
    assert size == len(_HELLO_BYTES)


def test_stat_size_raises_oversize(tmp_path: Path) -> None:
    """A file > max_bytes → ReportTooLargeError with job_id and size_bytes."""
    job_id = "0123456789ab"
    (tmp_path / f"{job_id}.md").write_bytes(b"x" * 11_000)
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    with pytest.raises(ReportTooLargeError) as excinfo:
        store.stat_size(job_id)
    assert excinfo.value.job_id == job_id
    assert excinfo.value.size_bytes == 11_000


def test_stat_size_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    """No file → FileNotFoundError (route maps to 500 report_unavailable)."""
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    with pytest.raises(FileNotFoundError):
        store.stat_size("0123456789ab")


# =====================================================================
# read
# =====================================================================


def test_read_returns_text(tmp_path: Path) -> None:
    """LocalReportStore.read returns the decoded string."""
    job_id = "0123456789ab"
    (tmp_path / f"{job_id}.md").write_bytes(_HELLO_BYTES)
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    text = store.read(job_id)
    assert text == "hello"


def test_read_raises_on_invalid_utf8(tmp_path: Path) -> None:
    """Raw bytes that are not valid UTF-8 → InvalidUTF8Error."""
    job_id = "0123456789ab"
    # b"\xff\xfe\x00\x01" is not valid UTF-8.
    (tmp_path / f"{job_id}.md").write_bytes(b"\xff\xfe\x00\x01")
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    with pytest.raises(InvalidUTF8Error) as excinfo:
        store.read(job_id)
    assert excinfo.value.job_id == job_id


def test_read_raises_on_symlink_escape(tmp_path: Path) -> None:
    """Symlink → outside the root → SymlinkEscapeError.

    Skipped on Windows because user-mode symlinks need elevation.
    """
    if os.name == "nt":
        pytest.skip("symlinks need elevation on Windows")
    job_id = "0123456789ab"
    # Create a "secret" file outside the root.
    secret_dir = tmp_path.parent / f"{tmp_path.name}_external_secret"
    secret_dir.mkdir(exist_ok=True)
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("top secret", encoding="utf-8")
    # Create a symlink at <root>/<id>.md → <secret_file>
    symlink = tmp_path / f"{job_id}.md"
    try:
        symlink.symlink_to(secret_file)
    except OSError:
        pytest.skip("symlink creation not supported in this environment")
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    try:
        with pytest.raises(SymlinkEscapeError) as excinfo:
            store.read(job_id)
        assert excinfo.value.job_id == job_id
    finally:
        with suppress(OSError):
            secret_file.unlink()
        with suppress(OSError):
            secret_dir.rmdir()


def test_read_rejects_oversize_via_stat(tmp_path: Path) -> None:
    """A file larger than max_bytes → ReportTooLargeError via read()."""
    job_id = "0123456789ab"
    (tmp_path / f"{job_id}.md").write_bytes(b"x" * 11_000)
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    with pytest.raises(ReportTooLargeError):
        store.read(job_id)


def test_read_rejects_path_truly_outside_root(tmp_path: Path) -> None:
    """A path truly outside the root → PathEscapeError or SymlinkEscapeError.

    The original test (round 1) used ``Path.resolve(strict=False)`` to
    build an "escape" path, but ``resolve`` normalizes ``..`` lexically
    so the escape was a no-op. This replacement creates a real file
    in ``tmp_path.parent`` and asserts that the store rejects it.

    The store's ``assert_inside_root`` catches any path that is NOT a
    sub-path of the root, regardless of how the caller constructed
    it. This is the defense-in-depth test for "what if a future
    refactor accidentally bypasses ``derive_report_path`` and passes
    a raw path to ``assert_inside_root``?".
    """
    # We use the job_id only to make the test name descriptive; the
    # actual exercise is on ``outside``, which lives in
    # ``tmp_path.parent`` (one level above the root).
    outside = tmp_path.parent / f"{tmp_path.name}_truly_outside.md"
    outside.write_bytes(b"data that should never be read")
    try:
        store = LocalReportStore(root=tmp_path, max_bytes=10_240)
        # The store must reject this path because it is NOT a
        # sub-path of ``tmp_path``. The exception class is either
        # PathEscapeError (lexical check) or SymlinkEscapeError
        # (realpath check) depending on the host filesystem; both
        # are part of the documented contract.
        # We exercise ``assert_inside_root`` directly because that is
        # the deterministic confinement surface; ``read`` would
        # re-derive from ``job_id`` (which is valid) and never see
        # the outside path.
        with pytest.raises((PathEscapeError, SymlinkEscapeError)):
            store.assert_inside_root(outside)
    finally:
        with suppress(OSError):
            outside.unlink()


def test_read_rejects_broken_symlink(tmp_path: Path) -> None:
    """A symlink to a non-existent target → exists() returns False.

    The route maps to 500 report_unavailable (logs may tag
    report_symlink_denied or report_missing). Skipped on Windows.
    """
    if os.name == "nt":
        pytest.skip("symlinks need elevation on Windows")
    job_id = "0123456789ab"
    # Broken symlink: target does not exist.
    symlink = tmp_path / f"{job_id}.md"
    try:
        symlink.symlink_to(tmp_path / "does_not_exist.md")
    except OSError:
        pytest.skip("symlink creation not supported in this environment")
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    # exists() returns False for a broken symlink.
    assert store.exists(job_id) is False
    # stat_size raises FileNotFoundError (target missing).
    with pytest.raises(FileNotFoundError):
        store.stat_size(job_id)


def test_read_rejects_invalid_job_id(tmp_path: Path) -> None:
    """Bad job_id → InvalidJobIdError on read."""
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    with pytest.raises(InvalidJobIdError):
        store.read("not-a-uuid")


# =====================================================================
# Bounded read (Slice 1C2 round 2): TOCTOU defense-in-depth
# =====================================================================


def test_read_is_bounded_even_if_file_grows_between_check_and_read(
    tmp_path: Path,
) -> None:
    """Even if a pre-read stat is stale, the read itself stays bounded.

    The new ``read`` method uses a single opened handle and reads at
    most ``max_bytes + 1`` bytes. If the file grew past ``max_bytes``
    between the open and the read, the read returns ``max_bytes + 1``
    bytes and we raise ``ReportTooLargeError`` without ever loading
    the unbounded payload into memory.

    This test exercises the bound by writing a file that is exactly
    ``max_bytes + 1`` bytes and asserting that ``read`` raises
    ``ReportTooLargeError``.
    """
    job_id = "0123456789ab"
    # max_bytes == 10_240; the file is max_bytes + 1 bytes long.
    payload = b"x" * (10_240 + 1)
    (tmp_path / f"{job_id}.md").write_bytes(payload)
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    with pytest.raises(ReportTooLargeError) as excinfo:
        store.read(job_id)
    assert excinfo.value.job_id == job_id
    # The reported size_bytes is the read length, not the on-disk
    # size. They are equal here because we wrote a single shot.
    assert excinfo.value.size_bytes == len(payload)


def test_read_accepts_file_exactly_at_max_bytes(tmp_path: Path) -> None:
    """A file of exactly ``max_bytes`` bytes is accepted (size == limit, not >)."""
    job_id = "0123456789ab"
    payload = b"x" * 10_240
    (tmp_path / f"{job_id}.md").write_bytes(payload)
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    text = store.read(job_id)
    assert text == "x" * 10_240
