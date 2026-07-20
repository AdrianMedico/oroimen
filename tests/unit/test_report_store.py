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
from pathlib import Path

import pytest

from hermes.jobs.exceptions import (
    InvalidJobIdError,
    InvalidUTF8Error,
    ReportTooLargeError,
    SymlinkEscapeError,
)
from hermes.jobs.report_store import LocalReportStore


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
    (tmp_path / f"{job_id}.md").write_text("hello", encoding="utf-8")
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
    (tmp_path / f"{job_id}.md").write_text("hello", encoding="utf-8")
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    size = store.stat_size(job_id)
    assert size == len("hello".encode("utf-8"))


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
    (tmp_path / f"{job_id}.md").write_text("hello", encoding="utf-8")
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
        # Cleanup.
        try:
            secret_file.unlink()
            secret_dir.rmdir()
        except OSError:
            pass


def test_read_rejects_oversize_via_stat(tmp_path: Path) -> None:
    """A file larger than max_bytes → ReportTooLargeError via read()."""
    job_id = "0123456789ab"
    (tmp_path / f"{job_id}.md").write_bytes(b"x" * 11_000)
    store = LocalReportStore(root=tmp_path, max_bytes=10_240)
    with pytest.raises(ReportTooLargeError):
        store.read(job_id)


def test_read_rejects_when_path_resolves_outside_root_via_dotdot(tmp_path: Path) -> None:
    """Manually constructing a path outside the root → caught by the store.

    ``derive_report_path`` rejects bad job_ids, so this is a defense
    in depth test. We construct a path that lives at
    ``<root>/<job_id>.md`` (valid by derivation) but the parent's
    ``resolve(strict=False)`` was somehow tampered with to escape.
    The store's ``assert_inside_root`` catches it.
    """
    job_id = "0123456789ab"
    # A file outside the root (in tmp_path's parent).
    outside = tmp_path.parent / f"{tmp_path.name}_escape.md"
    outside.write_text("escaped", encoding="utf-8")
    try:
        # Manually create a path object that LOOKS like it's inside
        # the root but resolves outside. This is only possible by
        # bypassing derive_path; the store is robust against it.
        inside_path = tmp_path / f"{job_id}.md"
        # Build a candidate that resolves via .. out of the root.
        escape = (tmp_path / job_id / ".." / f"{job_id}.md").resolve(strict=False)
        assert not escape.is_relative_to(tmp_path.resolve())
        store = LocalReportStore(root=tmp_path, max_bytes=10_240)
        with pytest.raises(Exception):  # PathEscapeError or SymlinkEscapeError
            store.read.__wrapped__(job_id) if hasattr(store.read, "__wrapped__") else None
        # The simpler direct test: stat_size on the escape path
        from hermes.jobs.exceptions import PathEscapeError, SymlinkEscapeError

        with pytest.raises((PathEscapeError, SymlinkEscapeError)):
            store.assert_inside_root(escape)
    finally:
        try:
            outside.unlink()
        except OSError:
            pass


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
