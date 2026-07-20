"""Unit tests for ``hermes.jobs.report_paths`` — path derivation and confinement.

Slice 1C2: the read path for ``GET /v1/jobs/{job_id}/report`` is derived
exclusively from ``<root>/<job_id>.md``. These tests exercise the two
helper functions:

    - ``derive_report_path(root, job_id)`` — validates UUID12 hex and
      returns the canonical path.
    - ``assert_inside_root(canonical, root)`` — verifies lexical
      resolve and symlink realpath stay inside the root.

All exceptions are from ``hermes.jobs.exceptions``. The tests are
offline: no network, no DB, no async. ``tmp_path`` is used for any
filesystem setup (symlinks); symlink tests are skipped on Windows
(``os.name == "nt"``) because user-mode symlinks need elevated
privileges on most Windows configurations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hermes.jobs.exceptions import (
    InvalidJobIdError,
    PathEscapeError,
    SymlinkEscapeError,
)
from hermes.jobs.report_paths import (
    assert_inside_root,
    derive_report_path,
)


# =====================================================================
# derive_report_path — happy path
# =====================================================================


def test_derive_path_happy_path(tmp_path: Path) -> None:
    """UUID12 hex → <root>/<job_id>.md."""
    job_id = "0123456789ab"
    out = derive_report_path(tmp_path, job_id)
    assert out == (tmp_path / f"{job_id}.md").resolve(strict=False)
    # Filename is exactly the job_id + .md
    assert out.name == "0123456789ab.md"


def test_derive_path_accepts_typical_uuid12(tmp_path: Path) -> None:
    """A typical UUID12 hex from uuid4().hex[:12] is accepted."""
    # e.g. "abcdef012345"
    out = derive_report_path(tmp_path, "abcdef012345")
    assert out.name == "abcdef012345.md"


# =====================================================================
# derive_report_path — validation
# =====================================================================


def test_derive_path_rejects_empty_string(tmp_path: Path) -> None:
    """Empty job_id → InvalidJobIdError."""
    with pytest.raises(InvalidJobIdError):
        derive_report_path(tmp_path, "")


def test_derive_path_rejects_nul_byte(tmp_path: Path) -> None:
    """NUL byte in job_id → InvalidJobIdError (NUL-truncation attack)."""
    with pytest.raises(InvalidJobIdError):
        derive_report_path(tmp_path, "abc\x00def4567")


def test_derive_path_rejects_too_long(tmp_path: Path) -> None:
    """13-char job_id → InvalidJobIdError."""
    with pytest.raises(InvalidJobIdError):
        derive_report_path(tmp_path, "0123456789abc")  # 13 chars


def test_derive_path_rejects_too_short(tmp_path: Path) -> None:
    """11-char job_id → InvalidJobIdError."""
    with pytest.raises(InvalidJobIdError):
        derive_report_path(tmp_path, "01234567890")  # 11 chars


def test_derive_path_rejects_uppercase_hex(tmp_path: Path) -> None:
    """Uppercase hex is NOT accepted (lowercase only per the pattern)."""
    with pytest.raises(InvalidJobIdError):
        derive_report_path(tmp_path, "0123456789AB")  # uppercase


def test_derive_path_rejects_non_hex(tmp_path: Path) -> None:
    """Non-hex character (e.g. 'g') → InvalidJobIdError."""
    with pytest.raises(InvalidJobIdError):
        derive_report_path(tmp_path, "0123456789gz")


def test_derive_path_rejects_dotdot(tmp_path: Path) -> None:
    """'..' in job_id → InvalidJobIdError (defense before path resolution)."""
    with pytest.raises(InvalidJobIdError):
        derive_report_path(tmp_path, "0123..56789a")


def test_derive_path_rejects_path_separator(tmp_path: Path) -> None:
    """Path separator in job_id → InvalidJobIdError."""
    with pytest.raises(InvalidJobIdError):
        derive_report_path(tmp_path, "0123/456789ab")


def test_derive_path_rejects_non_string(tmp_path: Path) -> None:
    """Non-str types (e.g. int) → InvalidJobIdError."""
    with pytest.raises(InvalidJobIdError):
        derive_report_path(tmp_path, 123456789012)  # type: ignore[arg-type]


# =====================================================================
# assert_inside_root — happy path
# =====================================================================


def test_assert_inside_root_happy(tmp_path: Path) -> None:
    """A canonical path inside the root is silent."""
    job_id = "0123456789ab"
    canonical = derive_report_path(tmp_path, job_id)
    # No exception expected.
    assert_inside_root(canonical, tmp_path)


def test_assert_inside_root_resolves_to_same_when_no_symlink(tmp_path: Path) -> None:
    """For a freshly created canonical file, realpath == canonical."""
    job_id = "0123456789ab"
    canonical = derive_report_path(tmp_path, job_id)
    # No file exists yet — but resolve(strict=False) still works.
    # realpath on a non-existent path returns its lexical resolve.
    real = os.path.realpath(canonical)
    assert real.startswith(str(tmp_path.resolve()))


# =====================================================================
# assert_inside_root — rejects escape via resolve
# =====================================================================


def test_assert_inside_root_rejects_dotdot_via_resolve(tmp_path: Path) -> None:
    """A path that resolves outside the root → PathEscapeError.

    We construct the escape vector manually because derive_report_path
    refuses bad job_ids. The route is the only legitimate caller and
    it MUST derive from data_root + job_id, so this test is a defense
    in depth: if the caller passes a manually-constructed path that
    escapes, the assert catches it.
    """
    # Create the job dir under tmp_path, then manually build a path
    # that traverses out via .. and use the real resolve() to confirm
    # the escape is caught.
    job_id = "0123456789ab"
    # A path that lives at <tmp>/<job_id>.md — that's inside the root.
    inside = derive_report_path(tmp_path, job_id)

    # Synthesize an escape: <tmp>/<job_id>.md/../../../etc/passwd
    escape_candidate = (tmp_path / job_id / ".." / ".." / "etc" / "passwd").resolve(strict=False)
    # Sanity: this resolves somewhere outside tmp_path.
    assert not escape_candidate.is_relative_to(tmp_path.resolve())

    # assert_inside_root must raise PathEscapeError for the escape.
    with pytest.raises(PathEscapeError):
        assert_inside_root(escape_candidate, tmp_path)

    # And the happy path (inside) still passes.
    assert_inside_root(inside, tmp_path)


# =====================================================================
# assert_inside_root — symlink confinement (skip on Windows)
# =====================================================================


@pytest.mark.skipif(os.name == "nt", reason="symlinks need elevation on Windows")
def test_assert_inside_root_rejects_symlink_to_outside(tmp_path: Path) -> None:
    """A symlink at <root>/<id>.md → outside the root → SymlinkEscapeError.

    The route derives the path as <root>/<id>.md, then assert_inside_root
    follows the symlink via os.path.realpath and checks the target is
    also inside the root. A symlink pointing to an external file is
    caught here even if the lexical resolve is fine.
    """
    job_id = "0123456789ab"
    # Create a "secret" file outside the root.
    secret_dir = tmp_path.parent / f"{tmp_path.name}_external"
    secret_dir.mkdir(exist_ok=True)
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("top secret", encoding="utf-8")

    # Create a symlink at <root>/<id>.md → <secret_file>
    symlink = tmp_path / f"{job_id}.md"
    try:
        symlink.symlink_to(secret_file)
    except OSError:
        pytest.skip("symlink creation not supported in this environment")

    # Derive the canonical path (the symlink itself).
    canonical = derive_report_path(tmp_path, job_id)
    # Sanity: lexical resolve is INSIDE root.
    assert canonical.is_relative_to(tmp_path.resolve())

    # assert_inside_root must raise SymlinkEscapeError.
    with pytest.raises(SymlinkEscapeError):
        assert_inside_root(canonical, tmp_path)

    # Cleanup the external file we created.
    try:
        secret_file.unlink()
        secret_dir.rmdir()
    except OSError:
        pass


@pytest.mark.skipif(os.name == "nt", reason="symlinks need elevation on Windows")
def test_assert_inside_root_allows_symlink_inside(tmp_path: Path) -> None:
    """A symlink inside the root is allowed (target is also inside)."""
    job_id = "0123456789ab"
    # Create a real file inside the root.
    real_target = tmp_path / "real_target.md"
    real_target.write_text("hello", encoding="utf-8")

    # Create a symlink at <root>/<id>.md → <real_target>
    symlink = tmp_path / f"{job_id}.md"
    try:
        symlink.symlink_to(real_target)
    except OSError:
        pytest.skip("symlink creation not supported in this environment")

    canonical = derive_report_path(tmp_path, job_id)
    # No exception expected — the symlink stays inside the root.
    assert_inside_root(canonical, tmp_path)
