"""LocalReportStore: synchronous, bounded, UTF-8-strict read of report files.

Slice 1C2 ships a concrete ``LocalReportStore`` (no Protocol yet — the
boundary is documented for Stage 1 but not enforced). The store reads
Markdown reports that ``DeepResearchService._phase_write`` produces
(via ``tmp + fsync + os.replace``) and exposes a small sync surface:

    - ``derive_path(job_id) -> Path``: validation + canonical derivation
    - ``assert_inside_root(canonical)``: confinement check
    - ``exists(job_id) -> bool``: file presence (after confinement)
    - ``stat_size(job_id) -> int``: file size in bytes; raises
      ``ReportTooLargeError`` if the size exceeds ``max_bytes``
    - ``read(job_id) -> str``: full UTF-8-strict read; raises on
      size, symlink escape, UTF-8 invalidity, or any file system error

All errors map to the Slice 1C2 report exceptions. The HTTP route
translates ALL of them to 500 ``report_unavailable``; the exception
class is for internal classification only.

Sync interface: ``read`` is synchronous. The HTTP route (which IS
async) calls it via FastAPI's threaded path — the read is bounded by
``max_bytes`` (default 5 MiB) and is dispatched off the event loop by
``asyncio.to_thread`` in the route. The store does no async I/O itself.
"""

from __future__ import annotations

import os
from pathlib import Path

from hermes.jobs.exceptions import (
    InvalidJobIdError,
    InvalidUTF8Error,
    PathEscapeError,
    ReportTooLargeError,
    SymlinkEscapeError,
)
from hermes.jobs.report_paths import (
    assert_inside_root as _assert_inside_root_module,
)
from hermes.jobs.report_paths import (
    derive_report_path as _derive_report_path_module,
)

# Re-export the exceptions for convenience so the route imports from
# one place. The route also imports ``assert_inside_root`` /
# ``derive_report_path`` for tests that exercise the path layer
# directly. ``FileNotFoundError`` is the built-in; we expose it for
# callers that want a single import surface.
__all__ = [
    "LocalReportStore",
    "PathEscapeError",
    "SymlinkEscapeError",
    "ReportTooLargeError",
    "InvalidUTF8Error",
    "InvalidJobIdError",
    "FileNotFoundError",
]


class LocalReportStore:
    """Sync, bounded, UTF-8-strict read of local Markdown report files.

    Construction validates the inputs (root is a Path, max_bytes is a
    positive integer >= 10 KiB). Construction does NOT touch the
    filesystem; the actual ``mkdir`` happens at the composition root
    (``hermes.__main__._compose_deep_research_runtime``), per the
    fail-closed contract.
    """

    # Hard cap on max_bytes. Mirrors the Settings validation
    # ``le=52_428_800`` (50 MiB) — this constant is the floor / minimum
    # the store will accept regardless of caller. The Settings layer
    # applies the upper bound at parse time.
    _MIN_MAX_BYTES: int = 10_240  # 10 KiB

    def __init__(self, root: Path, max_bytes: int) -> None:
        if root is None:
            raise ValueError("root is required")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
            raise ValueError("max_bytes must be an int")
        if max_bytes < self._MIN_MAX_BYTES:
            raise ValueError(
                f"max_bytes must be >= {self._MIN_MAX_BYTES} (got {max_bytes})"
            )
        self._root = Path(root)
        self._max_bytes = int(max_bytes)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def derive_path(self, job_id: str) -> Path:
        """Return the canonical report path for ``job_id`` (no I/O).

        Raises:
            InvalidJobIdError: if ``job_id`` is not a valid UUID12 hex.
        """
        return _derive_report_path_module(self._root, job_id)

    def assert_inside_root(self, canonical: Path) -> None:
        """Verify ``canonical`` is inside ``self._root`` (lexical + symlink).

        Raises:
            PathEscapeError: lexical resolve escapes the root.
            SymlinkEscapeError: a symlink target escapes the root.
        """
        _assert_inside_root_module(canonical, self._root)

    def exists(self, job_id: str) -> bool:
        """Return True iff a real report file exists for ``job_id``.

        Path validation + confinement are run first; the result is
        False (NOT a raise) if the job_id is invalid or escapes the
        root. Symlink-broken case (target does not exist) is also False.
        """
        try:
            canonical = self.derive_path(job_id)
            self.assert_inside_root(canonical)
        except (InvalidJobIdError, PathEscapeError, SymlinkEscapeError):
            return False
        return canonical.exists()

    def stat_size(self, job_id: str) -> int:
        """Return the report's size in bytes; raise if > ``max_bytes``.

        Raises:
            InvalidJobIdError: bad UUID12.
            PathEscapeError / SymlinkEscapeError: traversal attempt.
            FileNotFoundError: file does not exist (raised with job_id).
            ReportTooLargeError: size > max_bytes.
        """
        canonical = self.derive_path(job_id)
        self.assert_inside_root(canonical)
        if not canonical.exists():
            # Use the standard FileNotFoundError with job_id in the
            # message. The route catches FileNotFoundError specifically
            # and translates to 500 report_unavailable.
            raise FileNotFoundError(f"report not found: {job_id}")
        # ``stat`` follows symlinks — but we already ran the symlink
        # confinement check above. A symlink that points to a file
        # outside the root would have raised SymlinkEscapeError; the
        # only legal state here is a regular file inside the root.
        size = canonical.stat().st_size
        if size > self._max_bytes:
            raise ReportTooLargeError(job_id=job_id, size_bytes=size)
        return size

    def read(self, job_id: str) -> str:
        """Read and decode the report as strict UTF-8.

        Sequence:
          1. derive_path + assert_inside_root
          2. stat_size (catches missing + over-size)
          3. read bytes
          4. strict UTF-8 decode (raises ``InvalidUTF8Error`` on failure)

        Raises:
            InvalidJobIdError: bad UUID12.
            PathEscapeError / SymlinkEscapeError: traversal attempt.
            FileNotFoundError: file does not exist.
            ReportTooLargeError: file > max_bytes.
            InvalidUTF8Error: file is not valid UTF-8.
        """
        # (1)+(2) — size check first so an oversize file is rejected
        # WITHOUT reading the entire body into memory.
        self.stat_size(job_id)
        canonical = self.derive_path(job_id)
        # (3) read raw bytes. ``open(..., "rb")`` does not follow
        # symlinks (the OS resolves at open time, so a symlink whose
        # target escaped the root would have already raised
        # SymlinkEscapeError above).
        try:
            raw = canonical.read_bytes()
        except FileNotFoundError as exc:
            # Race: file deleted between stat and read. Surface as
            # not-found; the route maps to 500 report_unavailable.
            raise FileNotFoundError(f"report not found: {job_id}") from exc
        # (4) strict UTF-8 decode. ``strict`` mode raises ``UnicodeDecodeError``
        # on any invalid byte sequence; we translate to ``InvalidUTF8Error``.
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise InvalidUTF8Error(job_id=job_id) from exc
        return text
