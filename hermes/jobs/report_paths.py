"""Path derivation and confinement for Deep Research report content (Slice 1C2).

The report read path is derived EXCLUSIVELY from
``{data_root}/{job_id}.md`` where ``job_id`` is a 12-character lowercase
hex token (UUID12). The DB ``output_path`` column is NEVER consulted to
choose the file being read — it stays in the schema as an internal
completion/recovery marker only.

The functions here are pure path operations; no I/O beyond the minimum
required to resolve and check paths. They raise the Slice 1C2 report
exceptions (see ``hermes.jobs.exceptions``) so the HTTP route can
translate them to a single 500 ``report_unavailable`` envelope.

Privacy contract:
- The exception messages never contain filesystem paths.
- The exceptions carry ``job_id`` (12 hex chars) only.
- Logs at the call site may tag the stable internal category
  (report_path_escape, report_symlink_denied, report_invalid_job_id).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from hermes.jobs.exceptions import (
    InvalidJobIdError,
    PathEscapeError,
    SymlinkEscapeError,
)

# UUID12 hex: exactly 12 lowercase hex characters, no NUL, no '..',
# no whitespace, no path separators. The regex anchors on both ends.
# The route applies the same constraint via Path(pattern=...), but
# defense in depth here catches callers that bypass the route.
_UUID12_RE = re.compile(r"^[0-9a-f]{12}$")


def _validate_job_id(job_id: str) -> None:
    """Raise ``InvalidJobIdError`` unless ``job_id`` is a valid UUID12 hex token.

    Rules:
    - 12 lowercase hex characters (no uppercase).
    - No NUL, no '..', no whitespace, no path separators.
    - Type must be str (NOT bytes); reject other types outright.
    """
    if not isinstance(job_id, str):
        raise InvalidJobIdError(str(job_id))
    if "\x00" in job_id or ".." in job_id:
        raise InvalidJobIdError(job_id)
    if not _UUID12_RE.match(job_id):
        raise InvalidJobIdError(job_id)


def derive_report_path(root: Path, job_id: str) -> Path:
    """Return the canonical path ``<root>/<job_id>.md`` for the given job_id.

    The path is resolved (symlinks NOT followed yet — the read path
    follows symlinks explicitly via ``os.path.realpath`` inside
    ``assert_inside_root`` so the failure mode is observable). The
    caller MUST call ``assert_inside_root`` on the result before any
    read; this function only validates ``job_id`` and constructs the
    path.

    Raises:
        InvalidJobIdError: if ``job_id`` is not a valid UUID12 hex token.

    Note:
        ``strict=False`` because the file may not exist yet (read path
        must distinguish "missing" from "exists"; the store does that
        via ``.exists()`` separately).
    """
    _validate_job_id(job_id)
    return (root / f"{job_id}.md").resolve(strict=False)


def assert_inside_root(canonical: Path, root: Path) -> None:
    """Verify that ``canonical`` stays inside ``root`` after both symlink
    re-resolution and lexical resolve.

    Two checks, in order:
    1. Lexical resolve: ``canonical`` must be a sub-path of
       ``root.resolve()``. Catches ``../`` traversal via the URL path.
    2. Symlink realpath: ``os.path.realpath(canonical)`` (which follows
       ALL symlinks recursively) must also be inside ``root.resolve()``.
       Catches a symlink at the canonical path whose target escapes the
       root.

    Raises:
        PathEscapeError: if the lexical resolve escapes the root.
        SymlinkEscapeError: if a symlink's real target escapes the root.
        InvalidJobIdError: not raised here (job_id already validated at
            derivation time); provided for completeness in the call chain.
    """
    root_resolved = root.resolve(strict=False)
    canonical_resolved = canonical.resolve(strict=False)

    # (1) Lexical confinement. Use Path.is_relative_to when available
    # (Python 3.9+); fall back to string comparison otherwise.
    if hasattr(canonical_resolved, "is_relative_to"):
        if not canonical_resolved.is_relative_to(root_resolved):
            raise PathEscapeError(canonical.name.removesuffix(".md"))
    else:  # pragma: no cover - Python 3.9+ only
        try:
            canonical_resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise PathEscapeError(canonical.name.removesuffix(".md")) from exc

    # (2) Symlink realpath confinement. If the canonical path is a
    # symlink, realpath() follows it; the target must ALSO be inside
    # root. This catches a symlink that lives at <root>/<id>.md but
    # points to /etc/passwd.
    real = Path(os.path.realpath(canonical_resolved))
    if hasattr(real, "is_relative_to"):
        if not real.is_relative_to(root_resolved):
            raise SymlinkEscapeError(canonical.name.removesuffix(".md"))
    else:  # pragma: no cover - Python 3.9+ only
        try:
            real.relative_to(root_resolved)
        except ValueError as exc:
            raise SymlinkEscapeError(canonical.name.removesuffix(".md")) from exc


__all__ = [
    "assert_inside_root",
    "derive_report_path",
]
