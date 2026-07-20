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

Design notes (Slice 1C2):
- ``derive_report_path`` returns the CANDIDATE path constructed as
  ``root / <job_id>.md`` WITHOUT calling ``resolve``. Calling
  ``resolve(strict=False)`` would FOLLOW the symlink at that location
  and erase the distinction between the lexical and the realpath
  confinement checks. Derivation must leave the symlink identity
  intact so that ``assert_inside_root`` can identify it as a symlink
  and apply the realpath branch deterministically.
- ``assert_inside_root`` performs TWO distinct checks in order:
  1. Lexical confinement (no symlink follow) — the candidate path
     must be a sub-path of the resolved root.
  2. Realpath confinement (symlink follow) — IF the candidate is a
     symlink, the resolved real path must also be inside the root.
  The two checks produce different exception classes
  (PathEscapeError vs SymlinkEscapeError) so the HTTP route can log
  the correct internal category while still returning the same
  redacted 500 envelope.
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
    """Return the candidate path ``<root>/<job_id>.md`` for the given job_id.

    The returned path is the CANDIDATE constructed by joining ``root``
    and the validated filename. ``resolve`` is NOT called here because
    that would follow symlinks and erase the distinction between
    lexical and realpath confinement. The caller MUST run
    ``assert_inside_root`` on the result before any read; this function
    only validates ``job_id`` and constructs the path.

    Raises:
        InvalidJobIdError: if ``job_id`` is not a valid UUID12 hex token.
    """
    _validate_job_id(job_id)
    return root / f"{job_id}.md"


def assert_inside_root(canonical: Path, root: Path) -> None:
    """Verify that ``canonical`` stays inside ``root`` (lexical + realpath).

    Two checks, in order:
    1. Lexical confinement: ``canonical`` (compared as a pure path
       string, no symlink follow) must be a sub-path of
       ``root.resolve(strict=False)``. Catches ``../`` traversal
       via the URL path or via a manually-constructed candidate
       that the caller passed in.
    2. Realpath confinement: IF ``canonical`` is a symlink,
       ``os.path.realpath`` (which follows ALL symlinks recursively)
       must ALSO be inside ``root.resolve()``. Catches a symlink at
       ``<root>/<id>.md`` whose target escapes the root.

    Raises:
        PathEscapeError: the lexical resolve escapes the root.
        SymlinkEscapeError: a symlink target escapes the root.
        InvalidJobIdError: not raised here (job_id already validated at
            derivation time); provided for completeness in the call chain.
    """
    root_resolved = root.resolve(strict=False)

    # (1) Lexical confinement — NO symlink follow at this step. We
    # compare paths as pure strings after a lexical normalize
    # (``os.path.normpath`` only normalizes separators and ``..``; it
    # does NOT touch the filesystem and does NOT follow symlinks).
    # If root is relative, we anchor both to the same cwd via
    # ``Path.resolve(strict=False)`` on the lexical form of the
    # candidate so the comparison is apples-to-apples. This
    # ``resolve(strict=False)`` is purely lexical on the path STRING;
    # it does not open the file or follow symlinks because we are
    # operating on a ``normpath``-ed string and only normalizing
    # textual ``..`` segments.
    canonical_lex_str = os.path.normpath(str(canonical))
    canonical_lex = Path(canonical_lex_str)
    # Make both sides absolute-anchored (or both relative) so the
    # relative_to check is meaningful regardless of how the caller
    # constructed the paths. We do this with strict=False to avoid
    # following symlinks; we are only resolving the textual ``..``
    # segments against the current working directory.
    canonical_lex_anchored = canonical_lex if canonical_lex.is_absolute() else canonical_lex.resolve(strict=False)
    root_resolved_for_check = root_resolved if root_resolved.is_absolute() else Path(os.path.normpath(str(root_resolved)))

    try:
        canonical_lex_anchored.relative_to(root_resolved_for_check)
    except ValueError as exc:
        raise PathEscapeError(canonical.name.removesuffix(".md")) from exc

    # (2) Realpath confinement — only if the candidate is a symlink.
    # We check ``is_symlink()`` first to avoid the cost of an
    # ``os.path.realpath`` call on every read of a regular file.
    # The ``is_symlink()`` call does not follow the link; it inspects
    # the directory entry itself.
    if canonical.is_symlink():
        real = Path(os.path.realpath(canonical))
        if not real.is_relative_to(root_resolved):
            raise SymlinkEscapeError(canonical.name.removesuffix(".md"))


__all__ = [
    "assert_inside_root",
    "derive_report_path",
]
