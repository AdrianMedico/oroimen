"""Cross-platform path serialization helper (Sprint 19 §7).

Per Gemini Sprint 19 review: every path written to SQLite, JSON manifest,
log payload, or InfluxDB tag MUST use POSIX format. This helper is the
ONLY sanctioned way to convert at the persistence boundary.

Behavior:
- `to_posix(p)` accepts str, os.PathLike, or Path → returns POSIX-style string.
  Backslashes in Windows paths are converted to forward slashes.
- `from_posix(s)` round-trips back to a platform-aware Path.
- No information loss on round-trip (other than drive-letter casing on Windows).

Cross-platform invariant: any code that constructs a path for persistence
MUST go through `to_posix()`. Code reviewed via §11 audit checklist in TDD.
"""

from __future__ import annotations

from pathlib import Path


def to_posix(p: str | Path) -> str:
    """Convert a path to POSIX-format string for cross-platform persistence.

    POSIX format uses forward slashes exclusively. On Windows, drive letters
    remain (e.g., 'C:/Users/foo').

    >>> from pathlib import PurePosixPath, PureWindowsPath
    >>> to_posix(PurePosixPath('var/data/foo.pdf'))
    'var/data/foo.pdf'
    >>> to_posix(PureWindowsPath(r'C:\\Users\\foo\\bar.pdf'))
    'C:/Users/foo/bar.pdf'
    """
    return Path(p).as_posix()


def from_posix(s: str) -> Path:
    """Convert a persisted POSIX-format string back to a platform-aware Path.

    Idempotent with `to_posix()`: `from_posix(to_posix(p)) == Path(p)` for any
    path that Path can construct on the current platform.

    >>> from_posix("var/data/foo.pdf")
    PosixPath('var/data/foo.pdf')
    """
    return Path(s)
