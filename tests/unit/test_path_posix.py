"""Tests for hermes/util/paths.py (Sprint 19 §7 cross-platform constraint).

Per Gemini Sprint 19 review: every path written to SQLite, JSON manifest, log
payload, or InfluxDB tag MUST use POSIX format. These tests pin the behavior of
`to_posix()` and `from_posix()`.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from hermes.util.paths import from_posix, to_posix


class TestToPosix:
    """`to_posix(p)` converts any Path-like to POSIX-format string."""

    def test_posix_path_unchanged_forward_slashes(self) -> None:
        """POSIX path stays as-is (no backslashes, forward slashes preserved).

        Note: absolute POSIX paths keep their leading '/' (POSIX convention).
        Relative POSIX paths don't get a leading '/' (also POSIX).
        """
        result = to_posix(PurePosixPath("/data/shared/foo.pdf"))
        assert "/" in result
        assert "\\" not in result
        # Leading slash preserved for absolute paths.
        assert result == "/data/shared/foo.pdf"

    def test_windows_path_converted_backslashes_to_forward(self) -> None:
        """Windows backslashes become forward slashes; drive letter preserved."""
        result = to_posix(PureWindowsPath(r"C:\Users\foo\bar.pdf"))
        assert "/" in result
        assert "\\" not in result
        assert result == "C:/Users/foo/bar.pdf"

    def test_accepts_plain_string(self) -> None:
        """String input is tolerated."""
        result = to_posix("share/foo/bar.pdf")
        assert result == "share/foo/bar.pdf"

    def test_accepts_os_path_like(self) -> None:
        """os.PathLike (anything Path() accepts) is tolerated."""
        result = to_posix(Path("local/file.txt"))
        # On Linux: "local/file.txt"; on Windows: "local/file.txt" (Path normalizes).
        assert "\\" not in result

    def test_relative_path_unchanged(self) -> None:
        """Relative paths don't get a leading '/' (POSIX convention)."""
        result = to_posix(PurePosixPath("foo/bar/baz.md"))
        assert result == "foo/bar/baz.md"
        assert not result.startswith("/")


class TestFromPosix:
    """`from_posix(s)` roundtrips a POSIX string back to Path on current platform."""

    def test_posix_string_to_path(self) -> None:
        result = from_posix("share/foo.pdf")
        assert isinstance(result, Path)
        # Path() on the current platform will accept this. On Linux == PosixPath.
        # Don't assert equality with PosixPath because Windows Path() is WindowsPath.

    def test_posix_with_drive_letter_on_windows(self) -> None:
        """POSIX string with Windows-style drive letter still parses on Windows.

        On Linux: Path("C:/foo") is treated as relative 'C:/foo' (no special meaning).
        On Windows: Path("C:/foo") is absolute Windows path. Test that it doesn't crash
        in either case (we trust the OS Path() machinery).
        """
        # Just verify it doesn't raise on either platform.
        result = from_posix("C:/Users/foo/bar.pdf")
        assert isinstance(result, Path)


class TestRoundTrip:
    """Roundtrip property: from_posix(to_posix(p)) == Path(p) for any path."""

    @pytest.mark.parametrize(
        "pathlike",
        [
            Path("/tmp/foo.pdf"),
            Path("relative/file.txt"),
            Path("./foo/bar.md"),
        ],
    )
    def test_roundtrip_preserves_string_representation(self, pathlike: Path) -> None:
        """POSIX string roundtrips, doesn't change representation."""
        posix_str = to_posix(pathlike)
        recovered = from_posix(posix_str)
        assert to_posix(recovered) == posix_str


class TestIntegration:
    """Integration: to_posix is called in expected code paths."""

    def test_no_backslash_in_posix_output(self) -> None:
        """Anti-regression: never emit backslash from to_posix, no matter the input."""
        # Simulate a Windows path passed in (use PureWindowsPath so test is
        # cross-platform, not relying on Windows-only Path()).
        win_path = PureWindowsPath(r"D:\share\data\foo.md")
        assert "\\" not in to_posix(win_path)
