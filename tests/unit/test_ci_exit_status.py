"""Regression tests for trustworthy GitHub Actions exit codes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_github_actions_preserves_pytest_collection_error_exit_code() -> None:
    """A collection error under tests/ must never be reported as success."""
    missing_node = f"{Path(__file__).resolve()}::definitely_not_a_test"
    env = os.environ.copy()
    env["GITHUB_ACTIONS"] = "true"
    env.pop("PYTEST_ADDOPTS", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            missing_node,
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 4, result.stdout + result.stderr
