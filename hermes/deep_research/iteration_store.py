"""Atomic private persistence for C2 iterative Research state."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from hermes.deep_research.iteration import ResearchIterationState

_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
ITERATIONS_SUBDIR = "research_iterations"
STATE_FILE_SUFFIX = ".iteration.json"


class IterationStateStoreError(Exception):
    """Base class for local iteration-state persistence errors."""


class IterationStateInvalidJobIdError(IterationStateStoreError):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"invalid_iteration_job_id:{job_id!r}")


class IterationStateCorruptError(IterationStateStoreError):
    def __init__(self, job_id: str, detail: str) -> None:
        super().__init__(f"iteration_state_corrupt:{job_id!r}:{detail}")


class LocalIterationStateStore:
    """Confined, UTF-8, atomic checkpoint store for one Research job."""

    def __init__(self, root: Path) -> None:
        if root is None:
            raise ValueError("root is required")
        self._root = Path(root).resolve(strict=False)
        self._state_root = (self._root / ITERATIONS_SUBDIR).resolve(strict=False)
        self._state_root.mkdir(parents=True, exist_ok=True)
        self._assert_inside_root(self._state_root)

    def exists(self, job_id: str) -> bool:
        return self._path(job_id).exists()

    def load(self, job_id: str) -> ResearchIterationState | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level state must be an object")
            state = ResearchIterationState.from_dict(payload)
        except Exception as exc:
            raise IterationStateCorruptError(job_id, "unreadable or invalid JSON") from exc
        if state.job_id != job_id:
            raise IterationStateCorruptError(job_id, "state job binding mismatch")
        return state

    def write(self, job_id: str, state: ResearchIterationState) -> None:
        self._validate_job_id(job_id)
        if state.job_id != job_id:
            raise IterationStateCorruptError(job_id, "state job binding mismatch")
        path = self._path(job_id)
        payload = json.dumps(
            state.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{job_id}.",
            suffix=".tmp",
            dir=self._state_root,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _path(self, job_id: str) -> Path:
        self._validate_job_id(job_id)
        path = (self._state_root / f"{job_id}{STATE_FILE_SUFFIX}").resolve(strict=False)
        self._assert_inside_root(path)
        return path

    @staticmethod
    def _validate_job_id(job_id: str) -> None:
        if not isinstance(job_id, str) or not _JOB_ID_RE.fullmatch(job_id):
            raise IterationStateInvalidJobIdError(str(job_id))

    def _assert_inside_root(self, path: Path) -> None:
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise IterationStateStoreError("iteration state path escaped root") from exc


__all__ = [
    "ITERATIONS_SUBDIR",
    "STATE_FILE_SUFFIX",
    "IterationStateCorruptError",
    "IterationStateInvalidJobIdError",
    "IterationStateStoreError",
    "LocalIterationStateStore",
]
