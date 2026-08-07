"""Atomic private persistence for C2 iterative Research state."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO, ClassVar

try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX CI
    msvcrt = None  # type: ignore[assignment]

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows CI
    fcntl = None  # type: ignore[assignment]

from hermes.deep_research.iteration_state import ResearchIterationState

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


class IterationStateBusyError(IterationStateStoreError):
    """Another coordinator currently owns this job's state lease."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"iteration_state_busy:{job_id!r}")


class LocalIterationStateStore:
    """Confined, UTF-8, atomic checkpoint store for one Research job."""

    def __init__(self, root: Path) -> None:
        if root is None:
            raise ValueError("root is required")
        self._root = Path(root).resolve(strict=False)
        self._state_root = (self._root / ITERATIONS_SUBDIR).resolve(strict=False)
        self._state_root.mkdir(parents=True, exist_ok=True)
        self._assert_inside_root(self._state_root)
        self._claims: dict[str, BinaryIO] = {}

    _claim_guard: ClassVar[threading.Lock] = threading.Lock()
    _claimed_paths: ClassVar[set[Path]] = set()

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

    def claim(self, job_id: str) -> None:
        self._validate_job_id(job_id)
        lock_path = self._claim_path(job_id)
        with self._claim_guard:
            if lock_path in self._claimed_paths:
                raise IterationStateBusyError(job_id)
            handle = lock_path.open("a+b")
            try:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                if msvcrt is not None:
                    msvcrt.locking(  # type: ignore[attr-defined, unused-ignore]
                        handle.fileno(), msvcrt.LK_NBLCK, 1  # type: ignore[attr-defined, unused-ignore]
                    )
                elif fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                handle.close()
                raise IterationStateBusyError(job_id) from exc
            self._claimed_paths.add(lock_path)
            self._claims[job_id] = handle

    def release(self, job_id: str) -> None:
        self._validate_job_id(job_id)
        with self._claim_guard:
            handle = self._claims.pop(job_id, None)
            if handle is None:
                return
            try:
                if msvcrt is not None:
                    handle.seek(0)
                    msvcrt.locking(  # type: ignore[attr-defined, unused-ignore]
                        handle.fileno(), msvcrt.LK_UNLCK, 1  # type: ignore[attr-defined, unused-ignore]
                    )
                elif fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
                self._claimed_paths.discard(self._claim_path(job_id))

    def _path(self, job_id: str) -> Path:
        self._validate_job_id(job_id)
        path = (self._state_root / f"{job_id}{STATE_FILE_SUFFIX}").resolve(strict=False)
        self._assert_inside_root(path)
        return path

    def _claim_path(self, job_id: str) -> Path:
        path = (self._state_root / f"{job_id}.lock").resolve(strict=False)
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
    "IterationStateBusyError",
    "IterationStateCorruptError",
    "IterationStateInvalidJobIdError",
    "IterationStateStoreError",
    "LocalIterationStateStore",
]
