"""PRE2-C1A: atomic, versioned, UTF-8 plan store for SearchPlan artifacts.

This store is a PRIVATE / INTERNAL evidence seam. It is NOT a database
table and does NOT require a migration. It writes one JSON file per
research job under the existing confined Deep Research data root, in
a dedicated ``research_plans/`` subdirectory so reports and plans stay
physically separate.

Properties (per the C1A contract):

- Deterministic path derived from ``job_id``.
- Versioned JSON envelope (``schema_version`` carried by the plan +
  ``envelope_version`` on the wrapper).
- UTF-8.
- Atomic temp-write + flush + fsync + ``os.replace`` (consistent
  with the existing report durability style).
- Safe path construction (UUID12 hex job_id, no traversal).
- Write / load round-trip preserves the byte content.
- Load validates schema / version / hash / capability snapshot
  before returning a reusable plan.
- Corrupt JSON fails closed.
- Unsupported schema version fails closed.
- Research Brief hash mismatch fails closed.
- Capability / limit snapshot mismatch fails closed.
- Repeated write of byte-equivalent plan is idempotent.
- Conflicting overwrite of an existing valid plan is rejected (no
  silent replacement).

This module does NOT mutate ``research_jobs.query`` and does NOT
add a database column. The persisted job row remains the source of
the original Research Brief text; the plan file references the
brief by SHA-256 only.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Final

from hermes.jobs.planning import (
    SCHEMA_VERSION,
    CapabilitySnapshot,
    SearchPlan,
    deserialize_search_plan,
    serialize_search_plan,
    validate_search_plan,
)

#: Envelope version of the on-disk artifact. Bumped on breaking changes
#: to the wrapper (e.g. adding a required top-level field, switching
#: serialization formats). Bumping this constant is a hard contract
#: change that requires regenerating every persisted plan file.
ENVELOPE_VERSION: Final[int] = 1

#: Subdirectory under the Deep Research data root where plan artifacts
#: live. Kept short to keep paths compact; reports live in the parent
#: directory as ``<job_id>.md``.
PLANS_SUBDIR: Final[str] = "research_plans"

#: File extension for plan artifacts. ``.plan.json`` is explicit so
#: tooling (and human reviewers) can identify the artifact kind
#: without opening it.
PLAN_FILE_SUFFIX: Final[str] = ".plan.json"

#: UUID12 hex: exactly 12 lowercase hex characters. Same constraint as
#: the report path layer (``hermes.jobs.report_paths``). Keeping the
#: same surface here means a future C1B can share a single job-id
#: validator without surprise.
_JOB_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class PlanStoreError(Exception):
    """Base class for plan-store errors."""


class PlanInvalidJobIdError(PlanStoreError):
    """Raised when ``job_id`` is not a valid UUID12 hex token."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"invalid_job_id:{job_id!r}")


class PlanPathEscapeError(PlanStoreError):
    """Raised when the derived plan path escapes the configured root."""


class PlanNotFoundError(PlanStoreError):
    """Raised when a load() call cannot find an existing plan file."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"plan not found: {job_id}")


class PlanConflictError(PlanStoreError):
    """Raised when a write would silently replace an existing valid plan.

    The store compares the existing file bytes to the new bytes. If
    they differ, the write is rejected to prevent silent overwrite
    of an in-use plan. The caller can read the existing plan,
    decide whether the new plan supersedes it (future C1B), and
    re-issue the write with an explicit ``force=True`` after
    explicit operator action.
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(
            f"plan conflict: existing valid plan for {job_id!r} would "
            f"be silently replaced; pass force=True after operator action"
        )


class PlanCorruptError(PlanStoreError):
    """Raised when the persisted plan file is unreadable / unparseable."""

    def __init__(self, job_id: str, detail: str) -> None:
        self.job_id = job_id
        self.detail = detail
        super().__init__(f"plan_corrupt:{job_id!r}:{detail}")


class PlanSchemaMismatchError(PlanStoreError):
    """Raised when the persisted plan's schema version is unsupported."""

    def __init__(self, job_id: str, found: int, supported: int) -> None:
        self.job_id = job_id
        self.found = found
        self.supported = supported
        super().__init__(
            f"plan_schema_mismatch:{job_id!r}: found={found} supported={supported}"
        )


class PlanBriefHashMismatchError(PlanStoreError):
    """Raised when the persisted plan's brief hash does not match the expected."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"plan_brief_hash_mismatch:{job_id!r}")


class PlanCapabilitySnapshotMismatchError(PlanStoreError):
    """Raised when the persisted plan's capability snapshot does not match expected."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"plan_capability_snapshot_mismatch:{job_id!r}")


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _validate_job_id(job_id: str) -> None:
    if not isinstance(job_id, str):
        raise PlanInvalidJobIdError(str(job_id))
    if "\x00" in job_id or ".." in job_id:
        raise PlanInvalidJobIdError(job_id)
    if not _JOB_ID_RE.match(job_id):
        raise PlanInvalidJobIdError(job_id)


def _assert_inside_root(canonical: Path, root: Path) -> None:
    """Lexical confinement check (no symlink follow).

    The plan store does not follow symlinks: it writes and reads
    regular files only. We do not need the realpath branch used by
    ``report_paths.assert_inside_root`` because the writer
    explicitly opens the canonical path and never creates a
    symlink. We still apply a lexical confinement check so a
    misconfigured root cannot redirect the plan file outside the
    data area.
    """
    root_resolved = root.resolve(strict=False)
    try:
        canonical.resolve(strict=False).relative_to(root_resolved)
    except ValueError as exc:
        raise PlanPathEscapeError(
            f"path {canonical!s} escapes root {root_resolved!s}"
        ) from exc


# ---------------------------------------------------------------------------
# Public store
# ---------------------------------------------------------------------------


class LocalPlanStore:
    """Sync, atomic, versioned, UTF-8 plan store.

    The store is constructed with the parent Deep Research data
    root (e.g. ``settings.deep_research_data_root``); plans are
    written under ``<root>/<PLANS_SUBDIR>/<job_id><PLAN_FILE_SUFFIX>``.

    Construction validates the root and creates the ``<root>/<PLANS_SUBDIR>``
    directory (with parents) so the first write does not race a
    missing directory. The store does NOT touch the parent root's
    other subdirectories (e.g. report files), so a coexisting
    ``LocalReportStore`` on the same root is unaffected.
    """

    def __init__(self, root: Path) -> None:
        if root is None:
            raise ValueError("root is required")
        self._root = Path(root).resolve(strict=False)
        self._plans_root = (self._root / PLANS_SUBDIR).resolve(strict=False)
        # Create the plans subdirectory eagerly. ``parents=True``
        # handles the case where the parent data root does not yet
        # exist (e.g. fresh test fixture).
        self._plans_root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def plans_root(self) -> Path:
        return self._plans_root

    # ---- Path derivation -------------------------------------------------

    def derive_path(self, job_id: str) -> Path:
        """Return the candidate plan path for ``job_id`` (no I/O)."""
        _validate_job_id(job_id)
        candidate = self._plans_root / f"{job_id}{PLAN_FILE_SUFFIX}"
        # Confinement is purely lexical here; the path is constructed
        # from a validated job_id and a known root, so escape is not
        # possible. We still run the check to catch a future
        # misconfiguration (e.g. root becomes a symlink to elsewhere).
        _assert_inside_root(candidate, self._root)
        return candidate

    # ---- Existence -------------------------------------------------------

    def exists(self, job_id: str) -> bool:
        """Return True iff a real plan file exists for ``job_id``.

        Path validation happens first; an invalid or escaping
        job_id returns ``False`` (NOT a raise), so this method is
        safe to call on untrusted input.
        """
        try:
            canonical = self.derive_path(job_id)
        except (PlanInvalidJobIdError, PlanPathEscapeError):
            return False
        return canonical.is_file()

    # ---- Write -----------------------------------------------------------

    def write(
        self,
        job_id: str,
        plan: SearchPlan,
        *,
        force: bool = False,
    ) -> None:
        """Atomically write the plan to disk.

        Sequence:
          1. ``derive_path`` (validates job_id + confinement).
          2. If a file already exists at the target path AND its
             bytes are byte-equal to the new envelope, the write
             is a no-op (idempotent rewrite of the same plan).
          3. If a file already exists AND the bytes differ AND
             ``force`` is False, raise ``PlanConflictError``. The
             operator must explicitly opt in to overwriting a
             different valid plan (``force=True``); C1A's default
             is to fail closed.
          4. Serialize the plan, wrap it in a versioned envelope,
             write to a temp file in the same directory, fsync,
             and ``os.replace`` onto the target path. ``os.replace``
             is atomic on POSIX and on Windows when both paths are
             on the same filesystem (the temp file is created in
             the plans subdirectory, so this holds).

        Raises:
            PlanInvalidJobIdError: bad job_id.
            PlanPathEscapeError: path escapes the configured root.
            PlanConflictError: an existing valid plan would be
                silently replaced; pass ``force=True`` to overwrite.
            OSError: filesystem failure.
        """
        target = self.derive_path(job_id)

        new_bytes = self._build_envelope(plan)
        if target.is_file():
            existing_bytes = target.read_bytes()
            if existing_bytes == new_bytes:
                # Idempotent rewrite: nothing to do.
                return
            if not force:
                raise PlanConflictError(job_id)
            # ``force=True``: fall through and replace. We do not
            # attempt to compare plans semantically; the caller has
            # already decided that the new plan supersedes the old
            # one. The atomic write below guarantees that a
            # concurrent reader sees either the old bytes or the
            # new bytes, never a partial mix.

        # Atomic write: temp file in the SAME directory so
        # ``os.replace`` is guaranteed to be atomic on Windows.
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".{job_id}.plan.",
            suffix=".tmp.json",
            dir=str(self._plans_root),
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(new_bytes)
                f.flush()
                # ``os.fsync`` ensures the bytes are durable on
                # disk before the rename. On Windows, ``os.replace``
                # is atomic on NTFS when both paths are on the
                # same volume, which they are here.
                os.fsync(f.fileno())
            os.replace(tmp_path_str, target)
        except BaseException:
            # Best-effort cleanup if the rename or fsync failed.
            with contextlib.suppress(OSError):
                os.unlink(tmp_path_str)
            raise

    # ---- Load ------------------------------------------------------------

    def load(
        self,
        job_id: str,
        *,
        expected_research_brief_sha256: str | None = None,
        expected_capability_snapshot: CapabilitySnapshot | None = None,
    ) -> SearchPlan:
        """Load and validate a persisted plan.

        The returned plan is validated by
        ``validate_search_plan`` (the C1A hard rules) AND by the
        store-level envelope / version / hash / capability-snapshot
        checks. Callers can rely on the returned plan being
        structurally sound AND bound to the expected brief /
        capability snapshot.

        Raises:
            PlanInvalidJobIdError: bad job_id.
            PlanPathEscapeError: path escapes the configured root.
            PlanNotFoundError: no plan file for the given job_id.
            PlanCorruptError: file is not valid UTF-8 / JSON.
            PlanSchemaMismatchError: unsupported schema or envelope version.
            PlanBriefHashMismatchError: stored hash differs from expected.
            PlanCapabilitySnapshotMismatchError: stored capability snapshot
                differs from expected.
            PlanStoreError: structural validation failed.
        """
        target = self.derive_path(job_id)
        if not target.is_file():
            raise PlanNotFoundError(job_id)

        raw = target.read_bytes()
        envelope = self._parse_envelope(raw)
        # ``envelope["plan_bytes"]`` is the JSON-encoded plan (a
        # ``str`` per the envelope shape). ``deserialize_search_plan``
        # accepts bytes; encode here so the bytes-only contract is
        # preserved at the planning boundary.
        plan_bytes_field = envelope["plan_bytes"]
        if isinstance(plan_bytes_field, str):
            plan_bytes = plan_bytes_field.encode("utf-8")
        else:
            raise PlanCorruptError(
                job_id, "envelope.plan_bytes must be a JSON string"
            )
        plan = deserialize_search_plan(plan_bytes)

        # Envelope-level checks happen BEFORE structural validation
        # so a wrong envelope version is reported as a schema
        # mismatch (not as a structural defect).
        if envelope["schema_version"] != SCHEMA_VERSION:
            raise PlanSchemaMismatchError(
                job_id=job_id,
                found=int(envelope["schema_version"]),
                supported=SCHEMA_VERSION,
            )
        if envelope["envelope_version"] != ENVELOPE_VERSION:
            raise PlanSchemaMismatchError(
                job_id=job_id,
                found=int(envelope["envelope_version"]),
                supported=ENVELOPE_VERSION,
            )

        # Brief hash binding.
        if expected_research_brief_sha256 is not None and (
            plan.research_brief_sha256 != expected_research_brief_sha256
        ):
            raise PlanBriefHashMismatchError(job_id)

        # Capability snapshot binding. We compare the persisted
        # snapshot dict to the expected snapshot dict via JSON
        # encoding to avoid relying on dataclass equality
        # internals across versions.
        if expected_capability_snapshot is not None:
            persisted_snapshot_dict = plan.capability_snapshot.to_dict()
            expected_snapshot_dict = expected_capability_snapshot.to_dict()
            if json.dumps(
                persisted_snapshot_dict, sort_keys=True, separators=(",", ":")
            ) != json.dumps(
                expected_snapshot_dict, sort_keys=True, separators=(",", ":")
            ):
                raise PlanCapabilitySnapshotMismatchError(job_id)

        # Structural validation is the last gate; any structural
        # defect surfaces as ``PlanStoreError`` so the caller can
        # distinguish load failures from application-level errors.
        try:
            validate_search_plan(plan)
        except Exception as exc:
            raise PlanStoreError(
                f"plan for {job_id!r} failed structural validation: {exc}"
            ) from exc

        return plan

    # ---- Internal envelope ----------------------------------------------

    @staticmethod
    def _build_envelope(plan: SearchPlan) -> bytes:
        envelope = {
            "envelope_version": ENVELOPE_VERSION,
            "schema_version": SCHEMA_VERSION,
            # ``plan_bytes`` is itself deterministic JSON produced
            # by ``serialize_search_plan``. Storing it as a
            # separate JSON string keeps the envelope text-only
            # and avoids a nested JSON-in-JSON decoding step
            # unless a human wants to read the plan directly.
            "plan_bytes": serialize_search_plan(plan).decode("utf-8"),
        }
        return json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @staticmethod
    def _parse_envelope(blob: bytes) -> dict[str, Any]:
        try:
            text = bytes(blob).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PlanCorruptError("<unknown>", "not valid UTF-8") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanCorruptError("<unknown>", f"not valid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise PlanCorruptError("<unknown>", "envelope is not a JSON object")
        for key in ("envelope_version", "schema_version", "plan_bytes"):
            if key not in payload:
                raise PlanCorruptError(
                    "<unknown>", f"envelope missing required key {key!r}"
                )
        if not isinstance(payload["plan_bytes"], str):
            raise PlanCorruptError(
                "<unknown>", "envelope.plan_bytes must be a JSON string"
            )
        return payload


__all__ = [
    "ENVELOPE_VERSION",
    "PLANS_SUBDIR",
    "PLAN_FILE_SUFFIX",
    "LocalPlanStore",
    "PlanBriefHashMismatchError",
    "PlanCapabilitySnapshotMismatchError",
    "PlanConflictError",
    "PlanCorruptError",
    "PlanInvalidJobIdError",
    "PlanNotFoundError",
    "PlanPathEscapeError",
    "PlanSchemaMismatchError",
    "PlanStoreError",
]
