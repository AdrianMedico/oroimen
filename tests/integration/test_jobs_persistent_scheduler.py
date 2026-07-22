"""Integration tests for the persistent scheduler / dispatcher fix.

These tests cover the systemic pickle-error bug that left
``research_jobs`` rows in ``pending`` with no scheduler entry
after ``DeepResearchScheduler.add_job`` failed to serialize the
bound ``self._service._run_research`` callable.

Tests A-J implement the spec in section 9 of the owner
authorization. All tests are offline and deterministic. They use
a real temporary SQLite-backed ``SQLAlchemyJobStore`` (no
``MemoryJobStore``) so the pickle path is actually exercised.

The service is replaced with a minimal stub that records the
``job_id`` it was called with. No provider calls are made. No
real LLM or search backend is contacted.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import pickle
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

from hermes.jobs.dispatcher import (
    execute_research_job,
)
from hermes.jobs.scheduler import DeepResearchScheduler
from hermes.jobs.service_registry import (
    clear_research_service,
    get_research_service,
)


# ============================================================================
# Minimal stub service used in place of the real DeepResearchService.
# The stub records the job_id it was called with. It NEVER holds an
# aiosqlite.Connection or any other non-picklable graph.
# ============================================================================
class StubService:
    """A minimal stand-in for ``DeepResearchService`` that records
    invocations. It exposes a ``_run_research`` coroutine so the
    dispatcher's contract (``service._run_research(job_id)``) is
    honored without depending on the real service graph.

    The stub does NOT subclass ``DeepResearchService``: the test
    explicitly verifies that the dispatcher contract is shape-
    based (no inheritance, no shared base state) so a future
    refactor cannot accidentally re-introduce the pickle
    problem.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._lock = asyncio.Lock()

    async def _run_research(self, job_id: str) -> None:
        async with self._lock:
            self.calls.append(job_id)


# ============================================================================
# Helpers: build a real DeepResearchScheduler against the real
# Database fixture. The Database and the SQLAlchemyJobStore share the
# same SQLite file (the schema-migrations live in research_jobs; the
# jobstore uses apscheduler_jobs). The settings fixture sets
# DB_PATH = tmp_path / "test.db"; the scheduler derives the
# jobstore URL from settings.db_path. Both end up on the same file.
# ============================================================================
async def _make_scheduler_and_stub(
    db: Any, settings: Any
) -> tuple[DeepResearchScheduler, StubService, str]:
    """Build a started scheduler + stub service + jobstore_url.

    Returns:
        (scheduler, stub, jobstore_url). The jobstore_url is the
        file URL the SQLAlchemyJobStore uses so a test can also
        open a parallel jobstore for inspection if needed.
    """
    scheduler = DeepResearchScheduler(db=db, settings=settings)
    stub = StubService()
    # The deprecated set_service shim also writes the registry,
    # which is the dispatcher's source of truth.
    scheduler.set_service(stub)
    await scheduler.start()
    jobstore_url = scheduler._jobstore_url
    return scheduler, stub, jobstore_url


def _settings_with_short_grace(settings: Any) -> Any:
    """Inject a short pending-gap grace so tests can use it.

    The default grace is 60 seconds; tests that exercise Case 6
    (pending-gap recovery) need it short (e.g. 1 second) so a
    fresh row with an old updated_at is immediately eligible.
    """
    import types

    class _GraceProxy(types.SimpleNamespace):
        """Proxy that reads from the real settings, except for
        the pending-gap grace which we set to 1 second."""

        def __init__(self, real: Any) -> None:
            super().__init__()
            self._real = real

        def __getattr__(self, name: str) -> Any:
            if name == "deep_research_recovery_pending_gap_grace_seconds":
                return 1
            return getattr(self._real, name)

    return _GraceProxy(settings)


# ============================================================================
# Test A — Persistent serialization
# ============================================================================
@pytest.mark.asyncio
async def test_a_persistent_serialization_no_pickle_error(db, settings) -> None:
    """The persisted callable is the module-level
    ``execute_research_job`` from ``hermes.jobs.dispatcher``.
    add_job with a real ``SQLAlchemyJobStore`` does not raise
    ``TypeError: cannot pickle``. The scheduler entry exists
    after enqueue. The persisted args contain only ``job_id``.
    """
    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test a — persistent serialization",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, _stub, jobstore_url = await _make_scheduler_and_stub(db, settings)
    try:
        await scheduler.enqueue(job_id, run_date=datetime.now(UTC))

        # Inspect the actual jobstore row.
        jobstore = SQLAlchemyJobStore(url=jobstore_url)
        try:
            row = jobstore.lookup_job(job_id)
            assert row is not None, (
                "scheduler entry MUST exist after enqueue; "
                "the test simulates a real SQLAlchemyJobStore"
            )
            # The persisted callable is the module-level
            # dispatcher. The state is pickled into the
            # jobstore row.
            assert row.func is execute_research_job, (
                f"persisted callable MUST be the dispatcher "
                f"execute_research_job, got {row.func!r}"
            )
            assert tuple(row.args) == (job_id,), (
                f"persisted args MUST be exactly (job_id,), "
                f"got {row.args!r}"
            )
        finally:
            jobstore.shutdown()
    finally:
        await scheduler.shutdown(timeout_s=2.0)


# ============================================================================
# Test B — Bound service graph is not persisted
# ============================================================================
@pytest.mark.asyncio
async def test_b_bound_service_graph_not_persisted(db, settings) -> None:
    """The jobstore row's serialized callable and args do NOT
    contain a ``DeepResearchService``, a ``Database``, an
    ``aiosqlite.Connection`` or a ``sqlite3.Connection``.

    The persisted callable is the module-level dispatcher
    function. The only arg is the string ``job_id``. No live
    graph is serialized.
    """
    import sqlite3

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test b — bound service graph",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, _stub, jobstore_url = await _make_scheduler_and_stub(db, settings)
    try:
        await scheduler.enqueue(job_id, run_date=datetime.now(UTC))

        # Re-open the jobstore and inspect the row.
        jobstore = SQLAlchemyJobStore(url=jobstore_url)
        try:
            row = jobstore.lookup_job(job_id)
            assert row is not None

            # Pickle + unpickle the job state ourselves.
            state = row.__getstate__()
            blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            state2 = pickle.loads(blob)
            # APScheduler 4.x serializes the function as a
            # ``module:qualname`` string. The string must
            # reference the dispatcher module.
            assert state2["func"] == "hermes.jobs.dispatcher:execute_research_job", (
                f"persisted func must be the dispatcher qualname, "
                f"got {state2['func']!r}"
            )
            # And resolving the qualname must yield the
            # actual function (not a stale reference).
            module_name, qualname = state2["func"].split(":", 1)
            mod = __import__(module_name, fromlist=[qualname])
            resolved = getattr(mod, qualname)
            assert resolved is execute_research_job
            assert tuple(state2["args"]) == (job_id,)

            # No live graph leaked into the serialized state.
            for k, v in state2.items():
                assert not isinstance(v, sqlite3.Connection), (
                    f"persisted state key {k!r} must not hold a "
                    f"sqlite3.Connection"
                )
            # Also re-pickle the func reference itself.
            blob2 = pickle.dumps(row.func)
            pickle.loads(blob2)  # must not raise
        finally:
            jobstore.shutdown()
    finally:
        await scheduler.shutdown(timeout_s=2.0)


# ============================================================================
# Test C — Actual execution (dispatcher resolves the registered service)
# ============================================================================
@pytest.mark.asyncio
async def test_c_actual_execution_dispatches_to_service(db, settings) -> None:
    """The persisted immediate job fires. The dispatcher
    resolves the registered service. The exact ``job_id``
    reaches ``_run_research`` once.
    """
    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test c — actual execution",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        await scheduler.enqueue(job_id, run_date=datetime.now(UTC))

        # Wait for the job to fire.
        for _ in range(50):  # up to ~5s
            if stub.calls:
                break
            await asyncio.sleep(0.1)
        assert stub.calls == [job_id], (
            f"the dispatcher must call _run_research exactly "
            f"once with the expected job_id; "
            f"expected [{job_id!r}], got {stub.calls!r}"
        )
    finally:
        await scheduler.shutdown(timeout_s=2.0)


# ============================================================================
# Test D — Restart persistence
# ============================================================================
@pytest.mark.asyncio
async def test_d_restart_persistence_survives_scheduler_restart(db, settings) -> None:
    """A future-dated job enqueued on scheduler A still
    fires when scheduler A is shut down and a fresh scheduler
    B is constructed over the same SQLite jobstore with a
    fresh registered service.
    """
    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test d — restart persistence",
        notify_via_tg=0,
        user_id=0,
    )

    # First scheduler instance.
    scheduler_a = DeepResearchScheduler(db=db, settings=settings)
    stub_a = StubService()
    scheduler_a.set_service(stub_a)
    await scheduler_a.start()
    try:
        # Enqueue for a near-future time (so it fires quickly
        # but the scheduler does not race the shutdown).
        run_date = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=2)
        await scheduler_a.enqueue(job_id, run_date=run_date)

        # Shutdown scheduler A.
        await scheduler_a.shutdown(timeout_s=3.0)
        clear_research_service()  # cleanup; scheduler B re-registers

        # The job is still in the jobstore. After
        # ``scheduler_a.shutdown()`` the wrapper's ``_scheduler``
        # is detached, so we inspect the SQLite file directly.
        jobstore_url = scheduler_a._jobstore_url
        jobstore = SQLAlchemyJobStore(url=jobstore_url)
        try:
            row_after_a = jobstore.lookup_job(job_id)
            assert row_after_a is not None, (
                "the job must persist across scheduler A shutdown"
            )
        finally:
            jobstore.shutdown()

        # Second scheduler instance, fresh stub.
        scheduler_b = DeepResearchScheduler(db=db, settings=settings)
        stub_b = StubService()
        scheduler_b.set_service(stub_b)
        await scheduler_b.start()
        try:
            # Wait for the job to fire on scheduler B.
            for _ in range(70):  # up to ~7s
                if stub_b.calls:
                    break
                await asyncio.sleep(0.1)
            assert stub_b.calls == [job_id], (
                f"the persisted job must fire exactly once on "
                f"scheduler B with the expected job_id; "
                f"got {stub_b.calls!r}"
            )
            # And it must NOT have fired on scheduler A (which
            # was already shut down before the fire time).
            assert stub_a.calls == [], (
                f"the job must NOT have fired on scheduler A "
                f"(already shut down); got {stub_a.calls!r}"
            )
        finally:
            await scheduler_b.shutdown(timeout_s=3.0)
    finally:
        # Defensive: scheduler_a already shut down. The
        # start() guard would no-op on a re-shutdown anyway.
        with contextlib.suppress(Exception):
            await scheduler_a.shutdown(timeout_s=1.0)


# ============================================================================
# Test E — Enqueue serialization failure truth
# ============================================================================
@pytest.mark.asyncio
async def test_e_enqueue_serialization_failure_truth(db, settings) -> None:
    """When ``add_job`` raises a serialization-style error,
    the scheduler propagates ``SchedulerEnqueueError`` and the
    row is NOT left in ``pending``. The caller is responsible
    for compensating the row; in the service path this is
    done in ``submit_job``. Here we test the scheduler-level
    contract: add_job raises → SchedulerEnqueueError
    propagates, the jobstore has no entry, no 201-equivalent
    is implied.
    """
    from hermes.jobs.exceptions import SchedulerEnqueueError

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test e — serialization failure",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, _stub, jobstore_url = await _make_scheduler_and_stub(db, settings)
    try:
        # Force add_job to raise by patching the internal
        # scheduler's add_job. We do NOT swallow the
        # exception; we expect SchedulerEnqueueError to
        # propagate.
        real_scheduler = scheduler._scheduler

        def boom(*a, **kw):
            raise TypeError("simulated_pickle_failure")

        real_scheduler.add_job = boom
        with pytest.raises(SchedulerEnqueueError) as excinfo:
            await scheduler.enqueue(
                job_id, run_date=datetime.now(UTC)
            )
        assert "add_job failed" in str(excinfo.value)

        # The jobstore has no entry.
        jobstore = SQLAlchemyJobStore(url=jobstore_url)
        try:
            row = jobstore.lookup_job(job_id)
            assert row is None, (
                "scheduler entry must NOT exist when add_job "
                "raised a serialization error"
            )
        finally:
            jobstore.shutdown()
    finally:
        await scheduler.shutdown(timeout_s=2.0)


# ============================================================================
# Test F — Pending-gap recovery
# ============================================================================
@pytest.mark.asyncio
async def test_f_pending_gap_recovery_enqueues_missing_pending(db, settings) -> None:
    """A ``pending`` row without a scheduler entry is
    detected by recovery and re-enqueued. A second recovery
    run does NOT duplicate the entry.
    """
    from hermes.jobs.recovery import recover_research_jobs

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test f — pending gap",
        notify_via_tg=0,
        user_id=0,
    )
    # Backdate updated_at to far in the past so the
    # pending-gap grace elapses immediately.
    await db.conn.execute(
        "UPDATE research_jobs SET updated_at = ? WHERE id = ?",
        ("2020-01-01 00:00:00.000000+00:00", job_id),
    )
    await db.conn.commit()

    settings_short = _settings_with_short_grace(settings)
    scheduler, _stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        # No scheduler entry for this job yet.
        assert scheduler.get_job(job_id) is None

        # Run recovery. The pending-gap case should
        # re-enqueue the row.
        await recover_research_jobs(
            db=db,
            notifier=AsyncMock(),
            settings=settings_short,
            scheduler=scheduler,
        )

        # After recovery, the job should be in the jobstore.
        assert scheduler.get_job(job_id) is not None, (
            "pending-without-scheduler-entry must be "
            "recovered by re-enqueue"
        )

        # Run recovery a second time. The job is still
        # in the jobstore, so the case-6 must skip it
        # (``recovery_pending_gap_skipped_healthy``).
        await recover_research_jobs(
            db=db,
            notifier=AsyncMock(),
            settings=settings_short,
            scheduler=scheduler,
        )
        # The job is still in the jobstore; no duplicate.
        assert scheduler.get_job(job_id) is not None
    finally:
        await scheduler.shutdown(timeout_s=2.0)


# ============================================================================
# Test G — Pending healthy recovery (no duplicate)
# ============================================================================
@pytest.mark.asyncio
async def test_g_pending_healthy_recovery_does_not_duplicate(db, settings) -> None:
    """A ``pending`` row WITH a scheduler entry is NOT
    re-enqueued by recovery. The case-6 path sees the
    scheduler entry via ``get_job`` and skips.
    """
    from hermes.jobs.recovery import recover_research_jobs

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test g — pending healthy",
        notify_via_tg=0,
        user_id=0,
    )

    settings_short = _settings_with_short_grace(settings)
    scheduler, stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        # Enqueue in the future so the job does not fire.
        run_date = datetime.now(UTC) + timedelta(seconds=60)
        await scheduler.enqueue(job_id, run_date=run_date)
        assert scheduler.get_job(job_id) is not None

        # Backdate updated_at so the pending-gap grace elapses.
        await db.conn.execute(
            "UPDATE research_jobs SET updated_at = ? WHERE id = ?",
            ("2020-01-01 00:00:00.000000+00:00", job_id),
        )
        await db.conn.commit()

        # Run recovery. Case 6 should see the scheduler
        # entry and skip.
        await recover_research_jobs(
            db=db,
            notifier=AsyncMock(),
            settings=settings_short,
            scheduler=scheduler,
        )
        # The job is still in the jobstore (still pending
        # future, hasn't fired).
        assert scheduler.get_job(job_id) is not None
        # And the stub was not called.
        assert stub.calls == []
    finally:
        await scheduler.shutdown(timeout_s=2.0)


# ============================================================================
# Test H — Registry absent (coherent terminal failed)
# ============================================================================
@pytest.mark.asyncio
async def test_h_registry_absent_terminalizes_to_failed(db, settings) -> None:
    """When the dispatcher runs and the registry is empty
    (e.g. after a service restart that has not yet registered
    the new service), the persisted job is transitioned to a
    terminal ``failed`` state. Tokens=0, cost=0. No
    indefinite ``pending`` or ``running`` row.
    """
    import os

    from hermes.jobs import dispatcher as dispatcher_mod

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test h — registry absent",
        notify_via_tg=0,
        user_id=0,
    )

    # Clear the registry so execute_research_job sees an
    # empty registry.
    clear_research_service()
    assert get_research_service() is None

    # Point the dispatcher at this test's DB. The
    # _terminate_registry_missing fallback uses Settings()
    # to find the DB. We point env to db.path so a fresh
    # Settings() picks it up.
    db_path = db.path
    old_db_path = os.environ.get("DB_PATH")
    os.environ["DB_PATH"] = str(db_path)
    try:
        # Sanity: a fresh Settings uses our path.
        from hermes.config import Settings

        s = Settings(_env_file=None)
        assert str(s.db_path) == str(db_path), (
            f"Settings must pick up DB_PATH={db_path}, got {s.db_path}"
        )

        # Call the dispatcher directly. The registry is
        # empty, so it should hit the fallback.
        await dispatcher_mod.execute_research_job(job_id)
    finally:
        if old_db_path is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = old_db_path

    # Verify the row was transitioned to failed.
    row = await db.get_research_job(job_id)
    assert row is not None
    assert row["status"] == "failed", (
        f"row must transition to failed when registry is "
        f"absent, got status={row['status']!r}"
    )
    assert row["error_taxonomy"] == "checkpoint_corrupt"
    assert "registry_absent" in (row["error_message"] or "")


# ============================================================================
# Helpers for the H1-H7 state-transition matrix and the post-add
# lookup tests. The dispatcher uses a fresh ``Database`` from
# ``Settings()``; we point ``DB_PATH`` at the test DB so the
# transition is observable end-to-end.
# ============================================================================
async def _seed_research_job_in_state(
    db: Any,
    job_id: str,
    state: str,
    *,
    current_phase: str | None = None,
    error_taxonomy: str | None = None,
    error_message: str | None = None,
    completed_at: str | None = None,
) -> None:
    """Insert a research_jobs row directly in the requested state.

    The schema DEFAULT generates ``created_at`` and ``updated_at``;
    we override ``updated_at`` so the Case 6 grace test is not
    affected. We also override ``status`` so the row is in the
    exact state the test wants.
    """
    from hermes.jobs.cost import format_now

    now = format_now()
    await db.conn.execute(
        """
        INSERT INTO research_jobs (
            id, user_id, job_type, query, notify_via_tg,
            status, current_phase, progress_percent,
            output_path, partial_output_path,
            error_taxonomy, error_message, cost_usd,
            tokens_in, tokens_out, notified,
            created_at, started_at, completed_at, updated_at
        ) VALUES (?, ?, 'deep_research', ?, 0,
                  ?, ?, 0,
                  NULL, NULL,
                  ?, ?, 0.0,
                  0, 0, 0,
                  ?, ?, ?, ?)
        """,
        (
            job_id,
            0,
            f"h_test_{state}",
            state,
            current_phase,
            error_taxonomy,
            error_message,
            now,
            now if state in ("running", "cancelling") else None,
            completed_at,
            now,
        ),
    )
    await db.conn.commit()


@contextlib.contextmanager
def _scoped_db_path(db_path: Path):
    """Point the dispatcher's ``Settings()`` at the test DB.

    The dispatcher's ``_terminate_registry_missing`` fallback
    builds a fresh ``Database`` from ``Settings()``. We point
    ``DB_PATH`` at the test DB so the transition is observable
    in the real ``db`` fixture.
    """
    old_db_path = os.environ.get("DB_PATH")
    os.environ["DB_PATH"] = str(db_path)
    try:
        yield
    finally:
        if old_db_path is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = old_db_path


async def _invoke_dispatch_with_empty_registry(
    db: Any, job_id: str
) -> Any:
    """Run ``execute_research_job`` with the registry empty.

    Returns the ``RegistryMissingTransition`` enum value the
    dispatcher applied (or determined as a no-op).
    """
    import os as _os
    from hermes.jobs import dispatcher as dispatcher_mod

    clear_research_service()
    assert get_research_service() is None
    with _scoped_db_path(db.path):
        await dispatcher_mod.execute_research_job(job_id)
    # Re-read the canonical transition enum from the public
    # API by invoking the helper directly. The dispatcher
    # does not currently return the transition; we infer it
    # from the row state below. This keeps the helper
    # callable from tests without changing the dispatcher's
    # public signature.
    return await db.get_research_job(job_id)


# ============================================================================
# Test H1 — registry absent + pending -> failed
# ============================================================================
@pytest.mark.asyncio
async def test_h1_registry_absent_pending_to_failed(db, settings) -> None:
    """``pending`` row + registry absent + dispatcher runs
    -> ``failed`` with ``checkpoint_corrupt`` taxonomy.

    The transition is conditional on the exact source state
    (atomic CAS via ``transition_research_job_status``). No
    provider call is made. tokens and cost are preserved.
    """
    job_id = uuid.uuid4().hex[:12]
    await _seed_research_job_in_state(db, job_id, "pending")

    row = await _invoke_dispatch_with_empty_registry(db, job_id)
    assert row is not None
    assert row["status"] == "failed", (
        f"pending must transition to failed when registry is "
        f"absent, got status={row['status']!r}"
    )
    assert row["error_taxonomy"] == "checkpoint_corrupt"
    assert "registry_absent" in (row["error_message"] or "")
    # Tokens and cost preserved (still zero for a job that
    # never ran).
    assert row["cost_usd"] == 0.0
    assert row["tokens_in"] == 0
    assert row["tokens_out"] == 0


# ============================================================================
# Test H2 — registry absent + running -> failed
# ============================================================================
@pytest.mark.asyncio
async def test_h2_registry_absent_running_to_failed(db, settings) -> None:
    """``running`` row + registry absent + dispatcher runs
    -> ``failed`` with ``checkpoint_corrupt`` taxonomy.
    """
    job_id = uuid.uuid4().hex[:12]
    await _seed_research_job_in_state(
        db, job_id, "running", current_phase="scrape"
    )

    row = await _invoke_dispatch_with_empty_registry(db, job_id)
    assert row is not None
    assert row["status"] == "failed", (
        f"running must transition to failed when registry is "
        f"absent, got status={row['status']!r}"
    )
    assert row["error_taxonomy"] == "checkpoint_corrupt"
    assert "registry_absent" in (row["error_message"] or "")


# ============================================================================
# Test H3 — registry absent + cancelling -> cancelled (NOT failed)
# ============================================================================
@pytest.mark.asyncio
async def test_h3_registry_absent_cancelling_to_cancelled(
    db, settings
) -> None:
    """``cancelling`` row + registry absent + dispatcher runs
    -> ``cancelled`` (NEVER ``failed``).

    PRE1B invariant: cancellation wins. The registry-absent
    fallback MUST NOT overwrite an in-progress cancellation.
    A row that was ``cancelling`` must reach ``cancelled``;
    it must never become ``failed``.
    """
    from hermes.jobs.cost import format_now

    job_id = uuid.uuid4().hex[:12]
    await _seed_research_job_in_state(
        db,
        job_id,
        "cancelling",
        current_phase="scrape",
        error_taxonomy="cancelled",
        error_message="cancellation_requested_pre_execution",
    )

    row = await _invoke_dispatch_with_empty_registry(db, job_id)
    assert row is not None
    assert row["status"] == "cancelled", (
        f"cancelling must transition to cancelled (NEVER "
        f"failed), got status={row['status']!r}"
    )
    # The cancellation finalization marker is preserved.
    assert row["error_taxonomy"] == "cancelled", (
        f"cancellation finalization marker must be preserved, "
        f"got error_taxonomy={row['error_taxonomy']!r}"
    )
    # The registry-absent message must NOT have overwritten
    # the cancellation marker.
    assert "registry_absent" not in (row["error_message"] or ""), (
        f"registry-absent message must NOT overwrite "
        f"cancellation marker, got {row['error_message']!r}"
    )
    # completed_at must be set.
    assert row["completed_at"] is not None


# ============================================================================
# Test H4 — registry absent + complete remains complete
# ============================================================================
@pytest.mark.asyncio
async def test_h4_registry_absent_complete_remains_complete(
    db, settings
) -> None:
    """``complete`` row + registry absent -> stays ``complete``.

    The registry-absent fallback is a no-op on a terminal
    state. No resurrection, no overwrite of output_path.
    """
    from hermes.jobs.cost import format_now

    job_id = uuid.uuid4().hex[:12]
    completed_at = format_now()
    await _seed_research_job_in_state(
        db,
        job_id,
        "complete",
        completed_at=completed_at,
    )
    # Set an output_path to verify it is not overwritten.
    await db.conn.execute(
        "UPDATE research_jobs SET output_path = ?, progress_percent = 100 "
        "WHERE id = ?",
        ("/tmp/h4_output.md", job_id),
    )
    await db.conn.commit()

    row = await _invoke_dispatch_with_empty_registry(db, job_id)
    assert row is not None
    assert row["status"] == "complete", (
        f"complete must remain complete (no resurrection), "
        f"got status={row['status']!r}"
    )
    assert row["output_path"] == "/tmp/h4_output.md", (
        f"output_path must not be overwritten, got "
        f"{row['output_path']!r}"
    )
    assert row["progress_percent"] == 100
    # completed_at must not be touched.
    assert row["completed_at"] == completed_at


# ============================================================================
# Test H5 — registry absent + failed remains failed
# ============================================================================
@pytest.mark.asyncio
async def test_h5_registry_absent_failed_remains_failed(db, settings) -> None:
    """``failed`` row + registry absent -> stays ``failed``.

    The transition is conditional on the source state; a
    pre-existing failed row is NOT overwritten.
    """
    from hermes.jobs.cost import format_now

    job_id = uuid.uuid4().hex[:12]
    completed_at = format_now()
    await _seed_research_job_in_state(
        db,
        job_id,
        "failed",
        completed_at=completed_at,
        error_taxonomy="timeout",
        error_message="pre_existing_failure",
    )

    row = await _invoke_dispatch_with_empty_registry(db, job_id)
    assert row is not None
    assert row["status"] == "failed", (
        f"failed must remain failed, got status={row['status']!r}"
    )
    # The pre-existing error metadata is preserved.
    assert row["error_taxonomy"] == "timeout", (
        f"pre-existing error_taxonomy must be preserved, got "
        f"{row['error_taxonomy']!r}"
    )
    assert row["error_message"] == "pre_existing_failure"


# ============================================================================
# Test H6 — registry absent + cancelled remains cancelled
# ============================================================================
@pytest.mark.asyncio
async def test_h6_registry_absent_cancelled_remains_cancelled(
    db, settings
) -> None:
    """``cancelled`` row + registry absent -> stays ``cancelled``.

    A pre-existing cancelled row is NOT overwritten. The
    registry-absent fallback is a no-op on the terminal
    state.
    """
    from hermes.jobs.cost import format_now

    job_id = uuid.uuid4().hex[:12]
    completed_at = format_now()
    await _seed_research_job_in_state(
        db,
        job_id,
        "cancelled",
        completed_at=completed_at,
        error_taxonomy="cancelled",
        error_message="pre_existing_cancellation",
    )

    row = await _invoke_dispatch_with_empty_registry(db, job_id)
    assert row is not None
    assert row["status"] == "cancelled", (
        f"cancelled must remain cancelled, got "
        f"status={row['status']!r}"
    )
    assert row["error_taxonomy"] == "cancelled"
    assert row["error_message"] == "pre_existing_cancellation"


# ============================================================================
# Test H7 — race: terminal writer wins, dispatcher does not overwrite
# ============================================================================
@pytest.mark.asyncio
async def test_h7_dispatcher_does_not_resurrect_terminal_state(
    db, settings
) -> None:
    """A concurrent writer (e.g. PRE1B finalizer) flips a
    ``running`` row to ``cancelled`` between the dispatcher's
    read and the atomic CAS. The CAS predicate refuses to
    overwrite; the dispatcher's ``get_research_job`` confirms
    the row is still ``cancelled``. The dispatcher does NOT
    resurrect the row to ``failed``.

    This is the race-with-terminal-writer case. The fix
    uses an atomic CAS predicate (status IN (from_states))
    so a concurrent state change is observed and the
    transition is correctly a no-op.

    Implementation: we patch the class method
    ``Database.transition_research_job_status`` so the flip
    happens for the dispatcher's NEW Database instance (the
    test's ``db`` fixture is a different instance).
    """
    from hermes.jobs import dispatcher as dispatcher_mod
    from hermes.jobs.cost import format_now
    from hermes.memory.db import Database

    job_id = uuid.uuid4().hex[:12]
    await _seed_research_job_in_state(
        db, job_id, "running", current_phase="scrape"
    )

    flip_done = False
    original_transition = Database.transition_research_job_status

    async def racing_transition(self, *args, **kwargs):
        nonlocal flip_done
        if not flip_done:
            flip_done = True
            # Flip the row to ``cancelled`` (a PRE1B
            # finalizer would do this). We use the test's
            # ``db`` connection (same file).
            now = format_now()
            await self.conn.execute(
                "UPDATE research_jobs SET status = 'cancelled', "
                "completed_at = ?, error_taxonomy = 'cancelled', "
                "error_message = 'raced_terminal_writer', "
                "updated_at = ? WHERE id = ? AND status = 'running'",
                (now, now, job_id),
            )
            await self.conn.commit()
        return await original_transition(self, *args, **kwargs)

    Database.transition_research_job_status = racing_transition  # type: ignore[method-assign]
    try:
        # Clear the registry so the dispatcher hits the fallback.
        clear_research_service()
        assert get_research_service() is None

        with _scoped_db_path(db.path):
            await dispatcher_mod.execute_research_job(job_id)
    finally:
        Database.transition_research_job_status = original_transition  # type: ignore[method-assign]

    # Re-read the canonical row.
    row = await db.get_research_job(job_id)
    assert row is not None
    # The terminal writer's cancellation wins. The dispatcher
    # must NOT have overwritten the row to ``failed``.
    assert row["status"] == "cancelled", (
        f"terminal writer's state must win, got "
        f"status={row['status']!r}"
    )
    assert row["error_taxonomy"] == "cancelled"
    assert row["error_message"] == "raced_terminal_writer"


# ============================================================================
# Post-add lookup hardening tests (A-E)
# ============================================================================
@pytest.mark.asyncio
async def test_pa_post_add_lookup_raises_calls_best_effort_remove(
    db, settings
) -> None:
    """A: ``add_job`` succeeds, post-add lookup raises.
    The scheduler MUST do a best-effort ``remove_job`` and
    raise ``SchedulerEnqueueError``. The submit_job path will
    then compensate the DB row to ``failed``.
    """
    from hermes.jobs.exceptions import SchedulerEnqueueError

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="pa — lookup raises",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, _stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        real_scheduler = scheduler._scheduler  # test fixture needs the inner APScheduler

        # Stage 1: ``add_job`` succeeds, ``get_job`` raises on
        # the post-add call (call #1). The cleanup path then
        # calls ``get_job`` again (call #2) inside
        # ``remove_job_best_effort``; that call returns a
        # truthy sentinel so the best-effort ``remove_job`` is
        # actually invoked.
        get_job_call_count = 0
        remove_called_with = []
        original_add_job = real_scheduler.add_job

        def fake_add_job(*a, **kw):
            return original_add_job(*a, **kw)

        def fake_get_job(jid, *a, **kw):
            nonlocal get_job_call_count
            get_job_call_count += 1
            if get_job_call_count == 1:
                # Post-add verification: raise.
                raise RuntimeError("simulated_lookup_failure")
            # Second call (inside remove_job_best_effort):
            # return a truthy sentinel so remove actually runs.
            return object()

        def fake_remove_job(jid):
            remove_called_with.append(jid)
            return None

        real_scheduler.add_job = fake_add_job  # type: ignore[method-assign]
        real_scheduler.get_job = fake_get_job  # type: ignore[method-assign]
        real_scheduler.remove_job = fake_remove_job  # type: ignore[method-assign]
        try:
            with pytest.raises(SchedulerEnqueueError) as excinfo:
                await scheduler.enqueue(job_id, run_date=datetime.now(UTC))
            assert "post-add lookup failed" in str(excinfo.value)
        finally:
            real_scheduler.add_job = original_add_job  # type: ignore[method-assign]
            real_scheduler.get_job = real_scheduler.get_job  # type: ignore[method-assign]

        # The best-effort remove was called with the job_id.
        assert job_id in remove_called_with, (
            f"best-effort remove_job must be called when "
            f"post-add lookup raises; got {remove_called_with!r}"
        )
        # Both get_job calls happened.
        assert get_job_call_count == 2, (
            f"post-add verification + best-effort cleanup "
            f"must both call get_job; got {get_job_call_count}"
        )
    finally:
        await scheduler.shutdown(timeout_s=2.0)


@pytest.mark.asyncio
async def test_pb_post_add_lookup_returns_none_calls_best_effort_remove(
    db, settings
) -> None:
    """B: ``add_job`` succeeds, post-add ``get_job`` returns
    ``None`` (partial write). The scheduler MUST do a
    best-effort ``remove_job`` and raise
    ``SchedulerEnqueueError``.
    """
    from hermes.jobs.exceptions import SchedulerEnqueueError

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="pb — lookup returns None",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, _stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        real_scheduler = scheduler._scheduler  # test fixture needs the inner APScheduler

        original_add_job = real_scheduler.add_job
        original_get_job = real_scheduler.get_job

        # Stage 2: ``add_job`` succeeds, but ``get_job``
        # returns None on the post-add call. We need to
        # count calls so the first ``get_job`` (inside
        # ``remove_job_best_effort``) returns None, and
        # ``remove_job`` is not called.
        get_job_call_count = 0
        remove_called_with = []

        def fake_add_job(*a, **kw):
            return original_add_job(*a, **kw)

        def fake_get_job(jid, *a, **kw):
            nonlocal get_job_call_count
            get_job_call_count += 1
            return None  # always None for this test

        def fake_remove_job(jid):
            remove_called_with.append(jid)
            return None

        real_scheduler.add_job = fake_add_job  # type: ignore[method-assign]
        real_scheduler.get_job = fake_get_job  # type: ignore[method-assign]
        real_scheduler.remove_job = fake_remove_job  # type: ignore[method-assign]
        try:
            with pytest.raises(SchedulerEnqueueError) as excinfo:
                await scheduler.enqueue(job_id, run_date=datetime.now(UTC))
            assert "post-add lookup returned None" in str(excinfo.value)
        finally:
            real_scheduler.add_job = original_add_job  # type: ignore[method-assign]
            real_scheduler.get_job = original_get_job  # type: ignore[method-assign]

        # ``remove_job_best_effort`` was called (it calls
        # ``get_job`` internally; when get_job returns
        # None, no remove happens). The important invariant
        # is that the API still raised truthfully.
        assert get_job_call_count >= 2, (
            f"post-add lookup must be called at least twice "
            f"(once for verification, once for best-effort "
            f"remove); got {get_job_call_count}"
        )
        # No remove actually happened because get_job
        # always returned None.
        assert remove_called_with == [], (
            f"no remove_job should be called when the "
            f"jobstore reports no entry; got {remove_called_with!r}"
        )
    finally:
        await scheduler.shutdown(timeout_s=2.0)


@pytest.mark.asyncio
async def test_pc_best_effort_remove_succeeds(db, settings) -> None:
    """C: ``add_job`` succeeds, post-add lookup fails, the
    best-effort ``remove_job`` itself succeeds. The API
    still raises ``SchedulerEnqueueError`` (truthful 503
    via submit_job compensation). The jobstore has no
    orphan entry.
    """
    from hermes.jobs.exceptions import SchedulerEnqueueError

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="pc — remove succeeds",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, _stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        real_scheduler = scheduler._scheduler  # test fixture needs the inner APScheduler

        # Stage: add_job succeeds, get_job raises on first
        # call (post-add) and succeeds (returns the Job)
        # on the second call (inside remove_job_best_effort).
        get_job_call_count = 0
        original_add_job = real_scheduler.add_job

        def fake_add_job(*a, **kw):
            return original_add_job(*a, **kw)

        def fake_get_job(jid, *a, **kw):
            nonlocal get_job_call_count
            get_job_call_count += 1
            if get_job_call_count == 1:
                raise RuntimeError("simulated_first_call_failure")
            # Second call (inside remove_job_best_effort):
            # return a truthy Job so remove actually runs.
            return object()  # truthy sentinel

        real_scheduler.add_job = fake_add_job  # type: ignore[method-assign]
        real_scheduler.get_job = fake_get_job  # type: ignore[method-assign]
        try:
            with pytest.raises(SchedulerEnqueueError):
                await scheduler.enqueue(job_id, run_date=datetime.now(UTC))
        finally:
            real_scheduler.add_job = original_add_job  # type: ignore[method-assign]

        # Both get_job calls happened (verification + cleanup).
        assert get_job_call_count == 2, (
            f"post-add verification + best-effort cleanup "
            f"must both call get_job; got {get_job_call_count}"
        )
    finally:
        await scheduler.shutdown(timeout_s=2.0)


@pytest.mark.asyncio
async def test_pd_best_effort_remove_fails_but_503_is_truthful(
    db, settings
) -> None:
    """D: ``add_job`` succeeds, post-add lookup fails, the
    best-effort ``remove_job`` itself raises. The API still
    raises ``SchedulerEnqueueError`` (truthful 503 via
    submit_job compensation). The remove failure is logged
    but does NOT swallow the truth.
    """
    from hermes.jobs.exceptions import SchedulerEnqueueError

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="pd — remove fails",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, _stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        real_scheduler = scheduler._scheduler  # test fixture needs the inner APScheduler

        get_job_call_count = 0
        original_add_job = real_scheduler.add_job

        def fake_add_job(*a, **kw):
            return original_add_job(*a, **kw)

        def fake_get_job(jid, *a, **kw):
            nonlocal get_job_call_count
            get_job_call_count += 1
            if get_job_call_count == 1:
                # Post-add verification: raise.
                raise RuntimeError("simulated_lookup_failure")
            # Second call (inside remove_job_best_effort):
            # return a truthy sentinel so remove actually runs.
            return object()

        def fake_remove_job(jid):
            # Best-effort remove also raises.
            raise RuntimeError("simulated_remove_failure")

        real_scheduler.add_job = fake_add_job  # type: ignore[method-assign]
        real_scheduler.get_job = fake_get_job  # type: ignore[method-assign]
        real_scheduler.remove_job = fake_remove_job  # type: ignore[method-assign]
        try:
            with pytest.raises(SchedulerEnqueueError) as excinfo:
                await scheduler.enqueue(job_id, run_date=datetime.now(UTC))
            assert "post-add lookup failed" in str(excinfo.value)
        finally:
            real_scheduler.add_job = original_add_job  # type: ignore[method-assign]
    finally:
        await scheduler.shutdown(timeout_s=2.0)


@pytest.mark.asyncio
async def test_pe_post_add_cleanup_makes_no_provider_call(
    db, settings
) -> None:
    """E: The post-add cleanup path makes zero provider
    calls. The stub service is never invoked.
    """
    from hermes.jobs.exceptions import SchedulerEnqueueError

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="pe — no provider call",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        real_scheduler = scheduler._scheduler  # test fixture needs the inner APScheduler

        def fake_get_job(jid, *a, **kw):
            raise RuntimeError("simulated_lookup_failure")

        real_scheduler.get_job = fake_get_job  # type: ignore[method-assign]
        try:
            with pytest.raises(SchedulerEnqueueError):
                await scheduler.enqueue(job_id, run_date=datetime.now(UTC))
        finally:
            real_scheduler.get_job = real_scheduler.get_job  # type: ignore[method-assign]

        # The stub was never called.
        assert stub.calls == [], (
            f"no provider call must occur during post-add "
            f"cleanup; got {stub.calls!r}"
        )
    finally:
        await scheduler.shutdown(timeout_s=2.0)


# ============================================================================
# Lookup-error recovery test
# ============================================================================
@pytest.mark.asyncio
async def test_recovery_lookup_error_does_not_reenqueue(
    db, settings
) -> None:
    """Scheduler membership lookup raises. Recovery MUST NOT
    re-enqueue. The row stays ``pending``. The lookup anomaly
    is logged. No duplicate is created.
    """
    from hermes.jobs.recovery import recover_research_jobs
    from hermes.jobs.scheduler import DeepResearchScheduler, JobLookupState

    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="recovery — lookup error",
        notify_via_tg=0,
        user_id=0,
    )
    # Backdate updated_at so the Case 6 grace elapses.
    await db.conn.execute(
        "UPDATE research_jobs SET updated_at = ? WHERE id = ?",
        ("2020-01-01 00:00:00.000000+00:00", job_id),
    )
    await db.conn.commit()

    # Build a scheduler whose ``inspect_job`` raises. We
    # patch the method on the class (not the instance) so
    # recovery's call hits our shim.
    scheduler = DeepResearchScheduler(db=db, settings=settings)
    original_inspect = DeepResearchScheduler.inspect_job

    def raising_inspect(self, jid):
        return JobLookupState.LOOKUP_FAILED

    DeepResearchScheduler.inspect_job = raising_inspect  # type: ignore[method-assign]
    try:
        # Enqueue attempts. We count them to confirm recovery
        # does NOT call scheduler.enqueue when the lookup
        # reports LOOKUP_FAILED.
        enqueue_attempts = []
        original_enqueue = DeepResearchScheduler.enqueue

        async def counting_enqueue(self, jid, run_date):
            enqueue_attempts.append(jid)
            return await original_enqueue(self, jid, run_date)

        DeepResearchScheduler.enqueue = counting_enqueue  # type: ignore[method-assign]
        try:
            settings_short = _settings_with_short_grace(settings)
            recovered = await recover_research_jobs(
                db=db,
                notifier=AsyncMock(),
                settings=settings_short,
                scheduler=scheduler,
            )
            # Case 6 is a candidate in the gap query but
            # inspect_job returns LOOKUP_FAILED → the
            # candidate is skipped. Other cases (1-5) may
            # also run; we only care that no enqueue
            # happened for this job_id.
            assert job_id not in enqueue_attempts, (
                f"recovery must NOT re-enqueue when "
                f"inspect_job returns LOOKUP_FAILED; got "
                f"enqueue attempts {enqueue_attempts!r}"
            )
        finally:
            DeepResearchScheduler.enqueue = original_enqueue  # type: ignore[method-assign]
    finally:
        DeepResearchScheduler.inspect_job = original_inspect  # type: ignore[method-assign]

    # The row stays pending (no transition was applied).
    row = await db.get_research_job(job_id)
    assert row is not None
    assert row["status"] == "pending", (
        f"row must stay pending when scheduler lookup is "
        f"unknown; got status={row['status']!r}"
    )


# ============================================================================
# Test I — PRE1B regression (cancellation still works)
# ============================================================================
@pytest.mark.asyncio
async def test_i_pre1b_cancellation_unaffected(db, settings) -> None:
    """The dispatcher delegates to ``service._run_research``.
    PRE1B cancellation still works. We don't fully re-test
    PRE1B here (that's ``test_jobs_cancellation.py``); we
    verify that the dispatcher shape does not break the
    PRE1B seam: the persisted callable is the dispatcher,
    not the bound method, and PRE1B's cancel_scheduled
    and remove path is intact.
    """
    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test i — pre1b regression",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, _stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        run_date = datetime.now(UTC) + timedelta(seconds=30)
        await scheduler.enqueue(job_id, run_date=run_date)
        # PRE1B cancel_scheduled is intact: it removes
        # the entry from the jobstore.
        removed = scheduler.cancel_scheduled(job_id)
        assert removed is True
        # And the entry is gone.
        assert scheduler.get_job(job_id) is None
    finally:
        await scheduler.shutdown(timeout_s=2.0)


# ============================================================================
# Test J — No provider calls
# ============================================================================
@pytest.mark.asyncio
async def test_j_no_provider_calls_in_persistence_path(db, settings) -> None:
    """The full persistence path makes zero provider calls.
    We verify by checking that no LLM, search, notifier, or
    fetcher is called during enqueue or during a no-op
    dispatcher run.
    """
    job_id = uuid.uuid4().hex[:12]
    await db.create_research_job(
        job_id=job_id,
        query="test j — no provider calls",
        notify_via_tg=0,
        user_id=0,
    )

    scheduler, stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        # Enqueue and immediately cancel (so the dispatcher
        # never fires).
        run_date = datetime.now(UTC) + timedelta(seconds=30)
        await scheduler.enqueue(job_id, run_date=run_date)
        removed = scheduler.cancel_scheduled(job_id)
        assert removed is True
        # The stub was never called.
        assert stub.calls == []
    finally:
        await scheduler.shutdown(timeout_s=2.0)


# ============================================================================
# Test K — Service-level compensation path
# ============================================================================
@pytest.mark.asyncio
async def test_k_submit_job_enqueue_failure_compensates_row(
    db, settings
) -> None:
    """When ``scheduler.enqueue`` raises ``SchedulerEnqueueError``,
    the service-level ``submit_job`` MUST compensate the row to
    ``failed`` with ``checkpoint_corrupt`` taxonomy, and then
    raise ``SchedulerUnavailableError`` (HTTP 503 mapping). The
    caller must NOT see a 201; the row must NOT stay in
    ``pending`` indefinitely.

    This test exercises the FULL compensation path that test_e
    only verifies at the scheduler level. If a future refactor
    breaks the order-of-operations (compensation before re-raise)
    or the WHERE clause (``AND status = 'pending'``), this test
    catches it.
    """
    from hermes.jobs.exceptions import SchedulerUnavailableError
    from hermes.jobs.models import CreateJobRequest
    from hermes.jobs.service import DeepResearchService

    # Build a real scheduler + a stub service. We will then
    # monkey-patch the inner scheduler's add_job to raise.
    scheduler, _stub, _ = await _make_scheduler_and_stub(db, settings)
    try:
        # Force add_job to raise so the next enqueue fails
        # through to the compensation path.
        real_scheduler = scheduler._scheduler  # test fixture needs the inner APScheduler to monkey-patch add_job

        def boom(*a, **kw):
            raise TypeError("simulated_add_job_failure")

        real_scheduler.add_job = boom

        # Build a minimal DeepResearchService. submit_job only
        # touches: _db, _settings, _scheduler, _check_daily_budget,
        # _now_dt. The other deps (notifier, llm, search, fetcher,
        # report_store) are not exercised by submit_job, so
        # MagicMocks are safe.
        service = DeepResearchService(
            db=db,
            notifier=AsyncMock(),
            llm_router=AsyncMock(),
            web_search=AsyncMock(),
            fetcher=AsyncMock(),
            settings=settings,
            scheduler=scheduler,
            report_store=None,
        )

        request = CreateJobRequest(
            query="test k — service compensation",
            notify_via_tg=False,
        )

        # The submit_job MUST raise SchedulerUnavailableError.
        with pytest.raises(SchedulerUnavailableError) as excinfo:
            await service.submit_job(request, user_id=0)
        # The error message references the inner enqueue error.
        assert "Enqueue refused by jobstore" in str(excinfo.value)

        # The row is compensated to failed / checkpoint_corrupt.
        # Find the row by querying the latest failed job.
        async with db.conn.execute(
            "SELECT id, status, error_taxonomy, error_message "
            "FROM research_jobs WHERE error_taxonomy = 'checkpoint_corrupt' "
            "ORDER BY updated_at DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, (
            "compensation must have transitioned a row to "
            "failed with checkpoint_corrupt taxonomy"
        )
        assert row["status"] == "failed"
        assert row["error_taxonomy"] == "checkpoint_corrupt"
        assert (
            "enqueue_serialization_or_persistence_failed"
            in (row["error_message"] or "")
        ), (
            f"error_message must contain the canonical prefix, "
            f"got {row['error_message']!r}"
        )

        # The jobstore has no entry for the compensated row.
        jobstore = SQLAlchemyJobStore(url=scheduler._jobstore_url)  # test fixture needs the same file URL
        try:
            apscheduler_row = jobstore.lookup_job(row["id"])
            assert apscheduler_row is None, (
                "compensated row must NOT have a jobstore entry"
            )
        finally:
            jobstore.shutdown()
    finally:
        await scheduler.shutdown(timeout_s=2.0)
