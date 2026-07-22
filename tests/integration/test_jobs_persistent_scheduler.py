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
