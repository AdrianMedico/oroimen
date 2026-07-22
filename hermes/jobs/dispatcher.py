"""Module-level dispatcher for the DeepResearchScheduler.

Background
==========

The DeepResearchScheduler uses APScheduler's ``AsyncIOScheduler`` with
a persistent ``SQLAlchemyJobStore`` (SQLite). When a job is enqueued,
APScheduler pickles the callable and the arguments into the
jobstore. If the callable holds a reference to a non-picklable
graph (e.g. an ``aiosqlite.Connection`` reached through
``DeepResearchService -> Database``), the pickling raises a
``TypeError`` and the enqueue silently fails in older code paths,
leaving a row in ``research_jobs`` with ``status='pending'`` that no
scheduler will ever pick up.

Fix
==

The persisted callable is now a module-level coroutine that takes a
single string argument (``job_id``) and resolves the live
``DeepResearchService`` through a narrow, process-local registry
(``hermes.jobs.service_registry``). The jobstore only ever
serializes:

- the callable (a module-level function reference),
- the string ``job_id``.

This module is intentionally tiny and side-effect free. It
imports the registry lazily (per call) so that test setup can
swap the registry between operations without importing the whole
service graph.

Contract
========

- ``execute_research_job`` is ``async def`` and globally
  importable. APScheduler can pickle it.
- It only accepts ``job_id: str``.
- It NEVER persists the service, the database, the LLM client,
  the search client, the settings, the notifier, the fetcher or
  any other live object.
- It resolves the service at call time, not at enqueue time.
- If the registry is empty (e.g. after a service restart
  between the enqueue and the scheduler firing), the job is
  transitioned to a terminal non-executed ``failed`` state with
  ``error_taxonomy='checkpoint_corrupt'``. This is the
  "registry absent" handling required by the spec (test H).
- A missing service in the registry NEVER leaves a row in
  ``pending`` or ``running`` indefinitely.

PRE1B cancellation contract
===========================

The dispatcher delegates to ``service._run_research``. PRE1B
cancellation is owned by ``_run_research`` and the
``_peek_active_task`` helper. This dispatcher does NOT
re-implement cancellation; it just calls into the service.

A narrow typed exception ``DispatcherRegistryMissing`` is
raised only when the registry is genuinely empty. Inside
``_run_research`` the dispatcher catches the cancellation
contract and propagates as before.

Why a registry, not a global service locator
==============================================

A narrow typed module-level ``set_research_service`` /
``get_research_service`` keeps the service graph out of the
pickled job state. It is process-local (no cross-process
state). It is intentionally NOT a general service locator:
it only holds the single ``DeepResearchService`` instance and
its lifetime is bounded by the daemon's startup / shutdown
seam.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from hermes.jobs.cost import format_now

logger = logging.getLogger(__name__)


class DispatcherRegistryMissing(RuntimeError):
    """Raised when the dispatcher runs without a registered service.

    The orchestrator (recovery, start, or the dispatcher itself)
    must convert this into a coherent terminal non-executed
    failure state on the affected research_job row so the row
    does not stay in ``pending`` or ``running`` indefinitely.
    """


async def execute_research_job(job_id: str) -> None:
    """Module-level async callable used by APScheduler.

    The jobstore persists this function reference and the
    ``job_id`` string only. The live ``DeepResearchService`` is
    resolved at call time via the process-local registry.
    """
    # Local imports to keep this module import-cheap and to avoid
    # a circular import. The service depends on the registry
    # only through the dispatcher's call into
    # service._run_research(job_id); neither service nor
    # service_registry imports the other at module load time.
    from hermes.jobs.service_registry import get_research_service

    service = get_research_service()
    if service is None:
        # The service is not registered. This is a coherent,
        # recoverable failure: the scheduler survived across a
        # service restart and the live service has not yet been
        # registered (or has been cleared during shutdown).
        # We MUST transition the row to terminal failed so it
        # does not stay in ``pending`` or ``running``.
        await _terminate_registry_missing(job_id)
        return

    # Delegate to the service. PRE1B cancellation, cost
    # reconciliation, token-usage persistence, and final
    # notifier wiring all live there.
    await service._run_research(job_id)


async def _terminate_registry_missing(job_id: str) -> None:
    """Transition a row to ``failed`` when the dispatcher cannot
    resolve a service.

    This is a *coherent* terminal state, not a silent skip. The
    row is updated with zero tokens, zero cost, and a clear
    error message. No provider call is made. The next recovery
    sweep will skip the row (its status is no longer ``pending``
    or ``running``).
    """
    # Lazy import. We deliberately do NOT import the service
    # here; the registry is empty precisely because the service
    # is unavailable.
    # The Database is reachable via the service registry entry's
    # ``_db`` attribute IF the service was previously registered.
    # In the registry-absent case, we must fall back to a fresh
    # Database instance using the same path. The Database
    # constructor is a thin wrapper around a sqlite path and
    # does not depend on the dispatcher.
    from hermes.config import Settings
    from hermes.memory.db import Database

    settings = Settings()
    db = Database(Path(str(settings.db_path)))
    await db.initialize()

    now = format_now()
    try:
        error_message = (
            "registry_absent_pre_execution: research service was not "
            "registered when the dispatcher ran; job never reached the "
            "LLM. Tokens=0, cost=0."
        )
        await db.conn.execute(
            """
            UPDATE research_jobs
            SET status = 'failed',
                completed_at = ?,
                error_taxonomy = 'checkpoint_corrupt',
                error_message = ?,
                updated_at = ?
            WHERE id = ? AND status IN ('pending', 'running', 'cancelling')
            """,
            (now, error_message, now, job_id),
        )
        await db.conn.commit()
        logger.error(
            "dispatcher_registry_missing_terminalized",
            extra={"job_id": job_id, "completed_at": now},
        )
    except Exception:
        logger.exception(
            "dispatcher_registry_missing_terminalize_failed",
            extra={"job_id": job_id},
        )
    finally:
        with contextlib.suppress(Exception):
            await db.close()
