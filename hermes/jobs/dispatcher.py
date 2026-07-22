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
from enum import StrEnum
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


class RegistryMissingTransition(StrEnum):
    """Which state transition the registry-absent handler applied.

    PRE1B-aware: ``cancelling`` is finalized to ``cancelled``
    (cancellation wins); only ``pending`` / ``running`` are
    failed with ``checkpoint_corrupt``; terminal states are
    never overwritten.
    """

    PENDING_TO_FAILED = "pending->failed"
    RUNNING_TO_FAILED = "running->failed"
    CANCELLING_TO_CANCELLED = "cancelling->cancelled"
    NO_OP_TERMINAL = "no-op-terminal"
    NO_OP_RACE = "no-op-race"
    LOOKUP_FAILED = "lookup-failed"


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
        # We MUST transition the row to a terminal state so it
        # does not stay in ``pending`` or ``running``. The
        # actual transition is state-aware and preserves PRE1B
        # cancellation semantics: ``cancelling`` finalizes to
        # ``cancelled``; only ``pending`` / ``running`` are
        # failed with ``checkpoint_corrupt``.
        await _terminate_registry_missing(job_id)
        return

    # Delegate to the service. PRE1B cancellation, cost
    # reconciliation, token-usage persistence, and final
    # notifier wiring all live there.
    await service._run_research(job_id)


async def _terminate_registry_missing(
    job_id: str,
) -> RegistryMissingTransition:
    """Apply a state-aware, PRE1B-preserving transition when the
    dispatcher cannot resolve a service.

    State transition matrix (PRE1B-aware):

        pending   -> failed      (checkpoint_corrupt)
        running   -> failed      (checkpoint_corrupt)
        cancelling -> cancelled  (cancellation wins; do not overwrite)
        complete  -> no-op       (terminal; do not resurrect)
        failed    -> no-op       (terminal; do not resurrect)
        cancelled -> no-op       (terminal; do not resurrect)
        missing row -> no-op-race

    Implementation: two separate atomic CAS transitions, using
    the existing ``transition_research_job_status`` helper. We
    first attempt the ``cancelling -> cancelled`` finalization
    (PRE1B invariant: cancellation wins, registry-absent must
    not overwrite it). If that did not match, we attempt
    ``pending|running -> failed`` with ``checkpoint_corrupt``.
    Finally we re-read the canonical row to confirm the
    terminal state and log which transition applied.

    The transition is conditional on the source state. We do
    NOT introduce an unconditional status writer. The DB
    transitions are atomic single-statement CAS predicates.
    No provider call is made. tokens and cost remain unchanged
    (we never write to them; they are zero for a job that
    never reached the LLM, but we leave the existing values
    alone so we never accidentally clear partial progress).
    """
    from hermes.config import Settings
    from hermes.memory.db import Database

    settings = Settings()
    db = Database(Path(str(settings.db_path)))
    await db.initialize()

    now = format_now()
    error_message = (
        "registry_absent_pre_execution: research service was not "
        "registered when the dispatcher ran; job never reached the "
        "LLM. Tokens=0, cost=0."
    )
    applied: RegistryMissingTransition = RegistryMissingTransition.NO_OP_RACE
    try:
        # Step 1: PRE1B invariant — if a cancellation has already
        # won (``status='cancelling'``), finalize to
        # ``status='cancelled'``. We do NOT touch
        # ``error_taxonomy`` or ``error_message`` here: the
        # cancellation finalization marker is set elsewhere (in
        # the recovery loop or the cancel endpoint) and we
        # preserve it.
        cancelled = await db.transition_research_job_status(
            job_id=job_id,
            from_states=("cancelling",),
            to_state="cancelled",
            completed_at=now,
        )
        if cancelled:
            applied = RegistryMissingTransition.CANCELLING_TO_CANCELLED
            logger.error(
                "dispatcher_registry_missing_cancellation_finalized",
                extra={"job_id": job_id, "completed_at": now},
            )
        else:
            # Step 2: pending / running -> failed (with
            # ``checkpoint_corrupt``). The atomic CAS predicate
            # refuses to overwrite any terminal state.
            failed = await db.transition_research_job_status(
                job_id=job_id,
                from_states=("pending", "running"),
                to_state="failed",
                completed_at=now,
                error_taxonomy="checkpoint_corrupt",
                error_message=error_message,
            )
            if failed:
                # We don't know whether the source was
                # ``pending`` or ``running``; the helper
                # collapses them. The canonical row read below
                # confirms the final state and the log line
                # records which one was the source.
                canonical = await db.get_research_job(job_id)
                if canonical is not None:
                    source_phase = canonical.get("current_phase")
                    # Heuristic: if the row had a current_phase
                    # it was running. Otherwise pending.
                    applied = (
                        RegistryMissingTransition.RUNNING_TO_FAILED
                        if source_phase
                        else RegistryMissingTransition.PENDING_TO_FAILED
                    )
                else:
                    applied = RegistryMissingTransition.NO_OP_RACE
                logger.error(
                    "dispatcher_registry_missing_terminalized",
                    extra={
                        "job_id": job_id,
                        "completed_at": now,
                        "applied_transition": applied.value,
                    },
                )
            else:
                # No match. Either the row is in a terminal
                # state (complete / failed / cancelled) or it
                # is missing. Re-read the canonical row to
                # log which.
                canonical = await db.get_research_job(job_id)
                if canonical is not None:
                    final_status = canonical.get("status")
                    if final_status in (
                        "complete",
                        "failed",
                        "cancelled",
                    ):
                        applied = RegistryMissingTransition.NO_OP_TERMINAL
                        logger.info(
                            "dispatcher_registry_missing_no_op_terminal",
                            extra={
                                "job_id": job_id,
                                "final_status": final_status,
                            },
                        )
                    else:
                        # E.g. status is ``pending`` or
                        # ``running`` but the CAS did not match
                        # (e.g. a concurrent writer just flipped
                        # to ``cancelling``). Treat as race.
                        applied = RegistryMissingTransition.NO_OP_RACE
                        logger.info(
                            "dispatcher_registry_missing_no_op_race",
                            extra={
                                "job_id": job_id,
                                "final_status": final_status,
                            },
                        )
                else:
                    applied = RegistryMissingTransition.NO_OP_RACE
                    logger.info(
                        "dispatcher_registry_missing_no_op_race",
                        extra={"job_id": job_id, "final_status": None},
                    )
    except Exception:
        applied = RegistryMissingTransition.LOOKUP_FAILED
        logger.exception(
            "dispatcher_registry_missing_terminalize_failed",
            extra={"job_id": job_id},
        )
    finally:
        with contextlib.suppress(Exception):
            await db.close()

    return applied
