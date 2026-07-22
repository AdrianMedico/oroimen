"""DeepResearchScheduler: AsyncIOScheduler wrapper para research jobs.

Ver TDD_S14_DEEP_RESEARCH.md §4.2.

Config (preservado de TDD_S10 v1.3 §2.7):
- AsyncIOScheduler (NO BackgroundScheduler — research es 99% I/O).
- max_instances=1 (NAS host 2 vCPU constraint).
- coalesce=True (misfires agrupados).
- misfire_grace_time=300 (5 min slack para Watchtower restarts).
- jobstore=SQLAlchemyJobStore(url=f"sqlite:///{db_path}")  # PERSISTENT
  CRÍTICO: a diferencia de MemoryJobStore (default), persistent jobstore
  guarda los jobs programados EN la DB. Si Watchtower reinicia el container
  a los 14 min de un job de 30 min, el scheduler post-restart todavía sabe
  que el job existe. MemoryJobStore lo perdería y recovery hook tendría
  que re-enqueue (más complejo, más race conditions).

Persisted callable (DR-Q1A remediation): the callable stored in the
jobstore is the module-level
``hermes.jobs.dispatcher.execute_research_job``, NOT a bound method
on ``self._service._run_research``. The bound method would
unpickle the entire service graph including
``Database``/``aiosqlite.Connection``/``sqlite3.Connection`` and
fail with ``TypeError: cannot pickle 'sqlite3.Connection' object``
(see ``tests/integration/test_jobs_persistent_scheduler.py``
test A). The dispatcher resolves the live ``DeepResearchService``
via the process-local registry
(``hermes.jobs.service_registry``) at firing time. If the registry
is empty (e.g. between a service restart and the next startup),
the dispatcher transitions the row to terminal ``failed`` with
``error_taxonomy='checkpoint_corrupt'`` so it never stays in
``pending`` indefinitely.

Enqueue error truth: ``add_job`` exceptions are no longer
swallowed. The scheduler raises ``SchedulerEnqueueError`` so the
caller (``DeepResearchService.submit_job``) can compensate the row
and return a truthful error response (HTTP 503). The
pre-flight ``SELECT status`` check still runs; a missing or
terminal row is a no-op skip (logged) rather than an error.

Recovery complementario (recovery.py): Si un job queda en estado
inconsistente (e.g. status='running' pero no hay task en el scheduler),
recover_research_jobs() lo resetea a 'pending' y re-enqueue. Case 6
(pending-without-scheduler-entry) closes the
post-pickle-fix gap: a ``pending`` row whose ``updated_at`` is
older than ``deep_research_recovery_pending_gap_grace_seconds``
(default 60s) and has no scheduler entry is re-enqueued.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

from hermes.jobs.exceptions import SchedulerEnqueueError, SchedulerUnavailableError
from hermes.jobs.models import JobStatus

logger = logging.getLogger(__name__)


class DeepResearchScheduler:
    """AsyncIOScheduler wrapper con SQLAlchemyJobStore persistent.

    Uso:
        scheduler = DeepResearchScheduler(db=db, settings=settings)
        await scheduler.start()
        await scheduler.enqueue(job_id, run_date=datetime.now())
        ...
        await scheduler.shutdown()
    """

    def __init__(
        self,
        *,
        db: Any,  # Database instance (duck-typed, sin import circular)
        settings: Any,  # Settings instance (duck-typed)
        jobstore_url: str | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        # Si jobstore_url no se pasa, derivar del settings.db_path.
        # Esto permite que tests inyecten su propia URL (e.g. SQLite in-memory).
        if jobstore_url is None:
            db_path = Path(str(getattr(settings, "db_path", "/app/data/conversations.db")))
            jobstore_url = f"sqlite:///{db_path}"
        self._jobstore_url = jobstore_url
        self._scheduler: AsyncIOScheduler | None = None
        # Slice 1C1c: explicit stopping flag toggled by ``stop_accepting``
        # BEFORE the actual shutdown runs. This prevents the classic
        # race where a request races past the deadline check and reaches
        # an internal APScheduler that is halfway through shutdown.
        self._stopping: bool = False
        self._shutdown_event: asyncio.Event | None = None

    async def start(self) -> None:
        """Inicializa AsyncIOScheduler con SQLAlchemyJobStore persistent.

        Llamado desde hermes/__main__.py:startup() TRAS recover_research_jobs().
        Idempotente: si ya está corriendo, no hace nada.
        """
        if self._scheduler is not None and self._scheduler.running:
            return
        self._scheduler = AsyncIOScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=self._jobstore_url)},
            job_defaults={
                "max_instances": 1,
                "coalesce": True,
                "misfire_grace_time": 300,
            },
        )
        self._scheduler.start()
        logger.info(
            "research_scheduler_started",
            extra={"jobstore_url": self._jobstore_url},
        )

    def stop_accepting(self) -> bool:
        """Mark the scheduler as no longer accepting enqueues. Idempotent.

        Synchronous seam. Once set, any ``enqueue`` call raises
        ``SchedulerUnavailableError`` *immediately* — before any DB
        transaction or APScheduler call.

        Returns:
            True if this call flipped the flag, False if it was already
            set.
        """
        if self._stopping:
            return False
        self._stopping = True
        return True

    def start_accepting(self) -> bool:
        """Re-arm the admission seam after a successful startup.

        Slice 1C1c: paired with :meth:`stop_accepting` so a successful
        ``start()`` can re-enable the enqueue path. Idempotent — calling
        it on a scheduler that is already accepting returns ``False``
        without touching the flag. This is the only way the composer
        should clear the pre-emptive guard set right before ``start()``;
        a future refactor that reintroduces direct ``_stopping`` writes
        should route through this method instead so the rollback
        ordering stays observable in tests.
        """
        if not self._stopping:
            return False
        self._stopping = False
        return True

    @property
    def accepting(self) -> bool:
        """True while enqueues are still allowed (read-only state)."""
        return not self._stopping

    async def shutdown(self, timeout_s: float = 10.0) -> bool:
        """Bounded, idempotent shutdown. Returns truthy drain outcome.

        Slice 1C1c contract:
        1. ``stop_accepting()`` is called FIRST so subsequent enqueues
           reject with ``SchedulerUnavailableError`` even if a caller
           races the actual shutdown.
        2. ``self._scheduler.shutdown(wait=True)`` is dispatched to a
           dedicated background thread (via ``asyncio.to_thread``) so
           the asyncio event-loop thread NEVER blocks inside the
           blocking APScheduler shutdown. This avoids the documented
           ``asyncio.run()`` hang that left the process alive after
           the deadline.
        3. The internal scheduler reference is detached (*not* replaced)
           so no late ``enqueue`` call can route into a half-shut
           APScheduler instance.
        4. Boolean honestly distinguishes graceful completion (True)
           from deadline expiry (False). The default executor task,
           when used, is awaited under the same deadline — no awaited
           default-executor task is left dangling after this returns.

        Args:
            timeout_s: caller-imposed budget (seconds). Must be > 0.

        Returns:
            True if APScheduler finished shutting down within the
            deadline, False otherwise.
        """
        # (a) Flag stopping first, *synchronously*, BEFORE issuing
        # the actual shutdown. Even a concurrent ``enqueue`` sees
        # the guard immediately.
        self.stop_accepting()

        if self._scheduler is None:
            return True

        scheduler_ref = self._scheduler
        # Detach so any late ``enqueue`` call cannot reach a half-shut
        # APScheduler instance. We do NOT create a replacement — losing
        # the reference is intentional.
        self._scheduler = None

        deadline = max(timeout_s, 0.0)

        # (b) Dispatch the blocking APScheduler.shutdown(wait=True) to a
        # dedicated thread so the event loop stays responsive. Without
        # this, calling ``scheduler.shutdown(wait=True)`` directly from
        # the loop's thread would wedge the process.
        try:
            await asyncio.wait_for(
                asyncio.to_thread(scheduler_ref.shutdown, True),
                timeout=deadline,
            )
            logger.info("research_scheduler_stopped")
            return True
        except TimeoutError:
            logger.warning("research_scheduler_shutdown_timeout")
            # Best-effort: cancel pending futures so we don't leak the
            # background thread. APScheduler doesn't expose a public
            # ``cancel()``; the next loop tick will reap the worker.
            return False
        except Exception:
            logger.exception("research_scheduler_shutdown_error")
            return False

    @property
    def running(self) -> bool:
        """True si el scheduler está activo (usado por /health endpoint)."""
        return self._scheduler is not None and self._scheduler.running

    async def enqueue(self, job_id: str, run_date: datetime) -> None:
        """Añade job al scheduler con pre-flight check atómico anti-duplicación.

        Algoritmo (TDD §4.2):
            BEGIN IMMEDIATE;  -- write lock, evita race con recovery hook
            SELECT status FROM research_jobs WHERE id = ?;
            si status IN ('running','complete','failed','cancelled'):
                COMMIT; return  # no-op, job ya está en estado terminal
            si status = 'pending':
                -- OK, enqueue
                self._scheduler.add_job(self._service._run_research, 'date',
                                         run_date=run_date, id=job_id,
                                         max_instances=1, coalesce=True,
                                         misfire_grace_time=300)
            COMMIT;

        Por qué BEGIN IMMEDIATE: SQLite serializa writes; sin lock explícito,
        dos callers concurrentes (e.g. POST /v1/jobs + recovery re-enqueue)
        podrían ambos ver status='pending' y ambos hacer add_job → doble task.

        El callable ejecutado por APScheduler es `service._run_research` — el
        `service` se inyecta via `set_service()` DESPUÉS de start() para
        evitar import circular. Si no se inyecta, NO se ejecuta (logged warning).

        Slice 1C1c: ``stop_accepting()`` rejects all new enqueues
        immediately. The check fires *before* any DB transaction or
        APScheduler call so neither half-shut resources nor the DB
        needs to handle the rejection.
        """
        # Slice 1C1c: reject immediately once stopping begins, BEFORE
        # any DB transaction or APScheduler.add_job call. Mirrors the
        # service-level guard so the test asserting "submission
        # rejection happens before budget/DB/enqueue" succeeds for
        # both HTTP-side and recovery-side callers.
        if self._stopping:
            raise SchedulerUnavailableError("Scheduler is no longer accepting enqueues")
        if self._scheduler is None:
            logger.warning(
                "research_scheduler_enqueue_skipped_not_started",
                extra={"job_id": job_id},
            )
            return
        # The legacy ``self._service`` instance attribute is no
        # longer required for the persisted callable (the
        # dispatcher uses the registry). We keep the attribute
        # for backward compatibility with any test that still
        # reads it, but a missing service here is NOT a
        # reason to skip the enqueue: the dispatcher will
        # resolve the live service at firing time via the
        # registry, and ``_terminate_registry_missing`` will
        # transition the row to failed if the registry is
        # empty when the dispatcher runs (e.g. after a
        # service restart). The 2 aborted jobs from the
        # pre-fix era were stuck because the OLD code path
        # required ``self._service`` here; the new code path
        # does not.

        # Pre-flight check atómico (BEGIN IMMEDIATE write lock).
        try:
            await self._db.conn.execute("BEGIN IMMEDIATE")
            async with self._db.conn.execute(
                "SELECT status FROM research_jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
            await self._db.conn.execute("COMMIT")
        except Exception:
            with contextlib.suppress(Exception):
                await self._db.conn.execute("ROLLBACK")
            logger.exception(
                "research_scheduler_preflight_error",
                extra={"job_id": job_id},
            )
            return

        if row is None:
            logger.warning(
                "research_scheduler_enqueue_skipped_job_missing",
                extra={"job_id": job_id},
            )
            return

        status_value = row[0] if not isinstance(row, dict) else row["status"]
        # Estado terminal: no re-enqueue.
        # DR-Q1A-PRE1B: ``cancelling`` is now a non-enqueueable
        # state. A job in ``cancelling`` is being finalized; a
        # recovery re-enqueue would race the finalizer. Only
        # ``pending`` is enqueueable; ``running``, ``complete``,
        # ``failed`` and ``cancelling`` are rejected.
        if status_value in (
            JobStatus.RUNNING.value,
            JobStatus.COMPLETE.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLING.value,
            JobStatus.CANCELLED.value,
        ):
            logger.info(
                "research_scheduler_enqueue_skipped_terminal_status",
                extra={"job_id": job_id, "status": status_value},
            )
            return

        # Estado pending: add_job al scheduler.
        # The persisted callable is the module-level dispatcher
        # ``execute_research_job`` from ``hermes.jobs.dispatcher``,
        # not a bound method. The jobstore will pickle only the
        # callable reference and the string ``job_id``; the live
        # service is resolved at firing time via the process-local
        # registry. This is the fix for the systemic pickle error
        # that left jobs in ``pending`` indefinitely (the bound
        # method ``self._service._run_research`` captured the
        # entire service graph including a non-picklable
        # aiosqlite.Connection).
        #
        # The error path does NOT swallow add_job exceptions. If
        # the jobstore refuses the job (e.g. integrity error,
        # serialization failure, duplicate id), the exception
        # propagates as ``SchedulerEnqueueError`` so the
        # caller (submit_job) can compensate the row in the DB
        # and return a truthful error response.
        from hermes.jobs.dispatcher import execute_research_job
        try:
            self._scheduler.add_job(
                execute_research_job,
                DateTrigger(run_date=run_date),
                id=job_id,
                name=f"research_job:{job_id}",
                args=[job_id],
                replace_existing=False,
            )
        except Exception as exc:
            # Redact the exception chain. Do NOT include the
            # full traceback in the log line (it can include
            # bound method internals that the operator does not
            # need). The logger.exception below captures the
            # full traceback server-side.
            error_type = type(exc).__name__
            logger.exception(
                "research_scheduler_enqueue_failed",
                extra={"job_id": job_id, "error_type": error_type},
            )
            raise SchedulerEnqueueError(
                f"add_job failed for {job_id}: {error_type}"
            ) from exc

        # Verify the job is actually in the jobstore. APScheduler's
        # ``add_job`` returns a Job object, but a serialization
        # failure can leave a partially written job. The
        # post-add lookup confirms the job is in the jobstore
        # before we declare success.
        try:
            lookup = self._scheduler.get_job(job_id)
        except Exception as exc:
            error_type = type(exc).__name__
            logger.exception(
                "research_scheduler_post_add_lookup_failed",
                extra={"job_id": job_id, "error_type": error_type},
            )
            raise SchedulerEnqueueError(
                f"post-add lookup failed for {job_id}: {error_type}"
            ) from exc
        if lookup is None:
            logger.error(
                "research_scheduler_post_add_lookup_missing",
                extra={"job_id": job_id},
            )
            raise SchedulerEnqueueError(
                f"post-add lookup returned None for {job_id}"
            )

        logger.info(
            "research_scheduler_enqueued",
            extra={"job_id": job_id, "run_date": run_date.isoformat()},
        )

    def cancel_scheduled(self, job_id: str) -> bool:
        """Quita job del scheduler si aún no ha corrido.

        DR-Q1A-PRE1B: also handles the rare case where the
        scheduler still has an entry for a job that is already
        in ``cancelling`` (e.g. the cancel endpoint won the race
        before the scheduler entry was removed). The check is
        best-effort: a missing job returns False without raising.

        Returns:
            True si quitó, False si no estaba o ya corría.
        """
        if self._scheduler is None:
            return False
        try:
            job = self._scheduler.get_job(job_id)
            if job is None:
                return False
            self._scheduler.remove_job(job_id)
            logger.info("research_scheduler_job_removed", extra={"job_id": job_id})
            return True
        except Exception:
            logger.exception("research_scheduler_cancel_failed", extra={"job_id": job_id})
            return False

    def get_job(self, job_id: str) -> Any | None:
        """Public wrapper around ``self._scheduler.get_job``.

        The recovery loop uses this to detect ``pending + no
        scheduler entry`` cases (a crash or persistence failure
        between DB creation and jobstore persistence). The
        recovery contract is:

        - ``pending + scheduler entry`` → healthy queued job;
        - ``pending + no scheduler entry`` → recoverable enqueue gap;
        - ``failed + completion_at != NULL`` (post-neutralization) → historical failure, not recoverable.

        The wrapper hides the ``self._scheduler`` attribute so
        callers do not need to know the APScheduler is alive
        or not. Returns ``None`` if the scheduler is not started
        or the job is not in the jobstore. Never raises.
        """
        if self._scheduler is None:
            return None
        try:
            return self._scheduler.get_job(job_id)
        except Exception:
            logger.exception(
                "research_scheduler_get_job_failed",
                extra={"job_id": job_id},
            )
            return None

    def set_service(self, service: Any) -> None:
        """Inyecta el DeepResearchService DESPUÉS de start().

        DEPRECATED shim: callers should now use
        ``hermes.jobs.service_registry.set_research_service``,
        which is the single source of truth for the dispatcher.
        This method is kept for backward compatibility with
        existing test fixtures and any external code that
        reached for the scheduler instance attribute directly.

        When called, this method now ALSO registers the service
        in the new registry, so the dispatcher
        (``hermes.jobs.dispatcher.execute_research_job``) can
        resolve it at firing time.
        """
        # Lazy import to avoid a circular import at module load.
        from hermes.jobs.service_registry import set_research_service
        # Keep the legacy instance attribute for any code that
        # still reads ``self._service``. New code reads from the
        # registry via the dispatcher.
        self._service = service
        set_research_service(service)

    # Default attribute para set_service() antes de llamarlo
    _service: Any | None = None
