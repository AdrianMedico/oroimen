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

Recovery complementario (recovery.py): Si un job queda en estado
inconsistente (e.g. status='running' pero no hay task en el scheduler),
recover_research_jobs() lo resetea a 'pending' y re-enqueue.
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

from hermes.jobs.exceptions import SchedulerUnavailableError
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
        if self._service is None:
            logger.warning(
                "research_scheduler_enqueue_skipped_no_service",
                extra={"job_id": job_id},
            )
            return

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
        if status_value in (
            JobStatus.RUNNING.value,
            JobStatus.COMPLETE.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ):
            logger.info(
                "research_scheduler_enqueue_skipped_terminal_status",
                extra={"job_id": job_id, "status": status_value},
            )
            return

        # Estado pending (o cancelling → re-enqueue OK): add_job al scheduler.
        try:
            self._scheduler.add_job(
                self._service._run_research,
                DateTrigger(run_date=run_date),
                id=job_id,
                name=f"research_job:{job_id}",
                args=[job_id],
                replace_existing=False,
            )
            logger.info(
                "research_scheduler_enqueued",
                extra={"job_id": job_id, "run_date": run_date.isoformat()},
            )
        except Exception:
            logger.exception(
                "research_scheduler_enqueue_failed",
                extra={"job_id": job_id},
            )

    def cancel_scheduled(self, job_id: str) -> bool:
        """Quita job del scheduler si aún no ha corrido.

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

    def set_service(self, service: Any) -> None:
        """Inyecta el DeepResearchService DESPUÉS de start().

        Patrón para evitar import circular: el scheduler se construye
        primero (en __main__.py:startup) sin service, y luego se inyecta
        el service cuando está listo (tras recovery).
        """
        self._service = service

    # Default attribute para set_service() antes de llamarlo
    _service: Any | None = None
