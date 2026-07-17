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

import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

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

    async def shutdown(self) -> None:
        """Graceful shutdown. Await in-flight jobs hasta 30s."""
        if self._scheduler is None:
            return
        try:
            self._scheduler.shutdown(wait=True)
        except Exception:
            logger.exception("research_scheduler_shutdown_error")
        self._scheduler = None
        logger.info("research_scheduler_stopped")

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
        """
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
