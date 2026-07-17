"""Backup scheduler (Sprint 8 — S8.4).

CRÍTICO (Gemini review 2026-06-24): `sqlite3.Connection.backup()` es
síncrono y bloqueante. Si se ejecuta directamente en el AsyncIOScheduler,
congela el event loop de FastAPI durante el backup (segundos críticos).

Solución: `asyncio.to_thread()` ejecuta el código bloqueante en un
ThreadPoolExecutor dedicado de 2 workers (<CPU_MODEL> = 4 cores,
Oroimen tiene 2 asignados; backup es I/O bound, no necesita más).

References:
    docs/TDD_SQLITE_WAL_BACKUP.md §3.3
    Vikunja #130 [S8.4]
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from hermes.backup import backup_db_main
from hermes.memory.db import Database
from hermes.memory.sleep_cycle import SleepCycle

logger = logging.getLogger(__name__)

# Pool dedicado: 2 workers max (<CPU_MODEL> = 4 cores,
# Hermes tiene 2 asignados via docker-compose cpus: "2.0").
# Backup es I/O bound (read DB -> write DB), no CPU bound.
# Aislado del default executor de asyncio (que tiene min(32, cpu+4) threads).
_BACKUP_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="hermes-backup",
)


async def backup_job_wrapper() -> None:
    """Async wrapper que delega el backup bloqueante a un thread del pool.

    CRÍTICO: sin este wrapper, backup_db_main() se ejecutaría en el event
    loop de FastAPI y bloquearía HTTP API, /health, streaming, Telegram
    durante los segundos del backup. Con asyncio.to_thread, el código
    síncrono de sqlite3 va a un thread del worker pool, dejando el event
    loop libre.
    """
    try:
        result = await asyncio.to_thread(backup_db_main)
        if result is not None:
            logger.info("db_backup_scheduled_ok", extra={"backup": str(result)})
    except Exception:
        # backup_db_main ya loggea el error específico, no duplicar
        logger.exception("db_backup_scheduled_failed")


class BackupScheduler:
    """Wraps APScheduler for the daily WAL backup job.

    Usa un ThreadPoolExecutor dedicado de 2 workers (ver _BACKUP_EXECUTOR)
    para evitar saturar el <CPU_MODEL> con el default executor.
    """

    def __init__(self, *, hour: int = 3, minute: int = 30) -> None:
        self._scheduler = AsyncIOScheduler()
        self._hour = hour
        self._minute = minute

    async def start(self) -> None:
        self._scheduler.add_job(
            backup_job_wrapper,  # wrapper asíncrono (asyncio.to_thread)
            CronTrigger(hour=self._hour, minute=self._minute),
            id="backup_db_daily",
            name="Backup SQLite WAL (online, atomic, off-loop)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=600,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info(
            "backup_scheduler_started",
            extra={"hour": self._hour, "minute": self._minute},
        )

    async def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)
            logger.info("backup_scheduler_stopped")
        # Cerrar el pool dedicado
        _BACKUP_EXECUTOR.shutdown(wait=True)
        logger.info("backup_executor_shutdown")


# Sprint 9.2: Sleep Cycle (memory fact extraction)
# ----------------------------------------------------------------------
# Pipeline opt-in que corre diariamente (default 04:00 AM) para extraer
# memory facts de las conversaciones del dia anterior, deduplicar contra
# staging, y promover a consolidated tras N menciones (threshold).
#
# Diferencias con BackupScheduler:
# - Opt-in via settings.sleep_cycle_enabled (default False)
# - SleepCycle.run() es ASYNC (no bloqueante), no necesita ThreadPoolExecutor
# - Default 04:00 AM (antes que el backup de 03:30? No, despues: 04:00 > 03:30)
# - Si el user no tiene sleep_cycle_enabled, el scheduler no se inicia


async def sleep_cycle_job_wrapper() -> None:
    """Async wrapper que ejecuta el Sleep Cycle pipeline.

    SleepCycle.run() ya es async y maneja su propio event loop yield
    (asyncio.sleep(0.2) entre conversaciones). Solo loggeamos metricas
    y errores a nivel de scheduler.
    """
    # Singletons se inyectan via __main__ antes de start(). Ver
    # hermes.__main__.run() para ver el wire-up completo.
    sleep_cycle = _sleep_cycle_instance
    if sleep_cycle is None:
        logger.warning("sleep_cycle_job_skipped_no_instance")
        return
    try:
        metrics = await sleep_cycle.run()
        logger.info("sleep_cycle_scheduled_ok", extra=metrics)
    except Exception:
        logger.exception("sleep_cycle_scheduled_failed")


# Singleton inyectado por __main__.run() antes de iniciar el scheduler.
# None hasta entonces (job no se ejecuta si no hay instance).
# P2 Copilot review 2026-06-26: type explicito (no `object`) para que
# mypy/pyright detecten cambios de API en SleepCycle.
_sleep_cycle_instance: SleepCycle | None = None


class SleepCycleScheduler:
    """Wraps APScheduler for the daily memory fact extraction job.

    Sprint 9.2 — opt-in (settings.sleep_cycle_enabled). Si el user no
    activa esta feature, el scheduler no se inicia. Esto es importante
    porque el Sleep Cycle hace LLM calls (coste) y queremos que el user
    decida explicitamente si lo quiere.
    """

    def __init__(
        self,
        *,
        hour: int = 4,
        minute: int = 0,
        sleep_cycle: SleepCycle | None = None,
    ) -> None:
        self._scheduler = AsyncIOScheduler()
        self._hour = hour
        self._minute = minute
        # Inyectado por __main__.run() para evitar import circular
        global _sleep_cycle_instance
        _sleep_cycle_instance = sleep_cycle

    async def start(self) -> None:
        self._scheduler.add_job(
            sleep_cycle_job_wrapper,
            CronTrigger(hour=self._hour, minute=self._minute),
            id="sleep_cycle_daily",
            name="Memory fact extraction (Sleep Cycle, opt-in)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,  # 1h: si se atrasa, ejecutar ASAP
            coalesce=True,
        )
        self._scheduler.start()
        logger.info(
            "sleep_cycle_scheduler_started",
            extra={"hour": self._hour, "minute": self._minute},
        )

    async def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)
            logger.info("sleep_cycle_scheduler_stopped")
        global _sleep_cycle_instance
        _sleep_cycle_instance = None


# Sprint 9.4: Conversation cleanup (archive stale conversations)
# ----------------------------------------------------------------------
# Job periodico que archiva convs activas con updated_at viejo para
# prevenir el bug 9.3.2b (UNIQUE constraint violation en
# idx_conversations_unique_active cuando una conv huérfana queda sin
# archivar por crash previo).
#
# Diferencias con BackupScheduler / SleepCycleScheduler:
# - IntervalTrigger (cada N min), NO CronTrigger (diario).
# - Ejecuta directamente en el event loop (es SQL rapido, no I/O bound).
# - Opt-out via settings.cleanup_enabled (default True: el bug es real,
#   no queremos que el user tenga que activarlo explicitamente).
# - max_instances=1 para evitar overlap si una ejecucion tarda mas que
#   el intervalo (improbable, pero defensive).


async def conversation_cleanup_job_wrapper(db: Database, max_age_seconds: int) -> None:
    """Wrapper async que ejecuta el archive de convs stale.

    No usa asyncio.to_thread porque el SQL UPDATE es rapido
    (indexado por is_archived + updated_at). En el peor caso,
    1000s de convs = <100ms. Ejecutar en event loop es seguro.
    """
    try:
        n = await db.archive_stale_conversations(max_age_seconds=max_age_seconds)
        logger.info(
            "conversation_cleanup_scheduled_ok",
            extra={"archived": n, "max_age_seconds": max_age_seconds},
        )
    except Exception:
        logger.exception("conversation_cleanup_scheduled_failed")


# Sprint 12.1 (TDD_S12_DELETE_AND_SYNC.md §7.4): hard-delete de convs
# tombstoned (soft-deleted) que han pasado la ventana de retencion
# (default 7d, configurable via HERMES_CONVERSATION_RETENTION_DAYS).
# El CASCADE en messages.conversation_id borra los rows (que ya estan
# cifrados) en la misma operacion. TDD §11: hard delete es destructivo
# e irreversible. Por eso la ventana de 7d da tiempo al user para Undo.
async def tombstone_purge_job_wrapper(db: Database) -> None:
    """Wrapper async del job diario de purge de tombstoned conversations.

    Llamado por TombstonePurgeScheduler cada 24h. La query es rapida
    (indexada por idx_conversations_deleted_at, parcial sobre filas
    con deleted_at IS NOT NULL). En el peor caso, 1000s de convs
    tombstoned = <100ms. Ejecutar en event loop es seguro.
    """
    try:
        n = await db.purge_expired_conversations()
        logger.info(
            "tombstone_purge_scheduled_ok",
            extra={"purged": n},
        )
    except Exception:
        logger.exception("tombstone_purge_scheduled_failed")


class ConversationCleanupScheduler:
    """Wraps APScheduler for the periodic conversation archive job.

    Sprint 9.4 — previene bug 9.3.2b (UNIQUE constraint block por
    conversaciones huerfanas). Opt-out via settings.cleanup_enabled.
    """

    def __init__(
        self,
        *,
        interval_minutes: int = 5,
        max_age_minutes: int = 60,
        db: Database | None = None,
    ) -> None:
        self._scheduler = AsyncIOScheduler()
        self._interval_minutes = interval_minutes
        self._max_age_minutes = max_age_minutes
        # Inyectado por __main__.run() para evitar import circular
        self._db = db

    async def start(self) -> None:
        if self._db is None:
            logger.warning("conversation_cleanup_scheduler_skipped_no_db")
            return
        self._scheduler.add_job(
            conversation_cleanup_job_wrapper,
            IntervalTrigger(minutes=self._interval_minutes),
            args=(self._db, self._max_age_minutes * 60),
            id="conversation_cleanup_interval",
            name="Archive stale conversations (S9.4, every N min)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=300,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info(
            "conversation_cleanup_scheduler_started",
            extra={
                "interval_minutes": self._interval_minutes,
                "max_age_minutes": self._max_age_minutes,
            },
        )

    async def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)
            logger.info("conversation_cleanup_scheduler_stopped")


class TombstonePurgeScheduler:
    """Sprint 12.1: hard-delete diario de convs tombstoned (past retention).

    TDD_S12_DELETE_AND_SYNC.md §7.4: el job corre cada 24h y purga
    convs con `purge_at <= NOW()`. El CASCADE en `messages` borra los
    rows (cifrados) en la misma operacion. Opt-out: si la DB no esta
    inicializada, el scheduler es no-op (sigue el patron de los demas).
    """

    def __init__(
        self,
        *,
        interval_hours: int = 24,
        db: Database | None = None,
    ) -> None:
        self._scheduler = AsyncIOScheduler()
        self._interval_hours = interval_hours
        # Inyectado por __main__.run() para evitar import circular
        self._db = db

    async def start(self) -> None:
        if self._db is None:
            logger.warning("tombstone_purge_scheduler_skipped_no_db")
            return
        self._scheduler.add_job(
            tombstone_purge_job_wrapper,
            IntervalTrigger(hours=self._interval_hours),
            args=(self._db,),
            id="tombstone_purge_daily",
            name="Purge tombstoned conversations (S12.1, every 24h)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info(
            "tombstone_purge_scheduler_started",
            extra={"interval_hours": self._interval_hours},
        )

    async def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)
            logger.info("tombstone_purge_scheduler_stopped")


# Sprint 17 Slice 1.5 hot-patch (PR #113b, B2 fix).
# VaultScheduler corre los 2 jobs de IngestRouter:
#   1. process_inbox cada N minutos (default 5) — drena done/<job>.json
#      hacia vault.update_text.
#   2. janitor_running_jobs cada N minutos (default 5) — mueve
#      processing/<job> stale (> threshold_s sin touch) → pending/.
# Sin este scheduler, Slice 1.5 GREEN es dead code: los jobs se
# acumulan en done/ y processing/ sin que nadie los procese.
# Opt-out: si el IngestRouter no está inicializado, no-op.
class VaultScheduler:
    """Wraps APScheduler for the Slice 1.5 vault ingest jobs.

    TDD_VAULT_INGEST_WORKER.md §"Deployment" exige ambos jobs. PR #113b
    cierra el gap: este scheduler se registra en `__main__.run()` con
    un IngestRouter ya construido (que a su vez tiene Vault + FsInboxWriter
    + Settings inyectados).

    Frecuencias:
    - process_inbox: cada 5 minutos (default). En el futuro, podría
      acelerarse a 1 min para baja latencia. Coalesce=True absorbe
      múltiples triggers en uno solo (e.g. si Hermes estuvo down 1h).
    - janitor_running_jobs: cada 5 minutos. El threshold real (e.g.
      600s = 10 min) vive en Settings; este scheduler solo dispara.

    Trade-off de max_instances=1: si un run anterior está en curso
    cuando el trigger llega, el nuevo se descarta. Para jobs de
    filesystem esto es OK (son rápidos, <1s con inbox vacío). Si en
    el futuro process_inbox se vuelve lento (10k+ jobs en done/),
    reevaluar.
    """

    def __init__(
        self,
        *,
        interval_minutes: int = 5,
        ingest_router: object | None = None,
    ) -> None:
        self._scheduler = AsyncIOScheduler()
        self._interval_minutes = interval_minutes
        # Inyectado por __main__.run() para evitar import circular
        # (ingest_router importa Settings que importa ...). Typing
        # es `object | None` en runtime; el wrapper type-checks via
        # hasattr().
        self._router = ingest_router

    async def start(self) -> None:
        if self._router is None:
            logger.warning("vault_scheduler_skipped_no_router")
            return
        self._scheduler.add_job(
            _process_inbox_wrapper,
            IntervalTrigger(minutes=self._interval_minutes),
            args=(self._router,),
            id="vault_process_inbox_interval",
            name="Vault: process_inbox (drain done/ → vault, S17 Slice 1.5)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=300,
            coalesce=True,
        )
        self._scheduler.add_job(
            _janitor_wrapper,
            IntervalTrigger(minutes=self._interval_minutes),
            args=(self._router,),
            id="vault_janitor_interval",
            name="Vault: janitor_running_jobs (stale → pending/, S17 Slice 1.5)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=300,
            coalesce=True,
        )
        # Sprint 18 hardening (M6 vacuum): daily soft-vacuum of aged-out
        # 'applied'/'failed' rows. The interval is 1 day (24h) — runs
        # once per day around the same wall-clock time. The actual
        # age threshold (default 30 days) is on the router; this
        # scheduler just fires the trigger. If vacuum is disabled
        # (max_age_days <= 0), the router returns 0 quickly without
        # touching the DB.
        self._scheduler.add_job(
            _vacuum_wrapper,
            IntervalTrigger(hours=24),
            args=(self._router,),
            id="vault_vacuum_daily",
            name="Vault: vacuum_applied_jobs (S18 hardening, M6 soft-vacuum)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info(
            "vault_scheduler_started",
            extra={"interval_minutes": self._interval_minutes},
        )

    async def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)
            logger.info("vault_scheduler_stopped")


async def _process_inbox_wrapper(router: object) -> None:
    """Async wrapper for IngestRouter.process_inbox()."""
    try:
        applied = await router.process_inbox()  # type: ignore[attr-defined]
        logger.debug(
            "vault_process_inbox_scheduled_ok",
            extra={"applied": applied},
        )
    except Exception:
        # El método ya loggea specifics; no duplicar. Pero
        # un loop-runner exception NO debe matar el scheduler —
        # el siguiente trigger lo reintenta.
        logger.exception("vault_process_inbox_scheduled_failed")


async def _janitor_wrapper(router: object) -> None:
    """Async wrapper for IngestRouter.janitor_running_jobs()."""
    try:
        moved = await router.janitor_running_jobs()  # type: ignore[attr-defined]
        logger.debug(
            "vault_janitor_scheduled_ok",
            extra={"moved": moved},
        )
    except Exception:
        logger.exception("vault_janitor_scheduled_failed")


async def _vacuum_wrapper(router: object) -> None:
    """Sprint 18 hardening: async wrapper for IngestRouter.vacuum_applied_jobs().

    Runs daily. Uses settings.vault_done_archive_after_days (default 30).
    No-op if router._db is None (tests with FakeVault) or if max_age_days <= 0.
    """
    try:
        archived = await router.vacuum_applied_jobs()  # type: ignore[attr-defined]
        logger.debug(
            "vault_vacuum_scheduled_ok",
            extra={"archived": archived},
        )
    except Exception:
        # Vacuum failure should NEVER abort the scheduler. The next
        # daily run will retry. Log + move on.
        logger.exception("vault_vacuum_scheduled_failed")


# ----------------------------------------------------------------------------
# EmbedWatcherScheduler — Slice 2.5 GREEN (PR #113c)
# ----------------------------------------------------------------------------
# Poll loop that calls EmbedWatcher.run_once() on a fixed interval.
# Without this scheduler, the embedder is dead code: vault_files get
# text written by IngestRouter, but the embeddings never get
# computed and RAG search returns nothing.
#
# Opt-out: interval_s <= 0 → no-op (embedder is wired but disabled,
# useful for tests and one-off manual runs).
class EmbedWatcherScheduler:
    """Wraps APScheduler for the Slice 2.5 embed watcher.

    PR #113c: this scheduler was missing (B2 finding from
    plan_fc47f6af). Slice 2.5 GREEN without it = dead code: no
    embeddings computed, RAG empty.

    Frecuencia: `interval_s` desde settings.vault_watcher_poll_interval_s
    (default 300s = 5min). El threshold real vive en Settings.

    Trade-off de max_instances=1: si un run anterior está en curso
    cuando el trigger llega, el nuevo se descarta. Para el watcher
    esto es OK (cosine load es rápido, <1s con vault vacío). Si en
    el futuro hay >10K chunks, reevaluar.
    """

    def __init__(
        self,
        *,
        interval_s: int,
        watcher: object | None = None,
    ) -> None:
        self._scheduler = AsyncIOScheduler()
        self._interval_s = interval_s
        # Inyectado por __main__.run() para evitar import circular
        # (embedder importa Settings, etc.). Typing es `object | None`
        # en runtime; el wrapper type-checks via hasattr().
        self._watcher = watcher

    async def start(self) -> None:
        if self._watcher is None:
            logger.warning("embed_watcher_scheduler_skipped_no_watcher")
            return
        if self._interval_s <= 0:
            logger.warning(
                "embed_watcher_scheduler_disabled",
                extra={"interval_s": self._interval_s},
            )
            return
        self._scheduler.add_job(
            _embed_watcher_wrapper,
            IntervalTrigger(seconds=self._interval_s),
            args=(self._watcher,),
            id="embed_watcher_interval",
            name="Vault: EmbedWatcher (find_files_needing_embed → embed, S17 Slice 2.5)",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=60,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info(
            "embed_watcher_scheduler_started",
            extra={"interval_s": self._interval_s},
        )

    async def shutdown(self, *, timeout_s: float = 10.0) -> None:
        """Sprint 18 hardening (Gemini P0 #2): bounded shutdown.

        PR #118: previous version called APScheduler's
        `self._scheduler.shutdown(wait=True)`, which blocks until the
        currently-running job (EmbedWatcher.run_once()) completes
        naturally. If the embedder is mid-LLM call, this can take 60+
        seconds (OpenRouter free tier queue). Docker SIGKILLs the
        container mid-call → loss of tokens + potential DB connection
        poisoning.

        Fix: call `watcher.shutdown(timeout_s)` first (sets stop_event,
        drains in-flight under timeout), THEN shut down the APScheduler
        with `wait=False` (don't block on running jobs — they're either
        done or will be GC'd by Python when the process exits).

        Args:
            timeout_s: max seconds to wait for in-flight embed to drain.
                Default 10.0 matches Docker's default SIGTERM grace.
        """
        if not self._scheduler.running:
            return
        # Step 1: signal the watcher to stop + wait for in-flight to
        # drain under timeout. If the watcher's shutdown returns False
        # (timeout), we still proceed to shut down the scheduler —
        # the orphaned task will be GC'd on process exit.
        if self._watcher is not None and hasattr(self._watcher, "shutdown"):
            try:
                drained = await self._watcher.shutdown(timeout_s=timeout_s)
                if not drained:
                    logger.warning(
                        "embed_watcher_scheduler_shutdown_timeout",
                        extra={"timeout_s": timeout_s},
                    )
            except Exception:
                logger.exception("embed_watcher_scheduler_watcher_shutdown_error")
        # Step 2: shut down APScheduler with wait=False so we don't
        # block on any other running job (there shouldn't be one since
        # max_instances=1 + the watcher already bailed between files).
        self._scheduler.shutdown(wait=False)
        logger.info("embed_watcher_scheduler_stopped")


async def _embed_watcher_wrapper(watcher: object) -> None:
    """Async wrapper for EmbedWatcher.run_once()."""
    try:
        embedded = await watcher.run_once()  # type: ignore[attr-defined]
        logger.debug(
            "embed_watcher_scheduled_ok",
            extra={"embedded": embedded},
        )
    except Exception:
        # El método ya loggea specifics; no duplicar. Pero
        # un loop-runner exception NO debe matar el scheduler —
        # el siguiente trigger lo reintenta.
        logger.exception("embed_watcher_scheduled_failed")


# ---------------------------------------------------------------------------
# Sprint 19 Slice 4c: M6 Phase 5 zombie edge job recovery
# ---------------------------------------------------------------------------


class EdgeZombieScheduler:
    """Periodic M6 Phase 5 cleanup of stuck `edge_queued` rows.

    TDD §4.4.5: rows in `ocr_pending.status='edge_queued'` older than
    `timeout_hours` are reverted to `pending_review` and their orphan
    `request.json` in the edge queue is removed. Handles the case
    where a PC crashed mid-processing and never wrote a response.

    Frequency: `interval_s` from `settings.ocr_edge_zombie_scan_interval`
    (default 900s = 15min). Cheap partial-index SELECT, no perf concern.

    The actual recovery logic lives in
    `EdgeCoordinator.recover_zombies(timeout_hours)` — this scheduler
    is just the periodic trigger.
    """

    def __init__(
        self,
        *,
        interval_s: int,
        coordinator: object | None = None,
        timeout_hours: int = 2,
    ) -> None:
        self._scheduler = AsyncIOScheduler()
        self._interval_s = interval_s
        self._coordinator = coordinator
        self._timeout_hours = timeout_hours

    async def start(self) -> None:
        if self._coordinator is None:
            logger.warning("edge_zombie_scheduler_skipped_no_coordinator")
            return
        if self._interval_s <= 0:
            logger.warning(
                "edge_zombie_scheduler_disabled",
                extra={"interval_s": self._interval_s},
            )
            return
        self._scheduler.add_job(
            _edge_zombie_wrapper,
            IntervalTrigger(seconds=self._interval_s),
            args=(self._coordinator, self._timeout_hours),
            id="edge_zombie_recovery",
            name="M6 Phase 5: revert stuck edge_queued rows to pending_review",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=60,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info(
            "edge_zombie_scheduler_started",
            extra={
                "interval_s": self._interval_s,
                "timeout_hours": self._timeout_hours,
            },
        )

    async def shutdown(self, *, timeout_s: float = 10.0) -> None:
        if not self._scheduler.running:
            return
        self._scheduler.shutdown(wait=False)
        logger.info("edge_zombie_scheduler_stopped")


async def _edge_zombie_wrapper(coordinator: object, timeout_hours: int) -> None:
    """Async wrapper for EdgeCoordinator.recover_zombies()."""
    try:
        recovered = await coordinator.recover_zombies(  # type: ignore[attr-defined]
            timeout_hours=timeout_hours,
        )
        if recovered > 0:
            logger.info(
                "edge_zombie_recovered_batch",
                extra={"recovered": recovered, "timeout_hours": timeout_hours},
            )
    except Exception:
        # A loop-runner exception must NOT kill the scheduler.
        logger.exception("edge_zombie_scheduled_failed")


# ---------------------------------------------------------------------------
# Sprint 19 Slice 4d v2 (TDD_VAULT_COLLECTIONS_v0.5 §5): M6ReconcileScheduler
# Reconcile vault_files with filesystem (monitor roots) + re-queue dropped
# events. Distinct from the watcher's inline processing (v0.6 §9).
# ---------------------------------------------------------------------------


class M6ReconcileScheduler:
    """Sprint 19 Slice 4d v2: periodic M6 reconciliation for monitor roots.

    Three responsibilities per v0.5 §5 + v0.6 §9:
    1. Re-queue dropped events from the dropped_events table (created by
       the watcher's asyncio.Queue when full). M6 picks them up at the
       start of each cycle, calls drop_watcher.process_path(), marks
       processed_at = NOW() on success.
    2. Detect orphans: files in vault_files that are no longer at
       their source_path. Set orphaned_at = NOW() on the row.
    3. (FUTURE) Scan monitor roots for new files NOT in vault_files.

    The watcher (v0.6 §9) is authoritative for new file + move detection
    (inline content_sha256 SELECT). M6 is "cleanup after watcher" - does
    NOT do move detection. Watcher and M6 share db._write_lock.

    Pattern follows all 6 other schedulers in this file: start()/shutdown()
    uses AsyncIOScheduler + IntervalTrigger with max_instances=1, coalesce=True.
    """

    def __init__(
        self,
        *,
        db,
        drop_watcher,
        monitor_roots: list,
        interval_s: int = 300,
    ) -> None:
        self._scheduler = AsyncIOScheduler()
        self._db = db
        self._drop_watcher = drop_watcher
        self._monitor_roots = monitor_roots
        self._interval_s = interval_s
        # R1 #3 m4.4 fix: re-validate at scheduler init. Raises
        # ValueError on VAULT_INBOX_ROOT overlap (forbidden config).
        self._validate_partition()

    def _validate_partition(self) -> None:
        """R1 #3 m4.4: re-validate VAULT_MONITOR_ROOTS vs VAULT_INBOX_ROOT.

        Fix (R1 v0.6 M3): DropWatcher has `_drop_root`, NOT `_inbox_root`.
        The partition rule is: monitor_roots must NOT be under drop_root
        (which itself is under VAULT_INBOX_ROOT). Same rule is also
        enforced in config._get_monitor_roots, this is defense-in-depth.
        """
        drop_root = getattr(self._drop_watcher, "_drop_root", None)
        if drop_root is None:
            return
        drop_resolved = drop_root.resolve()
        for root in self._monitor_roots:
            try:
                root.resolve().relative_to(drop_resolved)
                raise ValueError(
                    f"M6ReconcileScheduler: {root} is under drop_root "
                    f"({drop_resolved}). M6 partition conflict."
                )
            except ValueError as e:
                if "is under" in str(e):
                    raise
                continue

    async def start(self) -> None:
        """Wire run_once() into AsyncIOScheduler as interval job."""
        self._scheduler.add_job(
            self._run_once_wrapper,
            IntervalTrigger(seconds=self._interval_s),
            id="m6_reconcile_interval",
            name="M6 Vault Reconciliation (v0.2 monitor roots)",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )
        self._scheduler.start()
        logger.info(
            "m6_reconcile_scheduler_started",
            extra={
                "interval_s": self._interval_s,
                "monitor_roots_count": len(self._monitor_roots),
            },
        )

    async def shutdown(self, *, timeout_s: float = 10.0) -> None:
        if not self._scheduler.running:
            return
        self._scheduler.shutdown(wait=False)
        logger.info("m6_reconcile_scheduler_stopped")

    async def _run_once_wrapper(self) -> None:
        """Async wrapper for the scheduled job."""
        import time

        start = time.monotonic()
        try:
            processed = await self.run_once()
            logger.info(
                "m6_reconcile_cycle_done",
                extra={
                    "processed": processed,
                    "elapsed_ms": int((time.monotonic() - start) * 1000),
                },
            )
        except Exception:
            logger.exception("m6_reconcile_cycle_failed")

    async def run_once(self) -> int:
        """M6 reconciliation cycle. Returns count of files processed."""
        processed = 0
        processed += await self._requeue_dropped_events()
        processed += await self._detect_orphans()
        return processed

    async def _requeue_dropped_events(self) -> int:
        """Re-queue events from dropped_events table."""
        rows = await self._db.conn.execute_fetchall(
            "SELECT event_id, source_path FROM dropped_events "
            "WHERE processed_at IS NULL "
            "ORDER BY detected_at LIMIT 1000"
        )
        processed = 0
        import datetime

        for event_id, source_path in rows:
            from pathlib import Path

            path = Path(source_path)
            if not path.exists():
                now_iso = datetime.datetime.now(datetime.UTC).isoformat()
                await self._db.conn.execute(
                    "UPDATE dropped_events SET processed_at = ? WHERE event_id = ?",
                    (now_iso, event_id),
                )
                continue
            try:
                await self._drop_watcher.process_path(path)
                now_iso = datetime.datetime.now(datetime.UTC).isoformat()
                await self._db.conn.execute(
                    "UPDATE dropped_events SET processed_at = ? WHERE event_id = ?",
                    (now_iso, event_id),
                )
                processed += 1
            except Exception:
                logger.exception(
                    "m6_requeue_dropped_event_failed",
                    extra={"path": source_path, "event_id": event_id},
                )
        return processed

    async def _detect_orphans(self) -> int:
        """Scan vault_files for rows where the physical file is gone."""
        from pathlib import Path

        rows = await self._db.conn.execute_fetchall(
            "SELECT file_id, source_path FROM vault_files WHERE orphaned_at IS NULL"
        )
        processed = 0
        import datetime

        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        for file_id, source_path in rows:
            path = Path(source_path)
            if not path.exists():
                await self._db.conn.execute(
                    "UPDATE vault_files SET orphaned_at = ? WHERE file_id = ?",
                    (now_iso, file_id),
                )
                processed += 1
        return processed
