"""Recovery hook para reconciliar jobs tras container restart.

Ver TDD_S14_DEEP_RESEARCH.md §7.

Llamado desde hermes/__main__.py:startup() ANTES de
DeepResearchScheduler.start(). Recorre 5 casos (TDD §7.1) que pueden
dejar la DB en estado inconsistente si el container fue killed mid-flight:

  1. status='pending' sin started_at y created_at > recovery_drop_orphan_hours
     → DROP (orphan, nadie lo recogió)
  2. status='running' y started_at > recovery_running_stuck_hours
     → RESET a 'pending', re-enqueue (probable kill mid-flight)
  3. status='running' y output_path IS NOT NULL
     → marca 'complete' (write OK pero DB update no llegó)
  4. status='complete'|'failed' con notified=0
     → re-envía notif (push no enviado pre-kill)
  5. status='cancelling'
     → marca 'cancelled' (cancel request perdido)

Memory-safety (Q2 round Gemini): cada query tiene LIMIT 100 para evitar
OOM spike al startup si hay cientos de jobs en estado inconsistente.
Si una query devuelve 100, procesamos esos 100 y la próxima invocación
de recovery (manual o tras próximo restart) coge los siguientes. NO
paginamos en una sola invocación porque el objetivo es recover rápido
para que el container quede operativo cuanto antes.

Trade-off explícito caso 4: 100 LIMIT implica que tras un crash masivo,
los primeros 100 users reciben re-notif y los siguientes 100+ no. Esto
es aceptable — la notif es nice-to-have, no critical state.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from hermes.jobs.cost import format_now, format_now_at

logger = logging.getLogger(__name__)


async def recover_research_jobs(
    db: Any,
    notifier: Any,
    settings: Any,
    scheduler: Any | None = None,
) -> int:
    """Reconcilia jobs en estado inconsistente tras container restart.

    Args:
        db: Database instance (duck-typed; uso `db.conn`).
        notifier: TelegramNotifier instance (duck-typed; uso `send_research_*`).
        settings: Settings instance (deep_research_recovery_* fields).
        scheduler: DeepResearchScheduler opcional (usado en caso 2 re-enqueue).

    Returns:
        # de jobs reconciliados (suma de los 5 casos).
    """
    recovered = 0
    now = format_now()

    # === Caso 1: orphans (pending sin started_at, viejos) ===
    try:
        orphan_hours = int(getattr(settings, "deep_research_recovery_drop_orphan_hours", 168))
    except (AttributeError, TypeError):
        orphan_hours = 168
    cutoff_orphan = datetime.now(UTC) - timedelta(hours=orphan_hours)
    cutoff_orphan_str = format_now_at(cutoff_orphan)

    orphans = []
    try:
        orphans = await db.list_research_jobs_pending_created_before(
            cutoff_str=cutoff_orphan_str, limit=100
        )
    except Exception:
        logger.exception("recovery_orphan_query_failed")

    for job in orphans:
        job_id = job["id"]
        try:
            await db.update_research_job_status(
                job_id,
                "failed",
                error_taxonomy="timeout",
                error_message="orphan_recovered",
                completed_at=now,
            )
            logger.warning(
                "recovery_orphan_dropped",
                extra={"job_id": job_id, "age_hours": orphan_hours},
            )
            recovered += 1
        except Exception:
            logger.exception("recovery_orphan_failed", extra={"job_id": job_id})

    # === Caso 2: running colgado (started_at viejo, sin output_path) ===
    try:
        stuck_hours = int(getattr(settings, "deep_research_recovery_running_stuck_hours", 2))
    except (AttributeError, TypeError):
        stuck_hours = 2
    cutoff_stuck = datetime.now(UTC) - timedelta(hours=stuck_hours)
    cutoff_stuck_str = format_now_at(cutoff_stuck)

    stuck = []
    try:
        stuck = await db.list_research_jobs_by_status_started_before(
            status="running",
            cutoff_str=cutoff_stuck_str,
            limit=100,
        )
    except Exception:
        logger.exception("recovery_stuck_query_failed")
    # Caso 2 solo aplica a running SIN output (caso 3 cubre los con output)
    for job in stuck:
        if job.get("output_path"):
            continue
        job_id = job["id"]
        try:
            await db.update_research_job_status(
                job_id,
                "pending",
                current_phase=None,
                progress_percent=0,
            )
            # Re-enqueue via scheduler si está disponible
            if scheduler is not None:
                try:
                    await scheduler.enqueue(job_id, run_date=datetime.now(UTC))
                except Exception:
                    logger.exception("recovery_reenqueue_failed", extra={"job_id": job_id})
            logger.warning(
                "recovery_running_restart",
                extra={"job_id": job_id, "was_phase": job.get("current_phase")},
            )
            recovered += 1
        except Exception:
            logger.exception("recovery_running_failed", extra={"job_id": job_id})

    # === Caso 3: complete con output pero status running ===
    almost_done = []
    try:
        almost_done = await db.list_research_jobs_running_with_output(limit=100)
    except Exception:
        logger.exception("recovery_almost_done_query_failed")
    for job in almost_done:
        job_id = job["id"]
        try:
            await db.update_research_job_status(
                job_id,
                "complete",
                progress_percent=100,
                completed_at=now,
            )
            logger.info(
                "recovery_complete_from_running",
                extra={"job_id": job_id},
            )
            recovered += 1
        except Exception:
            logger.exception("recovery_complete_failed", extra={"job_id": job_id})

    # === Caso 4: complete/failed sin notificar ===
    unnotified = []
    try:
        unnotified = await db.list_research_jobs_unnotified(limit=100)
    except Exception:
        logger.exception("recovery_unnotified_query_failed")
    for job in unnotified:
        job_id = job["id"]
        status_value = job["status"]
        output_path = job.get("output_path")
        cost_usd = float(job.get("cost_usd") or 0.0)
        try:
            if not output_path:
                # Failed sin output — mensaje más simple
                if hasattr(notifier, "send_research_failed"):
                    await notifier.send_research_failed(
                        job_id=job_id,
                        error_taxonomy="failed",
                        error_message="recovered_after_restart",
                        retryable=False,
                    )
            else:
                if hasattr(notifier, "send_research_complete"):
                    # Slice 1C2: signature is now (job_id, cost_usd) — no
                    # output_path. The Telegram template is the redacted
                    # form. The output_path column is still READ from
                    # the row (it's the skip-if-output marker) but it
                    # is NEVER passed to the notifier.
                    await notifier.send_research_complete(
                        job_id=job_id,
                        cost_usd=cost_usd,
                    )
            await db.mark_research_job_notified(job_id)
            logger.info(
                "recovery_notif_resent",
                extra={"job_id": job_id, "status": status_value},
            )
            recovered += 1
        except Exception:
            logger.exception("recovery_notif_failed", extra={"job_id": job_id})

    # === Caso 5: cancelling huérfano ===
    cancelling = []
    try:
        cancelling = await db.list_research_jobs_cancelling(limit=100)
    except Exception:
        logger.exception("recovery_cancelling_query_failed")
    for job in cancelling:
        job_id = job["id"]
        try:
            await db.update_research_job_status(
                job_id,
                "cancelled",
                completed_at=now,
            )
            logger.info("recovery_cancelling_finalized", extra={"job_id": job_id})
            recovered += 1
        except Exception:
            logger.exception("recovery_cancelling_failed", extra={"job_id": job_id})

    # === Caso 6: pending-without-scheduler-entry (recoverable enqueue gap) ===
    #
    # The pickle-error bug left rows in ``pending`` with no
    # scheduler entry. After the fix, fresh enqueues persist the
    # row and the scheduler entry in the same atomic-like step, so
    # a row in ``pending + no scheduler entry`` indicates a real
    # enqueue gap (a crash between DB create and scheduler
    # add_job, or a serialisation failure that we did not
    # compensate correctly).
    #
    # To avoid racing with an in-flight healthy enqueue, we
    # require a small grace period (default 60 seconds) since
    # the row was last updated. A row that is in ``pending`` and
    # has no scheduler entry AND was last updated more than
    # ``recovery_pending_gap_grace_seconds`` ago is a recoverable
    # gap and is re-enqueued.
    #
    # The 2 aborted jobs (19d4ed181ac3, 18014f2cc834) were
    # intentionally terminalized to ``failed`` in a previous
    # mission; they are not in ``pending`` and are therefore not
    # touched by this case.
    try:
        gap_grace_s = int(
            getattr(
                settings,
                "deep_research_recovery_pending_gap_grace_seconds",
                60,
            )
        )
    except (AttributeError, TypeError):
        gap_grace_s = 60
    gap_cutoff = datetime.now(UTC) - timedelta(seconds=gap_grace_s)
    gap_cutoff_str = format_now_at(gap_cutoff)
    gap_candidates = []
    try:
        gap_candidates = await db.list_research_jobs_pending_updated_before(
            cutoff_str=gap_cutoff_str, limit=100
        )
    except Exception:
        logger.exception("recovery_pending_gap_query_failed")
    for job in gap_candidates:
        job_id = job["id"]
        # Belt-and-braces: skip terminal states (the query should
        # already filter to pending, but the schema is the source
        # of truth).
        if (job.get("status") or "").lower() != "pending":
            continue
        # If the scheduler still has the entry, leave it alone:
        # this is a healthy queued job, not a gap.
        if scheduler is not None:
            try:
                if scheduler.get_job(job_id) is not None:
                    logger.info(
                        "recovery_pending_gap_skipped_healthy",
                        extra={"job_id": job_id},
                    )
                    continue
            except Exception:
                logger.exception(
                    "recovery_pending_gap_lookup_failed",
                    extra={"job_id": job_id},
                )
                # If we cannot determine the scheduler state,
                # be conservative: do NOT re-enqueue (avoid
                # duplicates if the entry actually exists).
                continue
        # No scheduler entry → re-enqueue. The persisted callable
        # is now the module-level dispatcher, so add_job will
        # not fail on pickling.
        if scheduler is not None:
            try:
                await scheduler.enqueue(
                    job_id, run_date=datetime.now(UTC)
                )
                logger.warning(
                    "recovery_pending_gap_reenqueued",
                    extra={"job_id": job_id, "grace_seconds": gap_grace_s},
                )
                recovered += 1
            except Exception:
                logger.exception(
                    "recovery_pending_gap_reenqueue_failed",
                    extra={"job_id": job_id},
                )

    if recovered:
        logger.info("recovery_summary", extra={"total_recovered": recovered})
    return recovered
