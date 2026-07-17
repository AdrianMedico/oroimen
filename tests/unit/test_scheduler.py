"""Tests for BackupScheduler (S8.4 — regression thread-off-loop).

Cubre:
1. test_scheduler_uses_to_thread_off_event_loop: REGRESSION Gemini 2026-06-24
   - Mockea asyncio.to_thread, llama backup_job_wrapper
   - Verifica que se llamó con backup_db_main (NO ejecución directa en event loop)
2. test_scheduler_handles_backup_failure: si backup_db_main lanza, el wrapper no crashea
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from hermes import scheduler as sched_mod


async def test_scheduler_uses_to_thread_off_event_loop() -> None:
    """REGRESSION (Gemini 2026-06-24): backup DEBE correr en thread del pool,
    NO en el event loop. Sin esto, sqlite3.backup() bloquea FastAPI.
    """
    called_with: list[object] = []

    async def fake_to_thread(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        called_with.append(fn)
        return Path("/tmp/fake_backup.db")

    with patch.object(sched_mod.asyncio, "to_thread", side_effect=fake_to_thread):
        await sched_mod.backup_job_wrapper()

    assert len(called_with) == 1, "to_thread debe ser llamado exactamente 1 vez"
    assert (
        called_with[0] is sched_mod.backup_db_main
    ), "to_thread debe recibir backup_db_main (no ejecución directa)"


async def test_scheduler_handles_backup_failure() -> None:
    """Si backup_db_main lanza (ej: DB no existe), el wrapper NO propaga la excepción.
    El scheduler debe sobrevivir al fallo y continuar con la próxima ejecución.
    """

    async def fake_to_thread_raises(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Simulated DB failure")

    with patch.object(sched_mod.asyncio, "to_thread", side_effect=fake_to_thread_raises):
        # NO debe lanzar
        await sched_mod.backup_job_wrapper()


# Sprint 9.4: ConversationCleanupScheduler (archive stale conversations)
# -----------------------------------------------------------------------
# Cubre:
# 1. test_conversation_cleanup_wrapper_calls_db: el wrapper invoca
#    db.archive_stale_conversations con los args correctos
# 2. test_conversation_cleanup_wrapper_handles_db_failure: si el DB
#    falla, el wrapper NO propaga (scheduler debe sobrevivir)


async def test_conversation_cleanup_wrapper_calls_db() -> None:
    """El wrapper llama db.archive_stale_conversations(max_age_seconds=...)."""
    from unittest.mock import AsyncMock

    fake_db = AsyncMock()
    fake_db.archive_stale_conversations.return_value = 5
    await sched_mod.conversation_cleanup_job_wrapper(fake_db, max_age_seconds=3600)
    fake_db.archive_stale_conversations.assert_awaited_once_with(max_age_seconds=3600)


async def test_conversation_cleanup_wrapper_handles_db_failure() -> None:
    """Si db.archive_stale lanza, el wrapper NO propaga la excepcion.
    El scheduler debe sobrevivir al fallo y continuar.
    """
    from unittest.mock import AsyncMock

    fake_db = AsyncMock()
    fake_db.archive_stale_conversations.side_effect = RuntimeError("DB locked")
    # NO debe lanzar
    await sched_mod.conversation_cleanup_job_wrapper(fake_db, max_age_seconds=3600)


async def test_conversation_cleanup_scheduler_skips_without_db() -> None:
    """Si db=None, el scheduler arranca pero no añade jobs (skip log)."""
    from hermes.scheduler import ConversationCleanupScheduler

    sched = ConversationCleanupScheduler(interval_minutes=5, max_age_minutes=60, db=None)
    await sched.start()
    # Scheduler no tiene jobs
    assert len(sched._scheduler.get_jobs()) == 0
    await sched.shutdown()
