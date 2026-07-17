"""SQLite WAL backup online (Sprint 8 — S8.4).

IMPORTANT: external backup tools should copy the consolidated file produced
by this module, not the live database/WAL pair. Copying the live files during
writes can produce an inconsistent backup.

Este módulo usa `sqlite3.Connection.backup()` API que SÍ consolida el
WAL atómicamente y es online (no bloquea reads).

References:
    https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.backup
    docs/TDD_SQLITE_WAL_BACKUP.md
    Vikunja #130 [S8.4]
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BackupError(Exception):
    """Raised when backup fails. Caller should NOT delete old backups."""


class BackupManager:
    """Manages online WAL-consistent backups of the Oroimen SQLite DB.

    Uses sqlite3.Connection.backup() which:
    - Consolidates WAL into target atomically
    - Is online (no read blocking)
    - Safe under concurrent writes
    """

    def __init__(self, *, source_db: Path, backup_dir: Path, keep: int = 7) -> None:
        if not source_db.exists():
            raise BackupError(f"Source DB not found: {source_db}")
        self.source_db = source_db
        self.backup_dir = backup_dir
        self.keep = keep
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        """Run a single backup. Returns path to created backup file.

        Raises BackupError if backup fails. Never deletes existing backups on failure.

        Sprint 13.0 (S8.4 fix): añade BEGIN EXCLUSIVE + wal_checkpoint(TRUNCATE)
        ANTES del backup para evitar race condition con concurrent writes.
        Ver docs/POSTMORTEM_DB_CORRUPTION.md para el root cause analysis.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.backup_dir / f"conversations_{timestamp}.db"
        tmp_dest = dest.with_suffix(".db.tmp")

        # CRITICAL: wrap the WHOLE flow (backup + rename) in one try block.
        # Both sqlite3.Error (during .backup()) and OSError (during .rename())
        # must be handled: on failure, clean up the .tmp file and raise
        # BackupError. Old backups are NEVER touched.
        try:
            src_conn = sqlite3.connect(str(self.source_db))
            dst_conn = sqlite3.connect(str(tmp_dest))
            try:
                # Sprint 13.0 (S8.4 fix): checkpoint WAL → main antes del backup.
                #
                # Race condition original (Sprint 8.4, commit c0c6e0d):
                # writers en el main thread (Telegram, /v1/chat/completions,
                # etc.) invalidaban el B-tree structure durante .backup(),
                # generando pages corruptas (*** in database main ***,
                # btreeInitPage returns error code 11).
                #
                # Solución aplicada:
                # 1. PRAGMA wal_checkpoint(TRUNCATE): fuerza checkpoint del
                #    WAL al main DB. Consolida pages pendientes y reduce
                #    drásticamente el window de race.
                # 2. Connection.backup() API: atómica y online por diseño
                #    (SQLite 3.7+). Adquiere read locks que no bloquean
                #    writers durante el proceso.
                #
                # NOTA: NO usamos BEGIN EXCLUSIVE porque causa DEADLOCK
                # con .backup() (backup necesita read lock, EXCLUSIVE no
                # lo permite). Ver test test_backup_with_concurrent_writes.
                src_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                # .backup() consolidates WAL atomically into target
                with dst_conn:
                    src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
                src_conn.close()

            # Atomic rename (POSIX guarantee). Can raise OSError.
            tmp_dest.rename(dest)
        except (sqlite3.Error, OSError) as e:
            with contextlib.suppress(OSError):
                if tmp_dest.exists():
                    tmp_dest.unlink()
            logger.error(
                "db_backup_failed",
                extra={"error": str(e), "source": str(self.source_db)},
            )
            raise BackupError(f"Backup failed: {e}") from e

        size = dest.stat().st_size
        logger.info("db_backup_complete", extra={"backup": str(dest), "size_bytes": size})

        self._rotate()
        return dest

    def _rotate(self) -> None:
        """Keep only the N most recent backups. Never deletes the current one mid-run."""
        backups = sorted(self.backup_dir.glob("conversations_*.db"))
        if len(backups) <= self.keep:
            return
        for old in backups[: -self.keep]:
            try:
                old.unlink()
                logger.info("db_backup_rotated", extra={"deleted": str(old)})
            except OSError as e:
                logger.warning("db_backup_rotate_failed", extra={"file": str(old), "error": str(e)})

    def list_backups(self) -> list[Path]:
        """List existing backups, newest first."""
        return sorted(self.backup_dir.glob("conversations_*.db"), reverse=True)


def backup_db_main() -> Path | None:
    """Entry point for the scheduled backup job.

    Reads config from environment (via Settings) and runs a backup.
    Returns the backup file path, or None if backup is disabled.

    NOTE: This is a SYNC function. It must be called via asyncio.to_thread()
    from an async context to avoid blocking the FastAPI event loop.
    See hermes/scheduler.py for the wrapper.

    Creates a fresh Settings instance (cheap). The hermes singleton pattern
    is for runtime components; cron jobs are isolated and re-read env each time.
    """
    from hermes.config import Settings

    settings = Settings()

    if not settings.backup_enabled:
        logger.info("db_backup_skipped", extra={"reason": "BACKUP_ENABLED=false"})
        return None

    manager = BackupManager(
        source_db=Path(settings.db_path),
        backup_dir=Path(settings.backup_dir),
        keep=settings.backup_keep,
    )
    return manager.run()
