"""Sprint 19 Slice 4d v2 commit 5: M6ReconcileScheduler tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    pass


async def test_m6_reconcile_starts_and_stops(
    tmp_path,
) -> None:
    """start() registers the job, shutdown() calls apscheduler.shutdown."""
    from hermes.scheduler import M6ReconcileScheduler

    db = MagicMock()
    db.conn = AsyncMock()
    drop_watcher = MagicMock()
    drop_watcher._inbox_root = None  # skip validation

    scheduler = M6ReconcileScheduler(
        db=db,
        drop_watcher=drop_watcher,
        monitor_roots=[tmp_path / "monitor1"],
        interval_s=60,
    )
    await scheduler.start()
    # Job registered
    assert len(scheduler._scheduler.get_jobs()) == 1
    assert scheduler._scheduler.get_job("m6_reconcile_interval") is not None

    # shutdown() doesn't raise (apscheduler handles the cleanup async)
    await scheduler.shutdown()


async def test_m6_reconcile_partition_validation(
    tmp_path,
) -> None:
    """_validate_partition raises ValueError on VAULT_INBOX_ROOT overlap.

    R1 #3 m4.4: monitor root under inbox root is FORBIDDEN (M6
    partition conflict).
    """
    from hermes.scheduler import M6ReconcileScheduler

    db = MagicMock()
    drop_watcher = MagicMock()
    drop_root = tmp_path / "drop"
    drop_root.mkdir()
    # Monitor root is UNDER drop_root (forbidden)
    monitor_root = drop_root / "subfolder"

    # R1 v0.6 M3 fix: the partition rule uses _drop_root, not _inbox_root
    drop_watcher._drop_root = drop_root

    with pytest.raises(ValueError, match="is under drop_root"):
        M6ReconcileScheduler(
            db=db,
            drop_watcher=drop_watcher,
            monitor_roots=[monitor_root],
            interval_s=60,
        )


async def test_m6_reconcile_partition_validation_passes(
    tmp_path,
) -> None:
    """_validate_partition passes when monitor root is NOT under inbox."""
    from hermes.scheduler import M6ReconcileScheduler

    db = MagicMock()
    drop_watcher = MagicMock()
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    monitor_root = tmp_path / "other"  # NOT under inbox

    drop_watcher._inbox_root = inbox_root

    # Should not raise
    M6ReconcileScheduler(
        db=db,
        drop_watcher=drop_watcher,
        monitor_roots=[monitor_root],
        interval_s=60,
    )


async def test_m6_reconcile_run_once_no_work(
    tmp_path,
) -> None:
    """run_once() returns 0 when there's nothing to do."""
    from hermes.scheduler import M6ReconcileScheduler

    db = MagicMock()
    db.conn.execute_fetchall = AsyncMock(return_value=[])
    drop_watcher = MagicMock()
    drop_watcher._inbox_root = None

    scheduler = M6ReconcileScheduler(
        db=db,
        drop_watcher=drop_watcher,
        monitor_roots=[],
        interval_s=60,
    )
    processed = await scheduler.run_once()
    assert processed == 0


async def test_m6_reconcile_requeues_dropped_events(
    tmp_path,
) -> None:
    """_requeue_dropped_events calls process_path on existing files."""
    from hermes.scheduler import M6ReconcileScheduler

    # Create a real file to be "dropped"
    real_file = tmp_path / "foo.pdf"
    real_file.write_bytes(b"hello")

    db = MagicMock()
    # First SELECT returns 1 dropped event
    # UPDATE statements return mock cursors
    update_cursor = MagicMock()
    update_cursor.rowcount = 1
    db.conn.execute_fetchall = AsyncMock(return_value=[(1, str(real_file))])
    db.conn.execute = AsyncMock(return_value=update_cursor)

    drop_watcher = MagicMock()
    drop_watcher.process_path = AsyncMock()
    drop_watcher._inbox_root = None

    scheduler = M6ReconcileScheduler(
        db=db,
        drop_watcher=drop_watcher,
        monitor_roots=[],
        interval_s=60,
    )
    processed = await scheduler._requeue_dropped_events()
    assert processed == 1
    drop_watcher.process_path.assert_called_once_with(real_file)


async def test_m6_reconcile_marks_gone_file_as_processed(
    tmp_path,
) -> None:
    """_requeue_dropped_events: if file no longer exists, mark processed (no-op)."""
    from hermes.scheduler import M6ReconcileScheduler

    # File that doesn't exist
    missing_file = tmp_path / "missing.pdf"

    db = MagicMock()
    update_cursor = MagicMock()
    update_cursor.rowcount = 0
    db.conn.execute_fetchall = AsyncMock(return_value=[(1, str(missing_file))])
    db.conn.execute = AsyncMock(return_value=update_cursor)

    drop_watcher = MagicMock()
    drop_watcher.process_path = AsyncMock()
    drop_watcher._inbox_root = None

    scheduler = M6ReconcileScheduler(
        db=db,
        drop_watcher=drop_watcher,
        monitor_roots=[],
        interval_s=60,
    )
    processed = await scheduler._requeue_dropped_events()
    assert processed == 0
    # process_path should NOT be called (file is gone)
    drop_watcher.process_path.assert_not_called()


async def test_m6_reconcile_detects_orphans(
    tmp_path,
) -> None:
    """_detect_orphans sets orphaned_at on files that no longer exist."""
    from hermes.scheduler import M6ReconcileScheduler

    # Create a vault_files row for a file that doesn't exist
    missing_file = tmp_path / "gone.pdf"

    db = MagicMock()
    update_cursor = MagicMock()
    db.conn.execute_fetchall = AsyncMock(return_value=[("file_id_1", str(missing_file))])
    db.conn.execute = AsyncMock(return_value=update_cursor)

    drop_watcher = MagicMock()
    drop_watcher._inbox_root = None

    scheduler = M6ReconcileScheduler(
        db=db,
        drop_watcher=drop_watcher,
        monitor_roots=[],
        interval_s=60,
    )
    processed = await scheduler._detect_orphans()
    assert processed == 1
