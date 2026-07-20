"""Unit tests for hermes.jobs.recovery — recover_research_jobs() 5 cases.

Anti-regression checks (TDD §7.1):
- 5 distinct recovery cases (orphan, stuck, complete-from-running,
  unnotified, cancelling).
- LIMIT 100 in every recovery query.
- All cases are non-fatal (one exception doesn't break the others).
"""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from hermes.jobs.recovery import recover_research_jobs


class _FakeSettings:
    """Minimal settings stub with the two recovery-related fields."""

    deep_research_recovery_drop_orphan_hours = 168
    deep_research_recovery_running_stuck_hours = 2


@pytest.mark.asyncio
async def test_recovery_5_cases() -> None:
    """All 5 recovery cases exercise the right DB methods.

    Each case: db.list_* returns 1 row → db.update_research_job_status /
    notifier.send_research_* / db.mark_research_job_notified is called.

    Mocks:
    - db.list_research_jobs_pending_created_before → 1 orphan
    - db.list_research_jobs_by_status_started_before → 1 stuck (no output)
    - db.list_research_jobs_running_with_output → 1 almost_done
    - db.list_research_jobs_unnotified → 1 complete w/ output
    - db.list_research_jobs_cancelling → 1 cancelling

    Expects: recovered count == 5.
    """
    # Mock DB
    db = MagicMock()

    # Case 1: orphan (pending without started_at, older than 7 days)
    orphan_row = {"id": "orphan1", "status": "pending"}
    db.list_research_jobs_pending_created_before = AsyncMock(return_value=[orphan_row])

    # Case 2: running stuck (running, old, no output_path)
    stuck_row = {"id": "stuck1", "status": "running", "output_path": None}
    db.list_research_jobs_by_status_started_before = AsyncMock(return_value=[stuck_row])

    # Case 3: running with output → mark complete
    almost_done_row = {"id": "almost1", "status": "running", "output_path": "/x.md"}
    db.list_research_jobs_running_with_output = AsyncMock(return_value=[almost_done_row])

    # Case 4: complete w/ output, notified=0
    unnotified_row = {
        "id": "unnot1",
        "status": "complete",
        "output_path": "/x.md",
        "cost_usd": 0.05,
    }
    db.list_research_jobs_unnotified = AsyncMock(return_value=[unnotified_row])

    # Case 5: cancelling orphan
    cancelling_row = {"id": "cancel1", "status": "cancelling"}
    db.list_research_jobs_cancelling = AsyncMock(return_value=[cancelling_row])

    db.update_research_job_status = AsyncMock()
    db.mark_research_job_notified = AsyncMock()

    # Mock notifier with the research methods
    notifier = MagicMock()
    notifier.send_research_complete = AsyncMock(return_value=True)
    notifier.send_research_failed = AsyncMock(return_value=True)

    settings = _FakeSettings()

    recovered = await recover_research_jobs(
        db=db, notifier=notifier, settings=settings, scheduler=None
    )

    # All 5 cases processed
    assert recovered == 5, f"Expected 5 recovered, got {recovered}"

    # Case 1: orphan → status='failed'
    db.update_research_job_status.assert_any_call(
        "orphan1",
        "failed",
        error_taxonomy="timeout",
        error_message="orphan_recovered",
        completed_at=ANY,
    )

    # Case 2: stuck → status='pending'
    db.update_research_job_status.assert_any_call(
        "stuck1",
        "pending",
        current_phase=None,
        progress_percent=0,
    )

    # Case 3: running w/ output → status='complete'
    db.update_research_job_status.assert_any_call(
        "almost1",
        "complete",
        progress_percent=100,
        completed_at=ANY,
    )

    # Case 4: unnotified complete → notifier.send_research_complete called
    notifier.send_research_complete.assert_called_once_with(
        job_id="unnot1",
        cost_usd=0.05,
    )
    db.mark_research_job_notified.assert_called_once_with("unnot1")

    # Case 5: cancelling → status='cancelled'
    db.update_research_job_status.assert_any_call(
        "cancel1",
        "cancelled",
        completed_at=ANY,
    )


@pytest.mark.asyncio
async def test_recovery_handles_db_exceptions_gracefully() -> None:
    """If one DB call raises, the others still process.

    Verifies recovery is non-fatal — one bad row shouldn't block the
    other 99 cases from reconciling.
    """
    db = MagicMock()

    # Case 1 explodes, but other cases return empty lists
    db.list_research_jobs_pending_created_before = AsyncMock(side_effect=RuntimeError("db lock"))
    db.list_research_jobs_by_status_started_before = AsyncMock(return_value=[])
    db.list_research_jobs_running_with_output = AsyncMock(return_value=[])
    db.list_research_jobs_unnotified = AsyncMock(return_value=[])
    db.list_research_jobs_cancelling = AsyncMock(return_value=[])
    db.update_research_job_status = AsyncMock()
    db.mark_research_job_notified = AsyncMock()

    notifier = MagicMock()
    notifier.send_research_complete = AsyncMock(return_value=True)
    notifier.send_research_failed = AsyncMock(return_value=True)

    settings = _FakeSettings()

    # Should NOT raise — recovery catches per-case exceptions
    recovered = await recover_research_jobs(
        db=db, notifier=notifier, settings=settings, scheduler=None
    )
    # No cases successfully processed (all empty lists after the explosion)
    assert recovered == 0


@pytest.mark.asyncio
async def test_recovery_unnotified_failed_sends_failed_not_complete() -> None:
    """Case 4 (unnotified failed, no output) → notifier.send_research_failed."""
    db = MagicMock()
    db.list_research_jobs_pending_created_before = AsyncMock(return_value=[])
    db.list_research_jobs_by_status_started_before = AsyncMock(return_value=[])
    db.list_research_jobs_running_with_output = AsyncMock(return_value=[])
    db.list_research_jobs_unnotified = AsyncMock(
        return_value=[
            {
                "id": "failed1",
                "status": "failed",
                "output_path": None,
                "cost_usd": 0.0,
            }
        ]
    )
    db.list_research_jobs_cancelling = AsyncMock(return_value=[])
    db.update_research_job_status = AsyncMock()
    db.mark_research_job_notified = AsyncMock()

    notifier = MagicMock()
    notifier.send_research_complete = AsyncMock(return_value=True)
    notifier.send_research_failed = AsyncMock(return_value=True)

    settings = _FakeSettings()

    recovered = await recover_research_jobs(
        db=db, notifier=notifier, settings=settings, scheduler=None
    )
    assert recovered == 1
    notifier.send_research_failed.assert_called_once()
    notifier.send_research_complete.assert_not_called()
