"""DR-Q1A-PRE1A cancel and retry documentation tests.

These tests prove that ``cancel_job`` and ``_run_phase_with_retry``
behave as the docstrings describe. They are a regression gate for
the documentation-only changes in DR-Q1A-PRE1A: no implementation
changed, but the docstrings are now accurate to the actual code
behavior, and these tests pin that contract.

Anti-regression checks (DR-Q1A-PRE1A):
- ``cancel_job(graceful=True)`` marks ``cancelling`` in the DB and
  returns immediately. It does NOT wait. It does NOT prove the
  cancellation of the running asyncio task. It does NOT prove the
  cancellation of an in-flight provider request.
- ``cancel_job(graceful=False)`` marks ``cancelled`` in the DB
  and returns immediately. Same caveats.
- Cancellation is NOT a hard monetary boundary.
- ``_run_phase_with_retry`` does at most 3 total attempts. Effective
  waits are 1 second after the first failure and 4 seconds after the
  second failure. The third failure TERMINATES the phase. The value
  16 in ``_RETRY_BACKOFF_SCHEDULE`` is NOT consumed by the current
  loop.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.jobs.models import JobStatus, PhaseName
from hermes.jobs.service import (
    _RETRY_BACKOFF_SCHEDULE,
    RETRYABLE_ERRORS,
    DeepResearchService,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeFetchResult:
    def __init__(self, body: bytes = b"<html></html>") -> None:
        self.body = body
        self.media_type = "text/html"
        self.status = 200
        self.redirect_count = 0


class _FakeFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch(self, url: str) -> _FakeFetchResult:
        self.calls.append(url)
        return _FakeFetchResult()


class _FakeSettings:
    deep_research_daily_budget_usd = 3.0
    deep_research_max_sources = 5
    deep_research_phase1_timeout_s = 30
    deep_research_phase2_timeout_s = 30
    deep_research_phase3_timeout_s = 90
    deep_research_phase4_timeout_s = 120
    deep_research_phase5_timeout_s = 5
    deep_research_per_source_max_tokens = 3000
    deep_research_data_root = "/tmp/hermes-test-lifecycle-docstring"


@pytest.fixture
def service_with_db(db, tmp_path: Path):
    """Build a DeepResearchService pointing at the real DB, with mock notifier/llm/search/fetcher."""
    settings = _FakeSettings()
    settings.deep_research_data_root = str(tmp_path / "jobs")
    notifier = MagicMock()
    notifier.send_research_complete = AsyncMock(return_value=True)
    notifier.send_research_failed = AsyncMock(return_value=True)
    llm = MagicMock()
    search = MagicMock()
    scheduler = MagicMock()
    fetcher = _FakeFetcher()
    service = DeepResearchService(
        db=db,
        notifier=notifier,
        llm_router=llm,
        web_search=search,
        fetcher=fetcher,
        settings=settings,
        scheduler=scheduler,
    )
    return service, tmp_path


# ---------------------------------------------------------------------------
# cancel_job: graceful=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_job_graceful_true_marks_cancelling_and_returns_immediately(
    service_with_db, db
) -> None:
    """``cancel_job(graceful=True)`` marks the job ``cancelling`` in the
    DB and returns immediately with ``status=CANCELLING``.

    The function does NOT wait. The function does NOT prove the
    cancellation of the running asyncio task or any in-flight
    provider request.
    """
    service, _ = service_with_db
    job_id = "can01"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Time the call: must be fast (no 10s wait).
    t0 = datetime.now(UTC)
    response = await service.cancel_job(job_id, graceful=True)
    elapsed = (datetime.now(UTC) - t0).total_seconds()
    assert elapsed < 1.0, (
        f"cancel_job(graceful=True) took {elapsed:.2f}s; must return "
        f"immediately (the previous docstring's 'await current phase "
        f"finish (max 10s)' was incorrect)."
    )

    # Response is CANCELLING (transient), graceful=True.
    assert response.status == JobStatus.CANCELLING
    assert response.graceful is True
    assert response.id == job_id

    # DB row is now 'cancelling'.
    row = await db.get_research_job(job_id)
    assert row is not None
    assert row["status"] == "cancelling", (
        f"DB status must be 'cancelling' after graceful cancel; got "
        f"{row['status']!r}"
    )


@pytest.mark.asyncio
async def test_cancel_job_graceful_true_does_not_prove_task_cancellation(
    service_with_db, db
) -> None:
    """``cancel_job(graceful=True)`` does NOT prove the cancellation
    of the running asyncio task. The current code only updates the
    DB; the running task, if any, continues until it next polls
    the status.
    """
    service, _ = service_with_db
    job_id = "can02"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Patch any possible sleep/wait helpers to detect if cancel_job
    # tries to wait. We patch asyncio.sleep and time.sleep to record
    # any sleep calls.
    with patch("hermes.jobs.service.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        await service.cancel_job(job_id, graceful=True)

    # If the docstring's "await current phase finish (max 10s)" had
    # been implemented, asyncio.sleep would have been called with
    # a value > 0. It must NOT be called in the current
    # implementation.
    mock_sleep.assert_not_called(), (
        "cancel_job(graceful=True) must NOT call asyncio.sleep. "
        "The previous docstring claimed it awaited the current "
        "phase (max 10s); the current implementation only updates "
        "the DB and returns."
    )


# ---------------------------------------------------------------------------
# cancel_job: graceful=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_job_graceful_false_marks_cancelled_and_returns_immediately(
    service_with_db, db
) -> None:
    """``cancel_job(graceful=False)`` marks the job ``cancelled`` in
    the DB and returns immediately. Same caveats as graceful=True.
    """
    service, _ = service_with_db
    job_id = "can03"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    t0 = datetime.now(UTC)
    response = await service.cancel_job(job_id, graceful=False)
    elapsed = (datetime.now(UTC) - t0).total_seconds()
    assert elapsed < 1.0, (
        f"cancel_job(graceful=False) took {elapsed:.2f}s; must return "
        f"immediately (the previous docstring's 'hard cancel' was "
        f"misleading; the current code only updates the DB)."
    )

    assert response.status == JobStatus.CANCELLED
    assert response.graceful is False
    assert response.id == job_id

    row = await db.get_research_job(job_id)
    assert row is not None
    assert row["status"] == "cancelled", (
        f"DB status must be 'cancelled' after hard cancel; got {row['status']!r}"
    )
    assert row.get("completed_at") is not None, (
        "DB row must have completed_at set after hard cancel"
    )


# ---------------------------------------------------------------------------
# cancel_job: error semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_job_raises_for_unknown_id(service_with_db) -> None:
    """``cancel_job`` raises JobNotFoundError for an unknown id."""
    from hermes.jobs.exceptions import JobNotFoundError

    service, _ = service_with_db
    with pytest.raises(JobNotFoundError):
        await service.cancel_job("does-not-exist", graceful=True)


@pytest.mark.asyncio
async def test_cancel_job_raises_for_terminal_status(service_with_db, db) -> None:
    """``cancel_job`` raises JobAlreadyTerminalError for jobs in
    ``complete``, ``failed``, or ``cancelled``."""
    from hermes.jobs.exceptions import JobAlreadyTerminalError

    service, _ = service_with_db
    job_id = "can04"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )
    # Mark complete via DB
    await db.update_research_job_status(job_id, "complete")
    with pytest.raises(JobAlreadyTerminalError):
        await service.cancel_job(job_id, graceful=True)


# ---------------------------------------------------------------------------
# Cancellation is not a hard monetary boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_job_does_not_call_provider(service_with_db, db) -> None:
    """``cancel_job`` does NOT make any external call (no Tavily, no
    MiniMax, no Ollama, no Telegram). It is a DB-only operation.

    This proves that cancellation is NOT a hard monetary boundary:
    the service cannot prove that a previously-dispatched provider
    call has been stopped.
    """
    service, _ = service_with_db
    job_id = "can05"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Patch the LLM router, search router, fetcher, and notifier to
    # detect any outbound call.
    with patch.object(
        service, "_llm", wraps=service._llm
    ) as mock_llm, patch.object(
        service, "_search", wraps=service._search
    ) as mock_search, patch.object(
        service, "_fetcher", wraps=service._fetcher
    ) as mock_fetcher, patch.object(
        service, "_notifier", wraps=service._notifier
    ) as mock_notifier:
        await service.cancel_job(job_id, graceful=True)
        await service.cancel_job(job_id, graceful=False)  # this raises (already terminal)
    # LLM, search, fetcher, notifier must NOT have been called.
    assert not mock_llm.method_calls, (
        f"cancel_job called LLM router: {mock_llm.method_calls}"
    )
    assert not mock_search.method_calls, (
        f"cancel_job called search router: {mock_search.method_calls}"
    )
    assert not mock_fetcher.method_calls, (
        f"cancel_job called fetcher: {mock_fetcher.method_calls}"
    )
    assert not mock_notifier.method_calls, (
        f"cancel_job called notifier: {mock_notifier.method_calls}"
    )


# ---------------------------------------------------------------------------
# _run_phase_with_retry: 3 attempts, [1, 4] effective waits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_phase_with_retry_succeeds_on_first_attempt(service_with_db) -> None:
    """``_run_phase_with_retry`` returns the result of the first
    successful attempt without retry.
    """
    service, _ = service_with_db
    calls: list[int] = []

    async def phase_fn() -> str:
        calls.append(1)
        return "ok"

    result = await service._run_phase_with_retry(
        job_id="retry01",
        phase_name=PhaseName.SEARCH,
        phase_fn=phase_fn,
    )
    assert result == "ok"
    assert len(calls) == 1, f"phase_fn called {len(calls)} times; expected 1"


@pytest.mark.asyncio
async def test_run_phase_with_retry_retries_3_times_with_1s_and_4s_waits(
    service_with_db
) -> None:
    """``_run_phase_with_retry`` retries up to 3 total attempts.

    Effective waits: 1s after the first failure, 4s after the
    second failure. The third failure TERMINATES the phase.
    """
    service, _ = service_with_db
    calls: list[int] = []

    async def phase_fn() -> None:
        calls.append(1)
        from hermes.jobs.exceptions import PhaseError
        raise PhaseError("llm_5xx", "test_failure", retryable=True)

    # Patch asyncio.sleep to record wait values without actually waiting.
    with patch("hermes.jobs.service.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        from hermes.jobs.exceptions import PhaseError
        with pytest.raises(PhaseError):
            await service._run_phase_with_retry(
                job_id="retry02",
                phase_name=PhaseName.SEARCH,
                phase_fn=phase_fn,
            )

    # 3 total attempts
    assert len(calls) == 3, (
        f"phase_fn called {len(calls)} times; expected exactly 3 total attempts"
    )
    # 2 sleeps: 1s and 4s
    assert mock_sleep.call_count == 2, (
        f"asyncio.sleep called {mock_sleep.call_count} times; expected 2 "
        f"(after 1st and 2nd failure; 3rd failure terminates)"
    )
    # The actual wait values: 1 and 4
    wait_values = [call.args[0] for call in mock_sleep.call_args_list]
    assert wait_values == [1, 4], (
        f"effective retry waits must be [1, 4]; got {wait_values}. "
        f"The value 16 in _RETRY_BACKOFF_SCHEDULE is NOT consumed by "
        f"the current loop."
    )


@pytest.mark.asyncio
async def test_run_phase_with_retry_terminates_after_3_attempts(service_with_db) -> None:
    """After 3 failed attempts, ``_run_phase_with_retry`` terminates
    by raising PhaseError. The 3rd failure does NOT trigger another
    sleep+retry.
    """
    service, _ = service_with_db
    calls: list[int] = []

    async def phase_fn() -> None:
        calls.append(1)
        from hermes.jobs.exceptions import PhaseError
        raise PhaseError("timeout", "test_timeout", retryable=True)

    with patch("hermes.jobs.service.asyncio.sleep", new=AsyncMock()):
        from hermes.jobs.exceptions import PhaseError
        with pytest.raises(PhaseError):
            await service._run_phase_with_retry(
                job_id="retry03",
                phase_name=PhaseName.FINAL_SYNTHESIS,
                phase_fn=phase_fn,
            )

    assert len(calls) == 3, (
        f"phase_fn called {len(calls)} times; expected exactly 3 (3rd "
        f"failure terminates, no 4th attempt)"
    )


@pytest.mark.asyncio
async def test_run_phase_with_retry_does_not_retry_non_retryable(service_with_db) -> None:
    """``_run_phase_with_retry`` does NOT retry on non-retryable
    PhaseError taxonomies (e.g., ``cancelled``, ``budget_exceeded``).
    """
    service, _ = service_with_db
    calls: list[int] = []

    async def phase_fn() -> None:
        calls.append(1)
        from hermes.jobs.exceptions import PhaseError
        raise PhaseError("cancelled", "user_cancelled", retryable=False)

    with patch("hermes.jobs.service.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        from hermes.jobs.exceptions import PhaseError
        with pytest.raises(PhaseError):
            await service._run_phase_with_retry(
                job_id="retry04",
                phase_name=PhaseName.SEARCH,
                phase_fn=phase_fn,
            )

    # Only 1 call (no retry on non-retryable)
    assert len(calls) == 1
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_run_phase_with_retry_succeeds_on_2nd_attempt(service_with_db) -> None:
    """``_run_phase_with_retry`` returns the result of the 2nd
    attempt if the 1st fails and the 2nd succeeds. The wait after
    the 1st failure is 1s.
    """
    service, _ = service_with_db
    calls: list[int] = []

    async def phase_fn() -> str:
        calls.append(1)
        if len(calls) == 1:
            from hermes.jobs.exceptions import PhaseError
            raise PhaseError("llm_5xx", "first_fail", retryable=True)
        return "second_ok"

    with patch("hermes.jobs.service.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        result = await service._run_phase_with_retry(
            job_id="retry05",
            phase_name=PhaseName.SEARCH,
            phase_fn=phase_fn,
        )
    assert result == "second_ok"
    assert len(calls) == 2
    # Only 1 sleep (after 1st failure, value 1)
    assert mock_sleep.call_count == 1
    assert mock_sleep.call_args_list[0].args[0] == 1


def test_retry_backoff_schedule_residue_is_16() -> None:
    """The value 16 in ``_RETRY_BACKOFF_SCHEDULE = (1, 4, 16)`` is a
    residue. The current loop reads only ``_RETRY_BACKOFF_SCHEDULE[0]``
    and ``_RETRY_BACKOFF_SCHEDULE[1]`` (when ``attempt < 2``). The
    third attempt (``attempt == 2``) terminates the phase before
    reading ``_RETRY_BACKOFF_SCHEDULE[2]``.

    This is a documentation/regression test. If a future change
    consumes the 16 (e.g., by changing the loop bound), the rate
    schedule changes and this test must be updated.
    """
    assert _RETRY_BACKOFF_SCHEDULE == (1, 4, 16), (
        f"_RETRY_BACKOFF_SCHEDULE must be (1, 4, 16); got {_RETRY_BACKOFF_SCHEDULE}. "
        f"The current code reads only _RETRY_BACKOFF_SCHEDULE[0..1] (when "
        f"attempt < 2); the 16 is a residue and is NOT consumed."
    )


def test_retryable_errors_set_is_documented() -> None:
    """The set of retryable error taxonomies is documented and
    fixed.
    """
    expected = frozenset({"search_5xx", "llm_5xx", "timeout", "network"})
    assert expected == RETRYABLE_ERRORS, (
        f"RETRYABLE_ERRORS must be {set(expected)}; got {set(RETRYABLE_ERRORS)}. "
        f"Adding a new retryable taxonomy is a behavior change; "
        f"document and update tests deliberately."
    )
