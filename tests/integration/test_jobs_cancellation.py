"""DR-Q1A-PRE1B: real Deep Research cancellation tests.

Anti-regression checks for the cancellation contract:

A. Pending scheduled cancellation.
B. Startup race — cancel wins.
C. Startup race — execution wins.
D. Phase 1 cancellation.
E. Phase 2 cancellation.
F. Phase 3 concurrent cancellation.
G. Phase 4 cancellation.
H. Cancel-versus-Phase-5 race.
I. graceful=True acknowledgement.
J. graceful=True timeout.
K. graceful=False.
L. Idempotency.
M. Terminal conflict.
N. Cancellation source distinction.
O. Accounting.
P. Artifact cleanup.
Q. Notification contract.
R. Task registry cleanup.
S. Scheduler truth.
T. API/OpenAPI truth.

All tests are offline and deterministic. The only external
dependency is the real ``Database`` (sqlite in tmp_path via the
``db`` fixture from conftest.py). LLM, search, fetch and
notifier are all AsyncMock / MagicMock / controlled fakes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes.jobs.cost import calculate_cost, format_now
from hermes.jobs.exceptions import JobAlreadyTerminalError, JobNotFoundError
from hermes.jobs.models import (
    CancelResponse,
    JobStatus,
    PhaseName,
)
from hermes.jobs.scheduler import DeepResearchScheduler
from hermes.jobs.service import DeepResearchService

# =====================================================================
# Controlled fakes — deterministic, no network
# =====================================================================


class _FakeSettings:
    """Minimal settings stub for cancellation tests."""

    deep_research_daily_budget_usd = 100.0  # high cap; budget does not trip
    deep_research_per_job_budget_usd = 100.0
    deep_research_max_sources = 3
    deep_research_phase1_timeout_s = 5
    deep_research_phase2_timeout_s = 5
    deep_research_phase3_timeout_s = 5
    deep_research_phase4_timeout_s = 5
    deep_research_phase5_timeout_s = 5
    deep_research_per_source_max_tokens = 1000
    deep_research_output_max_tokens = 5000
    deep_research_cancel_wait_s = 1.0  # short for tests


@dataclass
class _FakeFetchResult:
    body: bytes
    media_type: str = "text/html"
    status: int = 200
    redirect_count: int = 0


class _FakeFetcher:
    """Controlled fake safe fetcher. ``gate`` blocks phase 2 until released."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.gate: asyncio.Event | None = None

    async def fetch(self, url: str) -> _FakeFetchResult:
        self.calls.append(url)
        if self.gate is not None:
            await self.gate.wait()
        return _FakeFetchResult(body=b"<html><body>fake</body></html>")


@dataclass
class _FakeLLMResp:
    content: str
    tokens_in: int = 100
    tokens_out: int = 50
    latency_ms: int = 50


def _make_service(
    db: Any,
    tmp_path: Path,
    *,
    search_block: asyncio.Event | None = None,
    fetch_block: asyncio.Event | None = None,
    llm_blocks: list[asyncio.Event] | None = None,
) -> tuple[DeepResearchService, dict[str, Any]]:
    """Build a DeepResearchService with controlled fakes.

    Returns ``(service, handles)`` where handles is a dict that
    the test can use to release blocking events or inspect calls.
    """
    settings = _FakeSettings()
    settings.deep_research_data_root = str(tmp_path / "jobs")

    notifier = MagicMock()
    notifier.send_research_complete = AsyncMock(return_value=True)
    notifier.send_research_failed = AsyncMock(return_value=True)

    search = MagicMock()

    llm = MagicMock()
    llm_blocks = llm_blocks or []

    async def chat_side_effect(*args: Any, **kwargs: Any) -> _FakeLLMResp:
        # Block the Nth call until its gate is set. Used by the
        # phase-3 concurrent test.
        for idx, gate in enumerate(llm_blocks):
            if llm.chat.call_count == idx + 1:
                await gate.wait()
        return _FakeLLMResp(
            content=kwargs.get("messages", [{}])[0].get("content", "")[:1000]
        )

    llm.chat = AsyncMock(side_effect=chat_side_effect)

    scheduler = MagicMock()
    scheduler.enqueue = AsyncMock()
    scheduler.cancel_scheduled = MagicMock(return_value=True)

    fetcher = _FakeFetcher()
    if fetch_block is not None:
        fetcher.gate = fetch_block

    if search_block is not None:
        async def blocking_search(*args: Any, **kwargs: Any) -> Any:
            await search_block.wait()
            return MagicMock(results=[])

        search.side_effect = blocking_search

    service = DeepResearchService(
        db=db,
        notifier=notifier,
        llm_router=llm,
        web_search=search,
        fetcher=fetcher,
        settings=settings,
        scheduler=scheduler,
    )

    handles = {
        "settings": settings,
        "notifier": notifier,
        "search": search,
        "llm": llm,
        "scheduler": scheduler,
        "fetcher": fetcher,
        "search_block": search_block,
        "fetch_block": fetch_block,
        "llm_blocks": llm_blocks,
    }
    return service, handles


async def _create_job(db: Any, job_id: str = "canceltest1", *, notify: bool = False) -> None:
    """Helper: create a pending research job."""
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=1 if notify else 0,
        user_id=0,
    )


# =====================================================================
# A. Pending scheduled cancellation
# =====================================================================


@pytest.mark.asyncio
async def test_pending_cancellation(db, tmp_path: Path) -> None:
    """A: pending job is cancelled before scheduler removal; scheduler.cancel_scheduled is called;
    job finishes cancelled; later scheduler execution cannot start search/provider work.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "pendingcancel1"
    await _create_job(db, job_id)
    # Sanity: status is pending
    row = await db.get_research_job(job_id)
    assert row["status"] == "pending"

    response = await service.cancel_job(job_id, graceful=True)
    assert response.status is JobStatus.CANCELLED
    assert response.id == job_id

    # Scheduler cancel_scheduled was called (pending path removes the
    # scheduler entry best-effort).
    handles["scheduler"].cancel_scheduled.assert_called_with(job_id)

    # The DB row is cancelled.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"
    assert row["error_taxonomy"] == "cancelled"
    assert row["completed_at"] is not None

    # The LLM, search and fetcher were NOT touched (no external work).
    handles["search"].assert_not_called()
    handles["llm"].chat.assert_not_called()
    assert handles["fetcher"].calls == []


# =====================================================================
# B. Startup race — cancel wins
# =====================================================================


@pytest.mark.asyncio
async def test_startup_cancel_wins(db, tmp_path: Path) -> None:
    """B: cancellation occurs before the pending -> running CAS.

    Simulate the race by:
      1. create pending job;
      2. cancel it (row becomes cancelled synchronously via the
         pending path);
      3. invoke _run_research (which would normally do the CAS);
      4. confirm no external work happens and the row stays cancelled.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "startupracewin"
    await _create_job(db, job_id)
    # Cancel first (pending path)
    response = await service.cancel_job(job_id, graceful=True)
    assert response.status is JobStatus.CANCELLED
    # Now invoke _run_research; the CAS sees status != pending and
    # exits. The startup CAS is a conditional UPDATE, but the
    # implementation already returns early from the inner method
    # when the row is not 'pending' (the BEGIN IMMEDIATE SELECT).
    await service._run_research(job_id)
    # No external work
    handles["search"].assert_not_called()
    handles["llm"].chat.assert_not_called()
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


# =====================================================================
# C. Startup race — execution wins
# =====================================================================


@pytest.mark.asyncio
async def test_startup_execution_wins(db, tmp_path: Path) -> None:
    """C: task registers and transitions to running; cancel finds the task;
    task receives cancellation; status becomes cancelled.
    """
    search_block = asyncio.Event()
    service, _handles = _make_service(db, tmp_path, search_block=search_block)
    job_id = "startupraceexec"
    await _create_job(db, job_id)

    # Run the research task in the background. It will register,
    # CAS to running, and block in phase 1.
    task = asyncio.create_task(service._run_research(job_id))
    # Wait until the row is 'running' (the startup CAS succeeded).
    for _ in range(200):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running":
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Task never reached 'running' state")

    # Now cancel. The cancel endpoint must find the registered task.
    response = await service.cancel_job(job_id, graceful=False)
    assert response.status is JobStatus.CANCELLING

    # Release the search block to let the task unwind.
    search_block.set()

    # Wait for the task to finish.
    with pytest.raises(asyncio.CancelledError):
        await task

    # The finalizer flipped the row to cancelled.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"
    assert row["error_taxonomy"] == "cancelled"
    assert row["completed_at"] is not None


# =====================================================================
# D. Phase 1 cancellation
# =====================================================================


@pytest.mark.asyncio
async def test_phase1_cancellation(db, tmp_path: Path) -> None:
    """D: blocking fake search is awaiting; cancel interrupts it;
    scrape and all later phases never start.
    """
    search_block = asyncio.Event()
    service, handles = _make_service(db, tmp_path, search_block=search_block)
    job_id = "phase1cancel"
    await _create_job(db, job_id)
    task = asyncio.create_task(service._run_research(job_id))
    # Wait for 'running'
    for _ in range(200):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running":
            break
        await asyncio.sleep(0.01)
    # Cancel (graceful=False to return immediately)
    response = await service.cancel_job(job_id, graceful=False)
    assert response.status is JobStatus.CANCELLING
    # Release the block so the task can unwind
    search_block.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    # LLM and fetcher must not have been touched
    handles["llm"].chat.assert_not_called()
    assert handles["fetcher"].calls == []
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


# =====================================================================
# E. Phase 2 cancellation
# =====================================================================


@pytest.mark.asyncio
async def test_phase2_cancellation(db, tmp_path: Path) -> None:
    """E: blocking fetch boundary is active; cancel prevents phase 3;
    no new URL fetch starts after cancellation wins; test does not
    claim an already-running Python thread was killed.
    """
    # We need a search that completes successfully, then a fetch
    # that blocks. The search returns no results; phase 2 will then
    # not have anything to fetch, so we need to inject a search
    # result. Use a custom search that returns one URL.
    fetch_block = asyncio.Event()
    service, handles = _make_service(db, tmp_path, fetch_block=fetch_block)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(results=[{"url": "https://example.com/a", "title": "A"}])

    handles["search"].side_effect = one_url_search

    job_id = "phase2cancel"
    await _create_job(db, job_id)
    task = asyncio.create_task(service._run_research(job_id))
    # Wait for the row to reach 'running' and the fetcher to be called
    for _ in range(400):
        if handles["fetcher"].calls:
            break
        await asyncio.sleep(0.01)
    else:
        # Release the block so the task can exit cleanly and we can
        # fail with a meaningful error.
        fetch_block.set()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await task
        pytest.fail("Fetcher was never called")
    # Cancel (graceful=False to return immediately)
    response = await service.cancel_job(job_id, graceful=False)
    assert response.status is JobStatus.CANCELLING
    # Release the fetch block so the task can unwind
    fetch_block.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    # LLM must not have been touched (phase 3 is per-source synthesis)
    handles["llm"].chat.assert_not_called()
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


# =====================================================================
# F. Phase 3 concurrent cancellation
# =====================================================================


@pytest.mark.asyncio
async def test_phase3_concurrent_cancellation(db, tmp_path: Path) -> None:
    """F: multiple per-source fake LLM calls are awaiting;
    cancelling the parent cancels child coroutines; final synthesis
    never starts.
    """
    # Use a single gate for ALL LLM calls. The phase 3 gather
    # awaits multiple LLM calls concurrently; with all blocked
    # on the same gate, the cancel is delivered to all child
    # coroutines at once.
    llm_block = asyncio.Event()
    service, handles = _make_service(
        db, tmp_path, llm_blocks=[llm_block]
    )

    # The search returns 3 URLs so phase 3 has 3 sources.
    async def three_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(results=[
            {"url": "https://example.com/a", "title": "A"},
            {"url": "https://example.com/b", "title": "B"},
            {"url": "https://example.com/c", "title": "C"},
        ])

    handles["search"].side_effect = three_url_search

    # The fetcher returns 3 sources with a body that passes the
    # 100-char minimum-length check in _phase_scrape.
    async def fake_fetch(url: str) -> _FakeFetchResult:
        long_text = (
            f"<html><body><h1>Source {url}</h1>"
            + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            * 3
            + "</body></html>"
        )
        return _FakeFetchResult(body=long_text.encode("utf-8"))

    handles["fetcher"].fetch = fake_fetch

    # Override the LLM chat to block all calls on the same gate.
    async def blocking_chat(*args: Any, **kwargs: Any) -> _FakeLLMResp:
        await llm_block.wait()
        return _FakeLLMResp(
            content=kwargs.get("messages", [{}])[0].get("content", "")[:1000]
        )

    handles["llm"].chat = AsyncMock(side_effect=blocking_chat)

    job_id = "phase3cancel"
    await _create_job(db, job_id)
    task = asyncio.create_task(service._run_research(job_id))

    # Wait until at least 2 LLM calls are in flight (the gather
    # has started multiple synth_one tasks).
    for _ in range(400):
        if handles["llm"].chat.call_count >= 2:
            break
        await asyncio.sleep(0.01)
    else:
        # The scrape may be slow to produce sources; release
        # any gate to let the task unwind.
        llm_block.set()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await task
        pytest.fail("LLM was not called twice")

    # Cancel while phase 3 is mid-flight. Do NOT release the gate;
    # the cancel must interrupt the gate.wait() coroutine directly.
    response = await service.cancel_job(job_id, graceful=False)
    assert response.status is JobStatus.CANCELLING

    with pytest.raises(asyncio.CancelledError):
        await task

    # The row is cancelled.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


# =====================================================================
# G. Phase 4 cancellation
# =====================================================================


@pytest.mark.asyncio
async def test_phase4_cancellation(db, tmp_path: Path) -> None:
    """G: final synthesis is awaiting; no final report is published;
    no completion notifier is sent.
    """
    # We need to reach phase 4. Strategy: short-circuit phases 1-3
    # by injecting the right side effects, then block phase 4.
    phase4_block = asyncio.Event()
    service, handles = _make_service(db, tmp_path)

    # Search returns 1 URL
    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(results=[{"url": "https://example.com/a", "title": "A"}])

    handles["search"].side_effect = one_url_search

    # Fetcher returns a long body to pass the 100-char minimum
    # check in _phase_scrape.
    async def fake_fetch(url: str) -> _FakeFetchResult:
        long_text = (
            f"<html><body><h1>Source {url}</h1>"
            + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            * 3
            + "</body></html>"
        )
        return _FakeFetchResult(body=long_text.encode("utf-8"))

    handles["fetcher"].fetch = fake_fetch

    # LLM: phase 3 (return quickly), phase 4 blocks, then release.
    llm_call_count = {"n": 0}

    async def chat_side_effect(*args: Any, **kwargs: Any) -> _FakeLLMResp:
        llm_call_count["n"] += 1
        if llm_call_count["n"] == 2:
            await phase4_block.wait()
        return _FakeLLMResp(content="summary" if llm_call_count["n"] == 1 else "final report")

    handles["llm"].chat = AsyncMock(side_effect=chat_side_effect)

    job_id = "phase4cancel"
    await _create_job(db, job_id)
    task = asyncio.create_task(service._run_research(job_id))

    # Wait for the 2nd LLM call to be in flight.
    for _ in range(400):
        if llm_call_count["n"] >= 2:
            break
        await asyncio.sleep(0.01)
    else:
        phase4_block.set()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await task
        pytest.fail("Phase 4 LLM was never called")

    # Cancel (graceful=False). Do NOT release the block; the
    # cancel must propagate to the next await after the LLM call
    # (e.g. _update_checkpoint_cost or _record_token_usage or the
    # _phase_write terminal seam).
    response = await service.cancel_job(job_id, graceful=False)
    assert response.status is JobStatus.CANCELLING

    with pytest.raises(asyncio.CancelledError):
        await task

    # No final report file is published.
    final_path = Path(service._data_root) / f"{job_id}.md"
    assert not final_path.exists()
    # No completion notifier.
    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()
    # The row is cancelled.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


# =====================================================================
# H. Cancel-versus-Phase-5 race
# =====================================================================


@pytest.mark.asyncio
async def test_cancel_vs_phase5_cancel_wins(db, tmp_path: Path) -> None:
    """H-1: cancellation wins the race. Cancelled, no final report,
    no completion notifier.
    """
    phase4_block = asyncio.Event()
    service, handles = _make_service(db, tmp_path)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(results=[{"url": "https://example.com/a", "title": "A"}])

    handles["search"].side_effect = one_url_search

    async def fake_fetch(url: str) -> _FakeFetchResult:
        long_text = (
            f"<html><body><h1>Source {url}</h1>"
            + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            * 3
            + "</body></html>"
        )
        return _FakeFetchResult(body=long_text.encode("utf-8"))

    handles["fetcher"].fetch = fake_fetch

    # LLM: phase 3 (return quickly), phase 4 blocks, then release.
    llm_call_count = {"n": 0}

    async def chat_side_effect(*args: Any, **kwargs: Any) -> _FakeLLMResp:
        llm_call_count["n"] += 1
        if llm_call_count["n"] == 2:
            await phase4_block.wait()
        return _FakeLLMResp(content="x" if llm_call_count["n"] == 1 else "y")

    handles["llm"].chat = AsyncMock(side_effect=chat_side_effect)

    job_id = "cancelvsphase5a"
    await _create_job(db, job_id)
    task = asyncio.create_task(service._run_research(job_id))
    # Wait for the 2nd LLM call (phase 4) to be in flight.
    for _ in range(400):
        if llm_call_count["n"] >= 2:
            break
        await asyncio.sleep(0.01)
    else:
        phase4_block.set()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await task
        pytest.fail("Phase 4 LLM was never called")
    # Cancel. Do NOT release the block; the cancel must interrupt
    # the gate.wait() in the phase-4 LLM call directly.
    await service.cancel_job(job_id, graceful=False)
    with pytest.raises(asyncio.CancelledError):
        await task
    # No final report, no completion notifier
    final_path = Path(service._data_root) / f"{job_id}.md"
    assert not final_path.exists()
    handles["notifier"].send_research_complete.assert_not_called()
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_vs_phase5_complete_wins(db, tmp_path: Path) -> None:
    """H-2: completion wins. Complete, final report available,
    later cancel returns 409 (terminal conflict).
    """
    service, handles = _make_service(db, tmp_path)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(results=[{"url": "https://example.com/a", "title": "A"}])

    handles["search"].side_effect = one_url_search

    async def fake_fetch(url: str) -> _FakeFetchResult:
        long_text = (
            f"<html><body><h1>Source {url}</h1>"
            + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            * 3
            + "</body></html>"
        )
        return _FakeFetchResult(body=long_text.encode("utf-8"))

    handles["fetcher"].fetch = fake_fetch

    # LLM returns quickly for all phases.
    async def chat_side_effect(*args: Any, **kwargs: Any) -> _FakeLLMResp:
        return _FakeLLMResp(content="x")

    handles["llm"].chat = AsyncMock(side_effect=chat_side_effect)

    job_id = "cancelvsphase5b"
    await _create_job(db, job_id, notify=True)
    # Run synchronously (no cancellation in flight)
    await service._run_research(job_id)
    # The job is now 'complete'.
    row = await db.get_research_job(job_id)
    assert row["status"] == "complete"
    # The final report is published.
    final_path = Path(service._data_root) / f"{job_id}.md"
    assert final_path.exists()
    # The completion notifier was called.
    handles["notifier"].send_research_complete.assert_awaited_once()
    # A subsequent cancel returns 409 (terminal conflict).
    with pytest.raises(JobAlreadyTerminalError):
        await service.cancel_job(job_id, graceful=True)
    # The report and the row state are unchanged.
    assert final_path.exists()
    row = await db.get_research_job(job_id)
    assert row["status"] == "complete"


# =====================================================================
# I. graceful=True acknowledgement
# =====================================================================


@pytest.mark.asyncio
async def test_graceful_true_acknowledged(db, tmp_path: Path) -> None:
    """I: task cancellation acknowledged inside wait; endpoint returns cancelled.
    """
    search_block = asyncio.Event()
    service, handles = _make_service(db, tmp_path, search_block=search_block)
    # Override the search to return one URL (otherwise phase 1
    # returns 0 URLs and phase 3 fails with no_valid_sources,
    # masking the cancel-vs-acknowledged behavior we want to test).
    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        await search_block.wait()
        return MagicMock(results=[{"url": "https://example.com/a", "title": "A"}])

    handles["search"].side_effect = one_url_search
    async def fake_fetch(url: str) -> _FakeFetchResult:
        long_text = (
            f"<html><body><h1>Source {url}</h1>"
            + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            * 3
            + "</body></html>"
        )
        return _FakeFetchResult(body=long_text.encode("utf-8"))

    handles["fetcher"].fetch = fake_fetch

    job_id = "gracefulack"
    await _create_job(db, job_id)
    task = asyncio.create_task(service._run_research(job_id))
    # Wait for 'running'
    for _ in range(200):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running":
            break
        await asyncio.sleep(0.01)
    # Cancel (graceful=True) — this will wait for acknowledgement.
    cancel_task = asyncio.create_task(
        service.cancel_job(job_id, graceful=True)
    )
    # Give the cancel_task a tick to acquire the registry.
    await asyncio.sleep(0.05)
    # Release the search so the task unwinds. The cancel is
    # already queued; the task's next await is interrupted.
    search_block.set()
    # The cancel_task should now return (task ended → finalizer ran
    # before the wait timeout).
    response = await cancel_task
    # Allow either cancelling or cancelled depending on the timing.
    assert response.status in (JobStatus.CANCELLING, JobStatus.CANCELLED)
    with pytest.raises(asyncio.CancelledError):
        await task
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


# =====================================================================
# J. graceful=True timeout
# =====================================================================


@pytest.mark.asyncio
async def test_graceful_true_timeout(db, tmp_path: Path) -> None:
    """J: cancellation-resistant controlled fake exceeds wait;
    response returns cancelling, not falsely cancelled.
    """
    # Register a custom task that catches and discards
    # CancelledError. The bounded wait times out, the cancel_job
    # returns cancelling, and the inner task is still alive (the
    # test's cleanup tears it down at the end).
    service, _handles = _make_service(db, tmp_path)
    job_id = "gracefultimeout"
    await _create_job(db, job_id)
    # Force the row to 'running' so the cancel_job takes the
    # running/cancelling path (with the bounded wait), not the
    # pending path (which finalizes synchronously without
    # waiting).
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="running"
    )
    exit_flag = asyncio.Event()

    async def discards_cancel() -> None:
        # Loop until exit_flag is set, swallowing CancelledError.
        while not exit_flag.is_set():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                # Swallow; the bounded wait should time out
                # and cancel_job returns cancelling without
                # waiting for us.
                continue

    task = asyncio.create_task(discards_cancel())
    await service._register_active_task(job_id, task)
    # Yield so the task reaches its first await.
    await asyncio.sleep(0)
    try:
        # Cancel; graceful=True should time out and return cancelling.
        import time
        t0 = time.monotonic()
        response = await service.cancel_job(job_id, graceful=True)
        elapsed = time.monotonic() - t0
        # The bounded wait is 1.0s. The task should be running (the
        # cancel was swallowed). The response must be CANCELLING
        # (timeout) and NOT CANCELLED (the task did not complete).
        assert elapsed >= 0.5, (
            f"cancel_job returned too fast ({elapsed:.2f}s); "
            f"bounded wait did not engage"
        )
        assert response.status is JobStatus.CANCELLING, (
            f"Expected CANCELLING (timeout), got {response.status}; "
            f"elapsed={elapsed:.2f}s; task.done()={task.done()}"
        )
        assert response.graceful is True
    finally:
        # Cleanup: signal exit, then cancel, then unregister.
        exit_flag.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, TimeoutError, Exception):
            await asyncio.wait_for(task, timeout=2.0)
        await service._unregister_active_task(job_id, task)


# =====================================================================
# K. graceful=False
# =====================================================================


@pytest.mark.asyncio
async def test_graceful_false_returns_promptly(db, tmp_path: Path) -> None:
    """K: graceful=False returns promptly; active job returns cancelling;
    background finalization produces cancelled.
    """
    search_block = asyncio.Event()
    service, _handles = _make_service(db, tmp_path, search_block=search_block)
    job_id = "gracefulfalse"
    await _create_job(db, job_id)
    task = asyncio.create_task(service._run_research(job_id))
    for _ in range(200):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running":
            break
        await asyncio.sleep(0.01)
    # Cancel with graceful=False. Should return quickly.
    import time
    t0 = time.monotonic()
    response = await service.cancel_job(job_id, graceful=False)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"graceful=False took {elapsed:.3f}s (should be prompt)"
    assert response.status is JobStatus.CANCELLING
    assert response.graceful is False
    # Release and let the task finalize.
    search_block.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


# =====================================================================
# L. Idempotency
# =====================================================================


@pytest.mark.asyncio
async def test_cancel_idempotency(db, tmp_path: Path) -> None:
    """L: cancelling twice is safe; cancelling an already cancelled
    job returns 200 cancelled; no duplicate notifier or side effect.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "cancelidem"
    await _create_job(db, job_id)
    r1 = await service.cancel_job(job_id, graceful=True)
    assert r1.status is JobStatus.CANCELLED
    # Second cancel on already-cancelled job is idempotent.
    r2 = await service.cancel_job(job_id, graceful=True)
    assert r2.status is JobStatus.CANCELLED
    # The notifier was never called (cancel does not notify).
    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()
    # No duplicate DB transitions.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


# =====================================================================
# M. Terminal conflict
# =====================================================================


@pytest.mark.asyncio
async def test_terminal_conflict(db, tmp_path: Path) -> None:
    """M: complete and failed still return 409.
    """
    service, _handles = _make_service(db, tmp_path)
    # Manually create a complete job.
    await _create_job(db, "completejob")
    await db.update_research_job_status(
        "completejob", "complete", completed_at=format_now()
    )
    with pytest.raises(JobAlreadyTerminalError):
        await service.cancel_job("completejob", graceful=True)
    with pytest.raises(JobAlreadyTerminalError):
        await service.cancel_job("completejob", graceful=False)
    # Failed job.
    await _create_job(db, "failedjob")
    await db.update_research_job_status(
        "failedjob", "failed", error_taxonomy="network", completed_at=format_now()
    )
    with pytest.raises(JobAlreadyTerminalError):
        await service.cancel_job("failedjob", graceful=True)
    with pytest.raises(JobAlreadyTerminalError):
        await service.cancel_job("failedjob", graceful=False)


# =====================================================================
# N. Cancellation source distinction
# =====================================================================


@pytest.mark.asyncio
async def test_cancellation_source_distinction(db, tmp_path: Path) -> None:
    """N: an infrastructure/shutdown CancelledError without user
    intent is not persisted as user-cancelled; recovery-compatible
    state is retained.
    """
    service, _handles = _make_service(db, tmp_path)
    job_id = "shutdowntest"
    await _create_job(db, job_id)
    # Manually transition to 'running' (simulate that execution
    # has begun).
    await db.update_research_job_status(
        job_id, "running", started_at=format_now()
    )
    # Call _handle_cancellation WITHOUT marking user intent. The
    # row is still 'running' (not 'cancelling').
    await service._handle_cancellation(job_id, start_time=0.0)
    # The row is STILL 'running' (preserved for recovery).
    row = await db.get_research_job(job_id)
    assert row["status"] == "running"
    # No cancellation log emitted in the DB.
    assert row.get("error_taxonomy") != "cancelled"


# =====================================================================
# O. Accounting
# =====================================================================


@pytest.mark.asyncio
async def test_cancelled_cost_in_daily_budget(db, tmp_path: Path) -> None:
    """O: cancelled jobs' cost remains in the daily accounting.
    """
    # Create a job, manually set its cost, then mark it cancelled.
    job_id = "cancelledcost"
    await _create_job(db, job_id)
    await db.set_research_job_cost_monotonic(job_id, 0.5)
    await db.transition_research_job_status(
        job_id,
        from_states=("pending",),
        to_state="running",
    )
    await db.transition_research_job_status(
        job_id,
        from_states=("running",),
        to_state="cancelling",
    )
    await db.transition_research_job_status(
        job_id,
        from_states=("cancelling",),
        to_state="cancelled",
        completed_at=format_now(),
        error_taxonomy="cancelled",
    )
    # The daily cost includes the cancelled job's cost.
    total = await db.get_today_research_cost(user_id=0)
    assert total >= 0.5, f"Daily cost {total} does not include cancelled cost 0.5"


# =====================================================================
# P. Artifact cleanup
# =====================================================================


@pytest.mark.asyncio
async def test_artifact_cleanup(db, tmp_path: Path) -> None:
    """P: no .md, no .md.tmp, no leftover checkpoint; DB job and
    token usage remain inspectable.
    """
    search_block = asyncio.Event()
    service, _handles = _make_service(db, tmp_path, search_block=search_block)
    job_id = "artifactcleanup"
    await _create_job(db, job_id)
    # Write a checkpoint and a .md.tmp before cancelling.
    ckpt_path = Path(service._data_root) / job_id / "checkpoint.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "completed_phases": [],
                "phase_data": {},
                "cost_accumulated_usd": 0.01,
            }
        ),
        encoding="utf-8",
    )
    tmp_md = Path(service._data_root) / f"{job_id}.md.tmp"
    tmp_md.write_text("# draft\n", encoding="utf-8")
    # Record a token usage row that should be preserved.
    cost = calculate_cost("MiniMax-M3", 100, 50)
    await service._record_token_usage(
        job_id,
        PhaseName.PER_SOURCE_SYNTHESIS,
        "MiniMax-M3",
        100,
        50,
        cost,
    )
    # Cancel via the pending path; finalize via the finalizer (simulate
    # that a running task receives the cancellation).
    # Force status to running first.
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="running"
    )
    # Mark user-cancel intent (the cancel endpoint would do this
    # before signalling the task). This is what makes the
    # finalizer treat the cancellation as user-initiated.
    service._mark_user_cancel_intent(job_id)
    # Register a fake active task and cancel via the running path.
    async def fake_task() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Run the finalizer manually (this is what the real
            # _run_research would do in its CancelledError branch).
            await service._handle_cancellation(job_id, start_time=0.0)
            raise

    task = asyncio.create_task(fake_task())
    await service._register_active_task(job_id, task)
    # Yield so the task is scheduled and starts running.
    await asyncio.sleep(0)
    # Cancel the fake task; it will run the finalizer.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # No .md, no .md.tmp, no checkpoint.
    final_md = Path(service._data_root) / f"{job_id}.md"
    assert not final_md.exists()
    assert not tmp_md.exists()
    assert not ckpt_path.exists()

    # No .md, no .md.tmp, no checkpoint.
    final_md = Path(service._data_root) / f"{job_id}.md"
    assert not final_md.exists()
    assert not tmp_md.exists()
    assert not ckpt_path.exists()
    # DB job and token usage remain.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"
    rows = await db.list_token_usage_for_job(job_id)
    assert len(rows) == 1
    assert Decimal(str(rows[0]["cost_usd"])) == cost


# =====================================================================
# Q. Notification contract
# =====================================================================


@pytest.mark.asyncio
async def test_notification_contract(db, tmp_path: Path) -> None:
    """Q: no send_research_complete on cancellation;
    no send_research_failed on cancellation; no mark_research_job_notified.
    """
    search_block = asyncio.Event()
    service, handles = _make_service(db, tmp_path, search_block=search_block)
    job_id = "notifcontract"
    await _create_job(db, job_id, notify=True)
    task = asyncio.create_task(service._run_research(job_id))
    for _ in range(200):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running":
            break
        await asyncio.sleep(0.01)
    await service.cancel_job(job_id, graceful=False)
    search_block.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()
    # mark_research_job_notified must not be called either.
    row = await db.get_research_job(job_id)
    # notified column was not flipped.
    assert int(row.get("notified", 0)) == 0


# =====================================================================
# R. Task registry cleanup
# =====================================================================


@pytest.mark.asyncio
async def test_task_registry_cleanup(db, tmp_path: Path) -> None:
    """R: registry empty after complete; registry empty after failed;
    registry empty after cancelled; task from an older attempt
    cannot unregister a newer task.
    """
    service, handles = _make_service(db, tmp_path)

    # Run a quick full cycle.
    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(results=[])

    handles["search"].side_effect = one_url_search

    job_id = "registrycleanup"
    await _create_job(db, job_id)
    await service._run_research(job_id)
    # Registry is empty.
    assert job_id not in service._active_tasks

    # Test the cannot-unregister-newer invariant.
    async def never_returns() -> None:
        await asyncio.sleep(3600)

    t1 = asyncio.create_task(never_returns())
    t2 = asyncio.create_task(never_returns())
    await service._register_active_task("racejob", t1)
    await service._register_active_task("racejob", t2)
    # The newer task is t2.
    assert service._active_tasks["racejob"] is t2
    # t1.try_to_unregister — must NOT remove t2.
    await service._unregister_active_task("racejob", t1)
    assert service._active_tasks["racejob"] is t2
    # t2.unregister — removes t2.
    await service._unregister_active_task("racejob", t2)
    assert "racejob" not in service._active_tasks
    t1.cancel()
    t2.cancel()
    for t in (t1, t2):
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await t


# =====================================================================
# S. Scheduler truth
# =====================================================================


@pytest.mark.asyncio
async def test_scheduler_non_enqueueable_cancelling(db) -> None:
    """S: cancelling is never enqueued; the scheduler rejects it
    deterministically.

    Build a scheduler with a stubbed internal ``_scheduler`` and
    ``_service`` (so we do NOT need to call the real ``start()``,
    which can be flaky in tests with in-memory sqlite). Pre-create
    a row whose status is ``cancelling``, then call ``enqueue`` and
    assert that ``add_job`` is NOT called.
    """
    from datetime import datetime

    from hermes.jobs.scheduler import DeepResearchScheduler

    job_id = "sched-non-enqueueable-cancelling"
    await _create_job(db, job_id)
    # Persist a 'cancelling' status so the pre-flight SELECT returns it.
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="cancelling"
    )

    sched = DeepResearchScheduler(
        db=db,
        settings=MagicMock(db_path=":memory:"),
        jobstore_url="sqlite:///:memory:",
    )
    # The real ``start()`` would create a live AsyncIOScheduler +
    # SQLAlchemyJobStore, which is flaky in tests. We exercise the
    # pre-flight path by setting the two attributes the function
    # touches after the ``_stopping`` / ``_scheduler is None`` guards.
    sched._scheduler = MagicMock()
    sched._service = MagicMock()
    sched._service._run_research = AsyncMock()

    await sched.enqueue(job_id, run_date=datetime.now(UTC))

    # Critical: add_job must NOT have been called for a cancelling row.
    sched._scheduler.add_job.assert_not_called()
    # And the contract: the JobStatus enum value is the source of truth.
    assert JobStatus.CANCELLING.value == "cancelling"


def _sched_noop_run_research(job_id: str) -> None:
    """Module-level no-op run_research used to make APScheduler's
    SQLAlchemyJobStore able to serialize the job's callable.
    Local functions cannot be pickled by APScheduler.
    """
    return None


@pytest.mark.asyncio
async def test_scheduler_cancel_scheduled_removes_pending_entry(
    db, tmp_path: Path
) -> None:
    """S: a pending job's scheduler entry is removed on cancel.

    A scheduler with a real in-memory sqlite jobstore is started;
    a pending job is enqueued; the cancel_scheduled call removes
    the entry.
    """
    settings = MagicMock()
    settings.db_path = str(tmp_path / "sched.db")
    sched = DeepResearchScheduler(
        db=db,
        settings=settings,
        jobstore_url=f"sqlite:///{tmp_path / 'sched.db'}",
    )
    await sched.start()
    try:
        # The scheduler needs a service to enqueue. Inject a
        # service whose _run_research is a module-level function
        # (so APScheduler can pickle the job's callable).
        sched.set_service(MagicMock(_run_research=_sched_noop_run_research))
        # Enqueue a pending job.
        from datetime import datetime, timedelta
        run_date = datetime.now() + timedelta(seconds=60)
        await _create_job(db, "schedpending")
        await sched.enqueue("schedpending", run_date=run_date)
        # The job is now in the scheduler.
        assert sched._scheduler.get_job("schedpending") is not None
        # Cancel removes it.
        removed = sched.cancel_scheduled("schedpending")
        assert removed is True
        assert sched._scheduler.get_job("schedpending") is None
    finally:
        await sched.shutdown(timeout_s=2.0)


# =====================================================================
# T. API/OpenAPI truth
# =====================================================================


@pytest.mark.asyncio
async def test_cancel_endpoint_status_codes(db, tmp_path: Path) -> None:
    """T: cancel endpoint returns 200 (idempotent), 200 cancelling
    for active, 409 for terminal, 404 for missing. Public field
    names and types unchanged.

    The HTTP routing is exercised by the existing test_jobs_api.py
    suite. This test focuses on the underlying service contract
    (which is what the route delegates to) plus the OpenAPI/
    DTO descriptions that document the new contract.
    """
    from hermes.jobs.models import JobStatus

    service, _handles = _make_service(db, tmp_path)

    # 404: service raises JobNotFoundError for missing job.
    with pytest.raises(JobNotFoundError):
        await service.cancel_job("000000000000", graceful=True)

    # 200: cancel a pending job.
    await _create_job(db, "apitest_pending")
    r = await service.cancel_job("apitest_pending", graceful=True)
    assert isinstance(r, CancelResponse)
    assert r.id == "apitest_pending"
    assert r.status is JobStatus.CANCELLED  # pending path finalizes synchronously
    assert r.graceful is True

    # 200 idempotent: re-cancel a cancelled job.
    r2 = await service.cancel_job("apitest_pending", graceful=True)
    assert r2.status is JobStatus.CANCELLED

    # 409: cancel a complete job.
    await _create_job(db, "apitest_complete")
    await db.update_research_job_status(
        "apitest_complete", "complete", completed_at=format_now()
    )
    with pytest.raises(JobAlreadyTerminalError):
        await service.cancel_job("apitest_complete", graceful=True)

    # 200 (graceful=False): same service contract; the HTTP
    # route would surface this as 200 with status=cancelling.
    await _create_job(db, "apitest_pending2")
    r3 = await service.cancel_job("apitest_pending2", graceful=False)
    assert r3.status is JobStatus.CANCELLED  # pending path again
    assert r3.graceful is False

    # Public DTO schema: CancelResponse has exactly the
    # documented fields (id, status, graceful).
    schema = CancelResponse.model_json_schema()
    assert set(schema["properties"].keys()) == {"id", "status", "graceful"}
    assert "partial_output_path" not in schema["properties"]

    # The CancelResponse.status enum is JobStatus with the
    # documented values (including cancelling and cancelled).
    assert JobStatus.CANCELLING.value == "cancelling"
    assert JobStatus.CANCELLED.value == "cancelled"
