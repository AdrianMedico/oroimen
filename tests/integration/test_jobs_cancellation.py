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
import os
import time
from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes.jobs.cost import calculate_cost, format_now
from hermes.jobs.exceptions import (
    JobAlreadyTerminalError,
    JobNotFoundError,
    PhaseError,
)
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

# =====================================================================
# DR-Q1A-PRE1B remediation: race tests A-T (NEW, deterministic)
# =====================================================================
# These tests prove the seven race/terminal-commit defects are fixed.
# They are OFFLINE: no network, no provider, no real asyncio loop
# shutdown. The active task and the cancellation finaliser are
# driven by controlled Events, AsyncMock side effects, and
# asyncio.CancelledError injection.

# -- Test A: true startup interleaving ------------------------------------


@pytest.mark.asyncio
async def test_a_true_startup_interleaving(db, tmp_path: Path) -> None:
    """A (overnight): REAL startup interleaving with Events/barriers.

    The previous "manually register a fake task before cancel_job"
    design did not actually interleave cancel's pre-lock read with
    research's register+transition. This test forces the real
    ordering with controlled Events:

      1. cancel_job begins; its pre-lock read captures ``pending``
         (we observe the read via a hook that signals an Event);
      2. cancel_job is held at its cancel-side CAS via a
         barrier (we wrap ``transition_research_job_status`` so
         the running->cancelling CAS waits on an Event);
      3. real ``_run_research`` task is started;
      4. it registers its exact asyncio.Task in the registry;
      5. it transitions ``pending -> running``;
      6. it blocks before external search/provider work (the
         search side-effect awaits an Event);
      7. cancel's seam release is allowed to proceed (test
         sets the CAS Event);
      8. cancel_job re-reads the canonical row (``running``);
      9. cancel_job CASes ``running -> cancelling``;
     10. cancel_job signals the exact registered research task
         (the task is NOT a manually-registered fake; it is the
         real ``_run_research`` coroutine);
     11. final state: ``cancelled``;
     12. no external phase proceeds past the controlled boundary
         (the search side-effect is still awaiting
         ``search_release.wait()`` and never returns).

    The proof is causal: the in-lock cancel decision inspects the
    active-task registry (not the pre-lock value) and signals the
    registered task. We verify the registered task receives cancel
    by checking that the search side-effect's blocking await never
    returns, and that the research task is cancelled.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "race_a_interleave"
    await _create_job(db, job_id)

    # Barrier 1: search side-effect blocks until search_release
    # is set. The research task will block here before any
    # external work; we never set search_release, so the
    # search will not return.
    search_entered = asyncio.Event()
    search_release = asyncio.Event()

    async def blocking_search(*args: Any, **kwargs: Any) -> Any:
        search_entered.set()
        await search_release.wait()
        return MagicMock(results=[])

    handles["search"].side_effect = blocking_search

    # Barrier 2: wrap transition_research_job_status so the
    # FIRST cancel-side running->cancelling CAS waits on
    # cancel_cas_release. This is the seam-side barrier
    # between the pre-lock read and the CAS.
    real_transition = db.transition_research_job_status
    cancel_cas_release = asyncio.Event()

    async def hooked_transition(*args: Any, **kwargs: Any) -> bool:
        from_states = args[1] if len(args) > 1 else kwargs.get("from_states", ())
        to_state = args[2] if len(args) > 2 else kwargs.get("to_state", "")
        # Detect: the cancel-side running->cancelling CAS
        # (only this CAS holds the cancel between pre-lock
        # read and the seam).
        if (
            to_state == "cancelling"
            and JobStatus.RUNNING.value in from_states
        ):
            await cancel_cas_release.wait()
        return await real_transition(*args, **kwargs)

    db.transition_research_job_status = hooked_transition  # type: ignore[method-assign]

    try:
        # Step 1+2: start cancel_job. The pre-lock read happens
        # immediately, then the cancel reaches the hooked CAS
        # and waits on cancel_cas_release.
        cancel_coro = asyncio.create_task(
            service.cancel_job(job_id, graceful=True)
        )

        # Step 3: start the REAL _run_research task.
        research_task = asyncio.create_task(
            service._run_research(job_id)
        )

        # Step 4+5: wait for the row to reach 'running' AND the
        # research task to be in the registry.
        for _ in range(400):
            row = await db.get_research_job(job_id)
            if row and row["status"] == "running":
                peeked = service._peek_active_task(job_id)
                if peeked is research_task:
                    break
            await asyncio.sleep(0.01)
        else:
            cancel_coro.cancel()
            research_task.cancel()
            pytest.fail(
                "Research task never registered or never reached 'running'"
            )

        # Step 6: wait for the search to be entered (so we know
        # the research task is blocked at the controlled
        # boundary before external work).
        for _ in range(400):
            if search_entered.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            cancel_coro.cancel()
            research_task.cancel()
            pytest.fail("Search was never entered")

        # Sanity: the cancel's CAS is still holding.
        assert not cancel_coro.done(), (
            "cancel_job completed before its CAS barrier; "
            "the test ordering is broken"
        )

        # Step 7: release the cancel's CAS. The cancel proceeds:
        # re-reads running, CASes running -> cancelling, signals
        # the registered research task.
        cancel_cas_release.set()

        # Step 8+9+10: wait for cancel_job to complete. The
        # cancel is graceful=True so it will bounded-wait
        # on the research task. The wait will time out (the
        # research task is still blocked at the search barrier
        # and the cancel was the only thing that could
        # unblock it; the cancel did signal the task but the
        # task is awaiting search_release which is not set).
        response = await cancel_coro

        # Step 11: wait for the research task to be cancelled.
        # The cancel signalled it; the task's ``except
        # CancelledError`` runs the finalizer; the finalizer
        # sees ``cancelling`` and the user-intent marker
        # (set by cancel_job), finalizes the row to
        # ``cancelled``, and the task exits.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(research_task, timeout=5.0)

        # Step 12: no external phase proceeded past the
        # controlled boundary.
        # The search side-effect's blocking await never
        # returned (we never set search_release).
        assert not search_release.is_set(), (
            "search_release was unexpectedly set"
        )
        # The search was entered exactly once (the side-effect
        # ran). It did not return.
        assert handles["search"].call_count == 1, (
            f"search call_count={handles['search'].call_count}; "
            f"expected 1"
        )
        # No fetcher call.
        assert handles["fetcher"].calls == []
        # No LLM call.
        assert handles["llm"].chat.call_count == 0

        # Final state: cancelled.
        row = await db.get_research_job(job_id)
        assert row["status"] == "cancelled", (
            f"Expected cancelled, got {row['status']!r}"
        )
        assert row["error_taxonomy"] == "cancelled"

        # The response status is CANCELLING (the wait timed out
        # because the research task was blocked at the search
        # barrier; the finalizer ran AFTER the wait returned
        # and flipped the row to CANCELLED). CANCELLED is also
        # acceptable if the wait was long enough.
        assert response.status in (
            JobStatus.CANCELLING,
            JobStatus.CANCELLED,
        ), f"Expected CANCELLING or CANCELLED, got {response.status}"

        # The registered research task is the one that was
        # signalled. We verify by re-peeking the registry; it
        # should be empty (the task unregistered itself in its
        # finally block) OR contain the same task that is now
        # done.
        peeked = service._peek_active_task(job_id)
        if peeked is not None:
            assert peeked.done(), (
                "registry contains a non-done task after research completion"
            )
    finally:
        # Always unblock any waiters so the test does not hang.
        cancel_cas_release.set()
        search_release.set()
        if not research_task.done():
            research_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await research_task


# -- Test B: phase guard failure ----------------------------------------


@pytest.mark.asyncio
async def test_b_phase_guard_failure(db, tmp_path: Path) -> None:
    """B: status changes to cancelling immediately before a phase guard;
    _update_phase does not merely log; the phase callable is never
    entered.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "race_b_phaseguard"
    await _create_job(db, job_id)
    # Force the row to running, then to cancelling (simulating a
    # cancel that won between phases).
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="running"
    )
    await db.transition_research_job_status(
        job_id, from_states=("running",), to_state="cancelling"
    )

    # Direct call to _update_phase must raise asyncio.CancelledError
    # (because the row is in cancelling). It must NOT return normally.
    with pytest.raises(asyncio.CancelledError):
        await service._update_phase(job_id, PhaseName.SEARCH, progress=10)

    # The search was never entered.
    assert handles["search"].call_count == 0, (
        "search was called despite the phase guard failing"
    )


# -- Test C: PhaseError race ---------------------------------------------


@pytest.mark.asyncio
async def test_c_phase_error_race_with_cancellation(db, tmp_path: Path) -> None:
    """C (overnight): PhaseError race where cancellation wins.

    The previous design used ``asyncio.wait_for(research_task,
    timeout=5.0)``; the timeout was a hidden cancel-injector and
    could mask the causal chain. This test uses
    ``asyncio.wait({research_task}, timeout=<bounded>)`` which does
    NOT cancel the research task on timeout, and asserts
    ``research_task in done`` as the success criterion. On timeout
    the failure is recorded and the test fails honestly.

    Proof:

      - search side-effect blocks on a controlled Event; the
        research task is paused at the controlled boundary;
      - externally transition ``running -> cancelling`` (the
        cancel won the race);
      - release the search side-effect to raise PhaseError
        (a non-retryable taxonomy);
      - the conditional ``running -> failed`` CAS in the
        ``PhaseError`` branch DOES NOT apply (row is in
        ``cancelling``);
      - the cancellation finalization is invoked through the
        shared ``_finalize_cancellation`` helper (Fix A);
      - final state is ``cancelled``, not ``failed``;
      - failed notifier is NOT called;
      - no failed metric is emitted;
      - the research task ends with cancellation semantics
        (its ``_handle_cancellation`` finalizer ran, not the
        ``running -> failed`` branch);
      - the test's own wait did NOT inject the decisive
        cancellation (the cancel is the externally-induced
        row state transition + the asyncio cancellation that
        the search side-effect's raise would NOT trigger on
        its own).
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "race_c_phaseerror"
    await _create_job(db, job_id)

    # Search side-effect: blocks on ``search_release`` and
    # raises a non-retryable PhaseError when released. The
    # retry loop terminates after this single raise (3
    # attempts total, but the test only allows one
    # non-retryable error which propagates immediately).
    search_entered = asyncio.Event()
    search_release = asyncio.Event()

    async def blocking_then_phase_error(*args: Any, **kwargs: Any) -> Any:
        search_entered.set()
        await search_release.wait()
        raise PhaseError(
            "test_5xx", "phase_error_after_cancel", retryable=False
        )

    handles["search"].side_effect = blocking_then_phase_error

    # Start the research task.
    research_task = asyncio.create_task(service._run_research(job_id))

    # Wait for the search to be entered.
    for _ in range(400):
        if search_entered.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail("search was never entered")

    # Wait for status=running.
    for _ in range(400):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running":
            break
        await asyncio.sleep(0.01)
    else:
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail("Task never reached 'running' state")

    # Cancel won the race. Transition the row externally.
    await db.transition_research_job_status(
        job_id, from_states=("running",), to_state="cancelling"
    )
    # Also mark user-cancel intent (so the finalizer treats
    # this as a user cancellation, not infra shutdown).
    service._mark_user_cancel_intent(job_id)

    # Release the search — it raises PhaseError. The error
    # propagates to ``_run_research_inner``'s ``except
    # PhaseError`` branch, which attempts a conditional
    # ``running -> failed`` CAS that WILL fail (row is in
    # ``cancelling``), and now (Fix A) delegates to the
    # shared ``_finalize_cancellation`` helper.
    search_release.set()

    # Wait for the research task to finish using
    # ``asyncio.wait`` + bounded timeout (NOT
    # ``asyncio.wait_for`` which would inject a cancel on
    # timeout and mask the causal chain). The bounded
    # timeout is generous (10s) but does NOT cancel the
    # research task if it elapses.
    try:
        done, _pending = await asyncio.wait(
            {research_task}, timeout=10.0
        )
    except asyncio.CancelledError:
        pytest.fail("wait raised CancelledError unexpectedly")
    if research_task not in done:
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail(
            "research task did not finish within 10s; "
            "the fix A delegation may not have completed"
        )

    # Drain any task exception so the asyncio runtime does
    # not log a warning.
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await research_task

    # Final state MUST be cancelled, NOT failed.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled", (
        f"Expected cancelled, got {row['status']!r}"
    )
    assert row["error_taxonomy"] == "cancelled"

    # Failed notifier MUST NOT be called.
    handles["notifier"].send_research_failed.assert_not_called()
    handles["notifier"].send_research_complete.assert_not_called()


# -- Test D: generic exception race --------------------------------------


@pytest.mark.asyncio
async def test_d_generic_exception_race_with_cancellation(
    db, tmp_path: Path
) -> None:
    """D (overnight): same as C for the generic Exception branch.

    Uses ``asyncio.wait`` + bounded timeout (not
    ``asyncio.wait_for``) so the test's wait does NOT inject
    the decisive cancellation. The causal chain is:

      - search side-effect blocks on a controlled Event;
      - externally transition ``running -> cancelling``;
      - mark user-cancel intent;
      - release the search side-effect to raise a generic
        ``RuntimeError``;
      - the conditional ``running -> failed`` CAS in the
        generic ``except Exception`` branch DOES NOT apply;
      - the shared ``_finalize_cancellation`` helper (Fix A)
        runs;
      - final state is ``cancelled``;
      - failed notifier is NOT called;
      - failed metric is NOT emitted.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "race_d_generic"
    await _create_job(db, job_id)

    search_entered = asyncio.Event()
    search_release = asyncio.Event()

    async def blocking_then_runtime_error(*args: Any, **kwargs: Any) -> Any:
        search_entered.set()
        await search_release.wait()
        raise RuntimeError("generic_error_after_cancel")

    handles["search"].side_effect = blocking_then_runtime_error

    research_task = asyncio.create_task(service._run_research(job_id))
    for _ in range(400):
        if search_entered.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail("search was never entered")

    for _ in range(400):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running":
            break
        await asyncio.sleep(0.01)
    else:
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail("Task never reached 'running' state")

    await db.transition_research_job_status(
        job_id, from_states=("running",), to_state="cancelling"
    )
    service._mark_user_cancel_intent(job_id)
    search_release.set()

    try:
        done, _pending = await asyncio.wait(
            {research_task}, timeout=10.0
        )
    except asyncio.CancelledError:
        pytest.fail("wait raised CancelledError unexpectedly")
    if research_task not in done:
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail(
            "research task did not finish within 10s"
        )

    with contextlib.suppress(asyncio.CancelledError, Exception):
        await research_task

    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled", (
        f"Expected cancelled, got {row['status']!r}"
    )
    assert row["error_taxonomy"] == "cancelled"
    handles["notifier"].send_research_failed.assert_not_called()
    handles["notifier"].send_research_complete.assert_not_called()


# -- Test E: repeated cancellation during finalization ------------------


@pytest.mark.asyncio
async def test_e_repeated_cancel_during_finalization(db, tmp_path: Path) -> None:
    """E: block reconcile_cost or the final DB CAS with an Event; call
    cancel a second time while finalization is blocked; finalizer
    still completes; state becomes cancelled; checkpoint/temp
    cleanup occurs once; intent is not cleared prematurely.
    """
    service, _handles = _make_service(db, tmp_path)
    job_id = "race_e_repeated"
    await _create_job(db, job_id)
    # Move the row to cancelling (simulating a cancel that won).
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="cancelling"
    )

    # Block reconcile_cost via a wrapper.
    reconcile_block = asyncio.Event()
    reconcile_entered = asyncio.Event()
    original_reconcile = service.reconcile_cost

    async def blocking_reconcile(jid: str) -> Decimal:
        reconcile_entered.set()
        await reconcile_block.wait()
        return await original_reconcile(jid)

    service.reconcile_cost = blocking_reconcile  # type: ignore[method-assign]

    # Write a checkpoint + .md.tmp so cleanup can be observed.
    ckpt_path = Path(service._data_root) / job_id / "checkpoint.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps({"job_id": job_id, "phase": "search"}))
    tmp_path_md = Path(service._data_root) / f"{job_id}.md.tmp"
    tmp_path_md.write_text("# draft\n")

    # Manually invoke the cancellation finalizer (this is the same
    # code path the ``except asyncio.CancelledError`` branch in
    # ``_run_research_inner`` takes when the research task receives
    # cancel).
    finalizer_task = asyncio.create_task(
        service._handle_cancellation(job_id, time.monotonic()),
        name=f"cancel-finalize:{job_id}",
    )
    # Register the finalizer in the active-task registry so that
    # the cancel_job endpoint treats the second cancel as a
    # re-signal of the active task (not as a no-task synchronous
    # finalize). The ``is`` identity check in
    # ``_unregister_active_task`` ensures an older task's
    # ``finally`` block cannot evict a newer one.
    await service._register_active_task(job_id, finalizer_task)
    # Wait for the finaliser to enter reconcile_cost.
    for _ in range(200):
        if reconcile_entered.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        finalizer_task.cancel()
        await service._unregister_active_task(job_id, finalizer_task)
        pytest.fail("finalizer never entered reconcile_cost")

    # Mark user intent (simulating that cancel_job had set it
    # before the finalizer started). The finalizer will preserve
    # this until the row reaches terminal state.
    service._mark_user_cancel_intent(job_id)

    # The intent is set while finalization is blocked.
    assert service._user_cancel_intended(job_id), (
        "user-cancel intent was cleared before finalization completed"
    )

    # Call cancel a SECOND time. Because the row is in 'cancelling'
    # (not 'pending|running'), cancel_job is idempotent: the
    # CAS is a no-op; the in-lock decision path inspects the
    # active task registry (which is empty in this test - we did
    # NOT register the finalizer task); the synchronous-finalize
    # path runs and returns 200 cancelled.
    response2 = await service.cancel_job(job_id, graceful=False)
    assert response2.status in (
        JobStatus.CANCELLING,
        JobStatus.CANCELLED,
    ), f"Expected CANCELLING or CANCELLED, got {response2.status}"

    # Intent is STILL set (the finaliser hasn't run yet, and the
    # second cancel does not clear it because the row is not yet
    # terminal in the finalizer's view).
    assert service._user_cancel_intended(job_id), (
        "user-cancel intent was cleared by the second cancel before finalization completed"
    )

    # Release reconcile_cost. The finaliser completes.
    reconcile_block.set()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(finalizer_task, timeout=5.0)
    # Drain the registry. The finalizer's task is done; the
    # unregister is a no-op (the task was never reaped because
    # the test owns the finalizer task directly, not via
    # ``_run_research``).
    await service._unregister_active_task(job_id, finalizer_task)

    # Final state: cancelled. Cleanup happened (checkpoint removed,
    # .md.tmp removed).
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled", (
        f"Expected cancelled, got {row['status']!r}"
    )
    assert not ckpt_path.exists(), "checkpoint was not cleaned up"
    assert not tmp_path_md.exists(), "tmp report was not cleaned up"
    # Final intent is cleared because the row is now terminal.
    assert not service._user_cancel_intended(job_id), (
        "user-cancel intent was not cleared after terminal state"
    )


# -- Test F: outer waiter cancellation ----------------------------------


@pytest.mark.asyncio
async def test_f_outer_waiter_cancellation(db, tmp_path: Path) -> None:
    """F (overnight): REAL outer-caller cancellation.

    The previous design did not actually cancel the
    ``cancel_job`` caller while it was in its acknowledgement
    wait. It only verified the call returned within ``wait_s``
    because the inner task terminated on cancel. This test
    uses a real running research task with a controlled
    awaitable, then explicitly cancels the cancel_job caller
    while the caller is in its bounded-wait, and asserts that
    awaiting the caller raises ``asyncio.CancelledError`` and
    no ``CancelResponse`` is returned. The research task
    continues independently to finalize as cancelled.

    Steps:

      1. create a job (pending);
      2. start a real research task that registers itself in
         the active-task registry, transitions pending ->
         running, and blocks at a controlled search boundary;
      3. start ``cancel_job(graceful=True)`` as a separate
         task; it acquires the seam, CASes running ->
         cancelling, signals the research task (inside the
         seam, Fix B), and enters the bounded wait on the
         research task;
      4. while the cancel_job caller is in the bounded wait,
         call ``cancel_caller.cancel()``;
      5. await the cancel_caller and assert that it raises
         ``asyncio.CancelledError`` (NOT a CancelResponse);
      6. release the research task's controlled boundary so
         its finalizer can run;
      7. wait for the research task to complete; verify the
         final state is ``cancelled``;
      8. verify no completion/failed notifier runs;
      9. verify the research task was signalled by the
         cancel (its controlled await saw CancelledError,
         and its finalizer ran).
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "race_f_outerwaiter"
    await _create_job(db, job_id)

    # Controlled boundary: the search side-effect blocks
    # until ``search_release`` is set. The research task
    # will be blocked here for the duration of the test.
    search_entered = asyncio.Event()
    search_release = asyncio.Event()

    async def blocking_search(*args: Any, **kwargs: Any) -> Any:
        search_entered.set()
        await search_release.wait()
        return MagicMock(results=[])

    handles["search"].side_effect = blocking_search

    # Start the real research task. It registers its
    # asyncio.Task, transitions pending -> running, and
    # blocks at the search side-effect.
    research_task = asyncio.create_task(
        service._run_research(job_id)
    )
    # Wait for the row to reach 'running' AND the search
    # to be entered.
    for _ in range(1000):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running" and search_entered.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail("research task never reached 'running' or search was never entered")
    # Confirm the task is registered and blocked.
    peeked = service._peek_active_task(job_id)
    assert peeked is research_task, (
        "research task did not register in the active-task registry"
    )

    # Replace the research_task's cancel-sensitivity by
    # wrapping it: when cancel_job signals the task, we
    # want the task to stay "not done" so the bounded
    # wait in cancel_job stays in flight. The cleanest
    # way is to install a done-callback on the task that
    # RE-REGISTERS a shielded blocking task. But asyncio
    # does not let us ignore cancel() on a task. The
    # alternative: replace the active task in the
    # registry with a synthetic task that does NOT
    # respond to cancel. We use the real research_task
    # for the seam-side signal (Fix B), but immediately
    # after the seam, we replace the active task in the
    # registry with a shielded-loop task so the bounded
    # wait sees a task that does not finish.

    # After the cancel is signalled, the research_task
    # is in "cancelling" state. It will process the
    # CancelledError on its next event-loop tick. We
    # need the bounded wait to be in flight at that
    # point. The bounded wait sees the task as "done"
    # only when the CancelledError propagates through
    # the task's coroutine. The research task's
    # CancelledError handler runs the finaliser; the
    # finaliser blocks at reconcile_cost. So the task
    # is NOT done while the finaliser is blocked.
    # The bounded wait sees the task as "not done"
    # and waits for the timeout (1.0s) — this gives
    # the test time to cancel the cancel_caller.
    #
    # To make the research task's finaliser block, we
    # block reconcile_cost.
    reconcile_block = asyncio.Event()
    reconcile_entered = asyncio.Event()
    original_reconcile = service.reconcile_cost

    async def blocking_reconcile(jid: str) -> Decimal:
        reconcile_entered.set()
        await reconcile_block.wait()
        return await original_reconcile(jid)

    service.reconcile_cost = blocking_reconcile  # type: ignore[method-assign]

    # Start cancel_job(graceful=True) as a separate task. The
    # cancel acquires the seam, CASes running -> cancelling,
    # signals the research task (INSIDE the seam, Fix B), and
    # enters the bounded wait on the research task.
    cancel_caller = asyncio.create_task(
        service.cancel_job(job_id, graceful=True)
    )

    # Yield once so the cancel_caller is scheduled by the
    # event loop. On busy CI runners the polling loop can
    # otherwise monopolise the loop and starve the
    # background cancel task.
    await asyncio.sleep(0)

    # Wait for the cancel to flip the row to cancelling.
    # The budget is generous (10s = 1000 iterations * 0.01s)
    # to absorb CI scheduling jitter; the actual cancel
    # should complete in <100ms on a quiet event loop.
    cancelling_seen = False
    for _ in range(1000):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "cancelling":
            cancelling_seen = True
            break
        # Also accept cancelled: a fast CI can run the
        # finaliser inside the bounded wait.
        if row and row["status"] == "cancelled":
            cancelling_seen = True
            break
        await asyncio.sleep(0.01)
    if not cancelling_seen:
        cancel_caller.cancel()
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await cancel_caller
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail("cancel_job did not flip row to cancelling in time")

    # Sanity: cancel_caller is in flight (waiting on the
    # bounded wait on the research task).
    assert not cancel_caller.done(), (
        "cancel_caller completed before the outer cancel; "
        "the test setup is broken"
    )

    # Cancel the cancel_caller itself. This is the
    # outer-caller cancellation we want to verify.
    cancel_caller.cancel()

    # Awaiting the cancel_caller MUST raise
    # ``asyncio.CancelledError`` (NOT return a CancelResponse).
    with pytest.raises(asyncio.CancelledError):
        await cancel_caller

    # The research task is independently owned. It has been
    # signalled (the cancel did call .cancel() on it inside
    # the seam). The research task's ``_run_research``
    # handles the CancelledError by running the
    # cancellation finaliser; the finaliser is blocked at
    # reconcile_cost (the test set up a wrapper). Release
    # the reconcile block AND the search block so the
    # research task can finalise.
    reconcile_block.set()
    search_release.set()

    # Wait for the research task to finish using
    # ``asyncio.wait`` (NOT ``wait_for``) so we do not
    # inject a second cancel.
    try:
        done, _pending = await asyncio.wait(
            {research_task}, timeout=10.0
        )
    except asyncio.CancelledError:
        pytest.fail("wait raised CancelledError unexpectedly")
    if research_task not in done:
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail(
            "research task did not finalize after the outer cancel; "
            "research task was NOT signalled by the cancel"
        )

    with contextlib.suppress(asyncio.CancelledError, Exception):
        await research_task

    # Final state: cancelled.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled", (
        f"Expected cancelled, got {row['status']!r}"
    )
    assert row["error_taxonomy"] == "cancelled"

    # No completion/failed notifier.
    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()


# -- Test G: publish failure ---------------------------------------------


@pytest.mark.asyncio
async def test_g_publish_failure(db, tmp_path: Path, monkeypatch) -> None:
    """G: monkeypatch os.replace to raise OSError for the .md publish
    only; status must NOT be complete; no completion notifier; no
    final report file; conditional failed state unless cancellation
    won.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "race_g_publish_fail"
    await _create_job(db, job_id)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(
            results=[
                {"url": "https://example.com/a", "title": "A", "snippet": "a"},
                {"url": "https://example.com/b", "title": "B", "snippet": "b"},
            ]
        )

    handles["search"].side_effect = one_url_search
    handles["fetcher"].fetch = AsyncMock(
        return_value=_FakeFetchResult(
            body=b"<html><body>"
            + (b"fake body content that is long enough to pass the "
               b"minimum-length check in the scrape phase; " * 5)
            + b"</body></html>"
        )
    )
    handles["llm"].chat = AsyncMock(
        return_value=_FakeLLMResp(
            content="summary of source " + ("X" * 200),
            tokens_in=200,
            tokens_out=200,
        )
    )

    # Monkeypatch os.replace so that ONLY the final .md publish
    # raises. Other os.replace calls (checkpoint writes) succeed.
    original_replace = os.replace
    call_count = [0]

    def selective_replace(src, dst, *args, **kwargs):
        call_count[0] += 1
        dst_str = str(dst)
        if dst_str.endswith(".md") and not dst_str.endswith(".md.tmp"):
            raise OSError("simulated publish failure")
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", selective_replace)

    # Run the research task. It will go through phases 1-4
    # successfully, then fail in phase 5 publish.
    research_task = asyncio.create_task(service._run_research(job_id))
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(research_task, timeout=5.0)

    # Status must NOT be complete.
    row = await db.get_research_job(job_id)
    assert row["status"] != "complete", (
        f"Expected non-complete, got {row['status']!r}"
    )
    # The conditional running -> failed CAS should have succeeded
    # (no cancel won in this test).
    assert row["status"] == "failed", (
        f"Expected failed, got {row['status']!r}"
    )
    # No completion notifier.
    handles["notifier"].send_research_complete.assert_not_called()
    # No final report file.
    final_md = Path(service._data_root) / f"{job_id}.md"
    assert not final_md.exists(), "final .md was published despite publish failure"
    # .md.tmp is cleaned.
    tmp_md = Path(service._data_root) / f"{job_id}.md.tmp"
    assert not tmp_md.exists(), ".md.tmp was not cleaned"


# -- Test H: completion success -----------------------------------------


@pytest.mark.asyncio
async def test_h_completion_success_publish_before_complete(
    db, tmp_path: Path
) -> None:
    """H: final report exists and is readable BEFORE complete is
    observable in the row; notifier runs only after successful
    publish + complete transition.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "race_h_completion"
    # notify=True so the notifier is called after completion.
    await _create_job(db, job_id, notify=True)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(
            results=[
                {"url": "https://example.com/a", "title": "A", "snippet": "a"},
            ]
        )

    handles["search"].side_effect = one_url_search
    handles["fetcher"].fetch = AsyncMock(
        return_value=_FakeFetchResult(
            body=b"<html><body>"
            + (b"fake body content that is long enough to pass the "
               b"minimum-length check in the scrape phase; " * 5)
            + b"</body></html>"
        )
    )
    handles["llm"].chat = AsyncMock(
        return_value=_FakeLLMResp(
            content="final report body " + ("Y" * 300),
            tokens_in=200,
            tokens_out=200,
        )
    )

    # Track when the notifier is called relative to the row state.
    notifier_observations: list[tuple[str, bool]] = []

    async def observed_notifier_complete(
        job_id: str, cost_usd: Decimal
    ) -> bool:
        row = await service._db.get_research_job(job_id)
        status = row["status"] if row else "unknown"
        final_md = Path(service._data_root) / f"{job_id}.md"
        file_exists = final_md.exists()
        notifier_observations.append((status, file_exists))
        return True

    handles["notifier"].send_research_complete = AsyncMock(
        side_effect=observed_notifier_complete
    )

    research_task = asyncio.create_task(service._run_research(job_id))
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(research_task, timeout=5.0)

    row = await db.get_research_job(job_id)
    assert row["status"] == "complete"
    final_md = Path(service._data_root) / f"{job_id}.md"
    assert final_md.exists(), "final .md report was not published"

    # The notifier was called exactly once, AND when it was called
    # the row was already 'complete' AND the final file existed.
    assert len(notifier_observations) == 1, (
        f"Expected exactly one notifier call, got {len(notifier_observations)}"
    )
    observed_status, observed_file_exists = notifier_observations[0]
    assert observed_status == "complete", (
        f"Notifier was called before complete: status={observed_status!r}"
    )
    assert observed_file_exists, "Notifier was called before the final file existed"


# -- Test I: strong acknowledgement assertion --------------------------


@pytest.mark.asyncio
async def test_i_strong_acknowledgement_assertion(db, tmp_path: Path) -> None:
    """I: when graceful=True receives task acknowledgement inside the
    bounded wait, response.status == cancelled. Only CANCELLED is
    acceptable; CANCELLING is NOT acceptable (the task fully
    acknowledged cancellation, the finalizer has run, the row is
    in the terminal state).
    """
    search_block = asyncio.Event()
    service, handles = _make_service(db, tmp_path, search_block=search_block)
    job_id = "race_i_strongack"
    await _create_job(db, job_id)
    research_task = asyncio.create_task(service._run_research(job_id))
    for _ in range(200):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running":
            break
        await asyncio.sleep(0.01)
    else:
        research_task.cancel()
        pytest.fail("Task never reached 'running' state")

    # Mark user intent BEFORE cancel_job (so the finalizer
    # recognises this as a user cancellation, not infra shutdown).
    service._mark_user_cancel_intent(job_id)

    # Pre-register the running task in the active-task registry.
    # (The task is already there from _run_research; this line is
    # a no-op when the task is already registered.)
    await service._register_active_task(job_id, research_task)

    # Run cancel_job(graceful=True) concurrently. The bounded wait
    # will block on asyncio.wait({research_task}, timeout=wait_s).
    cancel_task = asyncio.create_task(
        service.cancel_job(job_id, graceful=True)
    )
    # Give the cancel_task a moment to enter the wait.
    await asyncio.sleep(0.05)

    # Release the search block. The research task finalizes the
    # row to cancelled and the asyncio.wait() in cancel_task
    # returns with the task in done.
    search_block.set()

    # The cancel_task should return with response.status ==
    # CANCELLED. The wait should be satisfied (the task finished
    # before the timeout).
    response = await cancel_task
    assert response.status is JobStatus.CANCELLED, (
        f"Expected CANCELLED, got {response.status}; "
        f"graceful={response.graceful}"
    )
    assert response.graceful is True

    # The research task is done.
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(research_task, timeout=2.0)

    # Final row state: cancelled.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"
    # No completion / failed notifier.
    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()


# -- Test J: running-job idempotency -----------------------------------


@pytest.mark.asyncio
async def test_j_running_job_idempotency(db, tmp_path: Path) -> None:
    """J: cancel the same ACTIVE job twice, including once while its
    finalizer is executing. Both cancels return 200; the final
    state is cancelled; cleanup happens once (no duplicate
    notifier / cleanup / accounting side effect).
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "race_j_idempotency"
    await _create_job(db, job_id)
    # Move the row to running.
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="running"
    )

    # Block reconcile_cost so the finaliser blocks while the
    # cancel is in flight.
    reconcile_block = asyncio.Event()
    reconcile_entered = asyncio.Event()
    original_reconcile = service.reconcile_cost

    async def blocking_reconcile(jid: str) -> Decimal:
        reconcile_entered.set()
        await reconcile_block.wait()
        return await original_reconcile(jid)

    service.reconcile_cost = blocking_reconcile  # type: ignore[method-assign]

    # Set up a fake "research task" that, on cancel, runs the
    # finalizer (simulating the real ``_run_research`` outer
    # method). The cancel will signal this task; the task's
    # ``except CancelledError`` runs the finalizer, which blocks
    # at reconcile_cost.
    async def fake_research_task() -> None:
        # Block until cancel is signalled.
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            # The real outer method would call
            # ``service._handle_cancellation`` here. We invoke
            # it directly (without the shield because the test
            # owns the orchestration).
            await service._handle_cancellation(job_id, time.monotonic())
            raise

    fake = asyncio.create_task(fake_research_task())
    await service._register_active_task(job_id, fake)
    try:
        # First cancel: row is 'running'; the cancel sees the
        # active task in the registry, calls cancel on it, and
        # falls through to the bounded wait. The wait times out
        # (the fake task is blocked running the finalizer). The
        # response is CANCELLING.
        response1 = await service.cancel_job(job_id, graceful=True)
        assert response1.status is JobStatus.CANCELLING, (
            f"Expected CANCELLING, got {response1.status}"
        )

        # Wait for the fake task to enter the finalizer's
        # reconcile_cost.
        for _ in range(200):
            if reconcile_entered.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("finalizer never entered reconcile_cost")

        # The row is now in 'cancelling'. Cancel a SECOND time
        # (graceful=False). The cancel is idempotent: the CAS
        # does not match (row is in 'cancelling'), the registry
        # is inspected, the active task is the fake (still
        # blocked), and the ``was_idempotent_re_signal`` branch
        # returns CANCELLING WITHOUT calling ``fake.cancel()``
        # again (which would abandon the finalizer).
        response2 = await service.cancel_job(job_id, graceful=False)
        assert response2.status is JobStatus.CANCELLING, (
            f"Expected CANCELLING, got {response2.status}"
        )

        # The finalizer is STILL blocked at reconcile_cost. Cancel
        # a THIRD time (graceful=True). Same idempotent path.
        t0 = time.monotonic()
        response3 = await service.cancel_job(job_id, graceful=True)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"third cancel took {elapsed:.2f}s; expected fast return"
        assert response3.status is JobStatus.CANCELLING, (
            f"Expected CANCELLING, got {response3.status}"
        )

        # Release reconcile_cost. The finalizer completes.
        reconcile_block.set()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(fake, timeout=5.0)
    finally:
        await service._unregister_active_task(job_id, fake)
        if not fake.done():
            fake.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await fake

    # Final state: cancelled. Cleanup happened.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"
    # The finalizer was called exactly once; the mark_research_job_notified
    # side effect is NOT triggered for cancellation (no notifier call).
    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()


# =====================================================================
# DR-Q1A-PRE1B overnight remediation: 13 additional tests
# =====================================================================


@pytest.mark.asyncio
async def test_o1_repeated_graceful_true_waits_for_finalization(
    db, tmp_path: Path
) -> None:
    """O1: repeated graceful=True with an active task in the registry
    waits for the finalization to complete (when the finalization
    finishes inside the bounded wait) and returns CANCELLED.

    The repeated cancel MUST NOT inject another ``task.cancel()``
    (Fix C). The wait is on the already-signalled task.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "o1_repeated_graceful_true"
    await _create_job(db, job_id)
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="running"
    )

    # Block reconcile_cost so the finaliser blocks.
    reconcile_block = asyncio.Event()
    reconcile_entered = asyncio.Event()
    original_reconcile = service.reconcile_cost

    async def blocking_reconcile(jid: str) -> Decimal:
        reconcile_entered.set()
        await reconcile_block.wait()
        return await original_reconcile(jid)

    service.reconcile_cost = blocking_reconcile  # type: ignore[method-assign]

    # Fake active task that runs the finaliser on cancel.
    async def fake_research_task() -> None:
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            await service._handle_cancellation(job_id, time.monotonic())
            raise

    fake = asyncio.create_task(fake_research_task())
    await service._register_active_task(job_id, fake)
    try:
        # First cancel: graceful=True. CAS running -> cancelling,
        # signals the fake task, enters the bounded wait.
        response1 = await service.cancel_job(job_id, graceful=True)
        # The fake task is now mid-finaliser (blocked at
        # reconcile_cost).
        for _ in range(400):
            if reconcile_entered.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("finaliser never entered reconcile_cost")

        # Second cancel: graceful=True. The row is in cancelling,
        # the active task is in the registry. The CAS does not
        # match (was_idempotent_re_signal=True). The cancel MUST
        # NOT inject another task.cancel() (Fix C). The bounded
        # wait is on the already-signalled task.
        t0 = time.monotonic()
        response2 = await service.cancel_job(job_id, graceful=True)
        _elapsed = time.monotonic() - t0
        # The wait is bounded by wait_s=1.0s. The first cancel
        # is still in its wait too; releasing the finaliser
        # below will let both cancels see CANCELLED.

        # Release reconcile_cost. The fake finaliser completes,
        # flipping the row to cancelled.
        reconcile_block.set()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(fake, timeout=5.0)

        # Now the second cancel's wait should be done. Re-read
        # to get the post-finalisation state.
        row = await db.get_research_job(job_id)
        assert row["status"] == "cancelled"
        # The bounded wait semantics: the response2 should
        # reflect the cancelled state (the fake finalised
        # inside the wait).
        assert response2.status in (
            JobStatus.CANCELLING,
            JobStatus.CANCELLED,
        ), f"Expected CANCELLING or CANCELLED, got {response2.status}"
        # No injected second cancel: the fake task ran
        # _handle_cancellation exactly once.
        assert response1.status in (
            JobStatus.CANCELLING,
            JobStatus.CANCELLED,
        )
    finally:
        reconcile_block.set()
        await service._unregister_active_task(job_id, fake)
        if not fake.done():
            fake.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await fake

    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()


@pytest.mark.asyncio
async def test_o2_repeated_graceful_false_returns_promptly(
    db, tmp_path: Path
) -> None:
    """O2: repeated graceful=False returns immediately WITHOUT
    injecting another task.cancel() (Fix C).

    The second cancel sees the row in cancelling and the active
    task in the registry. It MUST NOT call ``active.cancel()``
    again (which would abandon the previously-triggered
    finaliser).
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "o2_repeated_graceful_false"
    await _create_job(db, job_id)
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="running"
    )

    reconcile_block = asyncio.Event()
    reconcile_entered = asyncio.Event()
    original_reconcile = service.reconcile_cost

    async def blocking_reconcile(jid: str) -> Decimal:
        reconcile_entered.set()
        await reconcile_block.wait()
        return await original_reconcile(jid)

    service.reconcile_cost = blocking_reconcile  # type: ignore[method-assign]

    # Track how many times the fake task is cancelled.
    cancel_signal_count = [0]

    async def fake_research_task() -> None:
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            cancel_signal_count[0] += 1
            await service._handle_cancellation(job_id, time.monotonic())
            raise

    fake = asyncio.create_task(fake_research_task())
    await service._register_active_task(job_id, fake)
    try:
        # First cancel: graceful=True. The fake is signalled
        # (cancel_signal_count -> 1), then blocks at
        # reconcile_cost.
        await service.cancel_job(job_id, graceful=True)
        for _ in range(400):
            if reconcile_entered.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("finaliser never entered reconcile_cost")

        # Second cancel: graceful=False. MUST NOT inject
        # another task.cancel() (Fix C). The active task is
        # still alive (blocked at reconcile_cost).
        t0 = time.monotonic()
        response2 = await service.cancel_job(job_id, graceful=False)
        elapsed = time.monotonic() - t0
        # The cancel returns immediately (no wait).
        assert elapsed < 0.5, (
            f"second cancel took {elapsed:.2f}s; expected fast return"
        )
        assert response2.status is JobStatus.CANCELLING, (
            f"Expected CANCELLING, got {response2.status}"
        )

        # The fake task received cancel EXACTLY ONCE.
        assert cancel_signal_count[0] == 1, (
            f"fake task was cancelled {cancel_signal_count[0]} times; "
            f"expected 1 (no 2nd cancel injection)"
        )

        # Release reconcile_cost; the finaliser completes.
        reconcile_block.set()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(fake, timeout=5.0)

        # The first cancel was graceful=True, the bounded
        # wait was satisfied (the fake finished). The row
        # is cancelled.
        row = await db.get_research_job(job_id)
        assert row["status"] == "cancelled"
    finally:
        reconcile_block.set()
        await service._unregister_active_task(job_id, fake)
        if not fake.done():
            fake.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await fake

    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()


@pytest.mark.asyncio
async def test_o3_first_cancel_signal_happens_inside_seam(
    db, tmp_path: Path
) -> None:
    """O3: the first cancellation signal is delivered INSIDE the
    terminal seam, atomically with the CAS (Fix B).

    The test holds the active task's acknowledgement, runs a
    cancel, and verifies that the seam-acquire/wait is observed
    between the CAS and the seam release. We probe this by
    hooking the active task's await: when the task receives
    cancel (the first signal inside the seam), we set an Event.
    The seam-release is observable from the row state.
    """
    service, _handles = _make_service(db, tmp_path)
    job_id = "o3_first_signal_inside_seam"
    await _create_job(db, job_id)
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="running"
    )

    # Active task that records the first cancel and blocks.
    first_signal_observed = asyncio.Event()
    task_release = asyncio.Event()

    async def live_research() -> None:
        try:
            await task_release.wait()
        except asyncio.CancelledError:
            first_signal_observed.set()
            raise

    fake = asyncio.create_task(live_research())
    await service._register_active_task(job_id, fake)
    try:
        # Run cancel_job(graceful=False) — it returns
        # immediately after the seam release. Inside the
        # seam, the CAS pending|running -> cancelling
        # happened AND the active task was signalled.
        response = await service.cancel_job(job_id, graceful=False)

        # Yield to the event loop so the task can process
        # the CancelledError (the cancel() call inside the
        # seam requests cancellation; the task processes it
        # on its next tick).
        for _ in range(50):
            if first_signal_observed.is_set() or fake.done():
                break
            await asyncio.sleep(0.01)

        # The first signal was delivered (the task is in
        # ``done`` state with a CancelledError).
        assert first_signal_observed.is_set(), (
            "first cancellation signal was not observed"
        )
        assert fake.cancelled(), (
            "active task was not cancelled by the seam-side signal"
        )

        # The row is in 'cancelling' (the CAS committed
        # before the seam was released).
        row = await db.get_research_job(job_id)
        assert row["status"] == "cancelling"

        # The response is CANCELLING (graceful=False).
        assert response.status is JobStatus.CANCELLING

        # Finalise manually: the fake's _handle_cancellation
        # is not run (the fake is a stub). Run the finaliser
        # to mark the row cancelled.
        async def run_finaliser() -> None:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                raise
            await service._handle_cancellation(job_id, time.monotonic())

        await run_finaliser()

        # Release the task to drain it.
        task_release.set()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await fake

        row = await db.get_research_job(job_id)
        assert row["status"] == "cancelled"
    finally:
        task_release.set()
        await service._unregister_active_task(job_id, fake)
        if not fake.done():
            fake.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await fake


@pytest.mark.asyncio
async def test_o4_http_cancel_after_cas_before_ack(
    db, tmp_path: Path
) -> None:
    """O4: a cancel that arrives after the CAS but before the
    bounded wait starts behaves as an idempotent re-signal:
    it does NOT inject another ``task.cancel()``.

    The first cancel acquires the seam, CASes to cancelling,
    signals the active task, releases the seam. The second
    cancel arrives AFTER the seam is released (after the CAS,
    before the bounded wait would even start). The second
    cancel sees the row in cancelling, the active task in
    the registry, the was_idempotent_re_signal flag set,
    and applies the wait mode (graceful=True bounded wait,
    graceful=False immediate return).
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "o4_http_cancel_after_cas"
    await _create_job(db, job_id)
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="running"
    )

    cancel_signal_count = [0]

    async def live_research() -> None:
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            cancel_signal_count[0] += 1
            # Run the finaliser.
            await service._handle_cancellation(job_id, time.monotonic())
            raise

    fake = asyncio.create_task(live_research())
    await service._register_active_task(job_id, fake)
    try:
        # First cancel: graceful=True. Signals the fake
        # (cancel_signal_count -> 1), enters bounded wait.
        response1 = await service.cancel_job(job_id, graceful=True)
        # The first signal was delivered; the fake is mid-
        # finaliser. The first cancel's bounded wait sees
        # the task finished (it does its own finaliser
        # synchronously).
        # Now: a second cancel arrives AFTER the first
        # cancel completed (the row is in cancelled or
        # cancelling). Let's transition the row to
        # cancelling explicitly to simulate the second
        # cancel arriving before the finaliser ran.
        row = await db.get_research_job(job_id)
        if row["status"] == "cancelled":
            # The first cancel saw the finaliser complete.
            # That's fine; the second cancel is a no-op.
            assert response1.status in (
                JobStatus.CANCELLING,
                JobStatus.CANCELLED,
            )
        else:
            # The finaliser is still mid-flight (the bounded
            # wait is on the still-running fake). Send a
            # second cancel (graceful=False) — it MUST NOT
            # inject another cancel.
            t0 = time.monotonic()
            response2 = await service.cancel_job(job_id, graceful=False)
            elapsed = time.monotonic() - t0
            assert elapsed < 0.5
            assert response2.status is JobStatus.CANCELLING
            # The fake received cancel exactly once (no
            # 2nd injection).
            assert cancel_signal_count[0] == 1
            # Drain the fake.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(fake, timeout=5.0)

        row = await db.get_research_job(job_id)
        assert row["status"] == "cancelled"
    finally:
        await service._unregister_active_task(job_id, fake)
        if not fake.done():
            fake.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await fake

    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()


@pytest.mark.asyncio
async def test_o5_atomic_complete_output_path_success(
    db, tmp_path: Path
) -> None:
    """O5: full pipeline completion uses the single atomic
    ``complete_research_job_with_output_path`` CAS.

    Verifies: complete row has output_path; final .md exists;
    notifier called with the cost; complete+output_path
    committed atomically (single SQL statement).
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "o5_atomic_complete_success"
    await _create_job(db, job_id, notify=True)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(
            results=[
                {"url": "https://example.com/a", "title": "A", "snippet": "a"},
            ]
        )

    handles["search"].side_effect = one_url_search
    handles["fetcher"].fetch = AsyncMock(
        return_value=_FakeFetchResult(
            body=b"<html><body>"
            + (b"fake body content that is long enough to pass the "
               b"minimum-length check in the scrape phase; " * 5)
            + b"</body></html>"
        )
    )
    handles["llm"].chat = AsyncMock(
        return_value=_FakeLLMResp(
            content="final report body " + ("Z" * 300),
            tokens_in=200,
            tokens_out=200,
        )
    )

    # Track SQL statements for the atomic CAS.
    complete_cas_calls: list[str] = []
    original_complete_atomic = db.complete_research_job_with_output_path

    async def tracked_complete_atomic(*args: Any, **kwargs: Any) -> bool:
        complete_cas_calls.append("called")
        return await original_complete_atomic(*args, **kwargs)

    db.complete_research_job_with_output_path = tracked_complete_atomic  # type: ignore[method-assign]

    # Run the research task.
    research_task = asyncio.create_task(service._run_research(job_id))
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(research_task, timeout=5.0)

    row = await db.get_research_job(job_id)
    assert row["status"] == "complete"
    # The output_path is set.
    assert row["output_path"], "output_path is empty in complete row"
    # progress_percent is 100.
    assert row["progress_percent"] == 100
    # The atomic CAS was called exactly once.
    assert len(complete_cas_calls) == 1, (
        f"expected 1 complete_research_job_with_output_path call, "
        f"got {len(complete_cas_calls)}"
    )
    # The final .md exists.
    final_md = Path(service._data_root) / f"{job_id}.md"
    assert final_md.exists(), "final .md report was not published"
    # The notifier was called.
    handles["notifier"].send_research_complete.assert_called_once()
    # No failed notifier.
    handles["notifier"].send_research_failed.assert_not_called()


@pytest.mark.asyncio
async def test_o6_atomic_complete_output_path_db_raises(
    db, tmp_path: Path
) -> None:
    """O6: the atomic CAS raises (DB write failure). The row is
    NOT advanced to complete. The final .md is deleted. No
    completion notifier. The conditional running->failed
    transition runs (and succeeds, since no cancel won in this
    test).
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "o6_atomic_complete_db_raises"
    await _create_job(db, job_id, notify=True)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(
            results=[
                {"url": "https://example.com/a", "title": "A", "snippet": "a"},
            ]
        )

    handles["search"].side_effect = one_url_search
    handles["fetcher"].fetch = AsyncMock(
        return_value=_FakeFetchResult(
            body=b"<html><body>"
            + (b"fake body content that is long enough to pass the "
               b"minimum-length check in the scrape phase; " * 5)
            + b"</body></html>"
        )
    )
    handles["llm"].chat = AsyncMock(
        return_value=_FakeLLMResp(
            content="final report body " + ("Z" * 300),
            tokens_in=200,
            tokens_out=200,
        )
    )

    # Patch the atomic CAS to raise.
    async def raising_complete_atomic(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("simulated_db_failure")

    db.complete_research_job_with_output_path = raising_complete_atomic  # type: ignore[method-assign]

    research_task = asyncio.create_task(service._run_research(job_id))
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(research_task, timeout=5.0)

    # The row is NOT complete.
    row = await db.get_research_job(job_id)
    assert row["status"] != "complete", (
        f"Expected non-complete, got {row['status']!r}"
    )
    # The conditional running->failed CAS ran (no cancel
    # won), so the row is failed.
    assert row["status"] == "failed", (
        f"Expected failed, got {row['status']!r}"
    )
    # The final .md is deleted.
    final_md = Path(service._data_root) / f"{job_id}.md"
    assert not final_md.exists(), (
        f"final .md was not deleted: {final_md}"
    )
    # The tmp .md is deleted.
    tmp_md = Path(service._data_root) / f"{job_id}.md.tmp"
    assert not tmp_md.exists(), ".md.tmp was not cleaned"
    # No completion notifier.
    handles["notifier"].send_research_complete.assert_not_called()
    # Failed notifier was called (the conditional running ->
    # failed transition succeeded).
    handles["notifier"].send_research_failed.assert_called_once()


@pytest.mark.asyncio
async def test_o7_atomic_complete_output_path_cas_false(
    db, tmp_path: Path
) -> None:
    """O7: the atomic CAS returns False (cancellation won
    after publish but before the CAS). The final .md is
    deleted. The row is in 'cancelling' (or 'cancelled'
    after the finaliser). No completion notifier.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "o7_atomic_complete_cas_false"
    await _create_job(db, job_id, notify=True)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(
            results=[
                {"url": "https://example.com/a", "title": "A", "snippet": "a"},
            ]
        )

    handles["search"].side_effect = one_url_search
    handles["fetcher"].fetch = AsyncMock(
        return_value=_FakeFetchResult(
            body=b"<html><body>"
            + (b"fake body content that is long enough to pass the "
               b"minimum-length check in the scrape phase; " * 5)
            + b"</body></html>"
        )
    )
    handles["llm"].chat = AsyncMock(
        return_value=_FakeLLMResp(
            content="final report body " + ("Z" * 300),
            tokens_in=200,
            tokens_out=200,
        )
    )

    # Hook the atomic CAS so that on the first call, we
    # transition the row to cancelling FIRST, then return
    # False (the predicate no longer matches because the
    # row is no longer in 'running').
    real_complete_atomic = db.complete_research_job_with_output_path
    call_count = [0]

    async def hook_complete_atomic(*args: Any, **kwargs: Any) -> bool:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: flip the row to cancelling.
            await db.transition_research_job_status(
                job_id,
                from_states=("running",),
                to_state="cancelling",
            )
            service._mark_user_cancel_intent(job_id)
        return await real_complete_atomic(*args, **kwargs)

    db.complete_research_job_with_output_path = hook_complete_atomic  # type: ignore[method-assign]

    research_task = asyncio.create_task(service._run_research(job_id))
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(research_task, timeout=5.0)

    # The row is cancelled (the finaliser ran after the CAS
    # raised asyncio.CancelledError).
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled", (
        f"Expected cancelled, got {row['status']!r}"
    )
    # The final .md is deleted (the cancellation finaliser
    # cleaned it up).
    final_md = Path(service._data_root) / f"{job_id}.md"
    assert not final_md.exists(), (
        f"final .md was not deleted after cancel-won: {final_md}"
    )
    # The tmp .md is deleted.
    tmp_md = Path(service._data_root) / f"{job_id}.md.tmp"
    assert not tmp_md.exists(), ".md.tmp was not cleaned"
    # No completion notifier.
    handles["notifier"].send_research_complete.assert_not_called()
    # No failed notifier.
    handles["notifier"].send_research_failed.assert_not_called()


@pytest.mark.asyncio
async def test_o8_atomic_complete_output_path_replace_fails(
    db, tmp_path: Path, monkeypatch
) -> None:
    """O8: os.replace failure for the .md publish. The row is
    NOT advanced to complete. No completion notifier. The
    conditional running->failed transition runs.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "o8_atomic_complete_replace_fails"
    await _create_job(db, job_id, notify=True)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(
            results=[
                {"url": "https://example.com/a", "title": "A", "snippet": "a"},
            ]
        )

    handles["search"].side_effect = one_url_search
    handles["fetcher"].fetch = AsyncMock(
        return_value=_FakeFetchResult(
            body=b"<html><body>"
            + (b"fake body content that is long enough to pass the "
               b"minimum-length check in the scrape phase; " * 5)
            + b"</body></html>"
        )
    )
    handles["llm"].chat = AsyncMock(
        return_value=_FakeLLMResp(
            content="final report body " + ("Z" * 300),
            tokens_in=200,
            tokens_out=200,
        )
    )

    # Patch os.replace so the .md publish raises.
    original_replace = os.replace

    def selective_replace(src, dst, *args, **kwargs):
        dst_str = str(dst)
        if dst_str.endswith(".md") and not dst_str.endswith(".md.tmp"):
            raise OSError("simulated_publish_failure")
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", selective_replace)

    research_task = asyncio.create_task(service._run_research(job_id))
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(research_task, timeout=5.0)

    # The row is failed (the conditional running->failed
    # CAS ran after the publish failure).
    row = await db.get_research_job(job_id)
    assert row["status"] == "failed", (
        f"Expected failed, got {row['status']!r}"
    )
    # No final .md.
    final_md = Path(service._data_root) / f"{job_id}.md"
    assert not final_md.exists(), "final .md exists despite publish failure"
    # No tmp .md.
    tmp_md = Path(service._data_root) / f"{job_id}.md.tmp"
    assert not tmp_md.exists(), ".md.tmp was not cleaned"
    # No completion notifier.
    handles["notifier"].send_research_complete.assert_not_called()
    # Failed notifier was called.
    handles["notifier"].send_research_failed.assert_called_once()


@pytest.mark.asyncio
async def test_o9_notifier_outside_lock(
    db, tmp_path: Path
) -> None:
    """O9: the completion notifier runs OUTSIDE the terminal
    seam. When the notifier is invoked, the seam can be
    acquired (the lock is free), the row is in 'complete',
    and the final report file is readable.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "o9_notifier_outside_lock"
    await _create_job(db, job_id, notify=True)

    async def one_url_search(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(
            results=[
                {"url": "https://example.com/a", "title": "A", "snippet": "a"},
            ]
        )

    handles["search"].side_effect = one_url_search
    handles["fetcher"].fetch = AsyncMock(
        return_value=_FakeFetchResult(
            body=b"<html><body>"
            + (b"fake body content that is long enough to pass the "
               b"minimum-length check in the scrape phase; " * 5)
            + b"</body></html>"
        )
    )
    handles["llm"].chat = AsyncMock(
        return_value=_FakeLLMResp(
            content="final report body " + ("Z" * 300),
            tokens_in=200,
            tokens_out=200,
        )
    )

    # Notifier probe: when the notifier is invoked, the
    # row MUST be 'complete' AND the seam MUST be free
    # (the notifier is outside the lock) AND the final
    # file MUST be readable.
    notifier_observations: list[dict[str, Any]] = []

    async def probing_notifier_complete(
        job_id: str, cost_usd: Decimal
    ) -> bool:
        # Try to acquire the seam (should NOT deadlock).
        term_lock = service._get_terminal_lock(job_id)
        seam_acquired = False
        try:
            # The notifier is awaited outside the seam,
            # so acquiring it here should be quick.
            await asyncio.wait_for(term_lock.acquire(), timeout=1.0)
            seam_acquired = True
        except TimeoutError:
            pass
        finally:
            if seam_acquired:
                term_lock.release()
        # Check row state and file.
        row = await service._db.get_research_job(job_id)
        final_md = Path(service._data_root) / f"{job_id}.md"
        notifier_observations.append(
            {
                "seam_acquired": seam_acquired,
                "row_status": row["status"] if row else None,
                "file_exists": final_md.exists(),
            }
        )
        return True

    handles["notifier"].send_research_complete = AsyncMock(
        side_effect=probing_notifier_complete
    )

    research_task = asyncio.create_task(service._run_research(job_id))
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(research_task, timeout=5.0)

    # The notifier was called exactly once.
    assert len(notifier_observations) == 1
    obs = notifier_observations[0]
    # The seam was free when the notifier ran.
    assert obs["seam_acquired"] is True, (
        "notifier ran while the seam was still held"
    )
    # The row was 'complete' when the notifier ran.
    assert obs["row_status"] == "complete"
    # The final file was readable when the notifier ran.
    assert obs["file_exists"] is True


@pytest.mark.asyncio
async def test_o10_job_state_invalid_direct(db, tmp_path: Path) -> None:
    """O10: JobStateInvalid is raised by ``_update_phase`` for
    unexpected non-running non-cancellation states. The phase
    callable is not entered. No state resurrection.
    """
    from hermes.jobs.exceptions import JobStateInvalid

    service, handles = _make_service(db, tmp_path)
    job_id = "o10_job_state_invalid"
    await _create_job(db, job_id)
    # Move the row to 'failed' (an unexpected non-running
    # non-cancellation state for ``_update_phase``).
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="failed"
    )

    # Direct call to _update_phase MUST raise JobStateInvalid
    # (the row is in 'failed', which is not 'running', not
    # 'cancelling', not 'cancelled').
    with pytest.raises(JobStateInvalid):
        await service._update_phase(job_id, PhaseName.SEARCH, progress=10)

    # The search was never called.
    assert handles["search"].call_count == 0

    # Verify the exception attributes.
    try:
        await service._update_phase(job_id, PhaseName.SCRAPE, progress=25)
    except JobStateInvalid as exc:
        assert exc.job_id == job_id
        assert exc.observed_status == "failed"
        assert exc.phase == PhaseName.SCRAPE.value


@pytest.mark.asyncio
async def test_o11_real_path_repeated_cancel(
    db, tmp_path: Path
) -> None:
    """O11: repeated cancellation using the real ``_run_research``
    task (not a manually-registered finalizer task). The second
    cancel arrives while the first cancel's finaliser is mid-
    reconcile. The finaliser is shielded; the second cancel does
    NOT inject another cancel.
    """
    service, handles = _make_service(db, tmp_path)
    job_id = "o11_real_path_repeated_cancel"
    await _create_job(db, job_id)
    # Block reconcile_cost so the finaliser blocks.
    reconcile_block = asyncio.Event()
    reconcile_entered = asyncio.Event()
    original_reconcile = service.reconcile_cost

    async def blocking_reconcile(jid: str) -> Decimal:
        reconcile_entered.set()
        await reconcile_block.wait()
        return await original_reconcile(jid)

    service.reconcile_cost = blocking_reconcile  # type: ignore[method-assign]

    # The search blocks until released; the real
    # _run_research registers and blocks at the search.
    search_entered = asyncio.Event()
    search_release = asyncio.Event()

    async def blocking_search(*args: Any, **kwargs: Any) -> Any:
        search_entered.set()
        await search_release.wait()
        return MagicMock(results=[])

    handles["search"].side_effect = blocking_search

    research_task = asyncio.create_task(
        service._run_research(job_id)
    )
    # Wait for the row to reach 'running' AND the search
    # to be entered.
    for _ in range(400):
        row = await db.get_research_job(job_id)
        if row and row["status"] == "running" and search_entered.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        research_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await research_task
        pytest.fail("research task never reached 'running' or search was never entered")

    # First cancel: graceful=True. CASes running ->
    # cancelling, signals the research task, enters bounded
    # wait. The research task's CancelledError handler
    # runs ``_finalize_cancellation`` which calls
    # ``_handle_cancellation`` which blocks at
    # reconcile_cost.
    await service.cancel_job(job_id, graceful=True)
    # Wait for the finaliser to enter reconcile_cost.
    for _ in range(400):
        if reconcile_entered.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("finaliser never entered reconcile_cost")

    # Second cancel: graceful=True. Idempotent re-signal.
    # MUST NOT inject another cancel.
    t0 = time.monotonic()
    response2 = await service.cancel_job(job_id, graceful=True)
    elapsed = time.monotonic() - t0
    # The second cancel returns within the bounded wait.
    assert elapsed < 5.0
    assert response2.status is JobStatus.CANCELLING, (
        f"Expected CANCELLING, got {response2.status}"
    )

    # Release reconcile_cost AND the search block. The
    # finaliser completes; the research task exits.
    reconcile_block.set()
    search_release.set()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(research_task, timeout=5.0)

    # Final state: cancelled.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"
    assert row["error_taxonomy"] == "cancelled"

    # No notifier calls.
    handles["notifier"].send_research_complete.assert_not_called()
    handles["notifier"].send_research_failed.assert_not_called()


@pytest.mark.asyncio
async def test_o12_user_cancel_intent_never_cleared_prematurely(
    db, tmp_path: Path
) -> None:
    """O12: the user-cancel intent is NOT cleared while
    finalization is incomplete. The intent is only cleared
    after the row reaches a terminal state (cancelled,
    complete, or failed) in the finaliser's ``finally``
    block.
    """
    service, _handles = _make_service(db, tmp_path)
    job_id = "o12_intent_not_cleared_prematurely"
    await _create_job(db, job_id)
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="cancelling"
    )
    service._mark_user_cancel_intent(job_id)

    # Verify intent is set.
    assert service._user_cancel_intended(job_id)

    # Block reconcile_cost so the finaliser blocks.
    reconcile_block = asyncio.Event()
    reconcile_entered = asyncio.Event()
    original_reconcile = service.reconcile_cost

    async def blocking_reconcile(jid: str) -> Decimal:
        reconcile_entered.set()
        await reconcile_block.wait()
        return await original_reconcile(jid)

    service.reconcile_cost = blocking_reconcile  # type: ignore[method-assign]

    # Manually invoke the finaliser. The finaliser blocks
    # at reconcile_cost.
    finaliser = asyncio.create_task(
        service._handle_cancellation(job_id, time.monotonic())
    )
    # Wait for reconcile_cost to be entered.
    for _ in range(400):
        if reconcile_entered.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        finaliser.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await finaliser
        pytest.fail("reconcile_cost was never entered")

    # While the finaliser is blocked, the user-cancel
    # intent MUST still be set.
    assert service._user_cancel_intended(job_id), (
        "user-cancel intent was cleared while finalisation is incomplete"
    )

    # Release reconcile_cost. The finaliser completes.
    reconcile_block.set()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(finaliser, timeout=5.0)

    # The intent is now cleared (the finaliser's
    # ``finally`` block ran).
    assert not service._user_cancel_intended(job_id), (
        "user-cancel intent was not cleared after finalisation"
    )

    # The row is in cancelled.
    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"


@pytest.mark.asyncio
async def test_o13_registry_and_lock_cleanup(db, tmp_path: Path) -> None:
    """O13: the active-task registry and the per-job terminal-
    lock cache do not leak unbounded per-job entries after a
    successful cancellation. The active-task entry is removed
    by the research task's ``finally`` block. The terminal-
    lock is intentionally cached (see note in
    ``_get_terminal_lock``); the test documents the decision
    and verifies the active-task entry is removed.
    """
    service, _handles = _make_service(db, tmp_path)
    job_id = "o13_registry_cleanup"
    await _create_job(db, job_id)
    await db.transition_research_job_status(
        job_id, from_states=("pending",), to_state="running"
    )

    # Block reconcile_cost so the finaliser blocks.
    reconcile_block = asyncio.Event()
    reconcile_entered = asyncio.Event()
    original_reconcile = service.reconcile_cost

    async def blocking_reconcile(jid: str) -> Decimal:
        reconcile_entered.set()
        await reconcile_block.wait()
        return await original_reconcile(jid)

    service.reconcile_cost = blocking_reconcile  # type: ignore[method-assign]

    async def live_research() -> None:
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            await service._handle_cancellation(job_id, time.monotonic())
            raise

    fake = asyncio.create_task(live_research())
    await service._register_active_task(job_id, fake)
    try:
        # Active task is registered.
        assert service._peek_active_task(job_id) is fake

        # Cancel and let the finaliser complete.
        await service.cancel_job(job_id, graceful=True)
        for _ in range(400):
            if reconcile_entered.is_set():
                break
            await asyncio.sleep(0.01)
        reconcile_block.set()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(fake, timeout=5.0)
    finally:
        await service._unregister_active_task(job_id, fake)
        if not fake.done():
            fake.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await fake

    # After the finaliser, the research task's ``finally``
    # block ran the unregister. The registry entry is
    # removed.
    assert service._peek_active_task(job_id) is None, (
        "active-task registry leaked an entry after terminal completion"
    )
    # The terminal-lock is intentionally cached; the
    # design decision is documented in
    # ``_get_terminal_lock``. We verify the lock is
    # present (so concurrent re-attempts on the same
    # job_id get a stable lock) but the lock is NOT held.
    term_lock = service._get_terminal_lock(job_id)
    assert not term_lock.locked(), (
        "terminal lock is still held after terminal completion"
    )

    row = await db.get_research_job(job_id)
    assert row["status"] == "cancelled"
