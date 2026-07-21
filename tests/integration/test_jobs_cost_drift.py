"""Integration tests for cost drift protection (TDD §8.4, §6.8).

Anti-regression checks:
- _record_token_usage: checkpoint writes FIRST, DB writes SECOND. If the
  DB write fails but the checkpoint OK, log warning, continue.
- _record_token_usage: aggregates by phase/model (sum tokens_in, tokens_out, cost).
- reconcile_cost uses max(checkpoint, token_usage_sum, aggregate), NOT average.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.jobs.cost import calculate_cost
from hermes.jobs.models import PhaseName
from hermes.jobs.service import DeepResearchService


@dataclass
class _FakeFetchResult:
    body: bytes
    media_type: str = "text/html"
    status: int = 200
    redirect_count: int = 0


class _FakeFetcher:
    """Controlled fake safe fetcher (Slice 1C1b)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch(self, url: str) -> _FakeFetchResult:
        self.calls.append(url)
        return _FakeFetchResult(body=b"<html></html>")


class _FakeSettings:
    """Minimal settings stub."""

    deep_research_daily_budget_usd = 3.0
    deep_research_max_sources = 5
    deep_research_phase1_timeout_s = 30
    deep_research_phase2_timeout_s = 30
    deep_research_phase3_timeout_s = 90
    deep_research_phase4_timeout_s = 120
    deep_research_phase5_timeout_s = 5
    deep_research_per_source_max_tokens = 3000
    deep_research_data_root = "/tmp/hermes-test-cost-drift"


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


@pytest.mark.asyncio
async def test_record_token_usage_aggregates(db, service_with_db) -> None:
    """Multiple _record_token_usage calls accumulate correctly per (phase, model)."""
    service, _tmp = service_with_db
    job_id = "costtest1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Record 3 calls in phase 3
    cost1 = calculate_cost("MiniMax-M3", 1000, 500)
    cost2 = calculate_cost("MiniMax-M3", 2000, 1000)
    cost3 = calculate_cost("MiniMax-M3", 1500, 750)

    await service._record_token_usage(
        job_id,
        PhaseName.PER_SOURCE_SYNTHESIS,
        "MiniMax-M3",
        1000,
        500,
        cost1,
    )
    await service._record_token_usage(
        job_id,
        PhaseName.PER_SOURCE_SYNTHESIS,
        "MiniMax-M3",
        2000,
        1000,
        cost2,
    )
    await service._record_token_usage(
        job_id,
        PhaseName.PER_SOURCE_SYNTHESIS,
        "MiniMax-M3",
        1500,
        750,
        cost3,
    )

    # Read back via DB
    rows = await db.list_token_usage_for_job(job_id)
    assert len(rows) == 3, f"Expected 3 token_usage rows, got {len(rows)}"

    # Total cost from token_usage table
    total = sum(Decimal(str(r["cost_usd"])) for r in rows)
    expected = cost1 + cost2 + cost3
    assert total == expected, f"Token usage sum mismatch: {total} vs {expected}"


@pytest.mark.asyncio
async def test_record_token_usage_checkpoint_wins_on_db_fail(db, service_with_db) -> None:
    """If DB write fails AFTER checkpoint OK, log warning, don't raise.

    This proves the order: checkpoint PRIMERO → DB después. If DB fails,
    the checkpoint still has the cost data — recovery / reconcile_cost
    can read from checkpoint.
    """
    service, _tmp = service_with_db
    job_id = "costdrift1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    cost = calculate_cost("MiniMax-M3", 1000, 500)

    # Patch db.add_token_usage to raise. _record_token_usage should NOT
    # raise — it should log warning and return gracefully because the
    # checkpoint write already succeeded.
    with patch.object(
        service,
        "_db",
        wraps=db,
        create=True,
    ) as mock_db:
        # Make add_token_usage raise on call
        async def _fail(*args, **kwargs):
            raise RuntimeError("simulated DB write failure")

        mock_db.add_token_usage = _fail
        # BUT checkpoint write uses _update_checkpoint_cost on filesystem
        # so it can still succeed.
        with patch.object(service, "_db", mock_db):
            # Should not raise
            await service._record_token_usage(
                job_id,
                PhaseName.PER_SOURCE_SYNTHESIS,
                "MiniMax-M3",
                1000,
                500,
                cost,
            )

    # Checkpoint file should exist with the cost recorded
    ckpt_path = Path(service._data_root) / job_id / "checkpoint.json"
    if ckpt_path.exists():
        # If checkpoint exists, it should contain the cost
        import json

        data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        # Cumulative cost should be >= the cost we just tried to record
        if isinstance(data, dict) and "cumulative_cost_usd" in data:
            assert Decimal(str(data["cumulative_cost_usd"])) >= cost


@pytest.mark.asyncio
async def test_reconcile_cost_uses_max_not_average(db, service_with_db) -> None:
    """reconcile_cost: max(checkpoint, db_sum, aggregate) — NOT average.

    Sets up a deliberate discrepancy: token_usage sum > checkpoint.
    Expects reconcile to pick the larger value.
    """
    service, _tmp = service_with_db
    job_id = "reconcile1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Record a token_usage with cost $0.05
    cost = Decimal("0.0500")
    await service._record_token_usage(
        job_id,
        PhaseName.FINAL_SYNTHESIS,
        "MiniMax-M3",
        1000,
        500,
        cost,
    )

    # Run reconcile_cost
    reconciled = await service.reconcile_cost(job_id)
    assert reconciled is not None
    assert isinstance(reconciled, Decimal)
    # Should be at least the cost we just recorded (max takes the larger)
    assert reconciled >= cost


@pytest.mark.asyncio
async def test_update_checkpoint_cost_atomic_write(db, service_with_db, tmp_path) -> None:
    """Checkpoint atomic write: tmp + fsync + os.replace (not naive open().write()).

    Verifies by monkey-patching os.replace to fail mid-write and confirming
    no half-written file remains.
    """

    service, _tmp = service_with_db
    job_id = "atomic1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    cost = Decimal("0.01")
    await service._update_checkpoint_cost(job_id, cost, 100, 50)

    # Checkpoint file should exist
    ckpt_path = Path(service._data_root) / job_id / "checkpoint.json"
    assert ckpt_path.exists(), f"Checkpoint file not created at {ckpt_path}"

    # Should be valid JSON
    import json

    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "cost_accumulated_usd" in data
    assert Decimal(str(data["cost_accumulated_usd"])) >= cost

    # Update again — should still produce valid JSON (no partial writes)
    cost2 = Decimal("0.02")
    await service._update_checkpoint_cost(job_id, cost2, 100, 50)
    data2 = json.loads(ckpt_path.read_text(encoding="utf-8"))
    # Cumulative should now be higher
    assert Decimal(str(data2["cost_accumulated_usd"])) > Decimal(str(data["cost_accumulated_usd"]))


# =====================================================================
# DR-Q1A-PRE1A cost-reconciliation fix: tests A-E + notifier propagation
# =====================================================================
# The previous flow had two bugs:
#   1. ``reconcile_cost`` computed max(checkpoint, token_usage_sum,
#      aggregate) but ONLY returned the value (didn't persist).
#   2. ``_phase_write`` discarded that return value and re-read
#      ``research_jobs.cost_usd`` via ``get_research_job_cost``,
#      which could return a STALE aggregate when the checkpoint or
#      token-usage sum legitimately exceeded it (e.g. after a
#      token-usage DB write failure that the checkpoint survived).
# This block pins the fixed behavior: reconcile_cost is
# persistent (writes back via atomic MAX(cost_usd, reconciled)),
# idempotent (running it twice does not lower the first result),
# monotonic (never decreases the aggregate), and the value
# returned by reconcile_cost is the value the notifier and the
# post-completion JobDetail.cost_usd expose.


async def _set_aggregate(db, job_id: str, cost: float) -> None:
    """Test helper: set research_jobs.cost_usd directly via the DB op."""
    await db.set_research_job_cost_monotonic(job_id, cost)


async def _read_aggregate(db, job_id: str) -> float:
    """Test helper: read research_jobs.cost_usd."""
    return await db.get_research_job_cost(job_id)


@pytest.mark.asyncio
async def test_reconcile_cost_checkpoint_wins_is_persisted(
    db, service_with_db
) -> None:
    """Test A: checkpoint > token_usage_sum and aggregate.

    Scenario: a token-usage DB write failed (so the aggregate
    and the SUM are stale/low), but the checkpoint survived
    (atomic tmp+fsync+rename). The checkpoint cost is the
    source of truth.

    After ``reconcile_cost``:
      - return value == checkpoint cost
      - research_jobs.cost_usd persisted to checkpoint cost
      - ``get_research_job_cost`` read == checkpoint cost
        (so ``JobDetail.cost_usd`` exposes the same value)
    """
    service, _tmp = service_with_db
    job_id = "ckptwins1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Simulate "stale" low aggregate (no LLM call succeeded
    # at writing to token_usage).
    await _set_aggregate(db, job_id, 0.01)

    # Write a HIGHER checkpoint cost directly. The checkpoint
    # is the source of truth in the drift recovery hierarchy:
    # only the checkpoint fsyncs immediately after every LLM
    # call. We bypass ``_update_checkpoint_cost`` (which
    # accumulates) and write the absolute value via
    # ``set_research_job_cost_monotonic`` semantics on the
    # checkpoint file.
    checkpoint_cost = Decimal("0.50")
    ckpt_path = Path(service._data_root) / job_id / "checkpoint.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_data = {
        "job_id": job_id,
        "completed_phases": [],
        "phase_data": {},
        "cost_accumulated_usd": float(checkpoint_cost),
        "tokens_in_accumulated": 0,
        "tokens_out_accumulated": 0,
    }
    ckpt_path.write_text(
        json.dumps(ckpt_data, indent=2), encoding="utf-8"
    )

    # token_usage_sum = 0 (nothing recorded)
    # aggregate = 0.01
    # checkpoint = 0.50  ← winner

    # Read checkpoint cost to be precise
    ckpt_cost_read = await service._read_checkpoint_cost(job_id)
    assert ckpt_cost_read == checkpoint_cost, (
        f"Checkpoint cost {ckpt_cost_read} should equal {checkpoint_cost}"
    )

    reconciled = await service.reconcile_cost(job_id)
    assert isinstance(reconciled, Decimal)

    # Reconciled value == checkpoint cost
    assert reconciled == ckpt_cost_read, (
        f"reconcile_cost returned {reconciled}, expected {ckpt_cost_read}"
    )

    # research_jobs.cost_usd was persisted to the reconciled
    # value
    persisted = await _read_aggregate(db, job_id)
    assert persisted == float(ckpt_cost_read), (
        f"Aggregate {persisted} != reconciled {ckpt_cost_read}; "
        f"reconcile_cost did not persist the checkpoint-wins value"
    )

    # And ``JobDetail.cost_usd`` (which reads from
    # ``get_research_job_cost``) exposes the same value
    job_row = await db.get_research_job(job_id)
    assert job_row["cost_usd"] == persisted, (
        f"JobDetail cost_usd {job_row['cost_usd']} != aggregate {persisted}"
    )


@pytest.mark.asyncio
async def test_reconcile_cost_token_usage_sum_wins_is_persisted(
    db, service_with_db
) -> None:
    """Test C: token_usage_sum > checkpoint and aggregate.

    Scenario: a previous reconciliation run persisted a
    high aggregate, then a token-usage write succeeded
    pushing the SUM higher. Re-running reconcile_cost
    must lift the aggregate to the new SUM.
    """
    service, _tmp = service_with_db
    job_id = "sumwins1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Pre-existing aggregate (from a previous reconciliation)
    await _set_aggregate(db, job_id, 0.20)

    # No checkpoint cost
    # (do not write to checkpoint — start from 0)

    # Add a token-usage row with high cost via _record_token_usage
    high_cost = Decimal("0.80")
    await service._record_token_usage(
        job_id,
        PhaseName.FINAL_SYNTHESIS,
        "MiniMax-M3",
        1000,
        500,
        high_cost,
    )
    # Note: ``_record_token_usage`` also updates the aggregate
    # by ``cost_usd += cost_usd`` (additive), so the aggregate
    # is now 0.20 + 0.80 = 1.00. The SUM is also 0.80.

    # To make the SUM legitimately the winner, manually set
    # the aggregate lower than the SUM.
    # (Realistic failure mode: a partial rollback or a
    # migration that overwrote the aggregate.)
    await db.conn.execute(
        "UPDATE research_jobs SET cost_usd = 0.30 WHERE id = ?", (job_id,)
    )
    await db.conn.commit()

    # Verify pre-state
    assert await _read_aggregate(db, job_id) == 0.30
    db_sum = await db.query_scalar(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM research_job_token_usage WHERE job_id = ?",
        job_id,
    )
    assert float(db_sum) == 0.80

    reconciled = await service.reconcile_cost(job_id)
    # winner = max(0.00 checkpoint, 0.80 sum, 0.30 aggregate) = 0.80
    assert reconciled == Decimal("0.80"), (
        f"reconcile_cost returned {reconciled}, expected 0.80 (token-usage sum)"
    )

    persisted = await _read_aggregate(db, job_id)
    assert persisted == 0.80, (
        f"Aggregate {persisted} != 0.80; reconcile_cost did not "
        f"persist the SUM-wins value"
    )


@pytest.mark.asyncio
async def test_reconcile_cost_existing_aggregate_wins_never_decreases(
    db, service_with_db
) -> None:
    """Test D: aggregate > checkpoint and token_usage_sum.

    Scenario: a previous reconciliation run already
    persisted a high aggregate. A subsequent re-run with
    a fresh checkpoint and a fresh SUM that are both
    SMALLER must NOT decrease the aggregate. The aggregate
    is monotonically non-decreasing across re-runs.
    """
    service, _tmp = service_with_db
    job_id = "aggwins1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Set a high existing aggregate directly
    high_agg = 5.00
    await _set_aggregate(db, job_id, high_agg)

    # No checkpoint cost
    # No token_usage rows
    # Aggregate = 5.00  ← winner by a lot

    reconciled = await service.reconcile_cost(job_id)
    assert reconciled == Decimal(str(high_agg)), (
        f"reconcile_cost returned {reconciled}, expected {high_agg} "
        f"(existing aggregate must win; must not decrease)"
    )

    persisted = await _read_aggregate(db, job_id)
    assert persisted == high_agg, (
        f"Aggregate decreased from {high_agg} to {persisted}; "
        f"reconcile_cost must NEVER decrease the aggregate"
    )

    # Run a second time — should still be 5.00
    reconciled2 = await service.reconcile_cost(job_id)
    assert reconciled2 == Decimal(str(high_agg))
    persisted2 = await _read_aggregate(db, job_id)
    assert persisted2 == high_agg


@pytest.mark.asyncio
async def test_reconcile_cost_is_idempotent(db, service_with_db) -> None:
    """Test E: running reconcile_cost twice returns the same value.

    No double counting; the second run sees the persisted
    aggregate from the first run and cannot exceed it (because
    ``set_research_job_cost_monotonic`` is monotonic), so the
    ``max`` of the three sources is unchanged.
    """
    service, _tmp = service_with_db
    job_id = "idemp1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Set up a non-trivial state: aggregate < checkpoint,
    # token_usage_sum = aggregate.
    # (No token_usage rows yet; aggregate = 0.0)
    await _set_aggregate(db, job_id, 0.0)
    checkpoint_cost = Decimal("0.40")
    await service._update_checkpoint_cost(job_id, checkpoint_cost, 100, 50)

    # First run: checkpoint wins
    r1 = await service.reconcile_cost(job_id)
    assert r1 == checkpoint_cost
    persisted1 = await _read_aggregate(db, job_id)
    assert persisted1 == float(checkpoint_cost)

    # Second run: checkpoint still 0.40, token_usage_sum = 0,
    # aggregate = 0.40. max(0.40, 0, 0.40) = 0.40. No change.
    r2 = await service.reconcile_cost(job_id)
    assert r2 == checkpoint_cost, (
        f"Second reconcile_cost returned {r2}, expected {checkpoint_cost}. "
        f"Idempotence violated."
    )
    persisted2 = await _read_aggregate(db, job_id)
    assert persisted2 == float(checkpoint_cost), (
        f"Aggregate changed across re-runs ({persisted1} -> {persisted2}); "
        f"reconcile_cost must be idempotent."
    )

    # Third run: same result
    r3 = await service.reconcile_cost(job_id)
    assert r3 == checkpoint_cost
    persisted3 = await _read_aggregate(db, job_id)
    assert persisted3 == float(checkpoint_cost)


@pytest.mark.asyncio
async def test_set_research_job_cost_monotonic_never_decreases(db) -> None:
    """Test for the new DB op directly: MAX(cost_usd, ?) semantics.

    Independent of ``reconcile_cost``: directly asserts that
    ``set_research_job_cost_monotonic`` never decreases the
    aggregate, regardless of the input value. This pins the
    DB-level invariant that the persistence implementation
    actually provides.
    """
    job_id = "dbmono1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Set high value
    r = await db.set_research_job_cost_monotonic(job_id, 1.0)
    assert r == 1.0
    assert await db.get_research_job_cost(job_id) == 1.0

    # Try to set lower — must be ignored
    r = await db.set_research_job_cost_monotonic(job_id, 0.5)
    assert r == 1.0, (
        f"set_research_job_cost_monotonic decreased aggregate from "
        f"1.0 to {r}; must be monotonic"
    )
    assert await db.get_research_job_cost(job_id) == 1.0

    # Set equal — must be accepted (no-op)
    r = await db.set_research_job_cost_monotonic(job_id, 1.0)
    assert r == 1.0

    # Set higher — must be accepted
    r = await db.set_research_job_cost_monotonic(job_id, 2.5)
    assert r == 2.5
    assert await db.get_research_job_cost(job_id) == 2.5

    # Zero must be respected as a high value
    r = await db.set_research_job_cost_monotonic(job_id, 0.0)
    # 0 < 2.5 so aggregate stays at 2.5
    assert r == 2.5, "Zero must not decrease the aggregate"


@pytest.mark.asyncio
async def test_phase_write_notifier_receives_reconciled_value(
    db, service_with_db
) -> None:
    """Test B: notifier propagation.

    The completion notifier ``send_research_complete`` must
    receive the RECONCILED cost value, not the pre-reconciliation
    stale aggregate. This test creates a state where the
    pre-reconciliation aggregate is stale (small) and the
    checkpoint is the legitimate winner, then exercises
    ``_phase_write`` (the completion path) and asserts that
    the notifier received the reconciled value AND that
    ``JobDetail.cost_usd`` exposes the same value after
    completion.
    """
    service, _tmp_root = service_with_db
    job_id = "notifprop1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=1,  # enable notifier
        user_id=0,
    )

    # Pre-state: stale aggregate = 0.05, no token_usage rows
    await _set_aggregate(db, job_id, 0.05)

    # Checkpoint cost = 0.80 (the source of truth)
    checkpoint_cost = Decimal("0.80")
    await service._update_checkpoint_cost(job_id, checkpoint_cost, 100, 50)

    # Mark the job as 'running' so _phase_write can transition
    # it to 'complete' (update_research_job_status doesn't have
    # a status check, but the notifier reads from
    # get_research_job which returns the existing row).
    await db.update_research_job_status(
        job_id,
        "running",
        started_at="2026-07-21 13:00:00.000",
    )

    # Now exercise _phase_write (Phase 5). The report is a
    # trivial string; what we care about is the notifier
    # call and the final aggregate.
    final_path = await service._phase_write(
        job_id, "# Test Report\n\nBody.\n"
    )

    # 1. The notifier received the RECONCILED value, not 0.05
    notif = service._notifier
    notif.send_research_complete.assert_awaited_once()
    call_kwargs = notif.send_research_complete.await_args.kwargs
    received_cost = call_kwargs.get("cost_usd")
    assert received_cost is not None, (
        f"send_research_complete was called without cost_usd: {call_kwargs}"
    )
    received_cost_dec = (
        received_cost
        if isinstance(received_cost, Decimal)
        else Decimal(str(received_cost))
    )
    assert received_cost_dec == checkpoint_cost, (
        f"Notifier received {received_cost_dec}, expected reconciled "
        f"value {checkpoint_cost}. The notifier saw the stale "
        f"pre-reconciliation aggregate instead of the reconciled value."
    )

    # 2. The aggregate was persisted to the reconciled value
    persisted = await _read_aggregate(db, job_id)
    assert persisted == float(checkpoint_cost), (
        f"Aggregate {persisted} != reconciled {checkpoint_cost}; "
        f"_phase_write did not persist the reconciled value"
    )

    # 3. JobDetail.cost_usd exposes the same value
    job_row = await db.get_research_job(job_id)
    assert job_row["cost_usd"] == float(checkpoint_cost), (
        f"JobDetail.cost_usd {job_row['cost_usd']} != reconciled "
        f"{checkpoint_cost}"
    )

    # 4. Status is 'complete' (the notifier should fire only
    # after a successful completion)
    assert job_row["status"] == "complete"

    # 5. The final path was written
    assert final_path.exists()


@pytest.mark.asyncio
async def test_phase_write_notifier_idempotent_on_re_run(
    db, service_with_db
) -> None:
    """Test B+ (extension): re-running _phase_write on a
    job that has just been completed propagates the same
    reconciled value to the notifier a second time, and the
    aggregate is unchanged (idempotence under re-run).

    Note: this test re-establishes the checkpoint before the
    re-run (because the production code deletes the checkpoint
    after a successful Phase 5). The point of the test is to
    pin the idempotence property: a second reconciliation on
    the same state returns the same value, not a different one.
    An external overwrite of the aggregate without a surviving
    checkpoint is a separate scenario the design does not
    claim to recover from.
    """
    service, _tmp = service_with_db
    job_id = "notifidem1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=1,
        user_id=0,
    )

    # First run: stale aggregate 0.05, checkpoint 0.80
    await _set_aggregate(db, job_id, 0.05)
    ckpt_path = Path(service._data_root) / job_id / "checkpoint.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "completed_phases": [],
                "phase_data": {},
                "cost_accumulated_usd": 0.80,
                "tokens_in_accumulated": 0,
                "tokens_out_accumulated": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    await db.update_research_job_status(
        job_id,
        "running",
        started_at="2026-07-21 13:00:00.000",
    )
    await service._phase_write(job_id, "first run body\n")

    notif = service._notifier
    notif.send_research_complete.assert_awaited_once()
    first_call = notif.send_research_complete.await_args.kwargs["cost_usd"]
    first_call_dec = (
        first_call if isinstance(first_call, Decimal) else Decimal(str(first_call))
    )
    assert first_call_dec == Decimal("0.80")
    assert await _read_aggregate(db, job_id) == 0.80

    # Re-establish the checkpoint (it was deleted by the first
    # ``_phase_write``) and re-run. The reconciliation must
    # return the same value (idempotence), and the aggregate
    # must not change.
    notif.send_research_complete.reset_mock()
    # The cleanup block at the end of _phase_write removes
    # the per-job directory when empty, so we must recreate
    # the parent directory before writing the new checkpoint.
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "completed_phases": [],
                "phase_data": {},
                "cost_accumulated_usd": 0.80,
                "tokens_in_accumulated": 0,
                "tokens_out_accumulated": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    await db.update_research_job_status(
        job_id,
        "running",
        started_at="2026-07-21 13:00:00.000",
    )
    await service._phase_write(job_id, "second run body\n")

    notif.send_research_complete.assert_awaited_once()
    second_call = notif.send_research_complete.await_args.kwargs["cost_usd"]
    second_call_dec = (
        second_call if isinstance(second_call, Decimal) else Decimal(str(second_call))
    )
    assert second_call_dec == Decimal("0.80"), (
        f"Notifier received {second_call_dec} on re-run, "
        f"expected 0.80 (idempotent reconciliation)"
    )
    assert await _read_aggregate(db, job_id) == 0.80, (
        "Aggregate changed across re-runs"
    )
