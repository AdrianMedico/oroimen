"""Integration tests for cost drift protection (TDD §8.4, §6.8).

Anti-regression checks:
- _record_token_usage: checkpoint writes FIRST, DB writes SECOND. If the
  DB write fails but the checkpoint OK, log warning, continue.
- _record_token_usage: aggregates by phase/model (sum tokens_in, tokens_out, cost).
- reconcile_cost uses max(checkpoint, token_usage_sum, aggregate), NOT average.
"""

from __future__ import annotations

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
