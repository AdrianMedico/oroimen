"""Tests Sprint 9.3: BudgetTracker (Capa 3).

Cubre:
- has_budget: True si used < limit, False si reached, True si limit=-1
- record_usage: atomic increment + UPSERT pattern
- remaining: limit - used, cap en 0
- Month rollover: row con month='YYYY-MM' se resetea cuando current month != 'YYYY-MM'
- Race conditions: 10 requests concurrentes, exactamente `used+10` al final
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes.memory.db import Database
from hermes.services.search.budget import BudgetTracker


@pytest.fixture
async def db() -> Database:
    with tempfile.TemporaryDirectory() as td:
        d = Database(Path(td) / "test.db")
        await d.initialize()
        yield d
        await d.close()


# --- has_budget ---


@pytest.mark.asyncio
async def test_has_budget_true_when_below_limit(db: Database) -> None:
    """has_budget retorna True si used < limit."""
    budget = BudgetTracker(db, limits={"tavily": 1000})
    # Sin uso previo → budget completo
    assert await budget.has_budget("tavily") is True


@pytest.mark.asyncio
async def test_has_budget_false_when_at_limit(db: Database) -> None:
    """has_budget retorna False si used >= limit."""
    budget = BudgetTracker(db, limits={"tavily": 3})
    # Agotar budget
    for _ in range(3):
        await budget.record_usage("tavily", count=1)
    assert await budget.has_budget("tavily") is False


@pytest.mark.asyncio
async def test_has_budget_true_when_unlimited(db: Database) -> None:
    """has_budget retorna True si limit=-1 (ilimitado, e.g. SearXNG)."""
    budget = BudgetTracker(db, limits={"searxng": -1})
    # Aunque registremos mucho uso, ilimitado es True
    for _ in range(1000):
        await budget.record_usage("searxng", count=1)
    assert await budget.has_budget("searxng") is True


@pytest.mark.asyncio
async def test_has_budget_returns_false_for_unknown_backend(db: Database) -> None:
    """has_budget retorna False para backends no configurados (default limit=0)."""
    budget = BudgetTracker(db, limits={"tavily": 1000})
    # 'exa' no está en limits → trata como limit=0 → no budget
    assert await budget.has_budget("exa") is False


# --- record_usage ---


@pytest.mark.asyncio
async def test_record_usage_creates_row_if_not_exists(db: Database) -> None:
    """record_usage crea la fila (month, backend) en el primer uso."""
    budget = BudgetTracker(db, limits={"tavily": 1000})
    await budget.record_usage("tavily", count=1)
    # Verificar que la fila existe
    assert await budget.remaining("tavily") == 999


@pytest.mark.asyncio
async def test_record_usage_increments_atomic(db: Database) -> None:
    """record_usage incrementa `used` por `count` (atomic)."""
    budget = BudgetTracker(db, limits={"tavily": 1000})
    await budget.record_usage("tavily", count=5)
    assert await budget.remaining("tavily") == 995
    await budget.record_usage("tavily", count=3)
    assert await budget.remaining("tavily") == 992


@pytest.mark.asyncio
async def test_record_usage_under_concurrent_load(db: Database) -> None:
    """10 requests concurrentes, exactamente +10 al final (atomicidad)."""
    budget = BudgetTracker(db, limits={"tavily": 1000})
    # Lanzar 10 requests en paralelo
    await asyncio.gather(*[budget.record_usage("tavily", count=1) for _ in range(10)])
    assert await budget.remaining("tavily") == 990


# --- remaining ---


@pytest.mark.asyncio
async def test_remaining_returns_full_budget_for_unused(db: Database) -> None:
    """remaining retorna el limit completo si no hay uso."""
    budget = BudgetTracker(db, limits={"tavily": 1000})
    assert await budget.remaining("tavily") == 1000


@pytest.mark.asyncio
async def test_remaining_returns_negative_as_cap_zero(db: Database) -> None:
    """remaining nunca retorna negativo (cap en 0)."""
    budget = BudgetTracker(db, limits={"tavily": 2})
    await budget.record_usage("tavily", count=5)  # over-usage
    assert await budget.remaining("tavily") == 0  # cap en 0, no -3


@pytest.mark.asyncio
async def test_remaining_returns_minus_one_for_unlimited(db: Database) -> None:
    """remaining retorna -1 para unlimited (SearXNG)."""
    budget = BudgetTracker(db, limits={"searxng": -1})
    assert await budget.remaining("searxng") == -1


# --- Month rollover ---


@pytest.mark.asyncio
async def test_month_rollover_resets_used(db: Database) -> None:
    """Si el row tiene month != current month, se resetea a 0."""
    budget = BudgetTracker(db, limits={"tavily": 1000})
    # Simular uso en enero 2025
    await budget.record_usage("tavily", count=500)
    # Forzar month a enero en la DB
    cur = await db.conn.execute(
        "UPDATE search_budget SET month = '2025-01' WHERE backend = 'tavily'"
    )
    await db.conn.commit()
    # Llamar has_budget → debe detectar mes viejo y resetear
    is_ok = await budget.has_budget("tavily")
    assert is_ok is True  # reset a 0 → budget completo
    # Verificar que el row se actualizó
    cur = await db.conn.execute("SELECT used, month FROM search_budget WHERE backend = 'tavily'")
    row = await cur.fetchone()
    # month debe ser el mes actual (no 2025-01)
    current_month = datetime.now(UTC).strftime("%Y-%m")
    assert row[0] == 0  # reset
    assert row[1] == current_month  # mes actual


@pytest.mark.asyncio
async def test_month_rollover_does_not_affect_other_backends(db: Database) -> None:
    """Month rollover resetea solo el backend del row viejo."""
    budget = BudgetTracker(db, limits={"tavily": 1000, "exa": 1000})
    # Uso en tavily
    await budget.record_usage("tavily", count=500)
    # Uso en exa (mes actual)
    await budget.record_usage("exa", count=300)
    # Forzar tavily a enero
    await db.conn.execute("UPDATE search_budget SET month = '2025-01' WHERE backend = 'tavily'")
    await db.conn.commit()
    # Trigger rollover en tavily
    await budget.has_budget("tavily")
    # tavily: reset a 0
    assert await budget.remaining("tavily") == 1000
    # exa: intacto (mes actual)
    assert await budget.remaining("exa") == 700


# --- Edge cases ---


@pytest.mark.asyncio
async def test_record_usage_with_zero_count_is_noop(db: Database) -> None:
    """record_usage con count=0 no cambia used."""
    budget = BudgetTracker(db, limits={"tavily": 1000})
    await budget.record_usage("tavily", count=5)
    await budget.record_usage("tavily", count=0)
    assert await budget.remaining("tavily") == 995


@pytest.mark.asyncio
async def test_separate_backends_have_independent_budgets(db: Database) -> None:
    """Tavily y Exa tienen budgets independientes."""
    budget = BudgetTracker(db, limits={"tavily": 1000, "exa": 500})
    await budget.record_usage("tavily", count=100)
    await budget.record_usage("exa", count=50)
    assert await budget.remaining("tavily") == 900
    assert await budget.remaining("exa") == 450
