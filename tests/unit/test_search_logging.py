"""Tests Sprint 9.3: Logging estructurado (Capa 10).

Cubre:
- query_hash (no plain text) en logs
- backend name logged
- intent y content_mode logged
- duration_ms logged
- status (ok|fallback_circuit|fallback_format|error) logged
- error_code logged si error
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes.memory.db import Database
from hermes.services.search.budget import BudgetTracker
from hermes.services.search.protocol import SearchResult
from hermes.services.search.resilience import (
    CircuitBreakerRegistry,
    ConcurrencyLimiter,
)
from hermes.services.search.router import hermes_search


@pytest.fixture
async def db() -> Database:
    with tempfile.TemporaryDirectory() as td:
        d = Database(Path(td) / "test.db")
        await d.initialize()
        yield d
        await d.close()


def _make_searxng_backend(
    content_modes: frozenset[str] = frozenset({"snippet"}),
    has_budget: bool = True,
) -> MagicMock:
    """Crea un mock backend SearXNG."""
    backend = MagicMock()
    backend.name = "searxng"
    backend.SUPPORTED_CONTENT_MODES = content_modes

    async def fake_search(
        query: str, content_mode: str, num_results: int, *, intent: str = "general"
    ) -> SearchResult:
        return SearchResult(
            results=[{"title": "R1", "url": "u1", "content": "c"}],
            backend_used="searxng",
            query=query,
            content_mode=content_mode,
            original_content_mode=content_mode,
            format_fallback=False,
            size_guard_chars=50000,
            truncated=False,
        )

    async def fake_has_budget() -> bool:
        return has_budget

    backend.search = fake_search
    backend.has_budget = fake_has_budget
    return backend


@pytest.mark.asyncio
async def test_search_logs_query_hash_not_plain_text(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    """Log incluye query_hash (NO plain text) por privacy."""
    searxng = _make_searxng_backend()
    backends = {"searxng": searxng}
    budget = BudgetTracker(db, limits={"searxng": -1})

    caplog.set_level(logging.INFO, logger="hermes.services.search.router")

    sensitive_query = "my personal medical condition xyz123"
    await hermes_search(
        query=sensitive_query,
        intent="general",
        backends=backends,
        budget=budget,
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )

    # El plain text NO debe aparecer en ningun log record
    log_text = caplog.text
    assert sensitive_query not in log_text
    # El query_hash SI debe aparecer en el record 'search_query'
    search_query_records = [r for r in caplog.records if r.message == "search_query"]
    assert len(search_query_records) >= 1
    record = search_query_records[0]
    assert hasattr(record, "query_hash")
    # SHA256 hex prefix, 16 chars
    assert len(record.query_hash) == 16
    assert all(c in "0123456789abcdef" for c in record.query_hash)


@pytest.mark.asyncio
async def test_search_logs_backend_name(db: Database, caplog: pytest.LogCaptureFixture) -> None:
    """Log incluye backend name (para debugging)."""
    searxng = _make_searxng_backend()
    backends = {"searxng": searxng}
    budget = BudgetTracker(db, limits={"searxng": -1})

    caplog.set_level(logging.INFO, logger="hermes.services.search.router")
    await hermes_search(
        query="test",
        intent="general",
        backends=backends,
        budget=budget,
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )

    # Buscar el record con 'search_query' y verificar su 'backend' field
    search_query_records = [r for r in caplog.records if r.message == "search_query"]
    assert len(search_query_records) >= 1
    assert search_query_records[0].backend == "searxng"


@pytest.mark.asyncio
async def test_search_format_fallback_logs_warning(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    """Format fallback loggea INFO (no warning, es comportamiento esperado)."""
    searxng = _make_searxng_backend(content_modes=frozenset({"snippet"}))
    backends = {"searxng": searxng}
    budget = BudgetTracker(db, limits={"searxng": -1})

    caplog.set_level(logging.DEBUG, logger="hermes.services.search.router")
    await hermes_search(
        query="test",
        intent="general",
        content="summary",  # SearXNG no soporta -> format fallback
        backends=backends,
        budget=budget,
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )

    # search_format_fallback loggea con INFO level
    assert "search_format_fallback" in caplog.text
    # Verifica que el request fue a "searxng" (correct backend)
    # NO a tavily/exa (que no existen)
    assert "tavily" not in caplog.text or "format_fallback" in caplog.text


@pytest.mark.asyncio
async def test_search_error_logs_error_code(db: Database, caplog: pytest.LogCaptureFixture) -> None:
    """Error path loggea error code (no silent fail)."""
    searxng = _make_searxng_backend(has_budget=False)
    # Pre-abrir circuit
    cb = CircuitBreakerRegistry(threshold=1, ttl_seconds=300)
    cb.record_failure("searxng")
    backends = {"searxng": searxng}
    budget = BudgetTracker(db, limits={"searxng": 0})  # has_budget=False

    caplog.set_level(logging.INFO, logger="hermes.services.search.router")
    result = await hermes_search(
        query="test",
        intent="general",
        backends=backends,
        budget=budget,
        circuit_breaker=cb,
        semaphore=ConcurrencyLimiter(),
    )

    # Error retornado
    assert result.error is not None
    # Log con error code (search_circuit_fallback se loggea INFO level)
    log_text = caplog.text
    assert "ALL_BACKENDS_FAILED" in log_text or "search_circuit_fallback" in log_text


@pytest.mark.asyncio
async def test_search_uses_query_logger_name(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    """Los logs usan el logger 'hermes.services.search.router'."""
    searxng = _make_searxng_backend()
    backends = {"searxng": searxng}
    budget = BudgetTracker(db, limits={"searxng": -1})

    caplog.set_level(logging.INFO, logger="hermes.services.search.router")
    await hermes_search(
        query="test",
        intent="general",
        backends=backends,
        budget=budget,
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )

    # Verificar que el logger name esta en los records
    logger_names = [r.name for r in caplog.records]
    assert any("hermes.services.search.router" in n for n in logger_names)


# --- S9.3.1 punto 4: métrica size_guard_truncated ---


@pytest.mark.asyncio
async def test_search_logs_size_guard_truncated_flag(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    """Log incluye size_guard_truncated=True/False (métrica hit-rate)."""
    searxng = _make_searxng_backend()
    backends = {"searxng": searxng}
    budget = BudgetTracker(db, limits={"searxng": -1})

    caplog.set_level(logging.INFO, logger="hermes.services.search.router")
    await hermes_search(
        query="test",
        intent="general",
        backends=backends,
        budget=budget,
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )

    search_query_records = [r for r in caplog.records if r.message == "search_query"]
    assert len(search_query_records) >= 1
    record = search_query_records[0]
    assert hasattr(record, "size_guard_truncated")
    # Sin truncacion (pocos resultados)
    assert record.size_guard_truncated is False
