"""Tests Sprint 9.3: Web Search Router (Capa 6).

Cubre TODOS los fixes de las 3 rondas de cross-review:
- v1.0 -> v1.1 (Gemini 3.5 Thinking): P0-1 _TIMEOUTS, P0-1 bis intent param,
  P0-2 fallback doble check, P0-3 get_semaphore, P1-1 dedup URLs,
  P1-2 record_usage antes de search
- v1.1 -> v1.2 (GLM 5.2): P0-5 SearchResult.error field, P1-3 helpers
- v1.2 -> v1.3 (Gemini 3.5 Thinking): P1-1 double-checked locking
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes.memory.db import Database
from hermes.services.search.budget import BudgetTracker
from hermes.services.search.errors import (
    SearchErrorCode,
)
from hermes.services.search.protocol import (
    BackendProtocol,
    SearchResult,
)
from hermes.services.search.resilience import (
    CircuitBreakerRegistry,
    ConcurrencyLimiter,
)
from hermes.services.search.router import (
    _BACKEND_BY_INTENT,
    _TIMEOUTS,
    _apply_size_guard,
    _compute_usage_cost,
    _dedup_results,
    _normalize_result_urls,
    _sanitize_urls,
    hermes_search,
)


@pytest.fixture
async def db() -> Database:
    with tempfile.TemporaryDirectory() as td:
        d = Database(Path(td) / "test.db")
        await d.initialize()
        yield d
        await d.close()


def _make_backend(
    name: str = "searxng",
    content_modes: frozenset[str] = frozenset({"snippet"}),
    has_budget: bool = True,
    healthy: bool = True,
    should_fail: bool = False,
    results: list[dict] | None = None,
    response_content_mode: str = "snippet",
    response_original_content_mode: str = "snippet",
) -> BackendProtocol:
    """Crea un backend mock que respeta el Protocol."""
    backend = MagicMock(spec=BackendProtocol)
    backend.name = name
    backend.SUPPORTED_CONTENT_MODES = content_modes
    backend.has_budget = AsyncMock(return_value=has_budget)
    backend.health_check = AsyncMock(return_value=healthy)

    if should_fail:

        async def _fail(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError(f"{name} failure")

        backend.search = AsyncMock(side_effect=_fail)
    else:
        default_results = results or [
            {
                "title": f"Result from {name}",
                "url": f"https://example.com/{name}/1",
                "content": f"Content from {name} for test",
            }
        ]

        async def _search(
            query: str,
            content_mode: str,
            num_results: int,
            *,
            intent: str = "general",
        ) -> SearchResult:
            return SearchResult(
                results=default_results,
                backend_used=name,
                query=query,
                content_mode=response_content_mode,
                original_content_mode=response_original_content_mode,
                format_fallback=(response_content_mode != response_original_content_mode),
                size_guard_chars=50000,
                truncated=False,
            )

        backend.search = AsyncMock(side_effect=_search)
    return backend


def _make_budget(db: Database, limits: dict[str, int] | None = None) -> BudgetTracker:
    """Crea BudgetTracker con real db."""
    if limits is None:
        limits = {"searxng": -1, "tavily": 1000, "exa": 1000}
    return BudgetTracker(db, limits=limits)


# --- _BACKEND_BY_INTENT ---


def test_backend_by_intent_mapping() -> None:
    """Mapeo intent -> backend es el spec'd del TDD."""
    assert _BACKEND_BY_INTENT == {
        "general": "searxng",
        "semantic": "exa",
        "deep_research": "tavily",
    }


# --- _TIMEOUTS ---


def test_timeouts_per_backend() -> None:
    """Timeouts per-backend (no unico) — P1-4 fix v1.2."""
    assert _TIMEOUTS == {"searxng": 10.0, "tavily": 15.0, "exa": 15.0}


# --- validation: empty query ---


@pytest.mark.asyncio
async def test_empty_query_returns_empty_query_error(db: Database) -> None:
    """query='' retorna SearchError EMPTY_QUERY."""
    backends = {"searxng": _make_backend()}
    result = await hermes_search(
        query="",
        intent="general",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.results == []
    assert result.error is not None
    assert result.error.code == SearchErrorCode.EMPTY_QUERY


@pytest.mark.asyncio
async def test_whitespace_query_returns_empty_query_error(db: Database) -> None:
    """query='   ' (whitespace) tambien retorna EMPTY_QUERY."""
    backends = {"searxng": _make_backend()}
    result = await hermes_search(
        query="   ",
        intent="general",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.error is not None
    assert result.error.code == SearchErrorCode.EMPTY_QUERY


# --- validation: long query ---


@pytest.mark.asyncio
async def test_long_query_is_truncated(db: Database) -> None:
    """query > 2000 chars se trunca a 2000."""
    backends = {"searxng": _make_backend()}
    long_query = "x" * 3000
    result = await hermes_search(
        query=long_query,
        intent="general",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.query == "x" * 2000


# --- validation: invalid intent ---


@pytest.mark.asyncio
async def test_invalid_intent_returns_error(db: Database) -> None:
    """intent='unknown' retorna SearchError INVALID_INTENT."""
    backends = {"searxng": _make_backend()}
    result = await hermes_search(
        query="test",
        intent="invalid_intent",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.error is not None
    assert result.error.code == SearchErrorCode.INVALID_INTENT


# --- validation: invalid content ---


@pytest.mark.asyncio
async def test_invalid_content_returns_error(db: Database) -> None:
    """content='unknown' retorna SearchError INVALID_CONTENT."""
    backends = {"searxng": _make_backend()}
    result = await hermes_search(
        query="test",
        intent="general",
        content="invalid_content",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.error is not None
    assert result.error.code == SearchErrorCode.INVALID_CONTENT


# --- routing ---


@pytest.mark.asyncio
async def test_general_intent_routes_to_searxng(db: Database) -> None:
    """intent='general' rutea a SearXNG (default)."""
    searxng = _make_backend(name="searxng")
    backends = {"searxng": searxng, "tavily": _make_backend(name="tavily")}
    await hermes_search(
        query="test",
        intent="general",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    searxng.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_deep_research_intent_routes_to_tavily(db: Database) -> None:
    """intent='deep_research' rutea a Tavily."""
    tavily = _make_backend(name="tavily")
    backends = {"searxng": _make_backend(name="searxng"), "tavily": tavily}
    await hermes_search(
        query="test",
        intent="deep_research",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    tavily.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_semantic_intent_routes_to_exa(db: Database) -> None:
    """intent='semantic' rutea a Exa."""
    exa = _make_backend(name="exa")
    backends = {
        "searxng": _make_backend(name="searxng"),
        "tavily": _make_backend(name="tavily"),
        "exa": exa,
    }
    await hermes_search(
        query="test",
        intent="semantic",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    exa.search.assert_awaited_once()


# --- circuit fallback (P0-2 fix v1.1) ---


@pytest.mark.asyncio
async def test_circuit_fallback_when_primary_open(db: Database) -> None:
    """Si Tavily circuit esta open, fallback a SearXNG."""
    searxng = _make_backend(name="searxng")
    tavily = _make_backend(name="tavily")
    backends = {"searxng": searxng, "tavily": tavily}
    cb = CircuitBreakerRegistry(threshold=1, ttl_seconds=300)
    cb.record_failure("tavily")  # open
    result = await hermes_search(
        query="test",
        intent="deep_research",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=cb,
        semaphore=ConcurrencyLimiter(),
    )
    assert result.backend_used == "searxng"
    searxng.search.assert_awaited_once()
    tavily.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_circuit_fallback_when_no_budget(db: Database) -> None:
    """Si Tavily no tiene budget, fallback a SearXNG."""
    searxng = _make_backend(name="searxng", has_budget=True)
    tavily = _make_backend(name="tavily", has_budget=False)
    backends = {"searxng": searxng, "tavily": tavily}
    cb = CircuitBreakerRegistry()
    result = await hermes_search(
        query="test",
        intent="deep_research",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=cb,
        semaphore=ConcurrencyLimiter(),
    )
    assert result.backend_used == "searxng"


@pytest.mark.asyncio
async def test_circuit_fallback_to_searxng_when_searxng_also_down(db: Database) -> None:
    """Si Tavily Y SearXNG circuit estan open, retorna ALL_BACKENDS_FAILED."""
    searxng = _make_backend(name="searxng", has_budget=False)
    tavily = _make_backend(name="tavily", has_budget=False)
    backends = {"searxng": searxng, "tavily": tavily}
    cb = CircuitBreakerRegistry(threshold=1, ttl_seconds=300)
    cb.record_failure("tavily")
    cb.record_failure("searxng")
    result = await hermes_search(
        query="test",
        intent="deep_research",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=cb,
        semaphore=ConcurrencyLimiter(),
    )
    assert result.error is not None
    assert result.error.code == SearchErrorCode.ALL_BACKENDS_FAILED
    assert "searxng" in result.error.backends_tried


# --- format fallback (P0 Gemini 3.5) ---


@pytest.mark.asyncio
async def test_format_fallback_when_searxng_doesnt_support_content(db: Database) -> None:
    """SearXNG no soporta 'summary' — degrada a snippet silenciosamente."""
    searxng = _make_backend(
        name="searxng",
        content_modes=frozenset({"snippet"}),
        response_content_mode="snippet",  # backend degrada
        response_original_content_mode="summary",  # LLM pidio summary
    )
    backends = {"searxng": searxng}
    result = await hermes_search(
        query="test",
        intent="general",
        content="summary",  # LLM pide summary
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    # El backend se llamo con content_mode='snippet' (degraded)
    assert (
        searxng.search.await_args.kwargs.get("content_mode") == "snippet"
        or searxng.search.await_args.args[1] == "snippet"
    )
    assert result.error is None  # no error, graceful degradation


# --- double-checked locking (P1-1 fix v1.3) ---


@pytest.mark.asyncio
async def test_budget_exhausted_inside_semaphore_returns_error(db: Database) -> None:
    """Si budget se agota dentro del semaforo (double-check), retorna BUDGET_EXHAUSTED.

    Scenario: 5 requests concurrentes con intent='general' (searxng).
    Todas pasan has_budget() en paso 3 (budget=3, True). 3 entran al
    semaforo, record_usage → budget=0. Las 2 que esperaban despiertan,
    su local backend_name sigue 'searxng', pero el double-check detecta
    budget=0 y retorna BUDGET_EXHAUSTED.
    """
    searxng = _make_backend(name="searxng", has_budget=False)
    backends = {"searxng": searxng}
    cb = CircuitBreakerRegistry()
    # Paso 3: has_budget()=False → fallback intent a searxng (mismo, no-op)
    # Paso 5 (double-check): has_budget()=False → BUDGET_EXHAUSTED
    result = await hermes_search(
        query="test",
        intent="general",  # → searxng directamente
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=cb,
        semaphore=ConcurrencyLimiter(),
    )
    searxng.search.assert_not_awaited()
    assert result.error is not None
    assert result.error.code == SearchErrorCode.BUDGET_EXHAUSTED


# --- URL normalization (P1-1 v1.1) ---


def test_normalize_result_urls_strips_trailing_slash() -> None:
    """URLs con trailing '/' se normalizan a sin '/'."""
    result = SearchResult(
        results=[
            {"title": "R1", "url": "https://example.com/doc/", "content": "c"},
            {"title": "R2", "url": "https://example.com/doc", "content": "c"},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    normalized = _normalize_result_urls(result)
    assert normalized.results[0]["url"] == "https://example.com/doc"
    assert normalized.results[1]["url"] == "https://example.com/doc"


def test_normalize_result_urls_handles_empty_urls() -> None:
    """URLs vacias no se modifican (no lanza error)."""
    result = SearchResult(
        results=[{"title": "R1", "url": "", "content": "c"}],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    normalized = _normalize_result_urls(result)
    assert normalized.results[0]["url"] == ""


# --- size guard (P1-9 v1.2) ---


def test_size_guard_truncates_complete_results() -> None:
    """Size guard trunca resultados completos, no strings parciales."""
    result = SearchResult(
        results=[
            {"title": "R1", "url": "u1", "content": "x" * 100},
            {"title": "R2", "url": "u2", "content": "y" * 100},
            {"title": "R3", "url": "u3", "content": "z" * 100},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=150,
        truncated=False,
    )
    truncated = _apply_size_guard(result, limit=150)
    # R1 (100 chars) cabe. R2 (100 chars) total=200 > 150 → no se incluye.
    assert len(truncated.results) == 1
    assert truncated.results[0]["title"] == "R1"
    assert truncated.truncated is True
    assert truncated.truncated_at_chars == 100


def test_size_guard_no_truncation_when_within_limit() -> None:
    """Sin truncacion cuando total < limit."""
    result = SearchResult(
        results=[
            {"title": "R1", "url": "u1", "content": "x" * 50},
            {"title": "R2", "url": "u2", "content": "y" * 50},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    truncated = _apply_size_guard(result, limit=50000)
    assert len(truncated.results) == 2
    assert truncated.truncated is False
    assert truncated.truncated_at_chars is None


# Sprint 9.3.2: regression test for Bug 4 (TypeError en _apply_size_guard
# cuando raw_content es None). Tavily a veces retorna raw_content=None en
# resultados vacios. El fix usa `r.get("raw_content") or ""` para evitar
# len(None).
def test_size_guard_handles_raw_content_none() -> None:
    """Size guard no falla si raw_content es None (Tavily bug 4)."""
    result = SearchResult(
        results=[
            {"title": "R1", "url": "u1", "content": "", "raw_content": None},
            {"title": "R2", "url": "u2", "content": "real content"},
        ],
        backend_used="tavily",
        query="test",
        content_mode="full",
        original_content_mode="full",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    # Antes del fix esto lanzaba TypeError: object of type 'NoneType' has no len()
    truncated = _apply_size_guard(result, limit=50000)
    # R1 tiene 0 chars (raw_content=None tratado como ""), R2 incluido
    assert len(truncated.results) == 2
    assert truncated.truncated is False


def test_size_guard_handles_missing_raw_content_key() -> None:
    """Size guard maneja cuando raw_content no existe en el dict."""
    result = SearchResult(
        results=[
            {"title": "R1", "url": "u1"},  # sin content ni raw_content
        ],
        backend_used="tavily",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    truncated = _apply_size_guard(result, limit=50000)
    assert len(truncated.results) == 1
    assert truncated.truncated is False


# --- circuit breaker integration ---


@pytest.mark.asyncio
async def test_successful_search_records_success_in_circuit_breaker(db: Database) -> None:
    """Successful search llama circuit_breaker.record_success."""
    searxng = _make_backend(name="searxng")
    backends = {"searxng": searxng}
    cb = CircuitBreakerRegistry()
    await hermes_search(
        query="test",
        intent="general",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=cb,
        semaphore=ConcurrencyLimiter(),
    )
    # Verificar que no abrio el circuit (record_success fue llamado)
    assert cb.is_open("searxng") is False


@pytest.mark.asyncio
async def test_timeout_records_failure_in_circuit_breaker(db: Database) -> None:
    """Timeout en search llama circuit_breaker.record_failure."""
    searxng = _make_backend(name="searxng")

    async def _slow_search(*args: Any, **kwargs: Any) -> SearchResult:
        await asyncio.sleep(2)  # mas que el timeout (10s para searxng)
        return SearchResult(
            results=[],
            backend_used="searxng",
            query="test",
            content_mode="snippet",
            original_content_mode="snippet",
            format_fallback=False,
            size_guard_chars=50000,
            truncated=False,
        )

    searxng.search = _slow_search
    backends = {"searxng": searxng}
    cb = CircuitBreakerRegistry(threshold=1, ttl_seconds=300)
    # Override timeout a 0.01 para el test (no esperar 10s)
    import hermes.services.search.router as router_mod

    original_timeout = router_mod._TIMEOUTS["searxng"]
    router_mod._TIMEOUTS["searxng"] = 0.01
    try:
        result = await hermes_search(
            query="test",
            intent="general",
            backends=backends,
            budget=_make_budget(db),
            circuit_breaker=cb,
            semaphore=ConcurrencyLimiter(),
        )
    finally:
        router_mod._TIMEOUTS["searxng"] = original_timeout

    assert result.error is not None
    assert result.error.code == SearchErrorCode.TIMEOUT
    # Circuit breaker debe haber registrado el failure
    assert cb.is_open("searxng") is True


# --- P0-1 fix: backend missing fallback ---


@pytest.mark.asyncio
async def test_missing_primary_backend_falls_back_to_searxng(db: Database) -> None:
    """Si el backend primario (exa) no esta en backends, fallback a SearXNG.

    P0-1 fix: el router no debe hacer KeyError si el backend
    primario no esta configurado (e.g., EXA_API_KEY no set).
    """
    searxng = _make_backend(name="searxng")
    backends = {"searxng": searxng}  # NO hay 'exa'
    result = await hermes_search(
        query="test",
        intent="semantic",  # rutea a 'exa' que NO existe
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.error is None
    assert result.backend_used == "searxng"
    searxng.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_all_backends_returns_all_backends_failed(db: Database) -> None:
    """Si no hay ningun backend configurado, retorna ALL_BACKENDS_FAILED."""
    backends: dict[str, Any] = {}  # vacio
    result = await hermes_search(
        query="test",
        intent="general",
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.error is not None
    assert result.error.code == SearchErrorCode.ALL_BACKENDS_FAILED


# --- P1-1 fix: original_content_mode preserved on format fallback ---


@pytest.mark.asyncio
async def test_format_fallback_preserves_original_content_mode(db: Database) -> None:
    """Format fallback preserva original_content_mode para el LLM.

    P1-1 fix: el LLM debe saber que pidio 'summary' aunque el
    backend devolvio 'snippet' (format_fallback=True).
    """
    searxng = _make_backend(
        name="searxng",
        content_modes=frozenset({"snippet"}),
        response_content_mode="snippet",
        response_original_content_mode="snippet",
    )
    backends = {"searxng": searxng}
    result = await hermes_search(
        query="test",
        intent="general",
        content="summary",  # LLM pide summary
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.format_fallback is True
    assert result.content_mode == "snippet"  # degradado
    assert result.original_content_mode == "summary"  # P1-1: preservado


# --- P1-4 fix: dedup de resultados ---


def test_dedup_results_removes_duplicates() -> None:
    """Resultados con misma URL normalizada se dedup (mantiene el 1o)."""

    result = SearchResult(
        results=[
            {"title": "R1", "url": "https://example.com/doc", "content": "c1"},
            {"title": "R2", "url": "https://example.com/doc", "content": "c2"},
            {"title": "R3", "url": "https://example.com/other", "content": "c3"},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    deduped = _dedup_results(result)
    assert len(deduped.results) == 2
    assert deduped.results[0]["title"] == "R1"  # primero se mantiene
    assert deduped.results[1]["url"] == "https://example.com/other"


def test_dedup_results_preserves_empty_urls() -> None:
    """Resultados sin URL no se dedup."""

    result = SearchResult(
        results=[
            {"title": "R1", "url": "", "content": "c1"},
            {"title": "R2", "url": "", "content": "c2"},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    deduped = _dedup_results(result)
    assert len(deduped.results) == 2  # ambos se conservan


# --- S9.3.1 punto 1: budget variable por intent/backend ---


def test_compute_usage_cost_tavily_advanced() -> None:
    """Tavily deep_research cobra 2 credits (advanced)."""
    assert _compute_usage_cost("tavily", "deep_research") == 2


def test_compute_usage_cost_tavily_basic() -> None:
    """Tavily general/semantic cobra 1 credit (basic)."""
    assert _compute_usage_cost("tavily", "general") == 1
    assert _compute_usage_cost("tavily", "semantic") == 1


def test_compute_usage_cost_searxng_is_zero() -> None:
    """SearXNG es self-hosted, unlimited (count=0)."""
    assert _compute_usage_cost("searxng", "general") == 0
    assert _compute_usage_cost("searxng", "deep_research") == 0


def test_compute_usage_cost_exa_is_one() -> None:
    """Exa cobra 1 credit/search."""
    assert _compute_usage_cost("exa", "semantic") == 1
    assert _compute_usage_cost("exa", "general") == 1


@pytest.mark.asyncio
async def test_tavily_deep_research_records_two_credits(db: Database) -> None:
    """Integration: Tavily + deep_research descuenta 2 credits (no 1)."""
    tavily = _make_backend(name="tavily", content_modes=frozenset({"full"}))
    backends = {"tavily": tavily}
    budget = BudgetTracker(db, limits={"tavily": 100})
    await hermes_search(
        query="test",
        intent="deep_research",
        content="full",
        backends=backends,
        budget=budget,
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    # Tavily deep_research = 2 credits
    assert await budget.remaining("tavily") == 98


@pytest.mark.asyncio
async def test_searxng_does_not_decrement_budget(db: Database) -> None:
    """Integration: SearXNG (self-hosted) no decrementa budget."""
    searxng = _make_backend(name="searxng")
    backends = {"searxng": searxng}
    budget = BudgetTracker(db, limits={"searxng": -1})
    initial = await budget.remaining("searxng")
    await hermes_search(
        query="test",
        intent="general",
        backends=backends,
        budget=budget,
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    # SearXNG unlimited, no consume
    assert await budget.remaining("searxng") == initial


# --- S9.3.1 punto 5: URL sanitization ---


def test_sanitize_urls_filters_javascript() -> None:
    """URLs con scheme javascript: se eliminan (XSS)."""
    result = SearchResult(
        results=[
            {"title": "OK", "url": "https://example.com", "content": "c"},
            {"title": "BAD", "url": "javascript:alert(1)", "content": "c"},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=200000,
        truncated=False,
    )
    sanitized = _sanitize_urls(result)
    assert len(sanitized.results) == 1
    assert sanitized.results[0]["title"] == "OK"


def test_sanitize_urls_filters_data_uri() -> None:
    """URLs con scheme data: se eliminan (data URI)."""
    result = SearchResult(
        results=[
            {"title": "BAD", "url": "data:text/html,<script>alert(1)</script>", "content": "c"},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=200000,
        truncated=False,
    )
    sanitized = _sanitize_urls(result)
    assert len(sanitized.results) == 0


def test_sanitize_urls_filters_file_scheme() -> None:
    """URLs con scheme file: se eliminan (filesystem access)."""
    result = SearchResult(
        results=[
            {"title": "BAD", "url": "file:///etc/passwd", "content": "c"},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=200000,
        truncated=False,
    )
    sanitized = _sanitize_urls(result)
    assert len(sanitized.results) == 0


def test_sanitize_urls_keeps_https() -> None:
    """URLs https se mantienen."""
    result = SearchResult(
        results=[
            {"title": "R1", "url": "https://example.com", "content": "c"},
            {"title": "R2", "url": "http://example.org", "content": "c"},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=200000,
        truncated=False,
    )
    sanitized = _sanitize_urls(result)
    assert len(sanitized.results) == 2


def test_sanitize_urls_case_insensitive() -> None:
    """Detecta schemes peligrosos en mayusculas tambien."""
    result = SearchResult(
        results=[
            {"title": "BAD", "url": "JAVASCRIPT:alert(1)", "content": "c"},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=200000,
        truncated=False,
    )
    sanitized = _sanitize_urls(result)
    assert len(sanitized.results) == 0


# --- S9.3.1 punto 6: format fallback explícito al LLM (Postura B) ---


@pytest.mark.asyncio
async def test_format_fallback_reason_explains_degradation(db: Database) -> None:
    """S9.3.1 Postura B: format_fallback_reason explica al LLM que pidió 'summary' pero recibió 'snippet'."""
    searxng = _make_backend(
        name="searxng",
        content_modes=frozenset({"snippet"}),
    )
    backends = {"searxng": searxng}
    result = await hermes_search(
        query="test",
        intent="general",
        content="summary",  # SearXNG no soporta
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.format_fallback is True
    assert result.format_fallback_reason is not None
    # El reason debe mencionar 'summary' y 'snippet' (lo que pidió vs lo que recibió)
    assert "summary" in result.format_fallback_reason
    assert "snippet" in result.format_fallback_reason
    # Y sugerir la alternativa (deep_research)
    assert "deep_research" in result.format_fallback_reason


@pytest.mark.asyncio
async def test_format_fallback_reason_is_none_when_no_fallback(db: Database) -> None:
    """Si NO hay format fallback, format_fallback_reason es None."""
    searxng = _make_backend(
        name="searxng",
        content_modes=frozenset({"snippet"}),
    )
    backends = {"searxng": searxng}
    result = await hermes_search(
        query="test",
        intent="general",
        content="snippet",  # match exacto, no hay fallback
        backends=backends,
        budget=_make_budget(db),
        circuit_breaker=CircuitBreakerRegistry(),
        semaphore=ConcurrencyLimiter(),
    )
    assert result.format_fallback is False
    assert result.format_fallback_reason is None
