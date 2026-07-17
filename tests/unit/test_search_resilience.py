"""Tests Sprint 9.3: CircuitBreakerRegistry y ConcurrencyLimiter (Capa 11.1).

Cubre:
CircuitBreaker:
- Inicialmente closed (is_open=False, is_closed=True)
- Tras N fails, abre (is_open=True)
- Después de TTL, half-open (is_open=False, is_closed=True)
- Probe success → closed, fails reseteados
- Probe fail → open de nuevo
- record_success en estado closed resetea contador de fails

ConcurrencyLimiter:
- get_semaphore retorna asyncio.Semaphore nativo
- 3 acquires concurrentes OK, 4to bloquea
- Backends independientes tienen semaforos separados
"""

from __future__ import annotations

import asyncio

import pytest

from hermes.services.search.resilience import (
    CircuitBreakerRegistry,
    ConcurrencyLimiter,
)

# --- CircuitBreakerRegistry ---


@pytest.mark.asyncio
async def test_circuit_breaker_initially_closed() -> None:
    """Nuevo circuit breaker esta closed por default."""
    cb = CircuitBreakerRegistry(threshold=3, ttl_seconds=300)
    assert cb.is_open("tavily") is False
    assert cb.is_closed("tavily") is True


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold() -> None:
    """Tras N fails consecutivos, el circuit abre."""
    cb = CircuitBreakerRegistry(threshold=3, ttl_seconds=300)
    cb.record_failure("tavily")
    cb.record_failure("tavily")
    assert cb.is_open("tavily") is False  # 2 fails, threshold=3
    cb.record_failure("tavily")
    assert cb.is_open("tavily") is True  # 3 fails → open


@pytest.mark.asyncio
async def test_circuit_breaker_success_resets_fails() -> None:
    """Un success en estado closed resetea el contador de fails."""
    cb = CircuitBreakerRegistry(threshold=3, ttl_seconds=300)
    cb.record_failure("tavily")
    cb.record_failure("tavily")
    cb.record_success("tavily")
    # Despues del success, fails reseteados. Necesitamos 3 fails nuevos para abrir.
    cb.record_failure("tavily")
    cb.record_failure("tavily")
    assert cb.is_open("tavily") is False


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_after_ttl() -> None:
    """Despues de TTL, el circuit pasa a half-open (puede ser llamado)."""
    cb = CircuitBreakerRegistry(threshold=2, ttl_seconds=0.01)  # 10ms TTL
    cb.record_failure("tavily")
    cb.record_failure("tavily")
    assert cb.is_open("tavily") is True
    # Esperar a que TTL expire
    await asyncio.sleep(0.02)
    # Half-open: is_open=False pero is_closed=True (permite un probe)
    assert cb.is_open("tavily") is False
    assert cb.is_closed("tavily") is True


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_success_closes() -> None:
    """Probe success en half-open → closed, fails reseteados."""
    cb = CircuitBreakerRegistry(threshold=2, ttl_seconds=0.01)
    cb.record_failure("tavily")
    cb.record_failure("tavily")
    await asyncio.sleep(0.02)  # TTL expira
    # Half-open: un probe success debe cerrar el circuit
    cb.record_success("tavily")
    # Despues del success, no debe abrir con solo 1 fail nuevo
    cb.record_failure("tavily")
    assert cb.is_open("tavily") is False  # solo 1 fail, threshold=2


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure_reopens() -> None:
    """Probe fail en half-open → open de nuevo."""
    cb = CircuitBreakerRegistry(threshold=2, ttl_seconds=0.01)
    cb.record_failure("tavily")
    cb.record_failure("tavily")
    await asyncio.sleep(0.02)
    # Half-open: probe fail debe reabrir
    cb.record_failure("tavily")
    assert cb.is_open("tavily") is True


@pytest.mark.asyncio
async def test_circuit_breaker_independent_per_backend() -> None:
    """Backends tienen circuits independientes."""
    cb = CircuitBreakerRegistry(threshold=2, ttl_seconds=300)
    cb.record_failure("tavily")
    cb.record_failure("tavily")
    # Tavily abierto, Exa cerrado
    assert cb.is_open("tavily") is True
    assert cb.is_open("exa") is False


# --- ConcurrencyLimiter ---


@pytest.mark.asyncio
async def test_concurrency_limiter_get_semaphore_returns_native() -> None:
    """get_semaphore retorna asyncio.Semaphore nativo (compatible con async with)."""
    limiter = ConcurrencyLimiter(limits={"tavily": 3})
    sem = limiter.get_semaphore("tavily")
    assert isinstance(sem, asyncio.Semaphore)


@pytest.mark.asyncio
async def test_concurrency_limiter_third_acquire_blocks() -> None:
    """4 acquires: 3 OK, 4to bloquea hasta release."""
    limiter = ConcurrencyLimiter(limits={"tavily": 3})
    sem = limiter.get_semaphore("tavily")
    acquired = []

    async def hold() -> None:
        async with sem:
            acquired.append("enter")
            await asyncio.sleep(0.05)
            acquired.append("exit")

    # Lanzar 4 tasks
    tasks = [asyncio.create_task(hold()) for _ in range(4)]
    await asyncio.sleep(0.01)  # dar tiempo a que entren
    # 3 deberian estar dentro, 1 esperando
    assert len([x for x in acquired if x == "enter"]) == 3
    # Esperar a que todos terminen
    await asyncio.gather(*tasks)
    assert len(acquired) == 8  # 4 enters + 4 exits


@pytest.mark.asyncio
async def test_concurrency_limiter_independent_per_backend() -> None:
    """Backends tienen semaforos independientes (3 tavily + 3 exa simultaneos)."""
    limiter = ConcurrencyLimiter(limits={"tavily": 3, "exa": 3})
    sem_tavily = limiter.get_semaphore("tavily")
    sem_exa = limiter.get_semaphore("exa")

    async def hold_tavily() -> None:
        async with sem_tavily:
            await asyncio.sleep(0.05)

    async def hold_exa() -> None:
        async with sem_exa:
            await asyncio.sleep(0.05)

    # 2 tavily + 2 exa simultaneos (no compiten entre si)
    await asyncio.gather(hold_tavily(), hold_tavily(), hold_exa(), hold_exa())
    # Si no fueran independientes, habria bloqueos


@pytest.mark.asyncio
async def test_concurrency_limiter_unknown_backend_creates_default() -> None:
    """get_semaphore para backend desconocido crea un semaphore con max_concurrent default (6)."""
    limiter = ConcurrencyLimiter(limits={"tavily": 2})
    sem = limiter.get_semaphore("unknown_backend")
    assert isinstance(sem, asyncio.Semaphore)
    # Puede ser adquirido
    async with sem:
        pass


# --- S9.3.1 punto 2: per-backend limits ---


def test_concurrency_limiter_per_backend_defaults() -> None:
    """S9.3.1: ConcurrencyLimiter() con defaults per-backend (SearXNG=6, Tavily=10, Exa=5)."""
    limiter = ConcurrencyLimiter()
    assert limiter.get_max_concurrent("searxng") == 6
    assert limiter.get_max_concurrent("tavily") == 10
    assert limiter.get_max_concurrent("exa") == 5


def test_concurrency_limiter_per_backend_custom() -> None:
    """S9.3.1: limits custom sobrescribe defaults per-backend."""
    limiter = ConcurrencyLimiter(limits={"searxng": 2, "tavily": 20, "exa": 8})
    assert limiter.get_max_concurrent("searxng") == 2
    assert limiter.get_max_concurrent("tavily") == 20
    assert limiter.get_max_concurrent("exa") == 8


def test_concurrency_limiter_legacy_max_concurrent() -> None:
    """S9.3.1: backward compat con ConcurrencyLimiter(max_concurrent=N)."""
    limiter = ConcurrencyLimiter(max_concurrent=3)
    assert limiter.get_max_concurrent("tavily") == 3
    assert limiter.get_max_concurrent("searxng") == 3
    assert limiter.get_max_concurrent("unknown") == 3


@pytest.mark.asyncio
async def test_concurrency_limiter_per_backend_independent() -> None:
    """SearXNG=2 no bloquea Tavily=10 (independencia total)."""
    limiter = ConcurrencyLimiter(limits={"searxng": 2, "tavily": 10})

    # Saturamos searxng (2 slots)
    searxng_sem = limiter.get_semaphore("searxng")
    tavily_sem = limiter.get_semaphore("tavily")

    async def hold_searxng() -> None:
        async with searxng_sem:
            await asyncio.sleep(0.1)

    async def hold_tavily() -> None:
        async with tavily_sem:
            await asyncio.sleep(0.05)

    # 2 searxng + 3 tavily simultaneas: tavily no debe esperar
    import time

    start = time.monotonic()
    await asyncio.gather(
        hold_searxng(),
        hold_searxng(),
        hold_tavily(),
        hold_tavily(),
        hold_tavily(),
    )
    elapsed = time.monotonic() - start
    # Tavily termina en 0.05s, searxng en 0.1s. Total ~0.1s (max de ambos)
    assert elapsed < 0.15  # no hay bloqueo cruzado
