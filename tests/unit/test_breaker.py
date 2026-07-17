"""Tests para CircuitBreaker manual."""

from __future__ import annotations

import asyncio

import pytest

from hermes.llm.breaker import CircuitBreaker, CircuitOpenError


@pytest.mark.asyncio
async def test_starts_closed() -> None:
    cb = CircuitBreaker(fail_max=3, reset_timeout=0.1, name="t")
    assert cb.current_state == "closed"


@pytest.mark.asyncio
async def test_opens_after_fail_max() -> None:
    cb = CircuitBreaker(fail_max=3, reset_timeout=10.0, name="t")

    async def fail() -> None:
        raise RuntimeError("boom")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(fail)
    assert cb.current_state == "open"


@pytest.mark.asyncio
async def test_open_rejects_calls() -> None:
    cb = CircuitBreaker(fail_max=2, reset_timeout=10.0, name="t")

    async def fail() -> None:
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(fail)
    with pytest.raises(CircuitOpenError):
        await cb.call(fail)


@pytest.mark.asyncio
async def test_half_open_after_reset() -> None:
    cb = CircuitBreaker(fail_max=2, reset_timeout=0.1, name="t")

    async def fail() -> None:
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(fail)
    assert cb.current_state == "open"
    await asyncio.sleep(0.15)

    # Siguiente llamada: half-open, success → closed
    async def ok() -> str:
        return "ok"

    result = await cb.call(ok)
    assert result == "ok"
    assert cb.current_state == "closed"


@pytest.mark.asyncio
async def test_success_resets_fail_counter() -> None:
    cb = CircuitBreaker(fail_max=3, reset_timeout=10.0, name="t")

    async def fail() -> None:
        raise RuntimeError("boom")

    async def ok() -> str:
        return "ok"

    with pytest.raises(RuntimeError):
        await cb.call(fail)
    await cb.call(ok)
    assert cb.current_state == "closed"
