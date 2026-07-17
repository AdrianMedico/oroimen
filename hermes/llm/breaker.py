"""Circuit breaker minimalista (3 estados: closed/half_open/open).

Implementación propia para evitar dependencias externas con bugs en
Python 3.14. API compatible con el uso que hace LLMRouter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    HALF_OPEN = "half-open"
    OPEN = "open"


class CircuitOpenError(Exception):
    """Lanzada cuando el breaker está abierto y se intenta llamar."""


class CircuitBreaker:
    def __init__(
        self,
        *,
        fail_max: int = 5,
        reset_timeout: float = 60.0,
        name: str = "breaker",
    ) -> None:
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.name = name
        self._state = CircuitState.CLOSED
        self._fail_count = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def current_state(self) -> str:
        return self._state.value

    async def call(self, coro_factory: Callable[[], Awaitable[T]]) -> T:
        """Ejecuta la coroutine si el breaker lo permite."""
        async with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.OPEN:
                raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        try:
            result = await coro_factory()
        except Exception:
            await self._on_failure()
            raise
        else:
            await self._on_success()
            return result

    def _maybe_half_open(self) -> None:
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.reset_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            logger.info("circuit_half_open", extra={"breaker": self.name})

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state is not CircuitState.CLOSED:
                logger.info("circuit_closed", extra={"breaker": self.name})
            self._state = CircuitState.CLOSED
            self._fail_count = 0
            self._opened_at = None

    async def _on_failure(self) -> None:
        async with self._lock:
            self._fail_count += 1
            if self._state is CircuitState.HALF_OPEN or self._fail_count >= self.fail_max:
                if self._state is not CircuitState.OPEN:
                    logger.warning(
                        "circuit_opened",
                        extra={
                            "breaker": self.name,
                            "fails": self._fail_count,
                        },
                    )
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
