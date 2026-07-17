"""Sprint 9.3: CircuitBreakerRegistry y ConcurrencyLimiter (Capa 11.1).

Resiliencia operacional para el Web Search Router:
- CircuitBreakerRegistry: 3 fails -> open por TTL -> half-open -> probe
- ConcurrencyLimiter: asyncio.Semaphore per-backend, max_concurrent configurable
"""

from __future__ import annotations

import asyncio
import logging
import time
from threading import Lock
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class CircuitBreakerRegistry:
    """Circuit breaker per-backend con estados closed/open/half-open.

    Estados:
    - closed: backend operacional, requests pasan.
    - open: backend caido, requests NO pasan (fail fast).
    - half-open: despues de TTL, permite UN probe para testear.

    Transiciones:
    - closed -> open: tras `threshold` fails consecutivos.
    - open -> half-open: despues de `ttl_seconds`.
    - half-open -> closed: probe (request) success.
    - half-open -> open: probe (request) fail.

    Thread-safe: usa threading.Lock porque has_budget/record_usage
    son async pero el state mutation es in-memory rapido.
    """

    _FAIL_KEY: ClassVar[str] = "fails"

    def __init__(self, threshold: int = 3, ttl_seconds: int = 300) -> None:
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def is_open(self, backend: str) -> bool:
        """True si el circuit esta open (fail fast, NO llamar al backend).

        Side effect: si TTL ha expirado, transiciona el state a
        half-open y resetea opened_at. Esto evita reabrir
        inmediatamente en el siguiente record_failure sin un probe.
        """
        with self._lock:
            entry = self._state.get(backend)
            if entry is None:
                return False  # closed by default
            # Check TTL: si ya expiro, transiciona a half-open
            if (
                entry.get("state") == "open"
                and entry.get("opened_at", 0) + self._ttl_seconds <= time.monotonic()
            ):
                # Half-open: NO esta open (permite probe)
                entry["state"] = "half-open"
                self._state[backend] = entry
                logger.info(
                    "circuit_breaker_half_open",
                    extra={"backend": backend},
                )
                return False
            return entry.get("state") == "open"

    def is_closed(self, backend: str) -> bool:
        """True si el circuit esta closed u half-open (puede llamar)."""
        return not self.is_open(backend)

    def record_success(self, backend: str) -> None:
        """Registrar exito: reset fails, transition a closed.

        Llamado despues de un search() exitoso. Si el circuit estaba
        half-open, esto lo cierra. Si estaba closed, resetea el
        contador de fails.
        """
        with self._lock:
            entry = self._state.get(backend, {})
            entry[self._FAIL_KEY] = 0
            entry["state"] = "closed"
            entry["opened_at"] = 0
            self._state[backend] = entry
            logger.debug(
                "circuit_breaker_success",
                extra={"backend": backend, "fails": 0},
            )

    def record_failure(self, backend: str) -> None:
        """Registrar fallo: increment fails, abrir si >= threshold.

        Si el circuit estaba half-open y recibe failure, reabre
        inmediatamente (sin esperar threshold).
        """
        with self._lock:
            entry = self._state.get(backend, {})
            current_state = entry.get("state", "closed")
            if current_state == "half-open":
                # Probe fail: reabrir inmediatamente
                entry["state"] = "open"
                entry["opened_at"] = time.monotonic()
                self._state[backend] = entry
                logger.warning(
                    "circuit_breaker_reopened",
                    extra={"backend": backend, "reason": "probe_fail"},
                )
                return
            # closed: increment fails
            fails = entry.get(self._FAIL_KEY, 0) + 1
            entry[self._FAIL_KEY] = fails
            if fails >= self._threshold:
                entry["state"] = "open"
                entry["opened_at"] = time.monotonic()
                logger.warning(
                    "circuit_breaker_opened",
                    extra={
                        "backend": backend,
                        "fails": fails,
                        "threshold": self._threshold,
                    },
                )
            self._state[backend] = entry


class ConcurrencyLimiter:
    """Semáforo per-backend usando asyncio.Semaphore.

    P0-3 fix v1.1: get_semaphore() retorna la instancia nativa de
    asyncio.Semaphore, compatible con `async with`. NO usar
    `async with semaphore.acquire()` (eso es misuse de API).

    S9.3.1 punto 2: límites per-backend en lugar de un único max_concurrent.
    Razón: cada upstream tiene su rate limit:
    - SearXNG: depende de engines upstream (DuckDuckGo ~5 req/s). 6 max.
    - Tavily: 100 req/min documented. 10 max (margen conservador).
    - Exa: 50 req/min documented. 5 max (margen conservador).
    """

    _DEFAULTS: ClassVar[dict[str, int]] = {
        "searxng": 6,
        "tavily": 10,
        "exa": 5,
    }

    def __init__(
        self,
        limits: dict[str, int] | None = None,
        *,
        max_concurrent: int | None = None,
    ) -> None:
        """Inicializa con límites per-backend.

        Args:
            limits: dict[backend_name, max_concurrent]. Si None, usa defaults.
            max_concurrent: legacy (S9.3.0). Si se pasa, se aplica como
                límite para todos los backends (compat hacia atras).

        Uso:
            # S9.3.1: per-backend
            ConcurrencyLimiter(limits={"searxng": 6, "tavily": 10, "exa": 5})
            # S9.3.0 legacy
            ConcurrencyLimiter(max_concurrent=3)
        """
        if max_concurrent is not None:
            # Legacy mode: todos los backends comparten el mismo límite
            self._limits = {"__default__": max_concurrent}
            self._max_concurrent_default = max_concurrent
        else:
            self._limits = limits or self._DEFAULTS
            self._max_concurrent_default = self._limits.get("__default__", 6)
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    def get_semaphore(self, backend: str) -> asyncio.Semaphore:
        """Retorna el asyncio.Semaphore para el backend dado.

        Si el backend no tiene semaphore, se crea con el límite
        per-backend (o default si no está en limits). Crea lazily
        para evitar crear semaforos para backends que nunca se usan.

        Uso:
            async with limiter.get_semaphore("tavily"):
                await backend.search(...)
        """
        if backend not in self._semaphores:
            max_concurrent = self._limits.get(backend, self._max_concurrent_default)
            self._semaphores[backend] = asyncio.Semaphore(max_concurrent)
        return self._semaphores[backend]

    def get_max_concurrent(self, backend: str) -> int:
        """Retorna el max_concurrent configurado para el backend."""
        return self._limits.get(backend, self._max_concurrent_default)
