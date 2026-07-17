"""Tests Sprint 9.3: BackendProtocol contract (Capa 1).

Cubre:
- SearchResult dataclass: inmutabilidad, defaults, equality, repr
- ContentMode: tipos correctos
- BackendProtocol: estructura, runtime_checkable
- MockBackend: implementacion valida del Protocol
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import ClassVar

import pytest

from hermes.services.search.protocol import (
    ALL_CONTENT_MODES,
    BackendProtocol,
    ContentMode,
    SearchResult,
)

# --- SearchResult dataclass ---


def test_search_result_is_frozen() -> None:
    """SearchResult es inmutable (frozen dataclass).

    Importante: el resultado se loggea en metricas y se pasa por
    multiples capas del router. Inmutabilidad previene mutaciones
    accidentales que harian el debugging imposible.
    """
    result = SearchResult(
        results=[],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    with pytest.raises(FrozenInstanceError):
        result.query = "modified"  # type: ignore[misc]


def test_search_result_default_truncated_at_chars_is_none() -> None:
    """Si no se trunca, truncated_at_chars es None por default.

    El LLM usa este campo para detectar truncado. Si es None,
    sabe que recibio el contenido completo.
    """
    result = SearchResult(
        results=[],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    assert result.truncated_at_chars is None


def test_search_result_equality_by_value() -> None:
    """Dataclass equality: dos SearchResult con mismos campos son iguales.

    Util en tests (assertEqual) y en deduplicacion cross-backend
    (si dos backends devuelven mismo resultado, se pueden comparar).
    """
    r1 = SearchResult(
        results=[{"title": "A"}],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    r2 = SearchResult(
        results=[{"title": "A"}],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    assert r1 == r2


def test_search_result_supports_repr() -> None:
    """__repr__ es informativo (debugging en logs)."""
    result = SearchResult(
        results=[{"title": "A"}],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    repr_str = repr(result)
    assert "SearchResult" in repr_str
    assert "searxng" in repr_str
    assert "snippet" in repr_str


def test_search_result_with_truncation() -> None:
    """SearchResult con truncated=True tiene truncated_at_chars != None."""
    result = SearchResult(
        results=[{"title": "A", "content": "truncated content..."}],
        backend_used="tavily",
        query="long query",
        content_mode="full",
        original_content_mode="full",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=True,
        truncated_at_chars=50000,
    )
    assert result.truncated is True
    assert result.truncated_at_chars == 50000


def test_search_result_with_format_fallback() -> None:
    """SearchResult con format_fallback=True tiene content_mode != original."""
    result = SearchResult(
        results=[{"title": "A", "content": "snippet text"}],
        backend_used="searxng",
        query="test",
        content_mode="snippet",  # degraded
        original_content_mode="summary",  # originally requested
        format_fallback=True,
        size_guard_chars=50000,
        truncated=False,
    )
    assert result.format_fallback is True
    assert result.content_mode == "snippet"
    assert result.original_content_mode == "summary"


# --- ContentMode ---


def test_all_content_modes_contains_three_known_modes() -> None:
    """ALL_CONTENT_MODES contiene exactamente snippet, summary, full."""
    assert frozenset({"snippet", "summary", "full"}) == ALL_CONTENT_MODES


def test_content_mode_literal_values() -> None:
    """ContentMode es un Literal type con los 3 valores esperados."""
    valid_modes: list[ContentMode] = ["snippet", "summary", "full"]
    for mode in valid_modes:
        assert mode in ALL_CONTENT_MODES


# --- BackendProtocol ---


def test_backend_protocol_is_runtime_checkable() -> None:
    """BackendProtocol es runtime_checkable: isinstance funciona.

    Importante: el router puede verificar que un mock implementa
    el contrato en tests, sin necesidad de herencia.
    """

    class FakeBackend:
        name: ClassVar[str] = "fake"
        SUPPORTED_CONTENT_MODES: ClassVar[frozenset[str]] = frozenset({"snippet"})

        async def search(
            self,
            query: str,
            content_mode: str,
            num_results: int,
            *,
            intent: str = "general",
        ) -> SearchResult:
            return SearchResult(
                results=[],
                backend_used=self.name,
                query=query,
                content_mode=content_mode,
                original_content_mode=content_mode,
                format_fallback=False,
                size_guard_chars=50000,
                truncated=False,
            )

        async def has_budget(self) -> bool:
            return True

        async def health_check(self) -> bool:
            return True

    fake = FakeBackend()
    assert isinstance(fake, BackendProtocol)


def test_backend_protocol_class_not_instance() -> None:
    """Una clase sin los metodos requeridos NO es instancia del Protocol."""

    class Incomplete:
        name: ClassVar[str] = "incomplete"
        # Faltan SUPPORTED_CONTENT_MODES, search, has_budget, health_check

    incomplete = Incomplete()
    assert not isinstance(incomplete, BackendProtocol)


# --- MockBackend: implementation valida del Protocol ---


class MockBackend:
    """Mock backend que respeta BackendProtocol.

    Usado en tests del router (Capa 6) para verificar dispatch
    polimorfico sin tocar red. Configurable para simular failures
    y circuit breaker scenarios.
    """

    name: ClassVar[str] = "mock"
    SUPPORTED_CONTENT_MODES: ClassVar[frozenset[str]] = frozenset({"snippet", "summary"})

    def __init__(
        self,
        should_fail: bool = False,
        budget_available: bool = True,
        healthy: bool = True,
    ) -> None:
        self.should_fail = should_fail
        self.budget_available = budget_available
        self.healthy = healthy
        self.search_call_count = 0
        self.last_query: str | None = None
        self.last_content_mode: str | None = None
        self.last_num_results: int | None = None

    async def search(
        self,
        query: str,
        content_mode: str,
        num_results: int,
        *,
        intent: str = "general",
    ) -> SearchResult:
        self.search_call_count += 1
        self.last_query = query
        self.last_content_mode = content_mode
        self.last_num_results = num_results
        self.last_intent = intent
        if self.should_fail:
            raise RuntimeError("mock failure: backend error")
        return SearchResult(
            results=[
                {
                    "title": f"Mock result for {query}",
                    "url": f"http://mock.example/{query}",
                    "content": f"Mock content in {content_mode} mode",
                }
            ],
            backend_used=self.name,
            query=query,
            content_mode=content_mode,
            original_content_mode=content_mode,
            format_fallback=False,
            size_guard_chars=50000,
            truncated=False,
        )

    async def has_budget(self) -> bool:
        return self.budget_available

    async def health_check(self) -> bool:
        return self.healthy


@pytest.mark.asyncio
async def test_mock_backend_returns_valid_search_result() -> None:
    """MockBackend implementa el Protocol y retorna SearchResult valido."""
    backend = MockBackend()
    result = await backend.search(
        query="test query",
        content_mode="snippet",
        num_results=5,
    )
    assert isinstance(result, SearchResult)
    assert result.backend_used == "mock"
    assert result.query == "test query"
    assert result.content_mode == "snippet"
    assert result.original_content_mode == "snippet"
    assert len(result.results) == 1


@pytest.mark.asyncio
async def test_mock_backend_declares_supported_content_modes() -> None:
    """MockBackend declara explicitamente que content modes soporta.

    El router usa SUPPORTED_CONTENT_MODES para Format Fallback.
    """
    backend = MockBackend()
    assert "snippet" in backend.SUPPORTED_CONTENT_MODES
    assert "summary" in backend.SUPPORTED_CONTENT_MODES
    assert "full" not in backend.SUPPORTED_CONTENT_MODES  # no soportado


@pytest.mark.asyncio
async def test_mock_backend_has_budget_configurable() -> None:
    """has_budget() del mock es configurable (True por default)."""
    assert await MockBackend().has_budget() is True
    assert await MockBackend(budget_available=False).has_budget() is False


@pytest.mark.asyncio
async def test_mock_backend_health_check_configurable() -> None:
    """health_check() del mock es configurable (True por default)."""
    assert await MockBackend().health_check() is True
    assert await MockBackend(healthy=False).health_check() is False


@pytest.mark.asyncio
async def test_mock_backend_can_simulate_failure() -> None:
    """MockBackend con should_fail=True lanza RuntimeError.

    Usado en tests del router para verificar circuit breaker
    y graceful degradation cuando un backend falla.
    """
    backend = MockBackend(should_fail=True)
    with pytest.raises(RuntimeError, match="mock failure"):
        await backend.search(
            query="test",
            content_mode="snippet",
            num_results=5,
        )


@pytest.mark.asyncio
async def test_mock_backend_tracks_call_count() -> None:
    """MockBackend cuenta las llamadas (util para tests de rate limit)."""
    backend = MockBackend()
    await backend.search(query="q1", content_mode="snippet", num_results=5)
    await backend.search(query="q2", content_mode="snippet", num_results=5)
    await backend.search(query="q3", content_mode="summary", num_results=10)
    assert backend.search_call_count == 3
    assert backend.last_query == "q3"
    assert backend.last_content_mode == "summary"
    assert backend.last_num_results == 10
