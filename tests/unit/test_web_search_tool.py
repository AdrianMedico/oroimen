"""Tests Sprint 9.3: MCP tool web_search (Capa 7).

Cubre:
- SEARCH_TOOL_SCHEMA: formato OpenAI, no leakea backends
- make_search_tool_callable: callable async con dependencias inyectadas
- _serialize_result: SearchResult -> dict JSON-serializable
- num_results clamping (default + max)
"""

from __future__ import annotations

import asyncio
import inspect
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
from hermes.tools.web_search import (
    SEARCH_TOOL_SCHEMA,
    _serialize_result,
    make_search_tool_callable,
)


@pytest.fixture
async def db() -> Database:
    with tempfile.TemporaryDirectory() as td:
        d = Database(Path(td) / "test.db")
        await d.initialize()
        yield d
        await d.close()


# --- SEARCH_TOOL_SCHEMA ---


def test_schema_is_openai_function_format() -> None:
    """Schema es formato OpenAI Chat Completions (type=function)."""
    assert SEARCH_TOOL_SCHEMA["type"] == "function"
    assert "function" in SEARCH_TOOL_SCHEMA
    fn = SEARCH_TOOL_SCHEMA["function"]
    assert fn["name"] == "hermes_search"
    assert "description" in fn
    assert "parameters" in fn


def test_schema_description_does_not_leak_backend_names() -> None:
    """P2-1 v1.2: description NO menciona 'SearXNG', 'Tavily', 'Exa'."""
    description = SEARCH_TOOL_SCHEMA["function"]["description"].lower()
    for backend in ("searxng", "tavily", "exa"):
        assert (
            backend not in description
        ), f"Description leakea backend name '{backend}': {description}"


def test_schema_has_required_query_param() -> None:
    """query es el unico param required."""
    params = SEARCH_TOOL_SCHEMA["function"]["parameters"]
    assert params["required"] == ["query"]
    assert "query" in params["properties"]


def test_schema_intent_enum_has_three_values() -> None:
    """intent enum: general, semantic, deep_research."""
    params = SEARCH_TOOL_SCHEMA["function"]["parameters"]
    assert params["properties"]["intent"]["enum"] == ["general", "semantic", "deep_research"]


def test_schema_content_enum_has_three_values() -> None:
    """content enum: snippet, summary, full."""
    params = SEARCH_TOOL_SCHEMA["function"]["parameters"]
    assert params["properties"]["content"]["enum"] == ["snippet", "summary", "full"]


def test_schema_num_results_range() -> None:
    """num_results: min 1, max 50, default 10."""
    params = SEARCH_TOOL_SCHEMA["function"]["parameters"]
    n = params["properties"]["num_results"]
    assert n["minimum"] == 1
    assert n["maximum"] == 50
    assert n["default"] == 10


# --- make_search_tool_callable ---


@pytest.mark.asyncio
async def test_make_search_tool_callable_returns_callable(db: Database) -> None:
    """make_search_tool_callable retorna un callable async."""
    budget = BudgetTracker(db, limits={"searxng": -1})
    cb = CircuitBreakerRegistry()
    sem = ConcurrencyLimiter()
    backends = {"searxng": MagicMock(name="searxng")}

    fn = make_search_tool_callable(
        backends=backends,
        budget=budget,
        circuit_breaker=cb,
        semaphore=sem,
    )
    assert callable(fn)
    assert inspect.iscoroutinefunction(fn)


@pytest.mark.asyncio
async def test_callable_returns_serializable_dict(db: Database) -> None:
    """El callable retorna un dict JSON-serializable (no dataclass)."""
    import json

    budget = BudgetTracker(db, limits={"searxng": -1})
    cb = CircuitBreakerRegistry()
    sem = ConcurrencyLimiter()

    # Mock backend que retorna SearchResult valido
    searxng = MagicMock()
    searxng.SUPPORTED_CONTENT_MODES = frozenset({"snippet"})
    searxng.name = "searxng"

    async def fake_search(query, content_mode, num_results, *, intent="general"):
        return SearchResult(
            results=[{"title": "Test", "url": "https://x.com", "content": "c"}],
            backend_used="searxng",
            query=query,
            content_mode=content_mode,
            original_content_mode=content_mode,
            format_fallback=False,
            size_guard_chars=50000,
            truncated=False,
        )

    searxng.search = fake_search
    searxng.has_budget = MagicMock(return_value=asyncio.Future())
    searxng.has_budget.return_value.set_result(True)
    backends = {"searxng": searxng}

    fn = make_search_tool_callable(
        backends=backends,
        budget=budget,
        circuit_breaker=cb,
        semaphore=sem,
    )
    result = await fn(query="test", intent="general")
    # Debe ser dict JSON-serializable
    json_str = json.dumps(result)
    assert "Test" in json_str


# --- _serialize_result ---


def test_serialize_result_with_error() -> None:
    """SearchResult con error retorna dict success=False + error fields."""
    from hermes.services.search.errors import (
        SearchErrorCode,
        _build_structured_error,
    )

    error = _build_structured_error(
        code=SearchErrorCode.TIMEOUT,
        message="Timeout after 10s",
        backend="tavily",
        retryable=True,
    )
    result = SearchResult(
        results=[],
        backend_used="none",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
        error=error,
    )
    serialized = _serialize_result(result)
    assert serialized["success"] is False
    assert serialized["code"] == "TIMEOUT"
    assert serialized["backend"] == "tavily"
    assert serialized["retryable"] is True


def test_serialize_result_with_results() -> None:
    """SearchResult con results retorna success=True + results."""
    result = SearchResult(
        results=[
            {"title": "R1", "url": "u1", "content": "c1"},
            {"title": "R2", "url": "u2", "content": "c2"},
        ],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
    )
    serialized = _serialize_result(result)
    assert serialized["success"] is True
    assert len(serialized["results"]) == 2
    assert serialized["results"][0]["title"] == "R1"
    assert serialized["backend_used"] == "searxng"
    assert serialized["truncated"] is False


def test_serialize_result_with_truncation() -> None:
    """SearchResult con truncated=True incluye truncated_at_chars."""
    result = SearchResult(
        results=[{"title": "R1", "url": "u1", "content": "x" * 100}],
        backend_used="searxng",
        query="test",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=True,
        truncated_at_chars=100,
    )
    serialized = _serialize_result(result)
    assert serialized["truncated"] is True
    assert serialized["truncated_at_chars"] == 100


def test_serialize_result_with_format_fallback() -> None:
    """SearchResult con format_fallback=True tiene ambos modes."""
    result = SearchResult(
        results=[{"title": "R1", "url": "u1", "content": "c"}],
        backend_used="searxng",
        query="test",
        content_mode="snippet",  # degraded
        original_content_mode="summary",  # originally requested
        format_fallback=True,
        size_guard_chars=50000,
        truncated=False,
    )
    serialized = _serialize_result(result)
    assert serialized["format_fallback"] is True
    assert serialized["content_mode"] == "snippet"
    assert serialized["original_content_mode"] == "summary"


# --- num_results clamping ---


@pytest.mark.asyncio
async def test_num_results_clamped_to_max(db: Database) -> None:
    """num_results > max_num_results se clampa al max."""
    budget = BudgetTracker(db, limits={"searxng": -1})
    cb = CircuitBreakerRegistry()
    sem = ConcurrencyLimiter()
    searxng = MagicMock()
    searxng.SUPPORTED_CONTENT_MODES = frozenset({"snippet"})

    captured: list[dict] = []

    async def fake_search(query, content_mode, num_results, *, intent="general"):
        captured.append({"query": query, "n": num_results})
        return SearchResult(
            results=[],
            backend_used="searxng",
            query=query,
            content_mode=content_mode,
            original_content_mode=content_mode,
            format_fallback=False,
            size_guard_chars=50000,
            truncated=False,
        )

    searxng.search = fake_search
    searxng.has_budget = MagicMock(return_value=asyncio.Future())
    searxng.has_budget.return_value.set_result(True)
    backends = {"searxng": searxng}

    fn = make_search_tool_callable(
        backends=backends,
        budget=budget,
        circuit_breaker=cb,
        semaphore=sem,
        max_num_results=25,
    )
    await fn(query="test", intent="general", num_results=100)
    assert captured[0]["n"] == 25  # clamped


@pytest.mark.asyncio
async def test_num_defaults_when_not_passed(db: Database) -> None:
    """num_results=None usa default_num_results."""
    budget = BudgetTracker(db, limits={"searxng": -1})
    cb = CircuitBreakerRegistry()
    sem = ConcurrencyLimiter()
    searxng = MagicMock()
    searxng.SUPPORTED_CONTENT_MODES = frozenset({"snippet"})

    captured: list[dict] = []

    async def fake_search(query, content_mode, num_results, *, intent="general"):
        captured.append({"n": num_results})
        return SearchResult(
            results=[],
            backend_used="searxng",
            query=query,
            content_mode=content_mode,
            original_content_mode=content_mode,
            format_fallback=False,
            size_guard_chars=50000,
            truncated=False,
        )

    searxng.search = fake_search
    searxng.has_budget = MagicMock(return_value=asyncio.Future())
    searxng.has_budget.return_value.set_result(True)
    backends = {"searxng": searxng}

    fn = make_search_tool_callable(
        backends=backends,
        budget=budget,
        circuit_breaker=cb,
        semaphore=sem,
        default_num_results=15,
    )
    await fn(query="test", intent="general")
    assert captured[0]["n"] == 15
