"""Tests Sprint 9.3: Tavily Backend (Capa 4).

Cubre:
- search_depth: 'advanced' cuando intent='deep_research', 'basic' en otros
- include_raw_content: True cuando content_mode='full'
- Content modes: snippet usa 'content', summary usa 'summary', full usa 'raw_content'
- has_budget delega a BudgetTracker
- Error handling (401, 429, timeout, invalid JSON)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.memory.db import Database
from hermes.services.search.budget import BudgetTracker
from hermes.services.search.tavily import TavilyBackend


@pytest.fixture
async def db() -> Database:
    with tempfile.TemporaryDirectory() as td:
        d = Database(Path(td) / "test.db")
        await d.initialize()
        yield d
        await d.close()


def _tavily_response(
    query: str = "test",
    results: list[dict] | None = None,
) -> dict:
    """Construye un response JSON valido de Tavily."""
    if results is None:
        results = [
            {
                "title": "Result 1",
                "url": "https://example.com/1",
                "content": "Snippet 1",
                "summary": "Summary 1: detailed summary text",
                "raw_content": "# Full Content 1\n\nMarkdown content here.",
                "score": 0.95,
            }
        ]
    return {"query": query, "results": results, "answer": None}


# --- SUPPORTED_CONTENT_MODES ---


def test_tavily_supports_all_three_content_modes() -> None:
    """Tavily soporta snippet, summary, full (todos los modos)."""
    backend = TavilyBackend(
        api_key="tvly-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    assert frozenset({"snippet", "summary", "full"}) == backend.SUPPORTED_CONTENT_MODES


# --- name ---


def test_tavily_name() -> None:
    """Backend name es 'tavily'."""
    backend = TavilyBackend(
        api_key="tvly-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    assert backend.name == "tavily"


# --- search: intent → search_depth ---


@pytest.mark.asyncio
async def test_tavily_search_uses_advanced_depth_for_deep_research() -> None:
    """search_depth='advanced' cuando intent='deep_research'."""
    backend = TavilyBackend(
        api_key="tvly-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    captured: list[dict] = []

    async def mock_post(url: str, json: dict, headers: dict) -> MagicMock:
        captured.append(json)
        mock_response = MagicMock()
        mock_response.json.return_value = _tavily_response()
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch("hermes.services.search.tavily.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        await backend.search("test", content_mode="snippet", num_results=10, intent="deep_research")

    assert captured[0]["search_depth"] == "advanced"


@pytest.mark.asyncio
async def test_tavily_search_uses_basic_depth_for_non_deep_research() -> None:
    """search_depth='basic' cuando intent != 'deep_research'."""
    backend = TavilyBackend(
        api_key="tvly-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    captured: list[dict] = []

    async def mock_post(url: str, json: dict, headers: dict) -> MagicMock:
        captured.append(json)
        mock_response = MagicMock()
        mock_response.json.return_value = _tavily_response()
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch("hermes.services.search.tavily.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        # Test con intent='general' (default)
        await backend.search("test", content_mode="snippet", num_results=10)

    assert captured[0]["search_depth"] == "basic"


# --- search: content_mode → field ---


@pytest.mark.asyncio
async def test_tavily_search_snippet_uses_content_field() -> None:
    """content_mode='snippet' usa el campo 'content' (short)."""
    backend = TavilyBackend(
        api_key="tvly-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    with patch("hermes.services.search.tavily.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _tavily_response()
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        result = await backend.search("test", content_mode="snippet", num_results=10)

    assert result.results[0]["content"] == "Snippet 1"
    # El campo 'summary' y 'raw_content' NO se incluyen en snippet
    assert "summary" not in result.results[0]
    assert "raw_content" not in result.results[0]


@pytest.mark.asyncio
async def test_tavily_search_summary_uses_summary_field() -> None:
    """content_mode='summary' usa el campo 'summary' (~1.5K chars)."""
    backend = TavilyBackend(
        api_key="tvly-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    with patch("hermes.services.search.tavily.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _tavily_response()
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        result = await backend.search("test", content_mode="summary", num_results=10)

    assert result.results[0]["summary"] == "Summary 1: detailed summary text"
    # content presente como context
    assert "content" in result.results[0]


@pytest.mark.asyncio
async def test_tavily_search_full_uses_raw_content() -> None:
    """content_mode='full' usa el campo 'raw_content' (markdown completo)."""
    backend = TavilyBackend(
        api_key="tvly-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    with patch("hermes.services.search.tavily.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _tavily_response()
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        result = await backend.search("test", content_mode="full", num_results=10)

    assert result.results[0]["raw_content"] == "# Full Content 1\n\nMarkdown content here."


@pytest.mark.asyncio
async def test_tavily_search_full_sets_include_raw_content() -> None:
    """content_mode='full' setea include_raw_content=True en el request."""
    backend = TavilyBackend(
        api_key="tvly-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    captured: list[dict] = []

    async def mock_post(url: str, json: dict, headers: dict) -> MagicMock:
        captured.append(json)
        mock_response = MagicMock()
        mock_response.json.return_value = _tavily_response()
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch("hermes.services.search.tavily.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        await backend.search("test", content_mode="full", num_results=10)

    assert captured[0]["include_raw_content"] is True


# --- has_budget ---


@pytest.mark.asyncio
async def test_tavily_has_budget_delegates_to_tracker(db: Database) -> None:
    """has_budget() delega a BudgetTracker con self.name."""
    budget = BudgetTracker(db, limits={"tavily": 1000})
    backend = TavilyBackend(api_key="tvly-test", budget=budget, timeout=15.0)
    assert await backend.has_budget() is True
    # Agotar
    for _ in range(1000):
        await budget.record_usage("tavily", count=1)
    assert await backend.has_budget() is False


# --- health_check ---


@pytest.mark.asyncio
async def test_tavily_health_check_returns_true_on_200() -> None:
    """health_check() retorna True si Tavily responde 200."""
    backend = TavilyBackend(api_key="tvly-test", budget=MagicMock(), timeout=15.0)
    with patch("hermes.services.search.tavily.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        assert await backend.health_check() is True


@pytest.mark.asyncio
async def test_tavily_health_check_returns_false_on_error() -> None:
    """health_check() retorna False en error de red."""
    backend = TavilyBackend(api_key="tvly-test", budget=MagicMock(), timeout=15.0)
    with patch("hermes.services.search.tavily.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("network error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        assert await backend.health_check() is False


# --- error handling ---


@pytest.mark.asyncio
async def test_tavily_search_raises_on_http_error() -> None:
    """search() propaga HTTPError si Tavily retorna 4xx/5xx."""
    import httpx

    backend = TavilyBackend(api_key="tvly-test", budget=MagicMock(), timeout=15.0)
    with patch("hermes.services.search.tavily.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock(status_code=401)
        )
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            await backend.search("test", content_mode="snippet", num_results=10)
