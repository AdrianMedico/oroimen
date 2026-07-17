"""Tests Sprint 9.3: SearXNG Backend (Capa 2).

Cubre:
- URL encoding (queries con espacios, caracteres especiales)
- Parsing de JSON response de SearXNG
- Content mode snippet (unico soportado)
- Timeout via asyncio.wait_for
- Healthcheck via HTTP GET
- Error handling (connection refused, invalid JSON)
- has_budget delega a BudgetTracker
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.memory.db import Database
from hermes.services.search.budget import BudgetTracker
from hermes.services.search.searxng import SearXNGBackend


@pytest.fixture
async def db() -> Database:
    with tempfile.TemporaryDirectory() as td:
        d = Database(Path(td) / "test.db")
        await d.initialize()
        yield d
        await d.close()


def _searxng_response_json(results: list[dict] | None = None) -> dict:
    """Construye un response JSON valido de SearXNG."""
    if results is None:
        results = [
            {
                "title": "Result 1",
                "url": "https://example.com/1",
                "content": "Snippet 1 content",
                "engine": "duckduckgo",
            },
            {
                "title": "Result 2",
                "url": "https://example.com/2",
                "content": "Snippet 2 content",
                "engine": "mojeek",
            },
        ]
    return {"results": results}


# --- SUPPORTED_CONTENT_MODES ---


def test_searxng_supports_only_snippet() -> None:
    """SearXNG solo soporta snippet (no genera summaries nativamente)."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    assert frozenset({"snippet"}) == backend.SUPPORTED_CONTENT_MODES


# --- name ---


def test_searxng_name() -> None:
    """Backend name es 'searxng' (usado para circuit breaker, logs)."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    assert backend.name == "searxng"


# --- search: basic ---


@pytest.mark.asyncio
async def test_searxng_search_returns_results() -> None:
    """search() retorna SearchResult con los resultados parseados."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    response = _searxng_response_json()
    with patch("hermes.services.search.searxng.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = response
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        result = await backend.search("python pep", content_mode="snippet", num_results=10)

    assert len(result.results) == 2
    assert result.results[0]["title"] == "Result 1"
    assert result.results[0]["url"] == "https://example.com/1"
    assert result.results[0]["content"] == "Snippet 1 content"
    assert result.backend_used == "searxng"
    assert result.query == "python pep"
    assert result.content_mode == "snippet"
    assert result.original_content_mode == "snippet"
    assert result.format_fallback is False


# --- URL encoding ---


@pytest.mark.asyncio
async def test_searxng_search_url_encodes_query() -> None:
    """search() URL-encodes queries con espacios o caracteres especiales."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    captured_url: list[str] = []

    async def mock_get(url: str, *args: object, **kwargs: object) -> MagicMock:
        captured_url.append(url)
        mock_response = MagicMock()
        mock_response.json.return_value = _searxng_response_json([])
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch("hermes.services.search.searxng.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        await backend.search("python async await", content_mode="snippet", num_results=10)

    assert len(captured_url) == 1
    # "python async await" -> "python%20async%20await" o "python+async+await"
    assert "python" in captured_url[0]
    assert "async" in captured_url[0]
    # El espacio NO debe estar raw en la URL
    assert " " not in captured_url[0].split("?", 1)[1]


# --- num_results ---


@pytest.mark.asyncio
async def test_searxng_search_passes_num_results_to_query() -> None:
    """search() pasa num_results como param 'count' en la URL."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    captured_url: list[str] = []

    async def mock_get(url: str, *args: object, **kwargs: object) -> MagicMock:
        captured_url.append(url)
        mock_response = MagicMock()
        mock_response.json.return_value = _searxng_response_json([])
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch("hermes.services.search.searxng.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        await backend.search("test", content_mode="snippet", num_results=25)

    assert "count=25" in captured_url[0]


# --- has_budget delegation ---


@pytest.mark.asyncio
async def test_searxng_has_budget_delegates_to_tracker(db: Database) -> None:
    """has_budget() delega a BudgetTracker con self.name."""
    budget = BudgetTracker(db, limits={"searxng": -1})  # unlimited
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=budget,
        timeout=10.0,
    )
    assert await backend.has_budget() is True  # unlimited


@pytest.mark.asyncio
async def test_searxng_has_budget_false_when_exhausted(db: Database) -> None:
    """has_budget() retorna False cuando BudgetTracker dice que no hay budget."""
    budget = BudgetTracker(db, limits={"searxng": 0})  # 0 = no budget
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=budget,
        timeout=10.0,
    )
    assert await backend.has_budget() is False


# --- health_check ---


@pytest.mark.asyncio
async def test_searxng_health_check_returns_true_on_200() -> None:
    """health_check() retorna True si el endpoint root responde 200."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    with patch("hermes.services.search.searxng.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        result = await backend.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_searxng_health_check_returns_false_on_error() -> None:
    """health_check() retorna False si el endpoint no responde."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    with patch("hermes.services.search.searxng.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        result = await backend.health_check()

    assert result is False


@pytest.mark.asyncio
async def test_searxng_health_check_returns_false_on_non_200() -> None:
    """health_check() retorna False si el status code no es 200."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    with patch("hermes.services.search.searxng.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        result = await backend.health_check()

    assert result is False


# --- error handling ---


@pytest.mark.asyncio
async def test_searxng_search_raises_on_connection_error() -> None:
    """search() propaga exception si la conexion falla (router captura)."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    with patch("hermes.services.search.searxng.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        with pytest.raises(Exception, match="connection refused"):
            await backend.search("test", content_mode="snippet", num_results=10)


@pytest.mark.asyncio
async def test_searxng_search_raises_on_invalid_json() -> None:
    """search() propaga exception si el JSON response es invalido."""
    backend = SearXNGBackend(
        url="http://searxng:8888",
        budget=MagicMock(),
        timeout=10.0,
    )
    with patch("hermes.services.search.searxng.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("err", "doc", 0)
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        with pytest.raises(json.JSONDecodeError):
            await backend.search("test", content_mode="snippet", num_results=10)
