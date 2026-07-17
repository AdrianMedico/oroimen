"""Tests Sprint 9.3: Exa Backend (Capa 5).

Cubre:
- type: 'neural' cuando intent='semantic', 'keyword' en otros
- Content modes: solo snippet (campo 'highlights' de Exa)
- numResults en el request
- has_budget delega a BudgetTracker
- Error handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.memory.db import Database
from hermes.services.search.budget import BudgetTracker
from hermes.services.search.exa import ExaBackend


@pytest.fixture
async def db() -> Database:
    with tempfile.TemporaryDirectory() as td:
        d = Database(Path(td) / "test.db")
        await d.initialize()
        yield d
        await d.close()


def _exa_response(results: list[dict] | None = None) -> dict:
    """Construye un response JSON valido de Exa."""
    if results is None:
        results = [
            {
                "title": "Result 1",
                "url": "https://example.com/1",
                "highlights": ["highlight 1", "highlight 2"],
                "score": 0.92,
            }
        ]
    return {"requestId": "test", "results": results}


# --- SUPPORTED_CONTENT_MODES ---


def test_exa_supports_only_snippet() -> None:
    """Exa solo soporta snippet (no genera summaries nativamente)."""
    backend = ExaBackend(
        api_key="exa-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    assert frozenset({"snippet"}) == backend.SUPPORTED_CONTENT_MODES


# --- name ---


def test_exa_name() -> None:
    """Backend name es 'exa'."""
    backend = ExaBackend(
        api_key="exa-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    assert backend.name == "exa"


# --- search: intent → type ---


@pytest.mark.asyncio
async def test_exa_search_uses_neural_for_semantic_intent() -> None:
    """type='neural' cuando intent='semantic' (busqueda vector semantica)."""
    backend = ExaBackend(
        api_key="exa-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    captured: list[dict] = []

    async def mock_post(url: str, json: dict, headers: dict) -> MagicMock:
        captured.append(json)
        mock_response = MagicMock()
        mock_response.json.return_value = _exa_response()
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch("hermes.services.search.exa.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        await backend.search("test", content_mode="snippet", num_results=10, intent="semantic")

    assert captured[0]["type"] == "neural"


@pytest.mark.asyncio
async def test_exa_search_uses_keyword_for_non_semantic_intent() -> None:
    """type='keyword' cuando intent != 'semantic'."""
    backend = ExaBackend(
        api_key="exa-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    captured: list[dict] = []

    async def mock_post(url: str, json: dict, headers: dict) -> MagicMock:
        captured.append(json)
        mock_response = MagicMock()
        mock_response.json.return_value = _exa_response()
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch("hermes.services.search.exa.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        # intent='general' (default)
        await backend.search("test", content_mode="snippet", num_results=10)

    assert captured[0]["type"] == "keyword"


# --- search: numResults ---


@pytest.mark.asyncio
async def test_exa_search_passes_num_results_as_numResults() -> None:
    """numResults en el request de Exa."""
    backend = ExaBackend(
        api_key="exa-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    captured: list[dict] = []

    async def mock_post(url: str, json: dict, headers: dict) -> MagicMock:
        captured.append(json)
        mock_response = MagicMock()
        mock_response.json.return_value = _exa_response()
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch("hermes.services.search.exa.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        await backend.search("test", content_mode="snippet", num_results=15)

    assert captured[0]["numResults"] == 15


# --- search: results format ---


@pytest.mark.asyncio
async def test_exa_search_extracts_highlights_to_content() -> None:
    """Exa 'highlights' array se convierte a 'content' (joined)."""
    backend = ExaBackend(
        api_key="exa-test",
        budget=MagicMock(),
        timeout=15.0,
    )
    with patch("hermes.services.search.exa.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _exa_response(
            results=[
                {
                    "title": "R1",
                    "url": "https://e.com/1",
                    "highlights": ["h1", "h2", "h3"],
                    "score": 0.9,
                }
            ]
        )
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        result = await backend.search("test", content_mode="snippet", num_results=10)

    assert "h1" in result.results[0]["content"]
    assert "h2" in result.results[0]["content"]
    assert "h3" in result.results[0]["content"]


# --- has_budget ---


@pytest.mark.asyncio
async def test_exa_has_budget_delegates_to_tracker(db: Database) -> None:
    """has_budget() delega a BudgetTracker con self.name."""
    budget = BudgetTracker(db, limits={"exa": 1000})
    backend = ExaBackend(api_key="exa-test", budget=budget, timeout=15.0)
    assert await backend.has_budget() is True
    for _ in range(1000):
        await budget.record_usage("exa", count=1)
    assert await backend.has_budget() is False


# --- health_check ---


@pytest.mark.asyncio
async def test_exa_health_check_returns_true_on_200() -> None:
    """health_check() retorna True si Exa responde 200."""
    backend = ExaBackend(api_key="exa-test", budget=MagicMock(), timeout=15.0)
    with patch("hermes.services.search.exa.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        assert await backend.health_check() is True


@pytest.mark.asyncio
async def test_exa_health_check_returns_false_on_error() -> None:
    """health_check() retorna False en error de red."""
    backend = ExaBackend(api_key="exa-test", budget=MagicMock(), timeout=15.0)
    with patch("hermes.services.search.exa.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("network error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client

        assert await backend.health_check() is False
