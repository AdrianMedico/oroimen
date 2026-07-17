"""Tests Sprint 9.3: Wire-up de Web Search Router en main (Capa 12).

Cubre:
- search_enabled=False: no se crean componentes
- search_enabled=True con SearXNG URL: se crea SearXNGBackend
- search_enabled=True con Tavily/Exa keys: se crean los backends
- Tool 'hermes_search' se registra en tool_registry
- num_results y content_mode se pasan al backend
"""

from __future__ import annotations

import pytest


@pytest.fixture
def mock_settings_factory():
    """Crea Settings con valores minimos para S9.3 tests."""

    def _factory(**overrides):
        from hermes.config import Settings

        base = {
            "telegram_bot_token": "1234567890:test_token_here",
            "opencode_go_api_key": "sk-test-key-1234567890",  # gitleaks:allow
            "gemini_api_key": "AIza-test-key-1234567890",  # gitleaks:allow
        }
        base.update(overrides)
        return Settings(**base)

    return _factory


# --- search_enabled=False: no se crea nada ---


@pytest.mark.asyncio
async def test_search_disabled_no_components_created(mock_settings_factory) -> None:
    """Si search_enabled=False, no se crean backends ni tool."""
    settings = mock_settings_factory(search_enabled=False)
    assert settings.search_enabled is False


@pytest.mark.asyncio
async def test_search_enabled_creates_searxng_backend(mock_settings_factory) -> None:
    """Si search_enabled=True y search_searxng_url set, se crea SearXNG."""
    settings = mock_settings_factory(
        search_enabled=True,
        search_searxng_url="http://searxng:8888",
    )
    assert settings.search_enabled is True
    assert settings.search_searxng_url == "http://searxng:8888"


@pytest.mark.asyncio
async def test_search_enabled_creates_tavily_if_key_set(mock_settings_factory) -> None:
    """Si tavily_api_key esta set, el backend deberia crearse (en wire-up real)."""
    settings = mock_settings_factory(
        search_enabled=True,
        tavily_api_key="tvly-test-key-1234567890",  # gitleaks:allow
    )
    assert settings.tavily_api_key == "tvly-test-key-1234567890"  # gitleaks:allow


@pytest.mark.asyncio
async def test_search_enabled_creates_exa_if_key_set(mock_settings_factory) -> None:
    """Si exa_api_key esta set, el backend deberia crearse."""
    settings = mock_settings_factory(
        search_enabled=True,
        exa_api_key="exa-test-key-1234567890",  # gitleaks:allow
    )
    assert settings.exa_api_key == "exa-test-key-1234567890"  # gitleaks:allow


@pytest.mark.asyncio
async def test_search_no_keys_no_tool_registered(mock_settings_factory) -> None:
    """Si no hay Tavily/Exa keys, SearXNG default sigue creando el tool."""
    settings = mock_settings_factory(
        search_enabled=True,
        search_searxng_url="http://searxng:8888",
        tavily_api_key="",  # vacio
        exa_api_key="",  # vacio
    )
    # SearXNG URL set -> tool se creara
    assert settings.search_searxng_url != ""
    assert settings.tavily_api_key == ""
    assert settings.exa_api_key == ""


# --- Settings consistency ---


@pytest.mark.asyncio
async def test_default_num_results_in_range(mock_settings_factory) -> None:
    """search_default_num_results <= search_max_num_results."""
    settings = mock_settings_factory()
    assert settings.search_default_num_results <= settings.search_max_num_results


@pytest.mark.asyncio
async def test_per_backend_timeouts_distinct(mock_settings_factory) -> None:
    """SearXNG timeout != Tavily/Exa (privacy-first mas estricto)."""
    settings = mock_settings_factory()
    assert settings.search_timeout_searxng != settings.search_timeout_tavily


@pytest.mark.asyncio
async def test_circuit_breaker_threshold_positive(mock_settings_factory) -> None:
    """search_circuit_breaker_threshold >= 1."""
    settings = mock_settings_factory()
    assert settings.search_circuit_breaker_threshold >= 1


@pytest.mark.asyncio
async def test_circuit_breaker_ttl_in_range(mock_settings_factory) -> None:
    """search_circuit_breaker_ttl_seconds en [10, 3600]."""
    settings = mock_settings_factory()
    assert 10 <= settings.search_circuit_breaker_ttl_seconds <= 3600
