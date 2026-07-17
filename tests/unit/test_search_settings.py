"""Tests Sprint 9.3: Settings (Capa 8).

Cubre:
- search_enabled default False
- search_searxng_url default
- search_*_timeout per-backend
- tavily/exa API keys y monthly limits
- validation_alias para env vars
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hermes.config import Settings


def _make_settings(**overrides: object) -> Settings:
    """Crea Settings con valores minimos validos.

    Importante: pasamos _env_file=None para que pydantic-settings NO lea el
    .env local. Los tests asumen los defaults del codigo (no los valores
    del .env del dev). Si queres que el test respete .env, pasale env_file
    explicitamente via overrides.
    """
    base = {
        "telegram_bot_token": "1234567890:test_token_here",
        "opencode_go_api_key": "sk-test-key-1234567890",  # gitleaks:allow
        "gemini_api_key": "AIza-test-key-1234567890",  # gitleaks:allow
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


# --- Defaults ---


def test_search_enabled_default_false() -> None:
    """Network search is explicit opt-in."""
    s = _make_settings()
    assert s.search_enabled is False


def test_search_searxng_url_default() -> None:
    """search_searxng_url default = http://searxng:8888 (docker internal)."""
    s = _make_settings()
    assert s.search_searxng_url == "http://searxng:8888"


def test_search_size_guard_chars_default() -> None:
    """S9.3.1: search_size_guard_chars default = 200000 (50K tokens aprox).

    S9.3.0 era 50000. Subido a 200K para dar más contexto al LLM.
    A $0.21/M tokens (Go tier minimax-m3), 200K chars = ~$0.01/search.
    """
    s = _make_settings()
    assert s.search_size_guard_chars == 200000


def test_search_max_concurrent_default() -> None:
    """search_max_concurrent default = 3 (legacy, S9.3.0)."""
    s = _make_settings()
    assert s.search_max_concurrent == 3


def test_search_max_concurrent_searxng_default() -> None:
    """search_max_concurrent_searxng default = 6 (S9.3.1 punto 2)."""
    s = _make_settings()
    assert s.search_max_concurrent_searxng == 6


def test_search_max_concurrent_tavily_default() -> None:
    """search_max_concurrent_tavily default = 10 (S9.3.1 punto 2)."""
    s = _make_settings()
    assert s.search_max_concurrent_tavily == 10


def test_search_max_concurrent_exa_default() -> None:
    """search_max_concurrent_exa default = 5 (S9.3.1 punto 2)."""
    s = _make_settings()
    assert s.search_max_concurrent_exa == 5


def test_per_backend_concurrency_all_positive() -> None:
    """Todos los per-backend concurrency son > 0 y razonables."""
    s = _make_settings()
    assert s.search_max_concurrent_searxng > 0
    assert s.search_max_concurrent_tavily > 0
    assert s.search_max_concurrent_exa > 0
    # Tavily deberia ser >= Exa (Tavily permite 100 req/min vs Exa 50 req/min)
    assert s.search_max_concurrent_tavily >= s.search_max_concurrent_exa


def test_search_circuit_breaker_threshold_default() -> None:
    """search_circuit_breaker_threshold default = 3."""
    s = _make_settings()
    assert s.search_circuit_breaker_threshold == 3


def test_search_circuit_breaker_ttl_default() -> None:
    """search_circuit_breaker_ttl_seconds default = 300."""
    s = _make_settings()
    assert s.search_circuit_breaker_ttl_seconds == 300


# --- Per-backend timeouts (P1-4 v1.2) ---


def test_timeouts_per_backend_searxng() -> None:
    """search_timeout_searxng default = 10s."""
    s = _make_settings()
    assert s.search_timeout_searxng == 10.0


def test_timeouts_per_backend_tavily() -> None:
    """search_timeout_tavily default = 15s."""
    s = _make_settings()
    assert s.search_timeout_tavily == 15.0


def test_timeouts_per_backend_exa() -> None:
    """search_timeout_exa default = 15s."""
    s = _make_settings()
    assert s.search_timeout_exa == 15.0


def test_timeouts_per_backend_distinct() -> None:
    """SearXNG timeout != Tavily/Exa (privacy-first mas estricto)."""
    s = _make_settings()
    assert s.search_timeout_searxng != s.search_timeout_tavily


# --- Tavily config ---


def test_tavily_api_key_default_empty() -> None:
    """tavily_api_key default = '' (opt-in, no requerido)."""
    s = _make_settings()
    assert s.tavily_api_key == ""


def test_tavily_monthly_limit_default() -> None:
    """tavily_monthly_limit default = 1000."""
    s = _make_settings()
    assert s.tavily_monthly_limit == 1000


# --- Exa config ---


def test_exa_api_key_default_empty() -> None:
    """exa_api_key default = '' (opt-in, no requerido)."""
    s = _make_settings()
    assert s.exa_api_key == ""


def test_exa_monthly_limit_default() -> None:
    """exa_monthly_limit default = 1000."""
    s = _make_settings()
    assert s.exa_monthly_limit == 1000


# --- num_results ---


def test_search_default_num_results() -> None:
    """search_default_num_results = 10."""
    s = _make_settings()
    assert s.search_default_num_results == 10


def test_search_max_num_results() -> None:
    """search_max_num_results = 50 (hard cap)."""
    s = _make_settings()
    assert s.search_max_num_results == 50


# --- Validation ---


def test_tavily_monthly_limit_rejects_negative() -> None:
    """tavily_monthly_limit >= 0 (validation)."""
    with pytest.raises(ValidationError):
        _make_settings(tavily_monthly_limit=-1)


def test_search_size_guard_chars_rejects_too_small() -> None:
    """search_size_guard_chars >= 1000."""
    with pytest.raises(ValidationError):
        _make_settings(search_size_guard_chars=100)


def test_search_max_concurrent_rejects_too_large() -> None:
    """search_max_concurrent <= 10."""
    with pytest.raises(ValidationError):
        _make_settings(search_max_concurrent=100)


# --- Override via env vars (validation_alias) ---


def test_search_enabled_can_be_disabled() -> None:
    """search_enabled=False opt-out para tests/dev."""
    s = _make_settings(search_enabled=False)
    assert s.search_enabled is False


def test_tavily_api_key_can_be_set() -> None:
    """tavily_api_key se puede setear (opt-in)."""
    s = _make_settings(tavily_api_key="tvly-test-key-1234567890")  # gitleaks:allow
    assert s.tavily_api_key == "tvly-test-key-1234567890"  # gitleaks:allow


def test_searxng_url_can_be_overridden() -> None:
    """search_searxng_url se puede override (e.g., localhost:8888 en dev)."""
    s = _make_settings(search_searxng_url="http://localhost:8888")
    assert s.search_searxng_url == "http://localhost:8888"
