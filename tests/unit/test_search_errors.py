"""Tests Sprint 9.3: Errors module (Capa 9).

Cubre:
- SearchErrorCode enum: 9 codigos
- SearchError: dataclass con backends_tried, reasons
- _build_structured_error: defaults sensatos
- error_to_search_result: serializacion a dict para el LLM
"""

from __future__ import annotations

from hermes.services.search.errors import (
    ERROR_DEFAULTS,
    SearchErrorCode,
    _build_structured_error,
    error_to_search_result,
)

# --- SearchErrorCode enum ---


def test_search_error_code_has_nine_codes() -> None:
    """SearchErrorCode tiene exactamente 9 codigos (incluyendo EMPTY_QUERY)."""
    assert len(SearchErrorCode) == 9


def test_search_error_code_values_are_strings() -> None:
    """SearchErrorCode values son strings (compatibles con JSON)."""
    for code in SearchErrorCode:
        assert isinstance(code.value, str)


# --- SearchError dataclass ---


def test_search_error_includes_backends_tried() -> None:
    """P1-1 v1.2: SearchError tiene backends_tried (list)."""
    error = _build_structured_error(
        code=SearchErrorCode.ALL_BACKENDS_FAILED,
        message="All failed",
        backend="searxng",
        backends_tried=["tavily", "exa", "searxng"],
    )
    assert "tavily" in error.backends_tried
    assert "exa" in error.backends_tried
    assert "searxng" in error.backends_tried


def test_search_error_includes_reasons() -> None:
    """P1-1 v1.2: SearchError tiene reasons (dict backend -> code)."""
    error = _build_structured_error(
        code=SearchErrorCode.ALL_BACKENDS_FAILED,
        message="All failed",
        backend=None,
        backends_tried=["tavily", "exa", "searxng"],
        reasons={
            "tavily": "BUDGET_EXHAUSTED",
            "exa": "TIMEOUT",
            "searxng": "CIRCUIT_OPEN",
        },
    )
    assert error.reasons["tavily"] == "BUDGET_EXHAUSTED"
    assert error.reasons["exa"] == "TIMEOUT"
    assert error.reasons["searxng"] == "CIRCUIT_OPEN"


def test_search_error_is_frozen() -> None:
    """SearchError es inmutable (frozen dataclass)."""
    error = _build_structured_error(
        code=SearchErrorCode.EMPTY_QUERY,
        message="test",
        backend=None,
    )
    import dataclasses

    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        error.message = "modified"  # type: ignore[misc]


# --- _build_structured_error defaults ---


def test_build_structured_error_default_suggestion() -> None:
    """Default suggestion es la fallback universal."""
    error = _build_structured_error(
        code=SearchErrorCode.EMPTY_QUERY,
        message="test",
        backend=None,
    )
    assert "internal knowledge" in error.suggestion


def test_build_structured_error_default_not_retryable() -> None:
    """Default retryable=False (solo casos especificos son retryable)."""
    error = _build_structured_error(
        code=SearchErrorCode.EMPTY_QUERY,
        message="test",
        backend=None,
    )
    assert error.retryable is False


def test_build_structured_error_retryable_true() -> None:
    """retryable=True se puede setear explicitamente."""
    error = _build_structured_error(
        code=SearchErrorCode.TIMEOUT,
        message="test",
        backend="tavily",
        retryable=True,
    )
    assert error.retryable is True


def test_build_structured_error_default_empty_backends_tried() -> None:
    """Sin backends_tried, el campo es lista vacia (no None)."""
    error = _build_structured_error(
        code=SearchErrorCode.EMPTY_QUERY,
        message="test",
        backend=None,
    )
    assert error.backends_tried == []


def test_build_structured_error_default_empty_reasons() -> None:
    """Sin reasons, el campo es dict vacio (no None)."""
    error = _build_structured_error(
        code=SearchErrorCode.EMPTY_QUERY,
        message="test",
        backend=None,
    )
    assert error.reasons == {}


# --- error_to_search_result serialization ---


def test_error_to_search_result_serializes_all_fields() -> None:
    """error_to_search_result retorna dict con todos los campos del error."""
    error = _build_structured_error(
        code=SearchErrorCode.TIMEOUT,
        message="Timeout after 10s",
        backend="tavily",
        retryable=True,
        backends_tried=["tavily"],
        reasons={"tavily": "TIMEOUT"},
    )
    serialized = error_to_search_result(error)
    assert serialized["code"] == "TIMEOUT"
    assert serialized["error"] == "Timeout after 10s"
    assert serialized["backend"] == "tavily"
    assert serialized["retryable"] is True
    assert serialized["suggestion"]  # any non-empty
    assert serialized["backends_tried"] == ["tavily"]
    assert serialized["reasons"] == {"tavily": "TIMEOUT"}


def test_error_to_search_result_is_json_serializable() -> None:
    """error_to_search_result produce un dict JSON-serializable."""
    import json

    error = _build_structured_error(
        code=SearchErrorCode.ALL_BACKENDS_FAILED,
        message="All down",
        backend="searxng",
        backends_tried=["tavily", "searxng"],
        reasons={"tavily": "BUDGET_EXHAUSTED", "searxng": "CIRCUIT_OPEN"},
    )
    serialized = error_to_search_result(error)
    # json.dumps debe funcionar sin errores
    json_str = json.dumps(serialized)
    assert "ALL_BACKENDS_FAILED" in json_str


# --- ERROR_DEFAULTS ---


def test_error_defaults_has_all_codes() -> None:
    """ERROR_DEFAULTS tiene defaults para todos los codigos."""
    for code in SearchErrorCode:
        assert code in ERROR_DEFAULTS


def test_error_defaults_keys() -> None:
    """Cada default tiene retryable y suggestion."""
    for _code, defaults in ERROR_DEFAULTS.items():
        assert "retryable" in defaults
        assert "suggestion" in defaults
        assert isinstance(defaults["retryable"], bool)
        assert isinstance(defaults["suggestion"], str)
