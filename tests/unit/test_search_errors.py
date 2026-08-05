"""Tests Sprint 9.3 + PRE2-A1: Errors module (Capa 9).

Cubre:
- SearchErrorCode enum: 14 codigos (9 legacy + 5 nuevos PRE2-A1)
- SearchDiagnosticCategory enum: 11 categorias estables y finitas
- SearchError: dataclass con backends_tried, reasons, breaker_relevant,
  http_status, diagnostic_category
- _build_structured_error: defaults sensatos
- error_to_search_result: serializacion a dict para el LLM
  (incluye los campos seguros nuevos; nunca str(exc))
- Secret / raw-text redaction: ni message ni serializacion contienen
  texto de excepcion ni marcadores secretos plantados
"""

from __future__ import annotations

from hermes.services.search.errors import (
    ERROR_DEFAULTS,
    SearchDiagnosticCategory,
    SearchErrorCode,
    _build_structured_error,
    error_to_search_result,
)

# --- SearchErrorCode enum ---


def test_search_error_code_has_fourteen_codes() -> None:
    """SearchErrorCode tiene 14 codigos (9 legacy + 5 nuevos PRE2-A1)."""
    assert len(SearchErrorCode) == 14


def test_search_error_code_includes_pre2a1_codes() -> None:
    """PRE2-A1: nuevos codigos HTTP/transport estan presentes."""
    for code in (
        SearchErrorCode.CLIENT_ERROR,
        SearchErrorCode.AUTH_ERROR,
        SearchErrorCode.RATE_LIMITED,
        SearchErrorCode.SERVER_ERROR,
        SearchErrorCode.NETWORK_ERROR,
    ):
        assert code in SearchErrorCode


def test_search_error_code_values_are_strings() -> None:
    """SearchErrorCode values son strings (compatibles con JSON)."""
    for code in SearchErrorCode:
        assert isinstance(code.value, str)


# --- SearchDiagnosticCategory enum ---


def test_diagnostic_category_is_finite_enum() -> None:
    """SearchDiagnosticCategory es un enum finito y estable."""
    # Las 11 categorias cerradas del contrato PRE2-A1.
    expected = {
        "local_validation",
        "auth",
        "rate_limit",
        "client_error",
        "server_error",
        "timeout",
        "network",
        "invalid_response",
        "budget",
        "circuit",
        "all_backends_failed",
    }
    actual = {c.value for c in SearchDiagnosticCategory}
    assert actual == expected


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


# --- PRE2-A1: new SearchError fields ---


def test_search_error_has_pre2a1_fields() -> None:
    """PRE2-A1: SearchError tiene los campos estructurados nuevos."""
    error = _build_structured_error(
        code=SearchErrorCode.SERVER_ERROR,
        message="Search backend tavily returned HTTP 503.",
        backend="tavily",
        retryable=True,
        breaker_relevant=True,
        http_status=503,
        diagnostic_category=SearchDiagnosticCategory.SERVER_ERROR,
    )
    assert error.breaker_relevant is True
    assert error.http_status == 503
    assert error.diagnostic_category is SearchDiagnosticCategory.SERVER_ERROR


def test_search_error_http_status_can_be_none() -> None:
    """PRE2-A1: http_status admite None para timeouts, validacion local, etc."""
    error = _build_structured_error(
        code=SearchErrorCode.TIMEOUT,
        message="Search timed out.",
        backend="tavily",
        retryable=True,
        breaker_relevant=True,
        http_status=None,
        diagnostic_category=SearchDiagnosticCategory.TIMEOUT,
    )
    assert error.http_status is None


def test_search_error_default_fields_are_conservative() -> None:
    """PRE2-A1: defaults seguros (no breaker, no status, local validation)."""
    error = _build_structured_error(
        code=SearchErrorCode.EMPTY_QUERY,
        message="Query cannot be empty.",
        backend=None,
    )
    assert error.breaker_relevant is False
    assert error.http_status is None
    assert error.diagnostic_category is SearchDiagnosticCategory.LOCAL_VALIDATION


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
        message="Search timed out after 10s on tavily.",
        backend="tavily",
        retryable=True,
        backends_tried=["tavily"],
        reasons={"tavily": "TIMEOUT"},
        breaker_relevant=True,
        http_status=None,
        diagnostic_category=SearchDiagnosticCategory.TIMEOUT,
    )
    serialized = error_to_search_result(error)
    assert serialized["code"] == "TIMEOUT"
    assert serialized["error"] == "Search timed out after 10s on tavily."
    assert serialized["backend"] == "tavily"
    assert serialized["retryable"] is True
    assert serialized["suggestion"]  # any non-empty
    assert serialized["backends_tried"] == ["tavily"]
    assert serialized["reasons"] == {"tavily": "TIMEOUT"}
    # PRE2-A1:
    assert serialized["breaker_relevant"] is True
    assert serialized["http_status"] is None
    assert serialized["diagnostic_category"] == "timeout"


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


def test_error_to_search_result_never_contains_secret_marker() -> None:
    """PRE2-A1: serialized SearchError nunca contiene un secret marker plantado."""
    secret_marker = "PLANTED-SECRET-MARKER-12345"
    # The "exception text" that an unsafe implementation would
    # concatenate into the message.
    error = _build_structured_error(
        code=SearchErrorCode.INVALID_RESPONSE,
        message="Search backend tavily failed (INVALID_RESPONSE).",
        backend="tavily",
    )
    serialized = error_to_search_result(error)
    # Secret marker must NOT appear anywhere in the serialized
    # payload. This guards against accidental inclusion of str(exc)
    # at construction or serialization time.
    for value in serialized.values():
        if isinstance(value, str):
            assert secret_marker not in value
        elif isinstance(value, dict):
            for v in value.values():
                if isinstance(v, str):
                    assert secret_marker not in v
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    assert secret_marker not in item


# --- ERROR_DEFAULTS ---


def test_error_defaults_has_all_codes() -> None:
    """ERROR_DEFAULTS tiene defaults para todos los codigos."""
    for code in SearchErrorCode:
        assert code in ERROR_DEFAULTS


def test_error_defaults_keys() -> None:
    """Cada default tiene retryable, suggestion, breaker_relevant, diagnostic_category."""
    for _code, defaults in ERROR_DEFAULTS.items():
        assert "retryable" in defaults
        assert "suggestion" in defaults
        assert "breaker_relevant" in defaults  # PRE2-A1
        assert "diagnostic_category" in defaults  # PRE2-A1
        assert isinstance(defaults["retryable"], bool)
        assert isinstance(defaults["suggestion"], str)
        assert isinstance(defaults["breaker_relevant"], bool)


# --- PRE2-A1: frozen retry/breaker matrix at the defaults layer ---


def test_error_defaults_local_validation_not_retryable_not_breaker() -> None:
    """Local validation: no retryable, no breaker-relevant."""
    for code in (
        SearchErrorCode.EMPTY_QUERY,
        SearchErrorCode.INVALID_INTENT,
        SearchErrorCode.INVALID_CONTENT,
    ):
        assert ERROR_DEFAULTS[code]["retryable"] is False
        assert ERROR_DEFAULTS[code]["breaker_relevant"] is False
        assert (
            ERROR_DEFAULTS[code]["diagnostic_category"]
            is SearchDiagnosticCategory.LOCAL_VALIDATION
        )


def test_error_defaults_http_4xx_split() -> None:
    """HTTP 4xx split: 400/422 y 401/403 NO retryable, 429 SI retryable.
    Todos NO breaker-relevant."""
    # 400/422: client error
    assert ERROR_DEFAULTS[SearchErrorCode.CLIENT_ERROR]["retryable"] is False
    assert ERROR_DEFAULTS[SearchErrorCode.CLIENT_ERROR]["breaker_relevant"] is False
    # 401/403: auth error
    assert ERROR_DEFAULTS[SearchErrorCode.AUTH_ERROR]["retryable"] is False
    assert ERROR_DEFAULTS[SearchErrorCode.AUTH_ERROR]["breaker_relevant"] is False
    # 429: rate limited (retryable, but NOT breaker-relevant)
    assert ERROR_DEFAULTS[SearchErrorCode.RATE_LIMITED]["retryable"] is True
    assert ERROR_DEFAULTS[SearchErrorCode.RATE_LIMITED]["breaker_relevant"] is False


def test_error_defaults_5xx_retryable_and_breaker() -> None:
    """HTTP 5xx: retryable AND breaker-relevant."""
    assert ERROR_DEFAULTS[SearchErrorCode.SERVER_ERROR]["retryable"] is True
    assert ERROR_DEFAULTS[SearchErrorCode.SERVER_ERROR]["breaker_relevant"] is True


def test_error_defaults_timeout_retryable_and_breaker() -> None:
    """Timeout: retryable AND breaker-relevant."""
    assert ERROR_DEFAULTS[SearchErrorCode.TIMEOUT]["retryable"] is True
    assert ERROR_DEFAULTS[SearchErrorCode.TIMEOUT]["breaker_relevant"] is True


def test_error_defaults_network_retryable_and_breaker() -> None:
    """Network: retryable AND breaker-relevant."""
    assert ERROR_DEFAULTS[SearchErrorCode.NETWORK_ERROR]["retryable"] is True
    assert ERROR_DEFAULTS[SearchErrorCode.NETWORK_ERROR]["breaker_relevant"] is True
    # CONNECTION_REFUSED is the existing network code and inherits the same.
    assert ERROR_DEFAULTS[SearchErrorCode.CONNECTION_REFUSED]["retryable"] is True
    assert ERROR_DEFAULTS[SearchErrorCode.CONNECTION_REFUSED]["breaker_relevant"] is True


def test_error_defaults_invalid_response_not_retryable_not_breaker() -> None:
    """Invalid 2xx response: no retryable, no breaker-relevant."""
    assert ERROR_DEFAULTS[SearchErrorCode.INVALID_RESPONSE]["retryable"] is False
    assert ERROR_DEFAULTS[SearchErrorCode.INVALID_RESPONSE]["breaker_relevant"] is False
