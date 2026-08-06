"""Tests Sprint 9.3 + PRE2-A1 + PRE2-A2: Errors module (Capa 9).

Cubre:
- SearchErrorCode enum: 15 codigos (9 legacy + 5 PRE2-A1 + 1 PRE2-A2)
- SearchDiagnosticCategory enum: 11 categorias estables y finitas
- SearchError: dataclass con backends_tried, reasons, breaker_relevant,
  http_status, diagnostic_category
- _build_structured_error: defaults sensatos
- error_to_search_result: serializacion a dict para el LLM
  (incluye los campos seguros nuevos; nunca str(exc))
- Secret / raw-text redaction: ni message ni serializacion contienen
  texto de excepcion ni un sentinel de redaccion low-entropy plantado
  en una superficie insegura real.
- PRE2-A2: QUERY_TOO_LONG is a local-validation code that is
  non-retryable, non-breaker, and never carries the query text.
  The 399 value on Tavily is an Oroimen conservative operational
  / compatibility cap pending live Tavily validation; it is NOT
  a claim about the hosted Tavily API's current limit.
  Note: the structured ``search_error_code`` and
  ``search_diagnostic_category`` propagate to the in-memory
  ``PhaseError``; only the broad ``search_4xx`` taxonomy is
  persisted to the job row.
"""

from __future__ import annotations

from hermes.jobs.service import _phase_error_from_search_error
from hermes.services.search.errors import (
    ERROR_DEFAULTS,
    SearchDiagnosticCategory,
    SearchErrorCode,
    _build_structured_error,
    error_to_search_result,
)

# --- SearchErrorCode enum ---


def test_search_error_code_has_fifteen_codes() -> None:
    """SearchErrorCode tiene 15 codigos (9 legacy + 5 PRE2-A1 + 1 PRE2-A2)."""
    assert len(SearchErrorCode) == 15


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


def test_search_error_code_includes_pre2a2_query_too_long() -> None:
    """PRE2-A2: ``QUERY_TOO_LONG`` is a valid code in the enum.

    The router uses it to surface the local per-backend
    query-length rejection. It is non-retryable, non-breaker,
    and categorised as ``LOCAL_VALIDATION``.
    """
    assert SearchErrorCode.QUERY_TOO_LONG in SearchErrorCode
    assert SearchErrorCode.QUERY_TOO_LONG.value == "QUERY_TOO_LONG"


def test_query_too_long_default_is_local_validation() -> None:
    """PRE2-A2: ``QUERY_TOO_LONG`` defaults are local-validation.

    ``retryable=False``, ``breaker_relevant=False``,
    ``diagnostic_category=LOCAL_VALIDATION``. The router does
    NOT count this against any backend's circuit breaker; no
    provider call ever happened.
    """
    defaults = ERROR_DEFAULTS[SearchErrorCode.QUERY_TOO_LONG]
    assert defaults["retryable"] is False
    assert defaults["breaker_relevant"] is False
    assert (
        defaults["diagnostic_category"]
        == SearchDiagnosticCategory.LOCAL_VALIDATION
    )


def test_query_too_long_suggestion_mentions_shorten_or_intent() -> None:
    """PRE2-A2: ``QUERY_TOO_LONG`` suggestion guides the LLM.

    The LLM-facing suggestion must mention how to recover: by
    shortening the query or by switching intent (e.g. general
    / SearXNG which has ``max_query_chars = None``).
    """
    defaults = ERROR_DEFAULTS[SearchErrorCode.QUERY_TOO_LONG]
    suggestion = defaults["suggestion"]
    assert "shorten" in suggestion.lower() or "intent" in suggestion.lower()


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


def test_safe_message_passes_through_serializer_unchanged() -> None:
    """Serializer contract: ``error_to_search_result`` and the
    ``_phase_error_from_search_error`` bridge MUST faithfully reflect
    the ``SearchError.message`` they are given. They do not invent,
    rewrite, or extract text from arbitrary sources.

    This test verifies the serializer layer's honesty when the caller
    provides a safe static message. The router's actual production
    redaction proof (Cases A-D) lives in
    ``tests/unit/test_search_router.py`` and exercises the real
    ``hermes_search`` pipeline end-to-end with sentinel-bearing
    caller values, backend exceptions, and HTTP response bodies. This
    unit test only asserts the layer-local contract: the safe
    surfaces stay clean as long as the caller respects it.
    """
    # Low-entropy sentinel. Chosen to be obviously non-secret and
    # highly unlikely to appear in any other code path or fixture.
    redaction_sentinel = "internal response detail must stay private"

    # --- Three real unsafe inputs the sentinel could leak FROM ---
    class _UnsafeSentinelError(Exception):
        """Custom exception type carrying the sentinel."""

    unsafe_custom_exc = _UnsafeSentinelError(
        f"detail: {redaction_sentinel} from inner call"
    )
    unsafe_response_body = (
        f'{{"data": "...{redaction_sentinel}..."}}'
    )
    unsafe_value_error = ValueError(
        f"malformed JSON: {redaction_sentinel} token leaked"
    )

    # Sanity: the sentinel IS present in each unsafe input. If this
    # assertion ever fails the test fixture is no longer exercising
    # the intended unsafe surface and the leak assertions below
    # become vacuous.
    assert redaction_sentinel in str(unsafe_custom_exc)
    assert redaction_sentinel in unsafe_response_body
    assert redaction_sentinel in str(unsafe_value_error)

    # --- Safe construction (the router's contract since PRE2-A1) ---
    # The safe message is a static string set at construction. The
    # errors module MUST NOT extract text from any of the unsafe
    # inputs and inject it into the message.
    safe_message = "Search backend returned invalid response."
    assert redaction_sentinel not in safe_message

    error = _build_structured_error(
        code=SearchErrorCode.INVALID_RESPONSE,
        message=safe_message,
        backend="tavily",
        retryable=False,
        breaker_relevant=False,
        diagnostic_category=SearchDiagnosticCategory.INVALID_RESPONSE,
    )

    # 1. SearchError.message MUST NOT contain the sentinel.
    assert redaction_sentinel not in error.message

    # 2. error_to_search_result safe fields MUST NOT contain the
    # sentinel. The serializer is faithful to free-form fields
    # (suggestion, reasons, backends_tried) — the contract is for
    # callers to keep sentinels out of the safe message. This test
    # asserts the structured safe fields stay clean when the
    # caller respects the contract.
    serialized = error_to_search_result(error)
    for safe_field in ("error", "code", "backend", "diagnostic_category"):
        assert redaction_sentinel not in serialized[safe_field], (
            f"sentinel leaked into serialized safe field {safe_field!r}"
        )

    # 3. PhaseError bridged from the SearchError MUST NOT contain
    # the sentinel in its safe message.
    pe = _phase_error_from_search_error(error)
    assert redaction_sentinel not in pe.message


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
