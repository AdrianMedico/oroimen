"""Sprint 9.3 + PRE2-A1: Errores estructurados del Web Search Router.

P0 Gemini 3.1 Pro: el LLM debe recibir errores estructurados
(no strings vagos) para poder razonar sobre el fallback.

P1 Gemini 3.5 Thinking v1.1: el codigo del router hace referencia
a `_build_structured_error` que no existia en v1.0. Este modulo
lo implementa.

PRE2-A1 (truthful search-error bridge): extended contract so every
failure carries ``breaker_relevant``, ``http_status`` (optional), and
a stable ``SearchDiagnosticCategory``. The diagnostic category is a
finite enum, never arbitrary exception text. Deep Research and any
downstream consumer uses these structured fields to decide retry and
breaker behavior without parsing ``str(exc)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SearchErrorCode(StrEnum):
    """Codigos de error normalizados para el LLM.

    El LLM puede razonar sobre `code` y `suggestion` para decidir
    como responder al user (reformular, intentar con otro intent,
    usar knowledge interno, etc).

    PRE2-A1: added CLIENT_ERROR, AUTH_ERROR, RATE_LIMITED,
    SERVER_ERROR, NETWORK_ERROR so the router can classify typed
    HTTP/transport exceptions without parsing ``str(exc)``.
    """

    EMPTY_QUERY = "EMPTY_QUERY"
    INVALID_INTENT = "INVALID_INTENT"
    INVALID_CONTENT = "INVALID_CONTENT"
    TIMEOUT = "TIMEOUT"
    CONNECTION_REFUSED = "CONNECTION_REFUSED"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    ALL_BACKENDS_FAILED = "ALL_BACKENDS_FAILED"
    # PRE2-A1: HTTP/transport classification
    CLIENT_ERROR = "CLIENT_ERROR"  # HTTP 400, 422
    AUTH_ERROR = "AUTH_ERROR"  # HTTP 401, 403
    RATE_LIMITED = "RATE_LIMITED"  # HTTP 429
    SERVER_ERROR = "SERVER_ERROR"  # HTTP 5xx
    NETWORK_ERROR = "NETWORK_ERROR"  # transport-level failure
    # PRE2-A2: local validation against the SELECTED backend's
    # declared ``BackendQueryCapabilities.max_query_chars``. The
    # router raises this AFTER backend selection (including
    # fallback) and BEFORE acquiring the semaphore, debiting
    # budget, dispatching ``backend.search``, or mutating the
    # circuit breaker. The router validates the ORIGINAL query
    # (before the generic 2000-char truncation) so the surfaced
    # length is the user-visible length, not a post-truncation
    # artifact. The 399 value carried by Tavily is an Oroimen
    # conservative operational / compatibility cap pending live
    # Tavily validation; ``QUERY_TOO_LONG`` is the surface for
    # that cut, not a claim about the hosted API's current limit.
    # Length and backend identity are safe to surface; the query
    # text itself is NEVER included in the message, the
    # serialized dict, or the logs.
    QUERY_TOO_LONG = "QUERY_TOO_LONG"


class SearchDiagnosticCategory(StrEnum):
    """Categorias de diagnostico estables y finitas (PRE2-A1).

    Finite enum used to drive retry/breaker decisions and the
    mapping from ``SearchError`` to broad job taxonomy. The router
    assigns one of these values to every failure based on the
    exception type or HTTP status, never on exception text.

    Adding a new category requires a new ``SearchErrorCode``,
    explicit mapping in ``_classify_*`` helpers, and a focused
    regression test. Categories are not derived from arbitrary
    strings.
    """

    LOCAL_VALIDATION = "local_validation"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    NETWORK = "network"
    INVALID_RESPONSE = "invalid_response"
    BUDGET = "budget"
    CIRCUIT = "circuit"
    ALL_BACKENDS_FAILED = "all_backends_failed"


@dataclass(frozen=True)
class SearchError:
    """Error estructurado del Web Search Router.

    Attributes:
        code: codigo normalizado (SearchErrorCode).
        message: mensaje human-readable para el LLM. PRE2-A1: must
            be a safe static string set at construction time. Raw
            exception text, response bodies, headers, and credentials
            must NEVER enter this field.
        backend: nombre del backend que fallo (None si el error
            es anterior a la seleccion de backend, e.g.
            INVALID_INTENT).
        retryable: True si el LLM puede reintentar con el mismo
            intent (e.g., TIMEOUT es transitorio).
        suggestion: texto sugerido para el LLM sobre como
            responder al user. Ej: "Rely on internal knowledge
            or prompt the user for manual input".
        backends_tried: lista de backends intentados (P1-1 v1.2).
            Util para ALL_BACKENDS_FAILED.
        reasons: dict backend -> razon de fallo (P1-1 v1.2).
            Util para que el LLM sepa que fallo (CIRCUIT_OPEN,
            BUDGET_EXHAUSTED, TIMEOUT, etc).
        breaker_relevant: PRE2-A1. True si esta falla debe
            contar contra el circuit breaker del backend. False
            para errores deterministas (auth, rate limit, client
            error, invalid 2xx) o estados no-backend (budget,
            circuit ya abierto).
        http_status: PRE2-A1. HTTP status code cuando la falla
            viene de una respuesta HTTP. None para timeouts,
            errores de transporte, validacion local o estados
            no-HTTP.
        diagnostic_category: PRE2-A1. Categoria estable y finita
            del fallo. Usada por ``PhaseError`` y
            ``_run_phase_with_retry`` para mapear a taxonomia
            amplia sin parsear texto.
    """

    code: SearchErrorCode
    message: str
    backend: str | None
    retryable: bool
    suggestion: str
    backends_tried: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    # PRE2-A1: new structured fields
    breaker_relevant: bool = False
    http_status: int | None = None
    diagnostic_category: SearchDiagnosticCategory = (
        SearchDiagnosticCategory.LOCAL_VALIDATION
    )


def _build_structured_error(
    code: SearchErrorCode,
    message: str,
    backend: str | None,
    retryable: bool = False,
    suggestion: str = "Rely on internal knowledge or prompt the user for manual input.",
    backends_tried: list[str] | None = None,
    reasons: dict[str, str] | None = None,
    *,
    breaker_relevant: bool = False,
    http_status: int | None = None,
    diagnostic_category: SearchDiagnosticCategory = (
        SearchDiagnosticCategory.LOCAL_VALIDATION
    ),
) -> SearchError:
    """Construye un SearchError con defaults sensatos.

    PRE2-A1: ``breaker_relevant``, ``http_status`` and
    ``diagnostic_category`` are keyword-only. Existing callers that
    do not pass them get the conservative defaults (non-breaker,
    no status, LOCAL_VALIDATION category) so they remain safe.

    Args:
        code: codigo normalizado.
        message: mensaje tecnico. PRE2-A1: must be a safe static
            string. Do not concatenate ``str(exc)`` here.
        backend: nombre del backend o None.
        retryable: True si reintentar puede funcionar.
        suggestion: texto para el LLM.
        backends_tried: lista de backends intentados (P1-1 v1.2).
        reasons: dict backend -> razon de fallo (P1-1 v1.2).
        breaker_relevant: PRE2-A1. True if this failure should
            count against the circuit breaker. Defaults to False
            (conservative — only typed classifier opts in).
        http_status: PRE2-A1. Optional HTTP status code.
        diagnostic_category: PRE2-A1. Stable, finite category.

    Returns:
        SearchError inmutable.
    """
    return SearchError(
        code=code,
        message=message,
        backend=backend,
        retryable=retryable,
        suggestion=suggestion,
        backends_tried=backends_tried if backends_tried is not None else [],
        reasons=reasons if reasons is not None else {},
        breaker_relevant=breaker_relevant,
        http_status=http_status,
        diagnostic_category=diagnostic_category,
    )


# Mapeo de codigos a defaults utiles para tests/serializacion.
# PRE2-A1: each entry now also carries breaker_relevant and
# diagnostic_category so the defaults table is the single source of
# truth for the frozen retry/breaker matrix.
ERROR_DEFAULTS: dict[SearchErrorCode, dict[str, Any]] = {
    # --- Local validation (non-retryable, non-breaker) ---
    SearchErrorCode.EMPTY_QUERY: {
        "retryable": False,
        "suggestion": "Ask the user to clarify their query.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.LOCAL_VALIDATION,
    },
    SearchErrorCode.INVALID_INTENT: {
        "retryable": False,
        "suggestion": "Use one of: general, semantic, deep_research.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.LOCAL_VALIDATION,
    },
    SearchErrorCode.INVALID_CONTENT: {
        "retryable": False,
        "suggestion": "Use one of: snippet, summary, full.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.LOCAL_VALIDATION,
    },
    # PRE2-A2: local validation against the selected backend's
    # declared ``max_query_chars``. Non-retryable (the LLM
    # crafted a too-long query for this backend), non-breaker
    # (no provider call ever happened), and categorised as
    # ``local_validation`` so the job taxonomy treats it like
    # the other pre-dispatch rejections.
    SearchErrorCode.QUERY_TOO_LONG: {
        "retryable": False,
        "suggestion": (
            "Shorten the query to fit the selected backend's limit, "
            "or pick a different intent."
        ),
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.LOCAL_VALIDATION,
    },
    # --- Transport: timeout (retryable, breaker-relevant) ---
    SearchErrorCode.TIMEOUT: {
        "retryable": True,
        "suggestion": "Try again with a simpler query or different intent.",
        "breaker_relevant": True,
        "diagnostic_category": SearchDiagnosticCategory.TIMEOUT,
    },
    # --- Transport: network (retryable, breaker-relevant) ---
    SearchErrorCode.CONNECTION_REFUSED: {
        "retryable": True,
        "suggestion": "Check network connectivity. If persistent, use internal knowledge.",
        "breaker_relevant": True,
        "diagnostic_category": SearchDiagnosticCategory.NETWORK,
    },
    SearchErrorCode.NETWORK_ERROR: {
        "retryable": True,
        "suggestion": "Check network connectivity; this is a transient transport error.",
        "breaker_relevant": True,
        "diagnostic_category": SearchDiagnosticCategory.NETWORK,
    },
    # --- HTTP 4xx: client / auth / rate limit ---
    # PRE2-A1: all non-retryable EXCEPT 429, all non-breaker.
    SearchErrorCode.CLIENT_ERROR: {
        "retryable": False,
        "suggestion": "Verify request parameters; this is a deterministic request error.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.CLIENT_ERROR,
    },
    SearchErrorCode.AUTH_ERROR: {
        "retryable": False,
        "suggestion": "Verify the API key for the search backend.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.AUTH,
    },
    SearchErrorCode.RATE_LIMITED: {
        "retryable": True,
        "suggestion": "Reduce request rate or wait before retrying.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.RATE_LIMIT,
    },
    # --- HTTP 5xx (retryable, breaker-relevant) ---
    SearchErrorCode.SERVER_ERROR: {
        "retryable": True,
        "suggestion": "Try again later; this is a transient backend error.",
        "breaker_relevant": True,
        "diagnostic_category": SearchDiagnosticCategory.SERVER_ERROR,
    },
    # --- Invalid successful response (non-retryable, non-breaker) ---
    SearchErrorCode.INVALID_RESPONSE: {
        "retryable": False,
        "suggestion": "Report this as a bug. Use internal knowledge as fallback.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.INVALID_RESPONSE,
    },
    # --- Budget and circuit states (NOT backend breaker events) ---
    SearchErrorCode.BUDGET_EXHAUSTED: {
        "retryable": False,
        "suggestion": "Use a different intent (general/SearXNG is unlimited) or rely on internal knowledge.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.BUDGET,
    },
    SearchErrorCode.CIRCUIT_OPEN: {
        "retryable": True,
        "suggestion": "Wait 5 minutes for circuit to half-open, or use a different intent.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.CIRCUIT,
    },
    SearchErrorCode.ALL_BACKENDS_FAILED: {
        "retryable": False,
        "suggestion": "Rely on internal knowledge or prompt the user for manual input.",
        "breaker_relevant": False,
        "diagnostic_category": SearchDiagnosticCategory.ALL_BACKENDS_FAILED,
    },
}


def error_to_search_result(error: SearchError) -> dict[str, Any]:
    """Serializa un SearchError a dict (para retornar al LLM).

    PRE2-A1: extended with the safe structured fields
    ``breaker_relevant``, ``http_status``, ``diagnostic_category``.
    The output NEVER contains raw exception text, response bodies,
    response headers, or credentials. The ``error`` field is the
    safe static ``message`` set at construction time.

    Returns:
        dict con shape estable (P1-1 v1.2: incluye backends_tried
        y reasons para que el LLM sepa que se intento y por que fallo;
        PRE2-A1: incluye breaker_relevant, http_status, diagnostic_category):

        {
            "error": str (safe static message),
            "code": str (SearchErrorCode.value),
            "backend": str | None,
            "retryable": bool,
            "suggestion": str,
            "backends_tried": list[str],
            "reasons": dict[str, str],
            "breaker_relevant": bool,
            "http_status": int | None,
            "diagnostic_category": str (SearchDiagnosticCategory.value),
        }
    """
    return {
        "error": error.message,
        "code": error.code.value,
        "backend": error.backend,
        "retryable": error.retryable,
        "suggestion": error.suggestion,
        "backends_tried": error.backends_tried,
        "reasons": error.reasons,
        "breaker_relevant": error.breaker_relevant,
        "http_status": error.http_status,
        "diagnostic_category": error.diagnostic_category.value,
    }
