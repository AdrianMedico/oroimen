"""Sprint 9.3: Errores estructurados del Web Search Router.

P0 Gemini 3.1 Pro: el LLM debe recibir errores estructurados
(no strings vagos) para poder razonar sobre el fallback.

P1 Gemini 3.5 Thinking v1.1: el codigo del router hace referencia
a `_build_structured_error` que no existia en v1.0. Este modulo
lo implementa.
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


@dataclass(frozen=True)
class SearchError:
    """Error estructurado del Web Search Router.

    Attributes:
        code: codigo normalizado (SearchErrorCode).
        message: mensaje human-readable para el LLM.
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
    """

    code: SearchErrorCode
    message: str
    backend: str | None
    retryable: bool
    suggestion: str
    backends_tried: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


def _build_structured_error(
    code: SearchErrorCode,
    message: str,
    backend: str | None,
    retryable: bool = False,
    suggestion: str = "Rely on internal knowledge or prompt the user for manual input.",
    backends_tried: list[str] | None = None,
    reasons: dict[str, str] | None = None,
) -> SearchError:
    """Construye un SearchError con defaults sensatos.

    Args:
        code: codigo normalizado.
        message: mensaje tecnico.
        backend: nombre del backend o None.
        retryable: True si reintentar puede funcionar.
        suggestion: texto para el LLM.
        backends_tried: lista de backends intentados (P1-1 v1.2).
        reasons: dict backend -> razon de fallo (P1-1 v1.2).

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
    )


# Mapeo de codigos a defaults utiles para tests/serializacion.
ERROR_DEFAULTS: dict[SearchErrorCode, dict[str, Any]] = {
    SearchErrorCode.EMPTY_QUERY: {
        "retryable": False,
        "suggestion": "Ask the user to clarify their query.",
    },
    SearchErrorCode.INVALID_INTENT: {
        "retryable": False,
        "suggestion": "Use one of: general, semantic, deep_research.",
    },
    SearchErrorCode.INVALID_CONTENT: {
        "retryable": False,
        "suggestion": "Use one of: snippet, summary, full.",
    },
    SearchErrorCode.TIMEOUT: {
        "retryable": True,
        "suggestion": "Try again with a simpler query or different intent.",
    },
    SearchErrorCode.CONNECTION_REFUSED: {
        "retryable": True,
        "suggestion": "Check network connectivity. If persistent, use internal knowledge.",
    },
    SearchErrorCode.INVALID_RESPONSE: {
        "retryable": False,
        "suggestion": "Report this as a bug. Use internal knowledge as fallback.",
    },
    SearchErrorCode.BUDGET_EXHAUSTED: {
        "retryable": False,
        "suggestion": "Use a different intent (general/SearXNG is unlimited) or rely on internal knowledge.",
    },
    SearchErrorCode.CIRCUIT_OPEN: {
        "retryable": True,
        "suggestion": "Wait 5 minutes for circuit to half-open, or use a different intent.",
    },
    SearchErrorCode.ALL_BACKENDS_FAILED: {
        "retryable": False,
        "suggestion": "Rely on internal knowledge or prompt the user for manual input.",
    },
}


def error_to_search_result(error: SearchError) -> dict[str, Any]:
    """Serializa un SearchError a dict (para retornar al LLM).

    Returns:
        dict con shape estable (P1-1 v1.2: incluye backends_tried
        y reasons para que el LLM sepa que se intento y por que fallo):
        {
            "error": str (message),
            "code": str (SearchErrorCode.value),
            "backend": str | None,
            "retryable": bool,
            "suggestion": str,
            "backends_tried": list[str],
            "reasons": dict[str, str],
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
    }
