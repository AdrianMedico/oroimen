"""Sprint 9.3 + PRE2-A2: BackendProtocol y SearchResult.

Contrato comun para backends de busqueda web (SearXNG, Tavily, Exa).
El router usa este Protocol para dispatch polimorfico.

P0 Copilot review (Gemini 3.5 Thinking 2026-06-26): el campo
SUPPORTED_CONTENT_MODES declara que modos de contenido soporta
el backend nativamente. El router usa esto para Format Fallback
(degradar silenciosamente a 'snippet' si el modo pedido no
esta soportado, sin error al LLM).

P1-1 (Gemini 3.5 Thinking): Size Guard de 50K chars por default,
truncamiento determinista (text[:N]) independiente del tokenizador.

PRE2-A2 (backend query-length capabilities): adds the immutable
``BackendQueryCapabilities`` shape and a stable class-level
``QUERY_CAPABILITIES`` attribute on ``BackendProtocol``. The router
uses this to validate the ORIGINAL query against the SELECTED
backend's declared limit AFTER backend selection (including
fallback) and BEFORE the generic 2000-char truncation, so the
rejection surfaces the original length (not a truncated one).
``max_query_chars = None`` means unknown / no provider-specific
local rejection; it is NOT a guarantee of unlimited acceptance.

The 399 value carried by ``TAVILY_MAX_QUERY_CHARS`` is an
Oroimen-side conservative operational / compatibility cap. It is
a defensive cut that protects the local pipeline until live Tavily
measurements confirm the actual provider limit. It is NOT a claim
that Tavily's hosted API currently enforces a hard 399-character
limit on the ``query`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hermes.services.search.errors import SearchError

# Content modes soportados por al menos un backend.
# type-safe via Literal; frozenset para runtime membership checks.
ContentMode = Literal["snippet", "summary", "full"]
ALL_CONTENT_MODES: frozenset[str] = frozenset({"snippet", "summary", "full"})

# S9.3.1: shared constant para size guard default.
# 200K chars = ~50K tokens ≈ $0.015 input / $0.03 output por search
# (MiniMax-M3 a $0.30/$0.60 per M tokens, MiniMax API).
# Permite mas contexto al LLM (cross-reference) sin truncar agresivo.
# Backends y router referencian este constant para evitar stale 50000.
DEFAULT_SIZE_GUARD_CHARS: int = 200000


# PRE2-A2: Oroimen conservative operational / compatibility cap.
# 399 is a defensive cut that protects the local pipeline until live
# Tavily measurements confirm the actual provider limit. It is NOT
# a claim that the hosted Tavily API currently enforces a hard
# 399-character limit. The router validates the ORIGINAL query
# against this cap BEFORE the generic 2000-char truncation so the
# rejection surfaces the original length (not a truncated one).
TAVILY_MAX_QUERY_CHARS: int = 399


@dataclass(frozen=True)
class BackendQueryCapabilities:
    """Immutable capability declaration for a backend.

    PRE2-A2: minimal shape — the only capability the router needs to
    enforce per-backend query-length rejection is ``max_query_chars``.
    Adding new capability axes here requires updating the router's
    validation surface and adding a focused regression test.

    Attributes:
        max_query_chars: Oroimen-side operational cap for the
            query string the backend will accept. ``None`` means
            unknown / no provider-specific local rejection.
            ``None`` is NOT a guarantee of unlimited acceptance;
            the router does not create a provider-specific
            rejection for ``None``, but the generic
            ``_MAX_QUERY_CHARS`` ceiling (2000) still applies.
            The value reflects an Oroimen conservative operational
            / compatibility cut pending live provider validation;
            it is not a claim about the provider's current API limit.
    """

    max_query_chars: int | None


@dataclass(frozen=True)
class SearchResult:
    """Resultado normalizado de una busqueda web.

    Atributos:
        results: lista de dicts con shape:
            {
                "title": str,
                "url": str,
                "snippet": str,           # siempre presente (~300 chars)
                "content": str | None,    # solo si content_mode != 'snippet'
            }
        backend_used: nombre del backend que sirvio la peticion
            ('searxng' | 'tavily' | 'exa'). Util para logging y
            circuit breaker diagnostics.
        query: la query original (para contexto del LLM).
        content_mode: el modo de contenido REALMENTE usado (puede
            diferir de original_content_mode si hubo format fallback).
        original_content_mode: el modo que el LLM pidio originalmente.
        format_fallback: True si content_mode != original_content_mode
            (el router degrado el modo porque el backend no lo soporto).
        size_guard_chars: el limite configurado de chars para
            truncation. Default 200000 (~50K tokens aprox). El router
            aplica este limite via _apply_size_guard() y setea el
            field al valor real usado.
        truncated: True si el contenido fue truncado por size guard.
        truncated_at_chars: el punto de truncamiento real. None si
            no se trunco. El LLM usa esto para saber cuantos chars
            le faltan (no asumir silencio del final del documento).
        error: SearchError poblado si el router fallo. P0-5 v1.2:
            el router siempre retorna SearchResult; los errores viven
            en este campo (no en una union type). El LLM puede
            verificar `result.error is not None` antes de usar results.
    """

    results: list[dict[str, Any]]
    backend_used: str
    query: str
    content_mode: str
    original_content_mode: str
    format_fallback: bool
    size_guard_chars: int
    truncated: bool
    truncated_at_chars: int | None = None
    # S9.3.1 punto 6: format_fallback_reason explica al LLM por que se degrado.
    # None si no hubo fallback. Mensaje human-readable si hubo.
    format_fallback_reason: str | None = None
    error: SearchError | None = None


@runtime_checkable
class BackendProtocol(Protocol):
    """Interfaz comun para backends de busqueda web.

    Cada backend (SearXNG, Tavily, Exa) implementa este Protocol.
    El router (hermes_search) usa el Protocol para dispatch
    polimorfico y NO usa isinstance checks (duck typing via
    Protocol es suficiente gracias a @runtime_checkable).

    Capacidades declaradas:
        name: identificador del backend (para logs y routing).
        SUPPORTED_CONTENT_MODES: frozenset con los modos que el
            backend soporta nativamente. Usado por el router
            para Format Fallback. Cada backend concreta declara
            los suyos:

            - SearXNG: solo 'snippet' (no genera summaries nativamente)
            - Exa: solo 'snippet' (devuelve highlights, no summaries)
            - Tavily: 'snippet', 'summary', 'full' (todos)

        QUERY_CAPABILITIES: PRE2-A2. ``BackendQueryCapabilities``
            con el cap operacional/conservador de Oroimen para el
            tamano de query que el backend acepta. Declarado por
            cada backend concreta. El valor es un cut
            operacional defensivo pendiente de validacion con el
            provider real; NO es una afirmacion sobre el limite
            actual de la API del provider. Usado por el router
            para validar la query ORIGINAL contra el backend
            SELECCIONADO (incluyendo fallback), ANTES del
            truncamiento generico de 2000 chars y ANTES de
            adquirir semaforo, debitar budget, despachar
            ``search``, o mutar el circuit breaker.
            ``max_query_chars = None`` significa "desconocido /
            sin rechazo local provider-specific"; NO es una
            garantia de aceptacion ilimitada.

    Estado externo (NO en este Protocol, vive en BudgetTracker):
        - Budget mensual por backend (SQLite atomico).
        - Circuit breaker state (open/closed/half-open).

    Por que @runtime_checkable: el router puede hacer
    `isinstance(backend, BackendProtocol)` en tests para verificar
    que un mock cumple el contrato.
    """

    # Identificador del backend
    name: ClassVar[str]

    # Modos de contenido soportados nativamente (P0 Gemini 3.5)
    SUPPORTED_CONTENT_MODES: ClassVar[frozenset[str]]

    # PRE2-A2: query-length capability (inmutable, class-level).
    QUERY_CAPABILITIES: ClassVar[BackendQueryCapabilities]

    async def search(
        self,
        query: str,
        content_mode: str,
        num_results: int,
        *,
        intent: str = "general",
    ) -> SearchResult:
        """Ejecuta la busqueda en el backend.

        Args:
            query: la query de busqueda (no vacia, no >2K chars).
            content_mode: el modo de contenido pedido. El backend
                DEBE respetar este modo si esta en
                SUPPORTED_CONTENT_MODES. Si no, el router YA
                degrado a un modo soportado antes de llamar
                (Format Fallback).
            num_results: numero de resultados a retornar
                (default 10, max 50).
            intent: el intent original del LLM ('general',
                'semantic', 'deep_research'). Usado por Tavily
                para decidir search_depth ('advanced' para
                deep_research) y por Exa para type ('neural' para
                semantic). Fix P0-1 (bis) Gemini 3.5 Thinking
                2026-06-26: evita la variable `deep_research`
                indefinida del codigo original.

        Returns:
            SearchResult con los resultados. El backend es
            responsable de aplicar el size guard internamente
            (text[:size_guard_chars]) y marcar truncated=True
            si se trunco.

        Raises:
            RuntimeError: si la busqueda falla por timeout,
                error de red, o respuesta invalida. El router
                captura y decide si hacer circuit fallback.
        """
        ...

    async def has_budget(self) -> bool:
        """True si el backend todavia tiene budget este mes.

        Implementado por BudgetTracker (compartido entre backends).
        El backend consulta el tracker pasando self.name.
        """
        ...

    async def health_check(self) -> bool:
        """True si el backend esta operacional (no caido, no rate-limited).

        Para SearXNG: HTTP GET /healthz.
        Para Tavily/Exa: HTTP GET al endpoint mas barato
        (o cache del ultimo health check, TTL 5 min).

        Usado por el circuit breaker (3 fails -> open por 5 min).
        """
        ...
