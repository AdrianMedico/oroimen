"""Sprint 9.3: MCP tool registration para Web Search Router (Capa 7).

Define el schema MCP de `hermes_search` y expone el callable
que el agent loop registra. El schema NO leakea nombres de
backends (P2-1 v1.2) — el LLM solo sabe que hay un privacy-first
default y modos para casos especificos.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from hermes.services.search.errors import error_to_search_result
from hermes.services.search.protocol import SearchResult
from hermes.services.search.router import hermes_search

if TYPE_CHECKING:
    from hermes.services.search.budget import BudgetTracker
    from hermes.services.search.resilience import (
        CircuitBreakerRegistry,
        ConcurrencyLimiter,
    )

# P2-1 v1.2: description NO leakea "SearXNG", "Tavily", "Exa".
# El LLM solo sabe que hay privacy-first default y 3 modos de busqueda.
SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "hermes_search",
        "description": (
            "Web search (privacy-first by default). "
            "Use intent='semantic' for 'find similar to X', 'papers like X', "
            "or 'articles like Y' (good for research). "
            "Use intent='deep_research' for comprehensive research with "
            "extracted content and summaries. "
            "Use intent='general' for factual queries, current events, "
            "or anything else (privacy-first by default). "
            "The 'content' parameter controls result richness: 'snippet' "
            "(~300 chars/result, default), 'summary' (~1.5K chars, only "
            "for deep_research — others degrade to snippet), or 'full' "
            "(complete content, only for deep_research — others degrade). "
            "Results are truncated at size_guard_chars (default 200K chars) "
            "regardless of mode. If the search fails, you receive a "
            "structured error with retryable flag and suggestion — never "
            "a silent failure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (1-2000 chars).",
                    "minLength": 1,
                    "maxLength": 2000,
                },
                "intent": {
                    "type": "string",
                    "enum": ["general", "semantic", "deep_research"],
                    "default": "general",
                    "description": (
                        "Type of search. 'semantic' = vector similarity "
                        "(find similar to X). 'deep_research' = comprehensive "
                        "with content extraction. 'general' = factual/keyword "
                        "(privacy-first default)."
                    ),
                },
                "content": {
                    "type": "string",
                    "enum": ["snippet", "summary", "full"],
                    "default": "snippet",
                    "description": (
                        "Content richness per result. 'snippet' = ~300 chars. "
                        "'summary' = ~1.5K chars (deep_research only, others "
                        "degrade silently to snippet). 'full' = complete content "
                        "(deep_research only, others degrade silently to snippet). "
                        "Truncated at size_guard_chars (default 200K chars) "
                        "regardless of mode."
                    ),
                },
                "num_results": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Number of results to return (1-50).",
                },
            },
            "required": ["query"],
        },
    },
}


def make_search_tool_callable(
    backends: dict[str, Any],
    budget: BudgetTracker,
    circuit_breaker: CircuitBreakerRegistry,
    semaphore: ConcurrencyLimiter,
    size_guard_chars: int = 200000,
    default_num_results: int = 10,
    max_num_results: int = 50,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Crea un callable que el agent loop registra como `hermes_search`.

    El callable envuelve hermes_search con los componentes inyectados
    (backends, budget, circuit breaker, semaphore, size guard).
    Retorna un dict JSON-serializable para el LLM, NO un SearchResult
    (los dataclasses no son JSON-serializables directamente).

    Args:
        backends: dict inyectado con backends disponibles.
        budget: BudgetTracker compartido.
        circuit_breaker: CircuitBreakerRegistry compartido.
        semaphore: ConcurrencyLimiter compartido.
        size_guard_chars: limite de chars para truncation.
        default_num_results: num_results default si el LLM no lo pasa.
        max_num_results: hard cap para evitar abuse del LLM.

    Returns:
        Async callable(query, intent, content, num_results) -> dict.
    """

    async def search_tool(
        query: str,
        intent: str = "general",
        content: str = "snippet",
        num_results: int | None = None,
    ) -> dict[str, Any]:
        # Clamp num_results al rango valido
        n = num_results if num_results is not None else default_num_results
        n = max(1, min(max_num_results, n))

        result = await hermes_search(
            query=query,
            intent=intent,
            content=content,
            num_results=n,
            backends=backends,
            budget=budget,
            circuit_breaker=circuit_breaker,
            semaphore=semaphore,
            size_guard_chars=size_guard_chars,
        )
        return _serialize_result(result)

    return search_tool


def _serialize_result(result: SearchResult) -> dict[str, Any]:
    """Convierte SearchResult a dict JSON-serializable para el LLM.

    Si result.error esta poblado, retorna un error estructurado.
    Si no, retorna los results + metadata.
    """
    if result.error is not None:
        return {
            "success": False,
            **error_to_search_result(result.error),
        }
    return {
        "success": True,
        "results": result.results,
        "query": result.query,
        "backend_used": result.backend_used,
        "content_mode": result.content_mode,
        "original_content_mode": result.original_content_mode,
        "format_fallback": result.format_fallback,
        # S9.3.1 punto 6: format_fallback_reason explica al LLM por que se degrado
        "format_fallback_reason": result.format_fallback_reason,
        "truncated": result.truncated,
        "truncated_at_chars": result.truncated_at_chars,
    }
