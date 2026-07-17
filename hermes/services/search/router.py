"""Sprint 9.3: Web Search Router (Capa 6).

Implementa TODOS los fixes de las 3 rondas de cross-review:
- v1.0 -> v1.1 (Gemini 3.5 Thinking): 3 P0 + 2 P1
- v1.1 -> v1.2 (GLM 5.2): 7 P0 + 9 P1 + 6 P2
- v1.2 -> v1.3 (Gemini 3.5 Thinking): 2 P1 (concurrency)

Flow:
1. Validar inputs (empty/long query, invalid intent/content)
2. Resolver backend por intent
3. Circuit fallback (verificar SearXNG tambien, P0-2)
4. Format fallback (degradar content al maximo soportado, P0 Gemini)
5. Double-checked locking en budget (P1-1 v1.3)
6. record_usage ANTES de search (P1-2 v1.1)
7. asyncio.wait_for con _TIMEOUTS (P0-1 v1.1)
8. _normalize_result_urls (P1-1 v1.1)
9. _apply_size_guard (P1-9 v1.2)
10. search_query log estructurado (Capa 10, query_hash NO plain text)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from hermes.services.search.errors import (
    SearchError,
    SearchErrorCode,
    _build_structured_error,
)
from hermes.services.search.protocol import DEFAULT_SIZE_GUARD_CHARS, SearchResult

if TYPE_CHECKING:
    from hermes.services.search.budget import BudgetTracker
    from hermes.services.search.resilience import (
        CircuitBreakerRegistry,
        ConcurrencyLimiter,
    )

logger = logging.getLogger(__name__)


# Constantes del modulo
_BACKEND_BY_INTENT: dict[str, str] = {
    "general": "searxng",
    "semantic": "exa",
    "deep_research": "tavily",
}

# P0-1 fix v1.1: timeouts declarados a nivel de modulo
# P1-4 fix v1.2: per-backend, no unico
_TIMEOUTS: dict[str, float] = {
    "searxng": 10.0,
    "tavily": 15.0,
    "exa": 15.0,
}

_ALL_CONTENT_MODES: frozenset[str] = frozenset({"snippet", "summary", "full"})

# Maximo de chars en query (P0-1 v1.1: evita bills enormes)
_MAX_QUERY_CHARS = 2000

_DEFAULT_SIZE_GUARD_CHARS = DEFAULT_SIZE_GUARD_CHARS

# S9.3.1 punto 5: schemes peligrosos que se filtran pre-LLM.
# Case-insensitive (el router hace .lower() antes de comparar).
# Refs: OWASP XSS Filter Evasion Cheat Sheet.
_DANGEROUS_SCHEMES: tuple[str, ...] = (
    "javascript:",
    "data:",
    "file:",
    "vbscript:",
    "about:",
    "jar:",
)


async def hermes_search(
    query: str,
    intent: str = "general",
    content: str = "snippet",
    num_results: int = 10,
    *,
    backends: dict[str, Any],
    budget: BudgetTracker,
    circuit_breaker: CircuitBreakerRegistry,
    semaphore: ConcurrencyLimiter,
    size_guard_chars: int = _DEFAULT_SIZE_GUARD_CHARS,
) -> SearchResult:
    """Punto de entrada unico para el web search.

    Args:
        query: la query del LLM.
        intent: 'general' (default) | 'semantic' | 'deep_research'.
        content: 'snippet' (default) | 'summary' | 'full'.
        num_results: numero de resultados.
        backends: dict inyectado con los backends disponibles.
        budget: BudgetTracker para verificar/registrar uso.
        circuit_breaker: CircuitBreakerRegistry para state.
        semaphore: ConcurrencyLimiter para proteger rate limits.
        size_guard_chars: limite de chars para el size guard (default 50K).

    Returns:
        SearchResult con los resultados O con error populado.
    """
    start_time = time.monotonic()  # Capa 10: para duration_ms
    # 1. Validar inputs
    if not query or not query.strip():
        return _empty_result_with_error(
            _build_structured_error(
                code=SearchErrorCode.EMPTY_QUERY,
                message="Query cannot be empty",
                backend=None,
            )
        )
    if len(query) > _MAX_QUERY_CHARS:
        logger.warning(
            "search_query_truncated",
            extra={"original_len": len(query), "max": _MAX_QUERY_CHARS},
        )
        query = query[:_MAX_QUERY_CHARS]
    if intent not in _BACKEND_BY_INTENT:
        return _empty_result_with_error(
            _build_structured_error(
                code=SearchErrorCode.INVALID_INTENT,
                message=f"Unknown intent: {intent}. Use: general, semantic, deep_research.",
                backend=None,
            )
        )
    if content not in _ALL_CONTENT_MODES:
        return _empty_result_with_error(
            _build_structured_error(
                code=SearchErrorCode.INVALID_CONTENT,
                message=f"Unknown content mode: {content}. Use: snippet, summary, full.",
                backend=None,
            )
        )

    # 2. Resolver backend
    backend_name = _BACKEND_BY_INTENT[intent]

    # P0-1 fix: si el backend primario no esta configurado, fallback a SearXNG
    if backend_name not in backends:
        logger.info(
            "search_backend_missing_fallback",
            extra={"requested": backend_name, "to": "searxng"},
        )
        backend_name = "searxng"

    # 3. Circuit fallback PRIMERO (P0-2 fix v1.1)
    # P0-1 fix: verificar que el backend exista antes de acceder a el
    if (
        backend_name not in backends
        or circuit_breaker.is_open(backend_name)
        or not await backends[backend_name].has_budget()
    ):
        logger.info(
            "search_circuit_fallback",
            extra={"from_backend": backend_name, "to_backend": "searxng"},
        )
        backend_name = "searxng"
        # P0-1 fix: si SearXNG tampoco esta configurado o caido, ALL_BACKENDS_FAILED
        if "searxng" not in backends or circuit_breaker.is_open("searxng"):
            return _empty_result_with_error(
                _build_structured_error(
                    code=SearchErrorCode.ALL_BACKENDS_FAILED,
                    message="Primary and backup backends are down or not configured.",
                    backend="searxng",
                    backends_tried=list(_BACKEND_BY_INTENT.values()),
                    reasons={b: "CIRCUIT_OPEN" for b in _BACKEND_BY_INTENT.values()},
                )
            )
        backend = backends["searxng"]
    else:
        backend = backends[backend_name]

    # 4. Format fallback (P0 Gemini 3.5: capacidades del backend)
    original_content = content
    format_fallback = False
    format_fallback_reason: str | None = None
    if content not in backend.SUPPORTED_CONTENT_MODES:
        # S9.3.1 punto 6: Postura B (explÃ­cito al LLM).
        # Construimos un mensaje claro que el LLM ve en format_fallback_reason.
        # Asi el LLM puede decidir: reformular, usar deep_research, o aceptar el snippet.
        format_fallback_reason = (
            f"Requested content='{original_content}' but backend '{backend_name}' "
            f"only supports {sorted(backend.SUPPORTED_CONTENT_MODES)}. "
            f"Degraded to 'snippet'. Consider intent='deep_research' for richer content, "
            f"or accept this snippet-only result."
        )
        logger.info(
            "search_format_fallback",
            extra={
                "backend": backend_name,
                "requested": original_content,
                "fallback": "snippet",
                "supported_modes": sorted(backend.SUPPORTED_CONTENT_MODES),
            },
        )
        content = "snippet"
        format_fallback = True

    # 5. Double-checked locking (P1-1 v1.3) + execute
    # P0-3 fix v1.1: get_semaphore() retorna asyncio.Semaphore nativo
    # P1-2 fix v1.1: record_usage ANTES de search (budget leak prevention)
    async with semaphore.get_semaphore(backend_name):
        # P1-1 v1.3: re-verificar budget dentro del semaforo
        # (por si se agoto mientras esperabamos en cola)
        if not await backend.has_budget():
            return _empty_result_with_error(
                _build_structured_error(
                    code=SearchErrorCode.BUDGET_EXHAUSTED,
                    message=(f"Budget for {backend_name} exhausted while " "waiting in queue."),
                    backend=backend_name,
                    retryable=True,
                    suggestion=("Use intent='general' (unlimited SearXNG) " "or retry later."),
                )
            )

        # S9.3.1 punto 1: budget variable por intent/backend
        # Tavily: 1 credit basic, 2 credits advanced (deep_research)
        # Exa: 1 credit/search (no tier)
        # SearXNG: unlimited (count=0, has_budget() siempre True)
        usage_count = _compute_usage_cost(backend_name, intent)
        if usage_count > 0:
            await budget.record_usage(backend_name, count=usage_count)
        try:
            result = await asyncio.wait_for(
                backend.search(
                    query,
                    content,
                    num_results,
                    intent=intent,
                ),
                timeout=_TIMEOUTS[backend_name],
            )
        except TimeoutError:
            circuit_breaker.record_failure(backend_name)
            return _empty_result_with_error(
                _build_structured_error(
                    code=SearchErrorCode.TIMEOUT,
                    message=(f"Timeout after {_TIMEOUTS[backend_name]}s " f"on {backend_name}."),
                    backend=backend_name,
                    retryable=True,
                )
            )
        except Exception as exc:
            circuit_breaker.record_failure(backend_name)
            return _empty_result_with_error(
                _build_structured_error(
                    code=SearchErrorCode.INVALID_RESPONSE,
                    message=f"Runtime error: {str(exc)[:200]}",
                    backend=backend_name,
                    retryable=False,
                )
            )

    # 6. Success: reset circuit breaker
    circuit_breaker.record_success(backend_name)

    # 7. P1-1 v1.1: normalizar URLs (.rstrip('/'))
    # S9.3.1 punto 5: sanitizar URLs peligrosas (javascript:, data:, file:, vbscript:)
    # P1-4 fix: deduplicar URLs normalizadas
    # P1-2 v1.3: aplicar size guard sobre resultado NORMALIZADO + DEDUP
    normalized = _normalize_result_urls(result)
    sanitized = _sanitize_urls(normalized)
    deduped = _dedup_results(sanitized)
    final_result = _apply_size_guard(deduped, limit=size_guard_chars)

    # P1-1 fix: preservar original_content_mode y format_fallback
    if format_fallback:
        final_result = replace(
            final_result,
            format_fallback=True,
            format_fallback_reason=format_fallback_reason,
            original_content_mode=original_content,
        )

    # Capa 10: log estructurado (NO plain text query por privacy).
    duration_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "search_query",
        extra={
            "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
            "backend": final_result.backend_used,
            "intent": intent,
            "content_mode": final_result.content_mode,
            "num_results": len(final_result.results),
            "duration_ms": int(duration_ms),
            "status": "format_fallback" if final_result.format_fallback else "ok",
            # S9.3.1 punto 4: mÃ©trica hit-rate de size guard
            # truncated=True indica que el LLM recibiÃ³ MENOS info de la que pidiÃ³
            # hit_rate = count(truncated=True) / count(total queries)
            "size_guard_truncated": final_result.truncated,
            "size_guard_chars_limit": size_guard_chars,
            # S9.3.1 punto 6: format_fallback explÃ­cito al LLM
            "format_fallback": final_result.format_fallback,
        },
    )

    return final_result


# --- Helpers ---


def _compute_usage_cost(backend_name: str, intent: str) -> int:
    """S9.3.1: calcula el coste en credits segun backend + intent.

    Tavily: 1 credit basic (general/semantic), 2 credits advanced (deep_research)
    Exa: 1 credit/search
    SearXNG: 0 (self-hosted, unlimited)
    Otros: 1 credit/search (conservador)
    """
    if backend_name == "tavily":
        return 2 if intent == "deep_research" else 1
    if backend_name == "searxng":
        return 0
    return 1  # exa u otros


def _empty_result_with_error(error: SearchError) -> SearchResult:
    """Crea un SearchResult vacio con un error poblado.

    P0-5 fix v1.2: bridge entre SearchError y SearchResult.
    El router siempre retorna SearchResult; los errores viven
    en el campo `error`.
    """
    return SearchResult(
        results=[],
        backend_used=error.backend or "none",
        query="",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        format_fallback_reason=None,  # S9.3.1: errors no son format fallbacks
        size_guard_chars=0,
        truncated=False,
        error=error,
    )


def _normalize_result_urls(result: SearchResult) -> SearchResult:
    """P1-1 v1.1: normalizar URLs (.rstrip('/')) en cada resultado.

    Previene falsos duplicados tipo 'https://x.com/doc/' vs
    'https://x.com/doc'.
    """
    normalized_results = []
    for r in result.results:
        url = r.get("url", "")
        if url:
            r = {**r, "url": url.rstrip("/")}
        normalized_results.append(r)
    return replace(
        result,
        results=normalized_results,
    )


def _sanitize_urls(result: SearchResult) -> SearchResult:
    """S9.3.1 punto 5: filtra URLs con schemes peligrosos pre-LLM.

    Blacklist:
    - javascript: XSS injection
    - data: data URI (puede contener HTML/JS)
    - file: acceso a filesystem local
    - vbscript: VBScript injection (legacy IE)

    Tambien filtra URLs sin scheme vÃ¡lido (debe ser http: o https:).
    Results con URL peligrosa se ELIMINAN (no se enmascaran), porque
    el LLM podria decidir seguirlas.

    Refs: OWASP XSS Filter Evasion Cheat Sheet.
    """
    sanitized_results = []
    for r in result.results:
        url = r.get("url", "").strip().lower()
        if not url:
            sanitized_results.append(r)
            continue
        # Si el scheme no es http/https y no esta vacio, es peligroso
        if not (url.startswith("http://") or url.startswith("https://")):
            # Schemes peligrosos conocidos
            if any(url.startswith(scheme) for scheme in _DANGEROUS_SCHEMES):
                logger.warning(
                    "search_url_sanitized_dangerous_scheme",
                    extra={"url_prefix": url[:50]},
                )
                continue  # eliminar resultado
            # Otros schemes no-http (e.g., ftp://, ssh://) tambien se filtran
            # porque el LLM no deberia seguirlos sin contexto
            logger.warning(
                "search_url_sanitized_non_http_scheme",
                extra={"url_prefix": url[:50]},
            )
            continue
        sanitized_results.append(r)
    return replace(result, results=sanitized_results)


def _dedup_results(result: SearchResult) -> SearchResult:
    """P1-4 fix: elimina resultados con URL duplicada.

    Mantiene el primer resultado de cada URL (los backends suelen
    ordenar por relevance). Resultados sin URL se conservan todos
    (no se puede dedup sin clave).
    """
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in result.results:
        url = r.get("url", "")
        if not url:
            deduped.append(r)
            continue
        if url not in seen:
            seen.add(url)
            deduped.append(r)
    return replace(result, results=deduped)


def _apply_size_guard(result: SearchResult, limit: int) -> SearchResult:
    """P1-9 v1.2: trunca resultados completos, no strings parciales.

    Itera resultados ordenados, acumula chars del campo 'content',
    y corta cuando llega al limite. El LLM recibe solo resultados
    completos (no documentos a medias).
    """
    total = 0
    kept: list[dict[str, Any]] = []
    truncated = False
    for r in result.results:
        r_content = r.get("content", "")
        # Tambien considerar 'raw_content' (Tavily full mode)
        r_content_len = len(r_content) if r_content else len(r.get("raw_content") or "")
        if total + r_content_len > limit:
            truncated = True
            break
        kept.append(r)
        total += r_content_len
    return replace(
        result,
        results=kept,
        truncated=truncated,
        truncated_at_chars=total if truncated else None,
        size_guard_chars=limit,
    )
