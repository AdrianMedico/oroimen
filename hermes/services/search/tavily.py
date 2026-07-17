"""Sprint 9.3: Tavily Backend (Capa 4).

API hosted: Tavily search con content extraction optimizado para LLMs.
Soporta los 3 content modes: snippet, summary, full.

P0-3 fix v1.2: search_depth derivado de `intent == "deep_research"`.
P1-7 fix v1.2: usar `content_mode` (no `content`) en comparaciones.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from hermes.services.search.protocol import DEFAULT_SIZE_GUARD_CHARS, SearchResult

if TYPE_CHECKING:
    from hermes.services.search.budget import BudgetTracker

logger = logging.getLogger(__name__)


class TavilyBackend:
    """Tavily backend (deep_research intent, hosted API).

    Attributes:
        api_key: Tavily API key (env: TAVILY_API_KEY).
        budget: BudgetTracker para has_budget().
        timeout: HTTP request timeout en segundos (default 15.0).
    """

    name: str = "tavily"
    SUPPORTED_CONTENT_MODES: frozenset[str] = frozenset({"snippet", "summary", "full"})

    def __init__(
        self,
        api_key: str,
        budget: BudgetTracker,
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._budget = budget
        self._timeout = timeout

    async def search(
        self,
        query: str,
        content_mode: str,
        num_results: int,
        *,
        intent: str = "general",
    ) -> SearchResult:
        """Ejecuta busqueda en Tavily via HTTP POST.

        Args:
            query: la query.
            content_mode: 'snippet' (default), 'summary', o 'full'.
            num_results: numero de resultados (max 20 segun Tavily).
            intent: 'deep_research' usa search_depth='advanced'.

        Returns:
            SearchResult con los resultados. El campo 'content' del
            resultado depende de content_mode:
            - snippet: campo 'content' de Tavily (~300 chars)
            - summary: campo 'summary' de Tavily (~1.5K chars)
            - full: campo 'raw_content' de Tavily (markdown completo)

        Raises:
            httpx.HTTPError: si la conexion falla o retorna 4xx/5xx.
        """
        # P0-3 fix v1.2: search_depth derivado de intent
        # P1-7 fix v1.2: usar content_mode (no content) en comparaciones
        search_depth = "advanced" if intent == "deep_research" else "basic"
        include_raw = content_mode == "full"

        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": search_depth,
            "include_raw_content": include_raw,
        }
        headers = {
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        raw_results = data.get("results", [])
        results: list[dict[str, Any]] = []
        for r in raw_results:
            item: dict[str, Any] = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "score": r.get("score", 0.0),
            }
            # Content mode selecciona que campo incluir
            if content_mode == "full":
                item["raw_content"] = r.get("raw_content", "")
            elif content_mode == "summary":
                item["summary"] = r.get("summary", "")
                item["content"] = r.get("content", "")
            else:  # snippet
                item["content"] = r.get("content", "")
            results.append(item)

        return SearchResult(
            results=results,
            backend_used=self.name,
            query=query,
            content_mode=content_mode,
            original_content_mode=content_mode,
            format_fallback=False,
            size_guard_chars=DEFAULT_SIZE_GUARD_CHARS,
            truncated=False,
        )

    async def has_budget(self) -> bool:
        """Delega a BudgetTracker."""
        return await self._budget.has_budget(self.name)

    async def health_check(self) -> bool:
        """True si Tavily responde 200 a un GET liviano."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                # Tavily no tiene endpoint /healthz, usamos un GET
                # liviano. 200 = OK, 401/403 = OK tambien (key valida
                # pero ruta no autorizada, lo que significa API up).
                response = await client.get(
                    "https://api.tavily.com/",
                )
                return response.status_code in (200, 401, 403)
        except Exception as exc:
            logger.debug(
                "tavily_health_check_failed",
                extra={"error": str(exc)[:200]},
            )
            return False
