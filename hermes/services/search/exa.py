"""Sprint 9.3: Exa Backend (Capa 5).

API hosted: Exa neural search (semantic, find similar to X).
Soporta solo snippet (campo 'highlights' de Exa).

P2-4 fix v1.2: type='neural' cuando intent='semantic', 'keyword' en otros.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from hermes.services.search.protocol import DEFAULT_SIZE_GUARD_CHARS, SearchResult

if TYPE_CHECKING:
    from hermes.services.search.budget import BudgetTracker

logger = logging.getLogger(__name__)


class ExaBackend:
    """Exa backend (semantic intent, hosted API).

    Attributes:
        api_key: Exa API key (env: EXA_API_KEY).
        budget: BudgetTracker para has_budget().
        timeout: HTTP request timeout en segundos (default 15.0).
    """

    name: str = "exa"
    SUPPORTED_CONTENT_MODES: frozenset[str] = frozenset({"snippet"})

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
        """Ejecuta busqueda en Exa via HTTP POST.

        Args:
            query: la query.
            content_mode: solo 'snippet' (Exa no genera summaries).
            num_results: numero de resultados.
            intent: 'semantic' usa type='neural' (vector search).
                Otros usan type='keyword' (factual).

        Returns:
            SearchResult con los resultados. 'highlights' de Exa se
            concatenan en el campo 'content' del resultado.

        Raises:
            httpx.HTTPError: si la conexion falla o retorna 4xx/5xx.
        """
        # P2-4 fix v1.2: neural para semantic, keyword para otros
        search_type = "neural" if intent == "semantic" else "keyword"

        payload: dict[str, Any] = {
            "query": query,
            "numResults": num_results,
            "type": search_type,
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                "https://api.exa.ai/search",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        raw_results = data.get("results", [])
        results: list[dict[str, Any]] = []
        for r in raw_results:
            # Exa retorna 'highlights' como array. Concatenamos en 'content'.
            highlights = r.get("highlights", [])
            content = " ".join(highlights) if highlights else ""
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": content,
                    "score": r.get("score", 0.0),
                }
            )

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
        """True si Exa responde 200 (o 401/403 con key valida, API up)."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    "https://api.exa.ai/",
                    headers={"x-api-key": self._api_key},
                )
                return response.status_code in (200, 401, 403)
        except Exception as exc:
            logger.debug(
                "exa_health_check_failed",
                extra={"error": str(exc)[:200]},
            )
            return False
