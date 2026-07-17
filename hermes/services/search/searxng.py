"""Sprint 9.3: SearXNG Backend (Capa 2).

Self-hosted meta-search engine. Privacy-first default backend.
HTTP GET a /search?q={query}&format=json&count={n}.

P2-3 fix v1.2: URL-encode la query para soportar espacios y
caracteres especiales.

P1-5 fix v1.2: healthcheck usa endpoint root / (SearXNG no
expone /healthz).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx

from hermes.services.search.protocol import DEFAULT_SIZE_GUARD_CHARS, SearchResult

if TYPE_CHECKING:
    from hermes.services.search.budget import BudgetTracker

logger = logging.getLogger(__name__)


class SearXNGBackend:
    """SearXNG backend (self-hosted, default privacy-first).

    Attributes:
        url: base URL del SearXNG container (e.g., http://searxng:8888).
        budget: BudgetTracker para has_budget().
        timeout: HTTP request timeout en segundos.
    """

    name: str = "searxng"
    SUPPORTED_CONTENT_MODES: frozenset[str] = frozenset({"snippet"})

    def __init__(
        self,
        url: str,
        budget: BudgetTracker,
        timeout: float = 10.0,
    ) -> None:
        self._url = url.rstrip("/")
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
        """Ejecuta busqueda en SearXNG via HTTP GET.

        Args:
            query: la query (URL-encoded).
            content_mode: solo 'snippet' (SearXNG no genera summaries).
            num_results: numero de resultados.
            intent: ignorado (SearXNG no usa intent).

        Returns:
            SearchResult con los resultados parseados.

        Raises:
            httpx.HTTPError: si la conexion falla.
            json.JSONDecodeError: si el response no es JSON valido.
        """
        # P2-3 fix v1.2: URL-encode la query
        encoded_query = quote(query, safe="")
        url = f"{self._url}/search" f"?q={encoded_query}" f"&format=json" f"&count={num_results}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        # SearXNG retorna {"results": [{title, url, content, engine}, ...]}
        raw_results = data.get("results", [])
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "engine": r.get("engine", "unknown"),
            }
            for r in raw_results
        ]

        return SearchResult(
            results=results,
            backend_used=self.name,
            query=query,
            content_mode=content_mode,
            original_content_mode=content_mode,
            format_fallback=False,  # SearXNG solo soporta snippet, no hay fallback
            size_guard_chars=DEFAULT_SIZE_GUARD_CHARS,
            truncated=False,
        )

    async def has_budget(self) -> bool:
        """Delega a BudgetTracker."""
        return await self._budget.has_budget(self.name)

    async def health_check(self) -> bool:
        """True si el endpoint root responde 200.

        P1-5 fix v1.2: SearXNG no expone /healthz. Usamos root.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self._url}/")
                return response.status_code == 200
        except Exception as exc:
            logger.debug(
                "searxng_health_check_failed",
                extra={"url": self._url, "error": str(exc)[:200]},
            )
            return False
