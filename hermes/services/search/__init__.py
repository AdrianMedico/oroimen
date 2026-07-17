"""Sprint 9.3: Web Search Router (multi-backend).

Backends implementados:
- SearXNG (self-hosted, default para intent='general' y como fallback)
- Tavily (deep_research, output optimizado para LLMs)
- Exa (semantic vector search, find similar to X)

Arquitectura: un solo tool MCP (`hermes_search`) con routing interno
basado en `intent` declarado por el LLM. Circuit fallback (infra) +
format fallback (capacidades del backend) aplicados en orden.

Ver:
- docs/TDD_S9_3_WEB_SEARCH_ROUTER.md v1.3
- ADR-001 (cross-review disciplinado)
- 3 rondas de cross-review (Gemini 3.5 Thinking + GLM 5.2)
"""

from __future__ import annotations

from hermes.services.search.errors import (
    SearchError,
    SearchErrorCode,
)
from hermes.services.search.protocol import (
    ALL_CONTENT_MODES,
    BackendProtocol,
    ContentMode,
    SearchResult,
)

__all__ = [
    "ALL_CONTENT_MODES",
    "BackendProtocol",
    "ContentMode",
    "SearchError",
    "SearchErrorCode",
    "SearchResult",
]
