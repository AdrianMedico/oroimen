"""Sprint 9.1: SearchFilesTool - busqueda semantica sobre la library.

Tool invocable por el LLM para buscar archivos por similaridad
semantica usando embeddings (text-embedding-3-small por defecto).
Retorna los top-k file_ids mas relevantes al query.

El LLM recibe el resultado como JSON con file_id, filename, y score.
Tras recibir el resultado, el LLM puede referenciar el file_id
en su respuesta (o, en futuras versiones, recargar el file via
`message.files[]` para que el contenido completo se inyecte).

Defense in depth: aunque la tool está registrada, valida que
`embeddings_service.is_enabled` antes de ejecutarse. Si RAG está
disabled, retorna un mensaje claro en vez de crashear.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


SEARCH_FILES_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Texto natural de la busqueda. Describe semanticamente "
                "lo que el user quiere encontrar (e.g. 'papers sobre "
                "machine learning', 'documentos que mencionen Python 3.14')."
            ),
        },
        "top_k": {
            "type": "integer",
            "description": "Numero maximo de resultados a retornar (default 5).",
            "minimum": 1,
            "maximum": 20,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def search_files_tool_callable(
    *,
    query: str,
    top_k: int = 5,
    embeddings_service: Any,  # EmbeddingsService
    db: Any,  # Database
    vault_embedder: Any | None = None,
) -> str:
    """Implementacion de la tool search_files.

    Args:
        query: texto natural de la busqueda.
        top_k: maximo de resultados (default 5, max 20).
        embeddings_service: EmbeddingsService instance.
        db: Database instance (para resolver file_id -> filename).

    Returns:
        JSON string con la lista de resultados. Formato:
        `{"results": [{"file_id": "...", "filename": "...", "score": 0.92}, ...], "disabled": false, "count": N}`

        Si RAG está disabled, retorna:
        `{"results": [], "disabled": true, "reason": "..."}`
    """
    if not query or not query.strip():
        return json.dumps(
            {"results": [], "error": "query vacio"},
            ensure_ascii=False,
        )
    ensure_initialized = getattr(embeddings_service, "ensure_initialized", None)
    if callable(ensure_initialized):
        try:
            await ensure_initialized()
        except Exception as exc:
            logger.exception("search_files_embeddings_init_failed")
            return json.dumps(
                {
                    "results": [],
                    "disabled": True,
                    "reason": f"RAG initialization failed: {type(exc).__name__}",
                },
                ensure_ascii=False,
            )
    if not getattr(embeddings_service, "is_enabled", False):
        return json.dumps(
            {
                "results": [],
                "disabled": True,
                "reason": "RAG disabled: no embedding tier is configured.",
            },
            ensure_ascii=False,
        )
    # Clamp top_k a [1, 20]
    top_k = max(1, min(20, top_k))
    if vault_embedder is not None:
        try:
            hits = await vault_embedder.search(query, top_k=top_k)
        except Exception as exc:
            logger.exception(
                "search_files_chunk_search_failed",
                extra={"query_preview": query[:100]},
            )
            return json.dumps(
                {
                    "results": [],
                    "error": f"chunk search failed: {type(exc).__name__}",
                },
                ensure_ascii=False,
            )
        if hits:
            chunk_results = [
                {
                    "file_id": hit.file_id,
                    "filename": Path(hit.source_path).name if hit.source_path else hit.file_id,
                    "score": round(hit.score, 4),
                    "chunk_index": hit.chunk_index,
                    "text": hit.text,
                }
                for hit in hits
            ]
            return json.dumps(
                {"results": chunk_results, "count": len(chunk_results)},
                ensure_ascii=False,
            )
    try:
        scored = await embeddings_service.cosine_search(query, top_k=top_k)
    except Exception as exc:
        logger.exception(
            "search_files_cosine_search_failed",
            extra={"query_preview": query[:100]},
        )
        return json.dumps(
            {
                "results": [],
                "error": f"cosine_search fallo: {type(exc).__name__}",
            },
            ensure_ascii=False,
        )
    if not scored:
        return json.dumps(
            {
                "results": [],
                "count": 0,
                "hint": (
                    "No se encontraron archivos relevantes. "
                    "Posible causa: el query no esta relacionado con "
                    "ningun file indexado, o todos los matches tienen "
                    "score < min_similarity_threshold."
                ),
            },
            ensure_ascii=False,
        )
    # Enriquecer con filename
    results: list[dict[str, Any]] = []
    for fid, score in scored:
        file_entry = await db.get_file(fid)
        filename = file_entry.get("filename", fid) if file_entry else fid
        results.append(
            {
                "file_id": fid,
                "filename": filename,
                "score": round(score, 4),
            }
        )
    return json.dumps(
        {"results": results, "count": len(results)},
        ensure_ascii=False,
    )
