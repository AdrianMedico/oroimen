"""Tests Sprint 15 (US-3.1 §4 PR #69): search_files tool.

Cubre:
- Tool retorna resultados con file_id + filename + score.
- Tool retorna disabled=true si RAG está apagado.
- Tool clampa top_k a [1, 20].
- Tool retorna query vacio -> error claro.
- Tool no falla si cosine_search lanza excepcion.
- Tool enriquece file_id con filename (lookup en DB).

Patron: usa `embeddings_mock` fixture (vector fijo, no toca OpenRouter).
Cosine similarity siempre = 1.0 (vector constante), asi que search_files
devuelve todos los files embebidos en cualquier orden.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
async def seeded_db(tmp_path):
    """DB con 3 archivos embebidos (todos con el mismo vector constante)."""
    import numpy as np

    from hermes.memory.db import Database

    db = Database(tmp_path / "test.db")
    await db.initialize()
    files = [
        ("file_alpha", "alpha.md", "contenido alpha"),
        ("file_beta", "beta.md", "contenido beta"),
        ("file_gamma", "gamma.md", "contenido gamma"),
    ]
    for fid, fname, text in files:
        await db.add_file(fid, fname, "text/markdown", len(text), text, "raw", "vault")
        vec = np.full(4096, 0.5, dtype=np.float32)
        await db.add_file_embedding(fid, vec.tobytes(), model="test")
    yield db
    await db.close()


async def test_search_files_returns_results_with_scores(
    seeded_db: Any, embeddings_mock: Any
) -> None:
    """Tool retorna JSON con file_id, filename, score para cada match."""
    from hermes.tools.search_files import search_files_tool_callable

    result_str = await search_files_tool_callable(
        query="anything",
        top_k=5,
        embeddings_service=embeddings_mock,
        db=seeded_db,
    )
    result = json.loads(result_str)
    assert "results" in result
    assert "count" in result
    assert result["count"] == 3
    for r in result["results"]:
        assert "file_id" in r
        assert "filename" in r
        assert "score" in r
        # Score es 1.0 (vector constante en el mock).
        assert r["score"] == 1.0


async def test_search_files_clamps_top_k(seeded_db: Any, embeddings_mock: Any) -> None:
    """top_k fuera de [1, 20] se clampa a los limites.

    Defensa contra callers que pasen valores extremos (e.g. 1000
    daria un payload enorme; 0 no devolveria nada util).
    """
    from hermes.tools.search_files import search_files_tool_callable

    # top_k=100 se clampa a 20 -> no afecta a este test (solo hay 3)
    r = await search_files_tool_callable(
        query="x", top_k=100, embeddings_service=embeddings_mock, db=seeded_db
    )
    data = json.loads(r)
    assert data["count"] == 3  # solo 3 en la library

    # top_k=0 se clampa a 1 -> devolvemos solo 1
    r = await search_files_tool_callable(
        query="x", top_k=0, embeddings_service=embeddings_mock, db=seeded_db
    )
    data = json.loads(r)
    assert data["count"] == 1  # solo 1


async def test_search_files_empty_query(seeded_db: Any, embeddings_mock: Any) -> None:
    """Query vacio -> error claro, no llama al backend."""
    from hermes.tools.search_files import search_files_tool_callable

    result_str = await search_files_tool_callable(
        query="", embeddings_service=embeddings_mock, db=seeded_db
    )
    result = json.loads(result_str)
    assert result["results"] == []
    assert "error" in result
    assert "vacio" in result["error"]


async def test_search_files_rag_disabled_returns_disabled_true(
    seeded_db: Any,
) -> None:
    """Si embeddings.is_enabled = False -> retorna disabled=true con reason."""
    from hermes.tools.search_files import search_files_tool_callable

    class _Disabled:
        is_enabled = False

    result_str = await search_files_tool_callable(
        query="anything", embeddings_service=_Disabled(), db=seeded_db
    )
    result = json.loads(result_str)
    assert result["disabled"] is True
    assert "reason" in result
    assert result["results"] == []


async def test_search_files_handles_cosine_search_exception(
    seeded_db: Any,
) -> None:
    """Si cosine_search lanza excepcion, la tool retorna error claro.

    No debe propagar la excepcion (romperia el agent loop). En su lugar,
    log + JSON estructurado con error.
    """
    from hermes.tools.search_files import search_files_tool_callable

    class _FailingService:
        is_enabled = True

        async def cosine_search(self, query: str, top_k: int):
            raise RuntimeError("OpenRouter timeout")

    result_str = await search_files_tool_callable(
        query="anything",
        embeddings_service=_FailingService(),
        db=seeded_db,
    )
    result = json.loads(result_str)
    assert "error" in result
    assert "cosine_search" in result["error"]
    assert result["results"] == []


async def test_search_files_enriches_with_filename_from_db(
    seeded_db: Any, embeddings_mock: Any
) -> None:
    """Cada resultado tiene el filename original, no el file_id."""
    from hermes.tools.search_files import search_files_tool_callable

    result_str = await search_files_tool_callable(
        query="x", embeddings_service=embeddings_mock, db=seeded_db
    )
    result = json.loads(result_str)
    filenames = {r["filename"] for r in result["results"]}
    # Nombres originales sembrados.
    assert "alpha.md" in filenames
    assert "beta.md" in filenames
    assert "gamma.md" in filenames


async def test_search_files_prefers_chunk_search_and_returns_text(seeded_db: Any) -> None:
    """Drop-folder chunks are returned with text after lazy initialization."""
    from hermes.tools.search_files import search_files_tool_callable

    class _LazyEmbeddings:
        is_enabled = False

        async def ensure_initialized(self) -> None:
            self.is_enabled = True

        async def cosine_search(self, query: str, top_k: int):
            raise AssertionError("legacy file_embeddings path must not run")

    class _ChunkSearch:
        async def search(self, query: str, *, top_k: int):
            return [
                SimpleNamespace(
                    file_id="file_alpha",
                    source_path="/app/drop/alpha.md",
                    score=0.91,
                    chunk_index=2,
                    text="grounded chunk text",
                )
            ]

    result = json.loads(
        await search_files_tool_callable(
            query="grounded",
            embeddings_service=_LazyEmbeddings(),
            db=seeded_db,
            vault_embedder=_ChunkSearch(),
        )
    )
    assert result == {
        "results": [
            {
                "file_id": "file_alpha",
                "filename": "alpha.md",
                "score": 0.91,
                "chunk_index": 2,
                "text": "grounded chunk text",
            }
        ],
        "count": 1,
    }


async def test_search_files_empty_library(embeddings_mock: Any, tmp_path) -> None:
    """Si la DB no tiene embeddings -> result count=0, no crash."""
    import numpy as np  # noqa: F401  (fixture abajo usa)

    from hermes.memory.db import Database
    from hermes.tools.search_files import search_files_tool_callable

    db = Database(tmp_path / "empty.db")
    await db.initialize()
    try:
        result_str = await search_files_tool_callable(
            query="anything",
            embeddings_service=embeddings_mock,
            db=db,
        )
        result = json.loads(result_str)
        assert result["count"] == 0
        assert result["results"] == []
        assert "hint" in result  # hint para el LLM
    finally:
        await db.close()
