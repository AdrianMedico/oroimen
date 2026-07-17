"""Tests Sprint 9.0: files persistence + file_refs en messages.

Cubre:
- Schema migration v0.5.8-s9 (tablas files, file_embeddings,
  memory_facts_staging, memory_facts, memory_fact_embeddings).
- DB methods: add_file, get_file, list_files, touch_file, delete_file.
- CASCADE behavior: delete_file limpia file_embeddings.
- file_refs en add_message + get_history.
- set_message_file_refs update.
- SchemaMigrator: idempotencia con SQL complejo (ALTER + CREATE TABLEs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.db import MIGRATIONS, Database, SchemaMigrator


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


# --- Schema migration v0.5.8-s9.0 ---


@pytest.mark.asyncio
async def test_s9_0_creates_files_table(db: Database) -> None:
    async with db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='files'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_s9_0_creates_file_embeddings_table(db: Database) -> None:
    async with db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='file_embeddings'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_s9_0_creates_memory_facts_staging_table(db: Database) -> None:
    async with db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_facts_staging'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_s9_0_creates_memory_facts_table(db: Database) -> None:
    async with db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_facts'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_s9_0_creates_memory_fact_embeddings_table(db: Database) -> None:
    async with db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_fact_embeddings'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_s9_0_creates_files_indices(db: Database) -> None:
    """Indices idx_files_last_referenced y idx_files_source."""
    async with db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='files'"
    ) as cur:
        rows = await cur.fetchall()
    names = {r[0] for r in rows}
    assert "idx_files_last_referenced" in names
    assert "idx_files_source" in names


@pytest.mark.asyncio
async def test_s9_0_creates_staging_indices(db: Database) -> None:
    async with db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memory_facts_staging'"
    ) as cur:
        rows = await cur.fetchall()
    names = {r[0] for r in rows}
    assert "idx_staging_status" in names


@pytest.mark.asyncio
async def test_s9_0_adds_file_refs_to_messages(db: Database) -> None:
    """ALTER messages ADD file_refs TEXT."""
    async with db.conn.execute("PRAGMA table_info(messages)") as cur:
        rows = await cur.fetchall()
    cols = {r[1] for r in rows}
    assert "file_refs" in cols


@pytest.mark.asyncio
async def test_s9_0_migration_idempotent(tmp_path: Path) -> None:
    """Re-aplicar SchemaMigrator en una DB ya migrada es un no-op."""
    # Primera migración
    d1 = Database(tmp_path / "idem.db")
    await d1.initialize()
    versions_1 = await SchemaMigrator(d1.conn, MIGRATIONS).applied_versions()
    assert 5 in versions_1
    await d1.close()
    # Segunda migración (re-open): debe ser no-op
    d2 = Database(tmp_path / "idem.db")
    await d2.initialize()
    versions_2 = await SchemaMigrator(d2.conn, MIGRATIONS).applied_versions()
    assert versions_1 == versions_2
    await d2.close()


# --- DB methods: add_file, get_file, list_files, touch_file, delete_file ---


@pytest.mark.asyncio
async def test_add_file_creates_row(db: Database) -> None:
    await db.add_file(
        file_id="file_abc123",
        filename="paper.pdf",
        mime_type="application/pdf",
        size_bytes=150_000,
        extracted_text="Lorem ipsum dolor sit amet",
        extraction_method="pypdf",
    )
    entry = await db.get_file("file_abc123")
    assert entry is not None
    assert entry["filename"] == "paper.pdf"
    assert entry["size_bytes"] == 150_000
    assert entry["extracted_text"] == "Lorem ipsum dolor sit amet"
    assert entry["source"] == "upload"  # default


@pytest.mark.asyncio
async def test_get_file_returns_none_for_missing(db: Database) -> None:
    entry = await db.get_file("file_nonexistent")
    assert entry is None


@pytest.mark.asyncio
async def test_list_files_returns_all_by_default(db: Database) -> None:
    await db.add_file("file_1", "a.pdf", "application/pdf", 100, "text a", "pypdf")
    await db.add_file("file_2", "b.txt", "text/plain", 50, "text b", "")
    rows = await db.list_files()
    assert len(rows) == 2
    filenames = {r["filename"] for r in rows}
    assert filenames == {"a.pdf", "b.txt"}


@pytest.mark.asyncio
async def test_list_files_filters_by_source(db: Database) -> None:
    await db.add_file("file_1", "a.pdf", "application/pdf", 100, "text", "pypdf", source="upload")
    await db.add_file(
        "file_2",
        "b.pdf",
        "application/pdf",
        200,
        "text",
        "pypdf",
        source="google_drive",
    )
    rows = await db.list_files(source="google_drive")
    assert len(rows) == 1
    assert rows[0]["id"] == "file_2"


@pytest.mark.asyncio
async def test_touch_file_increments_count(db: Database) -> None:
    await db.add_file("file_x", "x.pdf", "application/pdf", 100, "text", "pypdf")
    await db.touch_file("file_x")
    await db.touch_file("file_x")
    entry = await db.get_file("file_x")
    assert entry is not None
    assert entry["reference_count"] == 2
    assert entry["last_referenced_at"] is not None


@pytest.mark.asyncio
async def test_touch_file_silent_on_missing(db: Database) -> None:
    """Touch de un file que no existe no raise (graceful degradation)."""
    await db.touch_file("file_nonexistent")  # no debe raise
    # Verificar que nada raro pasó
    assert await db.get_file("file_nonexistent") is None


@pytest.mark.asyncio
async def test_delete_file_removes_row(db: Database) -> None:
    await db.add_file("file_del", "del.pdf", "application/pdf", 100, "text", "pypdf")
    deleted = await db.delete_file("file_del")
    assert deleted is True
    assert await db.get_file("file_del") is None


@pytest.mark.asyncio
async def test_delete_file_cascades_to_embeddings(db: Database) -> None:
    """Borrar file limpia file_embeddings via CASCADE."""
    await db.add_file("file_c", "c.pdf", "application/pdf", 100, "text", "pypdf")
    # Inyectar un embedding dummy (post-v25 schema requiere policy + dim NOT NULL;
    # usamos los defaults que add_file_embedding aplicaria).
    await db.conn.execute(
        "INSERT INTO file_embeddings (file_id, embedding, policy, dim) "
        "VALUES (?, ?, 'vault_ingest', 1536)",
        ("file_c", b"\x00" * 6144),  # 6KB dummy
    )
    await db.conn.commit()
    # Verificar que existe
    async with db.conn.execute("SELECT file_id FROM file_embeddings WHERE file_id='file_c'") as cur:
        row = await cur.fetchone()
    assert row is not None
    # Borrar y verificar cascade
    await db.delete_file("file_c")
    async with db.conn.execute("SELECT file_id FROM file_embeddings WHERE file_id='file_c'") as cur:
        row = await cur.fetchone()
    assert row is None


@pytest.mark.asyncio
async def test_delete_file_returns_false_for_missing(db: Database) -> None:
    deleted = await db.delete_file("file_nonexistent")
    assert deleted is False


# --- file_refs en messages ---


@pytest.mark.asyncio
async def test_add_message_with_file_refs_persists(db: Database) -> None:
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(
        conversation_id=conv_id,
        role="user",
        content="pregunta del user",
        file_refs=["file_1", "file_2"],
    )
    history = await db.get_history(conv_id)
    assert len(history) == 1
    assert history[0]["file_refs"] is not None
    import json as _json

    refs = _json.loads(history[0]["file_refs"])
    assert refs == ["file_1", "file_2"]


@pytest.mark.asyncio
async def test_add_message_without_file_refs_is_null(db: Database) -> None:
    """Backward compat: mensajes sin file_refs tienen file_refs NULL."""
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(conv_id, "user", "hola")
    history = await db.get_history(conv_id)
    assert history[0]["file_refs"] is None


@pytest.mark.asyncio
async def test_add_message_file_refs_dedup_preserves_order(db: Database) -> None:
    """add_message con file_refs duplicados deduplica preservando orden."""
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(
        conv_id,
        "user",
        "pregunta",
        file_refs=["file_1", "file_2", "file_1", "file_3", "file_2"],
    )
    history = await db.get_history(conv_id)
    import json as _json

    refs = _json.loads(history[0]["file_refs"])
    assert refs == ["file_1", "file_2", "file_3"]


@pytest.mark.asyncio
async def test_set_message_file_refs_updates_existing(db: Database) -> None:
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    msg_id = await db.add_message(conv_id, "user", "pregunta")
    await db.set_message_file_refs(msg_id, ["file_a", "file_b"])
    history = await db.get_history(conv_id)
    import json as _json

    refs = _json.loads(history[0]["file_refs"])
    assert refs == ["file_a", "file_b"]


@pytest.mark.asyncio
async def test_set_message_file_refs_empty_list_noop(db: Database) -> None:
    """set_message_file_refs con lista vacía no modifica el row."""
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    msg_id = await db.add_message(conv_id, "user", "pregunta")
    await db.set_message_file_refs(msg_id, [])
    history = await db.get_history(conv_id)
    assert history[0]["file_refs"] is None


# --- CASCADE cross-table (memory_facts.source_file_id) ---


@pytest.mark.asyncio
async def test_delete_file_cascades_to_memory_facts_source_file_id(db: Database) -> None:
    """P0-3 Gemini fix: borrar file limpia facts derivados via CASCADE."""
    await db.add_file("file_f", "f.pdf", "application/pdf", 100, "text", "pypdf")
    # Insertar un memory_fact con source_file_id
    await db.conn.execute(
        "INSERT INTO memory_facts (id, category, content, source_file_id) VALUES (?, ?, ?, ?)",
        ("fact_1", "academic_fact", "User uploaded f.pdf", "file_f"),
    )
    await db.conn.commit()
    # Verificar que existe
    async with db.conn.execute("SELECT id FROM memory_facts WHERE source_file_id='file_f'") as cur:
        row = await cur.fetchone()
    assert row is not None
    # Borrar file
    await db.delete_file("file_f")
    # Verificar que el fact se eliminó (CASCADE)
    async with db.conn.execute("SELECT id FROM memory_facts WHERE source_file_id='file_f'") as cur:
        row = await cur.fetchone()
    assert row is None


# ==============================================================================
# Sprint 15 (PR #66): Schema v15 + content_hash dedup
# ==============================================================================
#
# Cubre:
# - Migración v15 aplica correctamente (ALTER files ADD content_hash +
#   CREATE UNIQUE INDEX idx_files_content_hash parcial).
# - DB method add_file acepta content_hash.
# - DB method find_file_by_content_hash retorna el file correcto (o None).
# - UNIQUE INDEX impide insertar dos files con el mismo content_hash.
# - Multiple files con content_hash=NULL son válidos (parcial WHERE NOT NULL).
# - get_file/list_files incluyen content_hash en su SELECT.
# - Idempotencia: re-aplicar migración sobre DB ya migrada es no-op.
# - schema_version >= 15 después de initialize().
# ==============================================================================


# --- Schema migration v15 ---


@pytest.mark.asyncio
async def test_s15_adds_content_hash_column_to_files(db: Database) -> None:
    """Migración v15: ALTER TABLE files ADD COLUMN content_hash TEXT."""
    async with db.conn.execute("PRAGMA table_info(files)") as cur:
        rows = await cur.fetchall()
    cols = {r[1] for r in rows}
    assert "content_hash" in cols
    # Verificar que la columna es nullable (SQLite TEXT sin NOT NULL).
    # Buscamos el row completo (cid, name, type, notnull, dflt_value, pk).
    col_row = next(r for r in rows if r[1] == "content_hash")
    assert col_row[3] == 0, "content_hash debe ser nullable (notnull=0)"


@pytest.mark.asyncio
async def test_s15_creates_unique_content_hash_index(db: Database) -> None:
    """Migración v15: CREATE UNIQUE INDEX idx_files_content_hash (parcial)."""
    async with db.conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='index' AND tbl_name='files' AND name='idx_files_content_hash'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None, "idx_files_content_hash no existe"
    sql = row[1]
    assert "UNIQUE INDEX" in sql.upper()
    assert "content_hash" in sql
    # Crítico: el WHERE clause (parcial). Sin él, no podríamos tener
    # múltiples files con content_hash=NULL sin violar la UNIQUE.
    assert "WHERE content_hash IS NOT NULL" in sql


@pytest.mark.asyncio
async def test_s15_schema_version_at_least_15(db: Database) -> None:
    """Después de initialize(), schema_version >= 15."""
    version = await db.get_schema_version()
    assert version >= 15, f"Schema version debe ser >=15, actual={version}"


@pytest.mark.asyncio
async def test_s15_migration_idempotent(tmp_path: Path) -> None:
    """Re-aplicar SchemaMigrator sobre DB ya migrada es no-op (no falla)."""
    d1 = Database(tmp_path / "idem15.db")
    await d1.initialize()
    v1 = await d1.get_schema_version()
    await d1.close()
    d2 = Database(tmp_path / "idem15.db")
    await d2.initialize()
    v2 = await d2.get_schema_version()
    await d2.close()
    assert (
        v1 == v2 == 28
    )  # Sprint 19.5 Slice 6 Commit 4: +v25 +v26 +v27 +v28 (X3 per-policy caches; see db.py MIGRATIONS dict)


# --- DB method add_file con content_hash ---


@pytest.mark.asyncio
async def test_add_file_with_content_hash_persists(db: Database) -> None:
    """add_file acepta content_hash y lo persiste."""
    hash_val = "a" * 64  # SHA256 hex válido ficticio
    await db.add_file(
        file_id="file_h1",
        filename="hashed.pdf",
        mime_type="application/pdf",
        size_bytes=100,
        extracted_text="text",
        content_hash=hash_val,
    )
    entry = await db.get_file("file_h1")
    assert entry is not None
    assert entry["content_hash"] == hash_val


@pytest.mark.asyncio
async def test_add_file_without_content_hash_is_null(db: Database) -> None:
    """Backward compat: add_file sin content_hash deja la columna NULL."""
    await db.add_file(
        file_id="file_nohash",
        filename="legacy.pdf",
        mime_type="application/pdf",
        size_bytes=100,
        extracted_text="text",
    )
    entry = await db.get_file("file_nohash")
    assert entry is not None
    assert entry["content_hash"] is None


@pytest.mark.asyncio
async def test_get_file_includes_content_hash(db: Database) -> None:
    """get_file incluye content_hash en su resultado."""
    hash_val = "b" * 64
    await db.add_file(
        "file_g",
        "g.pdf",
        "application/pdf",
        100,
        "text",
        content_hash=hash_val,
    )
    entry = await db.get_file("file_g")
    assert entry is not None
    assert "content_hash" in entry
    assert entry["content_hash"] == hash_val


@pytest.mark.asyncio
async def test_list_files_includes_content_hash(db: Database) -> None:
    """list_files incluye content_hash en cada row."""
    await db.add_file("file_l1", "l1.pdf", "application/pdf", 100, "t", content_hash="c" * 64)
    await db.add_file("file_l2", "l2.pdf", "application/pdf", 100, "t")
    rows = await db.list_files()
    assert len(rows) == 2
    for row in rows:
        assert "content_hash" in row
    by_id = {r["id"]: r for r in rows}
    assert by_id["file_l1"]["content_hash"] == "c" * 64
    assert by_id["file_l2"]["content_hash"] is None


# --- find_file_by_content_hash ---


@pytest.mark.asyncio
async def test_find_file_by_content_hash_returns_match(db: Database) -> None:
    """find_file_by_content_hash retorna el row completo si hay match."""
    hash_val = "d" * 64
    await db.add_file(
        "file_f1",
        "f1.pdf",
        "application/pdf",
        100,
        "text",
        content_hash=hash_val,
    )
    found = await db.find_file_by_content_hash(hash_val)
    assert found is not None
    assert found["id"] == "file_f1"
    assert found["filename"] == "f1.pdf"
    assert found["content_hash"] == hash_val


@pytest.mark.asyncio
async def test_find_file_by_content_hash_returns_none_for_missing(db: Database) -> None:
    """find_file_by_content_hash retorna None si no hay match."""
    await db.add_file(
        "file_f2",
        "f2.pdf",
        "application/pdf",
        100,
        "text",
        content_hash="e" * 64,
    )
    found = await db.find_file_by_content_hash("nonexistent_hash")
    assert found is None


@pytest.mark.asyncio
async def test_find_file_by_content_hash_handles_null(db: Database) -> None:
    """find_file_by_content_hash con string vacio/None retorna None
    (no busca filas con NULL, comportamiento esperado del método)."""
    # Files sin hash (legacy S9.0 o streams binarios).
    await db.add_file("file_f3", "f3.pdf", "application/pdf", 100, "text")
    # Buscar por un hash que no matchea nada.
    found = await db.find_file_by_content_hash("f" * 64)
    assert found is None
    # El file sin hash sigue ahí.
    legacy = await db.get_file("file_f3")
    assert legacy is not None
    assert legacy["content_hash"] is None


# --- UNIQUE INDEX enforcement (dedup cross-source) ---


@pytest.mark.asyncio
async def test_unique_index_blocks_duplicate_content_hash(db: Database) -> None:
    """Insertar 2 files con mismo content_hash viola UNIQUE INDEX.

    Esto es el corazón del dedup transparente de Sprint 15. El upload
    handler (PR #67) hará find_file_by_content_hash ANTES de insertar,
    así que en práctica el conflicto no debería ocurrir — pero el INDEX
    es la red de seguridad si dos requests llegan simultáneamente.
    """
    import sqlite3

    hash_val = "g" * 64
    await db.add_file(
        "file_u1",
        "u1.pdf",
        "application/pdf",
        100,
        "text",
        content_hash=hash_val,
    )
    # Segundo INSERT con mismo hash: debe fallar con IntegrityError.
    with pytest.raises(sqlite3.IntegrityError):
        await db.add_file(
            "file_u2",
            "u2.pdf",
            "application/pdf",
            200,
            "text",
            content_hash=hash_val,
        )


@pytest.mark.asyncio
async def test_multiple_null_content_hash_allowed(db: Database) -> None:
    """UNIQUE INDEX parcial: multiples files con content_hash=NULL son válidos.

    El WHERE content_hash IS NOT NULL en el índice significa que NULLs
    no participan en la UNIQUE constraint. Esto es CRÍTICO para backward
    compat: archivos S9.0 existentes (sin hash) y archivos no-texto
    futuros (streams binarios) pueden coexistir sin violar el índice.
    """
    await db.add_file("file_n1", "n1.pdf", "application/pdf", 100, "text")
    await db.add_file("file_n2", "n2.pdf", "application/pdf", 100, "text")
    await db.add_file("file_n3", "n3.pdf", "application/pdf", 100, "text")
    # No hubo error; los 3 coexisten.
    assert await db.get_file("file_n1") is not None
    assert await db.get_file("file_n2") is not None
    assert await db.get_file("file_n3") is not None


@pytest.mark.asyncio
async def test_dedup_scenario_upload_twice_returns_existing(db: Database) -> None:
    """Simula el flujo de upload dedup (PR #67 contrato).

    1. User sube PDF → add_file con hash X → crea file_X.
    2. User sube mismo PDF → find_file_by_content_hash(X) encuentra
       file_X → retorna sin crear nuevo.
    3. Solo hay 1 row en files (no se duplica el texto extraído).
    """
    hash_val = "h" * 64
    # 1. Upload inicial
    await db.add_file(
        file_id="file_first",
        filename="duplicate.pdf",
        mime_type="application/pdf",
        size_bytes=500,
        extracted_text="x" * 1000,  # texto grande para verificar dedup real
        content_hash=hash_val,
    )
    # 2. Upload "duplicado": el handler hace find + touch, no add
    existing = await db.find_file_by_content_hash(hash_val)
    assert existing is not None
    assert existing["id"] == "file_first"
    await db.touch_file(existing["id"])  # touch para registrar uso
    # 3. Verificar que sigue habiendo solo 1 row
    rows = await db.list_files()
    assert len(rows) == 1
    assert rows[0]["id"] == "file_first"
    # Y que el touch incrementó reference_count
    entry = await db.get_file("file_first")
    assert entry is not None
    assert entry["reference_count"] == 1


@pytest.mark.asyncio
async def test_list_embedded_file_ids_returns_only_ids(
    db: Database,
) -> None:
    """Chore 2026-07-05: list_embedded_file_ids devuelve solo IDs, no blobs.

    Caso de uso: cosine_search mock quiere devolver top-k sin
    descargar los 60MB de embeddings en libraries grandes.
    Verificamos que:
    1. Solo devuelve strings (file_ids), no tuples
    2. Respeta el limit
    3. Filtra files sin embedding (legacy)
    """
    import numpy as np

    # 3 files con embedding
    for fid in ["file_a", "file_b", "file_c"]:
        await db.add_file(fid, f"{fid}.md", "text/markdown", 10, "text")
        await db.add_file_embedding(fid, np.zeros(4096, dtype=np.float32).tobytes())
    # 1 file sin embedding
    await db.add_file("file_no_emb", "x.md", "text/markdown", 5, "text")
    # default limit=100 -> 3 IDs (file_no_emb no aparece)
    fids = await db.list_embedded_file_ids()
    assert len(fids) == 3
    assert all(isinstance(f, str) for f in fids)
    assert set(fids) == {"file_a", "file_b", "file_c"}
    # limit=2 -> solo 2 IDs (mas recientes primero)
    fids_top2 = await db.list_embedded_file_ids(limit=2)
    assert len(fids_top2) == 2
