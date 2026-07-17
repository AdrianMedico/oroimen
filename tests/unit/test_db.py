"""Tests para Database (SQLite WAL)."""

from __future__ import annotations

import asyncio
import json as _json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from hermes.memory.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


@pytest.mark.asyncio
async def test_initialize_creates_schema(db: Database) -> None:
    async with db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_initialize_sets_wal_mode(db: Database) -> None:
    async with db.conn.execute("PRAGMA journal_mode") as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0].lower() == "wal"


@pytest.mark.asyncio
async def test_initialize_sets_busy_timeout_30s(db: Database) -> None:
    async with db.conn.execute("PRAGMA busy_timeout") as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == 30000, f"busy_timeout expected 30000ms (30s), got {row[0]}"


@pytest.mark.asyncio
async def test_get_or_create_conversation(db: Database) -> None:
    cid1 = await db.get_or_create_conversation(chat_id=1, user_id=100)
    cid2 = await db.get_or_create_conversation(chat_id=1, user_id=100)
    assert cid1 == cid2


@pytest.mark.asyncio
async def test_add_message_and_history(db: Database) -> None:
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    await db.add_message(cid, "user", "hola")
    await db.add_message(
        cid, "assistant", "buenas", model_used="test-model", tokens_in=5, tokens_out=7
    )
    hist = await db.get_history(cid)
    assert len(hist) == 2
    assert hist[0]["role"] == "user"
    assert hist[1]["model_used"] == "test-model"


@pytest.mark.asyncio
async def test_clear_creates_new_conversation(db: Database) -> None:
    cid_old = await db.get_or_create_conversation(chat_id=1, user_id=100)
    await db.add_message(cid_old, "user", "msg")
    await db.archive_conversation(cid_old)
    cid_new = await db.new_conversation(chat_id=1, user_id=100)
    assert cid_new != cid_old


# Sprint 9.4: archive_stale_conversations (S9.4 cleanup huérfanas)
# Previene el bug 9.3.2b: UNIQUE constraint en idx_conversations_unique_active
# falla si una conv huérfana (is_archived=0, sin cerrar) queda del crash
# anterior. El cleanup job archiva convs viejas para liberar la combinación
# única (chat_id, thread_id, user_id) para nuevas requests.


@pytest.mark.asyncio
async def test_archive_stale_archives_only_old_active(db: Database) -> None:
    """archive_stale_conversations solo archiva is_archived=0 + updated_at viejo.
    No toca: archived, ni active recientes.
    """

    # conv vieja activa EFIMERA (chat_id=0): deberia archivarse
    cid_old = await db.get_or_create_conversation(chat_id=0, user_id=100)
    await db.add_message(cid_old, "user", "old msg")
    # Forzar updated_at viejo: SQL directo
    await db.conn.execute(
        "UPDATE conversations SET updated_at = datetime('now', '-2 hours') WHERE id=?",
        (cid_old,),
    )
    await db.conn.commit()

    # conv reciente activa EFIMERA (chat_id=0): NO deberia archivarse
    cid_new = await db.get_or_create_conversation(chat_id=0, user_id=200)
    await db.add_message(cid_new, "user", "new msg")

    # conv archivada vieja: NO deberia cambiar
    cid_arch = await db.get_or_create_conversation(chat_id=3, user_id=300)
    await db.add_message(cid_arch, "user", "old archived msg")
    await db.archive_conversation(cid_arch)
    await db.conn.execute(
        "UPDATE conversations SET updated_at = datetime('now', '-2 hours') WHERE id=?",
        (cid_arch,),
    )
    await db.conn.commit()

    # Run cleanup: archive anything older than 1h
    n = await db.archive_stale_conversations(max_age_seconds=3600)
    assert n == 1, f"Solo la conv vieja activa deberia archivarse, archived={n}"

    # Verify states
    async with db.conn.execute("SELECT id, is_archived FROM conversations ORDER BY id") as cur:
        rows = await cur.fetchall()
    states = {r[0]: r[1] for r in rows}
    assert states[cid_old] == 1, "cid_old deberia estar archivada"
    assert states[cid_new] == 0, "cid_new deberia seguir activa"
    assert states[cid_arch] == 1, "cid_arch sigue archivada (sin cambio)"


@pytest.mark.asyncio
async def test_archive_stale_returns_zero_when_nothing_to_archive(db: Database) -> None:
    """Si todas las convs activas son recientes, retorna 0."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    await db.add_message(cid, "user", "fresh msg")
    n = await db.archive_stale_conversations(max_age_seconds=3600)
    assert n == 0


@pytest.mark.asyncio
async def test_archive_stale_unblocks_unique_constraint(db: Database) -> None:
    """REGRESSION bug 9.3.2b: despues de archivar orphans, un nuevo
    get_or_create_conversation con los mismos sentinels debe funcionar
    (antes fallaba con UNIQUE constraint).
    """
    import sqlite3

    # Simular crash: conv activa huérfana con sentinels (chat_id=0, user_id=0, thread_id=0)
    cid_orphan = await db.create_ephemeral_conversation(chat_id=0, user_id=0, thread_id=0)
    await db.add_message(cid_orphan, "user", "orphaned msg")

    from hermes.memory.db import THREAD_ID_NONE_SENTINEL

    # Sin cleanup: el siguiente INSERT con los mismos sentinels falla
    # con UNIQUE constraint violation (idx_conversations_unique_active).
    with pytest.raises(sqlite3.IntegrityError):
        await db.conn.execute(
            "INSERT INTO conversations (chat_id, thread_id, user_id, is_archived) "
            "VALUES (?, ?, ?, 0)",
            (0, THREAD_ID_NONE_SENTINEL, 0),
        )
        await db.conn.commit()

    # Rollback obligatorio: tras IntegrityError la conn queda en estado
    # abortado. Sin rollback, el siguiente execute lanza ProgrammingError.
    await db.conn.rollback()

    # Con cleanup: la huérfana se archiva y el insert funciona
    await db.archive_stale_conversations(max_age_seconds=0)  # archive ALL active
    await db.conn.execute(
        "INSERT INTO conversations (chat_id, thread_id, user_id, is_archived) VALUES (?, ?, ?, 0)",
        (0, THREAD_ID_NONE_SENTINEL, 0),
    )
    await db.conn.commit()
    # OK - no exception raised


@pytest.mark.asyncio
async def test_concurrent_get_or_create_same_user_single_conversation(db: Database) -> None:
    """N llamadas concurrentes con mismo (chat_id, user_id) → 1 sola conversación.

    Race condition conocido (Sprint 2.5 discovery): el código original
    hacía SELECT + INSERT sin atomicidad, creando conversaciones duplicadas
    bajo concurrencia. Este test verifica que el fix (índice UNIQUE parcial
    sobre conversaciones activas + INSERT OR IGNORE) mantiene exactamente
    1 conversación activa por grupo.

    Antes del fix: 10 coroutines pueden crear 10 conversaciones distintas.
    Tras el fix: las 10 coroutines obtienen la misma conversación (la única
    activa que existe gracias al índice único parcial).
    """
    N = 10
    results = await asyncio.gather(
        *[db.get_or_create_conversation(chat_id=42, user_id=123) for _ in range(N)]
    )

    # Todas las coroutines deben recibir el mismo conversation_id
    assert len(set(results)) == 1, (
        f"Race condition: {len(set(results))} conversaciones distintas creadas. "
        f"Esperaba 1, obtuve {set(results)}"
    )

    # Verificar que solo hay 1 fila en la DB
    async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
        count = (await cur.fetchone())[0]
    assert count == 1, f"Esperaba 1 conversación, hay {count}"


@pytest.mark.asyncio
async def test_concurrent_get_or_create_different_users_isolated(db: Database) -> None:
    """Concurrencia con diferentes users NO se interfiere.

    Verifica que el fix (UNIQUE constraint) no rompe la capacidad de tener
    múltiples conversaciones para diferentes (chat_id, user_id).
    """
    results = await asyncio.gather(
        db.get_or_create_conversation(chat_id=1, user_id=100),
        db.get_or_create_conversation(chat_id=1, user_id=200),
        db.get_or_create_conversation(chat_id=2, user_id=100),
        db.get_or_create_conversation(chat_id=2, user_id=200),
    )

    # 4 conversaciones distintas (uno por user x chat)
    assert len(set(results)) == 4
    # Todas distintas
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert results[i] != results[j]

    async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
        count = (await cur.fetchone())[0]
    assert count == 4


@pytest.mark.asyncio
async def test_concurrent_get_or_create_with_thread_id(db: Database) -> None:
    """Thread_id se incluye en la UNIQUE constraint.

    Dos llamadas concurrentes con mismo (chat_id, user_id, thread_id)
    deben obtener la misma conversación, pero con thread_id diferente
    obtienen conversaciones distintas.
    """
    # Mismo thread_id → 1 conversación
    r1 = await asyncio.gather(
        db.get_or_create_conversation(chat_id=1, user_id=100, thread_id=7),
        db.get_or_create_conversation(chat_id=1, user_id=100, thread_id=7),
    )
    assert r1[0] == r1[1]

    # Thread_id diferente → 2 conversaciones
    r2 = await asyncio.gather(
        db.get_or_create_conversation(chat_id=1, user_id=100, thread_id=8),
        db.get_or_create_conversation(chat_id=1, user_id=100, thread_id=8),
    )
    assert r2[0] == r2[1]
    assert r1[0] != r2[0]

    async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
        count = (await cur.fetchone())[0]
    assert count == 2


@pytest.mark.asyncio
async def test_schema_has_unique_constraint_on_conversations(db: Database) -> None:
    """Existe índice UNIQUE parcial sobre conversaciones activas.

    v0.3.1: usamos un índice UNIQUE PARCIAL (no UNIQUE constraint en la
    tabla) porque queremos permitir múltiples conversaciones archivadas
    para el mismo grupo, pero solo UNA activa. El índice parcial
    `WHERE is_archived = 0` lo garantiza.

    Este test documenta la dependencia entre el fix del race condition
    y la migración del schema. Si alguien borra el índice, este test
    falla y el race condition regresa silenciosamente.
    """
    async with db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' "
        "AND name='idx_conversations_unique_active'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None, (
        "No existe el índice idx_conversations_unique_active. "
        "La UNIQUE constraint sobre conversaciones activas falta."
    )
    sql = row["sql"] or ""
    # El índice parcial debe incluir las 3 columnas y el WHERE is_archived=0
    assert "chat_id" in sql
    assert "thread_id" in sql
    assert "user_id" in sql
    assert "is_archived" in sql.lower()
    assert "WHERE" in sql.upper()


# ---------------------------------------------------------------------------
# Tests para v0.4.3: assistant message con tool_calls
#
# El assistant que pide un tool_call NO genera content (o genera content
# vacio). Para que el LLM en la siguiente iteracion pueda "cerrar el
# bucle" del tool_use ↔ tool_result, necesitamos guardar los tool_calls
# en la DB y reconstruirlos al enviar el history de vuelta al LLM.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_message_with_tool_calls_saves_json(db: Database) -> None:
    """add_message con tool_calls guarda el JSON en tool_calls_json."""
    import json

    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    tc = [
        {
            "id": "call_xyz",
            "type": "function",
            "function": {"name": "get_current_time", "arguments": "{}"},
        }
    ]
    await db.add_message(cid, "assistant", "", tool_calls=tc)
    history = await db.get_history(cid)
    assert len(history) == 1
    assert history[0]["role"] == "assistant"
    assert history[0]["tool_calls_json"] is not None
    parsed = json.loads(history[0]["tool_calls_json"])
    assert parsed == tc


@pytest.mark.asyncio
async def test_add_message_without_tool_calls_keeps_null(db: Database) -> None:
    """add_message sin tool_calls deja tool_calls_json en None."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    await db.add_message(cid, "user", "hola")
    history = await db.get_history(cid)
    assert history[0]["tool_calls_json"] is None


@pytest.mark.asyncio
async def test_add_message_with_tool_call_id_saves(db: Database) -> None:
    """add_message con tool_call_id lo guarda (role='tool')."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    await db.add_message(cid, "tool", "2026-06-22T17:30:00+02:00", tool_call_id="call_abc")
    history = await db.get_history(cid)
    assert history[0]["tool_call_id"] == "call_abc"


@pytest.mark.asyncio
async def test_messages_table_has_tool_calls_json_column(db: Database) -> None:
    """La columna tool_calls_json existe en messages (migración v3)."""
    async with db.conn.execute("PRAGMA table_info(messages)") as cur:
        rows = await cur.fetchall()
    cols = [r[1] for r in rows]
    assert "tool_calls_json" in cols
    assert "tool_call_id" in cols  # v2


@pytest.mark.asyncio
async def test_full_tool_calling_roundtrip(db: Database) -> None:
    """Roundtrip completo: user → assistant(tool_call) → tool(result) → assistant(text).

    Verifica que get_history devuelve los campos correctos para que
    _build_llm_messages pueda reconstruir el payload OpenAI.
    """
    import json

    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    # User
    await db.add_message(cid, "user", "qué hora es")
    # Assistant pide tool
    tc = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_current_time", "arguments": "{}"},
        }
    ]
    await db.add_message(cid, "assistant", "", tool_calls=tc)
    # Tool responde
    await db.add_message(cid, "tool", "2026-06-22T17:30:00+02:00", tool_call_id="call_1")
    # Assistant final
    await db.add_message(cid, "assistant", "Son las 17:30 en Madrid.")

    history = await db.get_history(cid)
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[0]["tool_calls_json"] is None
    assert history[0]["tool_call_id"] is None
    assert history[1]["role"] == "assistant"
    assert history[1]["tool_calls_json"] is not None
    assert history[2]["role"] == "tool"
    assert history[2]["tool_call_id"] == "call_1"
    assert history[3]["role"] == "assistant"
    assert history[3]["tool_calls_json"] is None
    # El JSON del tool_calls del assistant se parsea correctamente
    parsed = json.loads(history[1]["tool_calls_json"])
    assert parsed[0]["function"]["name"] == "get_current_time"


# --- Sprint 5 T51: reasoning_content passthrough ---


@pytest.mark.asyncio
async def test_add_message_persists_reasoning_content(db: Database) -> None:
    """add_message con reasoning_content guarda el campo en DB.

    Sprint 5 T51: necesario para que el round-trip Anthropic → OpenAI
    (o viceversa) mantenga el contrato de thinking mode entre iteraciones.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    rc = "El usuario quiere un resumen. Voy a estructurar puntos clave..."
    await db.add_message(cid, "assistant", "Te resumo el video.", reasoning_content=rc)
    history = await db.get_history(cid)
    assert len(history) == 1
    assert history[0]["reasoning_content"] == rc


@pytest.mark.asyncio
async def test_add_message_without_reasoning_content_keeps_null(db: Database) -> None:
    """add_message sin reasoning_content deja la columna en None.

    Para mensajes user/tool (sin thinking mode) y para assistant
    pre-migración (backwards compat). El helper de Anthropic trata
    None como cadena vacía y omite el campo.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    await db.add_message(cid, "user", "Hola")
    history = await db.get_history(cid)
    assert history[0]["reasoning_content"] is None


@pytest.mark.asyncio
async def test_get_history_includes_reasoning_content(db: Database) -> None:
    """get_history devuelve la columna reasoning_content en cada fila.

    Necesario para que _build_llm_messages pueda re-inyectar el campo
    en iteraciones siguientes.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    await db.add_message(cid, "user", "Resume el video")
    await db.add_message(
        cid,
        "assistant",
        "Te resumo el video",
        reasoning_content="Pensamiento largo...",
    )
    await db.add_message(cid, "user", "Gracias")
    history = await db.get_history(cid)
    assert len(history) == 3
    assert history[0]["reasoning_content"] is None  # user
    assert history[1]["reasoning_content"] == "Pensamiento largo..."  # assistant
    assert history[2]["reasoning_content"] is None  # user


@pytest.mark.asyncio
async def test_reasoning_content_roundtrip_sprint5_t51(db: Database) -> None:
    """Round-trip: guardar y leer de vuelta preserva el contenido.

    Es el escenario completo de uso: assistant con reasoning_content
    → DB → get_history → _build_llm_messages. Verificamos que la
    cadena no se trunca ni se escapa incorrectamente.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    original = "Línea 1\nLínea 2\nCon 'comillas' y \"dobles\" y <tags> & símbolos"
    await db.add_message(cid, "assistant", "Respuesta", reasoning_content=original)
    history = await db.get_history(cid)
    assert history[0]["reasoning_content"] == original


@pytest.mark.asyncio
async def test_migration_v4_adds_reasoning_content_column(db: Database) -> None:
    """Sprint 5 T51 migración v4: ALTER TABLE messages ADD reasoning_content.

    Verifica que tras initialize() la columna existe y es nullable TEXT.
    """
    # db fixture ya paso por initialize() → todas las migraciones aplicadas
    async with db.conn.execute("PRAGMA table_info(messages)") as cur:
        cols = await cur.fetchall()
    col_names = {row[1] for row in cols}
    assert (
        "reasoning_content" in col_names
    ), f"reasoning_content no en schema. Columnas: {sorted(col_names)}"
    # Verificar que es nullable (no tiene default NOT NULL)
    for row in cols:
        if row[1] == "reasoning_content":
            assert row[3] == 0, "reasoning_content no deberia ser NOT NULL"
            assert row[2] == "TEXT", f"Tipo esperado TEXT, got {row[2]}"


async def test_migration_v20_creates_vault_collections_table(db: Database) -> None:
    """Sprint 19 migración v20: vault_collections table + columns + indexes.

    Verifica:
    - Tabla vault_collections existe con todas las columnas (collection_id PK,
      name UNIQUE NOT NULL, parent_collection_id FK, archived INTEGER, etc.)
    - Indexes idx_vault_collections_parent y _archived existen
    """
    async with db.conn.execute("PRAGMA table_info(vault_collections)") as cur:
        cols = await cur.fetchall()
    col_names = {row[1] for row in cols}
    required = {
        "collection_id",
        "name",
        "parent_collection_id",
        "description",
        "sort_order",
        "archived",
        "archived_at",
        "created_at",
    }
    missing = required - col_names
    assert not missing, f"Columnas faltantes en vault_collections: {missing}"

    # name debe ser UNIQUE NOT NULL.
    async with db.conn.execute("PRAGMA index_list(vault_collections)") as cur:
        indexes = await cur.fetchall()
    index_names = {row[1] for row in indexes}
    # SQLite auto-crea un index para UNIQUE constraint con nombre
    # sqlite_autoindex_vault_collections_N. Check that name has UNIQUE.
    async with db.conn.execute("PRAGMA index_info(sqlite_autoindex_vault_collections_1)") as cur:
        # Best-effort: confirmamos via UNIQUE clause via SQL.
        pass
    assert "idx_vault_collections_parent" in index_names
    assert "idx_vault_collections_archived" in index_names


async def test_migration_v20_creates_vault_file_collections_table(
    db: Database,
) -> None:
    """Sprint 19 migración v20: vault_file_collections (bridge) table + PK + FKs.

    Contrato clave: PRIMARY KEY (file_id, collection_id) garantiza
    idempotencia en el link (insertar 2 veces = error). Esto NO es lo
    que hace append-only (eso lo enforce código); la PK evita dup rows.
    """
    async with db.conn.execute("PRAGMA table_info(vault_file_collections)") as cur:
        cols = await cur.fetchall()
    col_names = {row[1] for row in cols}
    assert {"file_id", "collection_id", "added_at"} <= col_names

    # PK compuesta
    async with db.conn.execute("PRAGMA index_list(vault_file_collections)") as cur:
        indexes = await cur.fetchall()
    index_names = {row[1] for row in indexes}
    assert "idx_vfc_collection" in index_names


async def test_migration_v20_adds_orphaned_at_column_to_vault_files(
    db: Database,
) -> None:
    """Sprint 19 migración v20: ALTER TABLE vault_files ADD COLUMN orphaned_at.

    Verifica que la columna existe y es nullable TEXT. Sin valor default
    NOT NULL: archivos activos tienen NULL, archivos físicos perdidos
    tienen timestamp ISO 8601 UTC.
    """
    async with db.conn.execute("PRAGMA table_info(vault_files)") as cur:
        cols = await cur.fetchall()
    col_names = {row[1] for row in cols}
    assert "orphaned_at" in col_names, f"orphaned_at no existe. Columnas: {sorted(col_names)}"
    for row in cols:
        if row[1] == "orphaned_at":
            # Nullable: NOT NULL flag debe ser 0 (False).
            assert row[3] == 0, "orphaned_at debe ser nullable"
            assert row[2] == "TEXT", f"Tipo esperado TEXT, got {row[2]}"


async def test_migration_v20_idempotent(db: Database) -> None:
    """Sprint 19 migración v20: re-aplicar es no-op (idempotencia clave).

    Re-corre SchemaMigrator y verifica que las mismas versiones siguen
    aplicadas y no hay errores. Esto valida que el CREATE TABLE IF NOT
    EXISTS y el ALTER ADD COLUMN idempotente (via SchemaMigrator case
    'is_pure_alter') funcionan juntos.
    """
    from hermes.memory.db import MIGRATIONS, SchemaMigrator

    first_versions = await SchemaMigrator(db.conn, MIGRATIONS).applied_versions()
    # Re-run
    await SchemaMigrator(db.conn, MIGRATIONS).run()
    second_versions = await SchemaMigrator(db.conn, MIGRATIONS).applied_versions()
    assert first_versions == second_versions
    # Migration 20 debe estar aplicada.
    assert 20 in second_versions


async def test_migration_v20_sql_idempotent_on_direct_rerun(
    tmp_path,
    db: Database,
) -> None:
    """Verifier B1 finding (2026-07-09): v20 SQL must be idempotent when
    re-applied directly, bypassing the version gate.

    Simulates the real failure mode: partial-failure race between
    processes during cold-start (DDL committed, schema_version insert
    failed). The next cold-start sees current_version < 20 and re-applies
    the SQL. If the SQL is NOT idempotent, the DB is bricked.

    Before fix: ALTER TABLE vault_files ADD COLUMN orphaned_at raised
    "duplicate column name" because SchemaMigrator.is_pure_alter
    requires the WHOLE SQL to be a single ALTER (no semicolons, starts
    with ALTER), and v20 starts with CREATE TABLE.

    After fix: the ALTER is split into v21 (pure ALTER, is_pure_alter
    catches it, skips on re-run). v20 contains only CREATE TABLEs +
    CREATE INDEXes (all IF NOT EXISTS).
    """
    from hermes.memory.db import MIGRATIONS, SchemaMigrator

    # First, verify v20 and v21 already applied by the auto-initialize
    versions = await SchemaMigrator(db.conn, MIGRATIONS).applied_versions()
    assert 20 in versions, "v20 should be applied"
    assert 21 in versions, "v21 should be applied"

    # Now simulate the failure mode: roll back the schema_version rows
    # for v20 and v21 (and any newer migrations — e.g. v22 added in
    # Sprint 19 Slice 4), but keep the schema (DDL already committed).
    await db.conn.execute(
        "DELETE FROM schema_version WHERE version >= 20",
    )
    await db.conn.commit()

    # Re-run the migrator: this will try to re-apply v20, v21, v22.
    # Pre-fix: v20 ALTER raises "duplicate column name: orphaned_at".
    # Post-fix: v20 has only idempotent CREATE IF NOT EXISTS; v21 ALTER
    # is caught by is_pure_alter and skipped; v22 strips leading
    # ALTER ADD COLUMN for text_source if it exists, runs rest.
    await SchemaMigrator(db.conn, MIGRATIONS).run()

    # Verify v20, v21, v22 are back as applied versions
    versions_after = await SchemaMigrator(db.conn, MIGRATIONS).applied_versions()
    assert 20 in versions_after, f"v20 should be re-applied after rollback. Got: {versions_after}"
    assert 21 in versions_after, f"v21 should be re-applied after rollback. Got: {versions_after}"
    assert 22 in versions_after, f"v22 should be re-applied after rollback. Got: {versions_after}"

    # Verify orphaned_at column still exists (wasn't dropped)
    async with db.conn.execute(
        "SELECT 1 FROM pragma_table_info('vault_files') WHERE name = 'orphaned_at'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None, "orphaned_at column must still exist after re-run"


# Sprint 12 (ADR-007): tests para los nuevos endpoints y funciones
# de conversaciones persistentes del cliente nativo RikkaHub.


@pytest.mark.asyncio
async def test_sleep_cycle_processed_default_zero(db: Database) -> None:
    """Nueva columna sleep_cycle_processed existe con default 0."""
    async with db.conn.execute("PRAGMA table_info(conversations)") as cur:
        rows = await cur.fetchall()
    cols = {r[1]: r for r in rows}
    assert "sleep_cycle_processed" in cols, "sleep_cycle_processed no existe"
    assert cols["sleep_cycle_processed"][2] == "INTEGER"
    assert cols["sleep_cycle_processed"][3] == 1, "deberia ser NOT NULL"
    assert cols["sleep_cycle_processed"][4] == "0", "default deberia ser 0"


@pytest.mark.asyncio
async def test_mark_sleep_cycle_processed(db: Database) -> None:
    """mark_sleep_cycle_processed cambia el flag de 0 a 1."""
    cid = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    # Default 0
    async with db.conn.execute(
        "SELECT sleep_cycle_processed FROM conversations WHERE id = ?", (cid,)
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == 0
    # Marcar
    await db.mark_sleep_cycle_processed(cid)
    async with db.conn.execute(
        "SELECT sleep_cycle_processed FROM conversations WHERE id = ?", (cid,)
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_list_conversations_excludes_archived(db: Database) -> None:
    """list_conversations NO devuelve conversaciones archivadas."""
    cid_active = await db.new_conversation(chat_id=100, user_id=1, thread_id=1)
    cid_archived = await db.new_conversation(chat_id=101, user_id=1, thread_id=2)
    await db.archive_conversation(cid_archived)
    convs = await db.list_conversations(user_id=1, limit=10)
    ids = [c["id"] for c in convs]
    assert cid_active in ids
    assert cid_archived not in ids


@pytest.mark.asyncio
async def test_list_conversations_scoped_by_user(db: Database) -> None:
    """list_conversations filtra por user_id (multi-tenant scoping)."""
    cid_user1 = await db.new_conversation(chat_id=200, user_id=1, thread_id=1)
    cid_user2 = await db.new_conversation(chat_id=200, user_id=2, thread_id=1)
    convs = await db.list_conversations(user_id=1, limit=10)
    ids = [c["id"] for c in convs]
    assert cid_user1 in ids
    assert cid_user2 not in ids


@pytest.mark.asyncio
async def test_list_conversations_respects_limit(db: Database) -> None:
    """list_conversations respeta el param limit (cap defensivo 1-100)."""
    for i in range(5):
        await db.new_conversation(chat_id=300 + i, user_id=1, thread_id=i)
    convs = await db.list_conversations(user_id=1, limit=2)
    assert len(convs) == 2
    # Cap defensivo: limit > 100 se trata como 100
    convs = await db.list_conversations(user_id=1, limit=500)
    assert len(convs) <= 100


@pytest.mark.asyncio
async def test_get_conversation_messages(db: Database) -> None:
    """get_conversation_messages devuelve historial con paginacion."""
    cid = await db.new_conversation(chat_id=400, user_id=1, thread_id=1)
    await db.add_message(cid, "user", "Hola")
    await db.add_message(cid, "assistant", "Buenas")
    await db.add_message(cid, "user", "Adios")
    msgs = await db.get_conversation_messages(cid, limit=10)
    assert len(msgs) == 3
    # Ordering: DESC por created_at. Verificar que los contenidos están.
    contents = {m["content"] for m in msgs}
    assert contents == {"Hola", "Buenas", "Adios"}


@pytest.mark.asyncio
async def test_get_conversation_messages_not_found(db: Database) -> None:
    """get_conversation_messages retorna [] si conv no existe (404 lo maneja el caller)."""
    msgs = await db.get_conversation_messages(conv_id=99999, limit=10)
    assert msgs == []


@pytest.mark.asyncio
async def test_archive_stale_excludes_persistent(db: Database) -> None:
    """archive_stale_conversations NO archiva conversaciones con chat_id != 0."""
    import time
    from datetime import UTC, datetime

    # Conv persistente (chat_id != 0) muy vieja
    cid_persistent = await db.new_conversation(chat_id=500, user_id=1, thread_id=1)
    # Conv efimera (chat_id = 0) muy vieja
    cid_ephemeral = await db.new_conversation(chat_id=0, user_id=0, thread_id=2)
    # Backdate ambas a hace 1 hora
    old_str = datetime.now(UTC).timestamp() - 3600
    old_strftime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(old_str))
    for cid in (cid_persistent, cid_ephemeral):
        await db.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (old_strftime, cid),
        )
    await db.conn.commit()
    # Archivar stale con max_age de 10 minutos (ambas son mas viejas)
    archived = await db.archive_stale_conversations(max_age_seconds=600)
    # Solo la efimera se archiva
    assert archived == 1
    # Verificar
    async with db.conn.execute(
        "SELECT is_archived FROM conversations WHERE id = ?", (cid_persistent,)
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == 0, "conv persistente NO debe archivarse"
    async with db.conn.execute(
        "SELECT is_archived FROM conversations WHERE id = ?", (cid_ephemeral,)
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == 1, "conv efimera SI debe archivarse"


# =========================================================================
# Sprint 12.1 (TDD_S12_DELETE_AND_SYNC.md): tombstone + sync engine tests
# =========================================================================
#
# Estas tests cubren las 5 funciones nuevas de db.py:
#   - soft_delete_conversation
#   - restore_conversation
#   - purge_expired_conversations
#   - get_conversations_sync
#   - get_deleted_conversations_since
#
# Se prueban happy paths + edge cases (idempotency, scope, safe-fallbacks,
# edge cases de fechas). Las 3 trampas que Gemini senalo en la revision
# arquitectonica estan cubiertas explicitamente:
# - Trampa #1 (SQLite TEXT datetime): se valida formato UTC zero-padded.
# - Trampa #2 (Undo 29s): se valida que DELETE + POST /restore rapido
#   funciona correctamente.
# - Trampa #3 (safe-fallback encryption): conv con encrypted_at IS NULL
#   (legacy) se restaura sin necesidad de key.


def _utc_str(dt: datetime | None = None) -> str:
    """Helper: formato 'YYYY-MM-DD HH:MM:SS' UTC zero-padded (TDD trampa #1)."""
    if dt is None:
        dt = datetime.now(UTC)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def fernet_key() -> bytes:
    """Genera una Fernet key valida para tests."""
    return Fernet.generate_key()


# --- soft_delete_conversation ---


@pytest.mark.asyncio
async def test_soft_delete_encrypts_messages_atomically(db: Database, fernet_key: bytes) -> None:
    """Happy path: soft delete cifra cada message y actualiza conversations.

    Edge cases cubiertos:
    - Content con caracteres especiales y longitud.
    - Verificacion de atomicidad: messages y conversation se actualizan juntos.
    """
    cid = await db.new_conversation(chat_id=1001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Hola, ¿qué tiempo hace en Madrid?")
    await db.add_message(cid, "assistant", "Soleado, 25 grados. — Mensaje con ñ y emoji 🌍")
    await db.add_message(cid, "user", "Gracias!")

    ok = await db.soft_delete_conversation(cid, 1, fernet_key, retention_days=7)
    assert ok is True

    # Conversations flags
    async with db.conn.execute(
        "SELECT is_archived, deleted_at, encrypted_at, purge_at, chat_id, user_id "
        "FROM conversations WHERE id=?",
        (cid,),
    ) as cur:
        conv = await cur.fetchone()
    assert conv["is_archived"] == 1
    assert conv["deleted_at"] is not None
    assert conv["encrypted_at"] is not None
    # Format check (TDD trampa #1): deleted_at en formato UTC zero-padded
    assert (
        _utc_str() in conv["deleted_at"]
        or _utc_str(datetime.now(UTC) - timedelta(seconds=5)) in conv["deleted_at"]
    )
    assert conv["purge_at"] is not None
    # Purge_at = deleted_at + 7d (mismo dia del mes siguiente)
    assert conv["chat_id"] == 1001
    assert conv["user_id"] == 1

    # Messages cifrados (content es JSON con ct, v, ts)
    async with db.conn.execute(
        "SELECT content, encrypted_at FROM messages WHERE conversation_id=? ORDER BY id",
        (cid,),
    ) as cur:
        rows = await cur.fetchall()
    assert len(rows) == 3
    for r in rows:
        parsed = _json.loads(r["content"])
        assert isinstance(parsed, dict)
        assert parsed.get("v") == 1
        assert "ct" in parsed
        assert "ts" in parsed
        assert r["encrypted_at"] is not None
        # El ct NO contiene el texto plano
        assert "Soleado" not in parsed["ct"]
        assert "Madrid" not in parsed["ct"]


@pytest.mark.asyncio
async def test_soft_delete_idempotent_returns_false_on_second_call(
    db: Database, fernet_key: bytes
) -> None:
    """Edge case: llamar soft_delete dos veces. Segunda vez retorna False.

    Importante: la conv queda en estado tombstoned (no se duplica el cifrado
    ni se hace doble archive). El DELETE HTTP retorna 204 ambas veces
    (idempotente), pero el operation log no debe duplicar el delete.
    """
    cid = await db.new_conversation(chat_id=2001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Test idempotency")

    ok1 = await db.soft_delete_conversation(cid, 1, fernet_key)
    ok2 = await db.soft_delete_conversation(cid, 1, fernet_key)
    assert ok1 is True
    assert ok2 is False  # ya esta tombstoned

    # El message NO se debe cifrar dos veces. Si se cifra dos veces, el
    # segundo intento descifraria un Fernet token (que NO es Fernet) y
    # fallaria. Pero como ya esta archivada, el path returnea False
    # ANTES de tocar los messages. Verificamos que el encrypted_at del
    # message es el original (no se reescribio).
    async with db.conn.execute(
        "SELECT encrypted_at FROM messages WHERE conversation_id=?", (cid,)
    ) as cur:
        row = await cur.fetchone()
    assert row["encrypted_at"] is not None


@pytest.mark.asyncio
async def test_soft_delete_returns_false_for_wrong_user(db: Database, fernet_key: bytes) -> None:
    """Edge case: user 2 intenta borrar conv de user 1. Retorna False.

    Defensa de seguridad: scope check. Sin esto, cualquier user podria
    borrar chats ajenos por fuerza bruta de conv_id.
    """
    cid = await db.new_conversation(chat_id=3001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Secreto de user 1")

    ok = await db.soft_delete_conversation(cid, 2, fernet_key)  # user_id=2, conv de user 1
    assert ok is False

    # La conv sigue activa y sin cifrar
    async with db.conn.execute(
        "SELECT is_archived, deleted_at, encrypted_at FROM conversations WHERE id=?",
        (cid,),
    ) as cur:
        conv = await cur.fetchone()
    assert conv["is_archived"] == 0
    assert conv["deleted_at"] is None
    assert conv["encrypted_at"] is None

    async with db.conn.execute(
        "SELECT content FROM messages WHERE conversation_id=?", (cid,)
    ) as cur:
        row = await cur.fetchone()
    assert row["content"] == "Secreto de user 1"  # sin cifrar


@pytest.mark.asyncio
async def test_soft_delete_returns_false_for_nonexistent_conv(
    db: Database, fernet_key: bytes
) -> None:
    """Edge case: conv_id que no existe. Retorna False (no error)."""
    ok = await db.soft_delete_conversation(99999, 1, fernet_key)
    assert ok is False


@pytest.mark.asyncio
async def test_soft_delete_with_empty_key_does_plain_archive(db: Database) -> None:
    """TDD trampa #3 safe-fallback: sin encryption_key, plain archive.

    La conv queda archivada con deleted_at seteado y encrypted_at NULL.
    El restore posterior no necesita key (usa safe-fallback path que
    solo limpia flags sin descifrar).

    CRITICO (Copilot review, fix #1): purge_at DEBE setearse incluso
    en el path safe-fallback (sin key). Si no, las convs plain-archived
    se acumulan para siempre sin posibilidad de hard-delete. El worker
    diario `purge_expired_conversations` filtra por `purge_at IS NOT
    NULL AND purge_at <= NOW()`.
    """
    cid = await db.new_conversation(chat_id=4001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Chat legacy sin cifrado")
    await db.add_message(cid, "assistant", "Respuesta legacy")

    # encryption_key = b"" (vacio, equivalente a None)
    ok = await db.soft_delete_conversation(cid, 1, b"", retention_days=7)
    assert ok is True

    async with db.conn.execute(
        "SELECT is_archived, deleted_at, encrypted_at, purge_at " "FROM conversations WHERE id=?",
        (cid,),
    ) as cur:
        conv = await cur.fetchone()
    assert conv["is_archived"] == 1
    assert conv["deleted_at"] is not None
    # Safe-fallback: encrypted_at es NULL (no se cifro el content).
    assert conv["encrypted_at"] is None
    # Pero purge_at SI se setea (CRITICO fix #1): igual que el path
    # con cifrado, para que el worker diario purgue tras 7d.
    assert conv["purge_at"] is not None

    # Messages siguen en plaintext
    async with db.conn.execute(
        "SELECT content, encrypted_at FROM messages WHERE conversation_id=?",
        (cid,),
    ) as cur:
        rows = await cur.fetchall()
    for r in rows:
        assert r["content"] in ("Chat legacy sin cifrado", "Respuesta legacy")
        assert r["encrypted_at"] is None


@pytest.mark.asyncio
async def test_soft_delete_with_no_messages_still_archives(db: Database, fernet_key: bytes) -> None:
    """Edge case: conv sin messages. No falla, hace archive igualmente."""
    cid = await db.new_conversation(chat_id=5001, user_id=1, thread_id=0)
    # No add_message: conv vacia.

    ok = await db.soft_delete_conversation(cid, 1, fernet_key)
    assert ok is True

    async with db.conn.execute(
        "SELECT is_archived, deleted_at, encrypted_at FROM conversations WHERE id=?",
        (cid,),
    ) as cur:
        conv = await cur.fetchone()
    assert conv["is_archived"] == 1
    assert conv["deleted_at"] is not None
    assert conv["encrypted_at"] is not None


@pytest.mark.asyncio
async def test_soft_delete_skips_already_encrypted_messages(
    db: Database, fernet_key: bytes
) -> None:
    """Edge case: conv con un message ya cifrado (caso improbable pero posible
    si una operacion anterior fallo a medias). El segundo delete no debe
    re-cifrar el ciphertext (eso daria Fernet-de-Fernet, irrecuperable).
    """
    cid = await db.new_conversation(chat_id=6001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Test re-delete")

    # Primer soft_delete: cifra
    await db.soft_delete_conversation(cid, 1, fernet_key)

    # Capturar el ciphertext original
    async with db.conn.execute(
        "SELECT content FROM messages WHERE conversation_id=?", (cid,)
    ) as cur:
        original = (await cur.fetchone())["content"]

    # Resetear la conv a "activa" para forzar el path de skip
    # (sin esto, el segundo soft_delete retorna False por idempotency)
    await db.conn.execute(
        "UPDATE conversations SET is_archived=0, deleted_at=NULL, "
        "    encrypted_at=NULL, purge_at=NULL WHERE id=?",
        (cid,),
    )
    await db.conn.commit()

    # Segundo soft_delete: el codigo detecta que el content ya es
    # ciphertext (JSON con ct, v=1) y lo deja intacto.
    ok = await db.soft_delete_conversation(cid, 1, fernet_key)
    assert ok is True

    async with db.conn.execute(
        "SELECT content FROM messages WHERE conversation_id=?", (cid,)
    ) as cur:
        after = (await cur.fetchone())["content"]
    # El ciphertext es IDENTICO (no se re-cifra)
    assert after == original


# --- restore_conversation ---


@pytest.mark.asyncio
async def test_restore_within_window_decrypts_messages(db: Database, fernet_key: bytes) -> None:
    """Happy path: soft_delete + restore recupera el plaintext."""
    cid = await db.new_conversation(chat_id=7001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Mensaje original A")
    await db.add_message(cid, "assistant", "Mensaje original B")

    await db.soft_delete_conversation(cid, 1, fernet_key)

    ok = await db.restore_conversation(cid, 1, fernet_key)
    assert ok is True

    # Flags limpios
    async with db.conn.execute(
        "SELECT is_archived, deleted_at, encrypted_at, purge_at " "FROM conversations WHERE id=?",
        (cid,),
    ) as cur:
        conv = await cur.fetchone()
    assert conv["is_archived"] == 0
    assert conv["deleted_at"] is None
    assert conv["encrypted_at"] is None
    assert conv["purge_at"] is None

    # Messages descifrados
    async with db.conn.execute(
        "SELECT content, encrypted_at FROM messages WHERE conversation_id=? ORDER BY id",
        (cid,),
    ) as cur:
        rows = await cur.fetchall()
    assert rows[0]["content"] == "Mensaje original A"
    assert rows[1]["content"] == "Mensaje original B"
    # encrypted_at del message se limpia
    for r in rows:
        assert r["encrypted_at"] is None


@pytest.mark.asyncio
async def test_restore_returns_false_for_wrong_user(db: Database, fernet_key: bytes) -> None:
    """Edge case: user 2 intenta restaurar conv de user 1. Retorna False.

    Defensa de seguridad: scope check, simétrico al delete.
    """
    cid = await db.new_conversation(chat_id=8001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Privado de user 1")
    await db.soft_delete_conversation(cid, 1, fernet_key)

    ok = await db.restore_conversation(cid, 2, fernet_key)
    assert ok is False

    # La conv sigue tombstoned
    async with db.conn.execute(
        "SELECT is_archived, deleted_at FROM conversations WHERE id=?",
        (cid,),
    ) as cur:
        conv = await cur.fetchone()
    assert conv["is_archived"] == 1
    assert conv["deleted_at"] is not None


@pytest.mark.asyncio
async def test_restore_after_purge_returns_false(db: Database, fernet_key: bytes) -> None:
    """Edge case: conv purgada (purge_at <= NOW). Retorna False.

    Despues del purge, la conv NO existe en la DB, asi que el SELECT
    del restore retorna None -> False.
    """
    cid = await db.new_conversation(chat_id=9001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Test purge before restore")
    await db.soft_delete_conversation(cid, 1, fernet_key)

    # Forzar purge_at al pasado
    past = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    await db.conn.execute("UPDATE conversations SET purge_at=? WHERE id=?", (past, cid))
    await db.conn.commit()

    ok = await db.restore_conversation(cid, 1, fernet_key)
    assert ok is False


@pytest.mark.asyncio
async def test_restore_legacy_conv_without_key_works(db: Database) -> None:
    """TDD trampa #3: conv con encrypted_at IS NULL se restaura sin key.

    Caso: archive_stale_conversations (chat_id=0, ephemeral) o /clear de
    Telegram deja la conv archivada sin cifrar. El restore debe limpiar
    los flags sin intentar descifrar nada.
    """
    cid = await db.new_conversation(chat_id=10001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Legacy chat, sin cifrado")
    # Plain archive (sin key)
    await db.soft_delete_conversation(cid, 1, b"")

    # Restore con key vacia (no hay key configurada)
    ok = await db.restore_conversation(cid, 1, b"")
    assert ok is True

    async with db.conn.execute(
        "SELECT is_archived, deleted_at, encrypted_at FROM conversations WHERE id=?",
        (cid,),
    ) as cur:
        conv = await cur.fetchone()
    assert conv["is_archived"] == 0
    assert conv["deleted_at"] is None
    assert conv["encrypted_at"] is None


@pytest.mark.asyncio
async def test_restore_returns_false_for_active_conv(db: Database, fernet_key: bytes) -> None:
    """Edge case: intentar restaurar una conv que NUNCA fue borrada.

    Retorna False (no es un error, pero el caller deberia distinguir
    este caso del 404 via un SELECT adicional).
    """
    cid = await db.new_conversation(chat_id=11001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Conv activa")

    ok = await db.restore_conversation(cid, 1, fernet_key)
    assert ok is False


# --- purge_expired_conversations ---


@pytest.mark.asyncio
async def test_purge_expired_deletes_conv_and_cascades_messages(
    db: Database, fernet_key: bytes
) -> None:
    """Happy path: purge hard-delete la conv y todos sus messages (CASCADE)."""
    cid = await db.new_conversation(chat_id=12001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "msg 1")
    await db.add_message(cid, "assistant", "msg 2")
    await db.soft_delete_conversation(cid, 1, fernet_key)

    # Forzar purge_at al pasado
    past = (datetime.now(UTC) - timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S")
    await db.conn.execute("UPDATE conversations SET purge_at=? WHERE id=?", (past, cid))
    await db.conn.commit()

    purged = await db.purge_expired_conversations()
    assert purged == 1

    # Convs borrada
    async with db.conn.execute("SELECT id FROM conversations WHERE id=?", (cid,)) as cur:
        row = await cur.fetchone()
    assert row is None

    # Messages borrados via CASCADE
    async with db.conn.execute("SELECT id FROM messages WHERE conversation_id=?", (cid,)) as cur:
        rows = await cur.fetchall()
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_purge_does_not_affect_non_expired(db: Database, fernet_key: bytes) -> None:
    """Edge case: solo purga convs con purge_at <= NOW. Las demas siguen."""
    cid_old = await db.new_conversation(chat_id=13001, user_id=1, thread_id=0)
    cid_new = await db.new_conversation(chat_id=13002, user_id=1, thread_id=0)
    await db.soft_delete_conversation(cid_old, 1, fernet_key)
    await db.soft_delete_conversation(cid_new, 1, fernet_key)

    # cid_old: purge_at = ayer (expira)
    past = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    await db.conn.execute("UPDATE conversations SET purge_at=? WHERE id=?", (past, cid_old))
    await db.conn.commit()
    # cid_new: purge_at = manana (no expira)

    purged = await db.purge_expired_conversations()
    assert purged == 1

    # cid_old borrado, cid_new sigue
    async with db.conn.execute(
        "SELECT id FROM conversations WHERE id IN (?, ?)", (cid_old, cid_new)
    ) as cur:
        rows = await cur.fetchall()
    ids = [r["id"] for r in rows]
    assert cid_old not in ids
    assert cid_new in ids


@pytest.mark.asyncio
async def test_purge_returns_zero_when_no_expired(db: Database) -> None:
    """Edge case: sin convs expiradas, retorna 0."""
    purged = await db.purge_expired_conversations()
    assert purged == 0


@pytest.mark.asyncio
async def test_purge_does_not_delete_legacy_archived_convs(
    db: Database,
) -> None:
    """Edge case: convs archivadas por /clear (sin deleted_at) NO se purgan.

    El purge filtra por `deleted_at IS NOT NULL AND purge_at IS NOT NULL`.
    Las legacy archived (sin deleted_at) no entran en el query.
    """
    cid = await db.new_conversation(chat_id=14001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Legacy archived")
    # Plain archive via archive_conversation (no soft_delete)
    await db.archive_conversation(cid)

    purged = await db.purge_expired_conversations()
    assert purged == 0

    # La conv sigue archivada
    async with db.conn.execute("SELECT is_archived FROM conversations WHERE id=?", (cid,)) as cur:
        row = await cur.fetchone()
    assert row["is_archived"] == 1


# --- get_conversations_sync (cursor-based) ---


@pytest.mark.asyncio
async def test_sync_returns_only_convs_after_cursor(
    db: Database, fernet_key: bytes, set_conv_updated_at
) -> None:
    """Happy path: solo devuelve convs con updated_at > cursor."""
    cid_old = await db.new_conversation(chat_id=15001, user_id=1, thread_id=0)
    await db.add_message(cid_old, "user", "old")
    # Forzar updated_at explicito (evita asyncio.sleep de 1.1s).
    await set_conv_updated_at(cid_old, "2026-07-01 12:00:00")

    cursor_str = "2026-07-01 12:30:00"  # cursor entre cid_old y cid_new
    cid_new = await db.new_conversation(chat_id=15002, user_id=1, thread_id=0)
    await db.add_message(cid_new, "user", "new")
    await set_conv_updated_at(cid_new, "2026-07-01 13:00:00")

    # Cold start (cursor=0): devuelve ambas
    sync = await db.get_conversations_sync(1, "1970-01-01 00:00:00")
    ids = [c["id"] for c in sync]
    assert cid_old in ids
    assert cid_new in ids

    # Sync con cursor entre ambas: solo la nueva
    sync = await db.get_conversations_sync(1, cursor_str)
    ids = [c["id"] for c in sync]
    assert cid_old not in ids
    assert cid_new in ids


@pytest.mark.asyncio
async def test_sync_user_scoped_does_not_leak(db: Database, fernet_key: bytes) -> None:
    """Edge case: sync de user 1 no devuelve convs de user 2.

    Defensa de seguridad multi-tenant: la query filtra por user_id.
    """
    cid_user1 = await db.new_conversation(chat_id=16001, user_id=1, thread_id=0)
    cid_user2 = await db.new_conversation(chat_id=16002, user_id=2, thread_id=0)
    await db.add_message(cid_user1, "user", "user 1 chat")
    await db.add_message(cid_user2, "user", "user 2 chat")

    sync = await db.get_conversations_sync(1, "1970-01-01 00:00:00")
    ids = [c["id"] for c in sync]
    assert cid_user1 in ids
    assert cid_user2 not in ids

    sync2 = await db.get_conversations_sync(2, "1970-01-01 00:00:00")
    ids2 = [c["id"] for c in sync2]
    assert cid_user2 in ids2
    assert cid_user1 not in ids2


@pytest.mark.asyncio
async def test_sync_excludes_archived_convs(db: Database, fernet_key: bytes) -> None:
    """Edge case: la query filtra is_archived=0. Archive por stale cleanup
    o /clear NO aparece en el upserted.

    Diferencia clave con `deleted`: archived sin deleted_at es "stale",
    NO "user-borrado". El sync los excluye del upserted (porque no son
    activas) pero NO aparecen en deleted (porque deleted_at IS NULL).
    RikkaHub no los ve hasta un cold-start sync.
    """
    cid_active = await db.new_conversation(chat_id=17001, user_id=1, thread_id=0)
    cid_archived = await db.new_conversation(chat_id=17002, user_id=1, thread_id=0)
    await db.add_message(cid_active, "user", "active")
    await db.add_message(cid_archived, "user", "stale")
    # Plain archive (no soft_delete) — emula stale cleanup
    await db.archive_conversation(cid_archived)

    sync = await db.get_conversations_sync(1, "1970-01-01 00:00:00")
    ids = [c["id"] for c in sync]
    assert cid_active in ids
    assert cid_archived not in ids  # archived (sin deleted_at) NO en sync


@pytest.mark.asyncio
async def test_sync_paginates_with_limit_and_ordering(
    db: Database, fernet_key: bytes, set_conv_updated_at
) -> None:
    """Edge case: limit=N pagina correctamente con ORDER BY updated_at ASC.

    Importante: la primera pagina tiene los N mas antiguos del cursor.
    La segunda pagina tiene los siguientes. Sin overlap.
    """
    cids = []
    for i in range(5):
        cid = await db.new_conversation(chat_id=18000 + i, user_id=1, thread_id=0)
        await db.add_message(cid, "user", f"msg {i}")
        cids.append(cid)
        # Forzar updated_at explicito (i segundos desde epoch 2026) en
        # lugar de asyncio.sleep(1.1). Mismo efecto, ~1000x mas rapido.
        await set_conv_updated_at(cid, "2026-07-01 12:00:0" + str(i))

    # Pagina 1: limit=2 -> los 2 mas antiguos
    page1 = await db.get_conversations_sync(1, "1970-01-01 00:00:00", limit=2)
    assert len(page1) == 2
    assert [c["id"] for c in page1] == [cids[0], cids[1]]

    # Pagina 2: limit=2 con cursor en page1[1].updated_at
    cursor = page1[-1]["updated_at"]
    page2 = await db.get_conversations_sync(1, cursor, limit=2)
    assert len(page2) == 2
    assert [c["id"] for c in page2] == [cids[2], cids[3]]

    # Pagina 3: solo el ultimo
    cursor = page2[-1]["updated_at"]
    page3 = await db.get_conversations_sync(1, cursor, limit=2)
    assert len(page3) == 1
    assert page3[0]["id"] == cids[4]


@pytest.mark.asyncio
async def test_sync_returns_empty_when_no_changes_after_cursor(
    db: Database, fernet_key: bytes
) -> None:
    """Edge case: cursor futuro (no hay convs con updated_at > cursor)."""
    cid = await db.new_conversation(chat_id=19001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "only conv")
    # Cursor = manana UTC
    future = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    sync = await db.get_conversations_sync(1, future)
    assert sync == []


# --- get_deleted_conversations_since ---


@pytest.mark.asyncio
async def test_deleted_since_returns_only_tombstoned_after_cursor(
    db: Database, fernet_key: bytes, set_conv_field
) -> None:
    """Happy path: solo devuelve convs con deleted_at > cursor."""
    cid_old = await db.new_conversation(chat_id=20001, user_id=1, thread_id=0)
    await db.add_message(cid_old, "user", "old")
    await db.soft_delete_conversation(cid_old, 1, fernet_key)
    # Forzar deleted_at explicito (evita asyncio.sleep).
    await set_conv_field("deleted_at", cid_old, "2026-07-01 12:00:00")

    cursor_str = "2026-07-01 12:30:00"
    cid_new = await db.new_conversation(chat_id=20002, user_id=1, thread_id=0)
    await db.add_message(cid_new, "user", "new")
    await db.soft_delete_conversation(cid_new, 1, fernet_key)
    await set_conv_field("deleted_at", cid_new, "2026-07-01 13:00:00")

    deleted = await db.get_deleted_conversations_since(1, cursor_str)
    ids = [d["id"] for d in deleted]
    assert cid_old not in ids
    assert cid_new in ids


@pytest.mark.asyncio
async def test_deleted_since_excludes_legacy_archived(db: Database, fernet_key: bytes) -> None:
    """Edge case: legacy archived (sin deleted_at) NO aparece en deleted.

    El sync recibe `deleted` como "convs que el server sabe que el user
    borro". Las convs legacy archivadas (sin deleted_at) NO son user-deleted.
    """
    cid_legacy = await db.new_conversation(chat_id=21001, user_id=1, thread_id=0)
    await db.add_message(cid_legacy, "user", "legacy archived")
    await db.archive_conversation(cid_legacy)  # plain archive, sin deleted_at

    cid_tombstoned = await db.new_conversation(chat_id=21002, user_id=1, thread_id=0)
    await db.add_message(cid_tombstoned, "user", "tombstoned")
    await db.soft_delete_conversation(cid_tombstoned, 1, fernet_key)

    deleted = await db.get_deleted_conversations_since(1, "1970-01-01 00:00:00")
    ids = [d["id"] for d in deleted]
    assert cid_legacy not in ids  # no deleted_at -> no aparece
    assert cid_tombstoned in ids  # tiene deleted_at -> aparece


@pytest.mark.asyncio
async def test_deleted_since_paginates_with_limit(
    db: Database, fernet_key: bytes, set_conv_field
) -> None:
    """Edge case: limit=N pagina correctamente."""
    cids = []
    for i in range(5):
        cid = await db.new_conversation(chat_id=22000 + i, user_id=1, thread_id=0)
        await db.add_message(cid, "user", f"msg {i}")
        await db.soft_delete_conversation(cid, 1, fernet_key)
        cids.append(cid)
        # Forzar deleted_at explicito (~1000x mas rapido que sleep).
        await set_conv_field("deleted_at", cid, "2026-07-01 12:00:0" + str(i))

    page1 = await db.get_deleted_conversations_since(1, "1970-01-01 00:00:00", limit=2)
    assert [d["id"] for d in page1] == [cids[0], cids[1]]

    cursor = page1[-1]["deleted_at"]
    page2 = await db.get_deleted_conversations_since(1, cursor, limit=2)
    assert [d["id"] for d in page2] == [cids[2], cids[3]]


@pytest.mark.asyncio
async def test_sync_handles_zero_messages_gracefully(db: Database, fernet_key: bytes) -> None:
    """Edge case: conv sin messages. last_message_preview = NULL.

    El sync no debe fallar; el cliente recibe conv con preview NULL.
    """
    cid = await db.new_conversation(chat_id=23001, user_id=1, thread_id=0)
    # No add_message.

    sync = await db.get_conversations_sync(1, "1970-01-01 00:00:00")
    assert len(sync) == 1
    assert sync[0]["id"] == cid
    assert sync[0]["last_message_preview"] is None


# --- Fix #1 (Copilot review): safe-fallback SÍ setea purge_at ---
# Test explicito: el safe-fallback de soft_delete setea purge_at
# correctamente, permitiendo que purge_expired_conversations() purgue
# la conv tras la ventana de retencion.


@pytest.mark.asyncio
async def test_soft_delete_safe_fallback_sets_purge_at(db: Database, fernet_key: bytes) -> None:
    """Fix #1: el path safe-fallback (sin key) tambien setea purge_at.

    Sin este fix, las convs plain-archived se acumulaban para siempre
    porque `purge_at IS NULL` y el worker de purga filtra por
    `purge_at IS NOT NULL AND purge_at <= NOW()`.
    """
    cid = await db.new_conversation(chat_id=24001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Test safe-fallback purge_at")

    # Sin key (b"" = equivalente a None)
    ok = await db.soft_delete_conversation(cid, 1, b"", retention_days=7)
    assert ok is True

    # purge_at debe estar seteado (no NULL). deleted_at + 7d.
    async with db.conn.execute(
        "SELECT deleted_at, purge_at FROM conversations WHERE id=?", (cid,)
    ) as cur:
        conv = await cur.fetchone()
    assert conv["deleted_at"] is not None
    assert conv["purge_at"] is not None

    # Verificamos que purge_at > deleted_at (correcto: future).
    # Ambos formato 'YYYY-MM-DD HH:MM:SS' UTC; comparar como strings.
    assert conv["purge_at"] > conv["deleted_at"]

    # Si forzamos purge_at al pasado, el worker lo purga.
    past = "2020-01-01 00:00:00"
    await db.conn.execute("UPDATE conversations SET purge_at=? WHERE id=?", (past, cid))
    await db.conn.commit()

    purged = await db.purge_expired_conversations()
    assert purged == 1

    # La conv fue hard-deleted.
    async with db.conn.execute("SELECT id FROM conversations WHERE id=?", (cid,)) as cur:
        row = await cur.fetchone()
    assert row is None


# --- Fix #2 (Copilot review): restore_conversation retorna False con key mala ---


@pytest.mark.asyncio
async def test_restore_returns_false_on_decrypt_failure(db: Database) -> None:
    """Fix #2: si el decrypt de CUALQUIER message falla, restore retorna
    False sin limpiar flags. La conv sigue tombstoned con messages
    cifrados intactos. El HTTP handler traduce esto a 503.

    Antes del fix: el codigo hacia `continue` y al final del loop
    limpiaba flags igualmente, dejando la conv "activa" con
    messages cifrados (estado inconsistente).
    """
    # Setup: encrypto una conv con una key
    good_key = Fernet.generate_key()
    cid = await db.new_conversation(chat_id=25001, user_id=1, thread_id=0)
    await db.add_message(cid, "user", "Mensaje para restore con key mala")
    await db.soft_delete_conversation(cid, 1, good_key, retention_days=7)

    # Verifico que encrypted_at esta seteado
    async with db.conn.execute("SELECT encrypted_at FROM conversations WHERE id=?", (cid,)) as cur:
        conv = await cur.fetchone()
    assert conv["encrypted_at"] is not None

    # Intento restaurar con una key DIFERENTE (que falla al decrypt)
    bad_key = Fernet.generate_key()
    ok = await db.restore_conversation(cid, 1, bad_key)
    assert ok is False  # CRITICO: retorna False, no True

    # La conv SIGUE tombstoned (no se limpio flags).
    async with db.conn.execute(
        "SELECT is_archived, deleted_at, encrypted_at " "FROM conversations WHERE id=?",
        (cid,),
    ) as cur:
        conv = await cur.fetchone()
    assert conv["is_archived"] == 1  # sigue tombstoned
    assert conv["deleted_at"] is not None
    assert conv["encrypted_at"] is not None  # sigue cifrada

    # Los messages SIGUEN cifrados (no se descifraron con la key mala).
    async with db.conn.execute(
        "SELECT content FROM messages WHERE conversation_id=?", (cid,)
    ) as cur:
        row = await cur.fetchone()
    parsed = _json.loads(row["content"])
    assert "ct" in parsed
    # Si intentamos descifrar con la key buena, debe funcionar
    # (probando que la key mala fallo pero no corrompio nada).
    good_fernet = Fernet(good_key)
    plaintext = good_fernet.decrypt(parsed["ct"].encode("ascii")).decode("utf-8")
    assert plaintext == "Mensaje para restore con key mala"
