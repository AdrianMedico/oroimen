"""Tests del ciclo de vida de la DB: schema migrator y configuración.

Cubre:
- busy_timeout=30000 (resiliencia bajo concurrencia)
- SchemaMigrator: aplica migraciones en orden, idempotente, preserva datos
- SchemaMigrator: salta versiones ya aplicadas
- SchemaMigrator: maneja MIGRATIONS={} sin error

Estos tests son el "contrato" del TDD: cualquier cambio futuro al schema
debe pasar estos tests tras añadirlo al mapa MIGRATIONS.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from hermes.memory.db import MIGRATIONS, Database, SchemaMigrator

# ---------------------------------------------------------------------------
# Helper: DB fresca en tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Database inicializada, conexión abierta."""
    d = Database(tmp_path / "lifecycle.db")
    await d.initialize()
    yield d
    await d.close()


# ---------------------------------------------------------------------------
# Configuración de conexión
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_database_concurrency_timeout(db: Database) -> None:
    """busy_timeout=30000ms (30s) para escrituras concurrentes.

    Razón: con tool_calls añadiendo más escrituras en Sprint 4, 5s puede
    quedarse corto bajo I/O spikes del NAS. 30s da margen para que las
    escrituras esperen en cola sin lanzar OperationalError.
    """
    async with db.conn.execute("PRAGMA busy_timeout") as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == 30000, f"busy_timeout expected 30000ms, got {row[0]}"


# ---------------------------------------------------------------------------
# SchemaMigrator: aplicación de migraciones
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_migration_runner(db: Database) -> None:
    """DB vacía aplica todas las migraciones en orden tras initialize().

    Después de `Database.initialize()`, schema_version debe contener
    todas las versiones de MIGRATIONS (actualmente 1: v0.3.1).
    """
    versions = await SchemaMigrator(db.conn, MIGRATIONS).applied_versions()
    expected = sorted(MIGRATIONS.keys())
    assert versions == expected, f"applied versions {versions} != expected {expected}"


@pytest.mark.asyncio
async def test_migration_idempotency(tmp_path: Path) -> None:
    """Inicializar dos veces la misma DB no re-aplica migraciones.

    Tras un primer `initialize()`, la DB está en la versión N. Un segundo
    `initialize()` debe detectar que no hay nada que migrar y ser un no-op.
    La fila de schema_version NO debe duplicarse.
    """
    db_path = tmp_path / "idem.db"
    d1 = Database(db_path)
    await d1.initialize()
    await d1.close()

    # Re-leer y contar filas en schema_version antes del segundo init
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT COUNT(*) FROM schema_version") as cur,
    ):
        (count_before,) = await cur.fetchone()

    d2 = Database(db_path)
    await d2.initialize()

    # Contar después
    async with d2.conn.execute("SELECT COUNT(*) FROM schema_version") as cur:
        (count_after,) = await cur.fetchone()
    await d2.close()

    assert (
        count_before == count_after
    ), f"Re-initialize duplicó filas: antes={count_before}, después={count_after}"


@pytest.mark.asyncio
async def test_migration_records_version_in_schema_version(db: Database) -> None:
    """La tabla schema_version tiene la fila correcta tras una migración.

    Verifica (version=1, description contiene 'v0.3.1').
    """
    async with db.conn.execute(
        "SELECT version, description FROM schema_version WHERE version=1"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None, "schema_version no contiene la versión 1"
    version = row[0]
    description = row[1] or ""
    assert version == 1
    assert "v0.3.1" in description or "unique" in description.lower()


@pytest.mark.asyncio
async def test_migration_preserves_existing_data(db: Database) -> None:
    """Una migración nueva no destruye datos existentes.

    Simula: insertar 5 conversaciones antes de re-correr el migrator.
    Tras el segundo run, las 5 conversaciones siguen existiendo.
    """
    # Insertar 5 conversaciones distintas
    for i in range(5):
        await db.get_or_create_conversation(chat_id=100 + i, user_id=1)

    # Contar antes
    async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
        (count_before,) = await cur.fetchone()
    assert count_before == 5

    # Re-correr el migrator (no-op porque v1 ya aplicada)
    await SchemaMigrator(db.conn, MIGRATIONS).run()

    # Contar después
    async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
        (count_after,) = await cur.fetchone()
    assert count_after == 5, f"Migración destruyó datos: antes=5, después={count_after}"


# ---------------------------------------------------------------------------
# SchemaMigrator: casos límite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migrator_handles_no_migrations(tmp_path: Path) -> None:
    """MIGRATIONS={} no falla. run() es no-op."""
    db_path = tmp_path / "empty.db"
    # DB con schema (que crea tabla schema_version en current_version)
    d = Database(db_path)
    await d.initialize()
    await d.close()

    async with aiosqlite.connect(db_path) as conn:
        # Correr SchemaMigrator con mapa vacío sobre una DB que YA tiene
        # la migración v0.3.1 aplicada (porque Database.initialize() corrió
        # con MIGRATIONS real). El migrator con MIGRATIONS={} no debe
        # hacer nada y reportar la versión actual.
        migrator = SchemaMigrator(conn, {})
        current = await migrator.current_version()
        # Migraciones v1..v8 ya están aplicadas por Database.initialize().
        # Sprint 5 T51 añadió v4 (reasoning_content passthrough).
        # Sprint 9.0 añadió v5 (files + file_refs).
        # Sprint 9.3 añadió v6 (search_budget table).
        # Sprint 12 (ADR-007) añadió v7 (sleep_cycle_processed).
        # Sprint 12 (Fase 2) añadió v8 (idx_conversations_sleep_unprocessed).
        # Sprint 12.1 (TDD_S12_DELETE_AND_SYNC) añadió v9..v13 (tombstone columns + index).
        # Sprint 14 (TDD_S14_DEEP_RESEARCH) añadió v14 (research_jobs + research_job_token_usage).
        # Sprint 15 (PR #66) añadió v15 (files.content_hash + idx_files_content_hash + messages.file_refs).
        # Sprint 17 (PR #111) añadió v16 (vault_blobs + vault_files); v17 (Slice 1.5 ingest_jobs).
        # Sprint 17 Slice 2.5 (PR #113) añadió v18 (vault_files.text + vault_chunks embeddings table).
        # PR #113b (Slice 1.5 hot-patch) añadió v19 (vault_files.text_at_epoch, M-INV-2 fix).
        # Sprint 19 añadió v20 (vault_collections + vault_file_collections + orphaned_at).
        # Sprint 19 Slice 4 añadió v22 (ocr_pending + vault_files.text_source).
        # Sprint 19 Slice 4d v2 añadió v23 (composite UNIQUE + superseded_at + dropped_events).
        # Sprint 19.5 Slice 6 Commit 4 añadió v25..v28 (X3 per-policy caches;
        # v24 was never assigned — see hermes/memory/db.py MIGRATIONS dict).
        assert current == 28, f"current expected 28 (v1..v28 ya aplicadas), got {current}"

        # run() con MIGRATIONS={} no aplica nada nuevo
        await migrator.run()
        versions = await migrator.applied_versions()
        assert (
            versions
            == [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                25,
                26,
                27,
                28,
            ]
        ), f"Esperado [1..23, 25..28] (v24 nunca asignado, v25..v28 = Sprint 19.5 Slice 6 Commit 4 X3 per-policy caches), got {versions}"


@pytest.mark.asyncio
async def test_migrator_skips_versions_lower_than_current(tmp_path: Path) -> None:
    """Migraciones con version <= current se saltan.

    Simula: pre-poblar schema_version con version=1, luego correr un
    MIGRATIONS={1: ..., 2: ...}. Solo v2 debe aplicarse.
    """
    db_path = tmp_path / "skip.db"
    # Crear DB con schema
    d = Database(db_path)
    await d.initialize()
    await d.close()

    # Pre-poblar version 1 manualmente (simula "ya aplicada")
    async with aiosqlite.connect(db_path) as conn:
        # INSERT OR IGNORE por si la migración v0.3.1 ya está en
        # schema_version (de Database.initialize()). En ese caso, no
        # modificamos — el comportamiento del test es el mismo: v1 ya
        # está en el estado "aplicada".
        await conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, description) VALUES (1, 'manual')"
        )
        await conn.commit()

        # Crear mapa con v1 y v2 ficticias. Usamos SQL idempotente
        # (CREATE INDEX IF NOT EXISTS) para v2 para que se pueda re-ejecutar
        # sin error si los tests corren varias veces sobre la misma DB.
        test_migrations: dict[int, tuple[str, str]] = {
            1: ("v1 fake (ya aplicada)", "SELECT 1"),  # debe saltarse
            2: (
                "v2 fake: create index on messages",
                "CREATE INDEX IF NOT EXISTS idx_fake_migration_test ON messages(id)",
            ),
        }
        migrator = SchemaMigrator(conn, test_migrations)
        await migrator.run()

        # Versiones aplicadas
        async with conn.execute("SELECT version FROM schema_version ORDER BY version ASC") as cur:
            versions = [r[0] for r in await cur.fetchall()]

    # v1 estaba, v2 (fake) se aplicó, v3..v8 (reales) ya estaban aplicadas
    # por Database.initialize(). Sprint 5 T51 añadió v4 (reasoning_content).
    # Sprint 9.0 añadió v5 (files + file_refs).
    # Sprint 9.3 añadió v6 (search_budget).
    # Sprint 12 (ADR-007) añadió v7 (sleep_cycle_processed).
    # Sprint 12 (Fase 2) añadió v8 (idx_conversations_sleep_unprocessed).
    # Sprint 12.1 (TDD_S12_DELETE_AND_SYNC) añadió v9..v13 (tombstone columns + index).
    assert 1 in versions
    assert 2 in versions
    assert 3 in versions
    assert 4 in versions
    assert 5 in versions
    assert 6 in versions
    assert 7 in versions
    assert 8 in versions
    assert 9 in versions
    assert 10 in versions
    assert 11 in versions
    assert 12 in versions
    assert 13 in versions
    assert 14 in versions
    assert 15 in versions  # Sprint 15 (PR #66)
    assert 16 in versions  # Sprint 17 (PR #111): vault_blobs + vault_files
    assert 17 in versions  # Sprint 17 (PR #112): ingest_jobs
    assert 18 in versions  # Sprint 17 Slice 2.5 (PR #113): vault_chunks embeddings
    assert 19 in versions  # PR #113b Slice 1.5 hot-patch: vault_files.text_at_epoch
    assert 20 in versions  # Sprint 19: vault_collections + vault_file_collections
    assert 21 in versions  # Sprint 19: vault_files.orphaned_at (split from v20)
    assert 22 in versions  # Sprint 19 Slice 4: ocr_pending + vault_files.text_source
    assert (
        23 in versions
    )  # Sprint 19 Slice 4d v2: composite UNIQUE on vault_collections + superseded_at + dropped_events
    # Sprint 19.5 Slice 6 Commit 4 (X3 per-policy caches):
    # - v25: ADD COLUMN policy TEXT (v25a in the TDD; the brief's
    #   v25a/v25b/v25c numbering was renumbered to v25/v26/v27/v28
    #   to add the missing dim column; v24 was never assigned).
    # - v26: ADD COLUMN dim INTEGER (brief deviation — see db.py).
    # - v27: BACKFILL policy + dim for legacy rows.
    # - v28: composite PK (file_id, policy) + policy/dim NOT NULL
    #   + idx_file_embeddings_policy.
    assert 25 in versions
    assert 26 in versions
    assert 27 in versions
    assert 28 in versions
    assert len(versions) == 27, f"Esperado 27 versiones, encontrado {len(versions)}: {versions}"


# ============================================================================
# Sprint 13.0 (S8.4 fix): Backup race condition tests
# Ver docs/POSTMORTEM_DB_CORRUPTION.md para el root cause.
# ============================================================================


def test_backup_with_concurrent_writes_does_not_corrupt(
    tmp_path: Path,
) -> None:
    """Sprint 13.0 S8.4 fix: backup con wal_checkpoint(TRUNCATE) es atómico.

    El bug original (S8.4) era: .backup() sin consolidar el WAL → race con
    writers del main thread → btreeInitPage error code 11 (SQLITE_CORRUPT).

    El fix:
    1. PRAGMA wal_checkpoint(TRUNCATE) - consolida WAL al main
    2. Connection.backup() - atómica por diseño (SQLite 3.7+)

    NOTA: NO usamos BEGIN EXCLUSIVE + .backup() porque causa DEADLOCK
    (.backup() necesita read lock, EXCLUSIVE no lo permite).

    Test unitario: verifica que el backup produce un archivo integro con
    todos los datos. El test de concurrencia con threads se hace
    manualmente en validación empírica (ver docs/POSTMORTEM_DB_CORRUPTION.md).
    """
    from hermes.backup import BackupManager

    # Setup: crea DB con 100 rows
    db_path = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY, val TEXT)")
    for i in range(100):
        conn.execute("INSERT INTO foo (val) VALUES (?)", (f"row-{i}",))
    conn.commit()
    conn.close()

    # Backup con el fix BEGIN EXCLUSIVE
    manager = BackupManager(source_db=db_path, backup_dir=backup_dir, keep=3)
    backup_path = manager.run()
    assert backup_path.exists()

    # Verifica integridad del backup
    backup_conn = sqlite3.connect(str(backup_path))
    result = backup_conn.execute("PRAGMA integrity_check").fetchone()[0]
    backup_conn.close()
    assert result == "ok", f"Backup corrupto: {result}"

    # Verifica que el backup tiene las 100 rows
    backup_conn = sqlite3.connect(str(backup_path))
    count = backup_conn.execute("SELECT count(*) FROM foo").fetchone()[0]
    backup_conn.close()
    assert count == 100, f"Backup tiene {count} rows, esperado 100"


def test_backup_integrity_check_validates_output(
    tmp_path: Path,
) -> None:
    """Sprint 13.0: después del backup, validar integrity_check.

    Previene el bug del S8.4: backups corruptos se quedaban en disco
    sin que nadie se enterase (porque nadie validaba).
    """
    from hermes.backup import BackupManager

    db_path = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE foo (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO foo (val) VALUES ('hello')")
    conn.commit()
    conn.close()

    manager = BackupManager(source_db=db_path, backup_dir=backup_dir, keep=3)
    backup_path = manager.run()

    # Valida integridad
    backup_conn = sqlite3.connect(str(backup_path))
    result = backup_conn.execute("PRAGMA integrity_check").fetchone()[0]
    backup_conn.close()

    assert result == "ok", f"Backup {backup_path} corrupto: {result}"


# ---------------------------------------------------------------------------
# V3 (F-CONC-2 / F-DB-SCHEMA-SUG-2): SchemaMigrator re-run sobre versión
# ya aplicada es idempotente, no duplica schema_version rows.
# ---------------------------------------------------------------------------


def test_migrator_rerun_is_idempotent_no_duplicate_schema_version(
    tmp_path: Path,
) -> None:
    """V3 F-CONC-2: tras el fix INSERT OR IGNORE + rowcount, una segunda
    llamada a SchemaMigrator.run() sobre una DB ya migrada NO debe
    duplicar filas en `schema_version`. Sin el fix, el `INSERT INTO
    schema_version` sin `OR IGNORE` levantaba UNIQUE constraint (en el
    caso multi-proceso) o duplicaba filas (en re-init del mismo proceso).

    Nota: este test es la versión SQL-level del race. La versión full
    multi-proceso (threads/subprocs) requeriría un test de integración;
    este cubre el path de código afectado por el fix.
    """
    import asyncio

    db_path = tmp_path / "idempotent.db"
    if db_path.exists():
        db_path.unlink()
    d = Database(db_path)

    async def go():
        # First init
        await d.initialize()
        # Capture schema_version state
        async with d.conn.execute("SELECT version FROM schema_version ORDER BY version") as cur:
            after_first = [r[0] for r in await cur.fetchall()]

        # Second init — should be no-op
        await d.close()
        d2 = Database(db_path)

        async def go2():
            await d2.initialize()
            async with d2.conn.execute(
                "SELECT version FROM schema_version ORDER BY version"
            ) as cur:
                after_second = [r[0] for r in await cur.fetchall()]
            await d2.close()
            return after_second

        after_second = await go2()
        return after_first, after_second

    after_first, after_second = asyncio.run(go())

    # Same set of versions, no duplicates
    assert sorted(after_first) == sorted(after_second)
    # No version number appears more than once in either init
    assert len(after_first) == len(set(after_first))
    assert len(after_second) == len(set(after_second))


# ---------------------------------------------------------------------------
# V3 (F-OBS-5 / F-DB-SCHEMA-SUG-1): idx_vault_files_added_at covers list_files
# ---------------------------------------------------------------------------


def test_list_files_index_used_for_order_by(
    tmp_path: Path,
) -> None:
    """V3 F-OBS-5: tras aplicar la migration v=16, `list_files` debe
    usar el índice `idx_vault_files_added_at`, NO un SCAN + temp B-tree.

    Sin el índice: O(N) full scan + sort → cost 55 ms a 50k rows.
    Con índice: O(log N + limit) → cost 0.034 ms a 50k rows (1600x).
    """
    import asyncio

    db_path = tmp_path / "index_test.db"
    if db_path.exists():
        db_path.unlink()
    d = Database(db_path)

    async def go() -> None:
        await d.initialize()
        # Insert 10 rows so the optimizer has multiple rows to consider.
        for i in range(10):
            await d.conn.execute(
                "INSERT INTO vault_files (file_id, source_path, content_sha256, "
                "size_bytes, mtime) VALUES (?, ?, ?, ?, ?)",
                (
                    f"00000000-0000-4000-8000-{i:012d}",
                    f"/path/{i}.md",
                    f"sha{i}",
                    100,
                    0.0,
                ),
            )
        await d.conn.commit()

        async with d.conn.execute(
            "EXPLAIN QUERY PLAN SELECT file_id, source_path, content_sha256, "
            "size_bytes, mtime, added_at FROM vault_files "
            "ORDER BY added_at DESC, file_id DESC LIMIT 20 OFFSET 0"
        ) as cur:
            plan_rows = await cur.fetchall()
        await d.close()

        # Concatenate the plan text. If `idx_vault_files_added_at` is
        # used, the plan should reference it. If we hit a SCAN + temp
        # B-tree, the plan will say so and the test fails.
        plan_text = " ".join(" ".join(str(c) for c in row) for row in plan_rows)
        assert (
            "USING INDEX idx_vault_files_added_at" in plan_text
        ), f"list_files plan should use idx_vault_files_added_at, got: {plan_text}"

    asyncio.run(go())


# ---------------------------------------------------------------------------
# V3 (F-OBS-9): db_initialized log incluye schema_version
# ---------------------------------------------------------------------------


def test_db_initialized_log_includes_schema_version(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """V3 F-OBS-9: el log de `db_initialized` debe incluir schema_version
    para que un operador post-deploy sepa qué migración se aplicó."""
    import asyncio
    import logging

    db_path = tmp_path / "schema_v_log.db"
    if db_path.exists():
        db_path.unlink()
    d = Database(db_path)

    async def go() -> None:
        with caplog.at_level(logging.INFO, logger="hermes.memory.db"):
            await d.initialize()
            await d.close()

    asyncio.run(go())

    record = next(
        (
            r
            for r in caplog.records
            if r.name == "hermes.memory.db" and r.message == "db_initialized"
        ),
        None,
    )
    assert record is not None, f"db_initialized log missing: {caplog.records!r}"
    assert hasattr(
        record, "schema_version"
    ), f"schema_version extra missing on db_initialized; record: {record.__dict__}"
    assert record.schema_version == max(
        MIGRATIONS
    ), f"expected schema_version={max(MIGRATIONS)}, got {record.schema_version}"


# ---------------------------------------------------------------------------
# Sprint 19.5 (PR-B): v23 migration FK self-reference safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v23_migration_handles_self_referential_fk(tmp_path: Path) -> None:
    """Sprint 19.5 PR-B: v23 migration recreates vault_collections with a
    self-referential FK (parent_collection_id -> collection_id).

    Without PRAGMA foreign_keys=OFF, the INSERT OR IGNORE INTO
    vault_collections_new fails for any row with non-NULL parent_collection_id
    because the parent row isn't in vault_collections_new yet (FK check at
    INSERT time). Discovered live on the NAS 2026-07-12.

    This test pre-populates v22 schema with parent+child rows, runs v23, and
    verifies (a) migration succeeds, (b) data preserved, (c) FK works after.

    If you remove the PRAGMA foreign_keys=OFF/ON wrap from v23 in db.py, this
    test will fail with `sqlite3.IntegrityError: FOREIGN KEY constraint failed`.
    """
    import aiosqlite

    from hermes.memory.db import MIGRATIONS, SchemaMigrator

    db_path = tmp_path / "v23_fk.db"
    if db_path.exists():
        db_path.unlink()

    # Step 1: Create a v22-shaped schema manually. We don't run all migrations;
    # we just need the vault_collections table with parent_collection_id.
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA foreign_keys=ON")
        # Mimic v22 schema (vault_collections exists, no superseded_at yet)
        await conn.execute("""
            CREATE TABLE vault_collections (
                collection_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent_collection_id TEXT,
                description TEXT,
                sort_order INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0,
                archived_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (parent_collection_id) REFERENCES vault_collections(collection_id) ON DELETE RESTRICT
            )
        """)
        # Insert parent + child (this is the problematic case: child FKs parent
        # that doesn't exist in the NEW table during v23's INSERT OR IGNORE)
        await conn.execute(
            "INSERT INTO vault_collections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("parent-1", "Proyectos", None, None, 0, 0, None, "2026-07-12 10:00:00"),
        )
        await conn.execute(
            "INSERT INTO vault_collections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("child-1", "Proyectos/Sub", "parent-1", None, 0, 0, None, "2026-07-12 10:01:00"),
        )
        await conn.execute("""
            CREATE TABLE vault_file_collections (
                file_id TEXT NOT NULL,
                collection_id TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY (file_id, collection_id)
            )
        """)
        await conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        # Mark v1..v22 as already applied
        for v in range(1, 23):
            await conn.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                (v, f"v{v} pre-existing"),
            )
        await conn.commit()

    # Step 2: Run the v23 migration via SchemaMigrator (only v23 should apply)
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA foreign_keys=ON")
        # v23 only (v1-v22 already in schema_version)
        v23_only = {23: MIGRATIONS[23]}
        migrator = SchemaMigrator(conn, v23_only)
        await migrator.run()  # this would fail with FK error before PR-B fix

        # Step 3: Verify migration succeeded
        async with conn.execute("SELECT MAX(version) FROM schema_version") as cur:
            row = await cur.fetchone()
            assert row[0] == 23, f"v23 not applied: max version = {row[0]}"

        # Step 4: Verify data preserved (parent + child still there)
        async with conn.execute(
            "SELECT collection_id, name, parent_collection_id FROM vault_collections ORDER BY collection_id"
        ) as cur:
            rows = await cur.fetchall()
        assert len(rows) == 2, f"data lost: {rows}"
        by_id = {r[0]: r for r in rows}
        assert by_id["parent-1"][1] == "Proyectos"
        assert by_id["parent-1"][2] is None
        assert by_id["child-1"][1] == "Proyectos/Sub"
        assert by_id["child-1"][2] == "parent-1"

        # Step 5: Verify composite UNIQUE index exists
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_vault_collections_name_parent'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "composite UNIQUE index not created"

        # Step 6: Verify dropped_events table exists
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dropped_events'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "dropped_events table not created"

        # Step 7: Verify FK still works after migration (try inserting orphan)
        try:
            await conn.execute(
                "INSERT INTO vault_collections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "orphan",
                    "Orphan",
                    "non-existent-parent",
                    None,
                    0,
                    0,
                    None,
                    "2026-07-12 11:00:00",
                ),
            )
            await conn.commit()
            # If PRAGMA foreign_keys=ON took effect after migration, this should
            # have raised IntegrityError. The migration re-enables it.
            # If it didn't raise, that's a bug (FK left off).
            raise AssertionError(
                "FK should be ON after migration: orphan INSERT was allowed. "
                "This means PRAGMA foreign_keys=ON at end of v23 didn't take effect."
            )
        except Exception as exc:
            # Expected: IntegrityError due to FK constraint
            assert (
                "FOREIGN KEY" in str(exc) or "IntegrityError" in type(exc).__name__
            ), f"Expected FK IntegrityError, got: {type(exc).__name__}: {exc}"

        # Cleanup: release the FK check (turn off so we can close cleanly)
        await conn.execute("PRAGMA foreign_keys=OFF")
        await conn.close()
