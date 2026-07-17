"""Sprint 19.5 Slice 6 Commit 4: tests for v25..v28 migration.

Covers (TDD §3 Commit 4, ~4 tests):
- v25 adds the `policy` column (Caso 1 ALTER ADD COLUMN).
- v27 backfills legacy rows (UPDATE WHERE policy IS NULL / dim IS NULL).
- v28 creates the composite PK + policy/dim NOT NULL + index.
- Re-running the migrations is a no-op (Caso 3 detects already-applied
  state for v28; v25/v26 use Caso 1; v27 is naturally idempotent).

Brief deviation: the brief listed 3 entries (v25a/v25b/v25c), but
v25b's UPDATE dim requires a `dim` column that v25a doesn't add. The
TDD itself (TDD_SPRINT_19_5_SLICE_6.md §2.7 #4) says "ADD COLUMN
`policy` (nullable), ADD COLUMN `dim` (nullable)" — both columns are
required. We added v26 for the missing dim column. The reproduction
of this brief bug is in db.py MIGRATIONS dict.
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


# --- Test 1: v25a adds policy column ------------------------------------


@pytest.mark.asyncio
async def test_v25a_adds_policy_column(db: Database) -> None:
    """v25 (v25a in the brief) adds the `policy` column to
    file_embeddings. Idempotent via Caso 1 PRAGMA table_info check.
    """
    # After initialize(), all migrations have run, so policy exists.
    async with db.conn.execute("PRAGMA table_info(file_embeddings)") as cur:
        cols = await cur.fetchall()
    col_names = {row[1] for row in cols}
    assert "policy" in col_names
    # policy is TEXT (nullable at this stage before v27 backfill)
    policy_col = next(row for row in cols if row[1] == "policy")
    assert policy_col[2].upper() == "TEXT"


# --- Test 2: v27 backfills legacy rows ----------------------------------


@pytest.mark.asyncio
async def test_v27_backfills_legacy_rows(tmp_path: Path) -> None:
    """v27 (v25b in the brief) backfills policy + dim for legacy rows.

    Strategy: build a custom MIGRATIONS dict that EXCLUDES v27 and
    v28 (the NOT NULL enforcement and recreation), keeping only
    v25 (ADD COLUMN policy) and v26 (ADD COLUMN dim). This gives
    us a schema where policy/dim columns exist but are nullable
    — the exact pre-backfill state. Then insert a legacy row
    (no policy, no dim) and run v27's UPDATE SQL against it.
    v27 is naturally idempotent (WHERE policy IS NULL / dim IS NULL).
    """
    import aiosqlite

    # Filter MIGRATIONS to exclude v27 and v28 — we want v25+v26 only.
    partial_migrations = {k: v for k, v in MIGRATIONS.items() if k not in (27, 28)}
    db_path = tmp_path / "backfill.db"
    conn = await aiosqlite.connect(db_path, timeout=30.0)
    conn.row_factory = aiosqlite.Row
    # Apply SCHEMA + partial MIGRATIONS (v1..v23 + v25 + v26).
    from hermes.memory.db import SCHEMA

    await conn.executescript(SCHEMA)
    await conn.commit()
    migrator = SchemaMigrator(conn, partial_migrations)
    await migrator.run()
    # Verify v25 + v26 applied (policy + dim columns exist, nullable)
    async with conn.execute("PRAGMA table_info(file_embeddings)") as cur:
        cols = await cur.fetchall()
    col_map = {row[1]: row for row in cols}
    assert "policy" in col_map
    assert "dim" in col_map
    # policy and dim should be NULLABLE at this point (v28 hasn't run)
    assert int(col_map["policy"][3]) == 0
    assert int(col_map["dim"][3]) == 0
    # Add a file + legacy row (no policy, no dim — simulating pre-v27).
    await conn.execute(
        "INSERT INTO files (id, filename, mime_type, size_bytes, extracted_text, "
        "extraction_method) VALUES (?, ?, ?, ?, ?, ?)",
        ("file_legacy_1", "l.pdf", "application/pdf", 100, "text", "pypdf"),
    )
    await conn.execute(
        "INSERT INTO file_embeddings (file_id, embedding) VALUES (?, ?)",
        ("file_legacy_1", b"\x00" * (384 * 4)),
    )
    await conn.commit()
    # Verify pre-state: policy and dim are NULL
    async with conn.execute(
        "SELECT policy, dim FROM file_embeddings WHERE file_id=?", ("file_legacy_1",)
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] is None
    assert row[1] is None
    # Run v27's UPDATEs (the same SQL as in MIGRATIONS dict entry 27)
    await conn.execute("UPDATE file_embeddings SET policy='vault_ingest' WHERE policy IS NULL")
    await conn.execute(
        "UPDATE file_embeddings SET dim = CAST(length(embedding) / 4 AS INTEGER) WHERE dim IS NULL"
    )
    await conn.commit()
    # Verify post-state
    async with conn.execute(
        "SELECT policy, dim, length(embedding) FROM file_embeddings WHERE file_id=?",
        ("file_legacy_1",),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    policy, dim, blob_len = row
    assert policy == "vault_ingest"
    # dim = length(embedding) / 4 (float32 = 4 bytes/elem).
    # We inserted 384 dims x 4 bytes = 1536 bytes; dim should be 384.
    assert int(dim) == 384
    assert int(blob_len) == 1536
    await conn.close()


# --- Test 3: v28 creates composite PK + index ---------------------------


@pytest.mark.asyncio
async def test_v28_creates_composite_pk_and_index(db: Database) -> None:
    """v28 (v25c in the brief) recreates file_embeddings with:
    - composite PRIMARY KEY (file_id, policy)
    - policy NOT NULL
    - dim NOT NULL
    - idx_file_embeddings_policy

    After initialize(), all of these are in place.
    """
    async with db.conn.execute("PRAGMA table_info(file_embeddings)") as cur:
        cols = await cur.fetchall()
    # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
    col_map = {row[1]: row for row in cols}
    # Composite PK: both file_id and policy have pk > 0
    assert int(col_map["file_id"][5]) > 0
    assert int(col_map["policy"][5]) > 0
    # policy and dim NOT NULL
    assert int(col_map["policy"][3]) == 1
    assert int(col_map["dim"][3]) == 1
    # Index on policy
    async with db.conn.execute("PRAGMA index_list(file_embeddings)") as cur:
        idxs = await cur.fetchall()
    idx_names = {row[1] for row in idxs}
    assert "idx_file_embeddings_policy" in idx_names


# --- Test 4: full migration is idempotent --------------------------------


@pytest.mark.asyncio
async def test_v25_is_idempotent(tmp_path: Path) -> None:
    """Re-running SchemaMigrator after a successful init is a no-op
    (no schema_version increment, no DDL applied). Caso 3 short-circuits
    v28 when all four constraints are already in place; v25/v26 use
    Caso 1 (PRAGMA table_info check); v27 is naturally idempotent.
    """
    # First init: applies v25, v26, v27, v28
    d1 = Database(tmp_path / "idem.db")
    await d1.initialize()
    v1 = await d1.get_schema_version()
    applied1 = await SchemaMigrator(d1.conn, MIGRATIONS).applied_versions()
    await d1.close()
    # Second init: should be a no-op (no new DDL, no version change)
    d2 = Database(tmp_path / "idem.db")
    await d2.initialize()
    v2 = await d2.get_schema_version()
    applied2 = await SchemaMigrator(d2.conn, MIGRATIONS).applied_versions()
    # Both DBs should have the same schema_version and the same set
    # of applied migrations.
    assert v1 == v2
    assert applied1 == applied2
    # Spot-check: the four v25..v28 entries should be in applied1
    for v in (25, 26, 27, 28):
        assert v in applied1, f"migration v{v} missing from applied1: {applied1}"
    await d2.close()
