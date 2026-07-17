"""Sprint 19.5 Slice 6 Commit 4: tests for v27 dim backfill logic.

Covers (TDD §3 Commit 4, ~2 tests):
- v27's `dim = length(embedding) / 4` backfill produces the right
  dim for a 1536-dim embedding (1536 * 4 = 6144 bytes → dim=1536).
- Same for a 4096-dim embedding (4096 * 4 = 16384 bytes → dim=4096).

The v27 SQL is: `UPDATE file_embeddings SET dim = CAST(length(embedding)
/ 4 AS INTEGER) WHERE dim IS NULL`. The CAST is important — without
it, SQLite returns a float and the NOT NULL constraint would fail
on the post-v28 schema. This test pins that behavior for the two
canonical dims we use in Sprint 19.5 (1536 for legacy text-embedding-3
-small, 4096 for the new qwen/qwen3-embedding-8b).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.db import MIGRATIONS, SchemaMigrator


async def _build_pre_backfill_db(tmp_path: Path, file_id: str, blob_len: int) -> object:
    """Build a DB in the pre-v27 state (columns exist, NOT NULL not yet
    enforced, row has policy=NULL + dim=NULL). Returns the open aiosqlite
    connection — caller is responsible for closing it.

    Uses a partial MIGRATIONS dict that excludes v27 (backfill) and
    v28 (recreation) to leave policy/dim nullable.
    """
    import aiosqlite

    from hermes.memory.db import SCHEMA

    partial_migrations = {k: v for k, v in MIGRATIONS.items() if k not in (27, 28)}
    db_path = tmp_path / "backfill_dim.db"
    conn = await aiosqlite.connect(db_path, timeout=30.0)
    conn.row_factory = aiosqlite.Row
    await conn.executescript(SCHEMA)
    await conn.commit()
    migrator = SchemaMigrator(conn, partial_migrations)
    await migrator.run()
    # Add file + legacy row (no policy, no dim).
    await conn.execute(
        "INSERT INTO files (id, filename, mime_type, size_bytes, extracted_text, "
        "extraction_method) VALUES (?, ?, ?, ?, ?, ?)",
        (file_id, "l.pdf", "application/pdf", 100, "text", "pypdf"),
    )
    await conn.execute(
        "INSERT INTO file_embeddings (file_id, embedding) VALUES (?, ?)",
        (file_id, b"\x00" * blob_len),
    )
    await conn.commit()
    return conn


# --- Test 1: 1536-dim backfill -------------------------------------------


@pytest.mark.asyncio
async def test_backfill_dim_from_blob_size_for_1536(tmp_path: Path) -> None:
    """Backfill `dim` for a 1536-dim BLOB (1536 * 4 = 6144 bytes)."""
    conn = await _build_pre_backfill_db(tmp_path, "file_1536", 1536 * 4)
    try:
        # Verify pre-state
        async with conn.execute(
            "SELECT policy, dim, length(embedding) FROM file_embeddings WHERE file_id=?",
            ("file_1536",),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] is None
        assert row[1] is None
        assert int(row[2]) == 1536 * 4
        # Run the v27 backfill UPDATE for dim
        await conn.execute(
            "UPDATE file_embeddings SET dim = CAST(length(embedding) / 4 AS INTEGER) "
            "WHERE dim IS NULL"
        )
        await conn.commit()
        # Verify post-state
        async with conn.execute(
            "SELECT dim FROM file_embeddings WHERE file_id=?", ("file_1536",)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert int(row[0]) == 1536
    finally:
        await conn.close()


# --- Test 2: 4096-dim backfill -------------------------------------------


@pytest.mark.asyncio
async def test_backfill_dim_from_blob_size_for_4096(tmp_path: Path) -> None:
    """Backfill `dim` for a 4096-dim BLOB (4096 * 4 = 16384 bytes)."""
    conn = await _build_pre_backfill_db(tmp_path, "file_4096", 4096 * 4)
    try:
        # Verify pre-state
        async with conn.execute(
            "SELECT policy, dim, length(embedding) FROM file_embeddings WHERE file_id=?",
            ("file_4096",),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] is None
        assert row[1] is None
        assert int(row[2]) == 4096 * 4
        # Run the v27 backfill UPDATE for dim
        await conn.execute(
            "UPDATE file_embeddings SET dim = CAST(length(embedding) / 4 AS INTEGER) "
            "WHERE dim IS NULL"
        )
        await conn.commit()
        # Verify post-state
        async with conn.execute(
            "SELECT dim FROM file_embeddings WHERE file_id=?", ("file_4096",)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert int(row[0]) == 4096
    finally:
        await conn.close()
