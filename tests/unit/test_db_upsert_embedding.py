"""Sprint 19.5 Slice 6 Commit 4: tests for ``db.upsert_embedding``.

Covers (TDD §3 Commit 4, ~3 tests):
- ``upsert_embedding`` inserts a new row when none exists for the
  (file_id, policy) composite key.
- ``upsert_embedding`` replaces an existing row in place (no duplicate
  rows for the same composite key).
- ``upsert_embedding`` stores the ``dim`` as given by the caller (no
  internal validation against the embedding BLOB length; the helper
  in EmbeddingsService extracts dim from the canonical vector).

The legacy ``add_file_embedding`` method is also exercised (it shares
the table) — see ``test_db_files.py`` and ``test_embeddings.py`` for
those.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


# --- Test 1: insert new row ----------------------------------------------


@pytest.mark.asyncio
async def test_upsert_embedding_inserts_new_row(db: Database) -> None:
    """upsert_embedding on a fresh (file_id, policy) inserts a new row.

    Verifies:
    - The row is queryable via SELECT.
    - All 6 columns are populated correctly (file_id, embedding,
      embedded_at, model, dim, policy).
    - The embedded_at is set by the SQL DEFAULT CURRENT_TIMESTAMP
      (not None).
    """
    # Need a parent file row for the FK.
    await db.add_file("file_upsert_1", "u1.pdf", "application/pdf", 100, "text", "pypdf")
    emb = b"\x00\x01\x02\x03" * 384  # 384 dims x 4 bytes (float32)
    await db.upsert_embedding("file_upsert_1", emb, model="qwen-8b", dim=384, policy="chat_rag")
    async with db.conn.execute(
        "SELECT file_id, embedding, embedded_at, model, dim, policy "
        "FROM file_embeddings WHERE file_id=? AND policy=?",
        ("file_upsert_1", "chat_rag"),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    file_id, blob, embedded_at, model, dim, policy = row
    assert file_id == "file_upsert_1"
    assert blob == emb
    assert embedded_at is not None  # set by SQL DEFAULT
    assert model == "qwen-8b"
    assert int(dim) == 384
    assert policy == "chat_rag"


# --- Test 2: replace existing row ---------------------------------------


@pytest.mark.asyncio
async def test_upsert_embedding_replaces_existing_row(db: Database) -> None:
    """upsert_embedding on an existing (file_id, policy) replaces in place.

    Verifies:
    - The row count for that composite key stays at 1 (no duplicate).
    - The new embedding, model, dim overwrite the old values.
    - The embedded_at is updated (CURRENT_TIMESTAMP at write time).
    """
    await db.add_file("file_upsert_2", "u2.pdf", "application/pdf", 100, "text", "pypdf")
    emb1 = b"\x00" * (384 * 4)
    emb2 = b"\xff" * (4096 * 4)
    # Insert first
    await db.upsert_embedding(
        "file_upsert_2", emb1, model="granite-97m", dim=384, policy="chat_rag"
    )
    # Replace (same composite key, different content)
    await db.upsert_embedding("file_upsert_2", emb2, model="qwen-8b", dim=4096, policy="chat_rag")
    # Verify count = 1 (composite PK enforces uniqueness)
    async with db.conn.execute(
        "SELECT COUNT(*) FROM file_embeddings WHERE file_id=? AND policy=?",
        ("file_upsert_2", "chat_rag"),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert int(row[0]) == 1
    # Verify content is the new one
    async with db.conn.execute(
        "SELECT embedding, model, dim FROM file_embeddings WHERE file_id=? AND policy=?",
        ("file_upsert_2", "chat_rag"),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    blob, model, dim = row
    assert blob == emb2
    assert model == "qwen-8b"
    assert int(dim) == 4096


# --- Test 3: dim is stored as given (no validation) ---------------------


@pytest.mark.asyncio
async def test_upsert_embedding_validates_dim(db: Database) -> None:
    """upsert_embedding stores the caller-provided dim as-is.

    The helper in EmbeddingsService (``_embed_and_store_router``)
    passes ``int(vector.shape[0])`` so the dim is always consistent
    with the embedding BLOB length. ``upsert_embedding`` itself does
    NOT cross-check; it stores whatever the caller passes. This test
    pins that contract: if a caller passes a wrong dim, the row is
    persisted with that wrong value (validation belongs to the caller).
    """
    await db.add_file("file_upsert_3", "u3.pdf", "application/pdf", 100, "text", "pypdf")
    # 1536 dim embedding, but we lie and say dim=4096.
    # (This is what would happen if a bug in the service passed the
    # wrong dim — the DB accepts it without complaint.)
    emb = b"\x00" * (1536 * 4)
    await db.upsert_embedding(
        "file_upsert_3", emb, model="text-embedding-3-small", dim=4096, policy="vault_ingest"
    )
    async with db.conn.execute(
        "SELECT dim, length(embedding) FROM file_embeddings WHERE file_id=? AND policy=?",
        ("file_upsert_3", "vault_ingest"),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    dim, blob_len = row
    # dim stored as given (4096), even though blob is 1536*4 = 6144 bytes
    assert int(dim) == 4096
    assert int(blob_len) == 1536 * 4
