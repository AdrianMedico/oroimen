"""Tests Sprint 9.2: memory_facts_staging + memory_facts + memory_fact_embeddings.

Cubre:
- staging CRUD (add, get, list, increment, promote, expire)
- facts CRUD (add, get, list, touch, delete)
- fact_embeddings CRUD
- CASCADE behavior
- find_similar_staging (heuristica textual placeholder)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes.memory.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


# --- memory_facts_staging ---


@pytest.mark.asyncio
async def test_add_staging_fact_creates_row(db: Database) -> None:
    """add_staging_fact inserta un fact en staging con defaults correctos."""
    await db.add_staging_fact(
        stg_id="stg_abc",
        category="user_preference",
        content="User prefers responses in espanol",
        confidence_score=0.85,
    )
    row = await db.get_staging_fact("stg_abc")
    assert row is not None
    assert row["category"] == "user_preference"
    assert row["content"] == "User prefers responses in espanol"
    assert row["confidence_score"] == 0.85
    assert row["occurrence_count"] == 1
    assert row["status"] == "pending"
    # v1.2 fix: source_conversation_ids siempre '[]' (nunca NULL)
    assert row["source_conversation_ids"] == "[]"


@pytest.mark.asyncio
async def test_add_staging_fact_with_source_conversations(db: Database) -> None:
    """add_staging_fact acepta lista de source_conversation_ids."""
    await db.add_staging_fact(
        stg_id="stg_1",
        category="project_context",
        content="Hermes Sprint 9 en desarrollo",
        confidence_score=0.92,
        source_conversation_ids=[1, 2, 3],
    )
    row = await db.get_staging_fact("stg_1")
    assert row is not None
    assert json.loads(row["source_conversation_ids"]) == [1, 2, 3]


@pytest.mark.asyncio
async def test_get_staging_fact_returns_none_for_missing(db: Database) -> None:
    assert await db.get_staging_fact("nonexistent") is None


@pytest.mark.asyncio
async def test_list_staging_facts_filters_by_status(db: Database) -> None:
    """list_staging_facts solo retorna los del status pedido."""
    await db.add_staging_fact("stg_a", "user_preference", "pref 1", 0.9)
    await db.add_staging_fact("stg_b", "user_preference", "pref 2", 0.8)
    # Marcar uno como rejected
    await db.conn.execute("UPDATE memory_facts_staging SET status='rejected' WHERE id='stg_b'")
    await db.conn.commit()
    pending = await db.list_staging_facts(status="pending")
    assert len(pending) == 1
    assert pending[0]["id"] == "stg_a"
    rejected = await db.list_staging_facts(status="rejected")
    assert len(rejected) == 1
    assert rejected[0]["id"] == "stg_b"


@pytest.mark.asyncio
async def test_list_staging_facts_min_occurrence(db: Database) -> None:
    """min_occurrence filtra facts con occurrence_count >= N."""
    await db.add_staging_fact("stg_low", "user_preference", "low", 0.5)
    await db.add_staging_fact("stg_high", "user_preference", "high", 0.5)
    # Incrementar high 3 veces
    for _ in range(2):
        await db.increment_staging_occurrence("stg_high")
    rows = await db.list_staging_facts(min_occurrence=3)
    assert len(rows) == 1
    assert rows[0]["id"] == "stg_high"


@pytest.mark.asyncio
async def test_increment_staging_occurrence(db: Database) -> None:
    """increment_staging_occurrence suma 1 y agrega conv_id."""
    await db.add_staging_fact("stg_1", "user_preference", "test", 0.8)
    await db.increment_staging_occurrence("stg_1", source_conversation_id=42)
    row = await db.get_staging_fact("stg_1")
    assert row is not None
    assert row["occurrence_count"] == 2
    assert json.loads(row["source_conversation_ids"]) == [42]
    # Segunda vez con mismo conv_id: NO se duplica
    await db.increment_staging_occurrence("stg_1", source_conversation_id=42)
    row = await db.get_staging_fact("stg_1")
    assert row["occurrence_count"] == 3
    assert json.loads(row["source_conversation_ids"]) == [42]
    # Con conv_id nuevo: se agrega
    await db.increment_staging_occurrence("stg_1", source_conversation_id=99)
    row = await db.get_staging_fact("stg_1")
    assert row["occurrence_count"] == 4
    assert json.loads(row["source_conversation_ids"]) == [42, 99]


@pytest.mark.asyncio
async def test_find_similar_staging_substring_match(db: Database) -> None:
    """find_similar_staging: match por substring (placeholder MVP)."""
    await db.add_staging_fact("stg_1", "user_preference", "User likes Python", 0.9)
    # Match exacto
    match = await db.find_similar_staging("user_preference", "User likes Python")
    assert match is not None
    assert match["id"] == "stg_1"
    # Match parcial (substring)
    match = await db.find_similar_staging("user_preference", "User likes Python and Rust")
    assert match is not None
    assert match["id"] == "stg_1"


@pytest.mark.asyncio
async def test_find_similar_staging_no_match_different_category(db: Database) -> None:
    """find_similar_staging: NO match si category distinta."""
    await db.add_staging_fact("stg_1", "user_preference", "test", 0.9)
    match = await db.find_similar_staging("project_context", "test")
    assert match is None


@pytest.mark.asyncio
async def test_promote_staging_to_fact_creates_fact(db: Database) -> None:
    """promote_staging_to_fact: inserta en memory_facts y marca staging."""
    # Crear conversaciones para satisfacer FK (thread_id distintos
    # para evitar UNIQUE constraint en (chat_id, thread_id, user_id))
    conv1 = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    conv2 = await db.new_conversation(chat_id=0, user_id=0, thread_id=2)
    await db.add_staging_fact(
        "stg_promote",
        "user_preference",
        "Test promote",
        0.9,
    )
    # Incrementar 2 veces para simular threshold
    await db.increment_staging_occurrence("stg_promote", source_conversation_id=conv1)
    await db.increment_staging_occurrence("stg_promote", source_conversation_id=conv2)
    await db.promote_staging_to_fact(
        stg_id="stg_promote",
        fact_id="fact_1",
        source_conversation_id=conv2,
    )
    # Fact creado
    fact = await db.get_fact("fact_1")
    assert fact is not None
    assert fact["category"] == "user_preference"
    assert fact["content"] == "Test promote"
    assert fact["source_conversation_id"] == conv2
    # Staging marcado como promoted
    stg = await db.get_staging_fact("stg_promote")
    assert stg["status"] == "promoted"


@pytest.mark.asyncio
async def test_promote_staging_is_permanent_flag(db: Database) -> None:
    """promote_staging_to_fact: is_permanent=True se persiste como 1."""
    await db.add_staging_fact("stg_1", "user_preference", "test", 0.9)
    await db.promote_staging_to_fact(stg_id="stg_1", fact_id="fact_1", is_permanent=True)
    fact = await db.get_fact("fact_1")
    assert fact["is_permanent"] == 1


@pytest.mark.asyncio
async def test_expire_old_staging(db: Database) -> None:
    """expire_old_staging marca como 'expired' los antiguos."""
    # Insertar uno "viejo" (manipulando last_seen_at directamente)
    await db.add_staging_fact("stg_old", "user_preference", "old", 0.5)
    await db.conn.execute(
        "UPDATE memory_facts_staging "
        "SET last_seen_at = DATE('now', '-100 days') WHERE id = 'stg_old'"
    )
    await db.conn.commit()
    # Insertar uno nuevo
    await db.add_staging_fact("stg_new", "user_preference", "new", 0.5)
    # Expirar a 90 días
    count = await db.expire_old_staging(days=90)
    assert count == 1
    # Verificar status
    old = await db.get_staging_fact("stg_old")
    assert old["status"] == "expired"
    new = await db.get_staging_fact("stg_new")
    assert new["status"] == "pending"


# --- memory_facts (consolidated) ---


@pytest.mark.asyncio
async def test_add_fact_creates_row(db: Database) -> None:
    await db.add_fact(
        fact_id="fact_1",
        category="user_preference",
        content="User prefers concise answers",
        is_permanent=False,
    )
    fact = await db.get_fact("fact_1")
    assert fact is not None
    assert fact["category"] == "user_preference"
    assert fact["content"] == "User prefers concise answers"
    assert fact["is_permanent"] == 0
    assert fact["is_verified"] == 0


@pytest.mark.asyncio
async def test_get_fact_returns_none_for_missing(db: Database) -> None:
    assert await db.get_fact("nonexistent") is None


@pytest.mark.asyncio
async def test_list_facts_filters_by_category(db: Database) -> None:
    await db.add_fact("f1", "user_preference", "pref", 0)
    await db.add_fact("f2", "project_context", "proj", 0)
    await db.add_fact("f3", "user_preference", "pref2", 0)
    prefs = await db.list_facts(category="user_preference")
    assert len(prefs) == 2
    assert {f["id"] for f in prefs} == {"f1", "f3"}


@pytest.mark.asyncio
async def test_touch_fact_updates_last_referenced_at(db: Database) -> None:
    await db.add_fact("f1", "user_preference", "test", 0)
    await db.touch_fact("f1")
    fact = await db.get_fact("f1")
    assert fact["last_referenced_at"] is not None


@pytest.mark.asyncio
async def test_delete_fact_removes_row(db: Database) -> None:
    await db.add_fact("f_del", "user_preference", "test", 0)
    assert await db.delete_fact("f_del") is True
    assert await db.get_fact("f_del") is None


@pytest.mark.asyncio
async def test_delete_fact_returns_false_for_missing(db: Database) -> None:
    assert await db.delete_fact("nonexistent") is False


@pytest.mark.asyncio
async def test_delete_fact_cascades_to_embeddings(db: Database) -> None:
    """CASCADE limpia memory_fact_embeddings al borrar el fact."""
    await db.add_fact("f_casc", "user_preference", "test", 0)
    await db.add_fact_embedding("f_casc", b"\x00" * 6144)
    async with db.conn.execute(
        "SELECT fact_id FROM memory_fact_embeddings WHERE fact_id='f_casc'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    await db.delete_fact("f_casc")
    async with db.conn.execute(
        "SELECT fact_id FROM memory_fact_embeddings WHERE fact_id='f_casc'"
    ) as cur:
        row = await cur.fetchone()
    assert row is None


# --- memory_fact_embeddings ---


@pytest.mark.asyncio
async def test_add_fact_embedding_upserts(db: Database) -> None:
    await db.add_fact("f1", "user_preference", "test", 0)
    await db.add_fact_embedding("f1", b"\x00" * 100, model="model-a")
    await db.add_fact_embedding("f1", b"\xff" * 100, model="model-b")
    rows = await db.get_all_fact_embeddings()
    assert len(rows) == 1
    fid, blob = rows[0]
    assert fid == "f1"
    assert blob == b"\xff" * 100


@pytest.mark.asyncio
async def test_get_all_fact_embeddings_empty(db: Database) -> None:
    assert await db.get_all_fact_embeddings() == []
