"""Regression test Sprint 9.2 P0-1 (minimax-m3 cross-review).

Verifica que un fallo en el upsert de UN candidato no contamina
la transaccion completa (los demas candidatos se persisten).

Sin SAVEPOINT per-candidate, el codigo original hacia que un error
en un solo candidato (e.g., transient DB error) se catche y continue,
pero los INSERTs previos ya estaban en la transaccion y se
commiteaban junto con los siguientes. Eso es "partial state".

Con SAVEPOINT per-candidate:
- Si candidato A falla: ROLLBACK TO SAVEPOINT solo afecta A
- Candidatos B, C, D (siguientes) continuan normalmente
- COMMIT final commitea B, C, D
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.config import Settings
from hermes.memory.db import Database
from hermes.memory.sleep_cycle import SleepCycle


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


@pytest.fixture
def mock_router() -> MagicMock:
    router = MagicMock()
    router.chat = AsyncMock()
    return router


@pytest.fixture
def settings() -> Settings:
    return Settings(
        telegram_bot_token="1234567890:test_token_here",
        opencode_go_api_key="sk-test-key",
        gemini_api_key="AIza-test-key",
        sleep_cycle_enabled=False,
        sleep_cycle_hour=4,
        memory_fact_min_mentions=3,
    )


@pytest.mark.asyncio
async def test_candidate_failure_does_not_block_other_candidates(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """P0-1 regression: 1 candidato falla, los demas se persisten.

    LLM devuelve 3 candidates. Mockeamos upsert_to_staging para que
    el SEGUNDO lance error. Verificamos que el 1ro y 3ro se persisten
    y metrics refleja 2 extracted + 1 error.
    """
    # LLM devuelve 3 candidates validos
    candidates_json = json.dumps(
        [
            {"category": "user_preference", "content": "Fact 0", "confidence_score": 0.9},
            {"category": "user_preference", "content": "Fact 1", "confidence_score": 0.9},
            {"category": "user_preference", "content": "Fact 2", "confidence_score": 0.9},
        ]
    )
    mock_router.chat.return_value = _FakeLLMResponse(candidates_json)

    # Crear conversacion
    conv = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    await db.add_message(conv, "user", "msg1")
    await db.add_message(conv, "user", "msg2")

    cycle = SleepCycle(db=db, router=mock_router, settings=settings)

    # Patch upsert_to_staging para que el 2do candidato falle
    original_upsert = cycle._extractor.upsert_to_staging
    call_count = {"n": 0}

    async def maybe_failing_upsert(cand: Any) -> str:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("Simulated transient DB error")
        return await original_upsert(cand)

    with patch.object(cycle._extractor, "upsert_to_staging", side_effect=maybe_failing_upsert):
        metrics = await cycle.run(lookback_hours=24, min_mentions=3, promote=False)

    # 2 facts extraidos (1 y 3), 1 error (2)
    assert metrics["facts_extracted"] == 2
    assert metrics["errors"] == 1
    assert metrics["conversations_processed"] == 1

    # Verificar que 1 y 3 estan en staging, 2 NO
    pending = await db.list_staging_facts(status="pending")
    persisted_contents = sorted([row["content"] for row in pending])
    assert persisted_contents == [
        "Fact 0",
        "Fact 2",
    ], f"Expected ['Fact 0', 'Fact 2'], got {persisted_contents}"


@pytest.mark.asyncio
async def test_all_candidates_failing_still_increments_errors(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """Si TODOS los candidates fallan, metrics refleja 0 extracted y N errors."""
    candidates_json = json.dumps(
        [
            {"category": "user_preference", "content": f"Fact {i}", "confidence_score": 0.9}
            for i in range(3)
        ]
    )
    mock_router.chat.return_value = _FakeLLMResponse(candidates_json)

    conv = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    await db.add_message(conv, "user", "msg")

    cycle = SleepCycle(db=db, router=mock_router, settings=settings)

    async def always_failing_upsert(cand: Any) -> str:
        raise RuntimeError("Always fails")

    with patch.object(cycle._extractor, "upsert_to_staging", side_effect=always_failing_upsert):
        metrics = await cycle.run(lookback_hours=24, min_mentions=3, promote=False)

    assert metrics["facts_extracted"] == 0
    assert metrics["errors"] == 3
    assert metrics["conversations_processed"] == 1
    # Ningun fact en staging
    pending = await db.list_staging_facts(status="pending")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_savepoint_isolates_failure_to_single_candidate(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """P0-1: SAVEPOINT aísla el failure. Verifica con DB query directa."""
    # 3 candidates, el del medio falla
    candidates_json = json.dumps(
        [
            {"category": "user_preference", "content": "A", "confidence_score": 0.9},
            {"category": "user_preference", "content": "B", "confidence_score": 0.9},
            {"category": "user_preference", "content": "C", "confidence_score": 0.9},
        ]
    )
    mock_router.chat.return_value = _FakeLLMResponse(candidates_json)

    conv = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    await db.add_message(conv, "user", "msg")

    cycle = SleepCycle(db=db, router=mock_router, settings=settings)

    original_upsert = cycle._extractor.upsert_to_staging
    call_count = {"n": 0}

    async def maybe_failing_upsert(cand: Any) -> str:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("DB transient error")
        return await original_upsert(cand)

    with patch.object(cycle._extractor, "upsert_to_staging", side_effect=maybe_failing_upsert):
        await cycle._process_conversation(
            conv_id=conv,
            metrics={
                "conversations_processed": 0,
                "facts_extracted": 0,
                "facts_promoted": 0,
                "staging_expired": 0,
                "errors": 0,
            },
        )

    # Verificar directamente en DB: A y C persisten, B no
    async with db.conn.execute("SELECT content FROM memory_facts_staging ORDER BY content") as cur:
        rows = await cur.fetchall()
    contents = [r[0] for r in rows]
    assert contents == ["A", "C"], f"Expected ['A', 'C'], got {contents}"


@pytest.mark.asyncio
async def test_savepoint_allows_subsequent_upserts_after_failure(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """P0-1: SAVEPOINT no aborta el loop. Candidato C se procesa tras fallo de B."""
    candidates_json = json.dumps(
        [
            {"category": "user_preference", "content": "First", "confidence_score": 0.9},
            {"category": "user_preference", "content": "Second", "confidence_score": 0.9},
            {"category": "user_preference", "content": "Third", "confidence_score": 0.9},
        ]
    )
    mock_router.chat.return_value = _FakeLLMResponse(candidates_json)

    conv = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    await db.add_message(conv, "user", "msg")

    cycle = SleepCycle(db=db, router=mock_router, settings=settings)

    original_upsert = cycle._extractor.upsert_to_staging
    call_count = {"n": 0}

    async def maybe_failing_upsert(cand: Any) -> str:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("Simulated")
        return await original_upsert(cand)

    metrics = {
        "conversations_processed": 0,
        "facts_extracted": 0,
        "facts_promoted": 0,
        "staging_expired": 0,
        "errors": 0,
    }

    with patch.object(cycle._extractor, "upsert_to_staging", side_effect=maybe_failing_upsert):
        await cycle._process_conversation(conv_id=conv, metrics=metrics)

    # Verificar que el loop proceso los 3 (no aborto en B)
    assert call_count["n"] == 3, f"Expected 3 upsert attempts, got {call_count['n']}"
    # 2 successful, 1 failed
    assert metrics["facts_extracted"] == 2
    assert metrics["errors"] == 1
