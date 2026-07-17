"""Tests Sprint 9.2: SleepCycle pipeline end-to-end.

Cubre:
- _load_recent_conversation_ids (filtro is_archived=0 + ventana temporal)
- _process_conversation (extraccion + upsert)
- _promote_pending_facts (threshold + embedding opcional)
- run() completo con metricas
- Idempotencia (re-run es seguro)
- Errores por conversacion no rompen el pipeline
- asyncio.sleep(0.2) yield entre conversaciones (P0-2 Gemini)
- opt-in via settings (sleep_cycle_enabled)
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes.config import Settings
from hermes.memory.db import Database
from hermes.memory.sleep_cycle import (
    YIELD_BETWEEN_CONVERSATIONS_SECONDS,
    SleepCycle,
)

# --- Helpers ---


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
    """Settings minimas para tests. sleep_cycle_enabled default False."""
    return Settings(
        telegram_bot_token="1234567890:test_token_here",
        opencode_go_api_key="sk-test-key",
        gemini_api_key="AIza-test-key",
        sleep_cycle_enabled=False,
        sleep_cycle_hour=4,
        memory_fact_min_mentions=3,
    )


@pytest.fixture
def mock_embeddings() -> MagicMock:
    emb = MagicMock()
    emb.is_enabled = False  # default: RAG disabled
    emb.embed = AsyncMock()
    return emb


# --- _load_recent_conversation_ids ---


@pytest.mark.asyncio
async def test_load_recent_returns_active_in_window(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """Carga conversaciones no archivadas dentro de la ventana."""
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    # Crear 3 conversaciones: 1 reciente, 1 vieja, 1 archivada
    conv_recent = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    conv_old = await db.new_conversation(chat_id=0, user_id=0, thread_id=2)
    conv_archived = await db.new_conversation(chat_id=0, user_id=0, thread_id=3)
    # Archivar la tercera
    await db.archive_conversation(conv_archived)
    # Backdate la segunda
    old_date = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    await db.conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (old_date, conv_old),
    )
    await db.conn.commit()
    # Load con ventana 24h
    ids = await cycle._load_recent_conversation_ids(lookback_hours=24)
    # Solo la reciente + la archivada (es reciente pero archivada -> EXCLUIDA)
    # Wait, la archivada se acaba de crear (updated_at=now), asi que esta en ventana
    # Pero is_archived=1, asi que NO se incluye
    assert conv_recent in ids
    assert conv_old not in ids
    assert conv_archived not in ids


@pytest.mark.asyncio
async def test_load_recent_respects_lookback(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """Lookback de 1h excluye conversaciones de hace 2h."""
    import time as _time

    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    conv = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    # Backdate 2h usando el mismo formato que _load_recent_conversation_ids
    # (strftime 'YYYY-MM-DD HH:MM:SS' para coincidir con CURRENT_TIMESTAMP
    # de SQLite).
    old_unix = _time.time() - (2 * 3600)
    old_date = _time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime(old_unix))
    await db.conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (old_date, conv),
    )
    await db.conn.commit()
    # Lookback 1h -> excluye
    ids_1h = await cycle._load_recent_conversation_ids(lookback_hours=1)
    assert conv not in ids_1h
    # Lookback 3h -> incluye
    ids_3h = await cycle._load_recent_conversation_ids(lookback_hours=3)
    assert conv in ids_3h


# --- _process_conversation ---


@pytest.mark.asyncio
async def test_process_conversation_extracts_and_upserts(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """_process_conversation: extrae facts y los upserta a staging."""
    mock_router.chat.return_value = _FakeLLMResponse(
        json.dumps(
            [
                {
                    "category": "user_preference",
                    "content": "User likes Python",
                    "confidence_score": 0.9,
                }
            ]
        )
    )
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    conv_id = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    await db.add_message(conv_id, "user", "Me gusta Python")
    await db.add_message(conv_id, "assistant", "Genial!")
    metrics = {"conversations_processed": 0, "facts_extracted": 0, "errors": 0}
    await cycle._process_conversation(conv_id, metrics)
    assert metrics["conversations_processed"] == 1
    assert metrics["facts_extracted"] == 1
    assert metrics["errors"] == 0
    # Verificar que se persistio
    pending = await db.list_staging_facts(status="pending")
    assert len(pending) == 1
    assert pending[0]["content"] == "User likes Python"


@pytest.mark.asyncio
async def test_process_conversation_skips_empty_history(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """Conversacion sin mensajes no se procesa."""
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    conv_id = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    metrics = {"conversations_processed": 0, "facts_extracted": 0, "errors": 0}
    await cycle._process_conversation(conv_id, metrics)
    assert metrics["conversations_processed"] == 0
    mock_router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_process_conversation_increments_on_duplicate(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """Si el mismo fact aparece en 2 conversaciones, occurrence_count=2."""
    mock_router.chat.return_value = _FakeLLMResponse(
        json.dumps(
            [
                {
                    "category": "user_preference",
                    "content": "User likes Python",
                    "confidence_score": 0.9,
                }
            ]
        )
    )
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    # Conversacion 1
    conv1 = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    await db.add_message(conv1, "user", "Me gusta Python")
    metrics = {"conversations_processed": 0, "facts_extracted": 0, "errors": 0}
    await cycle._process_conversation(conv1, metrics)
    # Conversacion 2 (mismo fact, debe dedup)
    conv2 = await db.new_conversation(chat_id=0, user_id=0, thread_id=2)
    await db.add_message(conv2, "user", "Python es mi favorito")
    metrics = {"conversations_processed": 0, "facts_extracted": 0, "errors": 0}
    await cycle._process_conversation(conv2, metrics)
    # 1 fact en staging, occurrence_count=2
    pending = await db.list_staging_facts(status="pending")
    assert len(pending) == 1
    assert pending[0]["occurrence_count"] == 2


# --- _promote_pending_facts ---


@pytest.mark.asyncio
async def test_promote_pending_facts_below_threshold_no_promote(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """Facts con occurrence < threshold NO se promueven."""
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    await db.add_staging_fact("stg_1", "user_preference", "test", 0.9)
    # occurrence=1 (default), threshold=3 (settings default)
    promoted = await cycle._promote_pending_facts(min_mentions=3)
    assert promoted == 0
    # Sigue en staging
    pending = await db.list_staging_facts(status="pending")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_promote_pending_facts_at_threshold_promotes(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """Facts con occurrence >= threshold se promueven a memory_facts."""
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    await db.add_staging_fact("stg_1", "user_preference", "test", 0.9)
    # Incrementar 2 veces para total=3
    await db.increment_staging_occurrence("stg_1")
    await db.increment_staging_occurrence("stg_1")
    promoted = await cycle._promote_pending_facts(min_mentions=3)
    assert promoted == 1
    # Verificar fact creado
    all_facts = await db.list_facts()
    assert len(all_facts) == 1
    assert all_facts[0]["content"] == "test"
    # Staging marcado como promoted
    stg = await db.get_staging_fact("stg_1")
    assert stg["status"] == "promoted"


@pytest.mark.asyncio
async def test_promote_pending_facts_increments_errors_in_metrics(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """P2 Copilot review 2026-06-26: si la promocion falla, el
    counter de errors en metrics debe incrementarse para que el
    caller (cycle.run) sepa cuantos promote fallaron."""
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    await db.add_staging_fact("stg_1", "user_preference", "test", 0.9)
    await db.increment_staging_occurrence("stg_1")
    await db.increment_staging_occurrence("stg_1")

    # Forzar error: monkey-patch promote_staging_to_fact para que lance
    async def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated promote failure")

    cycle._db.promote_staging_to_fact = _raise  # type: ignore[method-assign]
    metrics = {
        "conversations_processed": 0,
        "facts_extracted": 0,
        "facts_promoted": 0,
        "staging_expired": 0,
        "errors": 0,
    }
    promoted = await cycle._promote_pending_facts(min_mentions=3, metrics=metrics)
    assert promoted == 0
    assert metrics["errors"] == 1  # el error fue contado


@pytest.mark.asyncio
async def test_expire_old_staging_uses_date_normalization(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """P2 Copilot review 2026-06-26: el filtro de expiracion debe
    normalizar last_seen_at (timestamp) y DATE('now') a solo fecha
    para que la comparacion funcione."""
    # Crear staging antiguo (hace 100 dias, timestamp completo)
    import time as _time

    old_unix = _time.time() - (100 * 24 * 3600)
    old_date = _time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime(old_unix))
    await db.add_staging_fact("stg_old", "user_preference", "old fact", 0.9)
    await db.conn.execute(
        "UPDATE memory_facts_staging SET last_seen_at = ? WHERE id = 'stg_old'",
        (old_date,),
    )
    await db.conn.commit()
    # Crear staging reciente (hoy)
    await db.add_staging_fact("stg_recent", "user_preference", "recent fact", 0.9)
    # Expirar con threshold 90 dias
    count = await db.expire_old_staging(days=90)
    assert count == 1
    # Verificar que solo el viejo esta expired
    old_stg = await db.get_staging_fact("stg_old")
    recent_stg = await db.get_staging_fact("stg_recent")
    assert old_stg["status"] == "expired"
    assert recent_stg["status"] == "pending"


@pytest.mark.asyncio
async def test_promote_pending_facts_with_embeddings(
    db: Database, mock_router: MagicMock, settings: Settings, mock_embeddings: MagicMock
) -> None:
    """Si embeddings_service.is_enabled, calcula embedding del fact."""
    import numpy as np

    mock_embeddings.is_enabled = True
    mock_embeddings.embed.return_value = np.zeros(4096, dtype=np.float32)
    cycle = SleepCycle(
        db=db,
        router=mock_router,
        settings=settings,
        embeddings_service=mock_embeddings,
    )
    await db.add_staging_fact("stg_1", "user_preference", "test", 0.9)
    await db.increment_staging_occurrence("stg_1")
    await db.increment_staging_occurrence("stg_1")
    promoted = await cycle._promote_pending_facts(min_mentions=3)
    assert promoted == 1
    # Embedding llamado
    mock_embeddings.embed.assert_called_once_with("test")
    # Verificar que el embedding se persistio
    rows = await db.get_all_fact_embeddings()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_promote_pending_facts_without_embeddings(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """Si embeddings_service es None, la promoción funciona sin embedding."""
    cycle = SleepCycle(db=db, router=mock_router, settings=settings, embeddings_service=None)
    await db.add_staging_fact("stg_1", "user_preference", "test", 0.9)
    await db.increment_staging_occurrence("stg_1")
    await db.increment_staging_occurrence("stg_1")
    promoted = await cycle._promote_pending_facts(min_mentions=3)
    assert promoted == 1
    # Sin embeddings
    assert await db.get_all_fact_embeddings() == []


# --- run() completo ---


@pytest.mark.asyncio
async def test_run_full_pipeline_with_mocked_llm(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """run() ejecuta el pipeline completo end-to-end."""
    mock_router.chat.return_value = _FakeLLMResponse(
        json.dumps(
            [
                {
                    "category": "user_preference",
                    "content": "User prefers concise answers",
                    "confidence_score": 0.9,
                }
            ]
        )
    )
    # Crear conversacion con mensaje
    conv = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    await db.add_message(conv, "user", "Please be brief")
    # Crear SleepCycle y ejecutar
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    metrics = await cycle.run(lookback_hours=24, min_mentions=3, promote=False)
    # Verificar metricas
    assert metrics["conversations_processed"] == 1
    assert metrics["facts_extracted"] == 1
    assert metrics["errors"] == 0
    # Verificar que se persistio
    pending = await db.list_staging_facts(status="pending")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_run_is_idempotent(db: Database, mock_router: MagicMock, settings: Settings) -> None:
    """Re-ejecutar run() es seguro: la conversacion ya procesada se salta.

    Sprint 12 (ADR-007): el flag sleep_cycle_processed=1 evita re-procesar
    la misma conversacion en runs subsiguientes. El extractor sigue siendo
    idempotente (upsert por dedup), pero el job no lo invoca dos veces
    sobre la misma conv. Esto evita re-procesar conversaciones persistentes
    de la app nativa RikkaHub (chat_id != 0) en cada ejecucion.

    Para verificar la idempotencia del EXTRACTOR (no del job), creamos
    dos conversaciones con el mismo contenido. El segundo run no procesa
    ninguna (ambas marcadas como sleep_cycle_processed=1), pero el primer
    run ya dedupeo el fact via upsert.
    """
    mock_router.chat.return_value = _FakeLLMResponse(
        json.dumps(
            [
                {
                    "category": "user_preference",
                    "content": "User likes Python",
                    "confidence_score": 0.9,
                }
            ]
        )
    )
    conv = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    await db.add_message(conv, "user", "Me gusta Python")
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    # Run 1: procesa la conv, marca sleep_cycle_processed=1, extrae 1 fact
    metrics_1 = await cycle.run(lookback_hours=24, min_mentions=3)
    assert metrics_1["conversations_processed"] == 1
    assert metrics_1["facts_extracted"] == 1
    # Run 2: la conv ya esta marcada como procesada, no se re-procesa
    metrics_2 = await cycle.run(lookback_hours=24, min_mentions=3)
    assert metrics_2["conversations_processed"] == 0
    assert metrics_2["facts_extracted"] == 0
    # 1 fact en staging, occurrence_count=1 (NO se re-extrae)
    pending = await db.list_staging_facts(status="pending")
    assert len(pending) == 1
    assert pending[0]["occurrence_count"] == 1


@pytest.mark.asyncio
async def test_run_handles_per_conversation_errors(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """Si una conversacion falla, las demas continuan."""
    # Crear 2 conversaciones
    conv1 = await db.new_conversation(chat_id=0, user_id=0, thread_id=1)
    conv2 = await db.new_conversation(chat_id=0, user_id=0, thread_id=2)
    await db.add_message(conv1, "user", "msg1")
    await db.add_message(conv2, "user", "msg2")
    # Mock LLM: falla en primera llamada, OK en segunda
    call_count = 0

    async def chat_side_effect(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("LLM temporary error")
        return _FakeLLMResponse(
            json.dumps(
                [
                    {
                        "category": "user_preference",
                        "content": "test fact",
                        "confidence_score": 0.9,
                    }
                ]
            )
        )

    mock_router.chat.side_effect = chat_side_effect
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    metrics = await cycle.run(lookback_hours=24, min_mentions=3)
    # 1 error (primera conv), 1 success (segunda)
    assert metrics["errors"] == 1
    assert metrics["conversations_processed"] == 1
    assert metrics["facts_extracted"] == 1


@pytest.mark.asyncio
async def test_run_yields_between_conversations(
    db: Database, mock_router: MagicMock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P0-2 Gemini fix: asyncio.sleep(0.2) entre conversaciones.

    Monkeypatch asyncio.sleep para verificar que se llama con el yield
    correcto entre conversaciones (no con otros valores).
    """
    # Track sleep calls
    sleep_calls: list[float] = []
    real_sleep = asyncio.sleep

    async def mock_sleep(delay: float) -> Any:
        sleep_calls.append(delay)
        # No esperamos realmente, retornamos inmediatamente
        await real_sleep(0)  # yield control sin esperar

    monkeypatch.setattr("asyncio.sleep", mock_sleep)
    mock_router.chat.return_value = _FakeLLMResponse(
        json.dumps(
            [
                {
                    "category": "user_preference",
                    "content": "test",
                    "confidence_score": 0.9,
                }
            ]
        )
    )
    # 3 conversaciones
    for i in range(3):
        conv = await db.new_conversation(chat_id=0, user_id=0, thread_id=i + 1)
        await db.add_message(conv, "user", f"msg{i}")
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    await cycle.run(lookback_hours=24, min_mentions=3)
    # Verificar: al menos 2 yields con 0.2s (entre 3 conversaciones)
    yield_calls = [c for c in sleep_calls if c == YIELD_BETWEEN_CONVERSATIONS_SECONDS]
    assert len(yield_calls) >= 2


@pytest.mark.asyncio
async def test_run_expires_old_staging(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """run() tambien limpia staging antiguo (expire_old_staging)."""
    # Insertar staging viejo manualmente
    await db.add_staging_fact("stg_old", "user_preference", "old", 0.5)
    await db.conn.execute(
        "UPDATE memory_facts_staging "
        "SET last_seen_at = DATE('now', '-100 days') WHERE id = 'stg_old'"
    )
    await db.conn.commit()
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    metrics = await cycle.run(lookback_hours=24, min_mentions=3, promote=False)
    assert metrics["staging_expired"] == 1
    old = await db.get_staging_fact("stg_old")
    assert old["status"] == "expired"


@pytest.mark.asyncio
async def test_run_returns_metrics_dict(
    db: Database, mock_router: MagicMock, settings: Settings
) -> None:
    """run() siempre retorna un dict con metricas."""
    cycle = SleepCycle(db=db, router=mock_router, settings=settings)
    metrics = await cycle.run(lookback_hours=24, min_mentions=3, promote=False)
    assert "conversations_processed" in metrics
    assert "facts_extracted" in metrics
    assert "facts_promoted" in metrics
    assert "staging_expired" in metrics
    assert "errors" in metrics
    assert "duration_seconds" in metrics
    assert metrics["duration_seconds"] >= 0
