"""Tests Sprint 9.2: MemoryFactExtractor.

Cubre:
- Prompt formatting
- LLM response parsing (con variaciones: markdown wrap, JSON directo, invalido)
- Confidence threshold
- Categoria validation
- Valid candidates filtering
- Source conversation ID assignment
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes.memory.db import Database
from hermes.memory.facts import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_CANDIDATES_PER_CONVERSATION,
    VALID_CATEGORIES,
    FactCandidate,
    MemoryFactExtractor,
    _format_messages,
    _parse_llm_response,
    _validate_candidates,
)

# --- Helpers ---


class _FakeResponse:
    """Mock de LLM response (LLMRouter.chat)."""

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


# --- _format_messages ---


def test_format_messages_basic() -> None:
    msgs = [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Hola, como puedo ayudar?"},
    ]
    result = _format_messages(msgs)
    assert "user: Hola" in result
    assert "assistant: Hola, como puedo ayudar?" in result


def test_format_messages_truncates_long_content() -> None:
    long_content = "X" * 600
    msgs = [{"role": "user", "content": long_content}]
    result = _format_messages(msgs)
    assert "..." in result
    # El limite es 500 chars + elipsis
    assert len(result) < len(long_content) + 50


def test_format_messages_skips_empty() -> None:
    msgs = [
        {"role": "user", "content": "Hola"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "   "},
    ]
    result = _format_messages(msgs)
    assert result.count("user:") == 1


# --- _parse_llm_response ---


def test_parse_llm_response_direct_json() -> None:
    raw = json.dumps(
        [
            {"category": "user_preference", "content": "test", "confidence_score": 0.8},
        ]
    )
    result = _parse_llm_response(raw)
    assert len(result) == 1
    assert result[0]["content"] == "test"


def test_parse_llm_response_markdown_wrap() -> None:
    """LLM a veces envuelve JSON en markdown code blocks."""
    raw = (
        "Aqui tienes los facts:\n"
        "```json\n"
        + json.dumps(
            [
                {"category": "user_preference", "content": "test1", "confidence_score": 0.9},
                {"category": "project_context", "content": "test2", "confidence_score": 0.7},
            ]
        )
        + "\n```\n"
        "Espero que te sirva."
    )
    result = _parse_llm_response(raw)
    assert len(result) == 2


def test_parse_llm_response_invalid_json() -> None:
    result = _parse_llm_response("no es json {")
    assert result == []


def test_parse_llm_response_empty_array() -> None:
    result = _parse_llm_response("[]")
    assert result == []


def test_parse_llm_response_text_with_embedded_json() -> None:
    """El LLM puede dar texto explicativo + JSON. Extraemos solo el array."""
    raw = (
        "He analizado la conversacion. Los facts que detecte son:\n"
        + json.dumps(
            [
                {"category": "user_preference", "content": "test", "confidence_score": 0.85},
            ]
        )
        + "\nSi necesitas mas, preguntame."
    )
    result = _parse_llm_response(raw)
    assert len(result) == 1
    assert result[0]["category"] == "user_preference"


def test_parse_llm_response_handles_multiple_brackets_in_text() -> None:
    """P1 Copilot review 2026-06-26: el parser debe ser robusto ante
    texto con multiples bloques entre corchetes. Caso edge tipico:
    markdown con lista no-JSON ANTES del JSON array valido.

    Antes (regex greedy): el match tomaba '[' de la lista markdown
    hasta ']' del JSON valido, formando un string no parseable, y
    devolvia [] falsamente.

    Ahora (json.JSONDecoder.raw_decode): itera por cada '[' y prueba
    parsear; acepta el primero que produce una lista valida.
    """
    raw = (
        "Notas preliminares (no son facts):\n"
        "- mencion [a, b] en el texto\n"
        "- discusion [c] lateral\n"
        "\nAhora el JSON valido:\n"
        + json.dumps(
            [
                {"category": "user_preference", "content": "test", "confidence_score": 0.9},
            ]
        )
    )
    result = _parse_llm_response(raw)
    assert len(result) == 1
    assert result[0]["category"] == "user_preference"


def test_parse_llm_response_handles_nested_arrays_via_raw_decode() -> None:
    """P1-2 fix (greedy) cubria arrays anidados. Verificamos que el
    nuevo parser (raw_decode) sigue cubriendo ese caso edge."""
    raw = json.dumps(
        [{"category": "project_context", "content": "ver [1, 2]", "confidence_score": 0.85}]
    )
    result = _parse_llm_response(raw)
    assert len(result) == 1
    assert "ver [1, 2]" in result[0]["content"]


# --- _validate_candidates ---


def test_validate_candidates_filters_invalid_category() -> None:
    raw = [
        {"category": "invalid_category", "content": "test", "confidence_score": 0.8},
        {"category": "user_preference", "content": "valid", "confidence_score": 0.8},
    ]
    result = _validate_candidates(raw)
    assert len(result) == 1
    assert result[0]["category"] == "user_preference"


def test_validate_candidates_filters_invalid_confidence() -> None:
    raw = [
        {"category": "user_preference", "content": "test1", "confidence_score": 1.5},  # > 1
        {"category": "user_preference", "content": "test2", "confidence_score": -0.1},  # < 0
        {"category": "user_preference", "content": "test3", "confidence_score": 0.8},  # valid
        {
            "category": "user_preference",
            "content": "test4",
            "confidence_score": "high",
        },  # not number
    ]
    result = _validate_candidates(raw)
    assert len(result) == 1
    assert result[0]["content"] == "test3"


def test_validate_candidates_filters_empty_content() -> None:
    raw = [
        {"category": "user_preference", "content": "", "confidence_score": 0.8},
        {"category": "user_preference", "content": None, "confidence_score": 0.8},
        {"category": "user_preference", "content": "valid", "confidence_score": 0.8},
    ]
    result = _validate_candidates(raw)
    assert len(result) == 1


def test_validate_candidates_all_valid_categories() -> None:
    """Las 3 categorias validas son aceptadas."""
    for cat in VALID_CATEGORIES:
        raw = [{"category": cat, "content": f"test {cat}", "confidence_score": 0.8}]
        result = _validate_candidates(raw)
        assert len(result) == 1
        assert result[0]["category"] == cat


# --- FactCandidate ---


def test_fact_candidate_repr() -> None:
    cand = FactCandidate(
        category="user_preference",
        content="Test content here",
        confidence_score=0.85,
    )
    r = repr(cand)
    assert "user_preference" in r
    assert "Test content" in r
    assert "0.85" in r


def test_fact_candidate_to_dict() -> None:
    cand = FactCandidate(
        category="project_context",
        content="Project info",
        confidence_score=0.9,
        source_conversation_id=42,
    )
    d = cand.to_dict()
    assert d["category"] == "project_context"
    assert d["content"] == "Project info"
    assert d["confidence_score"] == 0.9
    assert d["source_conversation_id"] == 42


# --- MemoryFactExtractor.extract_from_conversation ---


@pytest.mark.asyncio
async def test_extract_from_conversation_returns_valid_candidates(
    db: Database, mock_router: MagicMock
) -> None:
    """Extraccion basica: LLM devuelve JSON con candidates validos."""
    mock_router.chat.return_value = _FakeResponse(
        json.dumps(
            [
                {
                    "category": "user_preference",
                    "content": "User likes Python",
                    "confidence_score": 0.9,
                },
                {
                    "category": "project_context",
                    "content": "Working on Hermes",
                    "confidence_score": 0.85,
                },
            ]
        )
    )
    extractor = MemoryFactExtractor(router=mock_router, db=db)
    candidates = await extractor.extract_from_conversation(
        conversation_id=1,
        messages=[
            {"role": "user", "content": "Me gusta Python"},
            {"role": "assistant", "content": "Genial, en que proyecto?"},
            {"role": "user", "content": "Estoy con Hermes"},
        ],
    )
    assert len(candidates) == 2
    assert candidates[0].category == "user_preference"
    assert candidates[0].source_conversation_id == 1


@pytest.mark.asyncio
async def test_extract_filters_below_confidence_threshold(
    db: Database, mock_router: MagicMock
) -> None:
    """Solo candidates con confidence >= threshold se aceptan."""
    mock_router.chat.return_value = _FakeResponse(
        json.dumps(
            [
                {"category": "user_preference", "content": "low", "confidence_score": 0.3},
                {"category": "user_preference", "content": "high", "confidence_score": 0.9},
            ]
        )
    )
    extractor = MemoryFactExtractor(
        router=mock_router, db=db, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
    )
    candidates = await extractor.extract_from_conversation(
        conversation_id=1,
        messages=[{"role": "user", "content": "test"}],
    )
    # Solo el de 0.9 pasa (threshold default 0.7)
    assert len(candidates) == 1
    assert candidates[0].content == "high"


@pytest.mark.asyncio
async def test_extract_limits_max_candidates(db: Database, mock_router: MagicMock) -> None:
    """El extractor limita el numero de candidates devueltos."""
    candidates_json = [
        {
            "category": "user_preference",
            "content": f"pref {i}",
            "confidence_score": 0.9,
        }
        for i in range(20)
    ]
    mock_router.chat.return_value = _FakeResponse(json.dumps(candidates_json))
    extractor = MemoryFactExtractor(router=mock_router, db=db)
    candidates = await extractor.extract_from_conversation(
        conversation_id=1,
        messages=[{"role": "user", "content": "test"}],
    )
    assert len(candidates) <= MAX_CANDIDATES_PER_CONVERSATION


@pytest.mark.asyncio
async def test_extract_returns_empty_on_empty_messages(
    db: Database, mock_router: MagicMock
) -> None:
    """Sin mensajes, no se llama al LLM."""
    extractor = MemoryFactExtractor(router=mock_router, db=db)
    candidates = await extractor.extract_from_conversation(conversation_id=1, messages=[])
    assert candidates == []
    mock_router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_extract_raises_on_llm_error(db: Database, mock_router: MagicMock) -> None:
    """Si el LLM falla, raise (caller hace el catch).

    El extractor NO silencia errores: el caller (_process_conversation
    en SleepCycle) necesita saber si la extraccion fallo para tracking
    de metricas. El caller hace el catch y decide si continuar.
    """
    mock_router.chat.side_effect = Exception("LLM API down")
    extractor = MemoryFactExtractor(router=mock_router, db=db)
    with pytest.raises(Exception, match="LLM API down"):
        await extractor.extract_from_conversation(
            conversation_id=1,
            messages=[{"role": "user", "content": "test"}],
        )


@pytest.mark.asyncio
async def test_extract_handles_invalid_json_gracefully(
    db: Database, mock_router: MagicMock
) -> None:
    """Si el LLM devuelve texto no parseable, retorna vacio."""
    mock_router.chat.return_value = _FakeResponse("no es json {{{")
    extractor = MemoryFactExtractor(router=mock_router, db=db)
    candidates = await extractor.extract_from_conversation(
        conversation_id=1,
        messages=[{"role": "user", "content": "test"}],
    )
    assert candidates == []


# --- MemoryFactExtractor.upsert_to_staging ---


@pytest.mark.asyncio
async def test_upsert_to_staging_inserts_new_fact(db: Database, mock_router: MagicMock) -> None:
    """upsert_to_staging inserta un fact nuevo en staging."""
    extractor = MemoryFactExtractor(router=mock_router, db=db)
    cand = FactCandidate(
        category="user_preference",
        content="User likes Python",
        confidence_score=0.9,
        source_conversation_id=42,
    )
    stg_id = await extractor.upsert_to_staging(cand)
    assert stg_id.startswith("stg_")
    # Verificar que se persistio
    row = await db.get_staging_fact(stg_id)
    assert row is not None
    assert row["content"] == "User likes Python"
    assert json.loads(row["source_conversation_ids"]) == [42]


@pytest.mark.asyncio
async def test_upsert_to_staging_dedups_existing(db: Database, mock_router: MagicMock) -> None:
    """Si el fact ya existe (substring match), incrementa occurrence."""
    extractor = MemoryFactExtractor(router=mock_router, db=db)
    # Primer insert
    cand1 = FactCandidate(
        category="user_preference",
        content="User likes Python",
        confidence_score=0.9,
        source_conversation_id=1,
    )
    stg_id_1 = await extractor.upsert_to_staging(cand1)
    # Segundo insert con texto similar (substring)
    cand2 = FactCandidate(
        category="user_preference",
        content="User likes Python and Rust",
        confidence_score=0.85,
        source_conversation_id=2,
    )
    stg_id_2 = await extractor.upsert_to_staging(cand2)
    # Mismo stg_id (dedup)
    assert stg_id_1 == stg_id_2
    # Occurrence count = 2
    row = await db.get_staging_fact(stg_id_1)
    assert row["occurrence_count"] == 2
    assert json.loads(row["source_conversation_ids"]) == [1, 2]
