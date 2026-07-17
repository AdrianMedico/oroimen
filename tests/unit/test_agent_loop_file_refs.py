"""Tests Sprint 9.0: AgentLoop._resolve_file_refs.

Cubre:
- _resolve_file_refs prepende el texto del archivo al content.
- _resolve_file_refs llama db.touch_file (tracking).
- _resolve_file_refs maneja refs huerfanas (file borrado) gracefully.
- _resolve_file_refs deduplica cache: file repetido en N msgs se lee 1 vez.
- _resolve_file_refs conserva el orden (file contents + content original).
- Backward compat: msgs sin file_refs pasan tal cual.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.agent.loop import AgentLoop
from hermes.config import Settings
from hermes.memory.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


@pytest.mark.asyncio
async def test_resolve_file_refs_prepends_extracted_text(db: Database) -> None:
    """file_refs resuelto: el texto del PDF se prepende al content del msg."""
    await db.add_file(
        "file_abc",
        "paper.pdf",
        "application/pdf",
        100,
        "Section A: Lorem ipsum dolor sit amet.",
        "pypdf",
    )
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(
        conv_id,
        "user",
        "Resume el paper",
        file_refs=["file_abc"],
    )
    history = await db.get_history(conv_id)
    assert history[0]["file_refs"] is not None  # DB persistence
    # Aplicar _resolve_file_refs
    loop = AgentLoop(
        router=None,  # type: ignore[arg-type]  # no usado en _resolve
        registry=None,  # type: ignore[arg-type]
        db=db,
    )
    resolved = await loop._resolve_file_refs(history)
    content = resolved[0]["content"]
    # Sprint 19.6 F2: file content is wrapped in <file_content> tags
    assert '<file_content source="paper.pdf">' in content
    assert 'source="paper.pdf"' in content
    assert "Section A: Lorem ipsum" in content
    assert "Resume el paper" in content  # pregunta del user preservada
    # El texto del PDF va ANTES de la pregunta
    assert content.index("Section A") < content.index("Resume el paper")


@pytest.mark.asyncio
async def test_resolve_file_refs_calls_touch_file(db: Database) -> None:
    """Cada file_id resuelto dispara db.touch_file (tracking de uso)."""
    await db.add_file("file_x", "x.pdf", "application/pdf", 100, "text x", "pypdf")
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(conv_id, "user", "pregunta", file_refs=["file_x"])
    history = await db.get_history(conv_id)
    loop = AgentLoop(
        router=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        db=db,
    )
    await loop._resolve_file_refs(history)
    entry = await db.get_file("file_x")
    assert entry is not None
    assert entry["reference_count"] == 1
    assert entry["last_referenced_at"] is not None


@pytest.mark.asyncio
async def test_resolve_file_refs_handles_orphan_gracefully(db: Database) -> None:
    """File_id que no existe en DB NO raise; el comportamiento fue
    actualizado en Sprint 15 (US-3.1 §8.3).

    Antes (S9.0): skip silencioso (file content omitido, LLM respondia
    sin saber que faltaba contexto). Esto causaba respuestas del LLM
    que asumian texto que el user NO habia dado.

    Ahora (S15): inyecta un MISSING_FILE_MARKER literal para que el
    LLM sepa advertir al user. Si en el futuro queremos cambiar esto,
    basta modificar el marker o restaurar skip silencioso en
    `_resolve_file_refs`.
    """
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(
        conv_id,
        "user",
        "pregunta",
        file_refs=["file_nonexistent"],
    )
    history = await db.get_history(conv_id)
    loop = AgentLoop(
        router=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        db=db,
    )
    resolved = await loop._resolve_file_refs(history)  # no debe raise
    # Content incluye el MISSING marker + la pregunta del user preservada
    content = resolved[0]["content"]
    assert "ARCHIVO NO DISPONIBLE" in content
    assert "file_nonexistent" in content  # file_id aparece en el marker
    assert "pregunta" in content  # la pregunta del user se preserva


@pytest.mark.asyncio
async def test_resolve_file_refs_dedup_cache_in_memory(db: Database) -> None:
    """Si el mismo file aparece en 2 msgs, get_file se llama 1 sola vez."""
    await db.add_file("file_dup", "dup.pdf", "application/pdf", 100, "duplicate text", "pypdf")
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(conv_id, "user", "msg 1", file_refs=["file_dup"])
    await db.add_message(conv_id, "user", "msg 2", file_refs=["file_dup"])
    history = await db.get_history(conv_id)
    assert len(history) == 2

    # Spy: wrappear db.get_file para contar invocaciones
    original_get_file = db.get_file
    call_count = 0

    async def counted_get_file(fid: str) -> dict | None:
        nonlocal call_count
        call_count += 1
        return await original_get_file(fid)

    db.get_file = counted_get_file  # type: ignore[assignment]
    try:
        loop = AgentLoop(
            router=None,  # type: ignore[arg-type]
            registry=None,  # type: ignore[arg-type]
            db=db,
        )
        await loop._resolve_file_refs(history)
        # Cache: 1 sola invocación a db.get_file
        assert call_count == 1
    finally:
        db.get_file = original_get_file  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_resolve_file_refs_passes_through_msg_without_refs(
    db: Database,
) -> None:
    """Backward compat: msgs sin file_refs pasan tal cual (S8.7 legacy)."""
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(conv_id, "user", "pregunta normal sin files")
    history = await db.get_history(conv_id)
    assert history[0]["file_refs"] is None
    loop = AgentLoop(
        router=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        db=db,
    )
    resolved = await loop._resolve_file_refs(history)
    assert resolved[0]["content"] == "pregunta normal sin files"


@pytest.mark.asyncio
async def test_resolve_file_refs_multiple_files_in_order(db: Database) -> None:
    """2+ files en file_refs: contenido en orden, ambos prepended."""
    await db.add_file("file_1", "a.pdf", "application/pdf", 100, "A1 content", "pypdf")
    await db.add_file("file_2", "b.pdf", "application/pdf", 200, "B2 content", "pypdf")
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(
        conv_id,
        "user",
        "compara estos dos",
        file_refs=["file_1", "file_2"],
    )
    history = await db.get_history(conv_id)
    loop = AgentLoop(
        router=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        db=db,
    )
    resolved = await loop._resolve_file_refs(history)
    content = resolved[0]["content"]
    # Ambos en el content, A1 antes que B2
    assert content.index("A1 content") < content.index("B2 content")
    assert "compara estos dos" in content


@pytest.mark.asyncio
async def test_resolve_file_refs_skips_empty_extracted_text(db: Database) -> None:
    """File con extracted_text vacio (PDF corrupto): skip silencioso."""
    await db.add_file("file_empty", "empty.pdf", "application/pdf", 100, "", "pypdf")
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_message(conv_id, "user", "pregunta", file_refs=["file_empty"])
    history = await db.get_history(conv_id)
    loop = AgentLoop(
        router=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        db=db,
    )
    resolved = await loop._resolve_file_refs(history)
    # Content no incluye <file_content> wrap porque extracted_text es vacio.
    assert "<file_content" not in resolved[0]["content"]
    assert resolved[0]["content"] == "pregunta"


@pytest.mark.asyncio
async def test_resolve_file_refs_invalid_json_logs_warning(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    """file_refs con JSON invalido: log warning, msg pasa sin enriquecimiento."""
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    # Insertar directamente con JSON invalido (bypass add_message sanitization)
    async with db.conn.execute(
        "INSERT INTO messages (conversation_id, role, content, file_refs) " "VALUES (?, ?, ?, ?)",
        (conv_id, "user", "pregunta", "not valid json {{{"),
    ) as cur:
        await cur.fetchone()
    await db.conn.commit()
    history = await db.get_history(conv_id)
    loop = AgentLoop(
        router=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        db=db,
    )
    resolved = await loop._resolve_file_refs(history)  # no debe raise
    assert resolved[0]["content"] == "pregunta"
    # Verificar que se loggeo el warning
    assert any("file_refs_invalid_json" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_resolve_file_refs_orphan_injects_missing_file_marker(
    db: Database,
) -> None:
    """Sprint 15: file_ref huérfana (file borrado) inyecta MISSING marker.

    Antes (S9.0) el orphan se omitia silenciosamente y el LLM respondia
    sin saber que faltaba contexto. Ahora (Sprint 15 §8.3) inyectamos
    un marcador semántico literal tipo "⚠️ ARCHIVO NO DISPONIBLE…"
    para que el LLM pueda advertir al user en su respuesta.
    """
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    # file_ref apunta a file_id que NO existe en la DB.
    await db.add_message(conv_id, "user", "resume el PDF", file_refs=["file_does_not_exist"])
    history = await db.get_history(conv_id)
    loop = AgentLoop(
        router=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        db=db,
    )
    resolved = await loop._resolve_file_refs(history)
    content = resolved[0]["content"]
    # El MISSING marker aparece literal:
    assert "ARCHIVO NO DISPONIBLE" in content
    assert "file_does_not_exist" in content
    # La pregunta original del user se preserva al final:
    assert "resume el PDF" in content
    # El orden es: marker ANTES de la pregunta del user.
    assert content.index("ARCHIVO NO DISPONIBLE") < content.index("resume el PDF")


@pytest.mark.asyncio
async def test_resolve_file_refs_orphan_markers_share_global_budget(
    db: Database,
    settings: Settings,
) -> None:
    """Missing-file notices cannot grow injected context beyond the cap."""
    settings = settings.model_copy(update={"read_tool_max_chars": 500})
    conv_id = await db.new_conversation(chat_id=0, user_id=0)
    await db.add_file("file_real_a", "a.pdf", "application/pdf", 100, "A" * 300, "pypdf")
    await db.add_file("file_real_b", "b.pdf", "application/pdf", 100, "B" * 300, "pypdf")
    await db.add_message(
        conv_id,
        "user",
        "pregunta",
        file_refs=["file_real_a", "file_orphan_1", "file_real_b", "file_orphan_2"],
    )
    history = await db.get_history(conv_id)
    loop = AgentLoop(
        router=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        db=db,
        settings=settings,
    )

    resolved = await loop._resolve_file_refs(history)
    content = resolved[0]["content"]
    assert content.endswith("pregunta")
    injected_prefix = content[: -len("pregunta")]
    assert len(injected_prefix) <= settings.read_tool_max_chars
    assert content.count("ARCHIVO NO DISPONIBLE") <= 1
    assert "A" * 300 in content
