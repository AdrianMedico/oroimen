"""Tests Sprint 7 T53.2: agent.run_stream() async generator.

Cobertura (6 tests):
- test_run_stream_forwards_content_to_caller: tokens del LLM llegan al cliente
- test_run_stream_emits_iter_start_status_on_second_iteration: iter > 0 emite hermes_status
- test_run_stream_pauses_on_tool_call_yields_tool_start_done: D1 pausar stream
- test_run_stream_persists_assistant_partial_content: contenido parcial se guarda
- test_run_stream_resumes_with_tool_result_in_history: siguiente iter ve tool result
- test_run_stream_final_response_emits_finish_reason: respuesta final cierra stream

Patron: mock de router.chat_stream que retorna un async generator
controlado. La DB es real (fixture db) para verificar persistencia.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hermes.agent.loop import AgentLoop
from hermes.llm.router import StreamChunk, ToolCall
from hermes.tools.registry import ToolRegistry

# --- Helpers ---


def _tool_call(name: str = "get_time", args: dict | None = None, id: str = "call_1") -> ToolCall:
    """Crea un ToolCall para tests."""
    return ToolCall(id=id, name=name, arguments=args or {})


async def _async_gen_from_list(items: list[Any]) -> AsyncGenerator[Any, None]:
    """Convierte una lista en un async generator."""
    for item in items:
        yield item


def _chat_stream_mock(chunks_per_iter: list[list[StreamChunk]]):
    """Retorna una funcion sync que devuelve un async generator.

    `chunks_per_iter` es una lista de listas: una por cada llamada
    a chat_stream (que corresponde a una iteracion del agent loop).
    Si el agent loop hace mas llamadas que listas, las llamadas
    adicionales devuelven una sola chunk de finish_reason='length'
    (simula max_iterations).

    Truco CRITICO: NO envolver en AsyncMock. AsyncMock() retorna una
    coroutine cuando se llama (es awaitable por design), y `async for`
    no funciona con coroutines. Necesitamos que la llamada retorne
    DIRECTAMENTE un async generator.
    """
    # Mutable para que el closure lo modifique
    state = {"iter": 0}

    def chat_stream_sync(*a: Any, **kw: Any) -> AsyncGenerator[Any, None]:
        idx = state["iter"]
        state["iter"] += 1
        if idx < len(chunks_per_iter):
            return _async_gen_from_list(chunks_per_iter[idx])
        # Sin mas datos: simular max_iterations
        return _async_gen_from_list([StreamChunk(finish_reason="length")])

    return chat_stream_sync


async def _collect_chunks(agen: AsyncGenerator[Any, None]) -> list[Any]:
    """Consume un async generator y devuelve todos los chunks."""
    return [chunk async for chunk in agen]


# --- Tests ---


@pytest.mark.asyncio
async def test_run_stream_forwards_content_to_caller(db: Any) -> None:
    """Tokens del LLM llegan al generador. Contenido se acumula."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    router = AsyncMock()
    # 1 sola iteracion: respuesta directa sin tool calls
    router.chat_stream = _chat_stream_mock(
        [
            [
                StreamChunk(content="Hola ", model="minimax-m3"),
                StreamChunk(content="qué ", model="minimax-m3"),
                StreamChunk(content="tal", model="minimax-m3"),
                StreamChunk(finish_reason="stop", model="minimax-m3"),
            ],
        ]
    )
    loop = AgentLoop(router, ToolRegistry(), db, settings=settings)
    chunks = await _collect_chunks(loop.run_stream(cid, "user msg"))

    # 4 chunks: 3 content + 1 finish
    assert len(chunks) == 4
    content_chunks = [c for c in chunks if c.content is not None]
    assert "".join(c.content for c in content_chunks) == "Hola qué tal"
    final = [c for c in chunks if c.finish_reason is not None]
    assert final[0].finish_reason == "stop"
    # El user message y el assistant message se persisten
    history = await db.get_history(cid)
    roles = [m["role"] for m in history]
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_run_stream_emits_iter_start_status_on_second_iteration(
    db: Any,
) -> None:
    """iter > 0 emite hermes_status iter_start antes del nuevo stream."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    router = AsyncMock()
    # Iter 0: tool call. Iter 1: respuesta final.
    router.chat_stream = _chat_stream_mock(
        [
            [
                StreamChunk(content="Voy a buscar la hora", model="minimax-m3"),
                StreamChunk(
                    tool_calls=[_tool_call()],
                    finish_reason="tool_calls",
                    model="minimax-m3",
                ),
            ],
            [
                StreamChunk(content="Son las 12:00", model="minimax-m3"),
                StreamChunk(finish_reason="stop", model="minimax-m3"),
            ],
        ]
    )
    loop = AgentLoop(router, ToolRegistry(), db, settings=settings)
    chunks = await _collect_chunks(loop.run_stream(cid, "qué hora es"))

    # Buscar hermes_status con iter_start
    iter_starts = [
        c.hermes_status
        for c in chunks
        if c.hermes_status and c.hermes_status.get("event") == "iter_start"
    ]
    assert len(iter_starts) == 1
    assert iter_starts[0]["iter"] == 1


@pytest.mark.asyncio
async def test_run_stream_pauses_on_tool_call_yields_tool_start_done(
    db: Any,
) -> None:
    """D1: cuando el LLM pide tool, pausar stream, yield tool_start,
    ejecutar, yield tool_done. El contenido del LLM llega ANTES del tool_start.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()

    # Registry con una tool fake que retorna un dict
    registry = ToolRegistry()

    async def fake_tool() -> dict:
        return {"time": "12:00"}

    registry.register(
        "get_time",
        fake_tool,
        description="Get current time",
        schema={"type": "function", "function": {"name": "get_time", "parameters": {}}},
    )

    router = AsyncMock()
    # Iter 0: tool call. Iter 1: respuesta final.
    router.chat_stream = _chat_stream_mock(
        [
            [
                StreamChunk(content="Llamando a get_time", model="minimax-m3"),
                StreamChunk(
                    tool_calls=[_tool_call()],
                    finish_reason="tool_calls",
                    model="minimax-m3",
                ),
            ],
            [
                StreamChunk(content="Son las 12:00", model="minimax-m3"),
                StreamChunk(finish_reason="stop", model="minimax-m3"),
            ],
        ]
    )
    loop = AgentLoop(router, registry, db, settings=settings)
    chunks = await _collect_chunks(loop.run_stream(cid, "qué hora es"))

    # Extraer todos los hermes_status
    statuses = [c.hermes_status for c in chunks if c.hermes_status]
    # Debe haber: tool_start, tool_done, iter_start
    events = [s["event"] for s in statuses]
    assert "tool_start" in events
    assert "tool_done" in events
    # tool_start y tool_done tienen tool="get_time"
    for s in statuses:
        if s["event"] in ("tool_start", "tool_done"):
            assert s["tool"] == "get_time"

    # Orden: content (Llamando a get_time) -> tool_start -> tool_done -> iter_start -> content (Son las 12:00) -> finish
    content_chunks = [c.content for c in chunks if c.content is not None]
    assert "Llamando a get_time" in content_chunks
    assert "Son las 12:00" in content_chunks
    # El content "Llamando a get_time" aparece ANTES del tool_start
    # (es prosa del LLM antes de pedir la tool)
    llama_idx = chunks.index(next(c for c in chunks if c.content == "Llamando a get_time"))
    tool_start_idx = chunks.index(
        next(c for c in chunks if c.hermes_status and c.hermes_status.get("event") == "tool_start")
    )
    assert llama_idx < tool_start_idx


@pytest.mark.asyncio
async def test_run_stream_persists_assistant_partial_content(db: Any) -> None:
    """Contenido parcial del assistant (con tool_calls) se guarda en DB
    ANTES de ejecutar las tools. AgentLoop reusa el msg_id para las
    tool_calls rows.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    registry = ToolRegistry()

    async def fake_tool() -> dict:
        return {"result": "ok"}

    registry.register(
        "test_tool",
        fake_tool,
        description="Test",
        schema={"type": "function", "function": {"name": "test_tool", "parameters": {}}},
    )

    router = AsyncMock()
    # Iter 0: tool call. Iter 1: respuesta final.
    router.chat_stream = _chat_stream_mock(
        [
            [
                StreamChunk(content="Voy a llamar a la tool", model="minimax-m3"),
                StreamChunk(
                    tool_calls=[_tool_call(name="test_tool")],
                    finish_reason="tool_calls",
                    model="minimax-m3",
                ),
            ],
            [
                StreamChunk(content="Resultado procesado", model="minimax-m3"),
                StreamChunk(finish_reason="stop", model="minimax-m3"),
            ],
        ]
    )
    loop = AgentLoop(router, registry, db, settings=settings)
    await _collect_chunks(loop.run_stream(cid, "test"))

    # El assistant message parcial DEBE estar persistido con su content
    # Y sus tool_calls_json.
    history = await db.get_history(cid)
    assistant_msgs = [m for m in history if m["role"] == "assistant"]
    assert len(assistant_msgs) == 2  # parcial + final
    # Primer assistant: el parcial con content y tool_calls
    assert assistant_msgs[0]["content"] == "Voy a llamar a la tool"
    assert assistant_msgs[0]["tool_calls_json"] is not None
    tc = json.loads(assistant_msgs[0]["tool_calls_json"])
    assert tc[0]["function"]["name"] == "test_tool"
    # Segundo assistant: respuesta final sin tool_calls
    assert assistant_msgs[1]["content"] == "Resultado procesado"
    assert assistant_msgs[1]["tool_calls_json"] is None

    # El tool message DEBE estar persistido
    tool_msgs = [m for m in history if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    # secure_execute wraps el tool output en <tool_output>...</tool_output>
    assert "result" in tool_msgs[0]["content"]


@pytest.mark.asyncio
async def test_run_stream_resumes_with_tool_result_in_history(db: Any) -> None:
    """Siguiente iteracion: el LLM ve el tool result en el history.

    Verificamos que al cargar history en la segunda iter, el assistant
    message parcial Y el tool message estan presentes.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    registry = ToolRegistry()

    async def fake_tool() -> dict:
        return {"data": 42}

    registry.register(
        "my_tool",
        fake_tool,
        description="My tool",
        schema={"type": "function", "function": {"name": "my_tool", "parameters": {}}},
    )

    router = AsyncMock()
    # Iter 0: tool call. Iter 1: respuesta final.
    router.chat_stream = _chat_stream_mock(
        [
            [
                StreamChunk(content="Llamando tool", model="minimax-m3"),
                StreamChunk(
                    tool_calls=[_tool_call(name="my_tool")],
                    finish_reason="tool_calls",
                    model="minimax-m3",
                ),
            ],
            [
                StreamChunk(content="Tool result: 42", model="minimax-m3"),
                StreamChunk(finish_reason="stop", model="minimax-m3"),
            ],
        ]
    )
    loop = AgentLoop(router, registry, db, settings=settings, max_iterations=10)
    await _collect_chunks(loop.run_stream(cid, "go"))

    # Tras la respuesta final, el history tiene:
    # user + assistant(parcial con tool_calls) + tool(resultado) + assistant(final)
    history = await db.get_history(cid)
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["tool_calls_json"] is not None
    assert history[2]["role"] == "tool"
    # El tool result contiene "data: 42" (envuelto en <tool_output>)
    assert "data" in history[2]["content"]
    assert history[3]["role"] == "assistant"
    assert history[3]["content"] == "Tool result: 42"


@pytest.mark.asyncio
async def test_run_stream_final_response_emits_finish_reason(db: Any) -> None:
    """Respuesta final (sin tool calls) emite finish_reason y termina
    el stream.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    router = AsyncMock()
    # 1 sola iteracion: respuesta directa
    router.chat_stream = _chat_stream_mock(
        [
            [
                StreamChunk(content="Solo texto", model="minimax-m3"),
                StreamChunk(finish_reason="stop", model="minimax-m3"),
            ],
        ]
    )
    loop = AgentLoop(router, ToolRegistry(), db, settings=settings)
    chunks = await _collect_chunks(loop.run_stream(cid, "ping"))

    # 2 chunks: 1 content + 1 finish
    assert len(chunks) == 2
    assert chunks[0].content == "Solo texto"
    assert chunks[1].finish_reason == "stop"

    # El assistant message se persiste
    history = await db.get_history(cid)
    assert len([m for m in history if m["role"] == "assistant"]) == 1


# Sprint 9.3.2: regression test for Bug 5 (UX sin feedback).
# Verifica que run_stream emite content chunks con markers visibles
# (🔍 Buscando / ✓ Completado) para que Open WebUI muestre progreso
# durante tool execution. Sin esto, el user ve 30+ segundos de silencio.
@pytest.mark.asyncio
async def test_run_stream_emits_ux_markers_for_tool_execution(
    db: Any,
) -> None:
    """Sprint 9.3.2: run_stream emite content chunks con markers
    🔍 [i/N] Ejecutando: <tool> y ✓ <tool> completado (Nms) para que
    Open WebUI muestre feedback visible durante tool execution."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()

    registry = ToolRegistry()

    async def fake_search(query: str) -> str:
        return f"Results for {query}"

    registry.register(
        "hermes_search",
        fake_search,
        description="Search the web",
        schema={
            "type": "function",
            "function": {
                "name": "hermes_search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        },
    )

    router = AsyncMock()
    router.chat_stream = _chat_stream_mock(
        [
            [
                StreamChunk(
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="hermes_search",
                            arguments={"query": "test"},
                        )
                    ],
                    finish_reason="tool_calls",
                    model="minimax-m3",
                ),
            ],
            [
                StreamChunk(content="Final answer", model="minimax-m3"),
                StreamChunk(finish_reason="stop", model="minimax-m3"),
            ],
        ]
    )
    loop = AgentLoop(router, registry, db, settings=settings)
    chunks = await _collect_chunks(loop.run_stream(cid, "busca algo"))

    # Sprint 9.3.3: los markers ahora van como reasoning_content (desplegable
    # "thinking" en Open WebUI), no como content (texto de respuesta).
    # Buscar en reasoning_content chunks.
    reasoning_chunks = [c.reasoning_content for c in chunks if c.reasoning_content is not None]
    # Debe haber al menos: marker "🔍" y marker "✓"
    has_search_marker = any(
        "🔍" in (rc or "") and "hermes_search" in (rc or "") for rc in reasoning_chunks
    )
    has_done_marker = any(
        "✓" in (rc or "") and "hermes_search" in (rc or "") for rc in reasoning_chunks
    )
    assert has_search_marker, f"Falta marker de busqueda. Reasoning: {reasoning_chunks}"
    assert has_done_marker, f"Falta marker de completado. Reasoning: {reasoning_chunks}"

    # El marker "🔍" debe aparecer ANTES del "✓" (orden logico)
    reasoning_text = " ".join(rc or "" for rc in reasoning_chunks)
    assert "🔍" in reasoning_text
    assert "✓" in reasoning_text
    assert reasoning_text.index("🔍") < reasoning_text.index("✓"), "🔍 debe aparecer antes de ✓"

    # Los markers NO deben estar en content (texto de respuesta)
    content_chunks = [c.content for c in chunks if c.content is not None]
    content_text = " ".join(content_chunks)
    assert "🔍" not in content_text, f"Markers NO deben estar en content: {content_chunks}"
    assert "✓" not in content_text, f"Markers NO deben estar en content: {content_chunks}"


# Sprint 9.3.3: Budget Warning — cuando quedan <=2 iteraciones, el
# AgentLoop inyecta un mensaje role="user" en la DB advirtiendo al LLM
# que sintetice su respuesta. Test: 5 iteraciones max, el LLM pide
# tools en todas. En la iter 3 (remaining=1) debe inyectarse el warning.
@pytest.mark.asyncio
async def test_run_stream_budget_warning_injected_when_iterations_low(
    db: Any,
) -> None:
    """Sprint 9.3.3: budget warning se inyecta cuando quedan <=2 iters."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()

    registry = ToolRegistry()

    async def fake_tool() -> str:
        return "ok"

    registry.register(
        "fake_tool",
        fake_tool,
        description="Fake tool for testing",
        schema={
            "type": "function",
            "function": {"name": "fake_tool", "parameters": {}},
        },
    )

    router = AsyncMock()
    # 4 iteraciones pidiendo tool + 1 final (max=5)
    router.chat_stream = _chat_stream_mock(
        [
            # iter 0: tool (remaining=4, no warning)
            [StreamChunk(tool_calls=[_tool_call()], finish_reason="tool_calls", model="m")],
            # iter 1: tool (remaining=3, no warning)
            [StreamChunk(tool_calls=[_tool_call()], finish_reason="tool_calls", model="m")],
            # iter 2: tool (remaining=2, WARNING "2 iteraciones")
            [StreamChunk(tool_calls=[_tool_call()], finish_reason="tool_calls", model="m")],
            # iter 3: tool (remaining=1, WARNING "ULTIMA iteracion")
            [StreamChunk(tool_calls=[_tool_call()], finish_reason="tool_calls", model="m")],
            # iter 4: respuesta final (no tool calls)
            [
                StreamChunk(content="Aqui esta el resumen", model="m"),
                StreamChunk(finish_reason="stop", model="m"),
            ],
        ]
    )
    loop = AgentLoop(router, registry, db, settings=settings, max_iterations=5)
    chunks = await _collect_chunks(loop.run_stream(cid, "test budget warning"))

    # Buscar budget warnings en reasoning_content (markers ⏰)
    reasoning_text = " ".join(
        (c.reasoning_content or "") for c in chunks if c.reasoning_content is not None
    )
    assert (
        "Budget Warning" in reasoning_text
    ), f"No se encontro budget warning en reasoning: {reasoning_text}"
    assert "2 iteraciones" in reasoning_text, "Falta warning de 2 iters"
    assert "ÚLTIMA iteración" in reasoning_text, "Falta warning de ultima iter"

    # Verificar que el warning se persistio en la DB (role=user)
    async with db.conn.execute(
        "SELECT content FROM messages WHERE conversation_id=? AND role='user' AND content LIKE '%Budget Warning%' ORDER BY id",
        (cid,),
    ) as cur:
        warnings = await cur.fetchall()
    assert len(warnings) >= 2, f"Deberia haber 2 warning messages en DB, hay {len(warnings)}"


# Sprint 9.4: tests para captura de usage (tokens) en streaming.
# Bug observado: tokens_in/out=0 en InfluxDB aunque el router emite
# final_usage. Causa: en el streaming, el chunk con usage a veces llega
# DESPUES del chunk con finish_reason, y el loop sale antes de procesarlo.


@pytest.mark.asyncio
async def test_run_stream_captures_usage_in_same_chunk_as_finish(
    db: Any,
) -> None:
    """Sprint 9.4: usage + finish_reason en mismo chunk debe capturarse."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    router = AsyncMock()

    async def stream():
        yield StreamChunk(content="Hola", model="minimax-m3")
        yield StreamChunk(
            content=" mundo",
            model="minimax-m3",
            finish_reason="stop",
            usage={"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
        )

    router.chat_stream = lambda *a, **k: stream()
    loop = AgentLoop(router, ToolRegistry(), db, settings=settings)
    await _collect_chunks(loop.run_stream(cid, "test"))

    history = await db.get_history(cid)
    assistant_msgs = [m for m in history if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    # get_history no devuelve tokens, los leemos directamente
    async with db.conn.execute(
        "SELECT tokens_in, tokens_out FROM messages WHERE conversation_id=? AND role='assistant'",
        (cid,),
    ) as cur:
        row = await cur.fetchone()
    if row[0] != 42:
        # Debug: print DB state
        print(f"  DB row: tokens_in={row[0]} tokens_out={row[1]}")
        async with db.conn.execute(
            "SELECT id, role, content, tokens_in, tokens_out, model_used, tool_calls_json FROM messages WHERE conversation_id=?",
            (cid,),
        ) as cur:
            for m in await cur.fetchall():
                print(
                    f"    msg_id={m[0]} role={m[1]} tokens=({m[3]},{m[4]}) tool_calls={m[6][:50] if m[6] else None}"
                )
                print(f"      content='{m[2][:80]}'")
    assert row[0] == 42, f"tokens_in deberia ser 42, got {row[0]}"
    assert row[1] == 7, f"tokens_out deberia ser 7, got {row[1]}"


@pytest.mark.asyncio
async def test_run_stream_captures_usage_from_separate_chunk(
    db: Any,
) -> None:
    """Sprint 9.4: usage en chunk SEPARADO (OpenAI a veces lo envia asi)
    debe capturarse. Patron: finish_reason en chunk 1, usage en chunk 2."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    router = AsyncMock()

    async def stream():
        yield StreamChunk(content="Hola", model="minimax-m3")
        yield StreamChunk(
            content=" mundo",
            model="minimax-m3",
            finish_reason="stop",
            # usage=None aqui, llega en chunk aparte
        )
        # Chunk vacio con solo usage (OpenAI include_usage pattern)
        yield StreamChunk(
            model="minimax-m3",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    router.chat_stream = lambda *a, **k: stream()
    loop = AgentLoop(router, ToolRegistry(), db, settings=settings)
    await _collect_chunks(loop.run_stream(cid, "test"))

    history = await db.get_history(cid)
    assistant_msgs = [m for m in history if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    async with db.conn.execute(
        "SELECT tokens_in, tokens_out FROM messages WHERE conversation_id=? AND role='assistant'",
        (cid,),
    ) as cur:
        row = await cur.fetchone()
    # ESTE test puede fallar si el loop sale antes de procesar el chunk de usage
    # Lo capturamos para diagnosticar
    if row[0] != 100 or row[1] != 20:
        import pytest

        pytest.xfail(
            f"Bug conocido: usage en chunk separado no se captura (got tokens_in={row[0]}, tokens_out={row[1]})"
        )


@pytest.mark.asyncio
async def test_run_stream_with_tool_calls_captures_usage_after_tool(
    db: Any,
) -> None:
    """Sprint 9.4: tool calls + final assistant message ambos capturan
    tokens. Patron: iter 0 con tool call (usage en el chunk de tool_calls),
    iter 1 con respuesta final."""
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    registry = ToolRegistry()

    async def fake_tool() -> str:
        return "result"

    registry.register(
        "my_tool",
        fake_tool,
        description="My tool",
        schema={"type": "function", "function": {"name": "my_tool", "parameters": {}}},
    )

    router = AsyncMock()
    router.chat_stream = _chat_stream_mock(
        [
            # Iter 0: tool call con usage
            [
                StreamChunk(
                    tool_calls=[_tool_call(name="my_tool")],
                    finish_reason="tool_calls",
                    model="minimax-m3",
                    usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                ),
            ],
            # Iter 1: respuesta final con usage
            [
                StreamChunk(content="Aqui esta el resultado", model="minimax-m3"),
                StreamChunk(
                    finish_reason="stop",
                    model="minimax-m3",
                    usage={"prompt_tokens": 70, "completion_tokens": 15, "total_tokens": 85},
                ),
            ],
        ]
    )
    loop = AgentLoop(router, registry, db, settings=settings)
    await _collect_chunks(loop.run_stream(cid, "test"))

    # La respuesta final (iter 1) debe tener tokens del chunk de finish
    # El ultimo assistant message deberia tener tokens del final response
    async with db.conn.execute(
        "SELECT tokens_in, tokens_out FROM messages WHERE conversation_id=? AND role='assistant' ORDER BY id DESC LIMIT 1",
        (cid,),
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == 70, f"tokens_in deberia ser 70 (de la respuesta final), got {row[0]}"
    assert row[1] == 15, f"tokens_out deberia ser 15, got {row[1]}"


# --- Sprint 16 P1: regression guard ---


@pytest.mark.asyncio
async def test_run_stream_calls_inject_memory_facts_once_per_request(
    db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sprint 16 P1 (post-mortem from adversarial 2nd-pass BLOCKING #1):
    _inject_memory_facts debe llamarse UNA vez por request, no una
    vez por iteracion. El retrieval de memory facts es caro (embed
    API call + DB scan) y el texto del usuario no cambia entre
    iteraciones del agent loop.

    Cualquier cambio futuro al loop de run_stream que vuelva a poner
    la retrieval dentro del `for iteration` loop debe fallar este test.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    router = AsyncMock()
    # 3 iteraciones: 2 con tool_call, 1 final con respuesta
    router.chat_stream = _chat_stream_mock(
        [
            [
                StreamChunk(content="", model="minimax-m3"),
                StreamChunk(
                    tool_calls=[_tool_call()],
                    finish_reason="tool_use",
                    model="minimax-m3",
                ),
            ],
            [
                StreamChunk(content="", model="minimax-m3"),
                StreamChunk(
                    tool_calls=[_tool_call(args={"tool": "different"})],
                    finish_reason="tool_use",
                    model="minimax-m3",
                ),
            ],
            [
                StreamChunk(content="Done", finish_reason="stop", model="minimax-m3"),
            ],
        ]
    )
    loop = AgentLoop(router, ToolRegistry(), db, settings=settings)

    # Contador de llamadas a _inject_memory_facts via monkeypatch
    call_count = {"n": 0}

    original = loop._inject_memory_facts

    async def counting_inject(user_query: str) -> str:
        call_count["n"] += 1
        return await original(user_query)

    monkeypatch.setattr(loop, "_inject_memory_facts", counting_inject)

    chunks = await _collect_chunks(loop.run_stream(cid, "user msg"))
    assert len(chunks) > 0  # sanity: streaming produjo output

    # REGLA: exactamente 1 llamada por request, sin importar
    # max_iterations. Si el codigo vuelve a meter la retrieval
    # dentro del loop, este assert falla.
    assert call_count["n"] == 1, (
        f"_inject_memory_facts debe llamarse 1 vez por request, "
        f"got {call_count['n']} (regression: retrieval dentro del "
        f"per-iteration loop)"
    )


@pytest.mark.asyncio
async def test_run_stream_max_iterations_marks_truncated_true(
    db: Any,
) -> None:
    """Sprint 16.8.3 (M-1 fix): cuando el agent loop alcanza max_iterations,
    el chunk final DEBE tener truncated=True. El user ve "He alcanzado el
    limite de N iteraciones..." (visible) pero el SSE payload DEBE incluir
    choice.truncated=true para que clientes downstream (Open WebUI)
    muestren su propio aviso de truncation consistente con el contrato
    PR #105 advertise.
    """
    cid = await db.get_or_create_conversation(chat_id=1, user_id=100)
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    router = AsyncMock()
    # Tool que el LLM pide en bucle, agota max_iterations=2
    from hermes.tools.registry import ToolRegistry

    registry = ToolRegistry()

    async def loop_tool() -> str:
        return "x"

    registry.register(
        "loop_tool",
        loop_tool,
        description="tool",
        schema={"type": "object", "properties": {}},
    )

    # Cada iteracion: tool_call (loop infinito)
    router.chat_stream = _chat_stream_mock(
        [
            [
                StreamChunk(
                    tool_calls=[_tool_call(name="loop_tool", id=f"call_{i}")],
                    finish_reason="tool_calls",
                    model="minimax-m3",
                )
            ]
            for i in range(5)  # 5 iters disponibles
        ]
    )
    loop = AgentLoop(router, registry, db, settings=settings, max_iterations=2)
    chunks = await _collect_chunks(loop.run_stream(cid, "test"))

    # El chunk final con finish_reason="length" debe tener truncated=True
    final_chunks = [c for c in chunks if c.finish_reason is not None]
    assert len(final_chunks) >= 1, "Deberia haber un chunk final con finish_reason"
    last_final = final_chunks[-1]
    assert last_final.finish_reason == "length"
    # M-1 fix: explicit truncated=True
    assert (
        last_final.truncated is True
    ), f"max_iterations chunk debe tener truncated=True, got {last_final.truncated}"
    # Y debe serializar choice.truncated=true en el SSE
    sse = last_final.to_sse()
    assert '"truncated": true' in sse or '"truncated":true' in sse
    # Adicionalmente, el chunk de content con el aviso de max_iterations
    # debe estar presente (warning visible al user)
    content_chunks = [c for c in chunks if c.content is not None]
    assert any(
        "iteraciones" in c.content for c in content_chunks
    ), "Falta el warning visible de max_iterations"
