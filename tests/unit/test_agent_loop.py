"""Tests del AgentLoop (Sprint 4 T1).

Cubre:
- Sin tools: 1 LLM call, devuelve content directo
- Con tools: ejecuta tool, llama LLM de nuevo con resultado
- max_iterations: aborta con error si el LLM pide tools en bucle
- Persistencia: tool_calls guardados en DB con arguments/result/success
- Errores de tool: se loggean y el loop continúa
- OpenAI tool_calls format: parsing correcto
- Anthropic tool_use format: parsing correcto
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from hermes.agent.loop import AgentLoop
from hermes.llm.router import LLMRouter, ToolCall
from hermes.memory.db import Database
from hermes.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "agent.db")
    await d.initialize()
    yield d
    await d.close()


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def sample_tool_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    }


# ---------------------------------------------------------------------------
# Test: caso simple sin tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_calls_llm_once_for_simple_prompt(
    db: Database, registry: ToolRegistry, settings: Any, respx_mock: Any
) -> None:
    """Sin tools registradas, 1 LLM call, retorna content directo."""
    # El LLM responde sin tool_calls
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    respx_mock.post(openai_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hola!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )
    )

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=100, user_id=1)
    try:
        result = await loop.run(conv_id, "Hola")
        assert result == "Hola!"
    finally:
        await router.aclose()

    # Solo 1 call HTTP
    assert len(respx_mock.calls) == 1

    # History guardada: user + assistant
    history = await db.get_history(conv_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Test: tool call simple
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_call_and_calls_llm_again(
    db: Database,
    registry: ToolRegistry,
    settings: Any,
    respx_mock: Any,
    sample_tool_schema: dict,
) -> None:
    """LLM pide tool, ejecuta, llama LLM de nuevo con resultado."""

    # Tool: get_weather que devuelve "22 grados, soleado"
    async def get_weather(city: str) -> str:
        return f"{city}: 22 grados, soleado"

    registry.register(
        "get_weather",
        get_weather,
        description="Get current weather for a city",
        schema=sample_tool_schema,
    )

    openai_url = f"{settings.opencode_go_base_url}/chat/completions"

    # 1ª respuesta: LLM pide tool
    respx_mock.post(openai_url).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "Madrid"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            ),
            # 2ª respuesta: LLM usa el resultado
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "En Madrid hace 22 grados."}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
                },
            ),
        ]
    )

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, settings=settings, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=101, user_id=1)
    try:
        result = await loop.run(conv_id, "¿Qué tiempo hace en Madrid?")
        assert result == "En Madrid hace 22 grados."
    finally:
        await router.aclose()

    # 2 calls HTTP (tool call + final)
    assert len(respx_mock.calls) == 2

    # Tool call persistido
    tcs = await db.list_tool_calls_for_conversation(conv_id)
    assert len(tcs) == 1
    assert tcs[0]["tool_name"] == "get_weather"
    assert json.loads(tcs[0]["arguments_json"]) == {"city": "Madrid"}
    # Sprint 5 T49: el result_json ahora viene envuelto en <tool_output>
    # por secure_execute (defense in depth centralizado)
    assert "Madrid: 22 grados, soleado" in tcs[0]["result_json"]
    assert "<tool_output" in tcs[0]["result_json"]
    assert tcs[0]["success"] == 1

    # History: user + assistant(tool_call) + tool + assistant(final)
    history = await db.get_history(conv_id)
    roles = [h["role"] for h in history]
    assert roles == ["user", "assistant", "tool", "assistant"]


# ---------------------------------------------------------------------------
# Test: max_iterations limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_max_iterations_limit(
    db: Database,
    registry: ToolRegistry,
    settings: Any,
    respx_mock: Any,
    sample_tool_schema: dict,
) -> None:
    """Si el LLM pide tools en bucle, aborta tras max_iterations."""

    async def useless_tool() -> str:
        return "useless"

    registry.register(
        "useless",
        useless_tool,
        description="Useless tool",
        schema={"type": "object", "properties": {}},
    )

    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    # El LLM pide la tool en cada iteración
    respx_mock.post(openai_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_x",
                                    "type": "function",
                                    "function": {"name": "useless", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=102, user_id=1)
    try:
        result = await loop.run(conv_id, "loop test")
        # Acepta el mensaje de error friendly
        assert "3 iteraciones" in result or "intenta" in result.lower()
    finally:
        await router.aclose()

    # 3 calls HTTP (max_iterations)
    assert len(respx_mock.calls) == 3

    # 3 tool_calls persistidos
    tcs = await db.list_tool_calls_for_conversation(conv_id)
    assert len(tcs) == 3


# ---------------------------------------------------------------------------
# Test: tool error se loggea y no rompe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_handles_tool_error_gracefully(
    db: Database,
    registry: ToolRegistry,
    settings: Any,
    respx_mock: Any,
    sample_tool_schema: dict,
) -> None:
    """Si una tool lanza excepción, el loop loggea y el LLM recibe el error."""

    async def broken_tool(city: str) -> str:
        raise ValueError(f"City '{city}' not found")

    registry.register(
        "broken",
        broken_tool,
        description="A broken tool",
        schema=sample_tool_schema,
    )

    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    respx_mock.post(openai_url).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_err",
                                        "type": "function",
                                        "function": {
                                            "name": "broken",
                                            "arguments": json.dumps({"city": "Atlantis"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "No pude encontrar Atlantis."}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            ),
        ]
    )

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=103, user_id=1)
    try:
        result = await loop.run(conv_id, "busca Atlantis")
        # El LLM recibe el error como tool result y responde accordingly
        assert "Atlantis" in result
    finally:
        await router.aclose()

    # Tool call persistido con success=0
    tcs = await db.list_tool_calls_for_conversation(conv_id)
    assert len(tcs) == 1
    assert tcs[0]["success"] == 0
    assert tcs[0]["error"] is not None
    assert "Atlantis" in tcs[0]["error"] or "not found" in tcs[0]["error"]


# ---------------------------------------------------------------------------
# Test: schema format OpenAI en payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_sends_openai_tools_format(
    db: Database,
    registry: ToolRegistry,
    settings: Any,
    respx_mock: Any,
    sample_tool_schema: dict,
) -> None:
    """El payload al LLM incluye `tools` con formato OpenAI."""

    async def my_tool(city: str) -> str:
        return "ok"

    registry.register(
        "my_tool",
        my_tool,
        description="Test tool",
        schema=sample_tool_schema,
    )

    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    captured_payloads: list[dict] = []

    def cb(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    respx_mock.post(openai_url).mock(side_effect=cb)

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=104, user_id=1)
    try:
        await loop.run(conv_id, "test")
    finally:
        await router.aclose()

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert "tools" in payload
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Test tool",
                "parameters": sample_tool_schema,
            },
        }
    ]


# ---------------------------------------------------------------------------
# Test: sin tools registradas, no envía `tools` en payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_no_tools_omits_tools_field(
    db: Database, registry: ToolRegistry, settings: Any, respx_mock: Any
) -> None:
    """Si no hay tools, el payload NO incluye el campo `tools`."""
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    captured_payloads: list[dict] = []

    def cb(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    respx_mock.post(openai_url).mock(side_effect=cb)

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=105, user_id=1)
    try:
        await loop.run(conv_id, "test")
    finally:
        await router.aclose()

    assert "tools" not in captured_payloads[0]


# ---------------------------------------------------------------------------
# Test: tool no registrada → error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_handles_unknown_tool_error(
    db: Database, registry: ToolRegistry, settings: Any, respx_mock: Any
) -> None:
    """Si el LLM pide una tool no registrada, se loggea como error."""
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    respx_mock.post(openai_url).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_unk",
                                        "type": "function",
                                        "function": {
                                            "name": "unknown_tool",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "No tengo esa tool."}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
        ]
    )

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=106, user_id=1)
    try:
        result = await loop.run(conv_id, "test")
        # El LLM recibe el KeyError como tool result
        assert "tool" in result.lower() or "No tengo" in result
    finally:
        await router.aclose()

    # Tool call registrado con success=0
    tcs = await db.list_tool_calls_for_conversation(conv_id)
    assert len(tcs) == 1
    assert tcs[0]["tool_name"] == "unknown_tool"
    assert tcs[0]["success"] == 0


# ---------------------------------------------------------------------------
# Test: Anthropic tool_use format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_parses_anthropic_tool_use(
    db: Database,
    registry: ToolRegistry,
    settings_v12: Any,  # settings con minimax-m3 primary
    respx_mock: Any,
    sample_tool_schema: dict,
) -> None:
    """Anthropic format: bloques tool_use en content.

    v0.5.7-revert: usamos qwen3.7-plus en vez de minimax-m3 porque
    minimax-m3 con tools va por path OpenAI (workaround bug 2013).
    qwen3.7-plus sigue usando path Anthropic, que es lo que este
    test quiere validar (parsing de tool_use blocks en formato Anthropic).
    """
    # Override primary a qwen3.7-plus (Anthropic path) para este test
    settings_v12.llm_text_primary = "qwen3.7-plus"

    async def my_tool(city: str) -> str:
        return f"{city}: soleado"

    registry.register(
        "my_tool",
        my_tool,
        description="Test",
        schema=sample_tool_schema,
    )

    anthropic_url = f"{settings_v12.opencode_go_base_url}/messages"
    respx_mock.post(anthropic_url).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "my_tool",
                            "input": {"city": "Madrid"},
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            ),
            httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "Madrid: soleado"}],
                    "usage": {"input_tokens": 15, "output_tokens": 8},
                },
            ),
        ]
    )

    router = LLMRouter(settings_v12)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=107, user_id=1)
    try:
        result = await loop.run(conv_id, "test")
        assert "Madrid" in result
    finally:
        await router.aclose()

    # Tool call persistido
    tcs = await db.list_tool_calls_for_conversation(conv_id)
    assert len(tcs) == 1
    assert tcs[0]["tool_name"] == "my_tool"
    assert json.loads(tcs[0]["arguments_json"]) == {"city": "Madrid"}


# ---------------------------------------------------------------------------
# Test: LLM error → mensaje friendly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_returns_friendly_message_on_llm_error(
    db: Database, registry: ToolRegistry, settings: Any, respx_mock: Any
) -> None:
    """Si el LLM falla, devuelve mensaje user-friendly en vez de raise."""
    # Deshabilitar fallback chain para que el 500 sea el último error.
    # Si dejamos el fallback, el router intenta el segundo modelo (que
    # también puede fallar). Para test unitario, queremos forzar el path
    # "todos los modelos del chain fallan" → LLMError.
    settings.llm_max_retries = 0  # sin retries
    # Mockear AMBOS endpoints del chain (deepseek + minimax) con 500
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    anthropic_url = f"{settings.opencode_go_base_url}/messages"
    respx_mock.post(openai_url).mock(return_value=httpx.Response(500))
    respx_mock.post(anthropic_url).mock(return_value=httpx.Response(500))

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=108, user_id=1)
    try:
        result = await loop.run(conv_id, "test")
        assert "error" in result.lower() or "inténtalo" in result.lower()
    finally:
        await router.aclose()


# ---------------------------------------------------------------------------
# T9: Thinking en tiempo real via step_callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_callback_invoked_on_each_step(
    db: Database,
    registry: ToolRegistry,
    settings: Any,
    respx_mock: Any,
    sample_tool_schema: dict,
) -> None:
    """El step_callback se invoca con cada paso del agent loop."""

    async def my_tool(city: str) -> str:
        return f"{city}: ok"

    registry.register("my_tool", my_tool, description="x", schema=sample_tool_schema)

    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    respx_mock.post(openai_url).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "my_tool",
                                            "arguments": json.dumps({"city": "Madrid"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "done"}}], "usage": {}},
            ),
        ]
    )

    step_messages: list[str] = []

    async def my_callback(msg: str) -> None:
        step_messages.append(msg)

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3, step_callback=my_callback)
    conv_id = await db.new_conversation(chat_id=200, user_id=1)
    try:
        await loop.run(conv_id, "test")
    finally:
        await router.aclose()

    # El callback debe haber sido invocado al menos 4 veces:
    # 1. Pensando inicial
    # 2. Analizando (iter 1+, v0.5.7-t49.2)
    # 3. Llamando my_tool
    # 4. my_tool completado
    assert len(step_messages) >= 4, f"Pocos callbacks ({len(step_messages)}): {step_messages!r}"
    assert any("🧠 Pensando" in m for m in step_messages), step_messages
    assert any("🧠 Analizando" in m for m in step_messages), step_messages
    assert any("🔧 Llamando" in m for m in step_messages), step_messages
    assert any("my_tool" in m for m in step_messages), step_messages
    assert any("✅" in m and "completado" in m for m in step_messages), step_messages
    # v0.5.7-t49.2: el "iter X/Y" se quitó (era ruido, se throttleaba
    # inmediatamente y nunca se mostraba)
    assert not any("iter" in m for m in step_messages), step_messages


@pytest.mark.asyncio
async def test_step_callback_silences_exceptions(
    db: Database, registry: ToolRegistry, settings: Any, respx_mock: Any
) -> None:
    """Si el callback lanza, el agent loop NO debe romperse."""
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    respx_mock.post(openai_url).mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )
    )

    async def broken_callback(msg: str) -> None:
        raise RuntimeError("callback broken")

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3, step_callback=broken_callback)
    conv_id = await db.new_conversation(chat_id=201, user_id=1)
    try:
        result = await loop.run(conv_id, "test")
        # El loop termina OK pese a que el callback falla
        assert result == "ok"
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_step_callback_optional(
    db: Database, registry: ToolRegistry, settings: Any, respx_mock: Any
) -> None:
    """AgentLoop funciona sin step_callback (default None)."""
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    respx_mock.post(openai_url).mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )
    )

    router = LLMRouter(settings)
    # Sin step_callback
    loop = AgentLoop(router, registry, db, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=202, user_id=1)
    try:
        result = await loop.run(conv_id, "test")
        assert result == "ok"
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_step_callback_can_be_set_via_attribute_after_init(
    db: Database, registry: ToolRegistry, settings: Any, respx_mock: Any
) -> None:
    """v0.5.7-t50.2: el step_callback puede asignarse al agent_loop
    DESPUES de su creacion. Esto es lo que hace messages.py: define
    el callback dentro de handle_message() y lo asigna al agent_loop
    antes de llamar a run().

    Antes (bug v0.5.7-t50): el callback se pasaba como parametro del
    constructor y quedaba como None. _report era un no-op y el user
    solo veia "Pensando..." para siempre.
    """
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    respx_mock.post(openai_url).mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )
    )

    step_calls: list[str] = []

    async def my_callback(status: str) -> None:
        step_calls.append(status)

    # Crear el loop SIN step_callback (como hace messages.py)
    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    assert loop._step_callback is None  # default

    # Asignar el callback despues (como hace messages.py:267-272)
    loop._step_callback = my_callback
    assert loop._step_callback is my_callback

    # run() debe invocar el callback
    conv_id = await db.new_conversation(chat_id=203, user_id=1)
    try:
        await loop.run(conv_id, "test")
    finally:
        await router.aclose()

    # El callback fue llamado al menos 1 vez ("Pensando...")
    assert step_calls, "step_callback no fue llamado"
    assert step_calls[0] == "🧠 Pensando…"


# ---------------------------------------------------------------------------
# Tests de _build_llm_messages (Sprint 4 v0.4.3)
#
# El agente reconstruye el payload desde la DB. Si el assistant pidió
# tool_calls, el mensaje debe llevar tool_calls (formato OpenAI) y content=None.
# Si el rol es tool, debe llevar tool_call_id. Esto es lo que el LLM espera
# para poder "cerrar el bucle" del tool_use ↔ tool_result.
# ---------------------------------------------------------------------------


def test_build_llm_messages_passes_through_user_and_assistant_text() -> None:
    """Mensajes user/assistant sin tools se pasan tal cual."""
    history = [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "hola, qué tal?"},
    ]
    from hermes.agent.loop import AgentLoop

    result = AgentLoop._build_llm_messages(history)
    assert result == [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "hola, qué tal?"},
    ]


def test_build_llm_messages_includes_tool_call_id_for_tool_role() -> None:
    """Mensajes role='tool' deben llevar tool_call_id (requerido por OpenAI/Anthropic)."""
    history = [
        {"role": "user", "content": "qué hora es"},
        {"role": "assistant", "content": "", "tool_calls_json": None},
        {"role": "tool", "content": "2026-06-22T17:30:00+02:00", "tool_call_id": "call_abc123"},
    ]
    from hermes.agent.loop import AgentLoop

    result = AgentLoop._build_llm_messages(history)
    assert len(result) == 3
    assert result[2] == {
        "role": "tool",
        "content": "2026-06-22T17:30:00+02:00",
        "tool_call_id": "call_abc123",
    }


def test_build_llm_messages_reconstructs_assistant_tool_calls() -> None:
    """Assistant con tool_calls_json se reconstruye con tool_calls + content=None."""
    tc_json = json.dumps(
        [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "arguments": "{}",
                },
            }
        ]
    )
    history = [
        {"role": "user", "content": "qué hora es"},
        {"role": "assistant", "content": "", "tool_calls_json": tc_json},
    ]
    from hermes.agent.loop import AgentLoop

    result = AgentLoop._build_llm_messages(history)
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] is None  # OpenAI requiere null
    assert result[1]["tool_calls"] == [
        {
            "id": "call_xyz",
            "type": "function",
            "function": {"name": "get_current_time", "arguments": "{}"},
        }
    ]


def test_build_llm_messages_assistant_with_text_and_tool_calls() -> None:
    """Assistant con content de texto + tool_calls se reconstruye con ambos."""
    tc_json = json.dumps(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Madrid"}'},
            }
        ]
    )
    history = [
        {"role": "assistant", "content": "Voy a consultar el tiempo", "tool_calls_json": tc_json},
    ]
    from hermes.agent.loop import AgentLoop

    result = AgentLoop._build_llm_messages(history)
    assert result[0]["content"] == "Voy a consultar el tiempo"
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["id"] == "call_1"


def test_build_llm_messages_handles_invalid_tool_calls_json() -> None:
    """Si tool_calls_json está corrupto, fallback a mensaje sin tool_calls."""
    history = [
        {"role": "assistant", "content": "hola", "tool_calls_json": "{invalid json"},
    ]
    from hermes.agent.loop import AgentLoop

    result = AgentLoop._build_llm_messages(history)
    assert result[0] == {"role": "assistant", "content": "hola"}


# --- Sprint 5 T51: reasoning_content passthrough ---


def test_build_llm_messages_includes_reasoning_content() -> None:
    """Assistant con reasoning_content lo inyecta en el payload LLM.

    Sprint 5 T51: necesario para round-trip con deepseek-v4-flash
    (OpenAI path) que exige reenviar el thinking mode en iteraciones
    siguientes tras un tool_call.
    """
    history = [
        {
            "role": "user",
            "content": "Resume el video",
            "tool_call_id": None,
            "tool_calls_json": None,
            "reasoning_content": None,
        },
        {
            "role": "assistant",
            "content": "Te resumo el video",
            "tool_call_id": None,
            "tool_calls_json": None,
            "reasoning_content": "Pensamiento largo de DeepSeek...",
        },
        {
            "role": "tool",
            "content": "<tool_output>...</tool_output>",
            "tool_call_id": "call_123",
            "tool_calls_json": None,
            "reasoning_content": None,
        },
        {
            "role": "user",
            "content": "Gracias",
            "tool_call_id": None,
            "tool_calls_json": None,
            "reasoning_content": None,
        },
    ]
    result = AgentLoop._build_llm_messages(history)
    assert len(result) == 4
    # El assistant message debe incluir reasoning_content
    asst_msg = result[1]
    assert asst_msg["role"] == "assistant"
    assert asst_msg["content"] == "Te resumo el video"
    assert asst_msg["reasoning_content"] == "Pensamiento largo de DeepSeek..."
    # Los user/tool NO deben tener reasoning_content
    assert "reasoning_content" not in result[0]
    assert "reasoning_content" not in result[2]
    assert "reasoning_content" not in result[3]


def test_build_llm_messages_omits_empty_reasoning_content() -> None:
    """Assistant sin reasoning_content no añade el campo al payload.

    Pre-migracion (rc=NULL en DB) o providers sin thinking mode
    (Anthropic). El campo NO debe aparecer en el payload para
    no contaminar providers que no lo esperan.
    """
    history = [
        {
            "role": "user",
            "content": "Hola",
            "tool_call_id": None,
            "tool_calls_json": None,
            "reasoning_content": None,
        },
        {
            "role": "assistant",
            "content": "Hola, en que puedo ayudarte?",
            "tool_call_id": None,
            "tool_calls_json": None,
            "reasoning_content": None,  # Pre-migracion o Anthropic
        },
    ]
    result = AgentLoop._build_llm_messages(history)
    assert len(result) == 2
    asst_msg = result[1]
    assert asst_msg == {"role": "assistant", "content": "Hola, en que puedo ayudarte?"}
    assert "reasoning_content" not in asst_msg


def test_build_llm_messages_tool_with_missing_call_id_defaults_empty() -> None:
    """Si role='tool' no tiene tool_call_id (mensaje legacy), usar string vacío
    para no romper el payload. El provider rechazará el tool_use_id que
    no coincida, pero al menos el payload será válido."""
    history = [
        {"role": "tool", "content": "resultado", "tool_call_id": None},
    ]
    from hermes.agent.loop import AgentLoop

    result = AgentLoop._build_llm_messages(history)
    assert result[0]["tool_call_id"] == ""


# Sprint 9.3.2: regression test for Bug 1 (latency_ms=0 en tool_calls).
# Verifica que _execute_tool_call_v2 persiste latency_ms correctamente.
# Antes del fix, add_tool_call() recibia todo menos latency_ms, por lo
# que la columna quedaba en 0 incluso cuando secure_execute media el
# tiempo real de ejecucion.
@pytest.mark.asyncio
async def test_execute_tool_call_v2_persists_latency_ms(
    db: Database,
    registry: ToolRegistry,
    settings: Any,
    respx_mock: Any,
    sample_tool_schema: dict,
) -> None:
    """Sprint 9.3.2: _execute_tool_call_v2 persiste latency_ms > 0 en tool_calls row."""
    import asyncio

    async def slow_tool(city: str) -> str:
        # Sleep para garantizar que latency_ms > 0
        await asyncio.sleep(0.05)
        return f"Result for {city}"

    registry.register(
        "slow_tool",
        slow_tool,
        description="Tool que duerme 50ms para forzar latency visible",
        schema=sample_tool_schema,
    )

    # Mock LLM: 1 tool call + 1 final
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    respx_mock.post(openai_url).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_latency",
                                        "type": "function",
                                        "function": {
                                            "name": "slow_tool",
                                            "arguments": json.dumps({"city": "Madrid"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "done"}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3},
                },
            ),
        ]
    )

    # Path: usar _execute_tool_call_v2 via run() (el test path normal)
    from hermes.agent.loop import AgentLoop
    from hermes.llm.router import LLMRouter

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, settings=settings, max_iterations=3)
    conv_id = await db.new_conversation(chat_id=999, user_id=1)
    try:
        await loop.run(conv_id, "test latency")
    finally:
        await router.aclose()

    # Verificar que latency_ms > 0 en el tool_calls row
    tcs = await db.list_tool_calls_for_conversation(conv_id)
    assert len(tcs) == 1
    tc = tcs[0]
    assert tc["tool_name"] == "slow_tool"
    # El fix: latency_ms debe ser > 0 (antes era 0)
    assert tc["latency_ms"] > 0, f"latency_ms deberia ser > 0, got {tc['latency_ms']}"
    assert (
        tc["latency_ms"] >= 40
    ), f"latency_ms deberia ser >= 40ms (sleep 50ms), got {tc['latency_ms']}"


# Sprint 9.5: regression test para Brecha 1 (refactor dedup _execute_tool_call
# y _execute_tool_call_v2). Valida que ambos wrappers producen el MISMO
# resultado (mismo tool_calls row, mismo telemetry, mismo role=tool message)
# usando el helper compartido _run_single_tool_call.
@pytest.mark.asyncio
async def test_execute_tool_call_v1_and_v2_produce_same_db_state(
    db: Any,
) -> None:
    """Sprint 9.5 Brecha 1 regression test: los 2 wrappers (_execute_tool_call
    para run() y _execute_tool_call_v2 para run_stream) deben producir el
    MISMO estado en DB. La unica diferencia es que _execute_tool_call_v2
    retorna un dict con hermes_status. La logica vive en _run_single_tool_call.
    """
    settings = type("S", (), {"read_tool_max_chars": 150000, "system_tool_max_chars": 2500})()
    registry = ToolRegistry()

    async def my_tool() -> dict:
        return {"data": 42}

    registry.register(
        "my_tool",
        my_tool,
        description="My tool",
        schema={"type": "function", "function": {"name": "my_tool", "parameters": {}}},
    )

    # Usar AsyncMock para evitar inicializar LLMRouter (que requiere
    # muchas settings de Hermes). Para _run_single_tool_call no se usa
    # el router, asi que es seguro mockear.
    router = AsyncMock()
    loop = AgentLoop(router, registry, db, settings=settings, max_iterations=3)

    # Run v1 (usado por run())
    conv1 = await db.new_conversation(chat_id=777, user_id=1)
    assistant1 = await db.add_message(conv1, "assistant", "")
    tool_call = ToolCall(id="tc1", name="my_tool", arguments={})
    await loop._execute_tool_call(conv1, assistant1, tool_call)

    # Run v2 (usado por run_stream) y verificar el dict retornado
    conv2 = await db.new_conversation(chat_id=888, user_id=1)
    assistant2 = await db.add_message(conv2, "assistant", "")
    tool_call2 = ToolCall(id="tc2", name="my_tool", arguments={})
    result = await loop._execute_tool_call_v2(conv2, assistant2, tool_call2)

    # Verificar hermes_status dict
    assert result["event"] == "tool_done"
    assert result["tool"] == "my_tool"
    assert result["tool_call_id"] == "tc2"
    assert result["success"] is True
    assert result["latency_ms"] >= 0

    # Comparar DB state de ambas runs
    tcs1 = await db.list_tool_calls_for_conversation(conv1)
    tcs2 = await db.list_tool_calls_for_conversation(conv2)
    assert len(tcs1) == 1
    assert len(tcs2) == 1
    # Mismo tool_name, success, error
    assert tcs1[0]["tool_name"] == tcs2[0]["tool_name"] == "my_tool"
    assert tcs1[0]["success"] == tcs2[0]["success"] == 1
    # Mismo result_json (wrapped en <tool_output>)
    assert "<tool_output" in tcs1[0]["result_json"]
    assert "<tool_output" in tcs2[0]["result_json"]
    # El JSON se serializa con comillas simples por secure_execute
    assert "42" in tcs1[0]["result_json"]
    assert "42" in tcs2[0]["result_json"]
    assert "data" in tcs1[0]["result_json"]
    assert "data" in tcs2[0]["result_json"]

    # Mismo role=tool message (mismo content salvo timestamp)
    import re

    async with db.conn.execute(
        "SELECT content FROM messages WHERE conversation_id IN (?, ?) AND role='tool'",
        (conv1, conv2),
    ) as cur:
        tool_msgs = await cur.fetchall()
    assert len(tool_msgs) == 2
    # Quitar timestamp (varia entre runs) y comparar estructura
    tstamp_re = re.compile(r'timestamp="[^"]+"')
    c1 = tstamp_re.sub('timestamp="X"', tool_msgs[0]["content"])
    c2 = tstamp_re.sub('timestamp="X"', tool_msgs[1]["content"])
    assert c1 == c2
    # Y verificar que el contenido funcional es el mismo
    assert "{'data': 42}" in c1
    assert "{'data': 42}" in c2


# --- Sprint 16.6: regression guard for legacy prefix asymmetry ---
# LLM cascade review of PR #93 (2026-07-06) caught that the legacy prefix
# "User memory (consolidated facts...)" was still present in run() but
# had been removed in run_stream() (3rd-pass adversarial MAJOR #3 only
# touched run_stream). This test prevents regression to that asymmetry.


async def test_run_does_not_include_legacy_user_memory_prefix(
    db: Database,
    registry: ToolRegistry,
    settings: Any,
    respx_mock: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sprint 16.6 regression guard: run() must not insert the legacy
    'User memory (consolidated facts, retrieved by semantic similarity):'
    prefix in the no-system-message branch. The <user_memory> wrapper
    provides the explanatory prose; the legacy prefix is redundant.

    Mirrors test_run_stream_calls_inject_memory_facts_once_per_request
    (in test_run_stream.py) but for run() instead of run_stream().
    """
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    # The router will retry on failure; we respond once with the
    # captured system content.
    captured: dict[str, str] = {}

    def callback(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        # Find the system message
        for msg in body.get("messages", []):
            if msg.get("role") == "system":
                captured["system_content"] = msg.get("content", "")
                break
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    respx_mock.post(openai_url).mock(side_effect=callback)

    # Stub _inject_memory_facts to return a fixed memory block
    memory_block = "<user_memory>\ntest facts here\n</user_memory>"

    async def fake_inject(user_query: str) -> str:
        return memory_block

    router = LLMRouter(settings)
    loop = AgentLoop(router, registry, db, max_iterations=3)
    monkeypatch.setattr(loop, "_inject_memory_facts", fake_inject)
    conv_id = await db.new_conversation(chat_id=100, user_id=1)
    try:
        await loop.run(conv_id, "test user message")
    finally:
        await router.aclose()

    # The injected memory block should be present (via wrapper)
    assert memory_block in captured.get("system_content", ""), (
        f"Expected the <user_memory> wrapper in system content, "
        f"got: {captured.get('system_content', '<none>')!r}"
    )
    # The LEGACY prefix must NOT be present (that's the regression)
    assert "User memory (consolidated facts" not in captured["system_content"]
    assert "retrieved by semantic similarity" not in captured["system_content"]
