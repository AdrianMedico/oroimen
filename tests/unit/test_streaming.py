"""Tests Sprint 7 T53.2: HTTP API streaming (SSE).

Cobertura:
- StreamChunk dataclass: to_sse(), to_done()
- _merge_tool_call_deltas: out-of-order, multi-tool, partial args, invalid JSON
- chat_stream: yields content/tool_calls/reasoning/final chunks
- chat_stream: chain fallback on failure
- chat_stream: raises LLMError if all models fail
- chat_stream: anthropic fallback to blocking chat() (R5)

Patron: respx_mock.stream() para simular el SSE upstream. Patron
heredado de test_router.py pero con side_effect que devuelve un
stream de lineas en lugar de un JSON unico.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from hermes.config import Settings
from hermes.llm.router import (
    LLMError,
    LLMResponse,
    LLMRouter,
    StreamChunk,
    ToolCall,
    _merge_tool_call_deltas,
)

# --- StreamChunk serialization ---


def test_stream_chunk_to_sse_content_only() -> None:
    """Chunk con content produce 'data: {"choices":[{"delta":{"content":"x"}}]}\\n\\n'."""
    chunk = StreamChunk(content="hola", model="minimax-m3")
    sse = chunk.to_sse()
    assert sse.startswith("data: ")
    assert sse.endswith("\n\n")
    payload = json.loads(sse[6:].strip())
    assert payload == {
        "choices": [{"delta": {"content": "hola"}}],
    }


def test_stream_chunk_to_sse_with_hermes_status() -> None:
    """Custom hermes_status extension se incluye en delta."""
    chunk = StreamChunk(
        hermes_status={"event": "tool_start", "tool": "x"},
        model="minimax-m3",
    )
    sse = chunk.to_sse()
    payload = json.loads(sse[6:].strip())
    assert payload["choices"][0]["delta"]["hermes_status"]["event"] == "tool_start"


def test_stream_chunk_to_sse_with_finish_reason() -> None:
    """finish_reason se incluye en choice (no delta)."""
    chunk = StreamChunk(finish_reason="stop", model="minimax-m3")
    sse = chunk.to_sse()
    payload = json.loads(sse[6:].strip())
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert "finish_reason" not in payload["choices"][0]["delta"]


def test_stream_chunk_to_done() -> None:
    """to_done() retorna 'data: [DONE]\\n\\n' literal."""
    chunk = StreamChunk()
    assert chunk.to_done() == "data: [DONE]\n\n"


# --- _merge_tool_call_deltas (Gemini's flagged risk) ---


def test_merge_arguments_concatenation_raises_llm_error_on_invalid_json() -> None:
    """Caso de concatenacion: arguments se acumula, pero si no es JSON
    valido al final, raise LLMError (la chain cae al siguiente modelo).

    Concatena correctamente (la longitud del string lo confirma), pero
    el parseo falla porque no es JSON. Esto es comportamiento esperado:
    la chain intenta con el siguiente modelo.
    """
    deltas = [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "x", "arguments": ""},
        },
        {"index": 0, "function": {"arguments": "PART_A"}},
        {"index": 0, "function": {"arguments": "PART_B"}},
        {"index": 0, "function": {"arguments": "PART_C"}},
    ]
    with pytest.raises(LLMError, match="invalid JSON") as exc_info:
        _merge_tool_call_deltas(deltas)
    # Verificar que el error message incluye los 3 parts concatenados
    assert "PART_APART_BPART_C" in str(exc_info.value)


def test_merge_single_tool_call_arguments_valid_json_via_helper() -> None:
    """Validacion adicional: con json.loads valido, el merge produce dict.

    Usa json.dumps para construir strings con llaves sin ambiguedad
    de sintaxis Python.
    """
    # Construir deltas con arguments validos JSON usando json.dumps
    deltas = [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Madrid"}),
            },
        },
    ]
    result = _merge_tool_call_deltas(deltas)
    assert len(result) == 1
    assert result[0].name == "get_weather"
    assert result[0].arguments == {"city": "Madrid"}


def test_merge_multiple_tool_calls_by_index() -> None:
    """Multiples tool calls, cada una con su index, se mergean por separado.

    Usamos json.dumps para construir arguments validos sin ambiguedad
    de sintaxis Python.
    """
    args_a = json.dumps({"x": 1})
    args_b = json.dumps({"y": 2})
    deltas = [
        {"index": 0, "id": "call_a", "function": {"name": "fn_a", "arguments": ""}},
        {"index": 1, "id": "call_b", "function": {"name": "fn_b", "arguments": ""}},
        {"index": 0, "function": {"arguments": args_a[:4]}},  # '{"x"'
        {"index": 1, "function": {"arguments": args_b[:4]}},  # '{"y"'
        {"index": 0, "function": {"arguments": args_a[4:]}},  # ': 1}'
        {"index": 1, "function": {"arguments": args_b[4:]}},  # ': 2}'
    ]
    result = _merge_tool_call_deltas(deltas)
    assert len(result) == 2
    by_id = {tc.id: tc for tc in result}
    assert by_id["call_a"].name == "fn_a"
    assert by_id["call_a"].arguments == {"x": 1}
    assert by_id["call_b"].name == "fn_b"
    assert by_id["call_b"].arguments == {"y": 2}


def test_merge_in_order_deltas() -> None:
    """Deltas en orden cronologico se mergean correctamente.

    Importante: el orden de llegada IMPORTA (concatenamos en orden de
    la lista). El upstream OpenAI garantiza orden de llegada.
    """
    full_args = json.dumps({"a": 1})  # '{"a": 1}'
    mid = len(full_args) // 2  # 4
    deltas = [
        # Primero llega el delta con id + name + primera mitad de args
        {
            "index": 0,
            "id": "call_1",
            "function": {"name": "x", "arguments": full_args[:mid]},
        },
        # Despues llega la segunda mitad de args
        {"index": 0, "function": {"arguments": full_args[mid:]}},
    ]
    result = _merge_tool_call_deltas(deltas)
    assert result[0].id == "call_1"
    assert result[0].arguments == {"a": 1}


def test_merge_deltas_preserve_argument_concat_order() -> None:
    """Concatenacion de arguments sigue el orden de la lista de deltas.

    Esto es importante para que el JSON sea valido al final.
    """
    deltas = [
        # arguments llega en 3 pedazos en este orden: "A", "B", "C"
        {"index": 0, "id": "call_1", "function": {"name": "x", "arguments": ""}},
        {"index": 0, "function": {"arguments": "A"}},
        {"index": 0, "function": {"arguments": "B"}},
        {"index": 0, "function": {"arguments": "C"}},
    ]
    # Concatenacion esperada: "A" + "B" + "C" = "ABC"
    # No es JSON valido, pero la concatenacion es correcta.
    with pytest.raises(LLMError, match="invalid JSON") as exc_info:
        _merge_tool_call_deltas(deltas)
    # El error message debe contener "ABC" (concatenacion correcta)
    assert "ABC" in str(exc_info.value)


def test_merge_empty_arguments_defaults_to_empty_dict() -> None:
    """Si arguments nunca llega (delta vacio), default a {}."""
    deltas = [
        {"index": 0, "id": "call_1", "type": "function", "function": {"name": "x"}},
    ]
    result = _merge_tool_call_deltas(deltas)
    assert result[0].arguments == {}


def test_merge_invalid_json_raises_llm_error() -> None:
    """JSON invalido al final raise LLMError (chain cae al siguiente modelo)."""
    deltas = [
        {
            "index": 0,
            "id": "call_1",
            "function": {"name": "x", "arguments": "this is not json"},
        },
    ]
    with pytest.raises(LLMError, match="arguments invalid JSON"):
        _merge_tool_call_deltas(deltas)


def test_merge_missing_id_raises_llm_error() -> None:
    """Si nunca llega el id, raise LLMError (stream corrupto)."""
    deltas = [
        {"index": 0, "function": {"name": "x", "arguments": "{}"}},
    ]
    with pytest.raises(LLMError, match="missing id"):
        _merge_tool_call_deltas(deltas)


def test_merge_arguments_must_be_json_object_not_array() -> None:
    """Si arguments es un array o escalar, raise LLMError (debe ser object)."""
    deltas = [
        {
            "index": 0,
            "id": "call_1",
            "function": {"name": "x", "arguments": "[1, 2, 3]"},
        },
    ]
    with pytest.raises(LLMError, match="must be a JSON object"):
        _merge_tool_call_deltas(deltas)


# --- chat_stream end-to-end (con respx stream) ---


def _make_sse_stream(lines: list[str]) -> bytes:
    """Helper: serializa lineas SSE al formato wire.

    Output: cada linea es 'data: {json}\\n' seguido de 'data: [DONE]\\n\\n'.
    El cliente httpx lo lee via aiter_lines() y nosotros lo parseamos.
    """
    parts = []
    for line in lines:
        if line.startswith("data: "):
            parts.append(line.encode() + b"\n")
        else:
            parts.append(f"data: {line}\n".encode())
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


@pytest.mark.asyncio
async def test_chat_stream_yields_content_chunks(settings: Settings, respx_mock: Any) -> None:
    """Stream simple: 3 chunks de content + 1 finish_reason.

    Forzamos chain OpenAI puro (deepseek-v4-flash) para no caer al
    fallback blocking de Anthropic (que en este test no queremos
    testear; hay otro test especifico para eso).
    """
    # Override chain: ambos OpenAI (deepseek). El text_chain es property
    # que retorna [primary, fallback], asi que cambiamos los dos.
    settings.llm_text_primary = "deepseek-v4-flash"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/chat/completions"
    sse = _make_sse_stream(
        [
            json.dumps({"choices": [{"delta": {"content": "Hola "}}]}),
            json.dumps({"choices": [{"delta": {"content": "qué "}}]}),
            json.dumps({"choices": [{"delta": {"content": "tal"}}]}),
            json.dumps({"choices": [{"finish_reason": "stop"}]}),
        ]
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "hola"}]):
            chunks.append(c)
    finally:
        await router.aclose()

    content_chunks = [c for c in chunks if c.content is not None]
    assert len(content_chunks) == 3
    assert "".join(c.content for c in content_chunks) == "Hola qué tal"
    final = [c for c in chunks if c.finish_reason is not None]
    assert len(final) == 1
    assert final[0].finish_reason == "stop"
    # Todos los chunks reportan el modelo
    assert all(c.model == "deepseek-v4-flash" for c in chunks if c.model)


@pytest.mark.asyncio
async def test_chat_stream_accumulates_and_emits_tool_calls(
    settings: Settings, respx_mock: Any
) -> None:
    """Tool call llega en 3 deltas, se acumulan y emiten en chunk final."""
    url = f"{settings.opencode_go_base_url}/chat/completions"
    sse = _make_sse_stream(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "get_weather", "arguments": ""},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]
                            }
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": ' "Madrid"}'}}
                                ]
                            }
                        }
                    ]
                }
            ),
            json.dumps({"choices": [{"finish_reason": "tool_calls"}]}),
        ]
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream(
            [{"role": "user", "content": "tiempo en Madrid"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
        ):
            chunks.append(c)
    finally:
        await router.aclose()

    # Solo UN chunk final con tool_calls (los deltas intermedios se acumulan)
    tool_call_chunks = [c for c in chunks if c.tool_calls is not None]
    assert len(tool_call_chunks) == 1
    assert len(tool_call_chunks[0].tool_calls) == 1
    tc = tool_call_chunks[0].tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Madrid"}
    # finish_reason en el mismo chunk
    assert tool_call_chunks[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_chat_stream_emits_reasoning_content(settings: Settings, respx_mock: Any) -> None:
    """DeepSeek thinking mode: delta.reasoning_content se emite como chunk custom."""
    url = f"{settings.opencode_go_base_url}/chat/completions"
    sse = _make_sse_stream(
        [
            json.dumps({"choices": [{"delta": {"reasoning_content": "Pensando "}}]}),
            json.dumps({"choices": [{"delta": {"reasoning_content": "más..."}}]}),
            json.dumps({"choices": [{"delta": {"content": "Respuesta"}}]}),
            json.dumps({"choices": [{"finish_reason": "stop"}]}),
        ]
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)
    finally:
        await router.aclose()

    reasoning_chunks = [c for c in chunks if c.reasoning_content is not None]
    assert len(reasoning_chunks) == 2
    assert "".join(c.reasoning_content for c in reasoning_chunks) == "Pensando más..."


@pytest.mark.asyncio
async def test_chat_stream_falls_back_to_next_model_on_failure(
    settings_v12: Settings, respx_mock: Any
) -> None:
    """Si minimax-m3 falla (Anthropic path), cae a deepseek-v4-flash (OpenAI path).

    Sprint 16.7: uses settings_v12 fixture which sets LLM_TEXT_PRIMARY=minimax-m3
    and LLM_TEXT_FALLBACK=deepseek-v4-flash (production chain). The previous
    version used the default settings fixture which had LLM_TEXT_PRIMARY=
    deepseek-v4-flash, so the test never actually exercised the fallback
    path it claimed to test.
    """
    openai_url = f"{settings_v12.opencode_go_base_url}/chat/completions"
    anthropic_url = f"{settings_v12.opencode_go_base_url}/messages"
    # Primary (minimax-m3, Anthropic path) fails with 500. Multiple
    # attempts to handle retries + Anthropic streaming fallback to blocking.
    respx_mock.post(anthropic_url).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(500),
        ]
    )
    # Fallback (deepseek-v4-flash, OpenAI path) succeeds
    respx_mock.post(openai_url).mock(
        return_value=httpx.Response(
            200,
            content=_make_sse_stream(
                [
                    json.dumps({"choices": [{"delta": {"content": "fallback OK"}}]}),
                    json.dumps({"choices": [{"finish_reason": "stop"}]}),
                ]
            ),
        )
    )

    router = LLMRouter(settings_v12)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream(
            [{"role": "user", "content": "x"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "x", "description": "noop tool", "parameters": {}},
                }
            ],
        ):
            chunks.append(c)
    finally:
        await router.aclose()

    # Fallback model emitted the content
    content_chunks = [c for c in chunks if c.content is not None]
    assert "".join(c.content for c in content_chunks) == "fallback OK"
    # Anthropic URL was hit (primary attempts before fallback)
    anthropic_calls = [c for c in respx_mock.calls if str(c.request.url).endswith("/messages")]
    assert (
        len(anthropic_calls) >= 1
    ), "Anthropic URL should have been hit at least once before fallback"
    # OpenAI URL was hit (fallback success, possibly with retries)
    openai_calls = [c for c in respx_mock.calls if str(c.request.url).endswith("/chat/completions")]
    assert len(openai_calls) >= 1, "OpenAI URL should have been hit at least once for fallback"


@pytest.mark.asyncio
async def test_chat_stream_raises_llm_error_if_all_models_fail(
    settings: Settings, respx_mock: Any
) -> None:
    """Si TODA la chain falla, raise LLMError tras agotar retries."""
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    anthropic_url = f"{settings.opencode_go_base_url}/messages"
    # Todos los modelos fallan en TODOS los retries
    respx_mock.post(openai_url).mock(return_value=httpx.Response(500))
    respx_mock.post(anthropic_url).mock(return_value=httpx.Response(500))

    router = LLMRouter(settings)
    try:
        with pytest.raises(LLMError, match="All models in chain failed"):
            async for _ in router.chat_stream([{"role": "user", "content": "x"}]):
                pass
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_chat_stream_anthropic_emits_text_and_finish(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 16.8: Anthropic-format streaming emite text_delta + finish_reason.

    Reemplaza el test viejo de "fallback a blocking". Verifica que el nuevo
    _stream_anthropic() parsea SSE de /v1/messages y emite StreamChunk con
    content incremental + finish_reason='stop' (mapeado desde 'end_turn').
    """
    # Forzar chain: minimax-m3 (Anthropic) primero
    settings.llm_text_primary = "minimax-m3"
    settings.llm_text_fallback = "deepseek-v4-flash"

    # Anthropic-format SSE para /v1/messages
    url = f"{settings.opencode_go_base_url}/messages"
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"minimax-m3","usage":{"input_tokens":12,"output_tokens":1}}}\n'
        b"\n"
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hola "}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"mundo"}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":0}\n'
        b"\n"
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}\n'
        b"\n"
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n'
        b"\n"
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)

        # Verificar chunks emitidos
        content_chunks = [c for c in chunks if c.content is not None]
        assert len(content_chunks) == 2
        assert content_chunks[0].content == "Hola "
        assert content_chunks[1].content == "mundo"

        # Finish reason mapeado de end_turn -> stop
        final = [c for c in chunks if c.finish_reason is not None]
        assert len(final) == 1
        assert final[0].finish_reason == "stop"

        # Usage tracking (Sprint 16.8.1 B-2: OpenAI keys para AgentLoop compat)
        usage_chunks = [c for c in chunks if c.usage is not None]
        assert len(usage_chunks) >= 1
        last_usage = usage_chunks[-1].usage
        assert last_usage is not None
        assert last_usage.get("prompt_tokens") == 12
        assert last_usage.get("completion_tokens") == 3
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_chat_stream_anthropic_emits_thinking_and_tool_use(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 16.8: Anthropic extended thinking + tool_use streaming.

    Verifica que thinking_delta -> StreamChunk(reasoning_content) y
    tool_use (con input_json_delta fragmentado) -> StreamChunk(tool_calls)
    """
    settings.llm_text_primary = "minimax-m3"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/messages"
    sse = (
        # Thinking block
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Pensando..."}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":0}\n'
        b"\n"
        # Text block
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Resultado"}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":1}\n'
        b"\n"
        # Tool use block (input fragmentado en 3 deltas)
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_abc","name":"get_weather","input":{}}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":"}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\\"Madrid\\""}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"}"}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":2}\n'
        b"\n"
        # Final
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}\n'
        b"\n"
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n'
        b"\n"
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)

        # Reasoning chunk
        reasoning_chunks = [c for c in chunks if c.reasoning_content is not None]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0].reasoning_content == "Pensando..."

        # Text chunk
        content_chunks = [c for c in chunks if c.content is not None]
        assert len(content_chunks) == 1
        assert content_chunks[0].content == "Resultado"

        # Tool call chunk (input JSON reconstruido desde 3 deltas)
        tool_chunks = [c for c in chunks if c.tool_calls is not None]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_calls
        assert tc is not None and len(tc) == 1
        assert tc[0].id == "toolu_abc"
        assert tc[0].name == "get_weather"
        assert tc[0].arguments == {"city": "Madrid"}

        # Finish reason mapeado de tool_use -> tool_calls
        final = [c for c in chunks if c.finish_reason is not None]
        assert len(final) == 1
        assert final[0].finish_reason == "tool_calls"
    finally:
        await router.aclose()


# --- Sprint 16.8.1 tests (adversarial review follow-ups B-1, B-2, M-1, M-2, M-3) ---


@pytest.mark.asyncio
async def test_chat_stream_anthropic_error_event_raises_llm_error(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 16.8.1 B-1: Anthropic error event raise LLMError (no KeyError crash).

    Verifica que un event: error mid-stream NO crashea con KeyError por
    LogRecord.message reserved-attribute. El fix fue renombrar el extra
    key de "message" a "error_message" en router.py:793.

    Usamos chain_override=[model] para forzar single-model chain y
    que el LLMError se propague sin caer al fallback OpenAI path.
    """
    settings.llm_text_primary = "minimax-m3"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/messages"
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg_x","type":"message","role":"assistant","content":[],"model":"minimax-m3","usage":{"input_tokens":1,"output_tokens":1}}}\n'
        b"\n"
        b"event: error\n"
        b'data: {"type":"error","error":{"type":"api_error","message":"overloaded"}}\n'
        b"\n"
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        # chain_override=[only_minimax] para evitar fallback chain
        with pytest.raises(LLMError) as exc_info:
            async for c in router.chat_stream(
                [{"role": "user", "content": "x"}],
                chain_override=["minimax-m3"],
            ):
                chunks.append(c)
        # Verificar que LLMError tiene el mensaje del upstream
        assert "overloaded" in str(exc_info.value) or "api_error" in str(exc_info.value)
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_chat_stream_anthropic_usage_compatible_with_agent_loop(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 16.8.1 B-2: Anthropic usage usa keys OpenAI-compatible.

    Verifica que el usage emitido por _stream_anthropic() usa
    prompt_tokens/completion_tokens (no input_tokens/output_tokens),
    para que AgentLoop.run_stream (loop.py:432-434) lo lea correctamente.
    """
    settings.llm_text_primary = "minimax-m3"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/messages"
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg_y","type":"message","role":"assistant","content":[],"model":"minimax-m3","usage":{"input_tokens":100,"output_tokens":1}}}\n'
        b"\n"
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":0}\n'
        b"\n"
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":42}}\n'
        b"\n"
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n'
        b"\n"
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)

        # Verificar que el chunk final tiene usage con keys OpenAI
        final = [c for c in chunks if c.finish_reason is not None]
        assert len(final) == 1
        usage = final[0].usage
        assert usage is not None
        # Keys deben ser OpenAI, no Anthropic
        assert "prompt_tokens" in usage, f"Missing prompt_tokens. Keys: {list(usage.keys())}"
        assert (
            "completion_tokens" in usage
        ), f"Missing completion_tokens. Keys: {list(usage.keys())}"
        assert "input_tokens" not in usage, "Anthropic key 'input_tokens' should be translated"
        assert "output_tokens" not in usage, "Anthropic key 'output_tokens' should be translated"
        # Valores correctos (100 input + 42 output = 100 + 42)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 42
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_chat_stream_anthropic_unknown_stop_reason_defaults_to_stop(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 16.8.1 M-1: stop_reason desconocido -> 'stop' + warning log."""
    settings.llm_text_primary = "minimax-m3"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/messages"
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg_z","type":"message","role":"assistant","content":[],"model":"minimax-m3","usage":{"input_tokens":5,"output_tokens":1}}}\n'
        b"\n"
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":0}\n'
        b"\n"
        # stop_reason inventado que NO esta en el mapping
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"weird_new_reason","stop_sequence":null}}\n'
        b"\n"
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n'
        b"\n"
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)

        final = [c for c in chunks if c.finish_reason is not None]
        assert len(final) == 1
        # Fallback defensivo: weird_reason -> "stop"
        assert final[0].finish_reason == "stop"
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_chat_stream_anthropic_multiple_tool_use_blocks(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 16.8.1 M-3: multiples tool_use blocks emiten chunks separados."""
    settings.llm_text_primary = "minimax-m3"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/messages"
    sse = (
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_a","name":"fn_a","input":{}}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"x\\":1}"}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":0}\n'
        b"\n"
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_b","name":"fn_b","input":{}}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"y\\":2}"}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":1}\n'
        b"\n"
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null}}\n'
        b"\n"
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n'
        b"\n"
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)

        tool_chunks = [c for c in chunks if c.tool_calls is not None]
        assert len(tool_chunks) == 2
        assert tool_chunks[0].tool_calls[0].name == "fn_a"
        assert tool_chunks[0].tool_calls[0].arguments == {"x": 1}
        assert tool_chunks[1].tool_calls[0].name == "fn_b"
        assert tool_chunks[1].tool_calls[0].arguments == {"y": 2}
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_chat_stream_anthropic_truncated_stream_emits_final_chunk(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 16.8.1 M-2: stream truncado (no message_stop) emite chunk final.

    Verifica que cuando el upstream cierra el stream sin enviar
    message_stop (despues de message_delta), el parser emite un chunk
    final con finish_reason + usage para que AgentLoop no quede
    silenciosamente sin info.
    """
    settings.llm_text_primary = "minimax-m3"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/messages"
    # SSE que termina abruptamente despues de message_delta (no message_stop)
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg_t","type":"message","role":"assistant","content":[],"model":"minimax-m3","usage":{"input_tokens":7,"output_tokens":1}}}\n'
        b"\n"
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"truncated"}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":0}\n'
        b"\n"
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}\n'
        # NO message_stop -> upstream cierra el stream aqui
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)

        # Verificar que hay content + chunk final con usage
        content_chunks = [c for c in chunks if c.content is not None]
        assert len(content_chunks) == 1
        assert content_chunks[0].content == "truncated"

        final = [c for c in chunks if c.finish_reason is not None]
        # M-2 fix: chunk final debe existir aunque no hubo message_stop
        assert len(final) == 1, "M-2: deberia haber un chunk final con finish_reason"
        assert final[0].finish_reason == "stop"
        # usage tambien debe estar (a pesar del truncation)
        assert final[0].usage is not None
        assert final[0].usage["prompt_tokens"] == 7
        assert final[0].usage["completion_tokens"] == 3
        # stop_reason=end_turn -> no es truncation, flag debe ser False
        assert final[0].truncated is False
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_chat_stream_anthropic_max_tokens_marks_truncated_true(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 16.8.2: stop_reason=max_tokens -> StreamChunk.truncated=True.

    El cliente (Open WebUI) lee choice.truncated=True y avisa al usuario
    que la respuesta fue truncada por el LLM.
    """
    settings.llm_text_primary = "minimax-m3"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/messages"
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg_x","type":"message","role":"assistant","content":[],"model":"minimax-m3","usage":{"input_tokens":5,"output_tokens":1}}}\n'
        b"\n"
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
        b"\n"
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}\n'
        b"\n"
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":0}\n'
        b"\n"
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":99}}\n'
        b"\n"
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n'
        b"\n"
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)

        final = [c for c in chunks if c.finish_reason is not None]
        assert len(final) == 1
        # Anthropic max_tokens -> OpenAI length
        assert final[0].finish_reason == "length"
        # Flag de truncation
        assert final[0].truncated is True

        # SSE serialization incluye el flag para que el cliente lo vea
        sse_payload = final[0].to_sse()
        # json.dumps default usa ": " (con espacio). Buscamos con y sin.
        assert '"truncated": true' in sse_payload or '"truncated":true' in sse_payload
        assert (
            '"truncation_reason": "max_tokens"' in sse_payload
            or '"truncation_reason":"max_tokens"' in sse_payload
        )
    finally:
        await router.aclose()


def test_stream_chunk_truncated_default_is_false() -> None:
    """Sprint 16.8.2: StreamChunk default truncated=False.

    Verifica que la nueva field existe y default es False para
    preservar el comportamiento pre-cambio en chunks que no son
    de finalizacion.
    """
    chunk = StreamChunk(content="hola")
    assert chunk.truncated is False
    # Y se puede setear explicitamente
    chunk2 = StreamChunk(content="adios", truncated=True)
    assert chunk2.truncated is True


def test_to_sse_does_not_mutate_hermes_status_in_place() -> None:
    """Sprint 16.8.3 (M-2 fix): to_sse() no muta self.hermes_status in-place.

    Antes el codigo hacia `existing_status = self.hermes_status or {}`
    seguido de `existing_status["truncated"] = True` — eso modificaba
    el dict original (por referencia). Si un caller retiene el dict
    entre llamadas a to_sse() o lo comparte, ve mutaciones inesperadas.

    Fix: shallow-copy `dict(self.hermes_status or {})`.
    """
    original_status: dict = {"phase": "thinking", "iter": 5}
    chunk = StreamChunk(
        finish_reason="length",
        model="minimax-m3",
        hermes_status=original_status,
        truncated=True,
    )
    # Snapshot antes
    before = dict(original_status)
    # Primera serializacion
    sse1 = chunk.to_sse()
    # El dict original NO debe haber mutado
    assert original_status == before, f"to_sse() muto self.hermes_status: {original_status}"
    # Segunda serializacion (idempotencia)
    sse2 = chunk.to_sse()
    assert sse1 == sse2, "to_sse() no es idempotente"
    # El SSE serializado SI tiene los nuevos keys (truncated, truncation_reason)
    assert '"truncated": true' in sse1 or '"truncated":true' in sse1
    assert "max_tokens" in sse1


@pytest.mark.asyncio
async def test_chat_stream_handles_malformed_sse_lines_gracefully(
    settings: Settings, respx_mock: Any
) -> None:
    """Lineas SSE malformadas se ignoran, el stream continua."""
    url = f"{settings.opencode_go_base_url}/chat/completions"
    # Stream con: 1 linea vacia, 1 JSON malformado, 1 OK, [DONE]
    sse = (
        b"\n"
        b"data: {esto no es json}\n\n"
        b"data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}).encode() + b"\n\n"
        b"data: [DONE]\n\n"
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)
    finally:
        await router.aclose()

    # Solo el chunk OK se emite (los malformados se skip)
    content_chunks = [c for c in chunks if c.content is not None]
    assert len(content_chunks) == 1
    assert content_chunks[0].content == "ok"


async def test_chat_stream_handles_choices_missing_or_empty(
    respx_mock: Any, settings: Settings
) -> None:
    """Sprint 7 T53.2 fase 4 regression: upstream puede enviar chunks
    sin 'choices' (e.g. error mid-stream envuelto en SSE). Antes esto
    crasheaba con IndexError en `_stream_openai`. Ahora se logea y se
    skip, continuando el stream hasta [DONE] o respuesta valida.
    """
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    sse_body = (
        b'data: {"id":"x","error":{"message":"upstream oops"}}\n\n'
        b'data: {"choices":[]}\n\n'
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    respx_mock.post(openai_url).mock(return_value=httpx.Response(200, content=sse_body))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)
    finally:
        await router.aclose()

    # Solo el chunk con content='ok' llega al cliente. Los chunks de
    # error/choices vacio se logean y se skip.
    content_chunks = [c for c in chunks if c.content is not None]
    assert len(content_chunks) == 1
    assert content_chunks[0].content == "ok"


# Sprint 9.3.2: regression test for Bug 2 (iter_tokens_in/out=0).
# (a) El payload del stream incluye stream_options.include_usage=true.
# (b) Cuando upstream retorna usage, el StreamChunk final lo incluye.
# (c) El usage NO se serializa a SSE (es interno, no se filtra al cliente).
@pytest.mark.asyncio
async def test_chat_stream_payload_includes_stream_options_usage(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 9.3.2: el payload del stream pide include_usage=true."""
    settings.llm_text_primary = "deepseek-v4-flash"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/chat/completions"
    captured: list[dict] = []

    def cb(req: httpx.Request) -> httpx.Response:
        import json as _json

        captured.append(_json.loads(req.content))
        sse = (
            b'data: {"choices":[{"delta":{"content":"hola"}}]}\n\n'
            b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":42,"completion_tokens":7}}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=sse)

    respx_mock.post(url).mock(side_effect=cb)

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)
    finally:
        await router.aclose()

    # (a) El payload incluye stream_options.include_usage=true
    assert len(captured) == 1
    payload = captured[0]
    assert payload.get("stream_options") == {"include_usage": True}
    assert payload.get("stream") is True

    # (b) El StreamChunk final tiene usage populated
    final = [c for c in chunks if c.finish_reason is not None]
    assert len(final) == 1
    assert final[0].usage == {"prompt_tokens": 42, "completion_tokens": 7}

    # (c) El usage NO se filtra al SSE (es interno)
    sse_output = final[0].to_sse()
    assert "usage" not in sse_output
    assert "prompt_tokens" not in sse_output


@pytest.mark.asyncio
async def test_chat_stream_usage_in_separate_chunk_with_empty_choices(
    settings: Settings, respx_mock: Any
) -> None:
    """Sprint 9.3.2: OpenAI con include_usage envia usage en chunk con choices=[]."""
    settings.llm_text_primary = "deepseek-v4-flash"
    settings.llm_text_fallback = "deepseek-v4-flash"

    url = f"{settings.opencode_go_base_url}/chat/completions"
    # Patron real de OpenAI: chunk con content, chunk vacio, chunk con usage
    sse = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}\n\n'
        b"data: [DONE]\n\n"
    )
    respx_mock.post(url).mock(return_value=httpx.Response(200, content=sse))

    router = LLMRouter(settings)
    try:
        chunks: list[StreamChunk] = []
        async for c in router.chat_stream([{"role": "user", "content": "x"}]):
            chunks.append(c)
    finally:
        await router.aclose()

    # El chunk final con finish_reason debe tener usage del chunk separado
    final = [c for c in chunks if c.finish_reason is not None]
    assert len(final) == 1
    assert final[0].usage == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


def test_stream_chunk_usage_field_not_serialized_to_sse() -> None:
    """Sprint 9.3.2: usage field es interno, no se serializa a SSE."""
    chunk = StreamChunk(
        content="hello",
        usage={"prompt_tokens": 5, "completion_tokens": 3},
    )
    sse = chunk.to_sse()
    assert "usage" not in sse
    assert "prompt_tokens" not in sse
    assert "hello" in sse
@pytest.mark.asyncio
async def test_chat_stream_local_only_dispatches_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public local-only chain emits functional SSE without opencode-go."""
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setenv("LLM_TEXT_PRIMARY_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_TEXT_PRIMARY", "qwen2.5:7b")
    monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "false")
    settings = Settings(_env_file=None)
    router = LLMRouter(settings)
    assert router._client is None
    assert router._ollama_client is not None
    router._ollama_client.chat = AsyncMock(
        return_value=LLMResponse(
            content="local response",
            model="qwen2.5:7b",
            tokens_in=7,
            tokens_out=3,
            latency_ms=4,
            tool_calls=[ToolCall(id="call_1", name="search_files", arguments={"query": "x"})],
            reasoning_content="local reasoning",
        )
    )
    try:
        chunks = [
            chunk
            async for chunk in router.chat_stream([{"role": "user", "content": "hi"}])
        ]
    finally:
        await router.aclose()

    assert [chunk.reasoning_content for chunk in chunks if chunk.reasoning_content] == [
        "local reasoning"
    ]
    assert [chunk.content for chunk in chunks if chunk.content] == ["local response"]
    final = chunks[-1]
    assert final.model == "qwen2.5:7b"
    assert final.finish_reason == "tool_calls"
    assert final.tool_calls and final.tool_calls[0].name == "search_files"
    assert final.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


@pytest.mark.asyncio
async def test_chat_stream_normal_chain_falls_through_to_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-override streaming escalates from failed local chat to enabled GPT."""
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setenv("LLM_TEXT_PRIMARY_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_TEXT_PRIMARY", "qwen2.5:7b")
    monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "true")
    monkeypatch.setenv("LLM_TEXT_FRONTIER__API_KEY", "sk-test-stream-frontier-1234567890")
    settings = Settings(_env_file=None)
    router = LLMRouter(settings)
    assert router._ollama_client is not None
    assert router._frontier_client is not None
    router._ollama_client.chat = AsyncMock(side_effect=LLMError("local unavailable"))
    router._frontier_client.chat = AsyncMock(
        return_value=LLMResponse(
            content="frontier response",
            model="gpt-5.6-sol",
            tokens_in=5,
            tokens_out=2,
            latency_ms=8,
        )
    )
    try:
        chunks = [
            chunk
            async for chunk in router.chat_stream([{"role": "user", "content": "hi"}])
        ]
    finally:
        await router.aclose()

    assert [chunk.content for chunk in chunks if chunk.content] == ["frontier response"]
    assert chunks[-1].model == "gpt-5.6-sol"
    assert chunks[-1].finish_reason == "stop"
