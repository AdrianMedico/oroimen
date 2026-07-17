"""Tests industriales para `hermes.handlers.messages`.

Cubre:
- Funciones puras: `_build_messages_payload`.
- Handler completo (`handle_message`) para texto y voz.
- Errores, edge cases, persistencia, telemetría, system prompt.
- Flujo de voz v1.2: STT queue (Gemini) + LLM con smart routing unificado.

Estrategia de mocking (justificación):
- `Database` y `Telemetry` reales (sin red, sin InfluxDB). Más fiel, más
  confianza en el comportamiento end-to-end del handler.
- `aiogram.Bot` real con `get_file` parcheado (la API exige token válido
  `botid:token`).
- `aiogram.Message` se mockea con `MagicMock` + spec selectivo. Razón:
  `Message` de aiogram es un modelo pydantic **frozen**, lo que impide
  reasignar `answer`. Además, `chat` no acepta `None` aunque la lógica
  defensiva del handler lo contempla. Mockearlo con `MagicMock` es la
  forma correcta de testear el comportamiento de `message.answer` sin
  atarnos al modelo pydantic.
- `LLMRouter` real con `httpx.AsyncClient` mockeado por `respx` (vía
  fixtures). Esto prueba el camino real del router sin tocar la red.
- `STTQueue` se inyecta con un `transcribe_fn` mockeado (AsyncMock).
  Esto evita llamadas reales a Gemini y hace los tests deterministas.
- El handler se extrae del Router con
  `router.message.handlers[0].callback` y se invoca directamente con un
  `Message` mock. Esto ejecuta la MISMA función que corre en producción
  (mismas dependencias capturadas en la closure: bot, db, llm, settings,
  telemetry, stt_queue).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiogram import Bot
from aiogram.types import Chat, User, Voice

from hermes.config import Settings
from hermes.handlers.messages import (
    _build_messages_payload,
    build_message_router,
    build_system_prompt,
)
from hermes.memory.db import Database
from hermes.telemetry import Telemetry

# Las fixtures (settings, db, telemetry, bot) y helpers (get_handler,
# make_text_message, _AnswerCapture) vienen automáticamente de
# conftest.py — pytest las descubre sin necesidad de importarlas.
from tests.conftest import (
    TEST_BOT_TOKEN,
    TEST_CHAT_ID,
    TEST_USER_ID,
    _AnswerCapture,
    get_handler,
    make_text_message,
)

# ---------------------------------------------------------------------------
# Helpers específicos de test_messages (no en conftest porque solo se usan aquí)
# ---------------------------------------------------------------------------


def make_voice_message(
    *,
    user_id: int = TEST_USER_ID,
    chat_id: int = TEST_CHAT_ID,
    file_id: str = "voice_file_abc",
    duration: int = 5,
    mime_type: str | None = "audio/ogg",
    file_size: int = 1024,
    thread_id: int | None = None,
) -> tuple[Any, _AnswerCapture]:
    """Mensaje de voz (Message.voice != None, text = None)."""
    user = User(id=user_id, is_bot=False, first_name="Test")
    chat = Chat(id=chat_id, type="private")
    voice = Voice(
        file_id=file_id,
        file_unique_id=f"uniq_{file_id}",
        duration=duration,
        mime_type=mime_type,  # type: ignore[arg-type]
        file_size=file_size,
    )
    capture = _AnswerCapture()
    msg = MagicMock(spec=["from_user", "chat", "text", "voice", "message_thread_id", "answer"])
    msg.from_user = user
    msg.chat = chat
    msg.text = None
    msg.voice = voice
    msg.message_thread_id = thread_id
    msg.answer = capture
    return msg, capture


def make_edge_message(
    *,
    with_user: bool = True,
    with_chat: bool = True,
    with_text: str | None = "hola",
    with_voice: Voice | None = None,
) -> Any:
    """Mensaje con campos arbitrariamente ausentes (None) para edge cases.

    A diferencia de `aiogram.Message` (que es pydantic frozen y rechaza
    None en `chat`), aquí permitimos cualquier combinación para ejercitar
    las guardas defensivas del handler.
    """
    user = User(id=TEST_USER_ID, is_bot=False, first_name="Test") if with_user else None
    chat = Chat(id=TEST_CHAT_ID, type="private") if with_chat else None
    msg = MagicMock(spec=["from_user", "chat", "text", "voice", "message_thread_id", "answer"])
    msg.from_user = user
    msg.chat = chat
    msg.text = with_text
    msg.voice = with_voice
    msg.message_thread_id = None
    msg.answer = _AnswerCapture()
    return msg


# ---------------------------------------------------------------------------
# Tests de la función pura `_build_messages_payload`
# ---------------------------------------------------------------------------


class TestBuildMessagesPayload:
    """`_build_messages_payload` es pura: dado history+payload+model, devuelve
    la lista de mensajes para el LLM. No toca I/O."""

    def test_includes_system_prompt_with_model(self) -> None:
        history = [
            {"role": "user", "content": "hola", "model_used": None, "created_at": "x"},
        ]
        out = _build_messages_payload(history, "¿qué hora es?", "deepseek-v4-flash")
        assert out[0]["role"] == "system"
        # El system prompt se formatea con el nombre del modelo
        assert "deepseek-v4-flash" in out[0]["content"]
        # La identidad y el motor se declaran sin contradicciones.
        assert "Oroimen" in out[0]["content"]
        assert "motor configurado" in out[0]["content"]

    def test_appends_history_in_order(self) -> None:
        history = [
            {"role": "user", "content": "msg-1", "model_used": None, "created_at": "a"},
            {"role": "assistant", "content": "resp-1", "model_used": None, "created_at": "b"},
            {"role": "user", "content": "msg-2", "model_used": None, "created_at": "c"},
        ]
        out = _build_messages_payload(history, "msg-3", "deepseek-v4-flash")
        # system + 3 history + current user = 5
        assert len(out) == 5
        assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "user"]
        assert [m["content"] for m in out[1:]] == ["msg-1", "resp-1", "msg-2", "msg-3"]

    def test_appends_current_payload_string(self) -> None:
        out = _build_messages_payload([], "texto plano", "minimax-m3")
        assert out[-1] == {"role": "user", "content": "texto plano"}

    def test_appends_current_payload_multimodal(self) -> None:
        """Para voz, el payload es una lista de bloques (text + input_audio)."""
        payload: list[dict] = [
            {"type": "text", "text": "transcribe"},
            {"type": "input_audio", "input_audio": {"data": "abc", "format": "ogg"}},
        ]
        out = _build_messages_payload([], payload, "mimo-v2.5")
        assert out[-1] == {"role": "user", "content": payload}

    def test_empty_history_only_system_and_user(self) -> None:
        out = _build_messages_payload([], "hola", "deepseek-v4-flash")
        assert len(out) == 2
        assert out[0]["role"] == "system"
        assert out[1] == {"role": "user", "content": "hola"}

    def test_different_model_changes_system_prompt(self) -> None:
        """El system prompt se formatea con el modelo; el resto es igual."""
        a = _build_messages_payload([], "x", "deepseek-v4-flash")
        b = _build_messages_payload([], "x", "mimo-v2.5")
        assert a[0]["content"] != b[0]["content"]
        assert "deepseek-v4-flash" in a[0]["content"]
        assert "mimo-v2.5" in b[0]["content"]


# ---------------------------------------------------------------------------
# Tests de `_build_voice_payload`
# ---------------------------------------------------------------------------


class TestBuildVoicePayload:
    """DEPRECATED en v1.2: _build_voice_payload ya no existe.

    v1.2: el flujo de voz se ha rediseñado para usar STT externo (Gemini).
    - Se elimina _build_voice_payload (la multimodalidad input_audio
      no funcionaba via OpenCode Go, bug #30389).
    - Se añade STTQueue en handlers/messages.py.

    Esta clase queda vacía como marcador histórico. Los tests del
    nuevo flujo están en TestHandleVoiceMessage más abajo.
    """

    def test_deprecated_placeholder(self) -> None:
        """Marcador: la función _build_voice_payload se eliminó en v1.2."""


# ---------------------------------------------------------------------------
# Tests del handler completo `handle_message` - mensajes de texto
# ---------------------------------------------------------------------------


class TestHandleTextMessage:
    """Flujo completo de un mensaje de texto. Mockeamos el LLM con respx para
    controlar la respuesta. DB y Telemetry son reales."""

    @pytest.fixture
    def llm_response_ok(self) -> dict:
        return {
            "choices": [{"message": {"content": "¡Hola! ¿En qué puedo ayudarte?"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }

    @pytest.fixture
    def llm_response_fallback(self) -> dict:
        # Anthropic-compat (qwen3.7-plus / minimax-m3) usa "content" en vez de "choices"
        return {
            "content": [{"type": "text", "text": "respuesta de fallback"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    @pytest.mark.asyncio
    async def test_text_message_happy_path(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        llm_response_ok: dict,
    ) -> None:
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(return_value=httpx.Response(200, json=llm_response_ok))

        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)
        msg, capture = make_text_message("hola")

        await handler(msg)

        # 1. Se llama a `answer` 2 veces: "…pensando…" y luego la respuesta
        assert capture.count() == 2
        assert "…pensando…" in capture.calls[0]
        assert "¿En qué puedo ayudarte" in capture.last()

        # 2. La conversación y los mensajes quedan en la DB
        cid = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        history = await db.get_history(cid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hola"
        assert history[1]["role"] == "assistant"
        assert history[1]["model_used"] == "deepseek-v4-flash"
        # Verificamos tokens y latency con SQL directo
        # (get_history no incluye esas columnas, ver nota en get_history)
        async with db.conn.execute(
            "SELECT tokens_in, tokens_out, latency_ms FROM messages "
            "WHERE conversation_id=? AND role='assistant'",
            (cid,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["tokens_in"] == 12
        assert row["tokens_out"] == 8
        assert row["latency_ms"] is not None
        assert row["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_text_uses_history_on_second_message(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        llm_response_ok: dict,
    ) -> None:
        url = f"{settings.opencode_go_base_url}/chat/completions"
        # Capturamos el payload enviado al LLM en la segunda llamada
        captured: list[dict] = []

        def callback(request: httpx.Request) -> httpx.Response:
            import json as _json

            body = _json.loads(request.content)
            captured.append(body)
            return httpx.Response(200, json=llm_response_ok)

        respx_mock.post(url).mock(side_effect=callback)

        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)

        # Primer mensaje: crea conversación y guarda en DB
        msg1, _ = make_text_message("primer mensaje")
        await handler(msg1)
        # Segundo mensaje: debe usar el history
        msg2, _ = make_text_message("segundo mensaje")
        await handler(msg2)

        # La segunda llamada al LLM debe incluir ambos mensajes
        assert len(captured) == 2
        second = captured[1]
        # El handler guarda el user message en DB ANTES de llamar al LLM,
        # así que el history ya contiene el msg2. Luego _build_messages_payload
        # añade el current_user_payload al final (que también es msg2).
        # Estructura: system + user(msg1) + assistant(resp1) + user(msg2 guardado) + user(msg2 current)
        roles = [m["role"] for m in second["messages"]]
        assert roles == ["system", "user", "assistant", "user", "user"]
        assert second["messages"][1]["content"] == "primer mensaje"
        assert second["messages"][3]["content"] == "segundo mensaje"
        assert second["messages"][4]["content"] == "segundo mensaje"

    @pytest.mark.asyncio
    async def test_text_fallback_to_minimax(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        llm_response_fallback: dict,
    ) -> None:
        # deepseek falla 3 veces, minimax responde
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        respx_mock.post(openai_url).mock(side_effect=httpx.Response(500, json={"error": "down"}))
        respx_mock.post(anthropic_url).mock(
            return_value=httpx.Response(200, json=llm_response_fallback)
        )

        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)
        msg, capture = make_text_message("hola")
        await handler(msg)

        # Se terminó usando minimax-m3
        cid = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        history = await db.get_history(cid)
        assert history[1]["model_used"] == "minimax-m3"
        async with db.conn.execute(
            "SELECT tokens_in, tokens_out FROM messages "
            "WHERE conversation_id=? AND role='assistant'",
            (cid,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["tokens_in"] == 10
        assert row["tokens_out"] == 5
        # La respuesta llega al usuario
        assert "fallback" in capture.last().lower()

    @pytest.mark.asyncio
    async def test_text_skips_command_messages(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        llm_response_ok: dict,
    ) -> None:
        """Los comandos (texto que empieza por /) los maneja build_command_router,
        no este handler. Aquí solo verificamos que NO se procesa ni se llama al LLM."""
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(return_value=httpx.Response(200, json=llm_response_ok))

        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)
        msg, capture = make_text_message("/start")

        await handler(msg)

        # No debe haber llamadas a answer ni a LLM
        assert capture.count() == 0
        # La DB no debe tener mensajes
        cid = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        assert await db.get_history(cid) == []
        # respx no recibió ninguna llamada
        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    async def test_text_skips_when_no_user(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        llm_response_ok: dict,
    ) -> None:
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(return_value=httpx.Response(200, json=llm_response_ok))

        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)
        # from_user=None
        msg = make_edge_message(with_user=False, with_text="hola")

        await handler(msg)

        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    async def test_text_skips_when_no_chat(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        llm_response_ok: dict,
    ) -> None:
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(return_value=httpx.Response(200, json=llm_response_ok))

        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)
        # chat=None
        msg = make_edge_message(with_chat=False, with_text="hola")

        await handler(msg)

        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    async def test_text_skips_when_neither_text_nor_voice(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        llm_response_ok: dict,
    ) -> None:
        """Mensajes tipo foto/documento/etc. sin texto ni voz se ignoran."""
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(return_value=httpx.Response(200, json=llm_response_ok))

        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)
        # text=None y voice=None
        msg = make_edge_message(with_text=None, with_voice=None)

        await handler(msg)

        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    async def test_text_llm_error_returns_user_message(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
    ) -> None:
        """Si todos los modelos del chain fallan, se notifica al usuario."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        err = httpx.Response(500, json={"error": "down"})
        respx_mock.post(openai_url).mock(side_effect=err)
        respx_mock.post(anthropic_url).mock(return_value=err)

        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)
        msg, capture = make_text_message("hola")

        await handler(msg)

        # Se enviaron 2 mensajes: "…pensando…" y el error
        assert capture.count() == 2
        assert "Error" in capture.last()
        # Pero el mensaje del usuario SÍ se guardó
        cid = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        history = await db.get_history(cid)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        # No se guardó respuesta del assistant
        assert all(m["role"] == "user" for m in history)

    @pytest.mark.asyncio
    async def test_text_response_uses_system_prompt_with_model_name(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
    ) -> None:
        """Verifica que el system prompt enviado al LLM incluye el modelo primario."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        captured_system: list[str] = []
        import json as _json

        def cb(req: httpx.Request) -> httpx.Response:
            body = _json.loads(req.content)
            for m in body.get("messages", []):
                if m.get("role") == "system":
                    captured_system.append(m["content"])
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        respx_mock.post(openai_url).mock(side_effect=cb)
        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)
        msg, _ = make_text_message("¿qué modelo eres?")
        await handler(msg)

        assert len(captured_system) == 1
        # El system prompt formateado incluye el modelo primario del chain de texto
        assert "deepseek-v4-flash" in captured_system[0]
        # Y previene alucinaciones de identidad
        assert "Oroimen" in captured_system[0]
        assert "motor configurado" in captured_system[0]


# ---------------------------------------------------------------------------
# Tests del handler completo - mensajes de voz
# ---------------------------------------------------------------------------


class TestHandleVoiceMessage:
    """Flujo completo de un mensaje de voz en v1.2.

    Pipeline:
    1. Descarga audio de Telegram (mockeado con respx).
    2. STT queue llama a Gemini externo (mockeado con mock function).
    3. LLM con smart routing unificado (mockeado con respx).
    4. Respuesta al usuario con prefijo 🎙️.

    Estrategia de mocking:
    - `bot.get_file` mockeado con AsyncMock.
    - Descarga de audio mockeada con respx.
    - `STTQueue` se construye con un `transcribe_fn` mockeado (no llama a Gemini).
    - LLM mockeado con respx.
    """

    @pytest.fixture
    def llm_response_voice(self) -> dict:
        """Respuesta mockeada del LLM para el flujo de voz."""
        return {
            "choices": [{"message": {"content": "Hola, ¿en qué puedo ayudarte?"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 15},
        }

    @pytest.fixture
    def mock_stt_transcribe(self):
        """Crea un mock de la función transcribe para inyectar en STTQueue.

        Por defecto retorna "hola, ¿qué hora es?" (transcripción exitosa).
        El test puede sobreescribir el side_effect para simular fallos.
        """
        from unittest.mock import AsyncMock

        mock = AsyncMock()
        mock.return_value = "hola, ¿qué hora es?"
        return mock

    @pytest.fixture
    def stt_queue(self, mock_stt_transcribe):
        """STTQueue real con transcribe mockeado (no llama a Gemini)."""
        from hermes.stt.queue import STTQueue

        return STTQueue(
            transcribe_fn=mock_stt_transcribe, max_concurrent=2, per_minute=12, timeout_s=5
        )

    @pytest.mark.asyncio
    async def test_voice_message_happy_path(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        llm_response_voice: dict,
        stt_queue,
        mock_stt_transcribe,
    ) -> None:
        """Happy path: descarga audio -> STT -> LLM -> respuesta con prefijo 🎙️."""
        # Mock de get_file en el bot
        file_path = "documents/voice/file_xyz.ogg"
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path=file_path)
        )
        # Mock de la descarga del audio de Telegram
        dl_url = f"https://api.telegram.org/file/bot{TEST_BOT_TOKEN}/{file_path}"
        respx_mock.get(dl_url).mock(return_value=httpx.Response(200, content=b"fake-ogg-bytes"))

        # Mock del LLM (smart routing unificado: text_chain[0] = deepseek-v4-flash
        # en el fixture settings por compatibilidad con tests pre-existentes).
        llm_url = f"{settings.opencode_go_base_url}/chat/completions"
        captured_payload: list[dict] = []
        import json as _json

        def cb(req: httpx.Request) -> httpx.Response:
            captured_payload.append(_json.loads(req.content))
            return httpx.Response(200, json=llm_response_voice)

        respx_mock.post(llm_url).mock(side_effect=cb)

        router = build_message_router(bot, db, settings, telemetry, stt_queue=stt_queue)
        handler = get_handler(router)
        msg, capture = make_voice_message(file_id="file_xyz")

        await handler(msg)

        # Se enviaron 2 mensajes: "…pensando…" y la respuesta con prefijo 🎙️
        assert capture.count() == 2
        assert "🎙️" in capture.last()
        assert "Hola" in capture.last()

        # STT fue llamado con los bytes correctos
        mock_stt_transcribe.assert_called_once()
        call_args = mock_stt_transcribe.call_args
        assert call_args.args[0] == b"fake-ogg-bytes"
        assert call_args.args[1] == "audio/ogg"

        # El LLM recibió la transcripción como mensaje de texto (no multimodal)
        assert len(captured_payload) == 1
        user_msg = captured_payload[0]["messages"][-1]
        assert user_msg["role"] == "user"
        # user_msg["content"] debe ser un string con la transcripción
        assert user_msg["content"] == "hola, ¿qué hora es?"

        # DB: la transcripción se almacena como contenido del mensaje user
        cid = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        history = await db.get_history(cid)
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hola, ¿qué hora es?"  # NO "[voice message]"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_voice_message_download_failure(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        stt_queue,
        mock_stt_transcribe,
    ) -> None:
        """Si no se puede descargar el audio, se notifica al usuario y NO se llama al STT ni al LLM."""
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path="voice/err.ogg")
        )
        dl_url = f"https://api.telegram.org/file/bot{TEST_BOT_TOKEN}/voice/err.ogg"
        respx_mock.get(dl_url).mock(return_value=httpx.Response(500))

        llm_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(llm_url).mock(return_value=httpx.Response(200))

        router = build_message_router(bot, db, settings, telemetry, stt_queue=stt_queue)
        handler = get_handler(router)
        msg, capture = make_voice_message()

        await handler(msg)

        # Mensaje de error al usuario
        assert capture.count() == 1
        assert "audio" in capture.last().lower()
        # STT NO fue llamado
        mock_stt_transcribe.assert_not_called()
        # El LLM NO fue llamado
        llm_calls = [c for c in respx_mock.calls if str(c.request.url) == llm_url]
        assert len(llm_calls) == 0
        # La DB no tiene mensajes
        cid = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        assert await db.get_history(cid) == []

    @pytest.mark.asyncio
    async def test_voice_message_get_file_failure(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        stt_queue,
        mock_stt_transcribe,
    ) -> None:
        """Si `bot.get_file()` lanza una excepción, se maneja gracefully."""
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            side_effect=Exception("Telegram API error")
        )

        router = build_message_router(bot, db, settings, telemetry, stt_queue=stt_queue)
        handler = get_handler(router)
        msg, capture = make_voice_message()

        await handler(msg)

        assert capture.count() == 1
        assert "audio" in capture.last().lower()
        mock_stt_transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_message_stt_empty_transcription(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        stt_queue,
        mock_stt_transcribe,
    ) -> None:
        """Si STT retorna transcripción vacía (silencio/ruido), se pide al usuario que repita."""
        # STT retorna string vacío (caso silencio o Gemini no entendió)
        mock_stt_transcribe.return_value = ""

        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path="voice/silent.ogg")
        )
        dl_url = f"https://api.telegram.org/file/bot{TEST_BOT_TOKEN}/voice/silent.ogg"
        respx_mock.get(dl_url).mock(return_value=httpx.Response(200, content=b"silent"))
        llm_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(llm_url).mock(return_value=httpx.Response(200))

        router = build_message_router(bot, db, settings, telemetry, stt_queue=stt_queue)
        handler = get_handler(router)
        msg, capture = make_voice_message()

        await handler(msg)

        # El usuario recibe un mensaje pidiendo que repita
        assert capture.count() == 1
        response = capture.last().lower()
        assert "repita" in response or "no te entend" in response or "no se entend" in response
        # El LLM NO fue llamado (no hay texto para procesar)
        llm_calls = [c for c in respx_mock.calls if str(c.request.url) == llm_url]
        assert len(llm_calls) == 0
        # La DB no tiene mensajes
        cid = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        assert await db.get_history(cid) == []

    @pytest.mark.asyncio
    async def test_voice_message_stt_timeout(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        mock_stt_transcribe,
    ) -> None:
        """Si STT queue da timeout (saturación), mensaje user-friendly."""
        from hermes.stt.queue import STTQueue

        # STT queue con transcribe que tarda más que el timeout
        async def slow_transcribe(audio: bytes, mime: str) -> str:
            await asyncio.sleep(2.0)  # más que timeout=0.1
            return "nunca"

        stt_queue = STTQueue(
            transcribe_fn=slow_transcribe, max_concurrent=1, per_minute=100, timeout_s=0.1
        )

        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path="voice/long.ogg")
        )
        dl_url = f"https://api.telegram.org/file/bot{TEST_BOT_TOKEN}/voice/long.ogg"
        respx_mock.get(dl_url).mock(return_value=httpx.Response(200, content=b"long-audio"))
        llm_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(llm_url).mock(return_value=httpx.Response(200))

        router = build_message_router(bot, db, settings, telemetry, stt_queue=stt_queue)
        handler = get_handler(router)
        msg, capture = make_voice_message()

        await handler(msg)

        # El usuario recibe mensaje de saturación
        assert capture.count() == 1
        response = capture.last().lower()
        assert "satur" in response or "intent" in response

    @pytest.mark.asyncio
    async def test_voice_message_stt_generic_error(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        stt_queue,
        mock_stt_transcribe,
    ) -> None:
        """Si STT lanza STTError genérico, mensaje user-friendly."""
        from hermes.stt.gemini import STTError

        mock_stt_transcribe.side_effect = STTError("Gemini down")

        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path="voice/err.ogg")
        )
        dl_url = f"https://api.telegram.org/file/bot{TEST_BOT_TOKEN}/voice/err.ogg"
        respx_mock.get(dl_url).mock(return_value=httpx.Response(200, content=b"audio"))
        llm_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(llm_url).mock(return_value=httpx.Response(200))

        router = build_message_router(bot, db, settings, telemetry, stt_queue=stt_queue)
        handler = get_handler(router)
        msg, capture = make_voice_message()

        await handler(msg)

        # El usuario recibe mensaje de error de transcripción
        assert capture.count() == 1
        response = capture.last().lower()
        assert "transcri" in response or "audio" in response
        # El LLM NO fue llamado
        llm_calls = [c for c in respx_mock.calls if str(c.request.url) == llm_url]
        assert len(llm_calls) == 0

    @pytest.mark.asyncio
    async def test_voice_response_includes_transcription_prefix(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
        llm_response_voice: dict,
        stt_queue,
        mock_stt_transcribe,
    ) -> None:
        """La respuesta al usuario lleva el prefijo 🎙️ + transcripción + respuesta."""
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path="voice/x.ogg")
        )
        dl_url = f"https://api.telegram.org/file/bot{TEST_BOT_TOKEN}/voice/x.ogg"
        respx_mock.get(dl_url).mock(return_value=httpx.Response(200, content=b"x"))
        llm_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(llm_url).mock(return_value=httpx.Response(200, json=llm_response_voice))

        router = build_message_router(bot, db, settings, telemetry, stt_queue=stt_queue)
        handler = get_handler(router)
        msg, capture = make_voice_message()

        await handler(msg)

        last_msg = capture.last()
        assert "🎙️" in last_msg
        # El prefijo indica "transcripción y respuesta"
        assert "transcrip" in last_msg.lower()


# ---------------------------------------------------------------------------
# Tests de errores y edge cases varios
# ---------------------------------------------------------------------------


class TestHandleMessageEdgeCases:
    """Casos límite del handler que no encajan en las secciones de texto/voz."""

    @pytest.mark.asyncio
    async def test_router_name_is_messages(
        self, settings: Settings, db: Database, telemetry: Telemetry, bot: Bot
    ) -> None:
        router = build_message_router(bot, db, settings, telemetry)
        assert router.name == "messages"

    @pytest.mark.asyncio
    async def test_each_user_has_own_conversation(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
    ) -> None:
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )
        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)

        msg_a, _ = make_text_message("user A", user_id=111, chat_id=222)
        msg_b, _ = make_text_message("user B", user_id=333, chat_id=444)
        await handler(msg_a)
        await handler(msg_b)

        cid_a = await db.get_or_create_conversation(chat_id=222, user_id=111)
        cid_b = await db.get_or_create_conversation(chat_id=444, user_id=333)
        assert cid_a != cid_b

        hist_a = await db.get_history(cid_a)
        hist_b = await db.get_history(cid_b)
        assert hist_a[0]["content"] == "user A"
        assert hist_b[0]["content"] == "user B"

    @pytest.mark.asyncio
    async def test_thread_id_is_preserved(
        self,
        settings: Settings,
        db: Database,
        telemetry: Telemetry,
        bot: Bot,
        respx_mock,
    ) -> None:
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )
        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)

        msg1, _ = make_text_message("msg en thread 7", thread_id=7)
        await handler(msg1)

        # Verificar que la conversación tiene thread_id=7
        async with db.conn.execute(
            "SELECT thread_id FROM conversations WHERE chat_id=?",
            (TEST_CHAT_ID,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["thread_id"] == 7

    @pytest.mark.asyncio
    async def test_telemetry_ok_on_success(
        self,
        settings: Settings,
        db: Database,
        bot: Bot,
        respx_mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """En el flujo feliz, la telemetría se llama con result='ok'."""
        # Habilitamos la telemetría con un cliente InfluxDB mockeado
        from hermes.telemetry import Telemetry as RealTelemetry

        class FakeWriteApi:
            def __init__(self) -> None:
                self.written: list = []

            def write(self, *, bucket: str, record: object, write_precision: object) -> None:
                self.written.append(record)

        class FakeInfluxClient:
            def __init__(self, *a: object, **kw: object) -> None:
                self._write_api = FakeWriteApi()
                self.write_api = lambda **_kw: self._write_api  # type: ignore[assignment]

            def close(self) -> None:
                pass

        monkeypatch.setattr("hermes.telemetry.InfluxDBClient", FakeInfluxClient)
        monkeypatch.setenv("INFLUX_URL", "http://fake:8086")
        monkeypatch.setenv("INFLUX_TOKEN", "fake-token")
        settings = Settings(_env_file=None)
        telemetry = RealTelemetry(settings)

        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )
        )

        router = build_message_router(bot, db, settings, telemetry)
        handler = get_handler(router)
        msg, _ = make_text_message("hola")
        await handler(msg)

        # Se llamó a record_message(result="ok") al menos una vez
        # (también puede haber record_llm_call y record_circuit_breaker_state)
        result_tags = [
            p._tags.get("result")  # type: ignore[attr-defined]
            for p in telemetry._write_api.written  # type: ignore[attr-defined]
            if hasattr(p, "_tags")
        ]
        assert "ok" in result_tags


# ---------------------------------------------------------------------------
# Sprint 4 T7: integración del AgentLoop en el handler
# ---------------------------------------------------------------------------


class TestAgentLoopIntegration:
    """Sprint 4 T7: cuando settings.tools_enabled=True y hay tools
    registradas, el handler usa AgentLoop en vez de router.chat() directo.

    Verifica:
    - Con tools: el path "agent loop" se activa.
    - Sin tools: el path legacy (router.chat) sigue funcionando.
    - Voz: NO usa agent loop (no aplica para STT → texto).
    - Error en agent loop: mensaje user-friendly al usuario.
    - Persistencia: tool_calls se guardan en DB.
    """

    @pytest.mark.asyncio
    async def test_text_uses_agent_loop_when_tools_available(
        self, bot: Any, settings: Settings, db: Database, telemetry: Telemetry, respx_mock: Any
    ) -> None:
        """Si hay tool_registry con tools Y tools_enabled=True, el handler
        invoca AgentLoop.run() en vez de llm.chat() directo."""
        from hermes.tools.registry import ToolRegistry

        async def get_weather(city: str) -> str:
            return f"{city}: 22 grados"

        registry = ToolRegistry()
        registry.register(
            "get_weather",
            get_weather,
            description="Get weather",
            schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        settings.tools_enabled = True

        # Mock LLM: primera respuesta pide tool, segunda da respuesta final
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        import json as _json

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
                                                "arguments": _json.dumps({"city": "Madrid"}),
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
                httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": "En Madrid hace 22 grados."}}],
                        "usage": {"prompt_tokens": 15, "completion_tokens": 8},
                    },
                ),
            ]
        )

        router = build_message_router(
            bot=bot, db=db, settings=settings, telemetry=telemetry, tool_registry=registry
        )
        handler = get_handler(router)
        msg, capture = make_text_message(text="¿Tiempo en Madrid?")
        await handler(msg)

        # El usuario recibe la respuesta final del agent loop
        answers = capture.calls
        assert any("22 grados" in a for a in answers)

    @pytest.mark.asyncio
    async def test_text_falls_back_to_plain_chat_when_no_tools(
        self, bot: Any, settings: Settings, db: Database, telemetry: Telemetry, respx_mock: Any
    ) -> None:
        """Si no hay tools registradas, el handler usa router.chat() directo
        (no AgentLoop)."""
        settings.tools_enabled = True

        # Mock LLM normal
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hola"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )
        )

        # Pasamos tool_registry=None explícito (no AgentLoop)
        router = build_message_router(
            bot=bot, db=db, settings=settings, telemetry=telemetry, tool_registry=None
        )
        handler = get_handler(router)
        msg, capture = make_text_message(text="hola")
        await handler(msg)

        # El path legacy: solo 1 call HTTP, sin tool_calls
        assert len(respx_mock.calls) == 1
        assert any("hola" in a for a in capture.calls)

    @pytest.mark.asyncio
    async def test_voice_still_uses_router_directly(
        self, bot: Any, settings: Settings, db: Database, telemetry: Telemetry, respx_mock: Any
    ) -> None:
        """Voz sigue usando el path legacy (STT → router.chat), NO AgentLoop.

        Razón: el STT ya produce texto, no tiene sentido que el LLM
        decida qué hacer con él. Voz es directo.
        """
        from hermes.stt.queue import STTQueue
        from hermes.tools.registry import ToolRegistry

        registry = ToolRegistry()

        async def useless() -> str:
            return "useless"

        registry.register(
            "useless", useless, description="x", schema={"type": "object", "properties": {}}
        )
        settings.tools_enabled = True

        # Mock STT (no Gemini real)
        stt_queue = STTQueue(
            transcribe_fn=AsyncMock(return_value="hola transcripción"),
            max_concurrent=1,
            per_minute=10,
            timeout_s=5.0,
        )

        # Mock descarga de audio
        file_path = "documents/voice/test.ogg"
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path=file_path)
        )
        dl_url = f"https://api.telegram.org/file/bot{TEST_BOT_TOKEN}/{file_path}"
        respx_mock.get(dl_url).mock(return_value=httpx.Response(200, content=b"fake-ogg"))

        # Mock LLM (path legacy de voz)
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "respuesta voz"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )
        )

        router = build_message_router(
            bot=bot,
            db=db,
            settings=settings,
            telemetry=telemetry,
            stt_queue=stt_queue,
            tool_registry=registry,
        )
        handler = get_handler(router)
        msg, capture = make_voice_message(file_id="v1", duration=3)
        await handler(msg)

        # 1 call LLM (NO agent loop)
        llm_calls = [c for c in respx_mock.calls if str(c.request.url) == openai_url]
        assert len(llm_calls) == 1
        # La respuesta incluye el prefijo 🎙️
        assert any("🎙️" in a for a in capture.calls)

    @pytest.mark.asyncio
    async def test_agent_loop_error_returns_friendly_message(
        self, bot: Any, settings: Settings, db: Database, telemetry: Telemetry, respx_mock: Any
    ) -> None:
        """Si AgentLoop lanza excepción, el usuario recibe mensaje friendly."""
        from hermes.tools.registry import ToolRegistry

        async def bad_tool() -> str:
            return "ok"

        registry = ToolRegistry()
        registry.register(
            "bad_tool", bad_tool, description="x", schema={"type": "object", "properties": {}}
        )
        settings.tools_enabled = True

        # Forzar error: 500 en todos los endpoints
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        respx_mock.post(openai_url).mock(return_value=httpx.Response(500))
        respx_mock.post(anthropic_url).mock(return_value=httpx.Response(500))

        router = build_message_router(
            bot=bot, db=db, settings=settings, telemetry=telemetry, tool_registry=registry
        )
        handler = get_handler(router)
        msg, capture = make_text_message(text="test")
        await handler(msg)

        # El usuario recibe un mensaje (puede ser error o fallback del LLM)
        assert len(capture.calls) >= 2  # "…pensando…" + respuesta final

    @pytest.mark.asyncio
    async def test_long_response_is_split_into_chunks(
        self, bot: Any, settings: Settings, db: Database, telemetry: Telemetry, respx_mock: Any
    ) -> None:
        """v0.5.7-t50: response > 4096 chars se divide en chunks.
        Regression test: bug UnboundLocalError donde 'prefix' no estaba
        definido en la rama is_first, dejando al bot en 'Pensando...'.
        """
        from hermes.tools.registry import ToolRegistry

        # Tool no-op (queremos que el LLM decida respuesta directa)
        async def noop() -> str:
            return "x"

        registry = ToolRegistry()
        registry.register(
            "noop", noop, description="x", schema={"type": "object", "properties": {}}
        )
        settings.tools_enabled = True

        # Mock LLM: respuesta LARGA (>4096 chars) sin tool calls
        long_response = "A" * 5000  # > 4096 limite de Telegram
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": long_response, "tool_calls": None}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
        )

        router = build_message_router(
            bot=bot, db=db, settings=settings, telemetry=telemetry, tool_registry=registry
        )
        handler = get_handler(router)
        msg, capture = make_text_message(text="test")

        # Antes del fix, esto lanzaba UnboundLocalError y el bot quedaba
        # colgado en "Pensando...". Ahora debe enviar multiples chunks.
        await handler(msg)

        # Debe haber enviado multiples mensajes (chunks + paginacion)
        # chunk 1: "A..." + "\n\n_(continuación, 1/2)_"
        # chunk 2: "_(continuación, 2/2)_\n\n" + "A..."
        # Ademas: 1 "Pensando..." + 1 "Llamando..." + chunks
        # Verificamos al menos 2 chunks (puede ser mas por throttling/log)
        assert len(capture.calls) >= 3, f"Pocos calls ({len(capture.calls)}): {capture.calls!r}"

        # El primer chunk es "Pensando..."
        assert "Pensando" in capture.calls[0]

        # Al menos uno de los calls contiene la respuesta larga (chunked)
        long_chunk_found = any(
            "A" * 100 in call
            for call in capture.calls  # substring de 100 As
        )
        assert long_chunk_found, f"Ningun chunk contiene la respuesta: {capture.calls!r}"

        # La paginacion debe estar presente (varios chunks)
        has_continuation = any("continuación" in call for call in capture.calls)
        assert has_continuation, f"Falta paginacion: {capture.calls!r}"

    @pytest.mark.asyncio
    async def test_tools_disabled_uses_legacy_path(
        self, bot: Any, settings: Settings, db: Database, telemetry: Telemetry, respx_mock: Any
    ) -> None:
        """Si settings.tools_enabled=False (default), el handler ignora
        tool_registry y usa router.chat() directo."""
        from hermes.tools.registry import ToolRegistry

        async def ignored_tool() -> str:
            return "ignored"

        registry = ToolRegistry()
        registry.register(
            "ignored_tool",
            ignored_tool,
            description="x",
            schema={"type": "object", "properties": {}},
        )
        settings.tools_enabled = False  # feature flag off

        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "legacy"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )

        router = build_message_router(
            bot=bot, db=db, settings=settings, telemetry=telemetry, tool_registry=registry
        )
        handler = get_handler(router)
        msg, capture = make_text_message(text="hola")
        await handler(msg)

        # Path legacy: 1 call, sin tool_calls
        assert len(respx_mock.calls) == 1
        assert any("legacy" in a for a in capture.calls)


def host_url() -> str:
    """Helper: la URL que aiogram usa para getFile (api.telegram.org)."""
    from tests.conftest import TEST_BOT_TOKEN

    return f"https://api.telegram.org/file/bot{TEST_BOT_TOKEN}/"


# ---------------------------------------------------------------------------
# Sprint 4 T8: build_system_prompt dinámico
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Sprint 4 T8: el system prompt se construye dinámicamente con la
    sección ## Available Tools cuando hay tools registradas."""

    def test_still_includes_identity_block(self) -> None:
        """El bloque de identidad SIEMPRE está, con o sin tools."""
        prompt_with = build_system_prompt(model="minimax-m3", tool_specs=[])
        prompt_without = build_system_prompt(model="minimax-m3", tool_specs=None)
        for prompt in (prompt_with, prompt_without):
            assert "Oroimen" in prompt
            assert "minimax-m3" in prompt
            # No deberia filtrar el nombre real del autor (Phase 2 audit F-007).
            assert "creado por project owner" not in prompt
            assert "asistente personal" in prompt

    def test_frontier_identity_does_not_deny_configured_engine(self) -> None:
        prompt = build_system_prompt(model="gpt-5.6-sol", tool_specs=[])
        assert "gpt-5.6-sol" in prompt
        assert "NO eres GPT" not in prompt
        assert "Oroimen" in prompt

    def test_no_tools_omits_tools_section(self) -> None:
        """Sin tools, no aparece la sección ## Available Tools ni
        ## Tool Output Quarantine."""
        prompt = build_system_prompt(model="minimax-m3", tool_specs=None)
        assert "## Available Tools" not in prompt
        assert "## Tool Output Quarantine" not in prompt

    def test_empty_tool_list_omits_tools_section(self) -> None:
        """tool_specs=[] (lista vacía) tampoco incluye tools."""
        prompt = build_system_prompt(model="minimax-m3", tool_specs=[])
        assert "## Available Tools" not in prompt

    def test_includes_tool_descriptions(self) -> None:
        """Con 2 tools, el prompt menciona ambas con description y schema."""
        from hermes.tools.registry import ToolSpec

        specs = [
            ToolSpec(
                name="get_weather",
                description="Get current weather for a city",
                schema={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
            ToolSpec(
                name="search_vault",
                description="Search text in Obsidian vault",
                schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ]
        prompt = build_system_prompt(model="minimax-m3", tool_specs=specs)
        assert "## Available Tools" in prompt
        assert "### get_weather" in prompt
        assert "### search_vault" in prompt
        assert "Get current weather for a city" in prompt
        assert "Search text in Obsidian vault" in prompt
        # Schema presente (como JSON string)
        assert '"city"' in prompt
        assert '"query"' in prompt

    def test_includes_anti_injection_warning(self) -> None:
        """Con tools, el prompt tiene la sección Tool Output Quarantine."""
        from hermes.tools.registry import ToolSpec

        specs = [
            ToolSpec(name="tool", description="x", schema={"type": "object", "properties": {}}),
        ]
        prompt = build_system_prompt(model="minimax-m3", tool_specs=specs)
        assert "## Tool Output Quarantine" in prompt
        assert "<tool_output" in prompt
        assert "ignore previous instructions" in prompt.lower()
        assert "you are now" in prompt.lower() or "manipulación" in prompt.lower()

    def test_instructs_to_use_tools(self) -> None:
        """El prompt anima al LLM a usar las tools disponibles."""
        from hermes.tools.registry import ToolSpec

        specs = [
            ToolSpec(
                name="get_weather",
                description="Weather",
                schema={"type": "object", "properties": {}},
            ),
        ]
        prompt = build_system_prompt(model="minimax-m3", tool_specs=specs)
        # Debe animar a usar tools
        assert "Puedes invocar" in prompt or "tools" in prompt.lower()
