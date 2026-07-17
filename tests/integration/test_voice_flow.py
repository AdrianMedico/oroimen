"""Integration tests: flujo de VOZ end-to-end.

Verifica el flujo completo de un mensaje de voz:
1. Telegram voice message arrives
2. Handler descarga el audio de Telegram (mockeado)
3. Handler codifica en base64
4. LLM call con input_audio (mockeado con respx)
5. Response con prefijo 🎙️

Para mockear la descarga de audio de Telegram, monkeypatchamos
`hermes.handlers.messages.httpx.AsyncClient` (que es como se importa
dentro de `_build_voice_payload`). El handler pasa el `http` que recibe
como argumento a `_build_voice_payload`, pero `_build_voice_payload` USA
el `http` que recibe (no crea uno nuevo). Sin embargo, al construir el
Router con `build_message_router`, se crea un `http = httpx.AsyncClient()`
nuevo en el closure.

Para testear, la estrategia más limpia es:
1. Mockear `bot.get_file` con AsyncMock
2. Mockear `httpx.AsyncClient` con un fake (esto afecta al `http` del
   closure del handler porque Python resuelve la clase en cada llamada)

El fake `httpx.AsyncClient` debe soportar el context manager protocol
(`async with`).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hermes.handlers.messages import build_message_router
from tests.conftest import TEST_CHAT_ID
from tests.integration.conftest import make_fake_http_for_voice, make_voice_message


def _build_router_with_fake_voice_http(bot, db, settings, telemetry, audio_bytes: bytes) -> Any:
    """Construye el message router con un fake http en el closure.

    Esta es la forma correcta de mockear la descarga de audio de voz
    en integration tests: el `http` que `build_message_router` crea en
    su closure es reemplazado por un fake que solo implementa `get()`.

    Returns:
        Tuple (router, handler).
    """
    # 1. Construir el router normalmente (su `http` interno es un httpx real)
    router = build_message_router(bot, db, settings, telemetry)
    handler = router.message.handlers[0].callback

    # 2. Reemplazar el `http` del closure del handler con nuestro fake
    #    (esto afecta SOLO al closure, no al global httpx)
    fake_http = make_fake_http_for_voice(audio_bytes)
    for cell, name in zip(handler.__closure__, handler.__code__.co_freevars, strict=False):
        if name == "http":
            cell.cell_contents = fake_http
            break

    return router, handler


# ---------------------------------------------------------------------------
# Tests: flujo de voz
# ---------------------------------------------------------------------------


class TestVoiceEndToEnd:
    """El flujo de voz se ejecuta correctamente de principio a fin.

    v1.2: estos tests asumen el flujo antiguo con mimo-v2.5 + input_audio.
    Se reescriben en S2.1.5 con STT queue (Gemini externo).
    Por ahora los saltamos para que CI no se cuelgue.
    """

    pytestmark = pytest.mark.skip(
        reason="Reescrito en S2.1.5: usa flujo voz con mimo-v2.5/input_audio obsoletos (bug #30389)"
    )

    @pytest.mark.asyncio
    async def test_voice_message_full_flow(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """Un mensaje de voz produce: descarga, base64, LLM call, response."""
        # 1. Mock bot.get_file
        file_path = "documents/voice/file_xyz.ogg"
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path=file_path)
        )

        # 2. Mock httpx.AsyncClient (download audio)
        audio_bytes = b"fake-ogg-bytes-for-test"

        # 3. Mock LLM (mimo-v2.5)
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "Transcripción: hola. Respuesta: ¡buenas!",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                },
            )
        )

        # 4. Build router and replace the http in the closure with our fake
        _router, handler = _build_router_with_fake_voice_http(
            bot, db, settings, telemetry, audio_bytes
        )

        msg, capture = make_voice_message(file_id="file_xyz")
        await handler(msg)

        # 5. Verify the response
        assert capture.count() == 2
        assert "pensando" in capture.calls[0]
        # Voice response has the 🎙️ prefix
        assert "🎙️" in capture.last()
        assert "Transcripción" in capture.last()

        # 6. Verify DB: 2 messages (user + assistant)

        async with db.conn.execute(
            "SELECT id FROM conversations WHERE chat_id=?", (TEST_CHAT_ID,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        conv_id = row["id"]

        async with db.conn.execute(
            "SELECT role, content, model_used, tokens_in, tokens_out "
            "FROM messages WHERE conversation_id=? ORDER BY id ASC",
            (conv_id,),
        ) as cur:
            messages = await cur.fetchall()
        assert len(messages) == 2
        # User message: stored as "[voice message]" placeholder
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "[voice message]"
        # Assistant message: from mimo-v2.5
        assert messages[1]["role"] == "assistant"
        assert messages[1]["model_used"] == "mimo-v2.5"
        assert "Transcripción" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_voice_message_handles_download_failure(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """Si la descarga del audio falla, el usuario recibe un error claro."""
        # Mock bot.get_file (success)
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path="voice/file.ogg")
        )

        # Mock httpx to return 500
        class FakeAsyncClient500:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def get(self, url):
                resp = MagicMock()
                resp.raise_for_status = MagicMock(side_effect=httpx.HTTPError("500 Server Error"))
                return resp

        with patch("hermes.handlers.messages.httpx.AsyncClient", FakeAsyncClient500):
            router = build_message_router(bot, db, settings, telemetry)
            handler = router.message.handlers[0].callback

            msg, capture = make_voice_message()
            await handler(msg)

        # User gets an error message
        assert capture.count() == 1
        assert "audio" in capture.last().lower()

        # No DB writes (no conversation, no messages)

        async with db.conn.execute(
            "SELECT id FROM conversations WHERE chat_id=?", (TEST_CHAT_ID,)
        ) as cur:
            row = await cur.fetchone()
        assert row is None

        # No LLM call
        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    async def test_voice_message_handles_get_file_failure(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """Si `bot.get_file()` falla (ej. file_id inválido), el usuario recibe un error."""
        # bot.get_file raises an exception
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            side_effect=Exception("file_id not found")
        )

        # Build router normally. The fake http is not needed because
        # the handler returns before calling it (bot.get_file fails first).
        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        msg, capture = make_voice_message()
        await handler(msg)

        assert capture.count() == 1
        assert "audio" in capture.last().lower()
        # No LLM call was made
        assert respx_mock.calls.call_count == 0
        # No DB writes

        async with db.conn.execute(
            "SELECT id FROM conversations WHERE chat_id=?", (TEST_CHAT_ID,)
        ) as cur:
            row = await cur.fetchone()
        assert row is None

    @pytest.mark.asyncio
    async def test_voice_message_default_format_when_no_mime_type(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """Sin mime_type, el formato por defecto es 'ogg'."""
        # bot.get_file
        file_path = "voice/file.ogg"
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path=file_path)
        )

        # Capture the LLM payload to verify format
        import json as _json

        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        respx_mock.post(openai_url).mock(side_effect=cb)

        # Voice without mime_type
        _router, handler = _build_router_with_fake_voice_http(
            bot, db, settings, telemetry, b"audio-data"
        )
        msg, _ = make_voice_message(mime_type=None)  # type: ignore[arg-type]
        await handler(msg)

        # Verify the audio format
        assert len(captured) == 1
        msgs = captured[0]["messages"]
        last_msg = msgs[-1]
        content = last_msg["content"]
        assert isinstance(content, list)
        # Find the input_audio block
        for block in content:
            if block.get("type") == "input_audio":
                assert block["input_audio"]["format"] == "ogg"
