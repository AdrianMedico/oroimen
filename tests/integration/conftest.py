"""Shared fixtures and helpers for integration tests.

Most fixtures are inherited from tests/conftest.py (settings, db,
telemetry, bot, _AnswerCapture, get_handler, make_text_message).

This conftest adds:
- A `message_router` fixture that returns the FULL Router (with both
  command and message routers registered, like in production).
- A helper `make_voice_message_with_file_id` that returns a Message
  mock with a Voice that can be used with our fake http factory.
- A `fake_http_for_voice` helper factory for mocking Telegram file
  downloads (respx doesn't intercept httpx.AsyncClient() created
  inside functions).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from aiogram.types import Chat, User, Voice

# Re-export shared fixtures from the parent conftest
from tests.conftest import (  # noqa: F401  (re-exported for tests)
    TEST_BOT_TOKEN,
    TEST_CHAT_ID,
    TEST_USER_ID,
    _AnswerCapture,
    bot,
    db,
    get_handler,
    make_text_message,
    settings,
    telemetry,
)

# ---------------------------------------------------------------------------
# Message factory for voice messages (integration-specific)
# ---------------------------------------------------------------------------


def make_voice_message(
    text: str | None = None,
    *,
    user_id: int = TEST_USER_ID,
    chat_id: int = TEST_CHAT_ID,
    file_id: str = "voice_file_abc",
    file_unique_id: str | None = None,
    duration: int = 5,
    mime_type: str | None = "audio/ogg",
    file_size: int = 1024,
    thread_id: int | None = None,
):
    """Mensaje de voz (Message.voice != None, text = None).

    Crea un Message con aiogram.types.Voice real y `_AnswerCapture` para
    capturar las llamadas a `message.answer(...)`.
    """

    user = User(id=user_id, is_bot=False, first_name="Test")
    chat = Chat(id=chat_id, type="private")
    voice = Voice(
        file_id=file_id,
        file_unique_id=file_unique_id or f"uniq_{file_id}",
        duration=duration,
        mime_type=mime_type,  # type: ignore[arg-type]
        file_size=file_size,
    )
    capture = _AnswerCapture()
    msg = MagicMock(spec=["from_user", "chat", "text", "voice", "message_thread_id", "answer"])
    msg.from_user = user
    msg.chat = chat
    msg.text = text
    msg.voice = voice
    msg.message_thread_id = thread_id
    msg.answer = capture
    return msg, capture


# ---------------------------------------------------------------------------
# Fake http factory for voice downloads
# ---------------------------------------------------------------------------


def make_fake_http_for_voice(audio_bytes: bytes = b"fake-ogg-bytes"):
    """Devuelve un MagicMock que simula httpx.AsyncClient para descarga de voz.

    El handler de voz crea un `httpx.AsyncClient()` internamente, y respx
    no intercepta clientes nuevos creados dentro de funciones. Por eso
    usamos un fake.

    Args:
        audio_bytes: bytes que devolverá el fake `get(url)`.

    Returns:
        MagicMock con `get = AsyncMock(...)` que devuelve un objeto con
        `content = audio_bytes` y `raise_for_status()` no-op.
    """
    client = MagicMock()

    async def get_factory(url: str):
        resp = MagicMock()
        resp.content = audio_bytes
        resp.raise_for_status = MagicMock()
        return resp

    client.get = AsyncMock(side_effect=get_factory)
    return client
