"""Integration tests: flujo de TEXTO end-to-end.

Verifica el flujo completo de un mensaje de texto:
1. Telegram message arrives
2. aiogram dispatcher routes to handler
3. Handler:
   - get_or_create_conversation
   - add_message (user)
   - get_history
   - call LLM (mocked with respx)
   - add_message (assistant)
   - record telemetry
4. Response is sent back via message.answer(...)

A diferencia de los unit tests, estos tests:
- Usan el Router REAL construido por `build_message_router`
- Verifican el camino completo: DB, LLM, respuesta
- Comprueban el estado FINAL del sistema (DB, response)
"""

from __future__ import annotations

import pytest

from hermes.handlers.messages import build_message_router
from tests.conftest import TEST_CHAT_ID, make_text_message

# ---------------------------------------------------------------------------
# Tests: flujo de texto básico
# ---------------------------------------------------------------------------


class TestTextEndToEnd:
    """El flujo de texto se ejecuta correctamente de principio a fin."""

    @pytest.mark.asyncio
    async def test_text_message_full_flow(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """Un mensaje de texto produce: DB write, LLM call, DB write, response."""
        import httpx

        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Hola desde el LLM"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )
        )

        # Build the full router
        router = build_message_router(bot, db, settings, telemetry)
        # Extract the message handler
        handler = router.message.handlers[0].callback

        # Create a message
        msg, capture = make_text_message("hola")

        # Execute the flow
        await handler(msg)

        # 1. Response was sent (2 calls: "pensando..." + final)
        assert capture.count() == 2
        assert "pensando" in capture.calls[0]
        assert "Hola desde el LLM" in capture.last()

        # 2. DB has 2 messages: user + assistant
        # Re-fetch conversation
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
        # User message
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hola"
        # Assistant message
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hola desde el LLM"
        assert messages[1]["model_used"] == "deepseek-v4-flash"
        assert messages[1]["tokens_in"] == 5
        assert messages[1]["tokens_out"] == 3

    @pytest.mark.asyncio
    async def test_text_message_uses_history_on_second_message(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """El segundo mensaje de un usuario incluye el history en el LLM call."""
        import json as _json

        import httpx

        # Mock LLM (capture all requests)
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        captured_requests: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured_requests.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        respx_mock.post(openai_url).mock(side_effect=cb)

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        # First message
        msg1, _ = make_text_message("primer mensaje")
        await handler(msg1)
        # Second message
        msg2, _ = make_text_message("segundo mensaje")
        await handler(msg2)

        # Two LLM calls were made
        assert len(captured_requests) == 2
        # Second call includes the history (user msg1, assistant resp1, user msg2)
        second = captured_requests[1]
        msgs = second["messages"]
        roles = [m["role"] for m in msgs]
        # system (hermes) + user (msg1) + assistant (resp1) + user (msg2 guardado) + user (msg2 current)
        assert roles[0] == "system"
        assert "user" in roles
        assert "assistant" in roles
        # Find the first user content (should be "primer mensaje")
        user_contents = [m["content"] for m in msgs if m["role"] == "user"]
        assert "primer mensaje" in user_contents
        assert "segundo mensaje" in user_contents

    @pytest.mark.asyncio
    async def test_text_message_creates_conversation_on_first_message(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """El primer mensaje de un user/chat crea una conversación nueva."""
        import httpx

        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        # Pre-condition: no conversations
        async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
            count = (await cur.fetchone())[0]
        assert count == 0

        msg, _ = make_text_message("hola", user_id=111, chat_id=222)
        await handler(msg)

        # Post-condition: 1 conversation
        async with db.conn.execute("SELECT chat_id, user_id FROM conversations") as cur:
            rows = await cur.fetchall()
        assert len(rows) == 1
        assert rows[0]["chat_id"] == 222
        assert rows[0]["user_id"] == 111

    @pytest.mark.asyncio
    async def test_text_message_reuses_existing_conversation(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """El segundo mensaje del mismo user/chat REUSA la conversación."""
        import httpx

        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        msg1, _ = make_text_message("msg 1")
        await handler(msg1)
        msg2, _ = make_text_message("msg 2")
        await handler(msg2)

        # Still only 1 conversation (reused)
        async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
            count = (await cur.fetchone())[0]
        assert count == 1

        # But 4 messages (2 user + 2 assistant)
        async with db.conn.execute("SELECT role FROM messages ORDER BY id ASC") as cur:
            roles = [r["role"] for r in await cur.fetchall()]
        assert roles == ["user", "assistant", "user", "assistant"]

    @pytest.mark.asyncio
    async def test_text_message_does_not_create_conversation_for_command(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """Comandos (texto que empieza por /) NO se procesan en message handler.

        El command router los maneja por separado (ver test_commands).
        """
        import httpx

        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        msg, capture = make_text_message("/start")
        await handler(msg)

        # No response, no DB writes, no LLM call
        assert capture.count() == 0
        async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
            count = (await cur.fetchone())[0]
        assert count == 0
        # No LLM call was made
        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    async def test_text_message_uses_correct_thread_id(
        self,
        settings,
        db,
        telemetry,
        bot,
        respx_mock,
    ) -> None:
        """Mensajes en threads distintos crean conversaciones distintas."""
        import httpx

        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        # Same user, same chat, but different thread
        msg1, _ = make_text_message("msg en thread 1", thread_id=1)
        await handler(msg1)
        msg2, _ = make_text_message("msg en thread 2", thread_id=2)
        await handler(msg2)

        # Two distinct conversations
        async with db.conn.execute("SELECT thread_id FROM conversations ORDER BY thread_id") as cur:
            threads = [r["thread_id"] for r in await cur.fetchall()]
        assert threads == [1, 2]
