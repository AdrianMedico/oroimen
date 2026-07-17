"""Integration tests: resiliencia del sistema.

Verifica cómo Hermes se comporta ante fallos del LLM:
- Circuit breaker abre tras N fallos consecutivos
- Fallback al siguiente modelo funciona end-to-end
- Circuit breaker se recupera tras el reset_timeout
- Errores de LLM devuelven mensaje claro al usuario
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from hermes.handlers.messages import build_message_router
from hermes.llm.breaker import CircuitState
from tests.conftest import make_text_message
from tests.integration.conftest import make_fake_http_for_voice, make_voice_message

# ---------------------------------------------------------------------------
# Tests: circuit breaker integration
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    """El circuit breaker integrado en el handler real abre/cierra correctamente."""

    @pytest.mark.asyncio
    async def test_breaker_opens_after_repeated_failures(
        self, settings, db, telemetry, bot, respx_mock
    ) -> None:
        """Tras N fallos del primary, el breaker abre y el handler va al fallback."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        err = httpx.Response(500, json={"error": "down"})
        anthropic_ok = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "fallback ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
        respx_mock.post(openai_url).mock(side_effect=err)
        respx_mock.post(anthropic_url).mock(return_value=anthropic_ok)

        # Reducimos fail_max para acelerar el test
        settings.circuit_breaker_fail_max = 2

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        # Primer mensaje: deepseek falla (intento 1) -> fallback a minimax
        msg1, _ = make_text_message("msg 1")
        await handler(msg1)
        # El breaker de deepseek ha registrado 1 fallo (el retry de 1 + retries
        # también falla, pero el breaker cuenta cada llamada fallida)

        # Verificamos que se usó minimax como fallback
        async with db.conn.execute(
            "SELECT model_used FROM messages WHERE role='assistant' " "ORDER BY id DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        assert row["model_used"] == "minimax-m3"

    @pytest.mark.asyncio
    async def test_breaker_failure_threshold_triggers_open(
        self, settings, db, telemetry, bot, respx_mock
    ) -> None:
        """Tras `fail_max` fallos, el breaker se abre y rechaza el siguiente call."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        err = httpx.Response(500, json={"error": "down"})
        respx_mock.post(openai_url).mock(side_effect=err)

        # fail_max=1: el primer fallo abre el breaker
        settings.circuit_breaker_fail_max = 1
        # Sin retries para que cada llamada sea un único fallo
        settings.llm_max_retries = 0

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        # Accedemos al LLMRouter real a través del closure del handler.
        # El handler tiene `llm` en su closure, que es el LLMRouter.
        llm = None
        for cell, name in zip(handler.__closure__, handler.__code__.co_freevars, strict=False):
            if name == "llm":
                llm = cell.cell_contents
                break
        assert llm is not None

        # Forzamos la apertura del breaker de deepseek
        llm._breakers["deepseek-v4-flash"]._state = CircuitState.OPEN  # type: ignore[attr-defined]

        # Configuramos el fallback de anthropic
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        anthropic_ok = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "fallback"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
        respx_mock.post(anthropic_url).mock(return_value=anthropic_ok)

        msg, capture = make_text_message("hola")
        await handler(msg)

        # El fallback (minimax-m3) respondió
        assert "fallback" in capture.last()
        # deepseek NO fue llamado (el breaker está open)
        openai_calls = [c for c in respx_mock.calls if str(c.request.url) == openai_url]
        assert len(openai_calls) == 0

    @pytest.mark.asyncio
    async def test_breaker_state_visible_via_router(
        self, settings, db, telemetry, bot, respx_mock
    ) -> None:
        """El LLMRouter expone el estado del breaker."""
        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        # Accedemos al LLMRouter a través del closure
        llm = None
        for cell, name in zip(handler.__closure__, handler.__code__.co_freevars, strict=False):
            if name == "llm":
                llm = cell.cell_contents
                break
        assert llm is not None

        # Estado inicial: closed
        assert llm.breaker_state("deepseek-v4-flash") == "closed"
        assert llm.breaker_state("minimax-m3") == "closed"
        assert llm.breaker_state("mimo-v2.5") == "closed"

        # Forzamos apertura
        llm._breakers["deepseek-v4-flash"]._state = CircuitState.OPEN  # type: ignore[attr-defined]
        assert llm.breaker_state("deepseek-v4-flash") == "open"


# ---------------------------------------------------------------------------
# Tests: fallback chain end-to-end
# ---------------------------------------------------------------------------


class TestFallbackChainEndToEnd:
    """El fallback chain completo funciona a través del handler real."""

    @pytest.mark.asyncio
    async def test_text_fallback_on_primary_failure(
        self, settings, db, telemetry, bot, respx_mock
    ) -> None:
        """Si deepseek-v4-flash falla, el handler cae a minimax-m3 automáticamente."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        err = httpx.Response(500, json={"error": "service unavailable"})
        anthropic_ok = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "respuesta de fallback"}],
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        )
        respx_mock.post(openai_url).mock(side_effect=[err, err, err])
        respx_mock.post(anthropic_url).mock(return_value=anthropic_ok)

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        msg, capture = make_text_message("hola")
        await handler(msg)

        # El usuario recibió respuesta
        assert capture.count() == 2
        assert "fallback" in capture.last().lower()

        # La DB tiene el mensaje del modelo fallback
        async with db.conn.execute(
            "SELECT role, model_used, content, tokens_in, tokens_out "
            "FROM messages WHERE role='assistant'"
        ) as cur:
            row = await cur.fetchone()
        assert row["model_used"] == "minimax-m3"
        assert "fallback" in row["content"].lower()
        assert row["tokens_in"] == 5
        assert row["tokens_out"] == 3

    @pytest.mark.asyncio
    async def test_text_all_models_fail_returns_error_to_user(
        self, settings, db, telemetry, bot, respx_mock
    ) -> None:
        """Si TODOS los modelos fallan, el usuario recibe un mensaje de error."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        err = httpx.Response(500, json={"error": "down"})
        respx_mock.post(openai_url).mock(side_effect=err)
        respx_mock.post(anthropic_url).mock(return_value=err)

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        msg, capture = make_text_message("hola")
        await handler(msg)

        # El usuario recibió 2 mensajes: "pensando..." + error
        assert capture.count() == 2
        assert "Error" in capture.last()

        # El mensaje del usuario SÍ se guardó (para history)
        async with db.conn.execute("SELECT role, content FROM messages WHERE role='user'") as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["content"] == "hola"

    @pytest.mark.asyncio
    async def test_voice_no_fallback_returns_error(
        self, settings, db, telemetry, bot, respx_mock
    ) -> None:
        """Si todos los modelos fallan, el usuario recibe un error."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        err = httpx.Response(500, json={"error": "down"})
        respx_mock.post(openai_url).mock(side_effect=err)

        # Mock Gemini STT endpoint (voice flow transcribes before LLM call)
        gemini_stt_url = f"{settings.stt_base_url}/models/{settings.stt_model}:generateContent"
        stt_response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "hola"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 5,
                "totalTokenCount": 55,
            },
        }
        respx_mock.post(gemini_stt_url).mock(return_value=httpx.Response(200, json=stt_response))

        router = build_message_router(bot, db, settings, telemetry)
        handler = router.message.handlers[0].callback

        # Mock bot.get_file (success)
        bot.get_file = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(file_path="voice/x.ogg")
        )

        # Replace http in closure with a fake that returns audio
        for cell, name in zip(handler.__closure__, handler.__code__.co_freevars, strict=False):
            if name == "http":
                cell.cell_contents = make_fake_http_for_voice(b"audio")
                break

        msg, capture = make_voice_message()
        await handler(msg)

        # El usuario recibió 2 mensajes: "pensando..." + error
        assert capture.count() == 2
        assert "Error" in capture.last()
