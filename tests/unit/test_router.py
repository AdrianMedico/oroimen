"""Tests industriales para `hermes.llm.router.LLMRouter`.

Cubre:
- Detección de familia Anthropic (`is_anthropic_model`).
- Routing: texto (chain) y voz (mimo).
- Fallback chain completo.
- Path Anthropic API: payload, system prompt split, response parsing.
- Path OpenAI API: payload, response parsing, malformed response.
- Circuit breaker integration: skip cuando open, recovery.
- Retry logic: backoff, exhausted retries.
- Headers de autenticación.
- Edge cases: mensajes vacíos, caracteres especiales, latencia.

Estrategia:
- `Settings` real desde env (fake credenciales).
- `httpx.AsyncClient` interno del router, mockeado con `respx` (vía
  `base_url` configurado en `Settings`).
- `LLMRouter` real (no se mockea) — así probamos el comportamiento
  end-to-end del routing, fallback, breaker, retry.
- Cada test cierra el router (`router.aclose()`) en un `finally` para
  evitar warnings de httpx.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from hermes.config import Settings
from hermes.llm.router import (
    LLMError,
    LLMResponse,
    LLMRouter,
    _convert_openai_vision_to_anthropic,
    _transform_to_anthropic_messages,
    is_anthropic_model,
)

# ---------------------------------------------------------------------------
# Helper: crea y cierra un router en un bloque try/finally
# ---------------------------------------------------------------------------


async def with_router(settings: Settings, respx_mock, mock_setup) -> tuple[LLMRouter, Any]:
    """Crea un router, aplica mocks, devuelve (router, mock). El caller debe
    hacer `await router.aclose()` al final.

    En lugar de un context manager async (que LLMRouter no implementa),
    proporcionamos un helper que crea el router y registra los mocks.
    """
    mock_setup(respx_mock)
    router = LLMRouter(settings)
    return router, respx_mock


@pytest.mark.asyncio
async def test_cloud_client_configured_for_standard_test_settings(
    settings: Settings,
) -> None:
    router = LLMRouter(settings)
    try:
        assert router.cloud_client_configured is True
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_cloud_client_not_configured_for_local_defaults() -> None:
    router = LLMRouter(Settings(_env_file=None))
    try:
        assert router.cloud_client_configured is False
    finally:
        await router.aclose()


# ---------------------------------------------------------------------------
# Tests de `is_anthropic_model` (función pura)
# ---------------------------------------------------------------------------


class TestIsAnthropicModel:
    """Detección de familia Anthropic por prefijo de nombre de modelo."""

    def test_anthropic_families(self) -> None:
        """Familias Anthropic (prefijos `minimax-`, `qwen3.`) son detectadas.
        Esta funcion NO considera excepciones (ver `force_openai_for_tools`).
        """
        # minimax family
        assert is_anthropic_model("minimax-m3") is True
        assert is_anthropic_model("minimax-m2.7") is True
        # Case-insensitive
        assert is_anthropic_model("MiniMax-M3") is True
        # qwen3 family
        assert is_anthropic_model("qwen3.7-plus") is True
        assert is_anthropic_model("qwen3.5-plus") is True
        # Modelos hipotéticos futuros con la misma familia
        assert is_anthropic_model("minimax-m4-pro") is True
        assert is_anthropic_model("qwen3.8-max") is True

    def test_force_openai_for_tools(self) -> None:
        """Workaround bug 2013: modelos en `_ANTHROPIC_FAMILIES_EXCEPTIONS`
        usan path OpenAI cuando hay tools en la peticion.

        v0.5.7-revert (reintroducido): workaround para minimax-M3 tras
        descubrir que el fix de freegate PR #19 no resolvia el bug 2013
        en opencode-go + MiniMax.

        Sprint 12+ (decision conservadora): OpenCode Go sigue siendo
        opcion viable a futuro (la quota se recupera), asi que MANTENEMOS
        el workaround activo incluso cuando usamos MiniMax API direct.
        Cuando el workaround no aplica (MiniMax direct sin bug 2013),
        forzar a OpenAI es un noop (mismo path al que iria de todas
        formas si `is_anthropic_model` retornara False).

        Sprint 16.7 (PR #98, 2026-07-06): EXCEPTIONS emptied. Empirical
        test (Sprint 16, 30 facts qwen3 + 10 queries via MiniMax API
        direct) confirmed that forcing minimax-M3 to OpenAI path
        produces URL `/anthropic/v1/chat/completions` which does NOT
        exist on MiniMax API (404). Anthropic path
        (`/anthropic/v1/messages`) is correct for all cases. The
        bug 2013 workaround is no longer needed since we don't use
        opencode-go anymore.
        """
        from hermes.llm.router import force_openai_for_tools

        # Sprint 16.7: EXCEPTIONS is now empty. force_openai_for_tools
        # always returns False regardless of model. This is the new
        # contract: all anthropic-family models go to `/messages` path,
        # all openai-family models go to `/chat/completions` path,
        # no model-specific overrides.
        assert force_openai_for_tools("minimax-M3") is False
        assert force_openai_for_tools("MINIMAX-M3") is False
        assert force_openai_for_tools("minimax-m3") is False
        # Otros modelos minimax-* tambien NO forzar (mismo contrato)
        assert force_openai_for_tools("minimax-m4") is False
        assert force_openai_for_tools("minimax-m2") is False
        # qwen3.* nunca en EXCEPTIONS -> nunca forzar
        assert force_openai_for_tools("qwen3.7-plus") is False
        assert force_openai_for_tools("qwen3.8-max") is False
        # Modelos no-Anthropic no cambian
        assert force_openai_for_tools("deepseek-v4-flash") is False
        assert force_openai_for_tools("glm-5.2") is False

    @pytest.mark.asyncio
    async def test_minimax_m3_with_tools_uses_openai_path(
        self, settings: Settings, respx_mock
    ) -> None:
        """v0.5.7-revert: cuando minimax-m3 recibe tools, debe usar path
        OpenAI (workaround bug 2013 en opencode-go + MiniMax).

        Sin tools, minimax-m3 usa path Anthropic normal (test
        `TestAnthropicPath` cubre ese caso).

        Este test valida end-to-end que el router elige el path
        correcto segun la presencia de tools. El path OpenAI es el
        unico que funciona con opencode-go + MiniMax cuando hay
        tool_use/tool_result en el payload.
        """
        import json as _json

        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        captured: list[tuple[str, dict]] = []

        def cb_openai(req: httpx.Request) -> httpx.Response:
            captured.append(("openai", _json.loads(req.content)))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "ok openai",
                                "tool_calls": [
                                    {
                                        "id": "call_test_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"Madrid"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        def cb_anthropic(req: httpx.Request) -> httpx.Response:
            captured.append(("anthropic", _json.loads(req.content)))
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok anthropic"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        respx_mock.post(openai_url).mock(side_effect=cb_openai)
        respx_mock.post(anthropic_url).mock(side_effect=cb_anthropic)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        router = LLMRouter(settings)
        try:
            # Caso 1: CON tools -> debe ir a Anthropic (bug 2013 workaround
            # removed in Sprint 16.7, PR #98). Antes iba a OpenAI path con
            # tools (legacy opencode-go bug workaround), pero eso producia
            # un 404 en MiniMax API direct (path `/anthropic/v1/chat/completions`
            # no existe). Ahora todos los modelos anthropic-family van a
            # `/messages` (que es lo que MiniMax API soporta).
            resp = await router._invoke(  # type: ignore[attr-defined]
                "minimax-m3",
                [{"role": "user", "content": "tiempo en Madrid"}],
                0.5,
                tools=tools,
            )
            assert resp.content == "ok anthropic"
            assert len(captured) == 1
            assert captured[0][0] == "anthropic"
            # El payload Anthropic incluye tools en formato nativo
            # Anthropic (name, description, input_schema), no en formato
            # OpenAI (type: function, function: {...}).
            assert "tools" in captured[0][1]
            anthropic_tools = captured[0][1]["tools"]
            assert len(anthropic_tools) == 1
            assert anthropic_tools[0]["name"] == "get_weather"
            assert anthropic_tools[0]["description"] == "Get weather"
            assert anthropic_tools[0]["input_schema"] == {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }

            # Caso 2: SIN tools -> debe ir a Anthropic
            captured.clear()
            resp2 = await router._invoke(  # type: ignore[attr-defined]
                "minimax-m3",
                [{"role": "user", "content": "hola"}],
                0.5,
            )
            assert resp2.content == "ok anthropic"
            assert len(captured) == 1
            assert captured[0][0] == "anthropic"
            # El payload Anthropic NO incluye tools
            assert "tools" not in captured[0][1]
        finally:
            await router.aclose()

    def test_non_anthropic_families(self) -> None:
        """Otras familias NO son Anthropic."""
        # glm, deepseek, kimi, mimo, hy3
        for model in (
            "glm-5.2",
            "glm-5.1",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "kimi-k2.6",
            "kimi-k2.7-code",
            "mimo-v2.5",
            "mimo-v2.5-pro",
            "mimo-v2-omni",
            "hy3-preview",
        ):
            assert is_anthropic_model(model) is False, f"{model} no debería ser Anthropic"


# ---------------------------------------------------------------------------
# Tests del path OpenAI API
# ---------------------------------------------------------------------------


class TestOpenAIPath:
    """Tests del path OpenAI-compatible (`/chat/completions`)."""

    @pytest.mark.asyncio
    async def test_payload_includes_model_messages_temperature(
        self, settings: Settings, respx_mock
    ) -> None:
        """El payload a /chat/completions incluye model, messages, temperature."""
        import json as _json

        url = f"{settings.opencode_go_base_url}/chat/completions"
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

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router.chat(
                [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                ],
                is_voice=False,
                temperature=0.7,
            )
        finally:
            await router.aclose()

        assert len(captured) == 1
        payload = captured[0]
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["messages"] == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        assert payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_payload_includes_top_p_and_repetition_penalty(
        self, settings: Settings, respx_mock
    ) -> None:
        """SPRINT 18 HOTFIX (2026-07-08): el path OpenAI envia top_p +
        repetition_penalty para prevenir token repetition loops.

        Bug production: respuestas largas del modelo MiMo v2.5
        terminaban en loops con caracteres chinos random
        ('适合适配适配...'). Root cause: top_p=1.0 (desactivado) +
        sin repetition_penalty.
        """
        import json as _json

        url = f"{settings.opencode_go_base_url}/chat/completions"
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

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router.chat(
                [{"role": "user", "content": "u"}],
                is_voice=False,
                temperature=0.7,
            )
        finally:
            await router.aclose()

        assert len(captured) == 1
        payload = captured[0]
        # Top_p universal (standard OpenAI).
        assert payload["top_p"] == settings.llm_top_p == 0.9
        # repetition_penalty NO es OpenAI-standard pero MiniMax OpenAI
        # -compat lo acepta como "additional parameter". Ver docs
        # MiniMax: platform.minimax.io/docs/api-reference/text-openai-api
        assert payload["repetition_penalty"] == settings.llm_repetition_penalty == 1.04

    @pytest.mark.asyncio
    async def test_top_p_and_repetition_penalty_use_runtime_settings(
        self, settings: Settings, respx_mock
    ) -> None:
        """Los valores se leen de settings en runtime (no hardcoded).
        Si en el futuro subimos top_p a 0.95 o repetition_penalty a
        1.06, el wiring los recoge sin tocar codigo del router.
        """
        import json as _json

        # Override via Pydantic model_copy (mantiene validators).
        new_settings = settings.model_copy(
            update={"llm_top_p": 0.95, "llm_repetition_penalty": 1.06}
        )

        url = f"{settings.opencode_go_base_url}/chat/completions"
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

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(new_settings)
        try:
            await router.chat(
                [{"role": "user", "content": "u"}],
                is_voice=False,
                temperature=0.7,
            )
        finally:
            await router.aclose()

        assert len(captured) == 1
        payload = captured[0]
        assert payload["top_p"] == 0.95
        assert payload["repetition_penalty"] == 1.06

    @pytest.mark.asyncio
    async def test_openai_max_tokens_override_reaches_payload(
        self, settings: Settings, respx_mock
    ) -> None:
        """Oroimen Slice 1C1a: explicit `max_tokens` override reaches the
        /chat/completions payload when the caller provides one.
        """
        import json as _json

        url = f"{settings.opencode_go_base_url}/chat/completions"
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

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router.chat(
                [{"role": "user", "content": "u"}],
                temperature=0.5,
                chain_override=["deepseek-v4-flash"],
                max_tokens=2468,
            )
        finally:
            await router.aclose()

        assert len(captured) == 1
        payload = captured[0]
        assert payload["max_tokens"] == 2468

    @pytest.mark.asyncio
    async def test_openai_payload_omits_max_tokens_when_no_override(
        self, settings: Settings, respx_mock
    ) -> None:
        """Oroimen Slice 1C1a: without an override, the /chat/completions
        payload must NOT carry a `max_tokens` field. This preserves
        backward compatibility for ordinary chat() callers (no override)
        and OpenAI providers that reject unknown keys.
        """
        import json as _json

        url = f"{settings.opencode_go_base_url}/chat/completions"
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

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router.chat(
                [{"role": "user", "content": "u"}],
                temperature=0.5,
                chain_override=["deepseek-v4-flash"],
            )
        finally:
            await router.aclose()

        assert len(captured) == 1
        payload = captured[0]
        assert "max_tokens" not in payload

    @pytest.mark.asyncio
    async def test_uses_default_temperature_when_not_provided(
        self, settings: Settings, respx_mock
    ) -> None:
        """Si temperature es None, usa settings.llm_temperature.

        Sprint 12+: el default cambio de 0.3 (calibrado para opencode-go)
        a 1.0 (recomendado por MiniMax). El test verifica la regla
        misma (que el router usa settings.llm_temperature cuando no se
        provee temperatura explicita), no un valor hardcoded, para
        ser robusto a futuros cambios del default.
        """
        import json as _json

        url = f"{settings.opencode_go_base_url}/chat/completions"
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

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router.chat([{"role": "user", "content": "u"}], is_voice=False)
        finally:
            await router.aclose()

        # El router usa settings.llm_temperature cuando no se pasa
        # temperature explicita al .chat(). Verificamos la regla, no
        # un valor hardcoded (1.0 MiniMax, 0.3 opencode-go legacy).
        assert captured[0]["temperature"] == settings.llm_temperature

    @pytest.mark.asyncio
    async def test_response_parsed_with_usage(self, settings: Settings, respx_mock) -> None:
        """Extrae content y usage de la respuesta OpenAI."""
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                },
            )
        )

        router = LLMRouter(settings)
        try:
            resp = await router.chat([{"role": "user", "content": "u"}], is_voice=False)
        finally:
            await router.aclose()

        assert resp.content == "hello"
        assert resp.tokens_in == 12
        assert resp.tokens_out == 8
        assert resp.latency_ms >= 0
        assert resp.model == "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_response_without_usage_returns_zero_tokens(
        self, settings: Settings, respx_mock
    ) -> None:
        """Si la respuesta no tiene `usage`, tokens_in/out son 0."""
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )
        )

        router = LLMRouter(settings)
        try:
            resp = await router.chat([{"role": "user", "content": "u"}], is_voice=False)
        finally:
            await router.aclose()

        assert resp.tokens_in == 0
        assert resp.tokens_out == 0

    @pytest.mark.asyncio
    async def test_malformed_openai_response_raises(self, settings: Settings, respx_mock) -> None:
        """Respuesta malformada (sin `choices`) → LLMError.

        Llamamos directamente a `_invoke_openai` (no a `chat()`) para
        evitar el fallback chain, que activaría el modelo Anthropic.
        """
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={"unexpected_field": "no choices here"},
            )
        )

        router = LLMRouter(settings)
        try:
            with pytest.raises(LLMError, match="Malformed OpenAI response"):
                await router._invoke_openai(  # type: ignore[attr-defined]
                    "deepseek-v4-flash",
                    [{"role": "user", "content": "u"}],
                    0.5,
                )
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_http_error_wrapped_in_llm_error(self, settings: Settings, respx_mock) -> None:
        """HTTPError (4xx/5xx) se convierte en LLMError.

        Llamamos directamente a `_invoke_openai` para evitar el chain
        de fallback.
        """
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(return_value=httpx.Response(500))

        router = LLMRouter(settings)
        try:
            with pytest.raises(LLMError, match="HTTP error"):
                await router._invoke_openai(  # type: ignore[attr-defined]
                    "deepseek-v4-flash",
                    [{"role": "user", "content": "u"}],
                    0.5,
                )
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_invalid_json_response_raises_llm_error(
        self, settings: Settings, respx_mock
    ) -> None:
        """Respuesta no-JSON (HTML de proxy roto, texto plano) → LLMError.

        Caso real: opencode-go o un proxy delante puede responder con
        HTML (página de error de Cloudflare, 502 Bad Gateway) en vez
        de JSON. Sin este fix, `resp.json()` lanza `JSONDecodeError`
        que NO está capturada y rompe el chain de fallback.
        """
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body>502 Bad Gateway</body></html>",
            )
        )

        router = LLMRouter(settings)
        try:
            with pytest.raises(LLMError, match="Invalid JSON from openai"):
                await router._invoke_openai(  # type: ignore[attr-defined]
                    "deepseek-v4-flash",
                    [{"role": "user", "content": "u"}],
                    0.5,
                )
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_invalid_json_response_logs_raw_snippet(
        self, settings: Settings, respx_mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Si la respuesta no es JSON, se loggea un snippet crudo truncado.

        Necesario para debug en producción: cuando opencode-go rompe,
        queremos ver QUÉ devolvió (HTML, vacío, basura) sin filtrar
        grandes volúmenes al log.
        """
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html>Cloudflare error: gateway timeout</html>",
            )
        )

        router = LLMRouter(settings)
        try:
            with caplog.at_level("WARNING"), pytest.raises(LLMError):
                await router._invoke_openai(  # type: ignore[attr-defined]
                    "deepseek-v4-flash",
                    [{"role": "user", "content": "u"}],
                    0.5,
                )
            # El log debe contener el evento y el snippet
            assert any("llm_invalid_json_response" in r.message for r in caplog.records)
            assert any(
                "Cloudflare error" in str(r.__dict__.get("raw_snippet", "")) for r in caplog.records
            )
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_response_not_dict_raises_llm_error(self, settings: Settings, respx_mock) -> None:
        """JSON válido pero NO es dict (null, list, string) → LLMError.

        Caso real: el provider responde con `null`, `[1,2,3]` o `"error"`
        (válidos JSON) pero no el dict esperado.
        """
        import json as _json

        url = f"{settings.opencode_go_base_url}/chat/completions"
        for invalid_payload in (None, [1, 2, 3], "plain string", 42):
            respx_mock.post(url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    content=_json.dumps(invalid_payload).encode(),
                )
            )
            router = LLMRouter(settings)
            try:
                with pytest.raises(LLMError, match="Unexpected JSON shape"):
                    await router._invoke_openai(  # type: ignore[attr-defined]
                        "deepseek-v4-flash",
                        [{"role": "user", "content": "u"}],
                        0.5,
                    )
            finally:
                await router.aclose()

    @pytest.mark.asyncio
    async def test_choices_field_wrong_type_raises_llm_error(
        self, settings: Settings, respx_mock
    ) -> None:
        """`choices` existe pero no es lista → LLMError (captura AttributeError).

        Sin este fix, `data["choices"][0]["message"]["content"]` lanza
        `TypeError: string indices must be integers` (si choices es str)
        o `AttributeError` (si choices es None).
        """
        url = f"{settings.opencode_go_base_url}/chat/completions"
        for invalid_choices in ("not a list", None, {"0": "fake"}):
            respx_mock.post(url).mock(
                return_value=httpx.Response(
                    200,
                    json={"choices": invalid_choices, "usage": {}},
                )
            )
            router = LLMRouter(settings)
            try:
                with pytest.raises(LLMError, match="Malformed OpenAI response"):
                    await router._invoke_openai(  # type: ignore[attr-defined]
                        "deepseek-v4-flash",
                        [{"role": "user", "content": "u"}],
                        0.5,
                    )
            finally:
                await router.aclose()


# ---------------------------------------------------------------------------
# Tests del path Anthropic API
# ---------------------------------------------------------------------------


class TestAnthropicPath:
    """Tests del path Anthropic-compatible (`/messages`)."""

    @pytest.mark.asyncio
    async def test_payload_includes_model_max_tokens_temperature(
        self, settings: Settings, respx_mock
    ) -> None:
        """Payload a /messages incluye model, max_tokens, temperature, messages (sin system)."""
        import json as _json

        url = f"{settings.opencode_go_base_url}/messages"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            # Llamamos directo al invoke privado para forzar el path Anthropic
            await router._invoke_anthropic(  # type: ignore[attr-defined]
                "minimax-m3",
                [{"role": "user", "content": "hola"}],
                0.5,
            )
        finally:
            await router.aclose()

        assert len(captured) == 1
        payload = captured[0]
        assert payload["model"] == "minimax-m3"
        assert payload["max_tokens"] == settings.llm_max_tokens
        assert payload["temperature"] == 0.5
        # El user message está en messages (NO en system)
        assert payload["messages"] == [{"role": "user", "content": "hola"}]
        # No hay system prompt → no se incluye el campo
        assert "system" not in payload

    @pytest.mark.asyncio
    async def test_anthropic_max_tokens_override_replaces_default(
        self, settings: Settings, respx_mock
    ) -> None:
        """Oroimen Slice 1C1a: explicit max_tokens override reaches the
        /messages payload. When the caller passes `max_tokens`, that
        value wins over settings.llm_max_tokens.
        """
        import json as _json

        url = f"{settings.opencode_go_base_url}/messages"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router.chat(
                [{"role": "user", "content": "hola"}],
                temperature=0.5,
                chain_override=["minimax-m3"],
                max_tokens=1234,
            )
        finally:
            await router.aclose()

        assert len(captured) == 1
        payload = captured[0]
        # Override is forwarded verbatim; settings.llm_max_tokens
        # default is NOT used in its place.
        assert payload["max_tokens"] == 1234
        assert payload["max_tokens"] != settings.llm_max_tokens

    @pytest.mark.asyncio
    async def test_payload_includes_top_p_but_not_repetition_penalty(
        self, settings: Settings, respx_mock
    ) -> None:
        """SPRINT 18 HOTFIX (2026-07-08): path Anthropic-compat envia
        top_p (standard Anthropic) pero NO repetition_penalty
        (Anthropic spec rechaza campos no estándar con 400).

        Trade-off: Anthropic path puede tener repetition loops
        residuales. Fix futuro: detectar path y mapear a
        `frequency_penalty` (OpenAI-standard equivalente).
        """
        import json as _json

        url = f"{settings.opencode_go_base_url}/messages"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router._invoke_anthropic(  # type: ignore[attr-defined]
                "minimax-m3",
                [{"role": "user", "content": "hola"}],
                0.5,
            )
        finally:
            await router.aclose()

        assert len(captured) == 1
        payload = captured[0]
        # top_p SI se envia (Anthropic lo soporta oficialmente).
        assert payload["top_p"] == settings.llm_top_p == 0.9
        # repetition_penalty NO se envia (Anthropic rechaza campos
        # no estándar con 400 Bad Request).
        assert "repetition_penalty" not in payload

    @pytest.mark.asyncio
    async def test_system_prompt_is_split_out(self, settings: Settings, respx_mock) -> None:
        """El system prompt se extrae a un campo `system` separado."""
        import json as _json

        url = f"{settings.opencode_go_base_url}/messages"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router._invoke_anthropic(  # type: ignore[attr-defined]
                "minimax-m3",
                [
                    {"role": "system", "content": "Eres un asistente"},
                    {"role": "user", "content": "hola"},
                ],
                0.5,
            )
        finally:
            await router.aclose()

        payload = captured[0]
        assert payload["system"] == "Eres un asistente"
        # El messages solo contiene user/assistant, NO system
        assert payload["messages"] == [{"role": "user", "content": "hola"}]

    @pytest.mark.asyncio
    async def test_multiple_system_messages_concatenated(
        self, settings: Settings, respx_mock
    ) -> None:
        """Múltiples system messages se concatenan con doble newline."""
        import json as _json

        url = f"{settings.opencode_go_base_url}/messages"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router._invoke_anthropic(  # type: ignore[attr-defined]
                "minimax-m3",
                [
                    {"role": "system", "content": "primera parte"},
                    {"role": "system", "content": "segunda parte"},
                    {"role": "user", "content": "hola"},
                ],
                0.5,
            )
        finally:
            await router.aclose()

        assert captured[0]["system"] == "primera parte\n\nsegunda parte"

    @pytest.mark.asyncio
    async def test_response_with_multiple_text_blocks(self, settings: Settings, respx_mock) -> None:
        """Múltiples bloques de texto se concatenan en content."""
        url = f"{settings.opencode_go_base_url}/messages"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "Hola, "},
                        {"type": "text", "text": "mundo"},
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            )
        )

        router = LLMRouter(settings)
        try:
            resp = await router._invoke_anthropic(  # type: ignore[attr-defined]
                "minimax-m3",
                [{"role": "user", "content": "u"}],
                0.5,
            )
        finally:
            await router.aclose()

        assert resp.content == "Hola, mundo"
        assert resp.tokens_in == 5
        assert resp.tokens_out == 3

    @pytest.mark.asyncio
    async def test_response_with_non_text_blocks_ignored(
        self, settings: Settings, respx_mock
    ) -> None:
        """Bloques que no son `text` (ej. tool_use) se ignoran."""
        url = f"{settings.opencode_go_base_url}/messages"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "Respuesta: "},
                        {"type": "tool_use", "id": "tool_1", "name": "x"},
                        {"type": "text", "text": "fin."},
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        )

        router = LLMRouter(settings)
        try:
            resp = await router._invoke_anthropic(  # type: ignore[attr-defined]
                "minimax-m3",
                [{"role": "user", "content": "u"}],
                0.5,
            )
        finally:
            await router.aclose()

        # Solo los bloques text se concatenan
        assert resp.content == "Respuesta: fin."

    @pytest.mark.asyncio
    async def test_response_with_empty_content_returns_empty_string(
        self, settings: Settings, respx_mock
    ) -> None:
        """Si `content` está ausente o vacío, se devuelve string vacío (no error).

        La implementación es tolerante: la ausencia de `content` se
        trata como respuesta vacía. El test verifica este comportamiento.
        """
        url = f"{settings.opencode_go_base_url}/messages"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={"unexpected_field": "no content here"},
            )
        )

        router = LLMRouter(settings)
        try:
            resp = await router._invoke_anthropic(  # type: ignore[attr-defined]
                "minimax-m3",
                [{"role": "user", "content": "u"}],
                0.5,
            )
            # Sin error, content es string vacío
            assert resp.content == ""
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_assistant_tool_calls_transformed_to_tool_use_blocks(
        self, settings: Settings, respx_mock
    ) -> None:
        """Assistant con tool_calls (formato OpenAI) se transforma a tool_use blocks.

        v0.4.3: cuando el AgentLoop reconstruye el history, los assistant
        messages con tool_calls vienen en formato OpenAI ({id, type,
        function: {name, arguments as JSON string}}). El router Anthropic
        los transforma a bloques tool_use dentro del content con arguments
        como dict.
        """
        import json as _json

        url = f"{settings.opencode_go_base_url}/messages"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            messages = [
                {"role": "user", "content": "qué hora es"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_current_time",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
            ]
            await router._invoke_anthropic(  # type: ignore[attr-defined]
                "minimax-m3",
                messages,
                0.5,
            )
        finally:
            await router.aclose()

        # El assistant message se transformó a tool_use blocks
        payload = captured[0]
        assistant_msg = payload["messages"][1]
        assert assistant_msg["role"] == "assistant"
        # El content ahora es una lista de blocks
        content_blocks = assistant_msg["content"]
        assert isinstance(content_blocks, list)
        tool_use_blocks = [b for b in content_blocks if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["id"] == "call_abc"
        assert tool_use_blocks[0]["name"] == "get_current_time"
        # El argumento (JSON string en OpenAI) se parsea a dict
        assert tool_use_blocks[0]["input"] == {}

    @pytest.mark.asyncio
    async def test_assistant_text_plus_tool_calls_preserves_both(
        self, settings: Settings, respx_mock
    ) -> None:
        """Assistant con content de texto + tool_calls: ambos se preservan.

        v0.4.3: el content de texto se añade como bloque text, y los
        tool_calls como bloques tool_use, todo dentro del content.
        """
        import json as _json

        url = f"{settings.opencode_go_base_url}/messages"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            messages = [
                {
                    "role": "assistant",
                    "content": "Voy a consultar el tiempo",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Madrid"}',
                            },
                        }
                    ],
                },
            ]
            await router._invoke_anthropic(  # type: ignore[attr-defined]
                "minimax-m3",
                messages,
                0.5,
            )
        finally:
            await router.aclose()

        payload = captured[0]
        assistant_msg = payload["messages"][0]
        content_blocks = assistant_msg["content"]
        # Hay 2 blocks: text + tool_use
        text_blocks = [b for b in content_blocks if b["type"] == "text"]
        tool_use_blocks = [b for b in content_blocks if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Voy a consultar el tiempo"
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["input"] == {"city": "Madrid"}

    @pytest.mark.asyncio
    async def test_invalid_json_response_raises_llm_error_anthropic(
        self, settings: Settings, respx_mock
    ) -> None:
        """Anthropic: respuesta no-JSON (HTML) → LLMError."""
        url = f"{settings.opencode_go_base_url}/messages"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html>502 Bad Gateway</html>",
            )
        )

        router = LLMRouter(settings)
        try:
            with pytest.raises(LLMError, match="Invalid JSON from anthropic"):
                await router._invoke_anthropic(  # type: ignore[attr-defined]
                    "minimax-m3",
                    [{"role": "user", "content": "u"}],
                    0.5,
                )
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_response_not_dict_raises_llm_error_anthropic(
        self, settings: Settings, respx_mock
    ) -> None:
        """Anthropic: JSON válido pero no dict → LLMError."""
        import json as _json

        url = f"{settings.opencode_go_base_url}/messages"
        for invalid_payload in (None, [1, 2, 3], "plain string"):
            respx_mock.post(url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    content=_json.dumps(invalid_payload).encode(),
                )
            )
            router = LLMRouter(settings)
            try:
                with pytest.raises(LLMError, match="Unexpected JSON shape"):
                    await router._invoke_anthropic(  # type: ignore[attr-defined]
                        "minimax-m3",
                        [{"role": "user", "content": "u"}],
                        0.5,
                    )
            finally:
                await router.aclose()

    @pytest.mark.asyncio
    async def test_content_blocks_not_list_raises_llm_error(
        self, settings: Settings, respx_mock
    ) -> None:
        """Anthropic: `content` no es lista → LLMError (captura AttributeError).

        Sin este fix, `data.get("content", [])` devuelve el valor tal
        cual (e.g. un dict), y el iter / block.get falla con
        `AttributeError: 'dict' object has no attribute 'get'`.
        """
        url = f"{settings.opencode_go_base_url}/messages"
        for invalid_content in (None, {"text": "x"}, "single string", 42):
            respx_mock.post(url).mock(
                return_value=httpx.Response(
                    200,
                    json={"content": invalid_content, "usage": {}},
                )
            )
            router = LLMRouter(settings)
            try:
                with pytest.raises(LLMError, match="Malformed Anthropic response"):
                    await router._invoke_anthropic(  # type: ignore[attr-defined]
                        "minimax-m3",
                        [{"role": "user", "content": "u"}],
                        0.5,
                    )
            finally:
                await router.aclose()


# ---------------------------------------------------------------------------
# Tests de headers de autenticación
# ---------------------------------------------------------------------------


class TestAuthHeaders:
    """Los headers de autenticación se configuran correctamente en el cliente."""

    @pytest.mark.asyncio
    async def test_authorization_header_is_bearer(self, settings: Settings, respx_mock) -> None:
        """El header Authorization lleva `Bearer <api_key>`."""
        url = f"{settings.opencode_go_base_url}/chat/completions"
        captured_headers: dict[str, str] = {}

        def cb(req: httpx.Request) -> httpx.Response:
            for k, v in req.headers.items():
                captured_headers[k] = v
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router.chat([{"role": "user", "content": "u"}], is_voice=False)
        finally:
            await router.aclose()

        # El header Authorization es "Bearer <api_key>"
        assert "authorization" in captured_headers
        assert captured_headers["authorization"] == f"Bearer {settings.opencode_go_api_key}"

    @pytest.mark.asyncio
    async def test_x_api_key_header_present(self, settings: Settings, respx_mock) -> None:
        """El header `x-api-key` está presente en TODOS los requests.

        Por qué: los modelos Anthropic-style de opencode-go (minimax-m3,
        minimax-m2.7, qwen3.7-max/plus, qwen3.6-plus) requieren el header
        `x-api-key` en vez de `Authorization: Bearer`. Sin este header,
        devuelven 401 "Missing API key" incluso con la misma API key que
        funciona en endpoints OpenAI-style.

        Bug histórico (S2.1.6 post-deploy): sin este header, minimax-m3
        fallaba con 401 y caía al fallback deepseek-v4-flash, añadiendo
        3-5s de latencia innecesaria.

        Ver docs oficiales: https://opencode.ai/docs/es/go/#endpoints
        """
        url = f"{settings.opencode_go_base_url}/chat/completions"
        captured_headers: dict[str, str] = {}

        def cb(req: httpx.Request) -> httpx.Response:
            for k, v in req.headers.items():
                captured_headers[k] = v
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router.chat([{"role": "user", "content": "u"}], is_voice=False)
        finally:
            await router.aclose()

        # x-api-key está presente y tiene el mismo valor que la API key.
        # Los modelos OpenAI-style lo ignoran; los Anthropic-style lo requieren.
        assert "x-api-key" in captured_headers
        assert captured_headers["x-api-key"] == settings.opencode_go_api_key

    @pytest.mark.asyncio
    async def test_content_type_is_json(self, settings: Settings, respx_mock) -> None:
        """El header Content-Type es application/json."""
        url = f"{settings.opencode_go_base_url}/chat/completions"
        captured_headers: dict[str, str] = {}

        def cb(req: httpx.Request) -> httpx.Response:
            for k, v in req.headers.items():
                captured_headers[k] = v
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            await router.chat([{"role": "user", "content": "u"}], is_voice=False)
        finally:
            await router.aclose()

        assert "content-type" in captured_headers
        assert "application/json" in captured_headers["content-type"]


# ---------------------------------------------------------------------------
# Tests de fallback chain completo
# ---------------------------------------------------------------------------


class TestFallbackChain:
    """El chain de fallback funciona end-to-end."""

    @pytest.mark.asyncio
    async def test_text_chain_primary_then_fallback(self, settings: Settings, respx_mock) -> None:
        """Si deepseek-v4-flash falla, cae a minimax-m3 (Anthropic)."""
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
        respx_mock.post(openai_url).mock(side_effect=[err, err, err])
        respx_mock.post(anthropic_url).mock(return_value=anthropic_ok)

        router = LLMRouter(settings)
        try:
            resp = await router.chat([{"role": "user", "content": "u"}], is_voice=False)
        finally:
            await router.aclose()

        assert resp.model == "minimax-m3"
        assert resp.content == "fallback ok"

    @pytest.mark.asyncio
    async def test_voice_chain_unified_falls_back(self, settings: Settings, respx_mock) -> None:
        """v1.2: voz usa el mismo chain que texto (deepseek → minimax).

        Si deepseek falla, fallback a minimax. Ya no hay
        NoVoiceFallbackError porque voz y texto comparten chain.
        """
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        err = httpx.Response(500, json={"error": "down"})
        ok_anthropic = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "voz fallback ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
        respx_mock.post(openai_url).mock(side_effect=[err, err, err])
        respx_mock.post(anthropic_url).mock(return_value=ok_anthropic)

        router = LLMRouter(settings)
        try:
            resp = await router.chat(
                [{"role": "user", "content": "hola"}],
                is_voice=True,
            )
            assert resp.model == "minimax-m3"
            assert resp.content == "voz fallback ok"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_voice_both_fail_raises_llm_error(self, settings: Settings, respx_mock) -> None:
        """Si TODOS los modelos del chain unificado fallan, LLMError."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        err = httpx.Response(500, json={"error": "down"})
        respx_mock.post(openai_url).mock(side_effect=[err, err, err])
        respx_mock.post(anthropic_url).mock(return_value=err)

        router = LLMRouter(settings)
        try:
            with pytest.raises(LLMError):
                await router.chat(
                    [{"role": "user", "content": "hola"}],
                    is_voice=True,
                )
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_text_all_fail_raises_llm_error(self, settings: Settings, respx_mock) -> None:
        """Si TODOS los modelos del chain fallan, LLMError."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        err = httpx.Response(500, json={"error": "down"})
        respx_mock.post(openai_url).mock(side_effect=[err, err, err])
        respx_mock.post(anthropic_url).mock(return_value=err)

        router = LLMRouter(settings)
        try:
            with pytest.raises(LLMError):
                await router.chat([{"role": "user", "content": "u"}], is_voice=False)
        finally:
            await router.aclose()


# ---------------------------------------------------------------------------
# Tests de circuit breaker integration
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    """El router respeta el circuit breaker."""

    @pytest.mark.asyncio
    async def test_breaker_records_success(self, settings: Settings, respx_mock) -> None:
        """Tras un éxito, el breaker para ese modelo está `closed`."""
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

        router = LLMRouter(settings)
        try:
            await router.chat([{"role": "user", "content": "u"}], is_voice=False)
            # Verificamos que el breaker para deepseek-v4-flash está closed
            state = router.breaker_state("deepseek-v4-flash")
            assert state == "closed"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_breaker_records_failure(self, settings: Settings, respx_mock) -> None:
        """Tras un fallo, el breaker abre (con fail_max=1).

        Llamamos a `_call_with_breaker` directamente para evitar el
        chain de fallback. Como fail_max=1, el primer fallo abre el
        breaker, y los siguientes intentos lanzan CircuitOpenError
        (que también se acepta como LLMError-compatible para este test).
        """
        from hermes.llm.breaker import CircuitOpenError

        url = f"{settings.opencode_go_base_url}/chat/completions"
        err = httpx.Response(500, json={"error": "down"})
        respx_mock.post(url).mock(side_effect=err)

        # Necesitamos fail_max pequeño para que se abra rápido
        settings.circuit_breaker_fail_max = 1
        # Sin retries para que el primer fallo abra el breaker
        settings.llm_max_retries = 0

        router = LLMRouter(settings)
        try:
            with pytest.raises((LLMError, CircuitOpenError)):
                await router._call_with_breaker(  # type: ignore[attr-defined]
                    router._breakers["deepseek-v4-flash"],  # type: ignore[attr-defined]
                    "deepseek-v4-flash",
                    [{"role": "user", "content": "u"}],
                    0.5,
                )
            # El breaker para deepseek-v4-flash debe estar open
            state = router.breaker_state("deepseek-v4-flash")
            assert state == "open"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_open_breaker_skips_model(self, settings: Settings, respx_mock) -> None:
        """Si el breaker de un modelo está open, el router lo salta y va al fallback."""
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        anthropic_ok = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "fallback"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
        # deepseek NO debe ser llamado (su breaker está open)
        respx_mock.post(anthropic_url).mock(return_value=anthropic_ok)

        router = LLMRouter(settings)
        try:
            # Forzamos el breaker de deepseek a open directamente
            from hermes.llm.breaker import CircuitState

            router._breakers["deepseek-v4-flash"]._state = CircuitState.OPEN  # type: ignore[attr-defined]

            resp = await router.chat([{"role": "user", "content": "u"}], is_voice=False)
            # El fallback (minimax) se usó
            assert resp.model == "minimax-m3"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_breaker_state_method(self, settings: Settings) -> None:
        """`breaker_state(model)` devuelve el estado actual del breaker."""
        router = LLMRouter(settings)
        try:
            # Estado inicial: closed (todos los modelos del chain unificado)
            assert router.breaker_state("deepseek-v4-flash") == "closed"
            assert router.breaker_state("minimax-m3") == "closed"
            # v1.2: mimo-v2.5 ya no se inicializa (fuera del chain unificado)
            # pero preguntar por su estado debe devolver algo sensato
            # (no AttributeError). Si se pregunta por un modelo no inicializado,
            # el router lo crea on-demand.
            state_mimo = router.breaker_state("mimo-v2.5")
            assert state_mimo in ("closed", "open", "half-open")
        finally:
            await router.aclose()


# ---------------------------------------------------------------------------
# Tests de retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """El router reintenta con backoff en caso de fallo."""

    @pytest.mark.asyncio
    async def test_retry_on_5xx(self, settings: Settings, respx_mock) -> None:
        """Si el primer intento falla con 5xx, se reintenta."""
        url = f"{settings.opencode_go_base_url}/chat/completions"
        ok = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
        # 1er intento: 500. 2º intento: 200
        respx_mock.post(url).mock(side_effect=[httpx.Response(500), ok])

        # Reducimos max_retries a 1 para test rápido
        settings.llm_max_retries = 1

        router = LLMRouter(settings)
        try:
            resp = await router.chat([{"role": "user", "content": "u"}], is_voice=False)
            # El segundo intento (retry) tuvo éxito
            assert resp.content == "ok"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self, settings: Settings, respx_mock) -> None:
        """Si TODOS los reintentos fallan, se lanza LLMError."""
        openai_url = f"{settings.opencode_go_base_url}/chat/completions"
        anthropic_url = f"{settings.opencode_go_base_url}/messages"
        err = httpx.Response(500, json={"error": "down"})
        # deepseek: 3 fallos (1 intento + 2 retries)
        respx_mock.post(openai_url).mock(side_effect=[err, err, err])
        respx_mock.post(anthropic_url).mock(return_value=err)

        router = LLMRouter(settings)
        try:
            with pytest.raises(LLMError):
                await router.chat([{"role": "user", "content": "u"}], is_voice=False)
        finally:
            await router.aclose()


# ---------------------------------------------------------------------------
# Tests de edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Casos límite del router."""

    @pytest.mark.asyncio
    async def test_empty_messages_array(self, settings: Settings, respx_mock) -> None:
        """Un array de mensajes vacío se procesa (envía al LLM, que probablemente rechace)."""
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
            )
        )

        router = LLMRouter(settings)
        try:
            resp = await router.chat([], is_voice=False)
            # El LLM respondió con algo
            assert resp.content == "ok"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_special_unicode_characters(self, settings: Settings, respx_mock) -> None:
        """Caracteres Unicode (emojis, acentos, CJK) no rompen el routing."""
        import json as _json

        url = f"{settings.opencode_go_base_url}/chat/completions"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "🎉 ¡Hola! 你好"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=cb)

        router = LLMRouter(settings)
        try:
            resp = await router.chat(
                [{"role": "user", "content": "🎉 ¡Hola! 你好 こんにちは"}],
                is_voice=False,
            )
            # El contenido viaja íntegro
            assert captured[0]["messages"][0]["content"] == "🎉 ¡Hola! 你好 こんにちは"
            assert resp.content == "🎉 ¡Hola! 你好"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_latency_is_measured(self, settings: Settings, respx_mock) -> None:
        """`latency_ms` refleja el tiempo real del request."""
        url = f"{settings.opencode_go_base_url}/chat/completions"

        async def delayed_response(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)  # 50ms
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        respx_mock.post(url).mock(side_effect=delayed_response)

        router = LLMRouter(settings)
        try:
            resp = await router.chat([{"role": "user", "content": "u"}], is_voice=False)
            # La latencia debe ser al menos 50ms
            assert resp.latency_ms >= 50
            # Y no excesiva (< 5s, holgura amplia)
            assert resp.latency_ms < 5000
        finally:
            await router.aclose()

    def test_voice_chain_unified_with_text_chain(self, settings: Settings) -> None:
        """v1.2: voice_chain == text_chain (unificado tras bug #30389).

        Antes había un chain separado solo con mimo-v2.5. El bug #30389
        confirmó que mimo-v2.5 no procesa audio via Go. Ahora voz y
        texto usan el mismo chain (deepseek-v4-flash → minimax-m3 en
        este fixture; o minimax-m3 → deepseek-v4-flash en v1.2).
        """
        chain = settings.voice_chain
        assert chain == settings.text_chain
        assert len(chain) == 2

    def test_text_chain_order(self, settings: Settings) -> None:
        """El chain de texto es [primary, fallback]."""
        chain = settings.text_chain
        assert chain[0] == "deepseek-v4-flash"  # primary
        assert chain[1] == "minimax-m3"  # fallback
        assert len(chain) == 2


# ---------------------------------------------------------------------------
# Tests S2.1.4: voice_chain unificado con smart routing v1.2
# ---------------------------------------------------------------------------


class TestVoiceChainUnifiedV12:
    """Verifica el comportamiento v1.2: voz y texto usan el mismo chain.

    El fixture `settings_v12` configura el smart routing invertido
    (minimax-m3 → deepseek-v4-flash). Esto refleja la decisión de v1.2
    de priorizar minimax (cuota x3, más capaz).
    """

    def test_v12_settings_have_minimax_primary(self, settings_v12: Settings) -> None:
        """v1.2: minimax-m3 es el primary del chain."""
        assert settings_v12.llm_text_primary == "minimax-m3"
        assert settings_v12.llm_text_fallback == "deepseek-v4-flash"
        assert settings_v12.text_chain == ["minimax-m3", "deepseek-v4-flash"]

    def test_v12_voice_chain_equals_text_chain(self, settings_v12: Settings) -> None:
        """v1.2: voice_chain == text_chain (unificado, no hay mimo)."""
        assert settings_v12.voice_chain == settings_v12.text_chain
        # mimo-v2.5 NO está en el chain
        assert "mimo-v2.5" not in settings_v12.voice_chain

    @pytest.mark.asyncio
    async def test_v12_chat_voice_uses_minimax_first(
        self, settings_v12: Settings, respx_mock
    ) -> None:
        """v1.2: is_voice=True intenta minimax-m3 primero (cuota x3)."""
        anthropic_url = f"{settings_v12.opencode_go_base_url}/messages"
        captured: list[dict] = []

        def cb(req: httpx.Request) -> httpx.Response:
            import json as _json

            captured.append(_json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "voz ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        respx_mock.post(anthropic_url).mock(side_effect=cb)

        router = LLMRouter(settings_v12)
        try:
            resp = await router.chat(
                [{"role": "user", "content": "hola"}],
                is_voice=True,
            )
            assert resp.model == "minimax-m3"
            assert resp.content == "voz ok"
            # El payload se envió al endpoint Anthropic-compatible
            assert captured[0]["model"] == "minimax-m3"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_v12_chat_voice_falls_back_to_deepseek(
        self, settings_v12: Settings, respx_mock
    ) -> None:
        """v1.2: si minimax falla, fallback a deepseek-v4-flash."""
        anthropic_url = f"{settings_v12.opencode_go_base_url}/messages"
        openai_url = f"{settings_v12.opencode_go_base_url}/chat/completions"
        err = httpx.Response(500, json={"error": "minimax down"})
        ok_openai = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "deepseek ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
        respx_mock.post(anthropic_url).mock(return_value=err)
        respx_mock.post(openai_url).mock(return_value=ok_openai)

        router = LLMRouter(settings_v12)
        try:
            resp = await router.chat(
                [{"role": "user", "content": "hola"}],
                is_voice=True,
            )
            assert resp.model == "deepseek-v4-flash"
            assert resp.content == "deepseek ok"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_v12_chat_text_same_chain_as_voice(
        self, settings_v12: Settings, respx_mock
    ) -> None:
        """v1.2: texto y voz usan exactamente el mismo chain."""
        anthropic_url = f"{settings_v12.opencode_go_base_url}/messages"
        ok = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
        respx_mock.post(anthropic_url).mock(return_value=ok)

        router = LLMRouter(settings_v12)
        try:
            # Texto
            resp_text = await router.chat([{"role": "user", "content": "hola"}], is_voice=False)
            # Voz
            resp_voice = await router.chat([{"role": "user", "content": "hola"}], is_voice=True)
            # Ambos deben usar minimax-m3 (primary del chain unificado)
            assert resp_text.model == "minimax-m3"
            assert resp_voice.model == "minimax-m3"
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_no_voice_fallback_error_raised(self, settings_v12: Settings, respx_mock) -> None:
        """v1.2: NoVoiceFallbackError ya no se lanza (chain unificado).

        Test de regresión: el router NO debe importar ni lanzar
        NoVoiceFallbackError, que era del diseño v1.0/v1.1 cuando
        voz solo tenía mimo-v2.5.
        """
        from hermes.llm import router as router_module

        # NoVoiceFallbackError ya no debe existir en el módulo
        assert not hasattr(router_module, "NoVoiceFallbackError"), (
            "NoVoiceFallbackError no debería existir en v1.2 (chain unificado)"
        )


# ---------------------------------------------------------------------------
# Sprint 5 T51: reasoning_content passthrough
# ---------------------------------------------------------------------------


class TestReasoningContentPassthrough:
    """Tests Sprint 5 T51: passthrough de reasoning_content end-to-end.

    Cobertura:
    - LLMResponse default factory
    - _invoke_openai extrae reasoning_content del JSON
    - _invoke_openai normaliza None/ausente a cadena vacia
    - _transform_to_anthropic_messages purga campos OpenAI-specific
      (TDD T51 v2 §3.7, fix de bug cross-contamination detectado en review)
    """

    def test_llm_response_default_reasoning_content_empty(self) -> None:
        """LLMResponse sin reasoning_content explicito usa default ''.

        Backwards compat: tests existentes que construyen LLMResponse
        sin pasar reasoning_content siguen funcionando.
        """
        resp = LLMResponse(
            content="ok",
            model="minimax-m3",
            tokens_in=10,
            tokens_out=5,
            latency_ms=100,
        )
        assert resp.reasoning_content == ""

    @pytest.mark.asyncio
    async def test_invoke_openai_extracts_reasoning_content(
        self, settings: Settings, respx_mock
    ) -> None:
        """_invoke_openai captura reasoning_content del response JSON.

        Escenario: deepseek-v4-flash responde con thinking mode.
        Hermes debe extraer el campo para persistirlo en DB.
        """
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "Resumen del video",
                                "reasoning_content": (
                                    "El usuario quiere un resumen. "
                                    "Voy a estructurar puntos clave..."
                                ),
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                },
            )
        )

        router = LLMRouter(settings)
        try:
            response = await router.chat([{"role": "user", "content": "resume"}])
        finally:
            await router.aclose()

        assert response.content == "Resumen del video"
        assert response.reasoning_content == (
            "El usuario quiere un resumen. Voy a estructurar puntos clave..."
        )

    @pytest.mark.asyncio
    async def test_invoke_openai_handles_missing_reasoning_content(
        self, settings: Settings, respx_mock
    ) -> None:
        """Si el response no incluye reasoning_content, queda en ''.

        Providers sin thinking mode (o modelos con la feature desactivada)
        devuelven JSON sin el campo. Hermes debe normalizar a cadena vacia
        para que el caller no tenga que distinguir entre None y ''.
        """
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

        router = LLMRouter(settings)
        try:
            response = await router.chat([{"role": "user", "content": "hola"}])
        finally:
            await router.aclose()

        assert response.reasoning_content == ""

    @pytest.mark.asyncio
    async def test_invoke_openai_handles_null_reasoning_content(
        self, settings: Settings, respx_mock
    ) -> None:
        """Si reasoning_content viene explicitamente null, queda en ''.

        Algunos providers devuelven `reasoning_content: null` cuando
        thinking mode esta desactivado. `or ""` lo normaliza.
        """
        url = f"{settings.opencode_go_base_url}/chat/completions"
        respx_mock.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                                "reasoning_content": None,
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )

        router = LLMRouter(settings)
        try:
            response = await router.chat([{"role": "user", "content": "hola"}])
        finally:
            await router.aclose()

        assert response.reasoning_content == ""


class TestTransformToAnthropicMessages:
    """Tests del helper _transform_to_anthropic_messages (T51 §3.7).

    Este helper es el corazon del fix de cross-contamination OpenAI->Anthropic
    detectado en review Gemini 3.5 Thinking de T51 v1.
    """

    def test_passes_through_user_messages(self) -> None:
        """Mensajes user pasan tal cual (role + content)."""
        out = _transform_to_anthropic_messages([{"role": "user", "content": "Hola"}])
        assert out == [{"role": "user", "content": "Hola"}]

    def test_strips_openai_reasoning_content(self) -> None:
        """CRITICO: assistant con reasoning_content lo ve purgado.

        Escenario de cross-contamination (TDD T51 §3.3.1):
        Turno previo con deepseek-v4-flash populo reasoning_content.
        Turno siguiente con minimax-m3 (Anthropic) lee historial.
        El else branch debe reconstruir el dict con solo role+content.
        """
        user_assistant = [
            {"role": "user", "content": "Resume el video"},
            {
                "role": "assistant",
                "content": "Te resumo el video",
                "reasoning_content": "Pensamiento largo de DeepSeek...",
            },
            {"role": "user", "content": "Gracias"},
        ]
        out = _transform_to_anthropic_messages(user_assistant)
        assert len(out) == 3
        # El assistant message NO debe contener reasoning_content
        asst_msg = out[1]
        assert asst_msg == {"role": "assistant", "content": "Te resumo el video"}
        assert "reasoning_content" not in asst_msg
        assert "tool_call_id" not in asst_msg

    def test_transforms_role_tool_to_user_with_tool_result(self) -> None:
        """role='tool' se transforma a role='user' con bloque tool_result."""
        out = _transform_to_anthropic_messages(
            [
                {
                    "role": "tool",
                    "content": "<tool_output>...</tool_output>",
                    "tool_call_id": "call_123",
                }
            ]
        )
        assert out == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "<tool_output>...</tool_output>",
                    }
                ],
            }
        ]

    def test_transforms_assistant_with_tool_calls_to_tool_use_blocks(self) -> None:
        """Assistant con tool_calls se convierte a bloques tool_use."""
        out = _transform_to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": '{"tz": "Europe/Madrid"}',
                            },
                        }
                    ],
                }
            ]
        )
        assert out == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_xyz",
                        "name": "get_time",
                        "input": {"tz": "Europe/Madrid"},
                    }
                ],
            }
        ]

    def test_assistant_with_content_and_tool_calls_includes_text_block(self) -> None:
        """Assistant con content + tool_calls genera bloques [text, tool_use]."""
        out = _transform_to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "Voy a llamar a la herramienta",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "x", "arguments": "{}"},
                        }
                    ],
                }
            ]
        )
        assert out == [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Voy a llamar a la herramienta"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "x",
                        "input": {},
                    },
                ],
            }
        ]

    def test_strips_arbitrary_openai_specific_fields(self) -> None:
        """Cualquier campo OpenAI-specific se purga en el else branch.

        Defense in depth (T51 §3.7): si en el futuro openai/responses
        introduce un nuevo campo (refusal, audio, etc.), este helper
        lo ignora sin necesidad de mantenimiento.
        """
        out = _transform_to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "respuesta",
                    "reasoning_content": "pensamiento",
                    "refusal": None,
                    "audio": {"id": "audio_1"},
                    "function_call": None,
                }
            ]
        )
        assert out == [{"role": "assistant", "content": "respuesta"}]
        # Verificar explicitamente que NINGUN campo extra se filtra
        msg = out[0]
        assert set(msg.keys()) == {"role", "content"}

    def test_handles_empty_input(self) -> None:
        """Lista vacia retorna lista vacia sin error."""
        assert _transform_to_anthropic_messages([]) == []


# ----------------------------------------------------------------------------
# SPRINT 18 HOTFIX (2026-07-08): OpenAI vision → Anthropic format
# conversion. Bug: MiniMax Anthropic-compat endpoint
# (https://api.minimax.io/anthropic/v1/messages) rechaza image_url blocks
# con HTTP 400, rompiendo TODA la chain cuando WebUI adjunta una imagen.
# Fix: _transform_to_anthropic_messages detecta content=list y convierte
# via _convert_openai_vision_to_anthropic.
# ----------------------------------------------------------------------------


class TestConvertOpenAIVisionToAnthropic:
    """SPRINT 18 HOTFIX: vision format conversion OpenAI -> Anthropic."""

    def test_data_url_becomes_base64_image_block(self) -> None:
        """data:image/png;base64,XXX → Anthropic base64 image block."""
        out = _convert_openai_vision_to_anthropic(
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                    },
                }
            ]
        )
        assert out == [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
                },
            }
        ]

    def test_data_url_with_jpeg_media_type(self) -> None:
        """JPEG data URL parsea media_type correctamente (no asume PNG)."""
        out = _convert_openai_vision_to_anthropic(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ"},
                }
            ]
        )
        assert out[0]["source"]["media_type"] == "image/jpeg"
        assert out[0]["source"]["data"] == "/9j/4AAQSkZJRgABAQ"

    def test_http_url_becomes_url_image_block(self) -> None:
        """HTTP URL → Anthropic url source (no base64)."""
        out = _convert_openai_vision_to_anthropic(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/screenshot.png"},
                }
            ]
        )
        assert out == [
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/screenshot.png"},
            }
        ]

    def test_mixed_text_and_image_preserves_order(self) -> None:
        """Multi-block content: text pasa tal cual, image se convierte.

        Reproduce WebUI vision message format:
        [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
        """
        out = _convert_openai_vision_to_anthropic(
            [
                {"type": "text", "text": "Que hay en esta imagen?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/webp;base64,XYZ"},
                },
            ]
        )
        assert len(out) == 2
        # text pasa tal cual (OpenAI y Anthropic usan el mismo format).
        assert out[0] == {"type": "text", "text": "Que hay en esta imagen?"}
        # image se convierte a Anthropic format.
        assert out[1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/webp",
                "data": "XYZ",
            },
        }

    def test_malformed_data_url_is_skipped(self) -> None:
        """data URL malformada (sin comma separador) se skipea.

        Defensa: perder 1 imagen < romper chat. Un input que NO es
        data: URL se trata como HTTP URL (otro path). Pero un data:
        URL sin comma no se puede parsear → skip.
        """
        out = _convert_openai_vision_to_anthropic(
            [
                {"type": "text", "text": "pre"},
                # data: URL malformada: "data:image/png;base64" sin el
                # comma + base64 data. No se puede parsear → skip.
                {"type": "image_url", "image_url": {"url": "data:image/png;base64"}},
                {"type": "text", "text": "post"},
            ]
        )
        # text blocks pasan, image malformada se skip.
        assert len(out) == 2
        assert out[0]["text"] == "pre"
        assert out[1]["text"] == "post"

    def test_empty_list_returns_empty_list(self) -> None:
        """Lista vacia → lista vacia."""
        assert _convert_openai_vision_to_anthropic([]) == []

    def test_unknown_block_type_passes_through(self) -> None:
        """Block type desconocido pasa tal cual (no silent drop)."""
        out = _convert_openai_vision_to_anthropic(
            [{"type": "future_block_type", "data": "unknown"}]
        )
        assert out == [{"type": "future_block_type", "data": "unknown"}]


class TestTransformAnthropicVisionEndToEnd:
    """End-to-end via _transform_to_anthropic_messages.

    Verifica que la integracion del helper en el transform principal
    detecta content=list y aplica la conversion.
    """

    def test_user_message_with_vision_list(self) -> None:
        """User message con vision list → Anthropic format con image blocks."""
        out = _transform_to_anthropic_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAA"},
                        },
                    ],
                }
            ]
        )
        assert len(out) == 1
        assert out[0]["role"] == "user"
        assert isinstance(out[0]["content"], list)
        assert out[0]["content"][0] == {"type": "text", "text": "Describe"}
        assert out[0]["content"][1]["type"] == "image"
        assert out[0]["content"][1]["source"]["type"] == "base64"

    def test_user_message_with_text_only_passes_through_unchanged(self) -> None:
        """User message con text plano (no vision) sigue funcionando."""
        out = _transform_to_anthropic_messages([{"role": "user", "content": "Hola"}])
        # Comportamiento existente (test_passes_through_user_messages).
        assert out == [{"role": "user", "content": "Hola"}]

    def test_assistant_message_with_vision_list(self) -> None:
        """Assistant message con vision list → conversion aplicada.

        Edge case SUGGESTION 2 (MiMo v2.5 review 2026-07-08): aunque
        assistant vision es inusual, el path deberia funcionar para
        cualquier role. Cubre el caso donde el assistant envia una
        imagen (e.g., tool_use result con preview).
        """
        out = _transform_to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Aqui tienes:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/preview.png"},
                        },
                    ],
                }
            ]
        )
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert isinstance(out[0]["content"], list)
        assert out[0]["content"][0] == {"type": "text", "text": "Aqui tienes:"}
        assert out[0]["content"][1] == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/preview.png"},
        }


class TestVisionDefenseGaps:
    """SUGGESTION 1 (MiMo v2.5 review 2026-07-08): defense gaps en el
    conversion helper. Cubre casos donde input es valido segun el type
    del block pero el payload interno esta vacio o ausente.
    """

    def test_empty_url_string_is_skipped(self) -> None:
        """image_url con url='' → skip block (no crea source url vacia)."""
        out = _convert_openai_vision_to_anthropic(
            [
                {"type": "text", "text": "pre"},
                {"type": "image_url", "image_url": {"url": ""}},
                {"type": "text", "text": "post"},
            ]
        )
        # text blocks pasan, image con url vacia se skip.
        assert len(out) == 2
        assert out[0]["text"] == "pre"
        assert out[1]["text"] == "post"
        # Ningun image block con source.url = "" debe quedar.
        for block in out:
            if block.get("type") == "image":
                assert block["source"].get("url") != ""

    def test_missing_image_url_field_is_skipped(self) -> None:
        """image_url sin campo 'url' (image_url vacio) → skip block."""
        out = _convert_openai_vision_to_anthropic(
            [
                {"type": "image_url", "image_url": {}},
            ]
        )
        assert out == []

    def test_missing_image_url_dict_is_skipped(self) -> None:
        """image_url sin dict 'image_url' → skip block."""
        out = _convert_openai_vision_to_anthropic(
            [
                {"type": "image_url"},
            ]
        )
        assert out == []


class TestUnknownBlockRecovery:
    """NIT 4 (MiMo v2.5 review 2026-07-08): pass-through de unknown
    block type NO debe romper el procesamiento de los bloques siguientes.

    Comportamiento esperado: unknown block pasa tal cual (defensa:
    no silenciar content del user), pero el siguiente bloque valido
    se procesa normalmente.
    """

    def test_unknown_block_then_valid_image(self) -> None:
        """Lista mixta: unknown → valid image. Ambos se preservan."""
        out = _convert_openai_vision_to_anthropic(
            [
                {"type": "future_block_type", "data": "preserved"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,XYZ"},
                },
            ]
        )
        assert len(out) == 2
        # Unknown block pasa tal cual.
        assert out[0] == {"type": "future_block_type", "data": "preserved"}
        # La imagen DESPUES del unknown block se procesa normalmente.
        assert out[1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "XYZ",
            },
        }

    def test_invalid_image_then_text_is_recovered(self) -> None:
        """Lista: image malformada → text. Text no se pierde."""
        out = _convert_openai_vision_to_anthropic(
            [
                {"type": "image_url", "image_url": {"url": "data:image/png"}},
                {"type": "text", "text": "texto importante"},
            ]
        )
        assert len(out) == 1
        assert out[0]["text"] == "texto importante"


class TestSSRFDefense:
    """SUGGESTION 1 (MiMo v2.5 review r2 2026-07-08): defense contra
    non-http URL schemes que pasarian tal cual a Anthropic.

    Risk class: SSRF/leaking. file:/// lee filesystem local,
    javascript: ejecuta code, data: ya manejado en otro branch.
    Whitelist http/https; skip+warning cualquier otro.
    """

    def test_file_scheme_url_is_skipped(self) -> None:
        """file:///etc/passwd → skip (no llega a Anthropic)."""
        out = _convert_openai_vision_to_anthropic(
            [
                {"type": "text", "text": "pre"},
                {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
            ]
        )
        assert len(out) == 1
        assert out[0]["text"] == "pre"

    def test_javascript_scheme_url_is_skipped(self) -> None:
        """javascript: URLs → skip (no llega a Anthropic)."""
        out = _convert_openai_vision_to_anthropic(
            [{"type": "image_url", "image_url": {"url": "javascript:alert(1)"}}]
        )
        assert out == []

    def test_data_scheme_still_works(self) -> None:
        """Regression: data: URL sigue funcionando (handled en otro branch)."""
        out = _convert_openai_vision_to_anthropic(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAA"},
                }
            ]
        )
        assert len(out) == 1
        assert out[0]["source"]["type"] == "base64"


class TestDataURLDefaults:
    """SUGGESTION 2 (MiMo v2.5 review r2 2026-07-08): data: URLs sin
    media_type explicito (`data:;base64,...`) deben defaultear a
    image/png en vez de mandar media_type vacio (que Anthropic rechaza
    con 400).
    """

    def test_data_url_without_mediatype_defaults_to_png(self) -> None:
        """data:;base64,XXX → media_type=image/png (no vacio)."""
        out = _convert_openai_vision_to_anthropic(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:;base64,AAA"},
                }
            ]
        )
        assert len(out) == 1
        assert out[0]["source"]["media_type"] == "image/png"
        assert out[0]["source"]["data"] == "AAA"

    def test_uppercase_https_scheme_passes(self) -> None:
        """BLOCKING #2 (Nemotron r3 2026-07-08): HTTP:// scheme
        case-insensitive (RFC 3986). url.lower() comparison."""
        out = _convert_openai_vision_to_anthropic(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "HTTPS://EXAMPLE.COM/X.PNG"},
                }
            ]
        )
        assert len(out) == 1
        assert out[0]["source"]["url"] == "HTTPS://EXAMPLE.COM/X.PNG"

    def test_empty_content_list_falls_through(self) -> None:
        """SUGGESTION 4 (Nemotron r3 2026-07-08): content=[] cae a
        string vacio en vez de mandar array vacio a Anthropic.
        """
        out = _transform_to_anthropic_messages([{"role": "user", "content": []}])
        # Empty list → fallback to empty string (Anthropic safe).
        assert out == [{"role": "user", "content": ""}]

    def test_unsupported_mediatype_falls_back_to_png(self) -> None:
        """SUGGESTION 3 (Nemotron r4 2026-07-08): media_types fuera
        de la allowlist de Anthropic (image/jpeg, image/png, image/gif,
        image/webp) caen a image/png en vez de mandar un valor que
        Anthropic rechaza con 400.
        """
        out = _convert_openai_vision_to_anthropic(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/svg+xml;base64,XYZ"},
                }
            ]
        )
        assert len(out) == 1
        assert out[0]["source"]["media_type"] == "image/png"
        # Data original se preserva.
        assert out[0]["source"]["data"] == "XYZ"

    def test_supported_mediatypes_pass_through(self) -> None:
        """Regression: los 4 media_types validos de Anthropic pasan
        tal cual sin fallback a image/png.
        """
        for mt in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            out = _convert_openai_vision_to_anthropic(
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mt};base64,AAA"},
                    }
                ]
            )
            assert out[0]["source"]["media_type"] == mt, f"failed for {mt}"
