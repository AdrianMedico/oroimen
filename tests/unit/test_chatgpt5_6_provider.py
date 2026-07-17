"""Tests for the ChatGPT 5.6 frontier provider (Sprint 19.6+ Phase 4).

Cubre:
- Settings validation: opt-in semantics, API key required when enabled
- Settings: text_chain_full reflects enabled state
- Client: request shape (URL, headers, payload)
- Client: response parsing (text content, latency, model, tokens)
- Client: error cases (401, 429, 5xx, network, timeout, malformed)
- Client: circuit breaker integration (opens after N fails, recovery)
- Router integration: opt-in (no client when disabled)
- Router integration: chat() silently skips frontier when not enabled
- Router integration: chat() dispatches to frontier client when enabled

Estrategia:
- `Settings` real desde env (fake credenciales, autouse via conftest).
- `httpx.AsyncClient` mockeado con `respx` (el client del frontier
  tiene su propio base_url, separado del main router).
- `LLMRouter` real para validar la integración end-to-end.
- Cada test cierra el router (`router.aclose()`) en un `finally` para
  evitar warnings de httpx.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
import respx

from hermes.config import Settings
from hermes.llm.breaker import CircuitOpenError
from hermes.llm.chatgpt5_6 import ChatGpt5_6Client
from hermes.llm.router import LLMError, LLMRouter

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

# Fake OpenAI API key for tests. Real OpenAI keys start with `sk-` or
# `sk-proj-`; we use a longer-than-10-chars string to satisfy any
# min_length validators. OpenAI's real keys are 50+ chars; we use a
# shorter one for readability in test output.
FAKE_OPENAI_KEY = "sk-test-fake-key-for-unit-tests-1234567890"

# Default frontier model name (matches Settings default).
FRONTIER_MODEL = "gpt-5.6-sol"

# Default OpenAI base URL (matches Settings default).
OPENAI_BASE_URL = "https://api.openai.com/v1"

# ---------------------------------------------------------------------------
# Autouse fixture: required env vars for Settings construction
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_required_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set required env vars for Settings construction (autouse).

    The Settings class requires `opencode_go_api_key` (min_length=10)
    and `gemini_api_key` (min_length=10) before any frontier tests
    can construct Settings. We set them via monkeypatch (which
    auto-cleans up after each test) so the test code itself doesn't
    reference the env-var names (avoids scrubber false positives on
    field-name substrings). Mirrors the pattern in
    `tests/unit/test_local_vision_ocr.py`.
    """
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")


# Standard OpenAI Chat Completions response payload builder.
def _openai_response(
    *,
    content: str = "ok",
    model: str = FRONTIER_MODEL,
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    tool_calls: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a realistic OpenAI /v1/chat/completions response body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# 1. Settings validation
# ---------------------------------------------------------------------------


class TestFrontierSettings:
    """Config validation for the frontier tier."""

    def test_default_disabled_no_key_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default: frontier disabled, no API key needed. Settings constructs cleanly.

        The frontier is opt-in (Sprint 19 north star: no automatic
        cloud calls). The default state should NOT require any env
        var to be set; users explicitly enable it.
        """
        # Default settings (from conftest) should construct without
        # any frontier env vars. The conftest already sets required
        # base env vars; we don't touch the frontier ones.
        s = Settings(_env_file=None)
        assert s.llm_text_frontier_enabled is False
        assert s.llm_text_frontier_api_key == ""
        assert s.llm_text_frontier_model == FRONTIER_MODEL
        # Default base URL is OpenAI's standard.
        assert s.llm_text_frontier_base_url == "https://api.openai.com/v1"
        # text_chain_full == text_chain when disabled (no frontier added).
        assert s.text_chain_full == s.text_chain
        assert len(s.text_chain_full) == 2

    def test_full_chain_without_cloud_keys_is_local_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The normal router chain must not add an uncredentialed cloud fallback."""
        monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
        monkeypatch.delenv("LLM_TEXT_FRONTIER__API_KEY", raising=False)
        monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "false")
        s = Settings(_env_file=None)
        assert s.text_chain == [s.llm_text_primary]
        assert s.text_chain_full == [s.llm_text_primary]

    def test_frontier_does_not_enable_uncredentialed_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frontier opt-in appends GPT without silently enabling MiniMax."""
        monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
        monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "true")
        monkeypatch.setenv("LLM_TEXT_FRONTIER__API_KEY", FAKE_OPENAI_KEY)
        s = Settings(_env_file=None)
        assert s.text_chain_full == [s.llm_text_primary, FRONTIER_MODEL]

    def test_enabled_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_TEXT_FRONTIER__ENABLED=true with no API key → ValueError at startup.

        This prevents the silent failure mode: "I enabled the
        frontier but the calls fail because the key is missing."
        Fail fast at startup instead.
        """
        monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "true")
        # LLM_TEXT_FRONTIER__API_KEY intentionally NOT set (empty).
        monkeypatch.delenv("LLM_TEXT_FRONTIER__API_KEY", raising=False)
        with pytest.raises(Exception, match="LLM_TEXT_FRONTIER__API_KEY"):
            Settings(_env_file=None)

    def test_enabled_with_key_constructs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_TEXT_FRONTIER__ENABLED=true with API key → Settings constructs.

        The text_chain_full now has 3 positions (primary, fallback, frontier).
        """
        monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "true")
        monkeypatch.setenv("LLM_TEXT_FRONTIER__API_KEY", FAKE_OPENAI_KEY)
        s = Settings(_env_file=None)
        assert s.llm_text_frontier_enabled is True
        assert s.llm_text_frontier_api_key == FAKE_OPENAI_KEY
        # text_chain_full = text_chain + frontier (3 positions)
        assert len(s.text_chain_full) == 3
        assert s.text_chain_full[2] == s.llm_text_frontier_model
        # text_chain is unchanged (backward compat).
        assert len(s.text_chain) == 2

    def test_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_TEXT_FRONTIER__MODEL can override the configured default."""
        monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "true")
        monkeypatch.setenv("LLM_TEXT_FRONTIER__API_KEY", FAKE_OPENAI_KEY)
        monkeypatch.setenv("LLM_TEXT_FRONTIER__MODEL", "gpt-5.6-terra")
        s = Settings(_env_file=None)
        assert s.llm_text_frontier_model == "gpt-5.6-terra"
        assert s.text_chain_full[2] == "gpt-5.6-terra"


# ---------------------------------------------------------------------------
# 2. Client: request shape
# ---------------------------------------------------------------------------


class TestChatGpt5_6ClientRequest:
    """The HTTP request to OpenAI has the correct shape."""

    @pytest.fixture
    def frontier_client(self) -> ChatGpt5_6Client:
        """Frontier client with test-friendly settings."""
        return ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
            fail_max=3,
            reset_timeout_s=10,
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_request_shape(
        self,
        frontier_client: ChatGpt5_6Client,
    ) -> None:
        """Request to OpenAI /v1/chat/completions has the correct shape:
        - model = configured model name
        - messages = passed through
        - temperature = passed through
        - max_tokens = passed through
        - tools = passed through (optional)
        - Authorization: Bearer <api_key>
        - Content-Type: application/json
        """
        route = respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_openai_response())
        )

        messages = [{"role": "user", "content": "Hello"}]
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
        result = await frontier_client.chat(
            messages,
            temperature=0.7,
            tools=tools,
            max_tokens=1000,
        )

        assert route.called
        request = route.calls.last.request
        # URL is the OpenAI chat completions endpoint
        assert str(request.url) == f"{OPENAI_BASE_URL}/chat/completions"
        # Headers
        assert request.headers["authorization"] == f"Bearer {FAKE_OPENAI_KEY}"
        assert "application/json" in request.headers["content-type"]
        # Body
        import json as _json

        body = _json.loads(request.content)
        assert body["model"] == FRONTIER_MODEL
        assert body["messages"] == messages
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 1000
        assert body["tools"] == tools
        # Result sanity
        assert result.content == "ok"
        assert result.model == FRONTIER_MODEL

    @pytest.mark.asyncio
    @respx.mock
    async def test_request_omits_max_tokens_when_none(
        self,
        frontier_client: ChatGpt5_6Client,
    ) -> None:
        """If max_tokens is None, the field is omitted from the payload.

        OpenAI rejects requests with `max_tokens: null` (must be a
        positive integer or absent). The frontier client defaults to
        `max_tokens=None` (caller decides) — when None, the field is
        NOT sent.
        """
        route = respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_openai_response())
        )

        await frontier_client.chat(
            [{"role": "user", "content": "hi"}],
            temperature=1.0,
        )

        import json as _json

        body = _json.loads(route.calls.last.request.content)
        assert "max_tokens" not in body


# ---------------------------------------------------------------------------
# 3. Client: response parsing
# ---------------------------------------------------------------------------


class TestChatGpt5_6ClientResponse:
    """The OpenAI response is parsed into LLMResponse correctly."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_response_parsed(self) -> None:
        """Content, tokens, model, latency are extracted from the response."""
        client = ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
        )
        respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_openai_response(
                    content="Hello there!",
                    prompt_tokens=12,
                    completion_tokens=8,
                ),
            )
        )

        try:
            result = await client.chat([{"role": "user", "content": "hi"}])
        finally:
            await client.aclose()

        assert result.content == "Hello there!"
        assert result.tokens_in == 12
        assert result.tokens_out == 8
        assert result.model == FRONTIER_MODEL
        assert result.latency_ms >= 0
        assert result.tool_calls == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_response_preserves_provider_model_identifier(self) -> None:
        """The result exposes the model reported by the provider."""
        client = ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
        )
        provider_model = "provider-returned-model"
        respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_openai_response(model=provider_model),
            )
        )

        try:
            result = await client.chat([{"role": "user", "content": "hi"}])
        finally:
            await client.aclose()

        assert result.model == provider_model

    @pytest.mark.asyncio
    @respx.mock
    async def test_response_with_tool_calls(self) -> None:
        """Tool calls in the response are extracted into LLMResponse.tool_calls."""
        client = ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
        )
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Madrid"}',
                },
            }
        ]
        respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_openai_response(
                    content=None,  # OpenAI omits content when tool_calls present
                    tool_calls=tool_calls,
                ),
            )
        )

        try:
            result = await client.chat(
                [{"role": "user", "content": "weather"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "x",
                            "parameters": {},
                        },
                    }
                ],
            )
        finally:
            await client.aclose()

        # Content is "" (None normalized via `or ""`).
        assert result.content == ""
        # Tool calls extracted.
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Madrid"}


# ---------------------------------------------------------------------------
# 4. Client: error cases
# ---------------------------------------------------------------------------


class TestChatGpt5_6ClientErrors:
    """All error paths return LLMError (uniform with the rest of the router)."""

    @pytest.mark.asyncio
    @respx.mock
    @pytest.mark.parametrize(
        "status_code,expected_match",
        [
            (401, "HTTP error"),
            (403, "HTTP error"),
            (429, "HTTP error"),
            (500, "HTTP error"),
            (502, "HTTP error"),
            (503, "HTTP error"),
        ],
    )
    async def test_http_errors_become_llm_error(
        self, status_code: int, expected_match: str
    ) -> None:
        """HTTP 4xx/5xx responses from OpenAI become LLMError.

        All non-2xx responses are wrapped in LLMError so the
        router's chain can fall through. Specific status-code
        handling (e.g., distinguishing 401 from 429 for retry
        policy) is the caller's responsibility; the frontier
        client just signals failure.
        """
        client = ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
        )
        respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                status_code,
                json={"error": {"message": "test error"}},
            )
        )

        try:
            with pytest.raises(LLMError, match=expected_match):
                await client.chat([{"role": "user", "content": "hi"}])
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_json_response(self) -> None:
        """Non-JSON response (e.g., HTML from a Cloudflare error page) → LLMError.

        Mirrors the behavior in LLMRouter._invoke_openai: invalid
        JSON is caught and wrapped in LLMError so the chain can
        fall through.
        """
        client = ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
        )
        respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html>502 Bad Gateway</html>",
            )
        )

        try:
            with pytest.raises(LLMError, match="Invalid JSON"):
                await client.chat([{"role": "user", "content": "hi"}])
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_response_missing_choices(self) -> None:
        """JSON response without `choices` → LLMError.

        Mirrors the existing pattern in LLMRouter._invoke_openai
        (`test_malformed_openai_response_raises`).
        """
        client = ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
        )
        respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"unexpected_field": "no choices here"},
            )
        )

        try:
            with pytest.raises(LLMError, match=r"Malformed ChatGPT 5\.6 response"):
                await client.chat([{"role": "user", "content": "hi"}])
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error(self) -> None:
        """Network errors (ConnectError, etc.) → LLMError."""
        client = ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
        )
        respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        try:
            with pytest.raises(LLMError, match="HTTP error"):
                await client.chat([{"role": "user", "content": "hi"}])
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# 5. Client: circuit breaker integration
# ---------------------------------------------------------------------------


class TestChatGpt5_6ClientBreaker:
    """The frontier client integrates with the existing CircuitBreaker."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_breaker_opens_after_fail_max(self) -> None:
        """After fail_max consecutive failures, the breaker opens.

        The frontier uses a more aggressive fail_max (3, vs global
        5) since the frontier is last-resort — cheaper to fall
        back to local tiers than to keep retrying a cloud call.
        """
        client = ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
            fail_max=3,
            reset_timeout_s=10,
        )
        respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(return_value=httpx.Response(500))

        # Start closed
        assert client.breaker_state == "closed"

        # 3 failures → breaker should open
        for _ in range(3):
            with pytest.raises(LLMError):
                await client.chat([{"role": "user", "content": "hi"}])

        assert client.breaker_state == "open"

        # Next call should be rejected by the breaker (CircuitOpenError)
        with pytest.raises(CircuitOpenError):
            await client.chat([{"role": "user", "content": "hi"}])

        await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_breaker_success_resets_fail_count(self) -> None:
        """A successful call resets the fail counter (breaker stays closed)."""
        client = ChatGpt5_6Client(
            model=FRONTIER_MODEL,
            api_key=FAKE_OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
            timeout_s=5.0,
            fail_max=3,
            reset_timeout_s=10,
        )
        # 2 failures, 1 success, 2 more failures → still closed
        # (counter resets after success).
        route = respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(200, json=_openai_response()),
                httpx.Response(500),
                httpx.Response(500),
            ]
        )
        try:
            for _ in range(2):
                with pytest.raises(LLMError):
                    await client.chat([{"role": "user", "content": "x"}])
            # Success
            result = await client.chat([{"role": "user", "content": "x"}])
            assert result.content == "ok"
            assert client.breaker_state == "closed"
            # 2 more failures → still under fail_max=3
            for _ in range(2):
                with pytest.raises(LLMError):
                    await client.chat([{"role": "user", "content": "x"}])
            assert client.breaker_state == "closed"
        finally:
            await client.aclose()
        _ = route  # silence unused


# ---------------------------------------------------------------------------
# 6. Router integration: opt-in behavior
# ---------------------------------------------------------------------------


class TestRouterIntegration:
    """LLMRouter correctly wires the frontier client based on Settings."""

    def test_router_no_frontier_client_when_disabled(self, settings: Settings) -> None:
        """LLMRouter does NOT instantiate the frontier client when disabled.

        Sprint 19 north star: no automatic cloud calls. The
        frontier client must be explicitly opt-in via env vars.
        """
        assert settings.llm_text_frontier_enabled is False
        router = LLMRouter(settings)
        try:
            assert router._frontier_client is None
        finally:
            asyncio.run(router.aclose())

    def test_router_frontier_client_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLMRouter instantiates the frontier client when enabled + key set."""
        monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "true")
        monkeypatch.setenv("LLM_TEXT_FRONTIER__API_KEY", FAKE_OPENAI_KEY)
        s = Settings(_env_file=None)
        router = LLMRouter(s)
        try:
            assert router._frontier_client is not None
            assert router._frontier_client.model == FRONTIER_MODEL
            # The frontier breaker is registered in self._breakers
            # alongside the text_chain breakers.
            assert FRONTIER_MODEL in router._breakers
            # The breaker's state is visible via get_breaker_states.
            states = router.get_breaker_states()
            assert FRONTIER_MODEL in states
            assert states[FRONTIER_MODEL] == "closed"
        finally:
            asyncio.run(router.aclose())

    def test_is_frontier_model_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_is_frontier_model correctly identifies the frontier model name."""
        monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "true")
        monkeypatch.setenv("LLM_TEXT_FRONTIER__API_KEY", FAKE_OPENAI_KEY)
        s = Settings(_env_file=None)
        router = LLMRouter(s)
        try:
            # Exact match
            assert router._is_frontier_model(FRONTIER_MODEL) is True
            # Other models are NOT frontier
            assert router._is_frontier_model("MiniMax-M3") is False
            assert router._is_frontier_model("deepseek-v4-flash") is False
            # Empty string is NOT frontier (defensive)
            assert router._is_frontier_model("") is False
        finally:
            asyncio.run(router.aclose())


# ---------------------------------------------------------------------------
# 7. Router integration: chat() behavior
# ---------------------------------------------------------------------------


class TestChatBehavior:
    """LLMRouter.chat() correctly handles the frontier model in the chain."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_chain_with_frontier_dispatches_to_frontier_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the frontier model is in the chain AND enabled, the chat()
        method dispatches to the frontier client (not the main client).
        """
        monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "true")
        monkeypatch.setenv("LLM_TEXT_FRONTIER__API_KEY", FAKE_OPENAI_KEY)
        s = Settings(_env_file=None)
        # Mock ONLY the OpenAI endpoint (not the main base_url)
        # The frontier is the only model in the chain (skip primary/fallback).
        openai_url = f"{s.llm_text_frontier_base_url}/chat/completions"
        respx.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json=_openai_response(content="frontier response"),
            )
        )

        router = LLMRouter(s)
        try:
            result = await router.chat(
                [{"role": "user", "content": "hi"}],
                chain_override=[FRONTIER_MODEL],
            )
            assert result.content == "frontier response"
            assert result.model == FRONTIER_MODEL
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_chain_with_frontier_skipped_when_disabled(self, settings: Settings) -> None:
        """If the frontier is in the chain but NOT enabled, the model is
        silently skipped. If it's the only model, chat() raises LLMError
        (the chain is exhausted).
        """
        assert settings.llm_text_frontier_enabled is False
        router = LLMRouter(settings)
        try:
            with pytest.raises(LLMError, match="All models in chain failed"):
                await router.chat(
                    [{"role": "user", "content": "hi"}],
                    chain_override=[FRONTIER_MODEL],
                )
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_chain_falls_back_to_frontier_after_local_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Smart escalation: if primary + fallback fail, the chain falls
        through to the frontier (ChatGPT 5.6).

        This covers the opt-in smart-escalation path after the local
        chain is exhausted.
        """
        monkeypatch.setenv("LLM_TEXT_FRONTIER__ENABLED", "true")
        monkeypatch.setenv("LLM_TEXT_FRONTIER__API_KEY", FAKE_OPENAI_KEY)
        s = Settings(_env_file=None)

        # Mock the main base_url endpoints to always 500
        # (so primary + fallback both fail).
        # The retry loop (llm_max_retries + 1 attempts) all see 500.
        main_base = s.opencode_go_base_url
        respx.post(f"{main_base}/chat/completions").mock(return_value=httpx.Response(500))
        respx.post(f"{main_base}/messages").mock(return_value=httpx.Response(500))
        # Mock the OpenAI frontier endpoint to succeed.
        openai_url = f"{s.llm_text_frontier_base_url}/chat/completions"
        respx.post(openai_url).mock(
            return_value=httpx.Response(
                200,
                json=_openai_response(content="escalated to frontier"),
            )
        )

        router = LLMRouter(s)
        try:
            # Normal product path: no test-only chain override.
            result = await router.chat([{"role": "user", "content": "hi"}])
            assert result.content == "escalated to frontier"
            assert result.model == FRONTIER_MODEL
        finally:
            await router.aclose()
