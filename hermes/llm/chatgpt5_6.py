"""ChatGPT 5.6 frontier-tier provider (Sprint 19.6+ Phase 4).

OpenAI Build Week hackathon (deadline 2026-07-21, $15k first prize)
requires use of ChatGPT 5.6 (OpenAI's new frontier model). This module
implements the frontier tier in Oroimen's multi-tier architecture:

  1. Primary    (llm_text_primary,    e.g. MiniMax-M3)
  2. Fallback   (llm_text_fallback,   e.g. MiniMax-M2.7-highspeed)
  3. Frontier   (llm_text_frontier_model, default gpt-5.6-sol)  <- NEW

The opt-in smart-escalation path can call ChatGPT 5.6 after the
configured primary and fallback tiers fail or are deemed insufficient.

Design decisions (Sprint 19 north star: no automatic cloud calls):

- **Opt-in via `llm_text_frontier_enabled`**. Default False. The
  frontier client is only instantiated when explicitly enabled. If
  the user does not set the API key, the client is not created and
  the model is silently skipped if it appears in a chain.
- **No dependency on the `openai` SDK**. Uses `httpx` directly (the
  project already uses httpx for `LocalVisionOcrProvider` and the
  main LLM router). Keeps the dependency surface minimal.
- **Own httpx client + own circuit breaker**. The frontier talks to
  `api.openai.com` (default) with a separate auth header. The
  per-tier circuit breaker can be tuned more aggressively (default
  fail_max=3 vs global 5) since falling back to local tiers is
  cheaper than retrying a cloud call.
- **Follows the existing `LLMResponse` / `LLMError` contract** so
  the chain integration is uniform with the rest of the router.

The model is configurable via `LLM_TEXT_FRONTIER__MODEL`.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

from hermes.llm.breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    # Type-only imports: not needed at runtime. The real imports
    # happen lazily inside functions to avoid a circular dependency
    # with hermes/llm/router.py (router imports this module's
    # ChatGpt5_6Client, this module needs router's LLMResponse /
    # _safe_json_parse / _parse_openai_tool_calls at function call time).
    from hermes.llm.router import LLMResponse

logger = logging.getLogger(__name__)

# Mirrors router.py: size of the raw snippet logged on parse failure.
# Sufficient to identify the root cause (Cloudflare HTML, proxy error,
# truncated JSON) without leaking large volumes of text.
_RAW_SNIPPET_MAX = 200


class ChatGpt5_6Client:
    """Async client for the ChatGPT 5.6 frontier tier (OpenAI-compatible).

    Talks to `https://api.openai.com/v1/chat/completions` by default.
    Configurable base URL for OpenAI-compatible proxies (Azure OpenAI,
    local llama.cpp OpenAI-compat server hosting a frontier model,
    etc.).

    Lifecycle:
    - Instantiated by `LLMRouter.__init__` ONLY when
      `settings.llm_text_frontier_enabled` is True.
    - Owns its own `httpx.AsyncClient` (own base_url, own auth).
    - Owns its own `CircuitBreaker` (per-tier tuning: more aggressive
      fail_max + longer reset_timeout than the global breaker).
    - Closed by `LLMRouter.aclose()` via `await client.aclose()`.

    Args:
        model: Model name to send to the API. The configured default
            is `gpt-5.6-sol`.
        api_key: OpenAI API key. Required when the frontier is
            enabled (validator in `hermes.config.Settings` rejects
            `enabled=True` + empty key at startup).
        base_url: API base URL. Default `https://api.openai.com/v1`.
        timeout_s: HTTP timeout (seconds). Default 60 (frontier
            models can be slow; 30s is too aggressive for a
            last-resort tier).
        fail_max: Circuit breaker failure threshold. Default 3.
        reset_timeout_s: Circuit breaker reset timeout (seconds).
            Default 120.
        name: Human-readable name for the breaker (used in logs).
            Default `gpt-5.6-sol`.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60.0,
        fail_max: int = 3,
        reset_timeout_s: int = 120,
        name: str = "gpt-5.6-sol",
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # Own httpx client — separate from the main router's client
        # (which talks to `opencode_go_base_url`, e.g. api.minimax.io).
        # This is the cleanest isolation: a frontier outage does NOT
        # affect local-model requests, and vice versa.
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout_s),
        )
        # Per-tier circuit breaker. The breaker's `name` is the
        # model name by default (visible in logs / health endpoints).
        self._breaker = CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=float(reset_timeout_s),
            name=name,
        )

    @property
    def model(self) -> str:
        """Configured model name (e.g. 'gpt-5.6-sol')."""
        return self._model

    @property
    def name(self) -> str:
        """Provider name. Used by the router for routing decisions
        (e.g. 'this model is the frontier tier, use the frontier
        client instead of the main client').
        """
        return self._model

    @property
    def breaker_state(self) -> str:
        """Circuit breaker state: 'closed', 'half-open', or 'open'.

        Used by `LLMRouter.get_breaker_states()` for the /health
        endpoint. Mirrors the per-model breakers in the main router.
        """
        return self._breaker.current_state

    @property
    def breaker(self) -> CircuitBreaker:
        """Direct breaker access for the router's chain integration.

        The router's `chat()` method wraps each chain call in
        `breaker.call(coro_factory)`. Exposing the breaker here
        avoids needing a separate `_call_with_breaker` implementation
        for the frontier.
        """
        return self._breaker

    async def aclose(self) -> None:
        """Close the httpx client. Idempotent."""
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Invoke the ChatGPT 5.6 frontier model.

        Talks OpenAI Chat Completions format (the API is
        OpenAI-compatible). Returns a standard `LLMResponse` so
        the chain integration in `LLMRouter` is uniform with the
        other models.

        Args:
            messages: List of OpenAI-format messages. Supports
                `role` in {system, user, assistant, tool}, plus
                `tool_calls` (assistant) and `tool_call_id`/`name`
                (tool).
            temperature: Sampling temperature. If None, uses
                `Settings.llm_temperature` (1.0 default).
            tools: Optional list of OpenAI-format tool definitions
                (passed through verbatim).
            max_tokens: Max output tokens. If None, uses
                `Settings.llm_text_frontier_max_tokens` (default 8192)
                via the caller. Direct callers can override.

        Returns:
            `LLMResponse` with content, model, token counts,
            latency, tool_calls (if any).

        Raises:
            CircuitOpenError: if the breaker is open (caller should
                fall through to the next model in the chain).
            LLMError: on any HTTP, parse, or model error.
        """
        # Build the OpenAI-compat payload. We intentionally do NOT
        # add `top_p` or `repetition_penalty` here — the OpenAI API
        # uses different params (`frequency_penalty`, `presence_penalty`)
        # and the project defaults (`llm_top_p`, `llm_repetition_penalty`)
        # are calibrated for the MiniMax path, not OpenAI. Frontier
        # callers can set their own sampling via env-level overrides
        # in a future Sprint (Sprint 22+).
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 1.0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools

        # Wrap the inner invoke in the circuit breaker. The breaker
        # records success/failure based on whether the inner call
        # raises. CircuitOpenError is propagated to the caller so
        # the chain can fall through.
        #
        # Lazy import: hermes.llm.router imports this module at
        # module load time, so we cannot import LLMError at the top
        # of this file (circular). The import here is cheap (Python
        # caches modules) and only happens on actual chat calls,
        # not at import time.
        from hermes.llm.router import LLMError

        try:
            return await self._breaker.call(
                # Lambda to delay coroutine creation until after the
                # breaker's open-check. Mirrors the existing pattern
                # in LLMRouter._call_with_breaker.
                lambda: self._invoke(payload)
            )
        except CircuitOpenError:
            # Propagate as-is so the router's chain falls through.
            raise
        except LLMError:
            # Already wrapped; propagate.
            raise

    async def _invoke(self, payload: dict[str, Any]) -> LLMResponse:
        """Inner HTTP call + parse. Wrapped by the circuit breaker.

        This is the function that actually talks to the OpenAI API.
        Raises `LLMError` on any failure (HTTP, parse, model error).
        The circuit breaker records the success/failure outcome.
        """
        # Lazy imports to break circular dependency: hermes.llm.router
        # imports this module (for ChatGpt5_6Client) at module load
        # time, so we cannot import router's helpers here at the top
        # level. We import them at first use, after router has
        # finished initializing.
        from hermes.llm.router import (
            LLMError,
            LLMResponse,
            _parse_openai_tool_calls,
            _safe_json_parse,
        )

        start = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # Capture body for diagnostics. Mirrors the pattern in
            # LLMRouter._invoke_openai.
            body = ""
            with contextlib.suppress(Exception):
                body = exc.response.text[:500] if hasattr(exc, "response") else ""
            logger.error(
                "chatgpt5_6_http_error",
                extra={
                    "model": self._model,
                    "provider": "chatgpt5_6",
                    "status": getattr(getattr(exc, "response", None), "status_code", None),
                    "body": body,
                    "url": str(getattr(exc, "request", None) and exc.request.url),
                },
            )
            raise LLMError(f"ChatGPT 5.6 HTTP error: {exc}") from exc

        # Parse the response. Reuses the helper from the main router
        # so behavior is consistent: invalid JSON / non-dict responses
        # become LLMError (not raw exceptions).
        data = _safe_json_parse(resp, provider="chatgpt5_6")
        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            usage = data.get("usage", {})
            tool_calls_raw = message.get("tool_calls")
            tool_calls = _parse_openai_tool_calls(tool_calls_raw)
            return LLMResponse(
                content=content,
                # Preserve the identifier returned by the provider. This lets
                # live verification detect aliases or routing mistakes.
                model=str(data["model"]),
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                tool_calls=tool_calls,
                # reasoning_content is supported by some OpenAI-compat
                # endpoints (o1, etc.) but ChatGPT 5.6 may or may not
                # expose it. The router handles None → "" for callers.
                reasoning_content=message.get("reasoning_content") or "",
            )
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            logger.warning(
                "chatgpt5_6_malformed_response",
                extra={
                    "model": self._model,
                    "error": str(exc),
                    "raw_snippet": str(data)[:_RAW_SNIPPET_MAX],
                },
            )
            raise LLMError(f"Malformed ChatGPT 5.6 response: {exc}") from exc


__all__ = [
    "ChatGpt5_6Client",
]
