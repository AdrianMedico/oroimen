"""Ollama chat provider (Sprint 19.6+ Phase 5 — OpenAI Build Week).

OpenAI Build Week hackathon (deadline 2026-07-21, $15k first prize)
requires that the demo "just work" out of the box. Before this
module, the default `text_chain` required an `OPENCODE_GO_API_KEY` to
be set in `.env` — a judge cloning the repo and running `docker
compose up` would see "API key missing" on the first chat attempt.

**This module fixes that.** Ollama runs in the `docker compose`
stack and serves an OpenAI-compatible API at `/v1`. The
`OllamaClient` here talks to that endpoint using the same
`openai.AsyncOpenAI` SDK we already depend on, so the chain
integration is uniform with the rest of `LLMRouter`.

Design decisions (Sprint 19 north star: private + local-first):

- **Lazy import of `openai` SDK** — the SDK is imported inside
  `chat()` so machines that never reach the Ollama branch of the
  chain (e.g., a deploy that only uses MiniMax) don't pay the
  import cost. The SDK is already a hard dep of the project
  (used by `ChatGpt5_6Client` and the embeddings path), so this
  isn't a new dependency — just a deferral.
- **Own circuit breaker** (per-tier tuning). The Ollama tier
  gets `fail_max=5, reset_timeout=60s` (default) which is more
  permissive than the frontier's `fail_max=3, reset=120s`
  because the Ollama container may need a moment to come up
  on cold start (model load) and we don't want to drop local
  requests just because the first 1-2 warmup calls are slow.
- **Follows the same `LLMResponse` / `LLMError` contract** so
  the chain integration in `LLMRouter` is uniform with
  `ChatGpt5_6Client` and the legacy OpenAI / Anthropic paths.
- **`api_key="ollama"`** — Ollama ignores the value of the
  Authorization header, but the `openai` SDK requires *some*
  value to be set when constructing the client (otherwise it
  raises). The literal `"ollama"` is the canonical placeholder
  used by the Ollama docs themselves.

**Default model**: `qwen2.5:7b` (~4.7 GB on disk, ~6 GB RAM at
Q4_K_M quantization, fits in the 6 GB memory limit of the
`ollama` service in `docker-compose.yml`). 7B-class models are
the sweet spot for "good enough to demo, small enough to run
on a 6 GB NAS".

Reference: https://github.com/ollama/ollama/blob/main/docs/openai.md
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

from hermes.llm.breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    # Type-only import. The real import is lazy (in chat/_invoke) to
    # avoid a circular dependency: hermes.llm.router imports this
    # module's OllamaClient, and this module needs router's
    # LLMResponse at function-call time.
    from hermes.llm.router import LLMResponse

logger = logging.getLogger(__name__)

# Mirrors router.py: size of the raw snippet logged on parse failure.
# Sufficient to identify the root cause (Ollama HTML error page, JSON
# truncated mid-stream, etc.) without leaking large volumes of text.
_RAW_SNIPPET_MAX = 200


class OllamaClient:
    """Async client for a local Ollama server (OpenAI-compat at /v1).

    Lifecycle:
    - Instantiated by `LLMRouter.__init__` ONLY when at least one
      model in the chain has `llm_text_primary_provider="ollama"`
      (or similar). The router registers the Ollama breaker for
      the model and instantiates this client on demand.
    - Owns its own `httpx.AsyncClient` (separate base_url, separate
      auth). The Ollama base_url points to `http://ollama:11434/v1`
      inside the docker network (or `http://localhost:11434/v1`
      for non-docker usage).
    - Owns its own `CircuitBreaker` (per-tier tuning).
    - Closed by `LLMRouter.aclose()` via `await client.aclose()`.

    Args:
        base_url: Ollama OpenAI-compat base URL. Default
            `http://localhost:11434/v1` (matches Ollama's own
            default; override to `http://ollama:11434/v1` inside
            the docker compose stack).
        model: Model name as served by Ollama. Default
            `qwen2.5:7b` (per the design plan in
            `docs/OLLAMA_DOCKER_COMPOSE_PLAN.md` §3.3).
        timeout_s: HTTP timeout (seconds). Default 120s. Ollama
            on CPU is slow: 5-10s/response for qwen2.5:7b; the
            first request after a cold start can take 30-60s as
            the model loads into memory.
        temperature: Sampling temperature. Default 0.7 (a
            reasonable middle ground; matches what most Ollama
            tutorials use).
        max_tokens: Max output tokens. Default 2048 (sufficient
            for chat-style responses; the router-level
            `llm_max_tokens` is applied to other providers).
        fail_max: Circuit breaker fail threshold. Default 5.
        reset_timeout_s: Circuit breaker reset timeout (seconds).
            Default 60.
        name: Breaker name (visible in logs and the /health
            endpoint via `get_breaker_states()`).
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        timeout_s: float = 120.0,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        fail_max: int = 5,
        reset_timeout_s: int = 60,
        name: str = "ollama",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Own httpx client. Reuses the project's existing httpx
        # dependency — no new transport stack. Timeout is per-call
        # (Ollama is local but CPU inference can be slow on first
        # request after a cold start, when the model is loading).
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                # Ollama ignores the value but requires *something* —
                # using the literal "ollama" placeholder per Ollama docs.
                "Authorization": "Bearer ollama",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout_s),
        )
        # Per-tier circuit breaker. The breaker's `name` defaults
        # to "ollama" but is typically overridden by the router to
        # match the model name (so health monitoring can correlate
        # the breaker with the model in the chain).
        self._breaker = CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=float(reset_timeout_s),
            name=name,
        )

    @property
    def model(self) -> str:
        """Configured model name (e.g., 'qwen2.5:7b')."""
        return self._model

    @property
    def name(self) -> str:
        """Provider name. Returns the model name (the breaker's
        name) so the router can log per-model breaker state.
        """
        return self._breaker.name

    @property
    def breaker_state(self) -> str:
        """Circuit breaker state: 'closed', 'half-open', or 'open'.

        Mirrors the pattern in `ChatGpt5_6Client.breaker_state`.
        Used by `LLMRouter.get_breaker_states()` for the /health
        endpoint.
        """
        return self._breaker.current_state

    @property
    def breaker(self) -> CircuitBreaker:
        """Direct breaker access for the router's chain integration.

        The router's `chat()` method wraps each chain call in
        `breaker.call(coro_factory)`. Exposing the breaker here
        avoids needing a separate `_call_with_breaker`
        implementation for the Ollama tier.
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
        """Invoke the local Ollama model.

        Uses the `openai.AsyncOpenAI` SDK against Ollama's
        OpenAI-compat endpoint at `/v1/chat/completions`. The
        SDK is imported LAZILY (inside this method) so that
        machines that never reach the Ollama branch of the
        chain don't pay the import cost.

        Args:
            messages: List of OpenAI-format messages. Passed
                through verbatim (Ollama's /v1 endpoint accepts
                the same format as OpenAI proper).
            temperature: Sampling temperature. If None, uses
                the constructor's default (0.7).
            tools: Optional list of OpenAI-format tool
                definitions. Most local 7B models don't support
                tool calls well, but we pass them through for
                larger Ollama models that do (e.g., qwen2.5:14b+).
            max_tokens: Max output tokens. If None, uses the
                constructor's default (2048).

        Returns:
            `LLMResponse` with content, model, tokens, latency.

        Raises:
            CircuitOpenError: if the breaker is open (caller
                should fall through to the next model in the
                chain).
            LLMError: on any HTTP, parse, or model error.
        """
        # Lazy import: avoid the openai SDK import cost on
        # machines that never reach the Ollama branch. The SDK
        # is a project dep (used elsewhere) so this is just a
        # deferral, not a new dep.
        import openai

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        if tools:
            payload["tools"] = tools

        # Lazy import for the same reason: hermes.llm.router
        # imports this module at module load time, so we
        # cannot import LLMError / LLMResponse at the top of
        # this file (circular). The import here is cheap
        # (Python caches modules) and only happens on actual
        # chat calls.
        from hermes.llm.router import LLMError

        try:
            return await self._breaker.call(
                # Lambda to delay coroutine creation until after
                # the breaker's open-check. Mirrors the pattern
                # in ChatGpt5_6Client.chat.
                lambda: self._invoke(payload, openai)
            )
        except CircuitOpenError:
            # Propagate as-is so the router's chain falls through.
            raise
        except LLMError:
            # Already wrapped; propagate.
            raise

    async def _invoke(self, payload: dict[str, Any], openai_module: Any) -> LLMResponse:
        """Inner HTTP call via the `openai` SDK. Wrapped by the
        circuit breaker.

        This is the function that actually talks to Ollama via
        the `openai.AsyncOpenAI` SDK. Raises `LLMError` on any
        failure. The circuit breaker records success/failure.

        Args:
            payload: The OpenAI-compat chat completions payload.
            openai_module: The `openai` module (passed in to
                avoid a second import — already imported by the
                caller for the lazy-import pattern).
        """
        # Lazy import for the same circular-dep reason as in chat().
        from hermes.llm.router import LLMError, LLMResponse

        client = openai_module.AsyncOpenAI(
            base_url=self._base_url,
            api_key="ollama",  # Ollama ignores; SDK requires non-empty.
            timeout=self._client.timeout,
            # Disable the SDK's built-in retry. We have our own
            # circuit breaker + the LLMRouter's outer retry loop.
            # Allowing both would cause compounded retry behavior
            # (SDK retries inside the breaker, then the router
            # retries again on the breaker open) and makes the
            # chain harder to reason about. Setting max_retries=0
            # makes the SDK a thin pass-through over httpx.
            max_retries=0,
        )
        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(**payload)
        except openai_module.APIError as exc:
            # OpenAI SDK wraps all upstream errors (HTTP, network,
            # model errors) as `openai.APIError` subclasses. We
            # catch the base class and wrap uniformly.
            logger.error(
                "ollama_http_error",
                extra={
                    "model": self._model,
                    "provider": "ollama",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                },
            )
            raise LLMError(f"Ollama API error: {exc}") from exc
        except Exception as exc:
            # Catch-all for non-APIError exceptions (e.g.,
            # httpx.ConnectError if the Ollama container is
            # down). Wrap in LLMError so the chain can fall
            # through to the next model.
            logger.error(
                "ollama_unexpected_error",
                extra={
                    "model": self._model,
                    "provider": "ollama",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                },
            )
            raise LLMError(f"Ollama unexpected error: {exc}") from exc
        finally:
            # Close the per-call client. We create a new client
            # per call (not ideal perf-wise, but the SDK is
            # simple and we want clean isolation per call). The
            # alternative — keeping a long-lived SDK client —
            # adds state we don't need: a single AsyncOpenAI
            # with the right base_url + api_key is essentially
            # free to construct, and per-call teardown
            # guarantees no leaked connections.
            with contextlib.suppress(Exception):
                await client.close()

        # Parse the SDK response. The SDK normalizes OpenAI
        # response shape so we can rely on attribute access.
        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            # Tool calls: SDK exposes them as a list of
            # `openai.types.chat.ChatCompletionMessageToolCall`
            # objects. We extract id, name, and parsed arguments
            # (the SDK already JSON-parses arguments for us).
            tool_calls_raw = message.tool_calls or []
            tool_calls: list = []
            from hermes.llm.router import ToolCall

            for tc in tool_calls_raw:
                # Each `tc` has .id, .function.name, .function.arguments
                # The SDK stores arguments as a JSON STRING (per
                # OpenAI API spec), but Pydantic v2 SDK decodes
                # it for us in some versions. Handle both.
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                args_raw = getattr(fn, "arguments", "{}")
                import json as _json

                if isinstance(args_raw, str):
                    try:
                        args_dict = _json.loads(args_raw) if args_raw.strip() else {}
                    except _json.JSONDecodeError:
                        args_dict = {}
                elif isinstance(args_raw, dict):
                    args_dict = args_raw
                else:
                    args_dict = {}
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(tc, "id", "")),
                        name=str(getattr(fn, "name", "")),
                        arguments=args_dict,
                    )
                )
            # Usage is optional; some Ollama versions don't
            # populate it. Default to 0.
            usage = getattr(response, "usage", None)
            tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            tokens_out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            return LLMResponse(
                content=content,
                model=str(getattr(response, "model", self._model)),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                tool_calls=tool_calls,
                # Ollama's OpenAI-compat endpoint doesn't expose
                # reasoning_content (local 7B models typically
                # don't have thinking mode exposed). Default to "".
                reasoning_content="",
            )
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            # Defensive: SDK may return shapes we don't expect
            # (e.g., streaming responses, partial completions).
            # Wrap in LLMError so the chain can fall through.
            logger.warning(
                "ollama_malformed_response",
                extra={
                    "model": self._model,
                    "error": str(exc),
                },
            )
            raise LLMError(f"Malformed Ollama response: {exc}") from exc


__all__ = [
    "OllamaClient",
]
