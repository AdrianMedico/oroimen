"""End-to-end tests for the Ollama provider (Sprint 19.6+ Phase 5).

These tests talk to a REAL local Ollama server. They SKIP if no
Ollama is reachable, so CI on machines without Ollama doesn't
break the suite.

The Ollama URL is configurable via the `E2E_OLLAMA_URL` env var
(default: `http://localhost:11434`). The `init-ollama` sidecar in
`docker-compose.yml` pulls `qwen2.5:7b` on first boot, so the
default URL works for a local docker compose stack.

Usage:
    # All tests skip if Ollama is not reachable
    pytest tests/e2e/test_ollama_provider.py --runnetwork
    # Force-run even if Ollama is not reachable (will skip with reason)
    pytest tests/e2e/test_ollama_provider.py --runnetwork --runollama=force
    # Or point to a custom Ollama URL
    E2E_OLLAMA_URL=http://edge-host:11434 pytest tests/e2e/test_ollama_provider.py --runnetwork

Reference: docs/OLLAMA_DOCKER_COMPOSE_PLAN.md §3.6.
"""

from __future__ import annotations

import os

import httpx
import pytest

from hermes.config import Settings
from hermes.llm.ollama import OllamaClient

pytestmark = pytest.mark.network

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default Ollama URL. Overridable via env for tests pointing at a
# remote Ollama (e.g., a edge host).
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------


def _ollama_reachable(base_url: str) -> bool:
    """Return True if the Ollama /api/tags endpoint is reachable.

    Used to skip e2e tests when no Ollama server is available
    (CI on machines without Ollama). Synchronous — called once
    per test session to decide skip vs run.
    """
    # The base_url ends with /v1; the health endpoint is at
    # `<host>:<port>/api/tags` (NOT under /v1 — that's the
    # OpenAI-compat path). Strip /v1 to find the Ollama root.
    ollama_root = base_url.rstrip("/")
    if ollama_root.endswith("/v1"):
        ollama_root = ollama_root[:-3]
    try:
        r = httpx.get(f"{ollama_root}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _model_available(base_url: str, model: str) -> bool:
    """Return True if `model` is in the Ollama /api/tags response.

    Used to skip e2e tests when the required model is not pulled.
    A docker compose stack on first boot may take 2-5 minutes
    to pull `qwen2.5:7b`; this check is a "skip if not ready"
    signal.
    """
    ollama_root = base_url.rstrip("/")
    if ollama_root.endswith("/v1"):
        ollama_root = ollama_root[:-3]
    try:
        r = httpx.get(f"{ollama_root}/api/tags", timeout=2.0)
        if r.status_code != 200:
            return False
        data = r.json()
        models = data.get("models", [])
        # The /api/tags response uses `name` (e.g., "qwen2.5:7b"),
        # which matches our model name. Some Ollama versions also
        # include the digest; we only check the name.
        names = {m.get("name", "") for m in models}
        return model in names
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ollama_base_url() -> str:
    """Ollama base URL from env, defaulting to localhost:11434/v1."""
    return os.environ.get("E2E_OLLAMA_URL", DEFAULT_OLLAMA_URL)


@pytest.fixture(scope="module")
def skip_if_no_ollama(ollama_base_url: str) -> None:
    """Skip the test if Ollama is not reachable or the model is not pulled."""
    if not _ollama_reachable(ollama_base_url):
        pytest.skip(
            f"Ollama not reachable at {ollama_base_url}. "
            f"Set E2E_OLLAMA_URL to a running Ollama instance, or run "
            f"`docker compose up` to start the local Ollama service."
        )
    if not _model_available(ollama_base_url, DEFAULT_MODEL):
        pytest.skip(
            f"Ollama reachable but {DEFAULT_MODEL} not pulled. "
            f"Run `docker compose up` to trigger the init-ollama "
            f"sidecar, or `ollama pull {DEFAULT_MODEL}` manually."
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOllamaE2E:
    """End-to-end tests against a real local Ollama server.

    All tests in this class are SKIPPED if no Ollama is reachable
    (see the `skip_if_no_ollama` fixture). This is the
    documented behavior per docs/OLLAMA_DOCKER_COMPOSE_PLAN.md
    §3.6: "Use the conftest to detect if Ollama is running. Skip
    if not (so e2e doesn't break on machines without Ollama)."

    The test exercises the full chain:
    1. Connect to Ollama via the OpenAI-compat /v1 endpoint.
    2. Send a simple chat completion request.
    3. Verify the response has the expected shape (content,
       tokens, model name).
    """

    @pytest.mark.asyncio
    async def test_real_chat_completion(
        self,
        ollama_base_url: str,
        skip_if_no_ollama: None,
    ) -> None:
        """Send a real chat completion to the local Ollama and verify the response.

        The model `qwen2.5:7b` is the default chain primary.
        We send a simple user message and expect a non-empty
        assistant response. This is a smoke test — the goal is
        to verify the wire-up works end-to-end, not to test
        the model's quality.

        The test sets a short timeout (60s) because Ollama on
        CPU can be slow, especially for the first request after
        a cold start (model load into memory).
        """
        client = OllamaClient(
            base_url=ollama_base_url,
            model=DEFAULT_MODEL,
            timeout_s=60.0,  # CPU inference can be slow
            fail_max=2,  # Low for a smoke test; 2 strikes you're out
            reset_timeout_s=10,
        )
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly 'pong' and nothing else. "
                        "This is a connectivity smoke test."
                    ),
                }
            ]
            response = await client.chat(
                messages,
                temperature=0.0,  # deterministic for smoke testing
            )
            # Basic response shape checks
            assert response.content, "Ollama returned empty content"
            assert response.model, "Ollama response missing model name"
            assert response.latency_ms > 0
            # The latency assertion is loose (just > 0) because
            # CPU inference times vary widely.
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_real_chat_via_router_dispatch(
        self,
        ollama_base_url: str,
        skip_if_no_ollama: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full router integration: LLMRouter.chat() dispatches to OllamaClient.

        This is the "end-to-end" version of the unit test
        `test_chat_dispatches_to_ollama_client` — it talks to a
        real Ollama, not a mock. It validates that the per-model
        provider dispatch in `LLMRouter._is_ollama_model` works
        in the real wire path.
        """
        # Set the Ollama URL to the real server (not the docker
        # default which assumes `http://ollama:11434/v1`).
        monkeypatch.setenv("LLM_TEXT_PRIMARY__BASE_URL", ollama_base_url)
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")

        s = Settings(_env_file=None)
        # Override the chain to just the Ollama primary
        # (avoids the MiniMax fallback that would need a real
        # API key to actually respond).
        router_chain = [s.llm_text_primary]

        from hermes.llm.router import LLMRouter

        router = LLMRouter(s)
        try:
            response = await router.chat(
                [
                    {
                        "role": "user",
                        "content": "Reply with just the word 'pong'.",
                    }
                ],
                chain_override=router_chain,
                temperature=0.0,
            )
            assert response.content, "Empty response from Ollama via router"
            assert response.model
        finally:
            await router.aclose()
