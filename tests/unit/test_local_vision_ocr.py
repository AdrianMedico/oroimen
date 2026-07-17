"""Unit tests for LocalVisionOcrProvider (Sprint 19.6+ Phase 3, TDD v0.6 §10).

Tests use `respx` to mock httpx (the LocalVisionOcrProvider uses
httpx.AsyncClient for Ollama HTTP calls). No real Ollama needed for CI.

The integration test (test 13: real qwen3-vl:8b) is in
`tests/integration/test_local_vision_ocr_integration.py` and is gated
by E2E_LOCAL_OLLAMA_URL.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from hermes.config import Settings
from hermes.llm.ocr import (
    LocalVisionOcrProvider,
    OcrError,
    OcrResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ollama_url() -> str:
    """Default Ollama URL for tests."""
    return "http://localhost:11434"


@pytest.fixture(autouse=True)
def _set_required_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set required env vars for Settings construction (autouse).

    The Settings class requires some fields (gemini_api_key, etc.)
    before local_vision_base_url can be tested. We set them via
    monkeypatch (which auto-cleans up after each test) so the test
    code itself doesn't reference the env-var names (avoids scrubber
    false positives on field-name substrings).
    """
    import os as _os

    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-test-minlength-1234567890")  # gitleaks:allow
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-1234567890")  # gitleaks:allow
    # Suppress unused import warning
    _ = _os


@pytest.fixture
def provider(ollama_url: str) -> LocalVisionOcrProvider:
    """A provider with default settings and a real (but unused) httpx client.

    Tests inject respx routes; the client is real but never makes network
    calls because respx intercepts.
    """
    return LocalVisionOcrProvider(
        base_url=ollama_url,
        model="qwen3-vl:2b",
        timeout_s=5.0,  # short for test speed
    )


@pytest.fixture
def tiny_png_bytes() -> bytes:
    """Minimal 1x1 PNG (67 bytes) for file-read tests."""
    # 1x1 transparent PNG, base64-decoded
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )


@pytest.fixture
def tmp_image_file(tmp_path: Path, tiny_png_bytes: bytes) -> Path:
    """Write a tiny PNG to a temp file and return the path."""
    p = tmp_path / "tiny.png"
    p.write_bytes(tiny_png_bytes)
    return p


def _ollama_response(
    *,
    response: str = "extracted text here",
    thinking: str = "",
    model: str = "qwen3-vl:2b",
    total_duration_ns: int = 1_000_000_000,  # 1s
) -> dict[str, Any]:
    """Build a fake Ollama /api/generate response JSON body."""
    return {
        "model": model,
        "created_at": "2026-07-16T00:00:00.000000Z",
        "response": response,
        "thinking": thinking,
        "done": True,
        "done_reason": "stop",
        "context": [151644, 872],
        "total_duration": total_duration_ns,
        "load_duration": 100_000_000,
        "prompt_eval_count": 17,
        "prompt_eval_duration": 200_000_000,
        "eval_count": 10,
        "eval_duration": 700_000_000,
    }


# ---------------------------------------------------------------------------
# 1. Request shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_request_shape(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Verify the request to Ollama has the correct shape.

    - model = provider's model
    - prompt = default OCR prompt
    - images = [base64 of file bytes]
    - stream = false
    - options.temperature = 0.0
    - options.num_predict = 4096
    """
    route = respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(200, json=_ollama_response())
    )

    result = await provider.ocr(str(tmp_image_file), "abc123")

    assert route.called
    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["model"] == "qwen3-vl:2b"
    assert "extract" in body["prompt"].lower()  # OCR intent
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["num_predict"] == 4096
    # images is a list with one base64 string
    assert isinstance(body["images"], list)
    assert len(body["images"]) == 1
    # Verify the base64 decodes to the original bytes
    assert base64.b64decode(body["images"][0]) == tmp_image_file.read_bytes()
    # Sanity on the result
    assert isinstance(result, OcrResult)
    assert result.provider == "local_vision"
    assert result.model == "qwen3-vl:2b"
    assert result.text == "extracted text here"


# ---------------------------------------------------------------------------
# 2. Response parsed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_response_parsed(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Verify response text is correctly extracted from Ollama JSON."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(
            200,
            json=_ollama_response(response="Hello World\n\nLine 2"),
        )
    )

    result = await provider.ocr(str(tmp_image_file), "abc123")
    assert result.text == "Hello World\n\nLine 2"
    assert result.confidence == 0.0  # No native OCR confidence
    assert result.provider == "local_vision"


# ---------------------------------------------------------------------------
# 3. Thinking stripped from response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_thinking_stripped_from_response(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """qwen3-vl quirk: thinking can leak into the `response` field.

    Verify the <think>...</think> blocks are removed, leaving only
    the final answer.
    """
    raw_response = (
        "<think>The user wants me to extract text. Let me look at the "
        "image carefully. I see 'HELLO' in the image.</think>"
        "HELLO"
    )
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(
            200,
            json=_ollama_response(response=raw_response, thinking="thinking content"),
        )
    )

    result = await provider.ocr(str(tmp_image_file), "abc123")
    assert result.text == "HELLO"  # thinking stripped
    assert "<think>" not in result.text


# ---------------------------------------------------------------------------
# 3b. Thinking stripped (multiline, nested whitespace)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_thinking_stripped_multiline(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Multiline <think> blocks with embedded newlines + whitespace."""
    raw_response = (
        "<think>\nStep 1: Look at image.\nStep 2: Find text.\nStep 3: "
        "Output it.\n</think>\nThe actual text is: ABC"
    )
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(200, json=_ollama_response(response=raw_response))
    )

    result = await provider.ocr(str(tmp_image_file), "abc123")
    assert result.text == "The actual text is: ABC"


# ---------------------------------------------------------------------------
# 4. Thinking in separate field (not leaked into OcrResult)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_thinking_in_separate_field_preserved(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """The `thinking` field from Ollama response is NOT in OcrResult.

    OcrResult only has: text, confidence, model, provider, latency_ms.
    The `thinking` field is internal to Ollama; we don't expose it.
    """
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(
            200,
            json=_ollama_response(
                response="final answer",
                thinking="internal chain of thought",
            ),
        )
    )

    result = await provider.ocr(str(tmp_image_file), "abc123")
    assert result.text == "final answer"
    # OcrResult has no `thinking` attribute
    assert not hasattr(result, "thinking") or result.thinking is None


# ---------------------------------------------------------------------------
# 5. Empty response returns empty OcrResult (not an error)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_empty_response_returns_empty_result(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Empty `response` field is NOT an error — caller decides what to do.

    This is the qwen3-vl case where num_predict cuts mid-thinking,
    leaving response="". The provider returns an empty OcrResult and
    does NOT raise OcrError. The drop_watcher / ocr_decision layer
    decides the next step.
    """
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(200, json=_ollama_response(response=""))
    )

    result = await provider.ocr(str(tmp_image_file), "abc123")
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.model == "qwen3-vl:2b"


# ---------------------------------------------------------------------------
# 6. Connection refused → retryable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_connection_refused_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Cannot connect to Ollama → OcrError(retryable=True)."""
    respx.post(f"{ollama_url}/api/generate").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is True
    assert exc_info.value.provider == "local_vision"
    msg = str(exc_info.value).lower()
    assert "ollama" in msg or "connect" in msg


# ---------------------------------------------------------------------------
# 7. Timeout → retryable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_timeout_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama request times out → OcrError(retryable=True)."""
    respx.post(f"{ollama_url}/api/generate").mock(
        side_effect=httpx.TimeoutException("Read timed out")
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is True
    msg = str(exc_info.value).lower()
    assert "timed out" in msg or "timeout" in msg


# ---------------------------------------------------------------------------
# 8. HTTP 500 → retryable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_http_500_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama returns 500 (server error) → OcrError(retryable=True)."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(500, json={"error": "model crashed: out of memory"})
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is True
    assert "500" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 9. HTTP 404 → not retryable, with helpful message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_http_404_model_not_found_message(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Model not found in Ollama → OcrError(retryable=False) with hint."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(404, json={"error": "model 'qwen3-vl:8b' not found"})
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is False
    msg = str(exc_info.value).lower()
    assert "not found" in msg
    assert "ollama pull" in msg  # the helpful hint
    assert "qwen3-vl:2b" in msg  # mentions the model name


# ---------------------------------------------------------------------------
# 10. HTTP 400 → not retryable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_http_400_not_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Bad request (model doesn't support images) → OcrError(retryable=False)."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(400, json={"error": "model does not support image input"})
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is False


# ---------------------------------------------------------------------------
# 11. File read failure → not retryable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_read_failure_not_retryable(
    provider: LocalVisionOcrProvider,
    tmp_path: Path,
) -> None:
    """File doesn't exist or can't be read → OcrError(retryable=False).

    No mocking needed — the failure happens before any HTTP call.
    """
    nonexistent = tmp_path / "does_not_exist.png"
    assert not nonexistent.exists()

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(nonexistent), "abc123")
    assert exc_info.value.retryable is False
    msg = str(exc_info.value).lower()
    # Accept any of: "failed to read", "failed to stat", "no such file",
    # "not found", or the OSError text from the underlying call.
    assert any(
        s in msg
        for s in (
            "failed to read",
            "failed to stat",
            "no such file",
            "not found",
            "winerror",
        )
    )


# ---------------------------------------------------------------------------
# 12. Settings defaults
# ---------------------------------------------------------------------------


def test_settings_defaults_present() -> None:
    """All 6 local_vision_* fields are in Settings with safe defaults.

    TDD v0.6 §10.4.5 specifies:
    - local_vision_enabled: bool = False (opt-in, per Sprint 19 north star)
    - local_vision_model: str = "qwen3-vl:2b"
    - local_vision_base_url: str = "http://localhost:11434"
    - local_vision_timeout_s: float = 120.0
    - local_vision_temperature: float = 0.0
    - local_vision_num_predict: int = 4096
    """
    fields = Settings.model_fields
    assert "local_vision_enabled" in fields
    assert fields["local_vision_enabled"].default is False
    assert "local_vision_model" in fields
    assert fields["local_vision_model"].default == "qwen3-vl:2b"
    assert "local_vision_base_url" in fields
    assert fields["local_vision_base_url"].default == "http://localhost:11434"
    assert "local_vision_timeout_s" in fields
    assert fields["local_vision_timeout_s"].default == 120.0
    assert "local_vision_temperature" in fields
    assert fields["local_vision_temperature"].default == 0.0
    assert "local_vision_num_predict" in fields
    assert fields["local_vision_num_predict"].default == 4096


# ---------------------------------------------------------------------------
# 13. acclose / client lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_idempotent() -> None:
    """Provider.aclose() is safe to call multiple times (idempotent)."""
    provider = LocalVisionOcrProvider()
    await provider.aclose()
    await provider.aclose()  # second call should be no-op, not raise


@pytest.mark.asyncio
async def test_aclose_does_not_close_injected_client(
    ollama_url: str,
) -> None:
    """If the client was injected, aclose() must NOT close it.

    The caller owns injected clients. Closing them would be a
    surprising side effect.
    """
    async with httpx.AsyncClient(timeout=5.0) as injected:
        provider = LocalVisionOcrProvider(client=injected)
        await provider.aclose()
        # The injected client should still be usable
        assert not injected.is_closed


# ---------------------------------------------------------------------------
# 14. Latency_ms is positive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_latency_ms_positive(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Verify latency_ms is reported and is a positive integer."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(200, json=_ollama_response())
    )

    result = await provider.ocr(str(tmp_image_file), "abc123")
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0  # could be 0 in very fast tests, but should be int


# ===========================================================================
# R1 v0.7 additions: extended error cases, URL validation, 3-pattern think
# ===========================================================================


# ---------------------------------------------------------------------------
# 15. Settings: URL validation (MAJOR-1 from security R1)
# ---------------------------------------------------------------------------


def test_settings_base_url_accepts_localhost() -> None:
    """local_vision_base_url accepts http://localhost:11434."""
    from hermes.config import Settings

    s = Settings(local_vision_base_url="http://localhost:11434")
    assert s.local_vision_base_url == "http://localhost:11434"


def test_settings_base_url_accepts_loopback_ip() -> None:
    """local_vision_base_url accepts 127.0.0.1."""
    from hermes.config import Settings

    s = Settings(local_vision_base_url="http://127.0.0.1:11434")
    assert s.local_vision_base_url == "http://127.0.0.1:11434"


def test_settings_base_url_rejects_remote_host() -> None:
    """local_vision_base_url REJECTS remote hosts (Sprint 19 north star)."""
    from pydantic import ValidationError

    from hermes.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings(local_vision_base_url="http://attacker.com:11434")
    assert "localhost" in str(exc_info.value).lower() or "127.0.0.1" in str(exc_info.value).lower()


def test_settings_base_url_rejects_non_http_scheme() -> None:
    """local_vision_base_url REJECTS non-http schemes (file://, gopher://, etc.)."""
    from pydantic import ValidationError

    from hermes.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings(local_vision_base_url="file:///etc/passwd")
    assert "http" in str(exc_info.value).lower() or "scheme" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 16. Error cases: 401, 403, 429, 503 (R1 v0.7 impl-3 MAJOR)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_http_401_not_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama returns 401 (auth required) → OcrError(retryable=False)."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(401, json={"error": "authentication required"})
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is False
    assert "401" in str(exc_info.value) or "credentials" in str(exc_info.value).lower()


@pytest.mark.asyncio
@respx.mock
async def test_http_403_not_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama returns 403 (auth failed) → OcrError(retryable=False)."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(403, json={"error": "invalid credentials"})
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
@respx.mock
async def test_http_429_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama returns 429 (rate limit) → OcrError(retryable=True)."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(429, json={"error": "rate limit exceeded"})
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
@respx.mock
async def test_http_503_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama returns 503 (overloaded) → OcrError(retryable=True)."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(503, json={"error": "model loading"})
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is True


# ---------------------------------------------------------------------------
# 17. Error cases: done:false, done_reason:error, missing field, malformed JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_done_false_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama returns done=false (unexpected for stream=false) → OcrError(retryable=True)."""
    body = _ollama_response(response="")
    body["done"] = False
    respx.post(f"{ollama_url}/api/generate").mock(return_value=httpx.Response(200, json=body))

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is True
    assert "done=false" in str(exc_info.value) or "done" in str(exc_info.value).lower()


@pytest.mark.asyncio
@respx.mock
async def test_done_reason_error_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama returns done_reason=error (model-side error) → OcrError(retryable=True)."""
    body = _ollama_response(response="")
    body["done_reason"] = "error"
    respx.post(f"{ollama_url}/api/generate").mock(return_value=httpx.Response(200, json=body))

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
@respx.mock
async def test_missing_response_field_not_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama response missing 'response' field → OcrError(retryable=False)."""
    body = _ollama_response(response="ignored")
    del body["response"]
    respx.post(f"{ollama_url}/api/generate").mock(return_value=httpx.Response(200, json=body))

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is False
    assert "response" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()


@pytest.mark.asyncio
@respx.mock
async def test_malformed_json_not_retryable(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """Ollama returns malformed JSON → OcrError(retryable=False)."""
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(200, content=b"not valid json {{{")
    )

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(tmp_image_file), "abc123")
    assert exc_info.value.retryable is False
    assert "json" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 18. Image size cap (TDD v0.7 §10.4.2, MINOR-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_size_cap_enforced(
    provider: LocalVisionOcrProvider,
    tmp_path: Path,
) -> None:
    """Files > 50 MB are rejected before reading (OOM DoS prevention)."""
    big_file = tmp_path / "huge.png"
    # Create a file of 51 MB. Skipping the actual write by directly
    # setting the size is tricky cross-platform; just write zeros.
    # Tests are slow if we write 51MB; can be optimized if needed.
    big_file.write_bytes(b"\x00" * (51 * 1024 * 1024))

    with pytest.raises(OcrError) as exc_info:
        await provider.ocr(str(big_file), "abc123")
    assert exc_info.value.retryable is False
    assert "exceeds max" in str(exc_info.value) or "50" in str(exc_info.value)


# Note: response size cap test removed in v0.7.1 — httpx 0.28 does not
# support `max_response_size` in Limits. Deferred to httpx upgrade
# (or custom transport in Sprint 22+). Image size cap (above) is the
# primary DoS defense.


# ---------------------------------------------------------------------------
# 19. Think-strip uses 3 patterns (TDD v0.7 §10.4.3, impl-1 MAJOR)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_thinking_patterns_all_three(
    provider: LocalVisionOcrProvider,
    tmp_image_file: Path,
    ollama_url: str,
) -> None:
    """All 3 thinking patterns are stripped: <thinking>, <reasoning>, <think>."""
    # Test <thinking>...</thinking>
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(
            200, json=_ollama_response(response="<thinking>internal</thinking>FINAL")
        )
    )
    result = await provider.ocr(str(tmp_image_file), "abc123")
    assert result.text == "FINAL"

    # Test <reasoning>...</reasoning>
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(
            200, json=_ollama_response(response="<reasoning>internal</reasoning>FINAL2")
        )
    )
    result = await provider.ocr(str(tmp_image_file), "abc123")
    assert result.text == "FINAL2"

    # Test unclosed <think> (qwen3-vl cut off mid-thinking)
    respx.post(f"{ollama_url}/api/generate").mock(
        return_value=httpx.Response(
            200, json=_ollama_response(response="<think>halfway through the answer...")
        )
    )
    result = await provider.ocr(str(tmp_image_file), "abc123")
    assert result.text == ""  # unclosed <think> → stripped to empty


# ---------------------------------------------------------------------------
# 20. Client lifecycle (TDD v0.7 §10.4.7, impl-2 MAJOR)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_creates_client_when_none_passed() -> None:
    """When client=None, the provider creates one on first ocr() call.

    Without an actual HTTP call, just verify the constructor doesn't
    fail and aclose() works. The actual auto-creation is verified by
    the existing respx-mocked tests (they pass client=httpx.AsyncClient
    explicitly, but the construction path is the same).
    """
    provider = LocalVisionOcrProvider()
    assert provider._client is None
    assert provider._owns_client is True
    await provider.aclose()
    # After aclose, the auto-owned client is None (was never created)
    assert provider._client is None


@pytest.mark.asyncio
async def test_aclose_closes_auto_created_client() -> None:
    """aclose() closes the client that was auto-created (not injected)."""
    provider = LocalVisionOcrProvider()
    # Trigger client creation
    client = await provider._get_client()
    assert client is not None
    await provider.aclose()
    # The client should now be closed
    assert provider._client is None
    # Sanity: the original client variable is now closed
    assert client.is_closed


# ---------------------------------------------------------------------------
# 21. Confidence documented as placeholder (TDD v0.7 §10.4.3, MINOR-4)
# ---------------------------------------------------------------------------


def test_confidence_documented_as_placeholder() -> None:
    """OcrResult.confidence docstring warns '0.0 means unknown'."""
    doc = OcrResult.__doc__ or ""
    assert "0.0" in doc
    assert "unknown" in doc.lower() or "trust signal" in doc.lower()
