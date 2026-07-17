"""Provider-agnostic OCR interface (Sprint 19 Slice 4d, B1 fix + Phase 3).

Two concrete implementations shipped:
- `HostedLlmOcrProvider` (Sprint 4d): wraps the existing LLM client
  (`LLMRouter.chat`). The actual model is whatever
  `Settings.llm_text_primary` is — default MiniMax-M3 (multimodal), but
  provider-agnostic via the same `llm_text_chain` mechanism the chat uses.
- `LocalVisionOcrProvider` (Sprint 19.6+ Phase 3, TDD v0.6 §10): HTTP
  client to Ollama's `/api/generate` with image input. Default
  qwen3-vl:2b on `http://localhost:11434`. No data leaves the local
  machine. Replaces the edge-PC requirement for users with a single
  edge host with a GPU.

Future (NO hardcoded dependency on any specific provider):
- `EdgeOcrV2Provider`: edge PC with vision models (LLaVA, Qwen-VL)

The decision logic in `hermes.memory.ocr_decision` is provider-agnostic.
It just receives an `OcrResult` and persists it. The handler
(`hermes.handlers.ocr_commands`) picks the provider and invokes it.

NORTH STAR:
- The LLM agent NEVER calls this module. /externalOCR is a USER-ONLY
  command (TDD §4.3.1). The model cannot escalate to a hosted VLM
  autonomously — the user must explicitly trigger it.
- HostedLlmOcrProvider is the EXPLICIT opt-in for hosted VLM. The
  default flow is Tesseract local + LocalVisionOcrProvider or edge PC
  (no network egress).
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from hermes.llm.router import LLMResponse, LLMRouter

logger = logging.getLogger(__name__)


# Sprint 19.6+ Phase 3 (TDD v0.7 §10.4.3, impl-1 MAJOR): think-strip
# uses 3 patterns, matching the §5 P0 classifier's THINKING_TRACE_PATTERNS
# for design consistency. Handles qwen3-vl `<think>` (smoke-tested),
# future models with `<thinking>` or `<reasoning>`.
THINKING_TRACE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
)

# Unclosed-tag fallback for qwen3-vl's `<think>` quirk: when the
# model is cut off mid-thinking, the closing `</think>` is missing.
# This greedy pattern catches the half-open block.
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove thinking-trace blocks from `text`. Returns stripped text.

    Defensive against the qwen3-vl quirk where thinking can leak into
    the `response` field (not just the separate `thinking` field).
    TDD v0.7 §10.4.3 documents this behavior. Uses the same 3-pattern
    set as the §5 P0 classifier (design consistency).

    Falls back to a greedy `<think>.*` pattern for unclosed tags
    (qwen3-vl cut off mid-thinking — the closing `</think>` is missing).
    """
    if not text:
        return text
    for pattern in THINKING_TRACE_PATTERNS:
        text = pattern.sub("", text)
    # If the response still starts with `<think>` (unclosed), strip
    # from there to the end. This catches the case where the model
    # ran out of `num_predict` mid-thinking.
    text = _UNCLOSED_THINK_PATTERN.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Result + error
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OcrResult:
    """Result from an OcrProvider invocation.

    Provider-agnostic: same shape regardless of implementation
    (hosted LLM, local vision, edge, etc.).

    **Sprint 19.6+ Phase 3 (TDD v0.7 §10.4.3 MINOR-4)**: `confidence`
    is a placeholder. Providers that don't report native OCR confidence
    (e.g., Ollama, OpenAI-compat APIs) set it to `0.0`. **Do not use
    `confidence` as a trust signal** — `0.0` means "unknown", NOT
    "low confidence". Callers should treat all OCR results as
    equally trustworthy (or untrustworthy) and apply the F2 fix
    downstream (see `hermes.agent.loop.wrap_file_content`).
    """

    text: str
    confidence: float
    model: str
    provider: str
    latency_ms: int = 0


class OcrError(Exception):
    """OCR provider failed. Caller decides whether to retry."""

    def __init__(self, message: str, provider: str, retryable: bool = True) -> None:
        self.provider = provider
        self.retryable = retryable
        super().__init__(message)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class OcrProvider(ABC):
    """Provider-agnostic OCR interface.

    Implementations:
    - HostedLlmOcrProvider (Sprint 4d, shipped): wraps the LLM client.
    - LocalVisionOcrProvider (Sprint 19.6+ Phase 3, TDD v0.7 §10): Ollama.
    - EdgeOcrV2Provider (Sprint 22+, future): edge PC vision models.

    The interface is intentionally minimal: read a file, return text.
    Format-specific handling (image vs PDF vs DOCX) is the LLM's job
    (or the provider's).

    Sprint 19.6+ Phase 3 (TDD v0.7 §10.4.7): added `aclose()` as a
    default no-op method. Subclasses that hold resources (httpx
    client, file handles, etc.) MUST override. Callers SHOULD call
    `aclose()` when done with the provider (typically at process
    shutdown).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'hosted_llm', 'local_vision')."""
        ...

    @abstractmethod
    async def ocr(self, file_path: str, file_id: str) -> OcrResult:
        """Run OCR on the file at `file_path`. Return extracted text.

        Args:
            file_path: Absolute path to the file (typically
                vault_files.source_path).
            file_id: SHA-256 file_id, for logging/audit.

        Returns:
            OcrResult with extracted text, confidence, model name.

        Raises:
            OcrError: if the OCR call fails (network, model, parse, etc.).
        """
        ...

    async def aclose(self) -> None:
        """Close any resources held by the provider. Default: no-op.

        Override in subclasses that hold resources (httpx client,
        file handles, etc.). Safe to call multiple times (idempotent).
        """
        return None


# ---------------------------------------------------------------------------
# Concrete: HostedLlmOcrProvider (Sprint 4d)
# ---------------------------------------------------------------------------


# OCR prompt. Conservative: extract text, preserve structure, no commentary.
# Same prompt works for images and PDFs (the LLM client converts as needed).
_OCR_PROMPT = (
    "Extract all text from this document. Preserve the structure "
    "(paragraphs, lists, tables, headings). Return ONLY the extracted "
    "text, no commentary or explanations. If the document has no text, "
    "return an empty string."
)


class HostedLlmOcrProvider(OcrProvider):
    """OCR via the existing LLM client (provider-agnostic via the chain).

    The actual model is whatever `Settings.llm_text_primary` resolves
    to. The default is MiniMax-M3 (multimodal, 1M context) but the
    code does NOT import or reference any specific model name —
    `LLMRouter.chat` handles the chain/fallback/breaker logic.

    File reading: the file at `file_path` is read as bytes, base64
    encoded, and sent as image_url content. PDF handling is delegated
    to the LLM (it accepts base64 PDFs in most modern APIs).

    Confidence: OpenAI-compatible APIs do not return OCR confidence.
    We use 0.0 as a placeholder — the audit log records what the LLM
    actually returned. If the user wants confidence scoring, Sprint
    22+ can add a heuristic (e.g., based on text length vs file size).
    """

    def __init__(self, llm: LLMRouter) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "hosted_llm"

    async def ocr(self, file_path: str, file_id: str) -> OcrResult:
        # Read file as bytes
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except OSError as e:
            raise OcrError(
                f"Failed to read file {file_path!r}: {e}",
                provider=self.name,
                retryable=False,
            ) from e

        b64 = base64.b64encode(file_bytes).decode("ascii")

        # Detect MIME type by extension (best effort; the LLM is
        # lenient about base64 image_url format)
        mime = _guess_mime(file_path)

        # Use the existing chat() method. The router handles provider-
        # specific conversion (OpenAI image_url vs Anthropic image
        # source format, etc.). The chain routes to the configured
        # primary model (Settings.llm_text_primary, default MiniMax-M3).
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                        },
                    },
                ],
            },
        ]

        t0 = time.monotonic()
        try:
            response: LLMResponse = await self._llm.chat(messages)
        except Exception as e:
            # LLMRouter already raised LLMError after circuit breaker.
            # Wrap in OcrError for caller uniformity.
            raise OcrError(
                f"LLM call failed for file_id {file_id!r}: {e}",
                provider=self.name,
                retryable=True,
            ) from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        text = response.content or ""
        return OcrResult(
            text=text,
            confidence=0.0,  # No native OCR confidence from OpenAI-compat APIs
            model=response.model,
            provider=self.name,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Concrete: LocalVisionOcrProvider (Sprint 19.6+ Phase 3, TDD v0.6 §10)
# ---------------------------------------------------------------------------


# Default OCR prompt for vision models. Conservative: extract text,
# preserve structure, no commentary. Same intent as the hosted LLM
# prompt; vision-tuned for qwen3-vl series.
_LOCAL_VISION_OCR_PROMPT = (
    "Extract all visible text from the image. Preserve the structure "
    "(paragraphs, lists, tables, headings). Output ONLY the extracted "
    "text, no commentary or explanations. If the image has no text, "
    "return an empty string."
)


class LocalVisionOcrProvider(OcrProvider):
    """OCR via Ollama-served vision model (qwen3-vl, llava, etc.).

    Runs locally — no network egress. Default endpoint is
    http://localhost:11434 (Ollama default). Works with any
    Ollama-served vision model that supports image input.

    Sprint 19.6+ Phase 3 (TDD v0.7 §10). Fills the Sprint 22+ stub
    that was previously documented in this module's docstring. Replaces
    the edge-PC requirement for users with a single edge host with a GPU.

    qwen3-vl:8b quirks handled in code:
    - `think: false` is IGNORED by qwen3-vl on Ollama (as of 2026-07);
      model always emits `<think>...</think>` before the final answer.
    - `num_predict` MUST be >=4096 — smaller values truncate
      mid-thinking, leaving `response=""`.
    - Heavy thinking overhead: ~30-45s latency on subsequent calls.

    R1 v0.7 fixes:
    - No `prompt` parameter (TDD §10.4.1). OCR prompt is a module
      constant. Eliminates the prompt-injection surface where a caller
      could turn the provider into a generic Ollama chat client.
    - Extended error cases (TDD §10.4.4): 18 conditions, all with
      explicit retryable flag and test name.
    - 3-pattern think-strip (TDD §10.4.3) matching the §5 P0
      classifier's THINKING_TRACE_PATTERNS for design consistency.
    - Image size cap (50MB) and response size cap (10MB) to prevent
      OOM DoS from misbehaving models.
    - URL validation is at the Settings layer (pydantic field_validator
      in hermes/config.py), not here. The provider trusts the URL it's
      given (after Settings validation has rejected non-localhost).
    - `aclose()` closes the auto-created httpx client. Injected
      clients are NOT closed (caller owns lifecycle).

    Args:
        base_url: Ollama API base URL. Default `http://localhost:11434`.
            For local edge host use, localhost is the only safe setting;
            the data must NOT leave the machine. Validated at the
            Settings layer (see `hermes.config.Settings.local_vision_base_url`).
        model: Ollama model name. Default `qwen3-vl:2b` (fastest, ~2GB
            VRAM). Override to `qwen3-vl:4b` (balanced) or
            `qwen3-vl:8b` (best quality, ~6GB VRAM). Also works with
            `llava:*`, `bakllava`, etc.
        timeout_s: HTTP timeout for the Ollama call. Default 120s to
            accommodate heavy thinking-mode models.
        temperature: Generation temperature. Default 0.0 (deterministic
            for OCR; variance is undesirable).
        num_predict: Max generation length. Default 4096 to
            accommodate qwen3-vl thinking mode. Smaller values cause
            truncation mid-thinking.
        client: Optional `httpx.AsyncClient` for dependency injection
            in tests. If None, a new client is created on first use
            and reused (connection pooling). The auto-created client
            is closed by `aclose()`; the injected client is NOT.
    """

    # Hardcoded safety limits (TDD v0.7 §10.4.2). These are NOT user-
    # configurable in v1; operators who need different limits should
    # fork. Future Sprint (22+) can move to Settings if needed.
    _MAX_IMAGE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    _MAX_RESPONSE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-vl:2b",
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        num_predict: int = 4096,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._temperature = temperature
        self._num_predict = num_predict
        # Client injection for tests; None means we create one on first use.
        # The auto-created client uses max_response_size cap (TDD §10.4.2).
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return "local_vision"

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the httpx client. Reused across calls for pooling.

        Note: response size cap is NOT enforced here. httpx 0.28 does
        not support `max_response_size` in Limits (deferred to httpx
        upgrade). The image size cap (50 MB, checked before this
        client is touched) is the primary DoS defense. Future Sprint
        22+ can add response size enforcement via a custom transport.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        return self._client

    async def aclose(self) -> None:
        """Close the auto-created httpx client. Safe to call multiple times.

        Only closes the client if we own it (i.e., it was created
        internally). If the client was injected, the caller manages
        its lifecycle. TDD v0.7 §10.4.7.
        """
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ocr(self, file_path: str, file_id: str) -> OcrResult:
        """Run OCR via Ollama /api/generate with the image as input.

        Args:
            file_path: Absolute path to the image file (JPG/PNG/etc.).
            file_id: SHA-256 file_id, for logging/audit.

        Returns:
            OcrResult with extracted text (thinking blocks stripped),
            model name, latency in ms. Confidence is 0.0 (placeholder;
            Ollama doesn't return native OCR confidence).

        Raises:
            OcrError: if the OCR call fails. The exception's
                `retryable` flag indicates whether the caller should
                retry (e.g., 5xx, timeout) or fail fast (e.g., 4xx,
                file not found).
        """
        # 1. Check file size BEFORE reading (TDD v0.7 §10.4.2, MINOR-3).
        # Prevents OOM from encoding a 1GB image.
        try:
            file_size = os.path.getsize(file_path)
        except OSError as e:
            raise OcrError(
                f"Failed to stat file {file_path!r}: {e}",
                provider=self.name,
                retryable=False,
            ) from e
        if file_size > self._MAX_IMAGE_BYTES:
            raise OcrError(
                f"Image file {file_path!r} is {file_size} bytes, "
                f"exceeds max {self._MAX_IMAGE_BYTES} bytes "
                f"({self._MAX_IMAGE_BYTES // (1024 * 1024)} MB). "
                f"Resize or compress the image before retrying.",
                provider=self.name,
                retryable=False,
            )

        # 2. Read file as bytes (with explicit close via context manager,
        # TDD v0.7 §10.4.5 MINOR-5).
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except OSError as e:
            raise OcrError(
                f"Failed to read file {file_path!r}: {e}",
                provider=self.name,
                retryable=False,
            ) from e

        b64 = base64.b64encode(file_bytes).decode("ascii")

        # 3. Build request payload. `stream: false` is REQUIRED (TDD
        # v0.7 §10.4.2): with `stream: true` the response is NDJSON
        # chunks and `response.json()` would fail.
        payload = {
            "model": self._model,
            "prompt": _LOCAL_VISION_OCR_PROMPT,
            "images": [b64],
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._num_predict,
            },
        }
        url = f"{self._base_url}/api/generate"

        # 4. POST to Ollama. Use the injected/lazy client.
        client = await self._get_client()
        t0 = time.monotonic()
        try:
            resp = await client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise OcrError(
                f"Cannot connect to Ollama at {self._base_url}: {e}. "
                f"Is the daemon running? Try `ollama serve`.",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.TimeoutException as e:
            # Covers both ConnectTimeout and ReadTimeout
            raise OcrError(
                f"Ollama request timed out after {self._timeout_s}s "
                f"for file_id {file_id!r}. The model may be slow or "
                f"num_predict too high.",
                provider=self.name,
                retryable=True,
            ) from e
        except httpx.HTTPError as e:
            # Catch-all for other httpx errors (read errors, protocol errors)
            raise OcrError(
                f"Ollama HTTP error for file_id {file_id!r}: {e}",
                provider=self.name,
                retryable=True,
            ) from e

        latency_ms = int((time.monotonic() - t0) * 1000)

        # 5. Parse status code (extended error cases, TDD v0.7 §10.4.4).
        if resp.status_code == 400:
            # Bad request — model doesn't support images, malformed payload
            detail = self._safe_response_detail(resp)
            raise OcrError(
                f"Ollama rejected the request (400): {detail}. "
                f"The model may not support image input.",
                provider=self.name,
                retryable=False,
            )
        if resp.status_code == 401:
            # Ollama auth required (OLLAMA_AUTH enabled)
            raise OcrError(
                "Ollama rejected the credentials (401). Check the "
                "OLLAMA_AUTH environment variable on the Ollama daemon.",
                provider=self.name,
                retryable=False,
            )
        if resp.status_code == 403:
            # Ollama auth failed
            raise OcrError(
                "Ollama rejected the credentials (403). The configured "
                "credentials may be invalid or expired.",
                provider=self.name,
                retryable=False,
            )
        if resp.status_code == 404:
            # Model not found. The error message hints the user how
            # to fix it.
            raise OcrError(
                f"Model {self._model!r} not found in Ollama. "
                f"Run `ollama pull {self._model}` to install it.",
                provider=self.name,
                retryable=False,
            )
        if resp.status_code == 429:
            # Rate limit (user has a proxy in front of Ollama)
            raise OcrError(
                "Ollama returned 429 (rate limit). The proxy in front "
                "of Ollama is throttling requests. Wait and retry.",
                provider=self.name,
                retryable=True,
            )
        if resp.status_code == 503:
            # Service unavailable (Ollama overloaded, model loading)
            raise OcrError(
                "Ollama returned 503 (service unavailable). The model "
                "may still be loading or the daemon is overloaded. "
                "Wait and retry.",
                provider=self.name,
                retryable=True,
            )
        if 500 <= resp.status_code < 600:
            # Generic 5xx (covers 500, 501, 502, 504, etc.)
            detail = self._safe_response_detail(resp)
            raise OcrError(
                f"Ollama server error {resp.status_code}: {detail}",
                provider=self.name,
                retryable=True,
            )
        if resp.status_code != 200:
            # Unexpected status (1xx, 3xx, etc.)
            raise OcrError(
                f"Unexpected Ollama response {resp.status_code}: {resp.text[:200]}",
                provider=self.name,
                retryable=False,
            )

        # 6. Parse JSON body. Malformed JSON is NOT retryable.
        try:
            data = resp.json()
        except (ValueError, KeyError) as e:
            raise OcrError(
                f"Ollama returned invalid JSON: {e}. Response: {resp.text[:200]}",
                provider=self.name,
                retryable=False,
            ) from e

        # 7. Check for `done: true` + `done_reason` (TDD v0.7 §10.4.4).
        # `done: false` shouldn't happen with `stream: false`, but
        # defensive. `done_reason: "error"` is a model-side error.
        if data.get("done") is False:
            raise OcrError(
                f"Ollama response has done=false (unexpected for "
                f"stream=false). Response: {str(data)[:200]}",
                provider=self.name,
                retryable=True,
            )
        if data.get("done_reason") == "error":
            raise OcrError(
                f"Ollama model error: done_reason=error. Response: {str(data)[:200]}",
                provider=self.name,
                retryable=True,
            )

        # 8. Extract response text. Defensive against missing field
        # (Ollama always sends it, but cheap to check).
        if "response" not in data:
            raise OcrError(
                f"Ollama response missing 'response' field. Response: {str(data)[:200]}",
                provider=self.name,
                retryable=False,
            )

        # 9. Strip thinking traces (3-pattern + unclosed-tag fallback,
        # TDD v0.7 §10.4.3). If response is empty after strip, return
        # empty OcrResult (NOT an error — caller decides what to do).
        raw_response = data.get("response", "") or ""
        text = _strip_thinking(raw_response)
        if not text and data.get("thinking"):
            # Best-effort: if response is empty but thinking has content,
            # the model ran out of num_predict mid-thinking. Log for
            # diagnostics. We do NOT surface thinking as the answer.
            logger.warning(
                "local_vision_response_empty_after_strip",
                extra={
                    "file_id": file_id,
                    "model": self._model,
                    "thinking_len": len(data.get("thinking", "")),
                },
            )

        # 10. Compute latency from total_duration (ns → ms); fallback
        # to wall clock if total_duration is missing.
        total_duration_ns = data.get("total_duration")
        if isinstance(total_duration_ns, (int, float)) and total_duration_ns > 0:
            latency_ms_from_ollama = int(total_duration_ns // 1_000_000)
        else:
            latency_ms_from_ollama = 0

        return OcrResult(
            text=text,
            confidence=0.0,  # 0.0 means "unknown" (Ollama doesn't report
            # confidence). Do not use as a trust signal. TDD v0.7 §10.4.3.
            model=data.get("model", self._model),
            provider=self.name,
            latency_ms=latency_ms_from_ollama or latency_ms,
        )

    @staticmethod
    def _safe_response_detail(resp: httpx.Response) -> str:
        """Extract error detail from response, defensive against malformed JSON."""
        try:
            body = resp.json()
            detail = body.get("error", "")
            if not isinstance(detail, str):
                detail = str(detail)
            return detail or resp.text[:200]
        except (ValueError, KeyError):
            return resp.text[:200]


def _guess_mime(file_path: str) -> str:
    """Best-effort MIME type from file extension.

    The LLM client is lenient; this is just for the base64 data URL.
    Sprint 22+ can use a real MIME library if precision matters.
    """
    lower = file_path.lower()
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".pdf"):
        return "application/pdf"
    # Default: binary octet-stream. The LLM might reject this; for
    # unsupported extensions, callers should fall back to text-only
    # extractors (PyMuPDF, python-docx, etc.).
    return "application/octet-stream"


__all__ = [
    "HostedLlmOcrProvider",
    "LocalVisionOcrProvider",
    "OcrError",
    "OcrProvider",
    "OcrResult",
]
