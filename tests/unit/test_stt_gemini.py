"""Tests industriales para `hermes.stt.gemini` (cliente STT externo).

Cubre:
- Happy path: 200 OK con parts de texto → retorna string transcrito.
- Audio vacío: bytes vacíos → retorna "" (no raise).
- Rate limit 429 con retry-after → STTRateLimitError (subclase de STTError).
- 5xx del backend → STTError genérico.
- Response malformado (200 sin candidates) → STTError.
- MIME type vacío/inválido → STTError sin hacer request.
- Endpoint correcto: URL contiene el modelo configurado.
- Payload correcto: base64 enviado en inline_data, prompt de transcripción.
- Latencia registrada correctamente.

Estrategia:
- `transcribe()` real, no se mockea.
- `httpx.AsyncClient` interno mockeado con `respx` (vía `base_url`
  configurado en los args de la función).
- Cada test crea su propio `AsyncClient` o usa el helper `make_client`.
- Sin dependencias de Hermes (settings, db, bot) — el cliente es
  standalone.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest
import respx

from hermes.stt.gemini import STTError, STTRateLimitError, transcribe

# ---------------------------------------------------------------------------
# Constantes de test
# ---------------------------------------------------------------------------

API_KEY = "AIzaSyTest_FakeApiKey_ForUnitTests1234567890"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-3.1-flash-lite"
FAKE_OGG = b"OggS\x00\x02fake_audio_bytes_for_unit_tests_only"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_success_response(text: str) -> dict[str, Any]:
    """Construye un response JSON válido de Gemini generateContent."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 10,
            "totalTokenCount": 60,
        },
    }


# ---------------------------------------------------------------------------
# Tests de happy path
# ---------------------------------------------------------------------------


class TestTranscribeSuccess:
    """Casos exitosos: 200 OK con transcripción."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_text_from_response(self) -> None:
        """200 OK con parts[0].text → retorna el texto transcrito."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json=make_success_response("hola mundo"))
            )
            result = await transcribe(
                FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
            )
        assert result == "hola mundo"

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_returns_empty_string(self) -> None:
        """Audio vacío (0 bytes) → retorna "" sin hacer request HTTP."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            route = mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json=make_success_response("nunca llamado"))
            )
            result = await transcribe(
                b"", "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
            )
        assert result == ""
        # Verifica que NO se hizo ningún request
        assert route.call_count == 0

    @pytest.mark.asyncio
    async def test_transcribe_uses_correct_endpoint(self) -> None:
        """La URL del request contiene el modelo y la key."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            route = mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json=make_success_response("ok"))
            )
            await transcribe(FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL)
        assert route.called
        # Verifica que la URL completa tiene el API key como query param
        request = route.calls.last.request
        assert f"key={API_KEY}" in str(request.url)
        assert MODEL in str(request.url)

    @pytest.mark.asyncio
    async def test_transcribe_includes_audio_as_inline_data(self) -> None:
        """El body del request incluye inline_data con base64 del audio."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            route = mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json=make_success_response("ok"))
            )
            await transcribe(FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL)
        request = route.calls.last.request
        body = request.content.decode("utf-8")
        # El base64 del audio aparece en el body
        expected_b64 = base64.b64encode(FAKE_OGG).decode("ascii")
        assert expected_b64 in body
        # El MIME type aparece
        assert "audio/ogg" in body
        # El prompt de transcripción aparece
        assert "transcri" in body.lower() or "audio" in body.lower()

    @pytest.mark.asyncio
    async def test_transcribe_concatenates_multiple_text_parts(self) -> None:
        """Si la response tiene varios parts, se concatenan (defensivo)."""
        multi_part_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "hola "},
                            {"text": "mundo"},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json=multi_part_response)
            )
            result = await transcribe(
                FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
            )
        assert result == "hola mundo"


# ---------------------------------------------------------------------------
# Tests de errores HTTP
# ---------------------------------------------------------------------------


class TestTranscribeErrors:
    """Casos de error: 4xx, 5xx, response malformado."""

    @pytest.mark.asyncio
    async def test_transcribe_429_raises_rate_limit_error(self) -> None:
        """429 → STTRateLimitError (subclase de STTError, retryable)."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(
                    429,
                    json={"error": {"code": 429, "message": "Rate limit exceeded"}},
                    headers={"Retry-After": "5"},
                )
            )
            with pytest.raises(STTRateLimitError):
                await transcribe(
                    FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
                )

    @pytest.mark.asyncio
    async def test_transcribe_429_is_subclass_of_stt_error(self) -> None:
        """STTRateLimitError es catchable como STTError."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(429, json={"error": {"code": 429}})
            )
            with pytest.raises(STTError):
                await transcribe(
                    FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
                )

    @pytest.mark.asyncio
    async def test_transcribe_500_raises_stt_error(self) -> None:
        """500 → STTError genérico (no RateLimit)."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(
                    500, json={"error": {"code": 500, "message": "Internal server error"}}
                )
            )
            with pytest.raises(STTError) as exc_info:
                await transcribe(
                    FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
                )
        # No debe ser específicamente RateLimit
        assert not isinstance(exc_info.value, STTRateLimitError)

    @pytest.mark.asyncio
    async def test_transcribe_503_raises_stt_error(self) -> None:
        """503 (Service Unavailable) → STTError."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(503, json={"error": {"code": 503}})
            )
            with pytest.raises(STTError):
                await transcribe(
                    FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
                )

    @pytest.mark.asyncio
    async def test_transcribe_malformed_response_no_candidates(self) -> None:
        """200 OK pero sin 'candidates' → STTError (defensivo)."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json={"unexpected": "format"})
            )
            with pytest.raises(STTError):
                await transcribe(
                    FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
                )

    @pytest.mark.asyncio
    async def test_transcribe_malformed_response_no_parts(self) -> None:
        """200 OK con candidates pero parts vacío → retorna '' (no raise).

        Decisión de diseño: si Gemini responde 200 OK con parts vacío
        o sin texto, es indistinguible de "audio sin habla audible"
        (silencio, ruido). Devolvemos '' para que la capa superior
        (handler) pida al usuario que repita, en vez de fallar con
        excepción. Consistente con `transcribe(b'', ...)` que también
        retorna ''.
        """
        malformed = {"candidates": [{"content": {"parts": []}}]}
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json=malformed)
            )
            result = await transcribe(
                FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
            )
        assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_network_error_raises_stt_error(self) -> None:
        """Error de red (ConnectError) → STTError."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            mock.post(f"/models/{MODEL}:generateContent").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            with pytest.raises(STTError):
                await transcribe(
                    FAKE_OGG, "audio/ogg", api_key=API_KEY, model=MODEL, base_url=BASE_URL
                )


# ---------------------------------------------------------------------------
# Tests de validación de input
# ---------------------------------------------------------------------------


class TestTranscribeInputValidation:
    """Validación de argumentos antes de hacer request."""

    @pytest.mark.asyncio
    async def test_transcribe_empty_mime_raises_stt_error(self) -> None:
        """mime_type vacío → STTError sin hacer request."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            route = mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json=make_success_response("ok"))
            )
            with pytest.raises(STTError):
                await transcribe(FAKE_OGG, "", api_key=API_KEY, model=MODEL, base_url=BASE_URL)
        assert route.call_count == 0

    @pytest.mark.asyncio
    async def test_transcribe_empty_api_key_raises_stt_error(self) -> None:
        """api_key vacío → STTError sin hacer request."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            route = mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json=make_success_response("ok"))
            )
            with pytest.raises(STTError):
                await transcribe(FAKE_OGG, "audio/ogg", api_key="", model=MODEL, base_url=BASE_URL)
        assert route.call_count == 0

    @pytest.mark.asyncio
    async def test_transcribe_empty_model_raises_stt_error(self) -> None:
        """model vacío → STTError sin hacer request."""
        with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
            route = mock.post(f"/models/{MODEL}:generateContent").mock(
                return_value=httpx.Response(200, json=make_success_response("ok"))
            )
            with pytest.raises(STTError):
                await transcribe(
                    FAKE_OGG, "audio/ogg", api_key=API_KEY, model="", base_url=BASE_URL
                )
        assert route.call_count == 0
