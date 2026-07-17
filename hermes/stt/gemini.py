"""Cliente STT (Speech-to-Text) basado en Gemini multimodal.

Por qué Gemini 3.1 Flash Lite:
- Multimodal nativo: acepta audio inline en `generateContent`.
- Free tier generoso: 500 RPD / 15 RPM (suficiente para uso personal/familiar).
- 0% CPU en el host (es API).
- Latencia: 1-2s para audios cortos (<1 min).

Endpoint oficial:
POST {base_url}/models/{model}:generateContent?key={api_key}
Body:
  {
    "contents": [{
      "parts": [
        {"text": "Transcribe..."},
        {"inline_data": {"mime_type": "audio/ogg", "data": "<base64>"}}
      ]
    }]
  }
Response:
  {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}

Ver:
- https://ai.google.dev/gemini-api/docs/audio
- Bug histórico opencode/opencode#30389 (Sprint 12-: ya no aplica
  porque MiniMax-M3 procesa audio nativo; STT Gemini se mantiene
  por coste y aislamiento de cuota — ver hermes/stt/__init__.py).
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Prompt de transcripción. Puro y directo: "solo el contenido del audio".
# Sin instrucciones extra: Gemini infiere idioma del audio.
_TRANSCRIBE_PROMPT = (
    "Transcribe este audio a texto. "
    "Responde SOLO con la transcripción, sin comillas, sin prefijos, sin comentarios."
)


class STTError(Exception):
    """Error genérico al transcribir audio."""


class STTRateLimitError(STTError):
    """Rate limit alcanzado (HTTP 429). Retryable."""


async def transcribe(
    audio_bytes: bytes,
    mime_type: str,
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout_s: float = 10.0,
) -> str:
    """Transcribe un audio usando Gemini multimodal.

    Args:
        audio_bytes: contenido del audio (OGG, MP3, WAV, etc.).
        mime_type: MIME type del audio (ej. "audio/ogg", "audio/mpeg").
        api_key: Google API key con acceso a Gemini.
        model: nombre del modelo Gemini (ej. "gemini-3.1-flash-lite").
        base_url: URL base de la API de Gemini.
        timeout_s: timeout HTTP en segundos. Default 10s (audios cortos).

    Returns:
        Texto transcrito. Cadena vacía si el audio está vacío.

    Raises:
        STTError: error genérico (input inválido, 5xx, response malformado).
        STTRateLimitError: HTTP 429 (subclase de STTError).
    """
    # Validación de input (fail fast, sin request HTTP)
    if not audio_bytes:
        logger.debug("stt_empty_audio_skip")
        return ""
    if not mime_type or not mime_type.strip():
        raise STTError("mime_type is required and cannot be empty")
    if not api_key or not api_key.strip():
        raise STTError("api_key is required and cannot be empty")
    if not model or not model.strip():
        raise STTError("model is required and cannot be empty")

    # Codificar audio en base64 (inline_data lo requiere)
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    # Construir payload (formato Gemini multimodal)
    payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"text": _TRANSCRIBE_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_b64,
                        }
                    },
                ]
            }
        ]
    }

    # Request HTTP
    url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
    params = {"key": api_key}
    timeout = httpx.Timeout(timeout_s)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, params=params, json=payload)
        except httpx.HTTPError as exc:
            # ConnectError, TimeoutException, etc.
            logger.warning("stt_network_error", extra={"error": str(exc)})
            raise STTError(f"Network error: {exc}") from exc

    # Manejo de errores HTTP
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "unknown")
        logger.warning("stt_rate_limit", extra={"retry_after": retry_after})
        raise STTRateLimitError(f"Rate limit exceeded (retry after {retry_after}s)")
    if resp.status_code >= 400:
        # 4xx/5xx genérico
        try:
            error_body = resp.json()
        except Exception:
            error_body = {"raw": resp.text[:500]}
        logger.warning(
            "stt_http_error",
            extra={"status": resp.status_code, "body": str(error_body)[:200]},
        )
        raise STTError(f"HTTP {resp.status_code}: {error_body}")

    # Parsear response exitoso
    try:
        data = resp.json()
    except Exception as exc:
        raise STTError(f"Malformed JSON response: {exc}") from exc

    try:
        candidates = data["candidates"]
        first_candidate = candidates[0]
        parts = first_candidate["content"]["parts"]
        # Concatenar todos los parts de texto (defensivo)
        texts = [p["text"] for p in parts if p.get("text")]
        transcription = "".join(texts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("stt_malformed_response", extra={"data_keys": list(data.keys())})
        raise STTError(f"Malformed Gemini response: {exc}") from exc

    if not transcription:
        # 200 OK pero transcripción vacía: Gemini no entendió el audio
        logger.info("stt_empty_transcription")
        return ""

    logger.debug("stt_success", extra={"chars": len(transcription)})
    return transcription
