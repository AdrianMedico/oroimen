"""LLMRouter con circuit breaker y smart routing (texto y voz unificados en v1.2).

Soporta dos formatos de API (detectados por nombre de modelo):
- OpenAI-compatible: /chat/completions
  - MiniMax (M3, M2.7, M2.5, M2.1, M2): via https://api.minimax.io/v1
  - Modelos legacy OpenCode Go (deprecados Sprint 12+): mimo, kimi, glm, deepseek
- Anthropic-compatible: /messages
  - Solo qwen3.*-plus, qwen3.*-max (legacy OpenCode Go)
  - NOTA: MiniMax tambien expone endpoint Anthropic en
    https://api.minimax.io/anthropic pero NO lo usamos por estabilidad
    con tools y thinking.

Smart routing v1.2 (unificado):
- Texto y voz: chain = [primary, fallback] (configurable).
- Default Sprint 12+ (MiniMax API): MiniMax-M3 → MiniMax-M2.7-highspeed.
- Antes (v1.0/v1.2, legacy OpenCode Go): el routing primario→fallback
  equivalente se hacia via el provider opencode-go. Esa capa intermedia
  fue removida en Sprint 12+ y el código ahora habla directo con
  MiniMax API (OPENCODE_GO_BASE_URL=https://api.minimax.io/v1 por default).
- Bug historico opencode/opencode#30389 motivo el bypass del audio
  via mimo-v2.5 (unico modelo de Go con audio que no procesaba
  input_audio). Ahora se resuelve nativamente con MiniMax-M3
  (multimodal con thinking nativo); STT externo Gemini Flash Lite se
  mantiene por aislamiento de cuota y coste (ver .env.example §STT).

Si en algun momento queremos volver al path legacy opencode-go (poco
probable — la quota se agoto y MiniMax API direct es estrictamente
mejor en coste/latencia/thinking), basta con setear
OPENCODE_GO_BASE_URL=https://opencode.ai/zen/go/v1 y los defaults v1.2
del routing vuelven a aplicar. Las env vars OPENCODE_GO_* se mantienen
por compat con deploy scripts y Credential Manager entries.

Referencia: https://platform.minimax.io/docs/llms.txt
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from hermes.llm.breaker import CircuitBreaker, CircuitOpenError
from hermes.llm.chatgpt5_6 import ChatGpt5_6Client

# Sprint 19.6+ Phase 5 (OpenAI Build Week): Ollama is a local LLM
# provider. The TYPE_CHECKING import is sufficient here because the
# real OllamaClient is only instantiated in LLMRouter.__init__ when
# the primary provider is "ollama"; machines that never use Ollama
# pay no import cost (the import is lazy inside the constructor's
# _maybe_instantiate_ollama helper).
if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.llm.ollama import OllamaClient

logger = logging.getLogger(__name__)

# SUGGESTION 3 (Nemotron r4 2026-07-08): Anthropic solo soporta estos
# media_types para image blocks. Cualquier otro (image/svg+xml,
# application/pdf, image/bmp, image/tiff) causa 400 "invalid media_type".
# Mantener allowlist sincronizado con la doc Anthropic API.
_ANTHROPIC_IMAGE_MEDIA_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)

# Tamaño máximo del snippet crudo que se loggea junto a un error de parseo.
# Suficiente para identificar la causa raíz (HTML de Cloudflare, error de proxy,
# JSON truncado) sin filtrar grandes volúmenes de texto al log.
_RAW_SNIPPET_MAX = 200


@dataclass
class ToolCall:
    """Un tool call solicitado por el LLM.

    Attributes:
        id: ID único del tool call (asignado por el LLM).
        name: nombre de la tool (debe estar registrada en ToolRegistry).
        arguments: argumentos parseados del LLM.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a OpenAI tool_call format (usado en SSE deltas).

        Formato: {"id": "call_1", "type": "function",
        "function": {"name": "tool", "arguments": "{...}"}}
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


def _safe_json_parse(resp: httpx.Response, provider: str) -> dict[str, Any]:
    """Parsea `resp.json()` capturando excepciones no-HTTP.

    Los providers upstream a veces devuelven HTML (errores de proxy,
    Cloudflare 5xx, gateways rotos) o texto plano en vez de JSON.
    `httpx.Response.json()` lanza `json.JSONDecodeError` (subclase
    de `ValueError`) o `ValueError` en esos casos.

    Esta función captura ambos y los convierte en `LLMError` con un
    snippet crudo truncado para debug. Esto evita que un error de
    parseo no-controlado propague como excepción genérica y rompa
    el chain de fallback.

    Args:
        resp: respuesta httpx ya validada por `raise_for_status()`.
        provider: "openai" o "anthropic" (solo para logs).

    Returns:
        Dict parseado del JSON.

    Raises:
        LLMError: si la respuesta no es JSON válido.
    """
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raw_snippet = resp.text[:_RAW_SNIPPET_MAX]
        logger.warning(
            "llm_invalid_json_response",
            extra={
                "provider": provider,
                "status": resp.status_code,
                "content_type": resp.headers.get("content-type", ""),
                "raw_snippet": raw_snippet,
                "error": str(exc),
            },
        )
        raise LLMError(
            f"Invalid JSON from {provider} (status={resp.status_code}, "
            f"content_type={resp.headers.get('content-type', '?')}): {exc}"
        ) from exc
    if not isinstance(data, dict):
        raw_snippet = resp.text[:_RAW_SNIPPET_MAX]
        logger.warning(
            "llm_response_not_dict",
            extra={
                "provider": provider,
                "status": resp.status_code,
                "data_type": type(data).__name__,
                "raw_snippet": raw_snippet,
            },
        )
        raise LLMError(
            f"Unexpected JSON shape from {provider} (expected dict, got {type(data).__name__})"
        )
    return data


class LLMError(Exception):
    """Error genérico al invocar el LLM."""


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Sprint 5 T51: reasoning_content es la cadena de pensamiento que
    # emiten los modelos en thinking mode (DeepSeek, OpenAI o1, etc.).
    # Se persiste en DB y se re-inyecta en iteraciones siguientes (e.g.
    # tras un tool_call) para cumplir el contrato de la API:
    # "reasoning_content must be passed back to the API" (DeepSeek).
    # Vacio para providers sin thinking mode (Anthropic) o cuando el
    # LLM no genera tokens de razonamiento.
    reasoning_content: str = ""


# Sprint 7 T53.2: chunk de streaming.
# Dataclass (no dict libre) para tener type checking, IDE autocomplete,
# y serializacion explicita a SSE via to_sse() / to_done().
#
# Cada chunk tiene UNO de los siguientes campos populated (los demas None):
# - content: delta de prosa del LLM (se reenvia al cliente)
# - tool_calls: lista de ToolCall (solo en el chunk final, cuando llega
#   finish_reason="tool_calls")
# - finish_reason: "stop" | "tool_calls" | "length" | "content_filter"
# - reasoning_content: delta de thinking mode (DeepSeek)
# - hermes_status: extension custom para tool progress (ver TDD v2.1 §2.2)
# - model: nombre del modelo que esta streameando (para tracking en agent loop)
@dataclass
class StreamChunk:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    reasoning_content: str | None = None
    hermes_status: dict | None = None
    model: str | None = None
    # Sprint 9.3.2: usage metadata (tokens). Internal only - not serialized
    # to SSE (clients compute their own from content length). Populated when
    # upstream sends {"usage": {...}} in the final chunk (requires
    # stream_options={"include_usage": true} in payload).
    usage: dict | None = None
    # Sprint 16.8.2 (2026-07-07): True cuando la respuesta fue truncada por
    # el LLM (stop_reason=max_tokens o equivalente). El cliente (Open WebUI)
    # debe mostrar un aviso "Respuesta truncada, pedí 'continúa' para más".
    truncated: bool = False

    def to_sse(self, *, model_override: str | None = None) -> str:
        """Serializa a SSE, optionally exposing a public API model alias."""
        delta: dict = {}
        if self.content is not None:
            delta["content"] = self.content
        if self.reasoning_content is not None:
            # Sprint 16.7 (2026-07-06) fix: passthrough de reasoning_content
            # para Open WebUI y clientes que renderizan thinking mode.
            # Antes se capturaba del delta del LLM (router.py:696) pero se
            # descartaba en la serializacion. Ahora se incluye en el delta
            # con la convencion OpenAI reasoning_content.
            delta["reasoning_content"] = self.reasoning_content
        if self.tool_calls is not None:
            delta["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.hermes_status is not None:
            delta["hermes_status"] = self.hermes_status
        choice: dict = {"delta": delta}
        if self.finish_reason:
            choice["finish_reason"] = self.finish_reason
        # Sprint 16.8.2: avisar al cliente si la respuesta fue truncada por el
        # LLM (max_tokens). Estilo OpenAI response field. Adicionalmente se
        # añade un trailer a hermes_status para que clientes que solo lean
        # ese campo (sin top-level 'truncated') también lo vean.
        if self.truncated:
            choice["truncated"] = True
            # Sprint 16.8.3 (M-2 fix): shallow-copy para no mutar
            # self.hermes_status in-place. Footgun para callers que
            # retengan referencia al dict original.
            existing_status = dict(self.hermes_status or {})
            existing_status["truncated"] = True
            existing_status["truncation_reason"] = "max_tokens"
            delta["hermes_status"] = existing_status
        payload: dict[str, Any] = {"choices": [choice]}
        if model_override is not None:
            payload["model"] = model_override
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def to_done(self) -> str:
        """Sentinel de fin de stream (data: [DONE]\\n\\n)."""
        return "data: [DONE]\n\n"


# Sprint 7 T53.2: helper para merge de tool_call deltas en streaming.
#
# OpenAI API entrega tool_call.arguments en CHUNKS de string JSON.
# Ejemplo real de stream (un solo tool call, 3 deltas):
#   delta.tool_calls = [{index: 0, id: "call_1", function: {name: "x", arguments: ""}}]
#   delta.tool_calls = [{index: 0, function: {arguments: "{"}}]
#   delta.tool_calls = [{index: 0, function: {arguments: "\"a\":"}}]
#   delta.tool_calls = [{index: 0, function: {arguments": "1}"}}]
# Tras el ultimo delta, arguments acumulado = "{\"a\":1}", json.loads OK.
#
# Robusta contra:
# - Out-of-order deltas (ordenamos por index)
# - Multiples tool calls en la misma respuesta (cada una con su index)
# - Campos parciales (algunos deltas solo tienen arguments, otros tienen id)
# - JSON invalido al final (raise LLMError para que la chain caiga al siguiente modelo)
# - arguments vacio (args = {})
def _merge_tool_call_deltas(
    deltas: list[dict],
) -> list[ToolCall]:
    """Convierte la lista de tool_call deltas del stream en ToolCall completos.

    Raises:
        LLMError: si el arguments acumulado no es JSON valido. Esto es
        bug del modelo upstream (deberia emitir JSON valido), no
        recuperable. La chain cae al siguiente modelo.
    """
    # Agrupar por index. Usamos dict[int, dict] para merge incremental.
    by_index: dict[int, dict] = {}
    for d in deltas:
        idx = d.get("index", 0)
        if idx not in by_index:
            by_index[idx] = {
                "id": None,
                "type": "function",
                "function": {"name": None, "arguments": ""},
            }
        agg = by_index[idx]
        # id: solo viene en el primer delta (a veces)
        if d.get("id"):
            agg["id"] = d["id"]
        # [type]: viene en el primer delta
        if d.get("type"):
            agg["type"] = d["type"]
        # function.name: viene en el primer delta
        fn_delta = d.get("function", {})
        if fn_delta.get("name"):
            agg["function"]["name"] = fn_delta["name"]
        # function.arguments: viene en TODOS los deltas, concatenar
        args_piece = fn_delta.get("arguments")
        if args_piece:
            agg["function"]["arguments"] += args_piece

    # Convertir a ToolCall, parsear arguments JSON.
    result: list[ToolCall] = []
    for idx in sorted(by_index.keys()):
        agg = by_index[idx]
        if agg["id"] is None:
            # No recibimos id en ningun delta. Stream corrupto.
            raise LLMError(
                f"tool_call delta missing id (index={idx}). Model emitted invalid stream."
            )
        if agg["function"]["name"] is None:
            raise LLMError(f"tool_call delta missing function name (index={idx}).")
        # Parsear arguments. Si vacio, default a {}.
        args_str = agg["function"]["arguments"].strip() or "{}"
        try:
            args = json.loads(args_str)
            if not isinstance(args, dict):
                # arguments debe ser un dict JSON, no un array o escalar.
                raise LLMError(
                    f"tool_call arguments must be a JSON object, got {type(args).__name__}: {args_str[:200]}"
                )
        except json.JSONDecodeError as exc:
            # Stream termino con arguments invalido. Bug del modelo upstream.
            raise LLMError(
                f"tool_call arguments invalid JSON (index={idx}): {args_str[:200]}"
            ) from exc

        result.append(
            ToolCall(
                id=agg["id"],
                name=agg["function"]["name"],
                arguments=args,
            )
        )
    return result


# Sprint 5 T51 (3.7): helper extraida de _invoke_anthropic para
# (a) testabilidad directa sin mockear HTTP y (b) defense in depth
# contra contaminacion de campos OpenAI-specific en el payload Anthropic.
# Si un turno previo populo reasoning_content en la DB (lo hacian modelos
# OpenAI-compatible tipo deepseek-v4-flash del chain legacy OpenCode Go),
# en un turno siguiente con un modelo Anthropic-compatible este helper
# purga cualquier campo OpenAI-specific (reasoning_content, tool_call_id,
# y futuros como refusal, audio, etc.) reconstruyendo el dict con solo
# role y content. Patron estandar de adaptadores de protocolo.
def _convert_openai_vision_to_anthropic(content_blocks: list[dict]) -> list[dict]:
    """Convierte content blocks OpenAI vision → Anthropic vision.

    OpenAI format (entrada):
        {"type": "text", "text": "..."}
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXX"}}
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}

    Anthropic format (salida):
        {"type": "text", "text": "..."}
        {"type": "image", "source": {"type": "base64",
                                   "media_type": "image/png",
                                   "data": "XXX"}}
        {"type": "image", "source": {"type": "url", "url": "https://..."}}

    SPRINT 18 HOTFIX (2026-07-08): MiniMax Anthropic-compat endpoint
    (https://api.minimax.io/anthropic/v1/messages) rechaza el formato
    OpenAI image_url con HTTP 400. Sin esta conversion, mensajes con
    imagenes adjuntas desde WebUI fallan en TODA la cadena de modelos
    (todos los del chain actual van por el path Anthropic).

    Bug class: contract violation. El internal format sigue la
    convencion OpenAI (vision support desde Sprint 7.3), pero el path
    Anthropic requiere conversion explicita. El helper principal
    (_transform_to_anthropic_messages) ahora detecta content=list y
    aplica esta conversion.

    Edge cases:
    - data: URL malformada → skip block (no raise). Logica defensiva:
      preferimos perder UNA imagen a fallar el mensaje entero (el LLM
      todavia recibe el text part).
    - HTTP URL → Anthropic url source (funciona para URLs publicamente
      accesibles).
    - Block type desconocido → pass-through (puede causar 400, pero no
      debemos silenciar content del user).
    """
    result: list[dict] = []
    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "text":
            # text blocks son identicos en OpenAI y Anthropic.
            result.append(block)
        elif block_type == "image_url":
            url = block.get("image_url", {}).get("url", "")
            # SUGGESTION 1 (MiMo v2.5 review 2026-07-08): defensa contra
            # url vacia o ausente. Sin esto el else branch creaba
            # source={"type": "url", "url": ""} → Anthropic 400.
            if not url:
                logger.warning(
                    "_convert_openai_vision_to_anthropic: empty url in "
                    "image_url block, skipping image block"
                )
                continue
            if url.startswith("data:"):
                # data:[<mediatype>];base64,<data>
                # Split en header y data. Si falla el formato, skip
                # el block (defensa: perder 1 imagen < romper el chat).
                # Si media_type falta (data:;base64,...), default a
                # image/png (Anthropic rechaza media_type vacio).
                try:
                    header, b64data = url.split(",", 1)
                    # media_type esta entre "data:" y ";".
                    # split(";")[0] drops el sufijo ";base64" antes de
                    # split(":"), asi media_type queda limpio.
                    parts = header.split(";")[0].split(":", 1)
                    media_type = parts[1] if len(parts) > 1 and parts[1] else "image/png"
                    # SUGGESTION 3 (Nemotron r4 2026-07-08): Anthropic
                    # solo soporta image/jpeg, image/png, image/gif,
                    # image/webp. Otro media_type (e.g., image/svg+xml,
                    # application/pdf) → 400. Allowlist + fallback.
                    if media_type not in _ANTHROPIC_IMAGE_MEDIA_TYPES:
                        logger.warning(
                            "_convert_openai_vision_to_anthropic: "
                            "unsupported media_type %r, defaulting to "
                            "image/png",
                            media_type,
                        )
                        media_type = "image/png"
                except (ValueError, IndexError):
                    logger.warning(
                        "_convert_openai_vision_to_anthropic: malformed "
                        "data URL, skipping image block"
                    )
                    continue
                result.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    }
                )
            else:
                # SUGGESTION 1 (MiMo v2.5 review r2 2026-07-08): SSRF
                # defense. Non-http schemes (file:///, javascript:, etc.)
                # pasarian tal cual a Anthropic. Whitelist http/https;
                # skip+warning cualquier otro scheme.
                # lower() para HTTP:// (RFC 3986 case-insensitive scheme).
                if not url.lower().startswith(("http://", "https://")):
                    logger.warning(
                        "_convert_openai_vision_to_anthropic: non-http "
                        "url scheme %r, skipping image block",
                        url[:32],
                    )
                    continue
                # HTTP/HTTPS URL → Anthropic url source.
                result.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    }
                )
        else:
            # Block type desconocido — pass-through. Puede causar 400
            # si Anthropic no lo reconoce, pero no debemos silenciar
            # content del user.
            result.append(block)
    return result


def _transform_to_anthropic_messages(user_assistant: list[dict]) -> list[dict]:
    """Transforma mensajes del formato interno a formato Anthropic /v1/messages.

    Reglas:
    - role="tool" → role="user" con content=[{type: tool_result, ...}]
    - role="assistant" con tool_calls → role="assistant" con
      content=[text, tool_use blocks] (formato Anthropic)
    - resto (user, system, assistant sin tools) → {role, content}
      RECONSTRUIDO (no se pasa m directamente, se blindan campos
      OpenAI-specific que podrian estar en DB por turnos previos).

    Si `content` es una LISTA (vision path, OpenAI ContentPart format),
    convierte los image_url blocks a Anthropic image blocks antes de
    enviar. Ver `_convert_openai_vision_to_anthropic` para el formato
    esperado y el bug class.

    Por que helper module-level (no metodo de la clase):
    - No necesita acceso a self (puro transformation).
    - Tests pueden importarla y verificar el output sin instanciar
      router ni mockear httpx.
    - Reutilizable si en el futuro queremos transformar mensajes para
      otro provider (e.g., Gemini native).
    """
    result: list[dict] = []
    for m in user_assistant:
        role = m.get("role")
        if role == "tool":
            # Convertir role="tool" a role="user" con content tool_result
            tool_call_id = m.get("tool_call_id", "")
            content = m.get("content", "")
            result.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content,
                        }
                    ],
                }
            )
        elif role == "assistant" and m.get("tool_calls"):
            # Convertir assistant con tool_calls (formato OpenAI) a
            # bloques tool_use en el content (formato Anthropic).
            tool_use_blocks: list[dict] = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args_raw = fn.get("arguments", "{}")
                try:
                    args_dict = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args_dict = {}
                tool_use_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args_dict,
                    }
                )
            # Si hay content ademas de tool_calls, anadirlo como text block
            text_content = m.get("content")
            content_blocks: list[dict] = []
            if text_content:
                content_blocks.append({"type": "text", "text": text_content})
            content_blocks.extend(tool_use_blocks)
            result.append({"role": "assistant", "content": content_blocks})
        else:
            # Defense in depth (T51 §3.7): reconstruir dict con solo
            # role + content. Asi evitamos que campos OpenAI-specific
            # (reasoning_content, tool_call_id en assistant sin tools,
            # refusal, audio, etc.) lleguen a la API de Anthropic y
            # causen 400 "Unknown field".
            content = m.get("content", "")
            if isinstance(content, list):
                # Vision content: OpenAI ContentPart format (text +
                # image_url). Convertir a Anthropic format antes de
                # enviar al endpoint /v1/messages. Sin esto, MiniMax
                # Anthropic-compat devuelve 400. Ver
                # `_convert_openai_vision_to_anthropic`.
                # Guard: content=[] (lista vacia) Anthropic rechaza;
                # caer al branch de text plano con string vacio.
                if not content:
                    result.append({"role": m["role"], "content": ""})
                else:
                    result.append(
                        {
                            "role": m["role"],
                            "content": _convert_openai_vision_to_anthropic(content),
                        }
                    )
            else:
                result.append({"role": m["role"], "content": content})
    return result


# Familias (prefijos) de modelos que usan el endpoint Anthropic-compatible
# (/v1/messages). Solo los modelos qwen3 de OpenCode Go. Los MiniMax
# (con capital M, MiniMax-M3 etc.) van por el path OpenAI directo
# porque MiniMax API expone ambos pero el OpenAI path soporta
# thinking, tool calls y streaming completo (recomendado).
# Lista oficial (verificada 2026-06-19 en /docs/es/go/): qwen3.*-plus
_ANTHROPIC_FAMILIES: tuple[str, ...] = (
    "minimax-",  # prefijos Anthropic-compatible (lowercased para matching
    # via model_id.lower() en is_anthropic_model). Incluye
    # legacy "minimax-m3" y familias MiniMax-style.
    "qwen3.",  # qwen3.5-plus, qwen3.6-plus, qwen3.7-plus, qwen3.7-max
)

# Excepciones: modelos que son Anthropic-compatible por familia pero
# que tenian bugs conocidos en el flujo de tool_use/tool_result via
# el provider opencode-go (legacy, ya no se usa desde Sprint 12).
#
# Sprint 16.7 (PR #98, 2026-07-06): emptied. Production base URL is
# https://api.minimax.io/anthropic/v1 (per current .env). The previous
# workaround forced minimax-M3 to OpenAI path (`/chat/completions`),
# producing URL `/anthropic/v1/chat/completions` which does NOT exist
# (404). With this empty tuple, all anthropic-family models use
# `/messages` (`/anthropic/v1/messages`) which IS supported by
# MiniMax API direct.
#
# Sprint 17+ (provider abstraction, see SP16_FOLLOWUP F12) will
# eliminate this concept entirely: routing will be provider-based,
# not model-specific.
#
# IMPORTANTE: los literales aqui son lowercase porque la comparacion
# en force_openai_for_tools usa model_id.lower() in tuple.
# Sprint 16.7 (PR #98, 2026-07-06): empty. The bug 2013 workaround
# was originally needed for opencode-go (legacy provider), but since
# Sprint 12+ we talk to MiniMax API directly. The workaround now
# forces OpenAI path (`/chat/completions`) on top of an Anthropic
# base URL (`https://api.minimax.io/anthropic/v1`), producing a URL
# that does NOT exist (`/anthropic/v1/chat/completions` = 404).
# With this empty tuple, all anthropic-family models use the correct
# Anthropic path (`/messages`), which works on MiniMax API direct.
# Provider-specific path routing is tracked in SP16_FOLLOWUP §F12.
_ANTHROPIC_FAMILIES_EXCEPTIONS: tuple[str, ...] = ()


def is_anthropic_model(model_id: str) -> bool:
    """Detecta si un modelo es de la familia Anthropic-compatible.

    Coincide por prefijo de familia (no por nombre exacto) para que
    funcione con modelos nuevos sin tener que actualizar el código.
    Ejemplos historicos: "MiniMax-M4" (legacy opencode-go), "qwen3.8-plus".

    Esta funcion NO considera las excepciones. Para chequear si un
    modelo debe usar path OpenAI forzado, usa
    `force_openai_for_tools(model_id)`.
    """
    m = model_id.lower()
    return any(m.startswith(prefix) for prefix in _ANTHROPIC_FAMILIES)


def force_openai_for_tools(model_id: str) -> bool:
    """True si el modelo debe usar path OpenAI cuando hay tools.

    Workaround historico para el bug 2013 en opencode-go + MiniMax:
    el path Anthropic fallaba con "tool call and result not match"
    cuando el payload tenia tool_use/tool_result. Forzamos path OpenAI
    (que funciona correctamente con tool_call_id) en ese caso.

    Sin tools, el path Anthropic normal funciona bien, asi que NO
    forzamos el cambio (mantenemos compatibilidad con todos los
    tests y casos de uso que no involucran tools).

    Nota: aunque ahora hablamos con MiniMax API directo (no via
    opencode-go), el workaround se mantiene por defensa (ver bloque
    de comentario sobre `_ANTHROPIC_FAMILIES_EXCEPTIONS`).
    """
    return model_id.lower() in _ANTHROPIC_FAMILIES_EXCEPTIONS


# Sprint 19.6+ Phase 5 (OpenAI Build Week): per-model provider
# dispatch. The LLMRouter stores a `_ollama_models` set built from
# the Settings (model name -> provider hint). When a chain iteration
# reaches a model in this set, the router dispatches to the
# OllamaClient (local, OpenAI-compat) instead of the main
# `httpx.AsyncClient` (which talks to the cloud provider).
#
# Detection rules:
# 1. Explicit provider hint: `llm_text_primary_provider == "ollama"`
#    AND model == `llm_text_primary` -> Ollama tier.
# 2. No fallback provider hint for now (the default fallback is a
#    MiniMax model with an API key). If we add a fallback Ollama
#    model in the future, set `llm_text_fallback_provider = "ollama"`
#    and add it here.
#
# This is NOT a prefix-based check: the model name `qwen2.5:7b` is
# the Ollama naming convention, but other models could also be
# served by Ollama (e.g., `llama3.1:8b`). Provider hint is explicit
# to avoid false-positives (a model named `qwen2.5:7b` served by a
# non-Ollama backend would NOT match).
_OLLAMA_PROVIDER_HINT = "ollama"


def _is_ollama_provider_hint(provider: str | None) -> bool:
    """True if a provider-hint string indicates the Ollama tier.

    Normalizes: trims whitespace, lower-cases, treats None/empty as
    NOT Ollama. This is the single source of truth for "is this
    provider hint ollama" so the chain-detection logic in
    `LLMRouter` doesn't have to repeat the normalization.
    """
    if not provider:
        return False
    return provider.strip().lower() == _OLLAMA_PROVIDER_HINT


def _await(coro: Awaitable):
    """Envuelve una coroutine en un callable. Evita capturar valores en lambdas."""

    async def _wrap() -> object:
        return await coro

    return _wrap


class LLMRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._breakers: dict[str, CircuitBreaker] = {}
        # v1.2: chain unificado (texto y voz usan el mismo).
        # Inicializamos circuit breakers para todos los modelos del chain.
        for model in settings.text_chain:
            self._breakers[model] = CircuitBreaker(
                fail_max=settings.circuit_breaker_fail_max,
                reset_timeout=float(settings.circuit_breaker_reset_timeout),
                name=model,
            )
        # Sprint 12 (ADR-007): tambien registramos breakers para los modelos
        # en `settings.llm_model_overrides` (las chains dedicadas de cada
        # alias). Sin esto, `self._breakers[model]` lanzaria KeyError
        # cuando un cliente pida `model: "oroimen-agent-fast"` y la chain
        # del alias incluya modelos fuera de `text_chain` (Copilot review).
        # Caso real: el alias `oroimen-agent-fast` mapea a
        # `[MiniMax-M2.7-highspeed]` en `.env.example` — ese modelo SI esta
        # en el text_chain por default (MiniMax-M3 primario, fallback
        # MiniMax-M2.7-highspeed), pero registrarlos aca blind contra
        # cambios futuros en el override.
        override_models: set[str] = set()
        for alias_chain in settings.llm_model_overrides.values():
            for m in alias_chain:
                override_models.add(m)
        for model in override_models:
            if model not in self._breakers:
                self._breakers[model] = CircuitBreaker(
                    fail_max=settings.circuit_breaker_fail_max,
                    reset_timeout=float(settings.circuit_breaker_reset_timeout),
                    name=model,
                )
        # Sprint 19.6+ Phase 4 (OpenAI Build Week): frontier tier breaker.
        # Registramos el breaker del frontier tier SIEMPRE que el
        # `llm_text_frontier_model` esté configurado (no vacío), para
        # que `self._breakers[model]` no lance KeyError cuando el modelo
        # aparezca en un chain_override. El cliente (httpx) solo se
        # instancia si `llm_text_frontier_enabled=True` — si no, el
        # frontier se omite silenciosamente en runtime (ver `chat()`
        # más abajo).
        if settings.llm_text_frontier_model.strip():
            frontier_model = settings.llm_text_frontier_model
            # Solo registrar si no está ya (puede coincidir con un
            # modelo del text_chain si el operador lo configuró así —
            # caso raro pero legítimo).
            if frontier_model not in self._breakers:
                self._breakers[frontier_model] = CircuitBreaker(
                    fail_max=settings.llm_text_frontier_breaker_fail_max,
                    reset_timeout=float(settings.llm_text_frontier_breaker_reset_timeout_s),
                    name=frontier_model,
                )
        # Sprint 19.6+ Phase 5 (OpenAI Build Week): Ollama local tier.
        # Build a set of model names whose provider hint is "ollama".
        # The set is consulted by `_is_ollama_model` and by `chat()`
        # to dispatch to the OllamaClient instead of the main client.
        # The breaker for each Ollama model is already registered
        # above (in the `for model in settings.text_chain` loop) if
        # the model is in the chain. We don't re-register; the chain
        # loop is the canonical source of truth.
        self._ollama_models: set[str] = set()
        if _is_ollama_provider_hint(getattr(settings, "llm_text_primary_provider", None)):
            self._ollama_models.add(settings.llm_text_primary)
        # Fallback is opt-in to Ollama too (future-proofing). If a
        # user sets LLM_TEXT_FALLBACK_PROVIDER=ollama, the fallback
        # model is dispatched to the OllamaClient.
        if _is_ollama_provider_hint(getattr(settings, "llm_text_fallback_provider", None)):
            self._ollama_models.add(settings.llm_text_fallback)
        # Headers compartidos por TODAS las requests.
        # - `Authorization: Bearer` → requerido por modelos OpenAI-style
        #   (modelos legacy OpenCode Go: deepseek, kimi, glm, mimo, y
        #   MiniMax-M3 via path OpenAI-compatible).
        # - `x-api-key` → requerido por modelos Anthropic-style
        #   (qwen3.*-plus/max legacy). Sin este header, devuelven
        #   401 "Missing API key" aunque la API key sea válida.
        #   Ver https://opencode.ai/docs/es/go/#endpoints (legacy)
        # - `anthropic-version` → opcional, solo relevante para Anthropic-style.
        # Ambos headers son estándar y los modelos del otro estilo los ignoran.
        # Sprint 19.6+ Phase 5: el cliente opencode-go es opcional, igual
        # que el frontier. Si no hay API key, el chain se queda en Ollama
        # local (text_chain gate) y el cliente no se necesita — no se
        # construye. Si hay key, se construye con los headers normales.
        # Patron consistente con `llm_text_frontier_enabled` (Sprint 19.6+
        # Phase 4) y `telegram_bot_token` (Sprint 11 ADR-004): Settings
        # valida la config, el runtime decide si la feature esta activa.
        self._client: httpx.AsyncClient | None = None
        if settings.opencode_go_api_key and settings.opencode_go_api_key.strip():
            self._client = httpx.AsyncClient(
                base_url=settings.opencode_go_base_url,
                headers={
                    "Authorization": f"Bearer {settings.opencode_go_api_key}",
                    "x-api-key": settings.opencode_go_api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                timeout=httpx.Timeout(settings.llm_timeout_seconds),
            )
        # Sprint 19.6+ Phase 4 (OpenAI Build Week): frontier tier client.
        # Opt-in: solo se instancia si `llm_text_frontier_enabled=True`.
        # Si no, `_frontier_client` queda en None y los modelos frontier
        # que aparezcan en un chain se omiten silenciosamente (logged
        # como info, no error — backward compat para `.env` que añadan
        # el modelo a un chain sin haber habilitado el tier).
        self._frontier_client: ChatGpt5_6Client | None = None
        if settings.llm_text_frontier_enabled:
            # El validador de Settings garantiza que la API key está
            # configurada cuando enabled=True. No hace falta check extra.
            self._frontier_client = ChatGpt5_6Client(
                model=settings.llm_text_frontier_model,
                api_key=settings.llm_text_frontier_api_key,
                base_url=settings.llm_text_frontier_base_url,
                timeout_s=settings.llm_text_frontier_timeout_s,
                fail_max=settings.llm_text_frontier_breaker_fail_max,
                reset_timeout_s=settings.llm_text_frontier_breaker_reset_timeout_s,
                name=settings.llm_text_frontier_model,
            )
            logger.info(
                "chatgpt5_6_client_initialized",
                extra={
                    "model": settings.llm_text_frontier_model,
                    "base_url": settings.llm_text_frontier_base_url,
                },
            )
        # Sprint 19.6+ Phase 5 (OpenAI Build Week): Ollama local
        # client. Opt-in via `llm_text_primary_provider == "ollama"`
        # (default for fresh installs). The client is instantiated
        # lazily: only when at least one model in the chain has the
        # Ollama provider hint. Machines that never use Ollama (e.g.,
        # a cloud-only deploy with `LLM_TEXT_PRIMARY_PROVIDER=minimax`)
        # don't pay the import or instantiation cost.
        self._ollama_client: OllamaClient | None = None
        if self._ollama_models:
            # Lazy import: the OllamaClient module is only needed
            # when the chain actually targets Ollama. The import
            # cost (~10ms on a warm interpreter) is acceptable on
            # the cold path and zero on the hot path.
            from hermes.llm.ollama import OllamaClient

            # Use the primary's URL/key for the client. If the
            # primary is Ollama, the URL is `llm_text_primary_base_url`
            # and the key is the literal "ollama" placeholder (Ollama
            # ignores the value). If only the fallback is Ollama, we
            # still use the primary's URL/key (future: per-tier
            # URL/key when more than one Ollama model is in the
            # chain). For Sprint 19.6+ the primary is always the
            # Ollama model when the hint is set.
            self._ollama_client = OllamaClient(
                base_url=settings.llm_text_primary_base_url,
                model=settings.llm_text_primary,
                timeout_s=settings.llm_timeout_seconds,
                fail_max=settings.circuit_breaker_fail_max,
                reset_timeout_s=settings.circuit_breaker_reset_timeout,
                name=settings.llm_text_primary,
            )
            logger.info(
                "ollama_client_initialized",
                extra={
                    "models": sorted(self._ollama_models),
                    "base_url": settings.llm_text_primary_base_url,
                },
            )

    async def aclose(self) -> None:
        # Sprint 19.6+ Phase 5: _client es opcional. Si no hay API
        # key, no se instancia, no hay que cerrarlo.
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        # Cerrar el frontier client si está instanciado. Idempotente.
        if self._frontier_client is not None:
            await self._frontier_client.aclose()
            self._frontier_client = None
        # Cerrar el Ollama client si está instanciado. Idempotente.
        if self._ollama_client is not None:
            await self._ollama_client.aclose()
            self._ollama_client = None

    def _is_ollama_model(self, model: str) -> bool:
        """True si `model` se sirve por Ollama (provider hint == 'ollama').

        Usado por `chat()` para despachar al OllamaClient en vez del
        cliente principal (que habla con el cloud provider).

        Matching: set lookup contra `self._ollama_models`, que se
        construye en `__init__` desde los settings (provider hint).
        El set se construye una sola vez; mutar el provider hint en
        runtime requiere re-instanciar el router (no es un caso
        soportado; los settings son inmutables tras el startup).
        """
        return model in self._ollama_models

    def _is_frontier_model(self, model: str) -> bool:
        """True si `model` es el frontier tier (debe usar el cliente
        dedicado en vez del cliente principal del router).

        Matching: exact match contra `settings.llm_text_frontier_model`.
        Cero ambigüedad: el frontier es UN modelo configurable, no un
        prefijo. Si en el futuro se quieren múltiples frontier models
        (e.g., gpt-5.6 + claude-4.5-opus), convertir a set.
        """
        return model == self.settings.llm_text_frontier_model and bool(
            self.settings.llm_text_frontier_model.strip()
        )

    def get_breaker_states(self) -> dict[str, str]:
        """Retorna {model_name: state} para monitoreo de salud.

        Sprint 6 T53 v3.1: API publica para que el endpoint /health
        del HTTP API inspeccione el estado de los circuit breakers
        sin acceder a self._breakers (atributo privado).

        Returns:
            dict con model_name como key y estado del breaker como
            value. Estados posibles: "closed" (operativo), "open"
            (bloqueado por fallos), "half-open" (probando recuperacion).
        """
        return {name: cb.current_state for name, cb in self._breakers.items()}

    # Sprint 7 T53.2: streaming chat con OpenAI-compatible format.
    # Yields StreamChunk. Si el primer modelo falla, cae al siguiente de
    # la chain (mismo comportamiento que chat()). Si todos fallan, raise
    # LLMError. El caller (agent.run_stream) maneja el fallback.
    #
    # Diferencia con chat(): yields incremental en vez de retorno unico.
    async def chat_stream(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        chain_override: list[str] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        chain = list(chain_override) if chain_override else list(self.settings.text_chain_full)
        temp = temperature if temperature is not None else self.settings.llm_temperature
        last_exc: Exception | None = None
        for model in chain:
            if self._is_frontier_model(model) and self._frontier_client is None:
                logger.info("frontier_skipped_not_enabled_stream", extra={"model": model})
                continue
            if self._is_ollama_model(model) and self._ollama_client is None:
                logger.info("ollama_skipped_not_configured_stream", extra={"model": model})
                continue
            breaker = self._breakers[model]
            if breaker.current_state == "open":
                logger.warning("circuit_open_skip_stream", extra={"model": model})
                continue
            try:
                # yield from en async generator
                async for chunk in self._stream_single_model(model, messages, tools, temp):
                    yield chunk
                return  # exito
            except LLMError as exc:
                logger.warning(
                    "llm_stream_attempt_failed",
                    extra={"model": model, "error": str(exc)},
                )
                last_exc = exc
                continue
        # Todos los modelos fallaron
        raise LLMError(f"All models in chain failed during streaming: {last_exc}")

    @staticmethod
    def _blocking_response_chunks(response: LLMResponse) -> list[StreamChunk]:
        """Adapt a non-streaming provider response to the SSE chunk contract."""
        chunks: list[StreamChunk] = []
        if response.reasoning_content:
            chunks.append(
                StreamChunk(reasoning_content=response.reasoning_content, model=response.model)
            )
        if response.content:
            chunks.append(StreamChunk(content=response.content, model=response.model))
        finish_reason = "tool_calls" if response.tool_calls else "stop"
        chunks.append(
            StreamChunk(
                tool_calls=response.tool_calls or None,
                finish_reason=finish_reason,
                model=response.model,
                usage={
                    "prompt_tokens": response.tokens_in,
                    "completion_tokens": response.tokens_out,
                    "total_tokens": response.tokens_in + response.tokens_out,
                },
            )
        )
        return chunks

    # Sprint 7 T53.2: stream de UN modelo via /chat/completions con stream=True.
    # Lee SSE chunks del upstream opencode-go, los parsea, y yields StreamChunk.
    # Acumula tool_calls (que vienen en pedazos) hasta ver finish_reason="tool_calls".
    async def _stream_single_model(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        # Dedicated Ollama and frontier clients currently expose a blocking
        # chat contract. Adapt their response to functional SSE instead of
        # misrouting them through the opencode-go transport.
        if self._is_frontier_model(model):
            if self._frontier_client is None:
                raise LLMError("frontier client is not enabled")
            response = await self._frontier_client.chat(
                messages,
                temperature=temperature,
                tools=tools,
                max_tokens=self.settings.llm_text_frontier_max_tokens,
            )
            for chunk in self._blocking_response_chunks(response):
                yield chunk
            return
        if self._is_ollama_model(model):
            if self._ollama_client is None:
                raise LLMError("Ollama client is not configured")
            response = await self._ollama_client.chat(
                messages,
                temperature=temperature,
                tools=tools,
            )
            for chunk in self._blocking_response_chunks(response):
                yield chunk
            return

        # Sprint 6 T53.1 v3: force_openai_for_tools se respeta tambien
        # en streaming. Si el modelo esta en EXCEPTIONS y hay tools,
        # forzamos path OpenAI (no path Anthropic /v1/messages).
        # v0.5.7-revert: workaround bug 2013 opencode-go + MiniMax.
        # Sprint 16.8: _ANTHROPIC_FAMILIES_EXCEPTIONS esta vacio (Sprint 16.7),
        # asi que el path "force OpenAI" rara vez aplica. Se mantiene por
        # defensa en profundidad si en el futuro se re-pobla el workaround.
        if force_openai_for_tools(model) and tools:
            async for chunk in self._stream_openai(model, messages, tools, temperature):
                yield chunk
            return
        # Sprint 16.8: Anthropic-format streaming implementado.
        # Parsea SSE de /v1/messages (MiniMax API compatible) y emite
        # StreamChunk con el mismo contrato que _stream_openai.
        # Antes (Sprint 7.1): fallback a chat() blocking. Ahora: stream
        # real con text_delta, thinking_delta, input_json_delta (tool_use).
        if is_anthropic_model(model) and not (force_openai_for_tools(model) and tools):
            async for chunk in self._stream_anthropic(model, messages, tools, temperature):
                yield chunk
            return
        # Default: path OpenAI
        async for chunk in self._stream_openai(model, messages, tools, temperature):
            yield chunk

    async def _stream_anthropic(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream Anthropic-compatible via /v1/messages.

        Sprint 16.8: implementa SSE parsing para el formato Anthropic
        (message_start, content_block_start/delta/stop, message_delta/stop).
        Emite StreamChunk con el mismo contrato que _stream_openai:
        - text_delta -> StreamChunk(content=...)
        - thinking_delta -> StreamChunk(reasoning_content=...)
        - tool_use block completo -> StreamChunk(tool_calls=[...])
        - message_stop -> StreamChunk(finish_reason=..., usage=...)
        """
        # Sprint 19.6+ Phase 5: el cliente opencode-go es opcional.
        # Si llegamos aquí sin cliente, es porque la chain incluyó un
        # modelo opencode-go sin API key — bug en el dispatch. Fallar
        # con error claro en vez de AttributeError.
        if self._client is None:
            raise LLMError("opencode-go client not configured: set OPENCODE_GO_API_KEY in .env")
        # Transformar messages al formato Anthropic (system separado, etc.)
        system_text, user_assistant = _split_system_from_messages(messages)
        anthropic_messages = _transform_to_anthropic_messages(user_assistant)
        payload: dict = {
            "model": model,
            "max_tokens": self.settings.llm_max_tokens,
            "temperature": temperature,
            "top_p": self.settings.llm_top_p,
            "stream": True,
        }
        if system_text:
            payload["system"] = system_text
        if tools:
            # OpenAI format -> Anthropic format (igual que _invoke_anthropic)
            payload["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
                if isinstance(t, dict) and "function" in t
            ]
        payload["messages"] = anthropic_messages

        # Estado del stream
        # Mapa de index -> tipo de content_block actual (text|thinking|tool_use)
        current_block_type: dict[int, str] = {}
        # Mapa de index -> metadata de tool_use (id, name) durante su construccion
        tool_use_meta: dict[int, dict] = {}
        # Acumulador de input JSON por tool_use index (viene fragmentado)
        tool_use_input_accum: dict[int, str] = {}
        # Stop reason + usage finales (de message_delta)
        final_stop_reason: str | None = None
        final_usage: dict | None = None

        try:
            async with self._client.stream("POST", "/messages", json=payload) as response:
                response.raise_for_status()
                current_event: str | None = None
                async for line in response.aiter_lines():
                    # Anthropic SSE format:
                    #   event: <type>
                    #   data: <json>
                    # Linea en blanco = fin de evento
                    line = line.strip()
                    if not line:
                        current_event = None
                        continue
                    if line.startswith("event: "):
                        current_event = line[len("event: ") :].strip()
                        continue
                    if not line.startswith("data: "):
                        # Comentario SSE o heartbeat, ignorar
                        continue
                    payload_str = line[len("data: ") :]
                    try:
                        data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        # Linea malformada, skip
                        continue
                    event_type = data.get("type") or current_event
                    if event_type == "message_start":
                        # Init message metadata. Capturar input_tokens.
                        # Sprint 16.8.1 (B-2 fix): translate Anthropic keys
                        # (input_tokens) to OpenAI keys (prompt_tokens)
                        # porque AgentLoop.run_stream (loop.py:432-434)
                        # lee los keys OpenAI.
                        msg = data.get("message", {})
                        usage_in = msg.get("usage", {}).get("input_tokens", 0)
                        if usage_in:
                            final_usage = {"prompt_tokens": usage_in, "completion_tokens": 0}
                    elif event_type == "content_block_start":
                        block = data.get("content_block", {})
                        idx = data.get("index")
                        btype = block.get("type")
                        current_block_type[idx] = btype
                        if btype == "tool_use":
                            tool_use_meta[idx] = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                            }
                            tool_use_input_accum[idx] = ""
                    elif event_type == "content_block_delta":
                        idx = data.get("index")
                        delta = data.get("delta", {})
                        dtype = delta.get("type")
                        if dtype == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield StreamChunk(content=text, model=model)
                        elif dtype == "thinking_delta":
                            thinking = delta.get("thinking", "")
                            if thinking:
                                yield StreamChunk(reasoning_content=thinking, model=model)
                        elif dtype == "input_json_delta":
                            # tool_use input viene fragmentado; acumular
                            partial = delta.get("partial_json", "")
                            # Sprint 16.8.1 (m-2 fix): warning si llega un
                            # input_json_delta sin content_block_start matching
                            # (upstream hiccup o proxy truncado).
                            if idx is not None:
                                if idx in tool_use_input_accum:
                                    tool_use_input_accum[idx] += partial
                                else:
                                    logger.warning(
                                        "anthropic_stream_orphan_tool_use_delta",
                                        extra={"model": model, "block_index": idx},
                                    )
                    elif event_type == "content_block_stop":
                        idx = data.get("index")
                        btype = current_block_type.get(idx)
                        if btype == "tool_use" and idx in tool_use_meta:
                            # Cerrar tool_use: parsear accumulated input JSON
                            raw_json = tool_use_input_accum.get(idx, "{}")
                            try:
                                args = json.loads(raw_json) if raw_json.strip() else {}
                            except json.JSONDecodeError:
                                # JSON malformado -> log warning, vaciar args
                                logger.warning(
                                    "anthropic_stream_malformed_tool_args",
                                    extra={
                                        "model": model,
                                        "tool_name": tool_use_meta[idx].get("name"),
                                        "raw_len": len(raw_json),
                                    },
                                )
                                args = {}
                            meta = tool_use_meta[idx]
                            tc = ToolCall(
                                id=meta["id"],
                                name=meta["name"],
                                arguments=args,
                            )
                            yield StreamChunk(tool_calls=[tc], model=model)
                        # Sprint 16.8.1 (m-3 fix): liberar memoria del block
                        # cerrado para evitar crecimiento lineal si el
                        # response emite muchos tool_use blocks.
                        current_block_type.pop(idx, None)
                        tool_use_meta.pop(idx, None)
                        tool_use_input_accum.pop(idx, None)
                    elif event_type == "message_delta":
                        # Stop reason + final output tokens
                        # Sprint 16.8.1 (B-2 fix): translate output_tokens ->
                        # completion_tokens para que AgentLoop.run_stream
                        # (loop.py:432-434) lo lea correctamente.
                        delta = data.get("delta", {})
                        stop = delta.get("stop_reason")
                        if stop:
                            final_stop_reason = stop
                        usage_out = data.get("usage", {}).get("output_tokens")
                        if usage_out is not None:
                            if final_usage is None:
                                final_usage = {"prompt_tokens": 0, "completion_tokens": 0}
                            final_usage["completion_tokens"] = usage_out
                    elif event_type == "message_stop":
                        # Fin del stream. Emitir chunk final con finish_reason.
                        # Sprint 16.8.1 (M-1 fix): mapping completo Anthropic
                        # -> OpenAI + fallback defensivo a "stop" con warning
                        # para stop_reasons desconocidos.
                        fr = final_stop_reason or "stop"
                        if fr == "end_turn":
                            fr = "stop"
                        elif fr == "tool_use":
                            fr = "tool_calls"
                        elif fr == "max_tokens":
                            fr = "length"
                        elif fr == "stop_sequence":
                            fr = "stop"
                        else:
                            # Fallback defensivo: cualquier stop_reason
                            # desconocido -> "stop" para que AgentLoop no se
                            # cuelgue. Log warning para investigación.
                            logger.warning(
                                "anthropic_unknown_stop_reason",
                                extra={"stop_reason": fr, "model": model},
                            )
                            fr = "stop"
                        yield StreamChunk(
                            finish_reason=fr,
                            model=model,
                            usage=final_usage,
                            truncated=(fr == "length"),
                        )
                        break
                    elif event_type == "ping":
                        # Heartbeat, ignorar
                        continue
                    elif event_type == "error":
                        err = data.get("error", {})
                        # Sprint 16.8.1 (B-1 fix): NO usar extra={"message": ...}
                        # porque 'message' es campo reservado de LogRecord
                        # en Python 3.14+ y crashea con KeyError. Ver convención
                        # documentada en hermes/agent/loop.py:146-147.
                        logger.error(
                            "anthropic_stream_error_event",
                            extra={
                                "model": model,
                                "error_type": err.get("type"),
                                "error_message": err.get("message"),
                            },
                        )
                        raise LLMError(
                            f"Anthropic stream error: {err.get('type')}: {err.get('message')}"
                        )
                else:
                    # Sprint 16.8.1 (M-2 fix): el loop termino sin message_stop
                    # (upstream cierra stream abruptamente despues de
                    # message_delta). Emitir chunk final con lo acumulado
                    # para que AgentLoop reciba usage + finish_reason.
                    if final_stop_reason is None:
                        logger.warning(
                            "anthropic_stream_truncated",
                            extra={"model": model},
                        )
                    fr = final_stop_reason or "stop"
                    if fr == "end_turn":
                        fr = "stop"
                    elif fr == "tool_use":
                        fr = "tool_calls"
                    elif fr == "max_tokens":
                        fr = "length"
                    elif fr == "stop_sequence":
                        fr = "stop"
                    else:
                        logger.warning(
                            "anthropic_unknown_stop_reason",
                            extra={"stop_reason": fr, "model": model},
                        )
                        fr = "stop"
                    yield StreamChunk(
                        finish_reason=fr,
                        model=model,
                        usage=final_usage,
                        truncated=(fr == "length"),
                    )
        except httpx.HTTPError as exc:
            logger.error(
                "llm_stream_http_error",
                extra={
                    "model": model,
                    "provider": "anthropic",
                    "status": getattr(getattr(exc, "response", None), "status_code", None),
                },
            )
            raise LLMError(f"HTTP error during stream: {exc}") from exc

    async def _stream_openai(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream OpenAI-compatible. Parsea SSE y yields StreamChunk."""
        # Sprint 19.6+ Phase 5: ver guard en _stream_anthropic.
        if self._client is None:
            raise LLMError("opencode-go client not configured: set OPENCODE_GO_API_KEY in .env")
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            # Sprint 9.3.2: pedirle al upstream que incluya usage en el
            # chunk final. OpenAI/OpenRouter soportan esto via stream_options.
            # Sin esto, iter_tokens_in/out siempre quedan en 0.
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        # Acumuladores
        accumulated_tool_deltas: list[dict] = []
        final_finish_reason: str | None = None
        final_usage: dict | None = None

        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload_str = line[6:]
                    if payload_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        # Linea SSE malformada. Skip y continuar.
                        continue
                    # Sprint 7 T53.2 fase 4 fix: si upstream devuelve
                    # una respuesta sin 'choices' (error mid-stream,
                    # e.g. upstream 5xx envuelto en SSE), no crashear
                    # con IndexError. Log warning y skip.
                    # Sprint 9.3.2: capturar usage metadata (tokens) en
                    # CUALQUIER chunk. OpenAI puede mandarlo con choices=[]
                    # (chunk final) O con choices=[{...finish_reason...}].
                    usage_data = data.get("usage")
                    if usage_data:
                        final_usage = usage_data
                        # Continuar procesando choices si las hay
                        if not data.get("choices"):
                            continue
                    choices = data.get("choices")
                    if not choices:
                        # Podria ser un chunk de error estilo
                        # {"error": {"message": "..."}}. Log y skip.
                        if "error" in data:
                            logger.warning(
                                "upstream_sse_error_chunk",
                                extra={"error": str(data.get("error"))},
                            )
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    # Reasoning content (DeepSeek thinking mode)
                    rc = delta.get("reasoning_content")
                    if rc:
                        yield StreamChunk(reasoning_content=rc, model=model)
                        continue
                    # Content (prosa del LLM)
                    content = delta.get("content")
                    if content:
                        yield StreamChunk(content=content, model=model)
                        continue
                    # Tool calls: acumular deltas, emitir en chunk final
                    tc_deltas = delta.get("tool_calls")
                    if tc_deltas:
                        accumulated_tool_deltas.extend(tc_deltas)
                        continue
                    # Finish reason (a veces viene en delta, otras en choice)
                    finish = choice.get("finish_reason") or delta.get("finish_reason")
                    if finish:
                        final_finish_reason = finish
                        # No yield aqui todavia: puede que falten tool_calls
                        continue
            # Stream terminado (llegamos a [DONE] o response.aiter_lines acabo)
            # Emitir tool_calls acumulados si los hay
            if accumulated_tool_deltas:
                merged = _merge_tool_call_deltas(accumulated_tool_deltas)
                yield StreamChunk(
                    tool_calls=merged,
                    finish_reason=final_finish_reason or "tool_calls",
                    model=model,
                    usage=final_usage,
                    truncated=(final_finish_reason == "length"),
                )
            elif final_finish_reason:
                yield StreamChunk(
                    finish_reason=final_finish_reason,
                    model=model,
                    usage=final_usage,
                    truncated=(final_finish_reason == "length"),
                )
        except httpx.HTTPError as exc:
            logger.error(
                "llm_stream_http_error",
                extra={
                    "model": model,
                    "status": getattr(getattr(exc, "response", None), "status_code", None),
                },
            )
            raise LLMError(f"HTTP error during stream: {exc}") from exc

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        is_voice: bool = False,
        tools: list[dict] | None = None,
        chain_override: list[str] | None = None,
    ) -> LLMResponse:
        """Invoca el LLM con smart routing unificado (v1.2).

        El parámetro `is_voice` se mantiene por compatibilidad con código
        existente, pero en v1.2 texto y voz usan el mismo chain
        (`settings.text_chain` == `settings.voice_chain`). Su valor se
        ignora en el routing.

        Si temperature es None, usa settings.llm_temperature (default 0.3,
        conservador para uso factual de asistencia).

        Si `tools` se proporciona, se envía al LLM como JSON Schema. El LLM
        puede responder con `tool_calls` (OpenAI) o `tool_use` blocks
        (Anthropic). Estos se extraen en `LLMResponse.tool_calls`.

        Si `chain_override` se proporciona, sustituye a `text_chain` para
        esta llamada. Sprint 12 (ADR-007): los model aliases de
        `/v1/models` (e.g. `oroimen-agent-fast`) se mapean a chains
        dedicadas via `Settings.model_overrides`, y el handler de chat
        las pasa aqui.

        Sprint 19.6+ Phase 4 (OpenAI Build Week): si un modelo en la
        chain es el frontier tier (`settings.llm_text_frontier_model`)
        pero el frontier no está habilitado (no hay API key configurada),
        se omite silenciosamente con un log de info. Esto permite que
        un `.env` con el modelo en un chain_override funcione sin
        requerir que el frontier esté configurado (el chain cae al
        siguiente modelo o falla con LLMError si era el último).
        """
        if temperature is None:
            temperature = self.settings.llm_temperature
        chain = list(chain_override) if chain_override else list(self.settings.text_chain_full)
        last_err: Exception | None = None
        for model in chain:
            # Sprint 19.6+ Phase 4 (OpenAI Build Week): opt-in frontier.
            # Si el modelo es el frontier tier pero el cliente no está
            # instanciado (enabled=False), lo saltamos como si el breaker
            # estuviera abierto. Log info, no warning, porque es el
            # comportamiento esperado cuando el operador no ha habilitado
            # el tier.
            if self._is_frontier_model(model) and self._frontier_client is None:
                logger.info(
                    "frontier_skipped_not_enabled",
                    extra={"model": model},
                )
                continue
            # Sprint 19.6+ Phase 5 (OpenAI Build Week): Ollama local tier.
            # Si el modelo es Ollama pero el cliente no está instanciado
            # (provider hint no es "ollama" en settings), lo saltamos
            # como info. Esto permite que un .env que tenga un modelo
            # tipo "qwen2.5:7b" en un chain_override funcione sin
            # requerir que Ollama esté configurado (el chain cae al
            # siguiente modelo o falla con LLMError si era el último).
            if self._is_ollama_model(model) and self._ollama_client is None:
                logger.info(
                    "ollama_skipped_not_configured",
                    extra={"model": model},
                )
                continue
            breaker = self._breakers[model]
            if breaker.current_state == "open":
                logger.warning("circuit_open_skip", extra={"breaker": model})
                continue
            try:
                return await self._call_with_breaker(breaker, model, messages, temperature, tools)
            except (CircuitOpenError, LLMError) as exc:
                last_err = exc
                logger.warning(
                    "llm_attempt_failed",
                    extra={"breaker": model, "error": str(exc)},
                )
                continue
        raise LLMError(f"All models in chain failed: {last_err}")

    async def _call_with_breaker(
        self,
        breaker: CircuitBreaker,
        model: str,
        messages: list[dict],
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        max_attempts = self.settings.llm_max_retries + 1
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            # Sprint 19.6+ Phase 4 (OpenAI Build Week): el frontier
            # tiene su propio breaker (mas agresivo) gestionado dentro
            # del ChatGpt5_6Client. Para el frontier, hacemos una sola
            # pasada (sin retry aqui) — el retry externo añadiria un
            # segundo loop encima del breaker interno, lo que es
            # redundante para un tier de ultimo recurso. El
            # circuit_breaker_fail_max=3 del frontier ya provee la
            # proteccion contra fallos transitorios. Si en el futuro
            # se quiere retry + breaker, refactorizar para que el
            # frontier tambien use el patron breaker.call.
            if self._is_frontier_model(model) and self._frontier_client is not None:
                return await self._frontier_client.chat(
                    messages,
                    temperature=temperature,
                    tools=tools,
                    max_tokens=self.settings.llm_text_frontier_max_tokens,
                )
            # Sprint 19.6+ Phase 5 (OpenAI Build Week): el Ollama tier
            # tiene su propio cliente (httpx + breaker) con retry
            # gestionado internamente. Mismo patron que el frontier:
            # una sola pasada, el breaker's fail_max (5) cubre los
            # transitorios. Sin retry externo encima del breaker.
            if self._is_ollama_model(model) and self._ollama_client is not None:
                return await self._ollama_client.chat(
                    messages,
                    temperature=temperature,
                    tools=tools,
                )
            coro = self._invoke(model, messages, temperature, tools)
            try:
                return await breaker.call(_await(coro))
            except (CircuitOpenError, LLMError) as exc:
                last_err = exc
                if attempt < max_attempts:
                    # Backoff exponencial: base * 2^(attempt-1), cap a 4s.
                    # Base configurable via settings.llm_retry_backoff_base
                    # (default 0.5s en prod, 0.001s en tests via conftest).
                    backoff = min(
                        self.settings.llm_retry_backoff_base * (2 ** (attempt - 1)),
                        4.0,
                    )
                    logger.debug(
                        "llm_retry",
                        extra={"breaker": model, "attempt": attempt, "backoff_s": backoff},
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise
        raise LLMError(f"Exhausted retries: {last_err}")

    async def _invoke(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        # Sprint 19.6+ Phase 4 (OpenAI Build Week): el frontier usa su
        # propio cliente (httpx + breaker dedicados). Esta rama solo
        # se alcanza si `_call_with_breaker` la invoca, y ese metodo ya
        # despacha el frontier por su cuenta. Dejamos el guard aqui
        # como defense-in-depth por si alguien llama `_invoke` directo
        # en un test.
        if self._is_frontier_model(model) and self._frontier_client is not None:
            return await self._frontier_client.chat(
                messages,
                temperature=temperature,
                tools=tools,
                max_tokens=self.settings.llm_text_frontier_max_tokens,
            )
        # Sprint 19.6+ Phase 5 (OpenAI Build Week): el Ollama tier
        # tiene su propio cliente (httpx + breaker) y se despacha
        # desde `_call_with_breaker` antes de llegar aquí. Este guard
        # es defense-in-depth: si alguien llama `_invoke` directo en
        # un test con un modelo Ollama, despachamos al OllamaClient
        # en vez de caer al path OpenAI principal (que hablaría con
        # la cloud API, no con Ollama).
        if self._is_ollama_model(model) and self._ollama_client is not None:
            return await self._ollama_client.chat(
                messages,
                temperature=temperature,
                tools=tools,
            )
        # v0.5.7-revert: workaround para bug 2013 en opencode-go + MiniMax.
        # Si el modelo esta en EXCEPTIONS Y hay tools en la peticion,
        # forzamos path OpenAI. Sin tools, el path Anthropic normal
        # funciona bien (el bug 2013 solo se manifiesta con tool_use/
        # tool_result en el payload). Esto minimiza el alcance del
        # workaround y mantiene compatibilidad con todos los tests
        # existentes que no usan tools.
        if force_openai_for_tools(model) and tools:
            return await self._invoke_openai(model, messages, temperature, tools)
        if is_anthropic_model(model):
            return await self._invoke_anthropic(model, messages, temperature, tools)
        return await self._invoke_openai(model, messages, temperature, tools)

    async def _invoke_openai(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        # Sprint 19.6+ Phase 5: ver guard en _stream_anthropic.
        if self._client is None:
            raise LLMError("opencode-go client not configured: set OPENCODE_GO_API_KEY in .env")
        start = time.perf_counter()
        # OpenAI requiere tool_call_id en mensajes role="tool". El
        # AgentLoop guarda este campo en messages.tool_call_id.
        # Si no está (mensajes de test o legacy), usamos string vacío
        # para evitar que el provider rechace el payload.
        openai_messages = [
            {
                **m,
                **({"tool_call_id": m.get("tool_call_id", "")} if m.get("role") == "tool" else {}),
            }
            for m in messages
        ]
        payload: dict = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "top_p": self.settings.llm_top_p,
            # repetition_penalty no es OpenAI-standard pero MiniMax-M3
            # OpenAI-compat lo acepta como "additional parameter".
            # Skip en paths Anthropic-compat (Anthropic spec rechaza
            # campos no estándar con 400).
            "repetition_penalty": self.settings.llm_repetition_penalty,
        }
        if tools:
            payload["tools"] = tools
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # Capturar body del error 400/500 para diagnóstico
            body = ""
            with contextlib.suppress(Exception):
                body = exc.response.text[:500] if hasattr(exc, "response") else ""
            logger.error(
                "llm_http_error",
                extra={
                    "model": model,
                    "provider": "openai",
                    "status": getattr(getattr(exc, "response", None), "status_code", None),
                    "body": body,
                    "url": str(getattr(exc, "request", None) and exc.request.url),
                },
            )
            raise LLMError(f"HTTP error: {exc}") from exc
        data = _safe_json_parse(resp, provider="openai")
        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            # Sprint 5 T51: capturar reasoning_content del response.
            # Providers OpenAI-compatibles exponen thinking mode via este
            # campo (MiniMax-M3 lo popula cuando temperature >= 1.0 y
            # el modelo decide pensar). Si la respuesta no lo incluye
            # (o viene null), `or ""` lo normaliza a cadena vacía para
            # que el caller no tenga que distinguir entre None y "".
            reasoning = message.get("reasoning_content") or ""
            usage = data.get("usage", {})
            tool_calls = _parse_openai_tool_calls(message.get("tool_calls"))
            return LLMResponse(
                content=content,
                model=model,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                tool_calls=tool_calls,
                reasoning_content=reasoning,
            )
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            logger.warning(
                "llm_malformed_openai_response",
                extra={
                    "model": model,
                    "error": str(exc),
                    "raw_snippet": str(data)[:_RAW_SNIPPET_MAX],
                },
            )
            raise LLMError(f"Malformed OpenAI response: {exc}") from exc

    async def _invoke_anthropic(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        # Sprint 19.6+ Phase 5: ver guard en _stream_anthropic.
        if self._client is None:
            raise LLMError("opencode-go client not configured: set OPENCODE_GO_API_KEY in .env")
        start = time.perf_counter()
        system_text, user_assistant = _split_system_from_messages(messages)
        # Sprint 5 T51: la transformacion a formato Anthropic se hace en
        # el helper module-level _transform_to_anthropic_messages (3.7)
        # para testabilidad directa y defense in depth contra campos
        # OpenAI-specific (reasoning_content, tool_call_id en assistants
        # sin tools, refusal, audio, etc.) que podrian estar en DB por
        # turnos previos con modelos OpenAI-compatible.
        anthropic_messages = _transform_to_anthropic_messages(user_assistant)
        payload: dict = {
            "model": model,
            "max_tokens": self.settings.llm_max_tokens,
            "temperature": temperature,
            "top_p": self.settings.llm_top_p,
        }
        if system_text:
            payload["system"] = system_text
        if tools:
            # Transformar tools de formato OpenAI ({type: function, function: {...}})
            # a formato Anthropic ({name, description, input_schema}).
            # El provider Minimax (vía opencode-go) interpreta /v1/messages
            # como Anthropic y rechaza el formato OpenAI con error
            # "function name or parameters is empty (2013)".
            payload["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
                if isinstance(t, dict) and "function" in t
            ]
        payload["messages"] = anthropic_messages
        try:
            resp = await self._client.post("/messages", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # Capturar body del error 400/500 para diagnóstico
            body = ""
            with contextlib.suppress(Exception):
                body = exc.response.text[:500] if hasattr(exc, "response") else ""
            logger.error(
                "llm_http_error",
                extra={
                    "model": model,
                    "provider": "anthropic",
                    "status": getattr(getattr(exc, "response", None), "status_code", None),
                    "body": body,
                    "url": str(getattr(exc, "request", None) and exc.request.url),
                },
            )
            raise LLMError(f"HTTP error: {exc}") from exc
        data = _safe_json_parse(resp, provider="anthropic")
        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            content_blocks = data.get("content", [])
            content = "".join(
                block.get("text", "") for block in content_blocks if block.get("type") == "text"
            )
            tool_calls = _parse_anthropic_tool_use(content_blocks)
            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                model=model,
                tokens_in=usage.get("input_tokens", 0),
                tokens_out=usage.get("output_tokens", 0),
                latency_ms=latency_ms,
                tool_calls=tool_calls,
            )
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            logger.warning(
                "llm_malformed_anthropic_response",
                extra={
                    "model": model,
                    "error": str(exc),
                    "raw_snippet": str(data)[:_RAW_SNIPPET_MAX],
                },
            )
            raise LLMError(f"Malformed Anthropic response: {exc}") from exc

    def breaker_state(self, model: str) -> str:
        """Estado del circuit breaker para `model`.

        Si el modelo no está inicializado (p. ej. no está en el chain
        actual), devuelve 'closed' por seguridad. Esto evita que la
        telemetría crashee si alguien pasa un modelo que ya no usamos
        (modelos legacy OpenCode Go como mimo/kimi/glm/deepseek tras
        la migración Sprint 12+ a MiniMax API).
        """
        breaker = self._breakers.get(model)
        if breaker is None:
            return "closed"
        return breaker.current_state


def _split_system_from_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Extrae el system prompt (Anthropic lo quiere separado) y devuelve user/assistant."""
    parts: list[str] = []
    rest: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            parts.append(m.get("content", ""))
        else:
            rest.append(m)
    return "\n\n".join(p for p in parts if p), rest


def _parse_openai_tool_calls(raw: list[dict] | None) -> list[ToolCall]:
    """Parsea tool_calls del formato OpenAI Chat Completions.

    Formato esperado:
    ```json
    [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Madrid\"}"  // string JSON
        }
      }
    ]
    ```
    Si los arguments no son JSON válido, devuelve ToolCall con arguments={}
    y loggea warning (no rompe el flujo).
    """
    if not raw:
        return []
    result: list[ToolCall] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        tc_id = str(tc.get("id", ""))
        fn = tc.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name", ""))
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, dict):
            args = args_raw
        else:
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "llm_invalid_tool_arguments",
                    extra={
                        "tool_call_id": tc_id,
                        "name": name,
                        "raw": str(args_raw)[:_RAW_SNIPPET_MAX],
                    },
                )
                args = {}
        result.append(ToolCall(id=tc_id, name=name, arguments=args))
    return result


def _parse_anthropic_tool_use(content_blocks: list[dict]) -> list[ToolCall]:
    """Parsea bloques `tool_use` del formato Anthropic Messages.

    Formato esperado:
    ```json
    [
      {"type": "text", "text": "..."},
      {
        "type": "tool_use",
        "id": "toolu_abc123",
        "name": "get_weather",
        "input": {"city": "Madrid"}
      }
    ]
    ```
    """
    if not content_blocks:
        return []
    result: list[ToolCall] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        result.append(
            ToolCall(
                id=str(block.get("id", "")),
                name=str(block.get("name", "")),
                arguments=block.get("input", {}) or {},
            )
        )
    return result
