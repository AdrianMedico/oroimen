"""Middleware HTTP compatible con OpenAI para Oroimen (Sprint 6 T53 fases 1-3).

v3.1 fixes aplicados (vs Gemini v3):
- create_app() factory (no `app` module-level) para testabilidad
- CORSMiddleware añadido (Open WebUI en navegador)
- Exception handler especifico para LLMError (no captura RequestValidationError)
- stream=true -> 501 (no 400) [fase 1]; SSE -> StreamingResponse [fase 3]
- System message del cliente se ignora (no duplica el de Oroimen)
- AgentLoop recibe settings explicitos (no defaults)
- new_conversation(chat_id=0, user_id=0, thread_id=0)
- Conversacion se archiva tras response (is_archived=1, evita DB growth)
- get_last_assistant_message() para usage en response

Sprint 7 T53 fase 3:
- Streaming SSE via StreamingResponse con `data: {...}\\n\\n` + `data: [DONE]`
- Vision passthrough (ContentPart) — se pasa al LLM, no se persiste
- /v1/files endpoint con pypdf (extraccion server-side de PDF)
- Files persistidos en DB (Sprint 9.0+). Sprint 15 eliminó el cache
  in-memory; cada read va directo a `db.get_file(file_id)`.

Sprint 8.7 v1.3:
- /v1/chat/completions inyecta extracted_text del file cuando
  el request incluye referencias a archivos subidos via /v1/files.
- Formato Open WebUI nativo: `message.files[]` con `{"type": "file", "id": "..."}`.
- Compatibilidad OpenAI Assistants v2: `content[].file_id` (defense in depth).
- Vision path: reconstruye la vision list desde cero (solo image_url + UN
  text part con enriched_text) para evitar duplicacion de la pregunta.
- Limite: settings.read_tool_max_chars (150K chars, mismo que tools).
- 404 explicito si file_id no existe en la DB.
- Ignora type='collection' (fuera de scope, manejo futuro).

Sprint 15 (US-3.1):
- POST /v1/files: dedup transparente por SHA256 del texto extraído.
  Devuelve HTTP 200 + `deduplicated=true` si el content_hash ya
  existe; HTTP 201 + `deduplicated=false` si es nuevo.
- GET /v1/files/{id}: lee directamente de la DB (sin cache).
- file_refs huérfanas (file borrado) ahora inyectan un marcador
  semántico literal via AgentLoop._resolve_file_refs en vez de
  omitirse silenciosamente (Gemini §8.3).

Endpoints:
- GET /v1/models (requerido por Open WebUI al arrancar)
- POST /v1/chat/completions (compatible con OpenAI, streaming + vision)
- POST /v1/files (multipart upload, pypdf para PDFs)
- GET /v1/files/{file_id} (recupera extracted_text)
- GET /health (activo: ping DB + check circuit breakers)

Coexistencia con Telegram:
- enable_http_api=False: solo Telegram, HealthServer en puerto 8000
- enable_http_api=True: Telegram + HTTP API, FastAPI expone /health en 8000
  (HealthServer desactivado para evitar conflicto de puerto)
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json as _json
import logging
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from hermes.agent.loop import LLM_ERROR_FALLBACK_MESSAGE, AgentLoop
from hermes.config import Settings
from hermes.jobs.preflight import DeepResearchCapabilities
from hermes.llm.router import LLMError
from hermes.memory.db import Database
from hermes.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

PUBLIC_MODEL_ID = "oroimen-agent"
PUBLIC_FRONTIER_MODEL_ID = "oroimen-agent-frontier"
LEGACY_MODEL_ID = "hermes-agent"
LEGACY_FRONTIER_MODEL_ID = "hermes-agent-frontier"


def _canonical_model_alias(model: str) -> str:
    """Return the public Oroimen spelling for a legacy built-in alias."""
    if model == LEGACY_MODEL_ID or model.startswith(f"{LEGACY_MODEL_ID}-"):
        return f"{PUBLIC_MODEL_ID}{model[len(LEGACY_MODEL_ID) :]}"
    return model


def _model_override(
    model: str,
    overrides: dict[str, list[str]],
) -> list[str] | None:
    """Resolve both spellings, with the canonical Oroimen key taking precedence."""
    canonical = _canonical_model_alias(model)
    canonical_override = overrides.get(canonical)
    if canonical_override is not None:
        return canonical_override

    # Defense in depth for Settings created via model_copy(update=...), which
    # bypasses field validation and may still contain a legacy key.
    if canonical == PUBLIC_MODEL_ID or canonical.startswith(f"{PUBLIC_MODEL_ID}-"):
        legacy = f"{LEGACY_MODEL_ID}{canonical[len(PUBLIC_MODEL_ID) :]}"
        return overrides.get(legacy)
    return overrides.get(model)


# --- Schemas Pydantic (OpenAI-compatible) ---


class ContentPart(BaseModel):
    """Una pieza de un mensaje multimodal (OpenAI vision format).

    Formato: {"type": "text", "text": "..."} o
    {"type": "image_url", "image_url": {"url": "data:image/..."}}.
    Solo soportamos estos dos tipos. Audio/video en sprint 9+.

    S8.7: el formato {"type": "file", "file_id": "..."} en content NO se
    usa en produccion (Open WebUI nativo usa message.files[]). Se acepta
    por defense in depth si un cliente OpenAI Assistants v2 lo envia.
    """

    type: str
    text: str | None = None
    image_url: dict[str, Any] | None = None
    # S8.7 v1.3: file reference (OpenAI Assistants v2 format, defense in
    # depth). NO es el formato nativo Open WebUI (que usa message.files).
    # Si type='file' y file_id presente, el texto del archivo se inyecta
    # en el user message. Ver _extract_file_ids_from_message.
    file_id: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    # Sprint 7.3: aceptar str (legacy) o list[ContentPart] (vision).
    # Si llega list, se extrae texto para DB y la lista va al LLM.
    content: str | list[ContentPart]
    # S8.7 v1.3: Open WebUI nativo envia files a nivel de message
    # (formato custom), NO en content. Formato: [{"type": "file", "id": "..."}].
    # Tambien soporta "collection" (fuera de scope S8.7, se ignora).
    files: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False
    user: str | None = None
    # Sprint 12 (ADR-007): metadata opcional para que el cliente nativo
    # RikkaHub indique que la conversacion es persistente (no efimera).
    # Si metadata.chat_id esta presente, Oroimen reusa o crea una conv
    # con ese chat_id (no se archiva al terminar). Si no esta, se
    # comporta como antes (efimera HTTP, archivada al final).
    metadata: dict[str, Any] | None = None


# --- Sprint 19: Vault Collections API models --------------------------------


class CreateCollectionRequest(BaseModel):
    """POST /v1/collections request body."""

    name: str = Field(min_length=1, max_length=200)
    parent_collection_id: str | None = None
    description: str | None = None
    sort_order: int = 0


class CollectionResponse(BaseModel):
    """Standard collection response shape."""

    collection_id: str
    name: str
    parent_collection_id: str | None
    description: str | None
    sort_order: int
    archived: int
    archived_at: str | None
    created_at: str


class ListCollectionsResponse(BaseModel):
    """GET /v1/collections response."""

    collections: list[CollectionResponse]


class PatchCollectionRequest(BaseModel):
    """PATCH /v1/collections/{id} body. All fields optional."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    parent_collection_id: str | None = None
    description: str | None = None
    sort_order: int | None = None


class AddFileToCollectionRequest(BaseModel):
    """POST /v1/collections/{id}/files body."""

    file_id: str = Field(min_length=32, max_length=32)


class FileInCollectionResponse(BaseModel):
    """File metadata for list_files_in_collection."""

    file_id: str
    path: str  # always posix (Sprint 19 §7)
    size_bytes: int
    text_version: str
    modified_at: str
    added_at: str


class ListFilesInCollectionResponse(BaseModel):
    """GET /v1/collections/{id}/files response."""

    files: list[FileInCollectionResponse]
    next_cursor: str | None = None


# --- Helpers de content extraction ---


def _extract_text_from_content(content: str | list[ContentPart]) -> str:
    """Extrae el texto de un content OpenAI (str o list[ContentPart]).

    Para DB storage: si es list, concatena las partes type='text'.
    Vision parts (image_url) se ignoran para la DB (no persistimos
    base64). Si no hay texto, retorna placeholder.
    """
    if isinstance(content, str):
        return content
    text_parts: list[str] = []
    for part in content:
        if part.type == "text" and part.text:
            text_parts.append(part.text)
    if not text_parts:
        return "[user sent a multimodal message]"
    return "\n".join(text_parts)


def _content_to_llm_parts(
    content: str | list[ContentPart],
) -> str | list[dict[str, Any]] | None:
    """Convierte content a formato que espera el LLM.

    Retorna:
    - str: si content es texto plano (no vision)
    - list[dict]: si content es list[ContentPart] (vision)
    - None: si input invalido (no deberia pasar, defense in depth)

    OpenAI vision format: [{"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {"url": "data:..."}}]
    """
    if isinstance(content, str):
        return content
    if not content:
        return None
    return [part.model_dump(exclude_none=True) for part in content]


def _extract_file_ids_from_message(msg: ChatMessage) -> list[str]:
    """S8.7 v1.3: extrae file_ids de un message (defense in depth).

    Busca en DOS ubicaciones:
    1. msg.content (si es list): ContentPart type='file' con file_id
       (formato OpenAI Assistants v2, compatibilidad legacy)
    2. msg.files: items con type='file' y 'id'
       (formato Open WebUI nativo, confirmado en docs.openwebui.com)

    Ignora type='collection' (fuera de scope S8.7, manejo futuro).
    Dedup preservando orden de primera aparicion.

    Returns:
        list[str] con file_ids unicos en orden.
    """
    file_ids: list[str] = []
    seen: set[str] = set()

    def _add(fid: str | None) -> None:
        if fid and fid not in seen:
            seen.add(fid)
            file_ids.append(fid)

    # 1. ContentParts (formato OpenAI Assistants v2)
    content = msg.content
    if isinstance(content, list):
        for part in content:
            if part.type == "file":
                _add(part.file_id)

    # 2. msg.files (formato Open WebUI nativo)
    if msg.files:
        for f in msg.files:
            if isinstance(f, dict) and f.get("type") == "file":
                _add(str(f.get("id", "")))

    return file_ids


async def _inject_file_contents(
    *,
    file_ids: list[str],
    db: Any,
    max_chars: int,
    user_text: str,
) -> str:
    """Prepend safely wrapped file text within one exact global budget.

    The budget covers escaped text, XML wrappers, separators, and any
    truncation note. This is especially important for multimodal requests,
    whose enriched text bypasses AgentLoop's normal file-reference resolver.
    """
    if not file_ids or max_chars <= 0:
        return user_text

    from hermes.agent.loop import wrap_file_content

    parts: list[str] = []
    truncated_files: list[str] = []
    # A successful prefix always ends with the two-character separator.
    remaining_budget = max_chars - 2

    for file_id in file_ids:
        entry = await db.get_file(file_id)
        if entry is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": (
                            f"File {file_id} not found. Did you upload it via POST /v1/files first?"
                        ),
                        "type": "not_found",
                    }
                },
            )
        text = str(entry.get("extracted_text") or "")
        filename = str(entry.get("filename") or file_id)
        if not text:
            logger.info(
                "http_api_file_inject_skipped_empty",
                extra={"file_id": file_id, "file_name": filename},
            )
            continue

        separator_cost = 2 if parts else 0
        available = remaining_budget - separator_cost
        empty_wrapper_len = len(wrap_file_content(filename, ""))
        if available <= empty_wrapper_len:
            truncated_files.append(filename)
            continue

        # XML escaping can expand a raw character (for example '&' -> '&amp;').
        # Binary-search the largest raw prefix whose final wrapped form fits.
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if len(wrap_file_content(filename, text[:mid])) <= available:
                low = mid
            else:
                high = mid - 1
        if low <= 0:
            truncated_files.append(filename)
            continue

        wrapped = wrap_file_content(filename, text[:low])
        parts.append(wrapped)
        remaining_budget -= separator_cost + len(wrapped)
        if low < len(text):
            truncated_files.append(filename)

    if not parts:
        return user_text

    prefix = "\n\n".join(parts) + "\n\n"
    if truncated_files and remaining_budget > 0:
        note = (
            "[Nota: contenido truncado al presupuesto global de "
            f"{max_chars} caracteres. Referencias truncadas: "
            f"{', '.join(truncated_files)}.]"
        )
        prefix += note[:remaining_budget]

    # Defense in depth: wrappers, escaping, separators, and notes all count.
    assert len(prefix) <= max_chars
    return prefix + user_text


# --- Exception handler especifico (v3.1 fix D) ---


async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    """Handler especifico para LLMError. NO es un catch-all de Exception.

    Por que especifico: un catch-all capturaria tambien RequestValidationError
    (que FastAPI lanza para payloads invalidos via Pydantic), regresionando
    422 (validation error) a 500 (server error). Este handler solo actua
    cuando LLMRouter.chat() lanza LLMError tras agotar retries.
    """
    logger.warning(
        "http_api_llm_error",
        extra={"path": str(request.url.path), "error": str(exc)},
    )
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": {
                "message": str(exc),
                "type": "api_error",
                "code": "llm_unavailable",
            }
        },
    )


# --- App factory (v3.1: no module-level app) ---


def create_app(
    settings: Settings,
    db: Database,
    router: Any,  # LLMRouter. Any para evitar import circular en tests.
    registry: ToolRegistry | None,
    embeddings_service: Any | None = None,  # EmbeddingsService, optional
    telemetry: Any | None = None,  # Telemetry, optional (Sprint 9.3.3)
    ocr_repo: Any | None = None,  # OcrPendingRepo, optional (Sprint 19 Slice 4d)
    edge_coordinator: Any | None = None,  # EdgeCoordinator, optional
    deep_research_capabilities: DeepResearchCapabilities | None = None,
) -> FastAPI:
    """Factory del FastAPI app. Recibe los singletons ya inicializados.

    Args:
        settings: Settings globales (de __main__.py).
        db: Database singleton (de __main__.py).
        router: LLMRouter singleton (de __main__.py).
        registry: ToolRegistry singleton o None si tools_enabled=False.

    Returns:
        FastAPI app lista para uvicorn.
    """
    app = FastAPI(
        title="Oroimen Middleware",
        version="0.5.8",
    )

    # Sprint 14 (US-2.1): expose settings on app.state so that Depends-based
    # auth (e.g. hermes.receivers.auth.authenticate_bearer used by jobs_api)
    # can read the configured `http_api_api_key`. The middleware path stays
    # unchanged; both auth code paths read the same value.
    app.state.settings = settings

    # Sprint 19 (post-PR #144 retro): wire collections_repo on app.state
    # so the /v1/collections endpoints don't 503. Before this fix, the
    # caller (tests, scripts, __main__) had to remember to set
    # `app.state.collections_repo = VaultCollectionsRepo(db)` manually.
    # Forgetting it = 503 Service Unavailable on every collections call,
    # with no error in the logs at startup. This is the same class of
    # bug as the original Sprint 14 settings wiring: a runtime
    # requirement that's only discovered on first request.
    #
    # Tested by tests/e2e/test_sprint_19_pipeline.py::test_collections_api
    # which does NOT wire collections_repo manually and expects 2xx.
    from hermes.memory.collections import VaultCollectionsRepo

    app.state.collections_repo = VaultCollectionsRepo(db)
    app.state.deep_research_capabilities = deep_research_capabilities or DeepResearchCapabilities()

    # Sprint 14 (US-2.1): mount deep research jobs router. Lazy import evita
    # circular dep (jobs_api importa authenticate_bearer de hermes.receivers.auth,
    # no de http_api; este include_router se ejecuta solo cuando create_app()
    # se llama en startup, ya con todos los modulos cargados).
    from hermes.receivers.jobs_api import router as jobs_router

    app.include_router(jobs_router)
    logger.info(
        "http_api_jobs_router_mounted",
        extra={"routes": len(jobs_router.routes)},
    )

    # Sprint 19 Slice 4d: OCR decision API (TDD §4.3.1 WebUI surface).
    # If ocr_repo is None (drop watcher disabled), skip.
    if ocr_repo is not None:
        from hermes.receivers.ocr_api import build_ocr_api_router

        ocr_api_router = build_ocr_api_router(db, ocr_repo, edge_coordinator, settings)
        app.include_router(ocr_api_router)
        logger.info(
            "http_api_ocr_router_mounted",
            extra={"routes": len(ocr_api_router.routes)},
        )

    # Sprint 9.0 (US-2.1): la DB es la fuente de verdad para files.
    # Sprint 15 (US-3.1): eliminamos el cache in-memory (`files_store`).
    # La DB es ahora la única fuente. Cada read va directo a SQLite
    # via `db.get_file(file_id)` (O(log n) en PK). Sin cache, sin
    # hydration, sin race entre escrituras concurrentes a multiples
    # workers de uvicorn.
    #
    # Antes (S9.0): dict local + _hydrate_files_cache() lazy + write-
    # through. Cada test que llamaba create_app() tenia su propio dict.
    # Sprint 15 lo elimina porque: (a) DB lookup es <1ms en LAN, (b)
    # el dict duplicaba la verdad (riesgo de drift), (c) complicaba
    # tests parametrizados (cada test re-hidrata), (d) hacia falta
    # serializar via asyncio.Lock si escalamos a mas workers.

    # Browser access is restricted to explicit, operator-configured origins.
    # Loopback publishing alone is not a browser security boundary: a hostile
    # website can still target localhost from the user's browser.
    cors_origins = [
        origin.strip() for origin in settings.http_api_cors_origins.split(",") if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    # Fase 0 Hardening: rate limiter por IP. Custom middleware (no dep
    # extra como slowapi). Ventana movil de 60s. /health y /v1/models
    # exentos (liveness checks del container + open-webui discovery que
    # hace multiples requests al arrancar). Si rate_limit_per_minute=0,
    # deshabilitado (dev workflow).
    _rate_limit = getattr(settings, "http_api_rate_limit_per_minute", 60) or 0
    if _rate_limit > 0:
        import time as _time
        from collections import deque as _deque

        from fastapi.responses import JSONResponse as _JSONResponse

        # Por IP: lista de timestamps en ventana movil. Cleanup lazy en cada
        # request (eliminamos timestamps fuera de la ventana).
        _rate_buckets: dict[str, _deque[float]] = {}

        @app.middleware("http")
        async def _rate_limit_middleware(request, call_next):
            # Exempt: liveness + open-webui discovery (ambas son GET cortas)
            if request.method == "OPTIONS" or request.url.path in ("/health", "/v1/models"):
                return await call_next(request)
            # IP del cliente (considerar X-Forwarded-For si reverse proxy)
            client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
                request.client.host if request.client else "unknown"
            )
            now = _time.monotonic()
            bucket = _rate_buckets.setdefault(client_ip, _deque())
            # Cleanup: quitar timestamps fuera de la ventana (60s)
            window_start = now - 60.0
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= _rate_limit:
                retry_after = max(1, int(60 - (now - bucket[0])))
                return _JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "type": "RateLimitError",
                            "message": f"Too many requests. "
                            f"Limit: {_rate_limit} req/min. Retry after {retry_after}s.",
                        }
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.append(now)
            return await call_next(request)

        logger.info("http_api_rate_limit_enabled", extra={"per_minute": _rate_limit})

    # Sprint 9.5: bearer token middleware. Si settings.http_api_api_key
    # esta configurado, requiere Authorization: Bearer <key> en TODOS los
    # endpoints EXCEPTO /health (liveness checks del container) y /v1/models
    # (open-webui lo consulta sin auth para discovery). /v1/chat/completions
    # y /v1/files/* requieren el token. Si http_api_api_key es None (default),
    # el middleware es no-op (legacy behavior, sin auth).
    _api_key = settings.http_api_api_key

    if _api_key:
        from fastapi import Request as _Req

        @app.middleware("http")
        async def _bearer_auth_middleware(request: _Req, call_next):
            # Exempt: liveness checks + model discovery
            if request.method == "OPTIONS" or request.url.path in ("/health", "/v1/models"):
                return await call_next(request)
            # Check Authorization header
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "AuthError",
                            "message": "Missing or invalid Authorization header. "
                            "Expected: Bearer <token>.",
                        }
                    },
                )
            token = auth_header[7:]  # strip "Bearer "
            if token != _api_key:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "AuthError",
                            "message": "Invalid API key.",
                        }
                    },
                )
            return await call_next(request)

        logger.info("http_api_bearer_auth_enabled")

    # Exception handler especifico (v3.1 fix D): solo LLMError.
    # FastAPI mantiene sus handlers nativos para RequestValidationError (422)
    # y HTTPException (status_code explicito), asi que no los sobreescribimos.
    app.add_exception_handler(LLMError, llm_error_handler)  # type: ignore[arg-type]

    # --- Endpoints ---

    @app.get("/v1/models")
    async def list_models() -> dict:
        """Endpoint requerido por Open WebUI al arrancar.

        Declara solo capabilities verificadas por el runtime publico. El upload
        y vision permanecen ocultos hasta cablear y probar ese contrato
        (Sprint 9.5 fix: open-webui >= 0.4 mira el campo `capabilities`
        en cada modelo de /v1/models; sin vision=true, open-webui
        deshabilita el upload aunque el LLM detras lo soporte).
        El smart router ya valida vision (todos los modelos del chain
        son vision-capaces via MiniMax API, ver MiniMax-M3 smoke test
        en scripts/mcp/mcp_influxdb/README.md).

        Sprint 12 (ADR-007): expone también los aliases definidos en
        `settings.llm_model_overrides`. Cada alias aparece como un modelo
        independiente. Capabilities se derivan heurísticamente: si
        cualquier modelo en su chain tiene vision (todos los actuales
        de MiniMax API), marcamos vision=True; tools=True siempre
        (todos los modelos actuales soportan tool calling).
        """
        # Default model: el chain primario con todas las capabilities.
        data: list[dict] = [
            {
                "id": PUBLIC_MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "oroimen",
                "capabilities": {
                    "vision": False,
                    "tools": True,
                    "file_upload": False,
                },
            }
        ]
        if settings.llm_text_frontier_enabled and settings.llm_text_frontier_api_key:
            data.append(
                {
                    "id": PUBLIC_FRONTIER_MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "oroimen",
                    "capabilities": {
                        "vision": False,
                        "tools": True,
                        "file_upload": False,
                    },
                }
            )
        # Aliases de llm_model_overrides. Capabilities basicas (sin vision
        # por defecto; el cliente que use el alias deberia saber que
        # es para tareas background).
        advertised_ids = {model["id"] for model in data}
        for configured_alias in settings.llm_model_overrides:
            alias = _canonical_model_alias(configured_alias)
            if alias in {PUBLIC_MODEL_ID, PUBLIC_FRONTIER_MODEL_ID}:
                continue
            if alias in advertised_ids:
                continue
            advertised_ids.add(alias)
            data.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "oroimen",
                    "capabilities": {
                        "vision": False,
                        "tools": True,
                        "file_upload": False,
                    },
                }
            )
        return {"object": "list", "data": data}

    @app.get("/health")
    async def health() -> Any:
        """Health activo: ping DB + reporte de circuit breakers.

        Sprint 9.3.1 fix: el healthcheck refleja el estado del SERVICIO
        (DB + HTTP API), no el estado de upstreams (LLM). Si el LLM está
        en rate limit (circuit breaker open), el servicio sigue funcional
        (chat persistence, tools, embeddings) — solo no puede responder
        queries hasta que el LLM vuelva. Reportamos 200 + status="degraded"
        para que Docker healthcheck no reinicie el container innecesariamente.
        El error 503 se reserva para cuando el servicio core está caído
        (DB inalcanzable, excepciones internas).
        """
        db_ok = await db.ping()
        if not db_ok:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unhealthy", "database": "disconnected"},
            )
        breaker_states = router.get_breaker_states()
        any_closed = any(state == "closed" for state in breaker_states.values())
        # Circuit breakers son soft failures (upstream LLM rate limit, no
        # problema del servicio). Reportar estado pero devolver 200.
        # El cliente puede leer "degraded" o "breakers" para decidir.
        return {
            "status": "ok" if any_closed else "degraded",
            "version": "0.5.8",
            "database": "connected",
            "breakers": breaker_states,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, _request: Request) -> Any:
        """Procesa una peticion OpenAI-compatible y ejecuta AgentLoop.

        Por cada request:
        1. Validar (al menos un user message)
        2. Crear conversacion efimera (chat_id=0 sentinel HTTP)
        3. Re-hidratar historial en DB (excluyendo system del cliente
           y el ultimo user message)
        4. Si stream=True: StreamingResponse con SSE de AgentLoop.run_stream
        5. Si stream=False: ejecutar AgentLoop.run, retornar JSON
        6. SIEMPRE archivar la conversacion (incluso si hubo error)

        Sprint 7.3: vision passthrough. Si last_user_msg es list[ContentPart],
        el texto se guarda en DB y la lista se pasa al LLM como vision
        (no se persiste base64).

        Sprint 8.7 v1.3: si el last_user_msg incluye referencias a archivos
        (en msg.files formato Open WebUI, o en content[] formato OpenAI),
        el texto extraido de files_store se inyecta en last_user_msg_text
        (DB persistence, P1-1). Si hay vision, la vision list se reconstruye
        desde cero para evitar duplicacion de la pregunta del usuario.
        404 si algun file_id no existe en files_store.
        """
        # Extraer ultimo user message
        user_msgs = [m for m in body.messages if m.role == "user"]
        if not user_msgs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "messages must include at least one user message",
                        "type": "invalid_request",
                    }
                },
            )
        requested_model = _canonical_model_alias(body.model)
        chain_override: list[str] | None
        if requested_model == PUBLIC_MODEL_ID:
            chain_override = None
        elif requested_model == PUBLIC_FRONTIER_MODEL_ID:
            if not (settings.llm_text_frontier_enabled and settings.llm_text_frontier_api_key):
                raise HTTPException(
                    status_code=400,
                    detail="Frontier model is not enabled for this deployment",
                )
            chain_override = [settings.llm_text_frontier_model]
        else:
            chain_override = _model_override(requested_model, settings.llm_model_overrides)
            if chain_override is None:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": {
                            "message": f"Model {body.model!r} was not found",
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    },
                )
        last_user_msg = user_msgs[-1]
        last_user_msg_content = last_user_msg.content
        # Texto para DB (extraido de vision si aplica)
        last_user_msg_text = _extract_text_from_content(last_user_msg_content)
        # Texto para el LLM (vision list text part). En caso vision +
        # file_refs, se enriquece con el file content directamente
        # (porque AgentLoop.run hace override de content en iter 0,
        # borrando el enriquecimiento de _resolve_file_refs).
        last_user_msg_text_for_llm = last_user_msg_text

        # S8.7 v1.3: extraer file_ids del message (defense in depth).
        # Busca en msg.content (OpenAI v2) Y msg.files (Open WebUI nativo).
        # S9.0: para el caso NO-vision ya NO inyectamos el texto del PDF
        # en `last_user_msg_text` (eso duplicaba el PDF en cada msg). En
        # su lugar, persistimos la REFERENCIA (`file_refs`) en el
        # message. AgentLoop resuelve el texto en runtime via
        # _resolve_file_refs cuando carga el historial. Esto da ~2,500x
        # reducción para PDFs reusados.
        # EXCEPCION: caso vision. El iter 0 del AgentLoop hace override
        # de `messages[-1]["content"]` con la vision list, REEMPLAZANDO
        # el content enriquecido por _resolve_file_refs. Para vision,
        # enriquecemos el text part directamente aquí (es la unica
        # manera de que el LLM vea el file content en la vision list).
        # Aun así, persistimos solo la pregunta en DB + file_refs, no
        # el texto duplicado.
        #
        # Sprint 15 (US-3.1): ya no hay fallback S8.7 con cache dict.
        # La DB es la única fuente: o el file_id existe o es 404.
        # _resolve_file_refs maneja orphans con MISSING_FILE_MARKER.
        file_ids = _extract_file_ids_from_message(last_user_msg)
        has_vision = isinstance(last_user_msg_content, list)
        if file_ids:
            if has_vision:
                # Caso vision: enriquecer SOLO el text para el LLM
                # (vision list text part). La DB guarda solo la
                # pregunta (last_user_msg_text) + file_refs. El helper
                # async lanza 404 si algun file_id no existe en DB.
                last_user_msg_text_for_llm = await _inject_file_contents(
                    file_ids=file_ids,
                    db=db,
                    max_chars=settings.read_tool_max_chars,
                    user_text=last_user_msg_text,
                )
            # Para no-vision, last_user_msg_text == last_user_msg_text_for_llm
            # (ambos son la pregunta sola). _resolve_file_refs hace
            # el trabajo de inyectar el file content al LLM.
            last_file_refs = file_ids
        else:
            last_file_refs = None

        # v1.3: reconstruir vision list DESDE CERO (fix duplicacion prosa).
        # Solo image_parts + UN text part con enriched_text.
        # NO usar _content_to_llm_parts aqui (mantiene text part original,
        # lo que duplicaria la pregunta del usuario).
        last_user_msg_parts: str | list[dict[str, Any]] | None = None
        if isinstance(last_user_msg_content, list):
            image_parts = [
                p.model_dump(exclude_none=True)
                for p in last_user_msg_content
                if p.type == "image_url"
            ]
            if image_parts:
                # UNA sola text part con last_user_msg_text_for_llm
                # (que en caso vision + file_refs ya tiene el file
                # content prepended, ver bloque de file_ids arriba).
                # image_parts solo contiene image_url.
                last_user_msg_parts = [
                    {"type": "text", "text": last_user_msg_text_for_llm},
                    *image_parts,
                ]
        if isinstance(last_user_msg_parts, str):
            last_user_msg_parts = None  # safety, no deberia pasar

        # 1. Crear o reusar conversacion.
        # Sprint 9.3.2c: antes new_conversation fallaba con UNIQUE
        # constraint si una conv huerfana existia (crash previo). Sprint
        # 9.3.2b uso get_or_create_conversation, pero eso REUSA la conv
        # huerfana con mensajes viejos que el LLM ve como historial
        # espurio (bug del "fantasma de Laufband": el nuevo 'Hola que
        # hora es' se mezclo con la query anterior en conv 182). Ahora
        # archivamos cualquier orphan con los mismos sentinels primero
        # y creamos una conv NUEVA vacia.
        #
        # La continuity entre chats de Open WebUI NO depende de la DB de
        # Hermes. WebUI guarda su propia copia local de cada chat y
        # reenvia TODO el historial en cada POST (re-hidratacion abajo).
        # Asi el LLM ve el contexto correcto del chat que se continua.
        #
        # Sprint 12 (ADR-007): si el cliente RikkaHub provee
        # metadata.chat_id, reusamos o creamos una conv PERSISTENTE con
        # ese chat_id (no se archiva al terminar). Si no, comportamiento
        # legacy: conv efimera HTTP (chat_id=0), archivada al final.
        client_chat_id_raw = None
        if body.metadata:
            cid = body.metadata.get("chat_id")
            if cid is not None:
                try:
                    client_chat_id_raw = int(cid)
                except (TypeError, ValueError):
                    client_chat_id_raw = None
        is_persistent = client_chat_id_raw is not None and client_chat_id_raw != 0
        if is_persistent:
            # El cliente RikkaHub ya envio el historial completo en
            # body.messages (mismo patron que OWUI). Reusamos la conv
            # con ese chat_id para que el LLM vea el contexto correcto.
            # user_id por defecto 1 (single-user setup del fork);
            # multi-user requeriria extraer de body.user o auth.
            assert client_chat_id_raw is not None  # narrowed by is_persistent
            conversation_id = await db.get_or_create_conversation(
                chat_id=client_chat_id_raw,
                user_id=1,
                thread_id=0,
            )
        else:
            conversation_id = await db.create_ephemeral_conversation(
                chat_id=0, user_id=0, thread_id=0
            )

        # 2. Re-hidratar historial: todos los mensajes EXCEPTO
        #    - el system message del cliente (AgentLoop tiene el suyo)
        #    - el ultimo user message (AgentLoop.run lo añadira)
        # Solo re-hidratamos mensajes con content=string (no vision raw).
        # Sprint 7.3: si un mensaje intermedio tiene vision, guardamos
        # solo el texto (vision es solo del turno actual).
        history_to_insert = [m for m in body.messages[:-1] if m.role != "system"]
        for msg in history_to_insert:
            text_content = _extract_text_from_content(msg.content)
            # S9.0: mensajes del historial con file references tambien
            # necesitan que se persistan sus file_refs para que el
            # AgentLoop los resuelva en runtime. Si el client re-envia
            # el mismo file en un mensaje intermedio (raro pero posible
            # en multi-turn), extraemos file_ids y los guardamos.
            history_file_refs = _extract_file_ids_from_message(msg)
            await db.add_message(
                conversation_id=conversation_id,
                role=msg.role,
                content=text_content,
                file_refs=history_file_refs or None,
            )

        # 3. Construir AgentLoop con settings explicitos
        # Sprint 12 (ADR-007): si el `model` del request coincide con
        # un alias de llm_model_overrides, pasar la chain correspondiente al
        # router. Asi un cliente que pide `model: "oroimen-agent-fast"`
        # usa solo MiniMax-M2.7-highspeed (sin fallback a MiniMax-M3).
        loop = AgentLoop(
            router=router,
            registry=registry,  # type: ignore[arg-type]
            db=db,
            settings=settings,
            step_callback=None,
            max_iterations=settings.agent_max_iterations,
            telemetry=telemetry,  # Sprint 9.3.3: para que record_llm_call/record_tool_call se ejecuten
            chain_override=chain_override,
            embeddings_service=embeddings_service,  # Sprint 16 (US-3.2)
        )

        # 4. Branch: streaming vs blocking
        if body.stream:
            # Sprint 7.3: streaming via StreamingResponse.
            # run_stream() yields StreamChunk. Convertimos a SSE.
            return StreamingResponse(
                _stream_response(
                    loop=loop,
                    conversation_id=conversation_id,
                    user_message_text=last_user_msg_text,
                    user_message_parts=last_user_msg_parts,
                    file_refs=last_file_refs,
                    is_persistent=is_persistent,
                    response_model=requested_model,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # disable nginx buffering
                },
            )

        # 5. Non-streaming (legacy)
        try:
            response_text = await loop.run(
                conversation_id,
                last_user_msg_text,
                user_message_parts=last_user_msg_parts,
                file_refs=last_file_refs,
            )

            # v3.1 fix 6: detectar cuando AgentLoop retorna el fallback
            # de error (todos los modelos del chain fallaron). En ese
            # caso, devolver 502 con formato OpenAI en vez de propagar
            # el mensaje de error como si fuera respuesta normal.
            if response_text == LLM_ERROR_FALLBACK_MESSAGE:
                return JSONResponse(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    content={
                        "error": {
                            "message": "All LLM models in chain failed. Try again later.",
                            "type": "api_error",
                            "code": "llm_unavailable",
                        }
                    },
                )

            # 4. Extraer usage del ultimo assistant message
            last_asst = await db.get_last_assistant_message(conversation_id)
            if last_asst is not None:
                tokens_in = last_asst.get("tokens_in") or 0
                tokens_out = last_asst.get("tokens_out") or 0
                usage = {
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                }
            else:
                usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
        finally:
            # 5. Archivar la conversacion SOLO si es efimera.
            # Sprint 12 (ADR-007): las conversaciones persistentes
            # (chat_id != 0) que crea el cliente nativo RikkaHub NO se
            # archivan: el usuario las retoma al abrir la app. El job
            # archive_stale_conversations tambien las exime.
            if not is_persistent:
                await db.archive_conversation(conversation_id)

    # Sprint 12 (ADR-007): endpoints para el cliente nativo RikkaHub.
    # Permiten listar conversaciones activas y leer su historial de
    # mensajes con paginacion forward (cursor antes de updated_at /
    # created_at). Defense in depth: el server filtra is_archived=0
    # a nivel SQL, asi el cliente nunca recibe chats archivados.
    @app.get("/v1/conversations")
    async def list_conversations(
        user_id: int = 1,
        limit: int = 20,
        before: str | None = None,
    ) -> dict:
        """Lista conversaciones activas del usuario.

        Query params:
            user_id: scoping por usuario (default 1, single-user fork).
            limit: numero maximo de conversaciones a devolver (1-100, default 20).
            before: cursor ISO 8601 (updated_at) para paginacion forward.

        Returns:
            dict OpenAI-like:
            {
                "object": "list",
                "data": [
                    {
                        "id": 42,
                        "chat_id": 1001,
                        "thread_id": 0,
                        "title": "...",
                        "created_at": "2026-06-30 12:00:00",
                        "updated_at": "2026-06-30 13:00:00",
                        "last_message_preview": "..."
                    },
                    ...
                ]
            }
        """
        conversations = await db.list_conversations(
            user_id=user_id,
            limit=limit,
            before=before,
        )
        return {
            "object": "list",
            "data": conversations,
        }

    @app.get("/v1/conversations/{conv_id}/messages")
    async def get_conversation_messages(
        conv_id: int,
        user_id: int = 1,
        limit: int = 100,
        before: str | None = None,
    ) -> dict:
        """Devuelve mensajes de una conversacion con paginacion forward.

        Path params:
            conv_id: id de la conversacion en Oroimen DB.

        Query params:
            user_id: scoping por usuario para verificar pertenencia.
            limit: numero maximo de mensajes (1-500, default 100).
            before: cursor ISO 8601 (created_at) para paginacion.

        Returns:
            dict OpenAI-like. Si la conversacion no existe o no pertenece
            al usuario, 404.
        """
        # Verificar que la conversacion pertenece al usuario
        async with db.conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"conversation {conv_id} not found",
                        "type": "not_found",
                    }
                },
            )
        messages = await db.get_conversation_messages(
            conv_id=conv_id,
            limit=limit,
            before=before,
        )
        return {
            "object": "list",
            "data": messages,
        }

    # Sprint 12.1 (TDD_S12_DELETE_AND_SYNC.md §4): DELETE + restore +
    # sync cursor-based para que el cliente RikkaHub pueda borrar y
    # sincronizar conversaciones persistentes (chat_id != 0).

    @app.delete("/v1/conversations/{conv_id}")
    async def delete_conversation(
        conv_id: int,
        user_id: int = 1,
    ) -> Response:
        """Soft delete + encrypt atómico del content de messages.

        Sprint 12.1 (ADR-007 / TDD §4.1):
        - Verifica pertenencia (conv_id + user_id).
        - Soft-archive + cifra cada message (Fernet) en una transaccion.
        - Marca deleted_at, encrypted_at, purge_at (+7d default).
        - Idempotente: si ya esta archivada, retorna 204 sin error.
        - Sin encryption_key: safe-fallback (plain archive, sin cifrado).

        Retorna 204 No Content.
        404 si la conv no existe.
        401 si bearer invalido (manejado por middleware).
        """

        enc_key_bytes = None
        if settings.conversation_encryption_key:
            enc_key_bytes = settings.conversation_encryption_key.encode("ascii")

        retention = settings.conversation_retention_days
        deleted = await db.soft_delete_conversation(
            conversation_id=conv_id,
            user_id=user_id,
            encryption_key=enc_key_bytes or b"",
            retention_days=retention,
        )
        if not deleted:
            # La conv no existe o no pertenece al user, o ya esta archivada
            # (idempotente: tambien retorna 204 — no es error).
            # Para distinguir "no existe" de "ya archivada" hariamos un
            # SELECT extra; por simplicidad retornamos 204 idempotente.
            # Si realmente no existe, el siguiente sync lo confirma.
            # Pero el cliente espera 404 para retry logic distinto.
            # Distingamos: SELECT primero.
            async with db.conn.execute(
                "SELECT id FROM conversations WHERE id=? AND user_id=?",
                (conv_id, user_id),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": {
                            "message": f"conversation {conv_id} not found",
                            "type": "not_found",
                        }
                    },
                )
            # row existe pero deleted=False: ya estaba archivada (idempotente)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/v1/conversations/{conv_id}/restore")
    async def restore_conversation(
        conv_id: int,
        user_id: int = 1,
    ) -> dict:
        """Revierte un soft_delete dentro de la ventana de 7d.

        Sprint 12.1 (TDD §4.2): descifra el content de messages y
        restaura is_archived=0.

        Retorna 200 con {"id": conv_id, "status": "restored"}.
        404 si la conv no existe.
        410 Gone si ya fue hard-deleted (>7d).
        503 si no hay encryption_key y la conv fue cifrada (no podemos
            descifrar). Safe-fallback: si encrypted_at IS NULL (TDD §7.2),
            el restore funciona aunque no haya key (legacy /clear de
            Telegram o archive sin cifrado).
        """
        enc_key_bytes = None
        if settings.conversation_encryption_key:
            enc_key_bytes = settings.conversation_encryption_key.encode("ascii")

        # Pre-check para distinguir 404 vs 410 vs 503
        from datetime import UTC, datetime

        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        async with db.conn.execute(
            "SELECT is_archived, deleted_at, encrypted_at, purge_at "
            "FROM conversations WHERE id=? AND user_id=?",
            (conv_id, user_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"conversation {conv_id} not found",
                        "type": "not_found",
                    }
                },
            )
        if not row["is_archived"] or row["deleted_at"] is None:
            # No tombstoned, nada que restaurar. Devolver 200 idempotente
            # (podria ser 409 Conflict pero idempotente es mas simple).
            return {"id": conv_id, "status": "already_active"}
        if row["purge_at"] is not None and row["purge_at"] <= now_str:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail={
                    "error": {
                        "message": f"conversation {conv_id} was hard-deleted (>retention window)",
                        "type": "purged",
                    }
                },
            )
        if row["encrypted_at"] is not None and not enc_key_bytes:
            # Cifrado y sin key: 503 (TDD §7.2 safe-fallback).
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "encryption key not configured; cannot restore encrypted content",
                        "type": "encryption_key_missing",
                    }
                },
            )

        restored = await db.restore_conversation(conv_id, user_id, enc_key_bytes or b"")
        if not restored:
            # Algo fallo durante el descifrado (key incorrecta?). Re-check.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "restore failed (encryption key mismatch?)",
                        "type": "restore_failed",
                    }
                },
            )
        return {"id": conv_id, "status": "restored"}

    @app.get("/v1/conversations/sync")
    async def sync_conversations(
        user_id: int = 1,
        updated_after: int = 0,  # epoch sec, 0 = cold start
        limit: int = 100,
    ) -> dict:
        """Sync cursor-based delta (TDD §4.4). Estilo Notion / Linear / WhatsApp.

        Devuelve dos listas:
        - upserted: convs activas con updated_at > cursor (nuevas o modificadas).
        - deleted: convs con deleted_at > cursor (borradas por user o admin).

        El cliente itera con next_cursor si upserted llena el limit.
        """
        import time as _time
        from datetime import UTC, datetime

        # Validacion basica del cursor (TDD §4.4-a).
        now = int(_time.time())
        if updated_after > now + 60:  # 60s de margen para clock skew
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": f"updated_after {updated_after} is in the future",
                        "type": "invalid_cursor",
                    }
                },
            )
        if updated_after < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "updated_after must be >= 0",
                        "type": "invalid_cursor",
                    }
                },
            )

        # Convertir epoch sec a 'YYYY-MM-DD HH:MM:SS' UTC (zero-padded).
        # IMPORTANTE (TDD §11 riesgo 1): SQLite no tiene DATETIME,
        # usa TEXT. La comparacion updated_at > ? es alfabetica.
        # Si el formato no es exactamente UTC zero-padded, el delta
        # sync falla silenciosamente. Validado contra db.py:802 que
        # usa el mismo patron.
        cutoff_dt = datetime.fromtimestamp(updated_after, tz=UTC)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")

        upserted = await db.get_conversations_sync(
            user_id=user_id,
            updated_after=cutoff_str,
            limit=limit,
        )
        deleted = await db.get_deleted_conversations_since(
            user_id=user_id,
            deleted_after=cutoff_str,
            limit=limit,
        )

        cursor_now = now
        # next_cursor: si CUALQUIERA de las dos listas (upserted o deleted)
        # lleno el limit, hay mas. Devolvemos el max timestamp
        # (updated_at de upserted o deleted_at de deleted) para que el
        # cliente continue desde ahi. Si ninguna lleno, next_cursor ==
        # cursor_now (el cliente actualiza lastSyncAt y la proxima sync
        # parte de cursor_now).
        #
        # CRITICO (Copilot review): antes solo se consideraba `upserted`
        # para avanzar el cursor. Si `deleted` llenaba el limit pero
        # `upserted` no, el cliente nunca recibia los tombstones restantes
        # (quedaban sin avisar al cliente para siempre). El fix:
        # `next_cursor = max(max_upserted, max_deleted)`.
        #
        # IMPORTANTE (TDD trampa #1 - SQLite TEXT datetime): el formato
        # es 'YYYY-MM-DD HH:MM:SS' UTC. Usamos calendar.timegm (no
        # _time.mktime) porque mktime interpreta como LOCAL TIME, lo que
        # falla en zonas horarias != UTC. En el NAS host con timezone
        # Europe/Madrid, mktime devolveria un epoch 1-2h menor.
        import calendar as _calendar

        def _to_epoch(ts: str) -> int | None:
            try:
                return int(_calendar.timegm(_time.strptime(ts, "%Y-%m-%d %H:%M:%S")))
            except (ValueError, TypeError):
                return None

        next_cursor = cursor_now
        max_ts: int | None = None
        if upserted and len(upserted) >= limit:
            for u in upserted:
                e = _to_epoch(u["updated_at"])
                if e is not None and (max_ts is None or e > max_ts):
                    max_ts = e
        if deleted and len(deleted) >= limit:
            for d in deleted:
                e = _to_epoch(d["deleted_at"])
                if e is not None and (max_ts is None or e > max_ts):
                    max_ts = e
        if max_ts is not None:
            next_cursor = max_ts

        return {
            "object": "list",
            "cursor_now": cursor_now,
            "next_cursor": next_cursor,
            "upserted": upserted,
            "deleted": deleted,
        }

    @app.post("/v1/files")
    async def upload_file(
        file: UploadFile = File(...),
        purpose: str = "assistants",
    ) -> Response:
        """Sube un archivo y extrae texto (PDFs via pypdf).

        Sprint 7.3 MVP: store in-memory (dict). Sprint 8+ persistente
        en filesystem. Sprint 9.0: persistente en DB (tabla `files`).
        El texto extraído se guarda en `files.extracted_text`.

        Sprint 13.0 (S8.6 fix): pypdf ahora corre en `asyncio.to_thread`
        para no bloquear el event loop. PDF 750KB: ~90s → ~5-8s. /health
        responde <100ms durante upload (regla del S8.4).

        Sprint 15 (US-3.1): dedup transparente por SHA256 del texto
        extraído. Si ya existe un file con el mismo content_hash
        (idx_files_content_hash UNIQUE INDEX en la DB), retornamos
        HTTP 200 + `deduplicated=true` con el file_id existente, en vez
        de crear un duplicado. Esto cubre tres casos:
        - User re-sube el mismo PDF dos veces → mismo file_id
        - Sprint 10 traera un file de Google Drive con el mismo texto
          → un solo file en la library
        - Open WebUI puede reintentar un upload tras timeout sin
          inflar la library

        Devuelve HTTP 201 + `deduplicated=false` si el file es nuevo.
        Devuelve HTTP 200 + `deduplicated=true` si es duplicado.

        Args:
            file: UploadFile de FastAPI (multipart/form-data).
            purpose: string OpenAI-compatible ("assistants", "fine-tune", etc.)

        Returns:
            JSONResponse con dict OpenAI file format + campos
            `extracted_text` (texto extraido) y `deduplicated` (bool).
        """
        content = await file.read()
        # Sprint 13.0 (S8.6 fix): extrae PDF async. Antes bloqueaba el
        # event loop entero. Ahora /health responde <100ms durante upload.
        extracted_text = await _extract_file_text_async(
            content, file.content_type or "", file.filename or ""
        )

        # Sprint 15 (US-3.1 §2.1): calcular SHA256 del TEXTO EXTRAIDO
        # (no del binario) para el dedup cross-source. Si extraemos
        # `manual.txt` y `extracted.txt` ambos terminan en el mismo
        # texto normalizado y colisionan. Para PDFs, los bytes difieren
        # si uno tiene bookmarks/payload pero el texto extraido es el
        # mismo → colisionan también. Esto es lo que queremos: dedup
        # SEMANTICO del contenido.
        #
        # Casos donde content_hash es None (no dedup):
        # - Archivos sin texto extraído (imágenes, binarios): el hash
        #   seria SHA256(b"") = siempre igual, lo cual provoca falsos
        #   positivos. Mejor guardar NULL y permitir duplicados.
        content_hash: str | None = None
        if extracted_text:
            content_hash = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()

        # Dedup check ANTES de generar file_id nuevo (no desperdiciamos
        # un UUID si vamos a dedupe).
        existing: dict | None = None
        if content_hash is not None:
            existing = await db.find_file_by_content_hash(content_hash)

        if existing is not None:
            # HIT: reutilizar el file_id existente. Devolver el row
            # completo con `deduplicated=true` + HTTP 200 (OK, no se
            # creo nada nuevo; idempotente). Si llegamos aqui, content_hash
            # es no-None (el branch requiere que lo fuera para el lookup),
            # pero mypy no lo sabe: usamos un local narrowing.
            assert content_hash is not None  # narrowing para mypy
            content_hash_prefix = content_hash[:16]
            logger.info(
                "http_api_file_upload_dedup_hit",
                extra={
                    "file_id": existing["id"],
                    "file_name": file.filename,
                    "content_hash": content_hash_prefix,
                },
            )
            # Sprint 15: created_at en DB es TIMESTAMP TEXT ("YYYY-MM-DD
            # HH:MM:SS"), no epoch. Convertir a epoch int para mantener
            # shape OpenAI-compatible en la respuesta. Si es None o no
            # parseable, fallback al tiempo actual (mejor que 500).
            created_at_raw = existing.get("created_at")
            created_epoch: int
            if created_at_raw is not None:
                try:
                    created_epoch = int(
                        datetime.strptime(created_at_raw, "%Y-%m-%d %H:%M:%S")
                        .replace(tzinfo=UTC)
                        .timestamp()
                    )
                except ValueError:
                    created_epoch = int(time.time())
            else:
                created_epoch = int(time.time())
            entry = {
                "id": existing["id"],
                "object": "file",
                "bytes": existing["size_bytes"],
                "created_at": created_epoch,
                "filename": existing["filename"],
                "purpose": purpose,
                "extracted_text": existing.get("extracted_text") or "",
                "deduplicated": True,
            }
            return JSONResponse(status_code=status.HTTP_200_OK, content=entry)

        # MISS: file nuevo. Generar file_id y persistir.
        file_id = f"file_{uuid.uuid4().hex[:24]}"
        try:
            await db.add_file(
                file_id=file_id,
                filename=file.filename or "untitled",
                mime_type=file.content_type,
                size_bytes=len(content),
                extracted_text=extracted_text,
                extraction_method=(
                    "pypdf"
                    if (
                        file.content_type == "application/pdf"
                        or (file.filename or "").lower().endswith(".pdf")
                    )
                    else ""
                ),
                source="upload",
                content_hash=content_hash,
            )
        except sqlite3.IntegrityError:
            # Sprint 17 (F4-1): race entre find_file_by_content_hash + add_file.
            # Dos requests concurrentes con el MISMO content_hash pasan el
            # dedup check (ambos ven None), y uno gana el INSERT. El otro
            # recibe UNIQUE INDEX violation. El UNIQUE INDEX es la red de
            # seguridad; nosotros debemos devolver una respuesta idempotente
            # 200 (el cliente retry ve el mismo file_id) en vez de 500.
            #
            # Refetch: el row ganador YA esta committed (SQLite WAL auto-commit
            # por INSERT). Si el refetch falla (no encontrado), retry una vez
            # con un pequeño sleep para tolerar la ventana de visibilidad.
            if content_hash is None:
                # Sin content_hash no deberia haber UNIQUE violation
                # (el UNIQUE INDEX es partial WHERE content_hash IS NOT
                # NULL), pero defensivamente re-raise.
                raise
            logger.info(
                "http_api_file_upload_dedup_race_winner",
                extra={
                    "file_id": file_id,
                    "content_hash": content_hash[:16],
                    "file_name": file.filename,
                },
            )
            existing_race = await db.find_file_by_content_hash(content_hash)
            if existing_race is None:
                # Visibilidad eventual: tiny window. Retry 1 vez con sleep
                # minimo (escrituras SQLite son auto-commit pero lectura
                # cross-connection puede ver WAL antes del commit).
                import asyncio as _asyncio

                await _asyncio.sleep(0.01)
                existing_race = await db.find_file_by_content_hash(content_hash)
            if existing_race is None:
                # No se pudo recuperar — propagar el error para no
                # perder visibilidad del fallo. Logged + return 500
                # semantico via HTTPException.
                raise
            # Construir respuesta idempotente a partir del row ganador.
            created_at_raw = existing_race.get("created_at")
            if created_at_raw is not None:
                try:
                    created_epoch_race = int(
                        datetime.strptime(created_at_raw, "%Y-%m-%d %H:%M:%S")
                        .replace(tzinfo=UTC)
                        .timestamp()
                    )
                except ValueError:
                    created_epoch_race = int(time.time())
            else:
                created_epoch_race = int(time.time())
            entry = {
                "id": existing_race["id"],
                "object": "file",
                "bytes": existing_race["size_bytes"],
                "created_at": created_epoch_race,
                "filename": existing_race["filename"],
                "purpose": purpose,
                "extracted_text": existing_race.get("extracted_text") or "",
                "deduplicated": True,
            }
            logger.info(
                "http_api_file_upload_dedup_race_resolved",
                extra={
                    "winner_file_id": existing_race["id"],
                    "loser_file_id": file_id,
                    "content_hash": content_hash[:16],
                },
            )
            return JSONResponse(status_code=status.HTTP_200_OK, content=entry)
        entry = {
            "id": file_id,
            "object": "file",
            "bytes": len(content),
            "created_at": int(time.time()),
            "filename": file.filename or "untitled",
            "purpose": purpose,
            "extracted_text": extracted_text,
            "deduplicated": False,
        }
        logger.info(
            "http_api_file_uploaded",
            extra={
                "file_id": file_id,
                # NOTA: 'filename' es reserved key en LogRecord de Python
                # 3.11+ (KeyError si lo pones en extra). Usamos 'file_name'.
                "file_name": file.filename,
                "bytes": len(content),
                "content_type": file.content_type,
                "content_hash": (content_hash[:16] if content_hash else None),
            },
        )
        # S9.1: auto-embed best-effort. Si RAG está disabled o la API
        # call falla, el upload sigue siendo exitoso (el file está
        # en la library, simplemente no es buscable semánticamente).
        # El user puede re-embed más tarde via tool o endpoint.
        if embeddings_service is not None and extracted_text:
            try:
                ensure_initialized = getattr(embeddings_service, "ensure_initialized", None)
                if callable(ensure_initialized):
                    await ensure_initialized()
                if getattr(embeddings_service, "is_enabled", False):
                    embedded = await embeddings_service.embed_and_store(file_id, extracted_text)
                    if embedded:
                        logger.info(
                            "http_api_file_embedded",
                            extra={"file_id": file_id},
                        )
            except Exception as exc:
                # No fallar el upload. Log warning y continuar.
                logger.warning(
                    "http_api_file_embed_failed",
                    extra={
                        "file_id": file_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:200],
                    },
                )
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=entry)

    @app.get("/v1/files")
    async def list_files(
        limit: int = 100,
        offset: int = 0,
        source: str | None = None,
    ) -> dict:
        """Lista archivos de la library (paginated cursor).

        Sprint 15 (US-3.1 §4 PR #69): antes de este PR, los files
        subidos solo eran accesibles si el cliente recordaba el
        file_id. No habia forma de listar "que tengo en la library".
        Open WebUI lo necesita para mostrar la library en su UI;
        el LLM agent lo usa indirectamente via la tool search_files,
        pero un script/dashboard puede querer browse manual.

        Args:
            limit: maximo de rows (default 100, max 500). Equilibrio
                entre payload y UX: 100 entries x ~500B = 50KB JSON.
            offset: numero de rows a saltar (default 0). Paginated
                cursor: el cliente incrementa offset += len(data)
                hasta que data == [].
            source: filtro opcional por source column ('upload',
                'google_drive' futuro). None = todos.

        Returns:
            dict shape:
            {
                "object": "list",
                "data": [...files...],
                "count": N,
                "limit": limit,
                "offset": offset,
                "has_more": bool,
            }
        """
        # Clamp defensivo para evitar queries accidentales pesados.
        # 500 es el techo absoluto (Gemini §8.1: rate limiting de DB).
        limit = max(1, min(500, limit))
        offset = max(0, offset)
        # Chore 2026-07-05 (Nemotron 3 Ultra 550B review): has_more
        # debe ser fiable cuando la última página tiene EXACTAMENTE
        # `limit` items. Truco clásico: pedimos limit+1 y si vuelven
        # limit+1 rows, descartamos la última y marcamos has_more=true.
        # Coste: +1 row transferido por página (pequeño vs simplicidad).
        rows = await db.list_files(source=source, limit=limit + 1, offset=offset)
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]
        return {
            "object": "list",
            "data": rows,
            "count": len(rows),
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
        }

    @app.get("/v1/files/{file_id}")
    async def get_file(file_id: str) -> dict:
        """Recupera un archivo subido por su ID.

        Sprint 9.0 (US-2.1): lee directo de la DB (PK lookup O(log n)).
        Sprint 15 (US-3.1): elimina el cache in-memory `files_store`.
        Cada request va directo a SQLite via `db.get_file(file_id)`.
        Sin hydration, sin cache, sin race entre workers.

        Returns:
            dict con TODAS las columnas de `files`, incluyendo:
            - id, filename, mime_type, size_bytes, extracted_text
            - extraction_method, source, source_metadata, tags
            - content_hash (Sprint 15 dedup)
            - created_at, last_referenced_at, **reference_count**
              (Sprint 15 §10 inode-like: expone el conteo de uso
              directamente para que clientes (Open WebUI, scripts) puedan
              decidir si un archivo vale la pena embed/re-embed sin
              tener que hacer /refs primero).
        """
        entry = await db.get_file(file_id)
        if entry is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"File {file_id} not found",
                        "type": "not_found",
                    }
                },
            )
        return entry

    @app.get("/v1/files/{file_id}/refs")
    async def get_file_refs(file_id: str, limit: int = 50) -> dict:
        """Lista las conversaciones + mensajes que referencian este file.

        Sprint 15 (US-3.1 §10 inode-like — antesala del modelo inode
        completo que se implementara en Sprint 10 con la integracion
        de Google Drive). Devuelve "quien usa este PDF?" sin que el
        cliente tenga que escanear todas las conversaciones. Util para:

        - LLM agent que necesita decidir si un archivo sigue siendo
          relevante ("este PDF fue mencionado en 3 conversaciones, pero
          ninguna en los ultimos 30 dias — probablemente obsoleto").
        - Open WebUI UI: mostrar "usado en N conversaciones" al lado del
          archivo en la library.
        - Auditoria / GDPR: "en que conversaciones aparece este file?"
          sin tener que cargar cada conv.

        Args:
            file_id: el ID del file.
            limit: maximo de rows (default 50, suficiente para una
                library tipica).

        Returns:
            dict con shape:
            {
                "object": "list.refs",
                "file_id": "file_abc...",
                "data": [
                    {
                        "message_id": int,
                        "conversation_id": int,
                        "role": "user"|"assistant"|"tool",
                        "created_at": "YYYY-MM-DD HH:MM:SS",
                        "content_snippet": str (primeros 200 chars),
                    },
                    ...
                ],
                "count": int,
            }
        404 si el file no existe (vs. 200 con data=[] si existe pero
        no tiene referencias — son cosas distintas semánticamente).

        Performance:
            Usa json_each() de SQLite (json1 module, habilitado por
            default en Python 3.12+). O(N) sobre messages con file_refs
            no-null. Para libraries <10k messages, ~ms en LAN.
        """
        # Verificar que el file existe primero (404 si no, vs 200+[] si
        # existe pero no tiene refs). Esto es una decision de API: el
        # cliente no deberia distinguir "file borrado" de "file sin
        # referencias" — son dos señales utiles distintas.
        if await db.get_file(file_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"File {file_id} not found",
                        "type": "not_found",
                    }
                },
            )
        # Clamp defensivo del limit (Gemini §8.5 equivalente + PR #68
        # free-LLM review): un cliente despistado o malicioso podria
        # pasar limit=999999 y forzar un query O(N) sobre todos los
        # messages con file_refs. Cap a 200 (mas que suficiente para
        # libraries tipicas; usuarios con >200 refs pueden paginar
        # cuando implementemos cursor-based pagination).
        limit = max(1, min(200, limit))
        rows = await db.find_messages_with_file_ref(file_id, limit=limit)
        return {
            "object": "list.refs",
            "file_id": file_id,
            "data": rows,
            "count": len(rows),
        }

    # --- Sprint 19: Vault Collections API endpoints -----------------------

    def _collections_repo_or_404() -> Any:
        """Fetch VaultCollectionsRepo from app.state, or raise 503."""
        repo = getattr(app.state, "collections_repo", None)
        if repo is None:
            raise HTTPException(
                status_code=503,
                detail="collections_repo not wired into app.state",
            )
        return repo

    def _collection_to_response(c: Any) -> dict[str, Any]:
        """Convert Collection dataclass to JSON-friendly dict."""
        return {
            "collection_id": c.collection_id,
            "name": c.name,
            "parent_collection_id": c.parent_collection_id,
            "description": c.description,
            "sort_order": c.sort_order,
            "archived": int(c.archived),
            "archived_at": c.archived_at,
            "created_at": c.created_at,
        }

    @app.post(
        "/v1/collections",
        status_code=201,
        response_model=CollectionResponse,
    )
    async def create_collection_endpoint(body: CreateCollectionRequest) -> Any:
        """POST /v1/collections — create a new collection."""
        from hermes.memory.collections import (
            CollectionNotFoundError,
            DuplicateCollectionError,
        )

        repo = _collections_repo_or_404()
        try:
            c = await repo.create_collection(
                name=body.name,
                parent_collection_id=body.parent_collection_id,
                description=body.description,
                sort_order=body.sort_order,
            )
        except DuplicateCollectionError as exc:
            raise HTTPException(status_code=409, detail="collection name already exists") from exc
        except CollectionNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="parent_collection_id not found",
            ) from exc
        return _collection_to_response(c)

    @app.get("/v1/collections", response_model=ListCollectionsResponse)
    async def list_collections_endpoint(include_archived: bool = False) -> Any:
        """GET /v1/collections — list all collections (active by default)."""
        repo = _collections_repo_or_404()
        rows = await repo.list_collections(include_archived=include_archived)
        return {"collections": [_collection_to_response(c) for c in rows]}

    @app.post(
        "/v1/collections/{collection_id}/restore",
        response_model=CollectionResponse,
    )
    async def restore_collection_endpoint(collection_id: str) -> Any:
        """POST /v1/collections/{id}/restore — unarchive a collection."""
        from hermes.memory.collections import CollectionNotFoundError

        repo = _collections_repo_or_404()
        if len(collection_id) != 32:
            raise HTTPException(status_code=404, detail="collection_id invalid format")
        try:
            await repo.restore_collection(collection_id)
        except CollectionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="collection not found") from exc
        c = await repo.get_collection(collection_id)
        if c is None:
            raise HTTPException(status_code=404, detail="collection not found")
        return _collection_to_response(c)

    @app.patch(
        "/v1/collections/{collection_id}",
        response_model=CollectionResponse,
    )
    async def patch_collection_endpoint(
        collection_id: str,
        body: PatchCollectionRequest,
    ) -> Any:
        """PATCH /v1/collections/{id} — update name, parent, description, sort_order."""
        import sqlite3

        repo = _collections_repo_or_404()
        if len(collection_id) != 32:
            raise HTTPException(status_code=404, detail="collection_id invalid format")

        # Cycle prevention: walk up parent chain of body.parent_collection_id
        # and ensure collection_id is NOT in that chain.
        if body.parent_collection_id is not None:
            cur = body.parent_collection_id
            seen: set[str] = set()
            for _ in range(50):  # safety cap
                if cur == collection_id:
                    raise HTTPException(
                        status_code=409,
                        detail="parent_collection_id would create a cycle",
                    )
                if cur in seen:
                    break  # pre-existing cycle in DB (shouldn't happen but safe)
                seen.add(cur)
                row = await repo.get_collection(cur)
                if row is None:
                    raise HTTPException(
                        status_code=404,
                        detail="parent_collection_id not found",
                    )
                cur = row.parent_collection_id
                if cur is None:
                    break

        # Apply updates. Use exclude_unset to distinguish "client passed
        # null" (e.g., to UNPARENT) from "client omitted field" (don't change).
        explicit_fields = body.model_dump(exclude_unset=True)
        if explicit_fields:
            try:
                async with db._write_lock:
                    await db.conn.execute("BEGIN IMMEDIATE")
                    try:
                        if "name" in explicit_fields:
                            await db.conn.execute(
                                "UPDATE vault_collections SET name = ? WHERE collection_id = ?",
                                (explicit_fields["name"], collection_id),
                            )
                        if "description" in explicit_fields:
                            await db.conn.execute(
                                "UPDATE vault_collections SET description = ? WHERE collection_id = ?",
                                (explicit_fields["description"], collection_id),
                            )
                        if "sort_order" in explicit_fields:
                            await db.conn.execute(
                                "UPDATE vault_collections SET sort_order = ? WHERE collection_id = ?",
                                (explicit_fields["sort_order"], collection_id),
                            )
                        if "parent_collection_id" in explicit_fields:
                            await db.conn.execute(
                                "UPDATE vault_collections SET parent_collection_id = ? WHERE collection_id = ?",
                                (explicit_fields["parent_collection_id"], collection_id),
                            )
                        await db.conn.execute("COMMIT")
                    except BaseException:
                        with suppress(Exception):
                            await db.conn.execute("ROLLBACK")
                        raise
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    raise HTTPException(
                        status_code=409,
                        detail="collection name already exists",
                    ) from e
                raise

        # Verify collection still exists (could have been deleted concurrently)
        c = await repo.get_collection(collection_id)
        if c is None:
            raise HTTPException(status_code=404, detail="collection not found")
        return _collection_to_response(c)

    @app.delete("/v1/collections/{collection_id}")
    async def delete_collection_endpoint(
        collection_id: str,
        confirm: bool = False,
    ) -> Any:
        """DELETE /v1/collections/{id}?confirm=true — hard delete (admin)."""
        from hermes.memory.collections import CollectionNotFoundError

        repo = _collections_repo_or_404()
        if len(collection_id) != 32:
            raise HTTPException(status_code=404, detail="collection_id invalid format")
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="must pass ?confirm=true to delete (safety check)",
            )
        # Pre-check existence
        if await repo.get_collection(collection_id) is None:
            raise HTTPException(status_code=404, detail="collection not found")
        try:
            # Cascade delete. Two protections against bugs:
            #
            # 1. PRAGMA foreign_keys=OFF (verifier 2 MAJOR fix):
            #    We disable FKs on the connection, do the deletes, then
            #    re-enable. If PRAGMA ON itself raises, we don't want to
            #    leave the connection with FK enforcement off (silent
            #    corruption for the rest of the process lifetime). So we
            #    restore in a nested try/except with a CRITICAL log if it
            #    fails.
            #
            # 2. Depth-bounded CTE (verifier 1 BLOCKING fix):
            #    Old code used `LIMIT 1000` which silently left orphan
            #    rows when the subtree had >1000 entries. Now we cap by
            #    depth (matches archive_collection's cap of 20 levels).
            #    Realistic PARA trees are <5 levels deep; 20 is comfortable
            #    headroom and prevents the cap-leaves-orphans failure mode.
            #
            # Safe because:
            #   - We hold Database._write_lock — no concurrent writes.
            #   - DELETE covers the entire subtree (depth-bounded, no
            #     orphans possible by construction).
            #   - FK restore is exception-safe (see point 1).
            async with db._write_lock:
                await db.conn.execute("PRAGMA foreign_keys = OFF")
                try:
                    # 1. Delete bridge rows for parent + all descendants
                    await db.conn.execute(
                        """
                        WITH RECURSIVE descendants(id, depth) AS (
                            SELECT collection_id, 0 FROM vault_collections
                            WHERE collection_id = ?
                            UNION ALL
                            SELECT vc.collection_id, d.depth + 1
                            FROM vault_collections vc
                            JOIN descendants d ON vc.parent_collection_id = d.id
                            WHERE d.depth < 19
                        )
                        DELETE FROM vault_file_collections
                        WHERE collection_id IN (SELECT id FROM descendants)
                        """,
                        (collection_id,),
                    )
                    # 2. Delete all collection rows in the subtree (any order)
                    await db.conn.execute(
                        """
                        WITH RECURSIVE descendants(id, depth) AS (
                            SELECT collection_id, 0 FROM vault_collections
                            WHERE collection_id = ?
                            UNION ALL
                            SELECT vc.collection_id, d.depth + 1
                            FROM vault_collections vc
                            JOIN descendants d ON vc.parent_collection_id = d.id
                            WHERE d.depth < 19
                        )
                        DELETE FROM vault_collections
                        WHERE collection_id IN (SELECT id FROM descendants)
                        """,
                        (collection_id,),
                    )
                finally:
                    try:
                        await db.conn.execute("PRAGMA foreign_keys = ON")
                    except Exception as exc:
                        # PRAGMA restore failed: the connection is in a
                        # dangerous state (FKs off). Log CRITICAL and
                        # re-raise so the next caller doesn't silently
                        # bypass FK enforcement.
                        logger.critical(
                            "collections_delete_pragma_restore_failed",
                            extra={
                                "error": str(exc),
                                "remediation": (
                                    "DB._conn has foreign_keys=OFF. "
                                    "Restart the process or run "
                                    "'PRAGMA foreign_keys=ON' manually."
                                ),
                            },
                        )
                        raise
        except CollectionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="collection not found") from exc
        # Verify gone
        if await repo.get_collection(collection_id) is not None:
            raise HTTPException(status_code=404, detail="collection not found")
        from starlette.responses import Response

        return Response(status_code=204)

    @app.post(
        "/v1/collections/{collection_id}/files",
        status_code=201,
    )
    async def add_file_to_collection_endpoint(
        collection_id: str,
        body: AddFileToCollectionRequest,
    ) -> Any:
        """POST /v1/collections/{id}/files — add a file to a collection."""
        from hermes.memory.collections import CollectionNotFoundError

        repo = _collections_repo_or_404()
        if len(collection_id) != 32:
            raise HTTPException(status_code=404, detail="collection_id invalid format")
        try:
            inserted = await repo.add_file_to_collection(body.file_id, collection_id)
        except CollectionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="collection not found") from exc
        if not inserted:
            raise HTTPException(
                status_code=409,
                detail="file already in collection",
            )
        from datetime import datetime

        added_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "collection_id": collection_id,
            "file_id": body.file_id,
            "added_at": added_at,
        }

    @app.delete("/v1/collections/{collection_id}/files/{file_id}")
    async def remove_file_from_collection_endpoint(
        collection_id: str,
        file_id: str,
    ) -> Any:
        """DELETE /v1/collections/{id}/files/{file_id} — remove bridge row."""
        repo = _collections_repo_or_404()
        if len(collection_id) != 32 or len(file_id) != 32:
            raise HTTPException(status_code=404, detail="invalid id format")
        removed = await repo.remove_file_from_collection(file_id, collection_id)
        if not removed:
            raise HTTPException(
                status_code=404,
                detail="file not in collection",
            )
        from starlette.responses import Response

        return Response(status_code=204)

    @app.get(
        "/v1/collections/{collection_id}/files",
        response_model=ListFilesInCollectionResponse,
    )
    async def list_files_in_collection_endpoint(
        collection_id: str,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Any:
        """GET /v1/collections/{id}/files — list files in collection."""
        from hermes.util.paths import to_posix

        if len(collection_id) != 32:
            raise HTTPException(status_code=404, detail="collection_id invalid format")
        # Paginate via file_id lexicographic ordering
        sql = """
        SELECT vf.file_id, vf.source_path, vf.size_bytes, vf.text_version,
               vf.mtime, vfc.added_at
        FROM vault_file_collections vfc
        JOIN vault_collections vc ON vc.collection_id = vfc.collection_id
        JOIN vault_files vf ON vf.file_id = vfc.file_id
        WHERE vfc.collection_id = ?
          AND vc.archived = 0
          AND vf.orphaned_at IS NULL
          AND vfc.superseded_at IS NULL
        """
        params: list[Any] = [collection_id]
        if cursor is not None:
            sql += " AND vf.file_id > ?"
            params.append(cursor)
        sql += " ORDER BY vf.file_id ASC LIMIT ?"
        params.append(min(limit, 200))

        async with db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        files = []
        for r in rows:
            mtime = r["mtime"]
            # SQLite REAL mtime: format as ISO 8601 UTC. Hermes already uses
            # POSIX timestamps in vault_files.mtime (added_at is TEXT ISO).
            # For consistency, prefer added_at if available, else mtime as epoch.
            try:
                from datetime import datetime

                modified_iso = datetime.fromtimestamp(
                    float(mtime),
                    tz=UTC,
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, TypeError):
                modified_iso = ""
            files.append(
                {
                    "file_id": r["file_id"],
                    "path": to_posix(r["source_path"]),
                    "size_bytes": r["size_bytes"],
                    "text_version": r["text_version"],
                    "modified_at": modified_iso,
                    "added_at": r["added_at"],
                }
            )
        next_cursor = files[-1]["file_id"] if len(files) == limit else None
        return {"files": files, "next_cursor": next_cursor}

    return app


# --- Helpers module-level (visibles para tests) ---


async def _stream_response(
    loop: AgentLoop,
    conversation_id: int,
    user_message_text: str,
    user_message_parts: list[dict] | None,
    file_refs: list[str] | None = None,
    is_persistent: bool = False,
    response_model: str | None = None,
):
    """Async generator que produce SSE desde AgentLoop.run_stream.

    Formato OpenAI SSE:
    - data: {"choices": [{"delta": {"content": "..."}}]}\\n\\n
    - data: {"choices": [{"delta": {"hermes_status": {...}}}]}\\n\\n
    - data: [DONE]\\n\\n  (sentinel de fin)

    La conversacion se archiva al final (incluso si hubo error).

    S9.0: file_refs (lista de file_ids) se persiste en el user message
    y se resuelve en runtime por AgentLoop._resolve_file_refs.
    """
    try:
        async for chunk in loop.run_stream(
            conversation_id,
            user_message_text,
            user_message_parts=user_message_parts,
            file_refs=file_refs,
        ):
            yield chunk.to_sse(model_override=response_model)
        yield "data: [DONE]\n\n"
    except LLMError as exc:
        # Si todos los modelos del chain fallan, emitimos un chunk de
        # error con formato OpenAI y luego [DONE] para que el cliente
        # cierre el stream limpiamente.
        error_payload = {
            "error": {
                "message": f"All LLM models in chain failed: {exc}",
                "type": "api_error",
                "code": "llm_unavailable",
            }
        }
        yield f"data: {_json.dumps(error_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        # Cualquier otra excepcion: emitimos chunk de error generico.
        # Defense in depth: NO dejamos que un error sin manejar cierre
        # el stream abruptamente (cliente OpenAI-compatible espera [DONE]).
        logger.exception("http_api_stream_unexpected_error")
        error_payload = {
            "error": {
                "message": f"Internal server error: {type(exc).__name__}",
                "type": "api_error",
                "code": "internal_error",
            }
        }
        yield f"data: {_json.dumps(error_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        # Sprint 12 (ADR-007): espejo del path no-streaming (linea 881).
        # Las conversaciones persistentes (chat_id != 0) que crea el
        # cliente nativo RikkaHub NO se archivan: el usuario las retoma
        # al abrir la app. archive_stale_conversations tambien las exime.
        if not is_persistent:
            try:
                await loop._db.archive_conversation(conversation_id)
            except Exception:
                logger.exception("http_api_archive_conversation_failed")


def _extract_file_text(content: bytes, content_type: str, filename: str) -> str:
    """Extrae texto de un archivo subido (SÍNCRONO, interno).

    PDFs: pypdf (lazy import — el modulo es pesado).
    text/*: decode utf-8.
    Otros: vacio (text extraction fuera de scope sprint 7.3).

    Sprint 13.0 (S8.6 fix): Esta función es SÍNCRONA. NO la llames
    directamente desde un handler async. Usa `_extract_file_text_async`
    que la envuelve en `asyncio.to_thread` para no bloquear el event
    loop de FastAPI.
    """
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        return _extract_pdf_text(content)
    if content_type.startswith("text/") or filename.lower().endswith((".txt", ".md", ".py")):
        return content.decode("utf-8", errors="replace")
    return ""


def _extract_pdf_text(content: bytes) -> str:
    """Extrae texto de un PDF usando pypdf (SÍNCRONO, interno).

    Lazy import: pypdf es pesado (~5MB) y solo se carga cuando
    se sube un PDF. Si pypdf no esta instalado, retornamos vacio
    con warning (en lugar de fallar el upload).

    Sprint 13.0 (S8.6 fix): Esta función es SÍNCRONA. NO la llames
    directamente desde un handler async. Usa `_extract_pdf_text_async`
    que la envuelve en `asyncio.to_thread` para no bloquear el event
    loop de FastAPI.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf_not_installed_pdf_text_extraction_skipped")
        return ""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        reader = PdfReader(tmp_path)
        pages_text: list[str] = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                # Pagina corrupta o sin texto extraible. Skip y continuar.
                continue
        return "\n".join(pages_text)
    except Exception:
        logger.exception("pdf_text_extraction_failed")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


async def _extract_pdf_text_async(content: bytes) -> str:
    """Versión async de _extract_pdf_text (Sprint 13.0 / S8.6 fix).

    Envuelve la llamada síncrona a pypdf en `asyncio.to_thread` para
    no bloquear el event loop de FastAPI durante la extracción.
    Patrón del S8.4 backup (asyncio.to_thread) pero aplicado a
    pypdf.

    Performance:
    - ANTES (síncrono): PDF 750KB tarda ~90s, /health bloqueado todo el rato
    - DESPUÉS (async): PDF 750KB tarda ~5-8s, /health responde <100ms

    Ver:
    - docs/POSTMORTEM_DB_CORRUPTION.md (S13.0 plan)
    - Vikunja #156 (task tracking)
    """
    return await asyncio.to_thread(_extract_pdf_text, content)


async def _extract_file_text_async(content: bytes, content_type: str, filename: str) -> str:
    """Versión async de _extract_file_text (Sprint 13.0 / S8.6 fix).

    Wrapper async para extraer texto de cualquier tipo de archivo sin
    bloquear el event loop. Despacha a:
    - `_extract_pdf_text_async` para PDFs (envuelve pypdf en thread)
    - decode síncrono para text/* (es rápido, no necesita thread)
    - vacío para otros tipos

    Ver `_extract_file_text` (versión síncrona) para implementación.
    """
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        return await _extract_pdf_text_async(content)
    # text/* y extensiones simples: decode es rápido (<10ms), no necesita thread
    if content_type.startswith("text/") or filename.lower().endswith((".txt", ".md", ".py")):
        return content.decode("utf-8", errors="replace")
    return ""
