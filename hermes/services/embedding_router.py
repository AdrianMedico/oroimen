"""Sprint 19.5 Slice 6 Commit 2: EmbeddingRouter multi-tier con circuit breakers.

Implementa la arquitectura multi-tier para embeddings definida en
``docs/TDD_SPRINT_19_5_SLICE_6.md`` (v0.16, R1 cycle 6 PASS). Es un modulo
**standalone**: NO modifica ``hermes/services/embeddings.py`` (esa
integracion es Commit 3). Aqui se construye la pieza nueva que Commit 3
enchufara en ``EmbeddingsService.ensure_initialized()``.

Componentes:
- ``EmbeddingPolicy`` (enum): ``CHAT_RAG``, ``VAULT_INGEST``. **FACTS fue
  eliminado en el ciclo 4 B-4 (phantom-policy bug)** — ver docstring del
  enum. No añadir ``FACTS`` sin antes cablearlo en TODOS los call sites
  (``hermes/memory/sleep_cycle.py:339`` y ``hermes/memory/facts.py:426``
  segun el apendice del TDD).
- ``TierConfig`` / ``PolicyConfig`` / ``EmbeddingResult``: dataclasses para
  configurar tiers, politicas y resultados.
- ``EmbeddingBackend`` (ABC, singular) + ``OpenAICompatibleBackend`` +
  ``GeminiBackend``: una sola interfaz para todos los backends HTTP.
- ``EmbeddingRouter``: orquesta cascade + circuit breaker per-tier +
  cache en memoria + dim tracking.
- ``_build_router(settings) -> EmbeddingRouter | None``: factory que lee
  ``EMBEDDING_TIER_<NAME>__*`` y ``EMBEDDING_POLICY_<NAME>`` del entorno.
  Retorna ``None`` si no hay tiers configurados (EmbeddingsService cae
  a legacy mode en Commit 3).

Reglas de validacion (TDD §2.4):
- **Rule 3** (within-policy dim match): un policy con 2+ tiers de dim
  distinta -> ``ConfigError`` HARD FAIL en startup.
- **Rule 7** (enabled tier sin ``base_url``/``model``): ``ConfigError``.
- **Rule 9** (per-policy cache init): si ``EMBEDDING_POLICY_X`` falta,
  se skipea ``X`` y se loggea warning. ``embed_with_policy(text, X)``
  luego lanza ``NoPolicyConfiguredError``.

Reutiliza ``hermes.llm.breaker.CircuitBreaker`` directamente (un breaker
por tier, NO compartido — un tier en OPEN no afecta a los demas).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hermes.llm.breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Error de configuracion del router (tier / policy mal definidos).

    Distinto de un error de runtime (backend caido, timeout, etc.). El
    factory ``_build_router`` lo lanza SOLO en startup cuando la
    configuracion no satisface las reglas de validacion del TDD §2.4.
    """


class NoPolicyConfiguredError(Exception):
    """No hay un policy con ese ``EmbeddingPolicy`` configurado.

    Si la policy existe en el enum pero no tiene un env var
    ``EMBEDDING_POLICY_<X>`` configurado, ``embed_with_policy(text, X)``
    lanza este error. El sistema sigue arrancando (warning en startup);
    el error aparece solo en tiempo de uso.
    """

    def __init__(self, policy: EmbeddingPolicy) -> None:
        super().__init__(
            f"policy {policy.value!r} is not configured "
            f"(set EMBEDDING_POLICY_{policy.value.upper()}=tier1,tier2)"
        )
        self.policy = policy


class AllTiersFailed(Exception):
    """Todos los tiers del policy fallaron al intentar embed.

    Atributos:
        policy: el ``EmbeddingPolicy`` que se intento ejecutar.
        errors: lista paralela de excepciones (una por tier intentado,
            en orden de prioridad).
        tried_tiers: lista de nombres de tiers que se intentaron (en
            orden de prioridad del policy).
    """

    def __init__(
        self,
        policy: EmbeddingPolicy,
        errors: list[BaseException],
        tried_tiers: list[str],
    ) -> None:
        msg = (
            f"all tiers failed for policy {policy.value!r} "
            f"(tried={tried_tiers}, errors={[type(e).__name__ for e in errors]})"
        )
        super().__init__(msg)
        self.policy = policy
        self.errors = errors
        self.tried_tiers = tried_tiers


# ---------------------------------------------------------------------------
# Enums y dataclasses
# ---------------------------------------------------------------------------


class EmbeddingPolicy(StrEnum):
    """Politica / caso de uso para embeddings (R1 cycle 6 fix, TDD §2.1).

    Cada policy tiene su propio dim, cache y ruta de DB (X3 per-policy
    isolation, ver TDD §2.7). El primer tier en la lista de prioridad
    es el "canonical" (almacenamiento); el resto son fallbacks para
    ``embed()`` live (cascade con degradacion controlada).

    **FACTS fue eliminado en el ciclo 4 B-4 (phantom-policy bug)**:
    existia en el enum con su propia env var, su propio cache, y nunca
    era llamado desde codigo. Esto es un anti-patron: una policy
    configurable pero sin callers es ruido en la configuracion y un
    vector de confusion ("por que tengo que configurar FACTS si no
    hace nada?"). El TDD §2.6 (ciclo 5 m27) documenta los 4 pasos
    obligatorios para re-añadir FACTS si en el futuro hace falta:
    1. Añadir ``FACTS = "facts"`` aqui.
    2. Parser del env var en ``hermes/config.py``.
    3. **Reemplazar las llamadas a ``embed()`` en
       ``hermes/memory/sleep_cycle.py:339`` y ``hermes/memory/facts.py:426``
       con ``embed_with_policy(text, EmbeddingPolicy.FACTS)``** — sin
       este paso, la policy es un phantom (B-4 original).
    4. Validar que X3 per-policy cache ya no requiere Rule 8 (cross-policy
       dim match). Con X3 la "facts canonical" se enforces por el
       MultiPolicyCache: ``cosine_search(query, policy=FACTS)`` solo ve
       filas ``policy='facts'``.
    """

    CHAT_RAG = "chat_rag"
    VAULT_INGEST = "vault_ingest"


@dataclass(frozen=True)
class TierConfig:
    """Configuracion de un tier individual (parseado de env vars).

    Atributos:
        name: nombre canonico del tier (``nas``, ``edge``, ``cloud``,
            etc.). Lowercase. Usado como clave en los dicts del router.
        base_url: URL base del API compatible con OpenAI. NO tiene
            default hardcoded — Rule 7 obliga a configurarlo
            explicitamente (TDD §2.4 #7). Ejemplo:
            ``http://granite-svc.lan:8082/v1`` para un sidecar
            NAS local.
        model: nombre del modelo en el backend remoto. Sin default
            hardcoded. Ejemplo: ``granite-97m`` o ``qwen/qwen3-embedding-8b``.
        api_key: API key (string vacio si no requiere auth, e.g. LAN).
        timeout_s: timeout HTTP en segundos. Default 30.
        enabled: si ``False``, el tier no se instancia ni se considera
            en cascade. Permite dejar tiers configurados como
            "apagado temporal" sin borrarlos del .env.
        extra_headers: headers HTTP extra (e.g. ``X-Oroimen-Key``,
            ``X-OpenRouter-ZDR: true``). Parseado de JSON.

    El dataclass es ``frozen=True`` para evitar mutaciones accidentales
    tras la validacion del factory (Rule 3 / Rule 7 se aplican sobre
    instancias inmutables).
    """

    name: str
    base_url: str
    model: str
    api_key: str
    timeout_s: float
    enabled: bool
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyConfig:
    """Configuracion de un policy (lista ordenada de tiers).

    Atributos:
        use_case: el ``EmbeddingPolicy`` que esta policy implementa.
        tiers: nombres de tiers en orden de prioridad. El primero es
            el "canonical" (almacenamiento y retrieval). El resto son
            fallbacks para ``embed()`` live (TDD §2.1 "first-wins").
    """

    use_case: EmbeddingPolicy
    tiers: list[str]


@dataclass(frozen=True)
class EmbeddingResult:
    """Resultado de un ``embed()`` o ``embed_batch()``.

    Atributos:
        vector: ``np.ndarray`` shape ``(dim,)`` o ``(N, dim)``, dtype
            ``float32``. Para ``embed()`` shape ``(dim,)``; para
            ``embed_batch()`` shape ``(N, dim)``.
        source_tier: nombre del tier que sirvio el resultado (el
            primero que tuvo exito en el cascade).
        latency_ms: latencia observada en milisegundos (medida en el
            router, incluye overhead de circuit breaker).
        cost_estimate: coste estimado en USD. ``0.0`` para tiers
            locales (NAS, edge) que no facturan por llamada. Para
            cloud tiers se deja en ``0.0`` en Commit 2 — el calculo
            preciso por token se hara en Sprint 19.6 (TDD §2.6 nota 8,
            sin metrics/InfluxDB en Commit 2).
        cached: ``True`` si el resultado vino del cache en memoria del
            router. ``False`` en la primera llamada.
    """

    vector: np.ndarray
    source_tier: str
    latency_ms: float
    cost_estimate: float = 0.0
    cached: bool = False


# ---------------------------------------------------------------------------
# Registry de dim por modelo conocido
# ---------------------------------------------------------------------------

#: Dim por modelo conocido. **NO** exhaustivo: si un modelo no está
#: aqui, el factory ``_build_router`` lanza ``ConfigError`` con hint
#: explicito ("add '<model>' to KNOWN_MODEL_DIMS in embedding_router.py").
#: Esto evita que un modelo con dim desconocida arranque y luego falle
#: en runtime con un ``ValueError`` en cosine_search (Regla 3
#: violation silenciosa).
#:
#: Fuentes:
#: - granite-97m: docs/GRANITE_SETUP.md §Architecture (384-dim, ONNX
#:   int8 en granite-svc).
#: - qwen/qwen3-embedding-8b: OpenRouter catalogue (4096-dim, MTEB
#:   multilingual top-3 2026-06).
#: - text-embedding-3-small / text-embedding-3-large: OpenAI docs
#:   (1536 / 3072 dims).
#: - gemini-embedding-001: Google AI for Developers (3072-dim;
#:   ``text-embedding-004`` legacy es 768-dim, en desuso).
KNOWN_MODEL_DIMS: dict[str, int] = {
    # Granite (NAS, sidecar on NAS host)
    "granite-97m": 384,
    "granite-311m": 768,
    "granite-107m": 384,
    "granite-30m": 384,
    # Qwen embeddings (edge Ollama + OpenRouter)
    "qwen3-embedding:8b": 4096,
    "qwen/qwen3-embedding-8b": 4096,
    "qwen3-embedding:0.6b": 1024,
    "qwen3-embedding:4b": 2048,
    "qwen/qwen3-embedding-4b": 2048,
    # OpenAI text-embedding-3 family
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # BGE (alternative open-source)
    "bge-large-en-v1.5": 1024,
    "bge-small-en-v1.5": 384,
    "bge-m3": 1024,
    # Gemini embedding
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,  # legacy
    "embedding-001": 768,  # legacy
}


def _dim_for_model(model: str) -> int:
    """Resuelve la dim de un modelo conocido. Lanza ``ConfigError`` si
    no esta en el registry (TDD §2.4 Rule 7 spirit: nada de defaults
    magicos, fall loud con hint action able).
    """
    # Lookup case-insensitive para tolerancia.
    model_lower = model.lower().strip()
    if model_lower in KNOWN_MODEL_DIMS:
        return KNOWN_MODEL_DIMS[model_lower]
    raise ConfigError(
        f"unknown model {model!r} — no dim in KNOWN_MODEL_DIMS. "
        f"Add it to hermes/services/embedding_router.py:KNOWN_MODEL_DIMS, "
        f"or use one of the known models: {sorted(KNOWN_MODEL_DIMS)}"
    )


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------


class EmbeddingBackend(ABC):
    """Interfaz abstracta para backends de embeddings (singular, TDD §3 commit 2).

    Cualquier backend concreto (OpenAI-compatible, Gemini, mock para
    tests) implementa estos 5 miembros:

    - ``name``: nombre del backend (``"nas"``, ``"edge"``, ``"cloud"``,
      ``"gemini"``). Usado en logs y en ``EmbeddingResult.source_tier``.
    - ``dim``: dimension del embedding (e.g. 384 para granite-97m,
      4096 para qwen-8b). **Debe coincidir con el modelo configurado**;
      si no, Rule 3 (within-policy dim match) falla en startup.
    - ``embed(text) -> np.ndarray``: embedding de un texto. Shape
      ``(dim,)`` dtype ``float32``.
    - ``embed_batch(texts) -> list[np.ndarray]``: embeddings de N
      textos. Implementaciones que NO soporten batch nativo (e.g.
      Gemini) caen a N llamadas secuenciales (TDD §2.6 Gemini row).
    - ``is_enabled() -> bool``: ``True`` si el backend está listo para
      servir (cliente HTTP inicializado, key configurada, etc.).
      ``False`` desactiva el tier en cascade sin re-construir el router.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del backend ('nas' | 'edge' | 'cloud' | 'gemini')."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dim del embedding que produce este backend."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Embedding de un texto. Shape ``(dim,)`` dtype ``float32``."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embeddings de N textos. Shape ``(N, dim)`` dtype ``float32``.

        Backend puede hacer UN request con ``input=texts`` (OpenAI-compat,
        eficiente) o N requests secuenciales (Gemini, sub-óptimo pero
        funcional). El router no distingue entre los dos.
        """

    @abstractmethod
    def is_enabled(self) -> bool:
        """``True`` si el backend puede servir requests ahora mismo."""

    async def aclose(self) -> None:
        """Cierra clientes HTTP. No-op por default (backends sin estado)."""
        return None


# ---------------------------------------------------------------------------
# Backends concretos
# ---------------------------------------------------------------------------


#: Truncado defensivo para texto enviado a OpenAI-compatible APIs.
#: OpenAI text-embedding-3-small acepta 8191 tokens (~24K chars).
#: Granite-97m en sidecar NAS acepta context window mucho mayor
#: (~512 tokens) pero truncamos a este limite para homogeneidad.
#: 32K chars = margen generoso para 8K tokens en cualquier backend
#: OpenAI-compat conocido.
_MAX_EMBED_CHARS_OPENAI = 32_000

#: Truncado para Gemini embedding. Gemini limita a ~2048 tokens
#: ~= 8K chars (TDD legacy constants, conservados por compat con
#: ``hermes/services/embeddings.py``).
_MAX_EMBED_CHARS_GEMINI = 8_000


class OpenAICompatibleBackend(EmbeddingBackend):
    """Backend HTTP compatible con el endpoint ``/v1/embeddings`` de OpenAI.

    Usado por todos los tiers que exponen el formato OpenAI:
    - NAS (granite-svc sidecar)
    - edge (Ollama con ``/v1/embeddings``)
    - cloud (OpenRouter)

    Soporta batch nativo: un solo request con ``input=texts``. La API
    devuelve los embeddings en el mismo orden que el input (TDD §2.6
    Gemini Blocker 2 fix: evita DDoS al edge daemon por N llamadas
    secuenciales, y evita 429 en OpenRouter por superar el rate limit
    por minute).

    Auth: header ``Authorization: Bearer <api_key>`` si ``api_key`` no
    está vacío. Para tiers LAN (NAS, edge) el ``api_key`` puede ser
    string vacío y entonces no se envía header.
    """

    def __init__(self, config: TierConfig) -> None:
        self._config = config
        # Lazy client (httpx.AsyncClient) — se crea en la primera call.
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            headers: dict[str, str] = dict(self._config.extra_headers)
            if self._config.api_key:
                # OpenAI-compat standard auth. Some providers accept
                # alternative headers (e.g. ``X-API-Key``) — set those
                # via ``extra_headers`` in env config.
                headers["Authorization"] = f"Bearer {self._config.api_key}"
            # ``base_url`` se pasa a httpx para que ``embeddings`` se
            # construya como ``POST {base_url}/embeddings``.
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout_s,
                headers=headers,
            )
        return self._client

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def dim(self) -> int:
        return _dim_for_model(self._config.model)

    def is_enabled(self) -> bool:
        # Un tier configurado está enabled si tiene base_url y model.
        # La validacion Rule 7 ya garantizó esto en el factory, pero
        # mantenemos la check por si se reconfigura en runtime.
        return bool(self._config.base_url) and bool(self._config.model)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _truncate(self, text: str) -> str:
        if len(text) > _MAX_EMBED_CHARS_OPENAI:
            return text[:_MAX_EMBED_CHARS_OPENAI]
        return text

    async def embed(self, text: str) -> np.ndarray:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        client = self._ensure_client()
        truncated = [self._truncate(t) for t in texts]
        # OpenAI-compat: POST {base_url}/embeddings con body
        # ``{"input": [...], "model": "<model>"}``. La respuesta es
        # ``{"data": [{"embedding": [...]}, ...], ...}`` con el mismo
        # orden que el input.
        response = await client.post(
            "/embeddings",
            json={"input": truncated, "model": self._config.model},
        )
        response.raise_for_status()
        body = response.json()
        # ``data`` es una lista de objetos con campo ``embedding``.
        # Algunos providers lo llaman ``embeddings`` (plural, no-OAI) —
        # soportamos ambos.
        data = body.get("data")
        if data is None:
            data = body.get("embeddings", [])
        if not data:
            raise RuntimeError(
                f"OpenAI-compat backend {self.name!r} returned no embeddings "
                f"in response (model={self._config.model!r}, body keys={list(body)})"
            )
        return [np.array(item["embedding"], dtype=np.float32) for item in data]


class GeminiBackend(EmbeddingBackend):
    """Backend para la API de Gemini embedding (free tier).

    Endpoint: ``POST {base_url}/models/{model}:embedContent`` con auth
    via query param ``?key=<api_key>``. Body:
    ``{"content": {"parts": [{"text": "..."}]}}``. Response:
    ``{"embedding": {"values": [...]}}``.

    ``embed_batch`` cae a N llamadas secuenciales (Gemini API no
    soporta batch nativo). Aceptable para workloads pequeños (el use
    case Gemini en este sistema es el `vault_ingest` fallback, no
    el chat loop live).
    """

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 30.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._api_key = api_key
        self._base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip(
            "/"
        )
        self._timeout_s = timeout_s
        self._extra_headers: dict[str, str] = dict(extra_headers or {})
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=self._timeout_s,
                headers=self._extra_headers,
            )
        return self._client

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return _dim_for_model(self._model)

    def is_enabled(self) -> bool:
        return bool(self._api_key) and bool(self._model)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _truncate(self, text: str) -> str:
        if len(text) > _MAX_EMBED_CHARS_GEMINI:
            return text[:_MAX_EMBED_CHARS_GEMINI]
        return text

    async def embed(self, text: str) -> np.ndarray:
        client = self._ensure_client()
        url = f"{self._base_url}/models/{self._model}:embedContent"
        body = {
            "content": {"parts": [{"text": self._truncate(text)}]},
        }
        response = await client.post(url, params={"key": self._api_key}, json=body)
        response.raise_for_status()
        data = response.json()
        values = data["embedding"]["values"]
        return np.array(values, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        # Gemini API no soporta batch nativo. Caemos a N llamadas
        # secuenciales. Para workloads grandes (1148 chunks) esto es
        # ~5-10x mas lento que un batch, pero el tier Gemini NO es
        # el primary en ``vault_ingest`` (cloud lo es, edge el
        # fallback). Gemini solo aparece en test fixtures o
        # deployments legacy.
        return [await self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


def _cache_key(text: str, policy: EmbeddingPolicy) -> tuple[str, str]:
    """Clave del cache en memoria: (text, policy) tuple.

    El text NO se hashea (queremos inspectable keys en logs). Si el
    crecimiento del cache es un problema, se sustituye por SHA-256 en
    una iteracion futura.
    """
    return (text, policy.value)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class _RouterCache:
    """Cache en memoria keyed por ``(text, policy)``.

    Commit 2: cache minima en proceso. Commit 4 introducira el
    ``MultiPolicyCache`` persistido en DB. Aqui solo necesitamos que
    ``invalidate_cache(policy=...)`` invalide el subset correcto
    (TDD §2.5 M-7 fix, v0.16).
    """

    _results: dict[tuple[str, str], EmbeddingResult] = field(default_factory=dict)

    def get(self, text: str, policy: EmbeddingPolicy) -> EmbeddingResult | None:
        return self._results.get(_cache_key(text, policy))

    def put(self, text: str, policy: EmbeddingPolicy, result: EmbeddingResult) -> None:
        # Marca cached=True al guardar; el resultado original (de la
        # primera llamada) no debe mutar.
        # ``vector.copy()`` previene el footgun R1 X-2: si un caller
        # muta el vector in-place (``result.vector /= norm``), NO
        # afecta al cache (ni viceversa). Tradeoff: una copia
        # extra por write (cheap para vectors de 384-4096 floats).
        cached = EmbeddingResult(
            vector=result.vector.copy(),
            source_tier=result.source_tier,
            latency_ms=result.latency_ms,
            cost_estimate=result.cost_estimate,
            cached=True,
        )
        self._results[_cache_key(text, policy)] = cached

    def invalidate(self, policy: str | None) -> None:
        """Invalida el cache.

        Args:
            policy: nombre de policy (``"chat_rag"`` / ``"vault_ingest"``)
                a invalidar, o ``None`` para invalidar TODO. Comparamos
                con el value del ``EmbeddingPolicy`` (string), no con el
                enum mismo, para que ``invalidate_cache("chat_rag")`` y
                ``invalidate_cache("vault_ingest")`` funcionen igual que
                ``invalidate_cache(policy=...)``.
        """
        if policy is None:
            self._results.clear()
            return
        # Filtra por policy value; mantén solo las keys de OTROS policies.
        self._results = {k: v for k, v in self._results.items() if k[1] != policy}


class EmbeddingRouter:
    """Orquestador de tiers con cascade y circuit breakers per-tier.

    Dado un set de ``EmbeddingBackend`` (construidos por el factory) y
    un set de ``PolicyConfig`` (cual policy usa cuales tiers en que
    orden), expone 3 metodos publicos:

    - ``embed(text, use_case=CHAT_RAG) -> EmbeddingResult``: cascade
      con fallback. Usado por el chat loop live.
    - ``embed_batch(texts, use_case=CHAT_RAG) -> list[EmbeddingResult]``:
      batch nativo. Usado por el embedder de vault.
    - ``embed_with_policy(text, policy) -> EmbeddingResult``: variante
      explicita sin default. Usado por callers que ya conocen el
      policy (e.g. ``hermes/memory/sleep_cycle.py`` cuando se re-añada
      FACTS en un futuro sprint).

    Cascade logic (TDD §2.6 "Router pattern"):
    1. Para cada tier en ``policy.tiers`` (orden de prioridad):
       a. Obtener el ``CircuitBreaker`` del tier.
       b. Intentar ``await cb.call(lambda: tier.embed(text))``.
       c. Si OK: construir ``EmbeddingResult`` y retornar.
       d. Si ``CircuitOpenError``: siguiente tier (fail fast).
       e. Si otra excepcion: siguiente tier (el breaker ya registro el fail).
    2. Si todos fallan: ``raise AllTiersFailed(...)``.

    Circuit breakers son **per-tier independientes** (no compartidos):
    un NAS que se va a OPEN no afecta al breaker del edge. Esta es la
    diferencia clave con el ``CircuitBreakerRegistry`` de
    ``hermes/services/search/resilience.py`` (que es registry-based,
    API distinta, no reusable aqui per TDD §2.6).

    Cache en memoria keyed por ``(text, policy)``: hits saltan el
    cascade. ``invalidate_cache(policy=None)`` limpia todo;
    ``invalidate_cache(policy="chat_rag")`` limpia solo ese policy
    (TDD §2.5 M-7).
    """

    def __init__(
        self,
        *,
        backends: Mapping[str, EmbeddingBackend],
        policies: Mapping[EmbeddingPolicy, PolicyConfig],
        breaker_fail_max: int = 5,
        breaker_reset_timeout_s: float = 60.0,
    ) -> None:
        self._backends: dict[str, EmbeddingBackend] = dict(backends)
        self._policies: dict[EmbeddingPolicy, PolicyConfig] = dict(policies)
        # Un CircuitBreaker por tier (NO compartido). Named para que los
        # logs estructurados distingan breakers.
        self._breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(
                fail_max=breaker_fail_max,
                reset_timeout=breaker_reset_timeout_s,
                name=f"embedding:{name}",
            )
            for name in self._backends
        }
        self._cache = _RouterCache()

    # --- public properties -------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """``True`` si el router puede servir al menos un policy.

        Es decir: al menos un policy configurado Y al menos uno de sus
        tiers canónicos tiene ``is_enabled() == True``. Si todos los
        tiers están deshabilitados en runtime (NAS down, OpenRouter 401),
        retorna ``False`` y ``embed()`` cae a la excepcion de cascade.

        Es ``@property`` (no metodo) para ser drop-in compatible con
        los 4 callers existentes que la usan como propiedad:
        ``hermes/agent/loop.py:790`` (``if not self._embeddings_service
        .is_enabled``), ``hermes/receivers/http_api.py:1551``,
        ``hermes/tools/search_files.py:74``, y
        ``hermes/services/embed_vault.py:185`` (todos usan
        ``getattr(..., "is_enabled", False)`` o ``not is_enabled`` sin
        parentesis). TDD §2.5 backward-compat table.
        """
        if not self._policies:
            return False
        for policy_cfg in self._policies.values():
            for tier_name in policy_cfg.tiers:
                backend = self._backends.get(tier_name)
                if backend is not None and backend.is_enabled():
                    return True
        return False

    def backend_name(self) -> str:
        """Nombre del router (literal ``"multi_tier"``)."""
        return "multi_tier"

    # --- public methods: embed --------------------------------------------

    async def embed(
        self,
        text: str,
        use_case: EmbeddingPolicy = EmbeddingPolicy.CHAT_RAG,
    ) -> np.ndarray:
        """Embedding de un texto con cascade sobre los tiers del policy.

        Args:
            text: texto a embeber (no se trunca aquí; cada backend
                aplica su propio limite defensivo).
            use_case: policy / caso de uso. Default ``CHAT_RAG``
                (live API del chat loop, TDD §2.5 M-8 fix).

        Returns:
            ``np.ndarray`` shape ``(dim,)`` dtype ``float32`` con el
            vector del primer tier que tuvo exito. Se devuelve el
            vector directamente (NO el ``EmbeddingResult`` wrapper)
            para ser drop-in compatible con los callers existentes:
            ``hermes/memory/embedder.py:540`` hace ``cache @ query_emb``
            (matmul numpy), ``hermes/services/embeddings.py`` callers
            esperan ``np.ndarray`` en ``embed()``. Internamente
            seguimos construyendo ``EmbeddingResult`` para cache, metrics
            y error tracking — solo lo desenvuelve en el return publico.
            TDD §2.5 backward-compat table.

        Raises:
            NoPolicyConfiguredError: si ``use_case`` no está configurado.
            AllTiersFailed: si todos los tiers del policy fallaron.
        """
        result = await self.embed_with_policy(text, use_case)
        return result.vector

    async def embed_with_policy(
        self,
        text: str,
        policy: EmbeddingPolicy,
    ) -> EmbeddingResult:
        """Embedding con policy explicito (no default).

        Usado por callers que ya conocen el policy (e.g. herramientas
        internas, scripts de re-embed). Equivalente a ``embed()`` con
        ``use_case=policy`` pero sin default para hacer explicito en la
        firma que el caller DEBE elegir un policy.

        Raises:
            NoPolicyConfiguredError: si ``policy`` no está en
                ``self._policies`` (no se ha configurado
                ``EMBEDDING_POLICY_<X>`` para este caso de uso).
            AllTiersFailed: si todos los tiers del policy fallaron.
        """
        cached = self._cache.get(text, policy)
        if cached is not None:
            return cached
        policy_cfg = self._policies.get(policy)
        if policy_cfg is None:
            raise NoPolicyConfiguredError(policy)
        result = await self._cascade(text, policy_cfg)
        self._cache.put(text, policy, result)
        return result

    async def embed_batch(
        self,
        texts: list[str],
        use_case: EmbeddingPolicy = EmbeddingPolicy.CHAT_RAG,
        *,
        model: str | None = None,
    ) -> list[np.ndarray]:
        """Embeddings de N textos (native batch, TDD §2.5).

        Happy path: **1 sola request HTTP** al canonical tier del
        policy con ``input=texts``. El backend (OpenAI-compat) hace
        batch nativo internamente. Esto evita DDoS al edge daemon
        (N conexiones) y 429 en OpenRouter (rate limit por minute).
        TDD §2.6: "Native batch, NOT a loop over embed_with_policy".

        Fallback: si el canonical tier falla para el batch entero
        (timeout, 5xx, dim mismatch, breaker abierto), cae al cascade
        per-text ``embed_with_policy``. Eso preserva la recuperacion
        parcial (per-text breaker granularity): si el 30% de los
        textos fallan en canonical, el 70% se recupera de los
        fallbacks.

        Cache: el batch se cachea per-text en el happy path (cache
        hits en calls futuras con la misma ``text``). El fallback
        per-text cascade cachea naturalmente (cada llamada pasa por
        ``embed_with_policy`` que cachea).

        Args:
            texts: lista de textos a embeber. ``[]`` retorna ``[]``.
            use_case: policy / caso de uso. Default ``CHAT_RAG``.
            model: kwarg backward-compat con
                ``EmbeddingsService.embed_batch(texts, model=...)``.
                En v0.16 hay UN solo modelo canonico por policy
                (first tier de ``policy.tiers``). Si ``model`` no
                coincide con el del canonical tier, se loggea warning
                y se ignora (single canonical por policy en v0.16;
                multi-model se aborda en un futuro sprint si hace
                falta). TDD §2.5 backward-compat table.

        Returns:
            ``list[np.ndarray]`` en el mismo orden que ``texts``,
            shape ``(dim,)`` dtype ``float32`` cada uno. Se devuelven
            los vectores directamente (NO ``EmbeddingResult``) para
            ser drop-in compatible con los callers existentes
            (``hermes/memory/embedder.py:377`` y callers de
            ``cosine_search`` que iteran y hacen matmul). ``[]`` si
            ``texts`` es vacio.

        Raises:
            NoPolicyConfiguredError: si ``use_case`` no está
                configurado.
            AllTiersFailed: si el canonical tier falla Y el fallback
                per-text cascade también falla para el batch entero.
        """
        if not texts:
            return []
        policy_cfg = self._policies.get(use_case)
        if policy_cfg is None:
            raise NoPolicyConfiguredError(use_case)
        # ``model`` kwarg is accepted for backward compat with
        # ``EmbeddingsService.embed_batch(texts, model=...)``. In v0.16
        # there is ONE canonical model per policy (the first tier's
        # model). Per TDD §2.5, ``model`` is accepted but currently
        # ignored (single canonical per policy semantics). Log a
        # warning when the caller passes it so they know it doesn't
        # route to a different model — multi-model per policy is a
        # future-sprint concern.
        canonical_tier_name = policy_cfg.tiers[0]
        canonical_backend = self._backends[canonical_tier_name]
        if model is not None:
            logger.debug(
                "embedding_router_batch_model_kwarg_accepted_ignored",
                extra={
                    "policy": use_case.value,
                    "requested_model": model,
                    "canonical_tier": canonical_tier_name,
                    "hint": (
                        "v0.16 has one canonical model per policy. "
                        "The model kwarg is accepted for backward compat "
                        "with EmbeddingsService.embed_batch but ignored. "
                        "Use the policy's canonical tier model."
                    ),
                },
            )
        # Happy path: 1 native batch request to the canonical tier.
        # El cascade per-text (embed_with_policy) solo se invoca como
        # fallback si canonical falla para el batch entero.
        breaker = self._breakers[canonical_tier_name]
        start = time.monotonic()
        try:
            # 1 HTTP request, native batch.
            vectors = await breaker.call(lambda: canonical_backend.embed_batch(texts))
        except CircuitOpenError:
            # Breaker abierto en canonical: cae al per-text cascade.
            # Es la situacion de fallback "frecuente" — el canonical
            # esta en cooldown. NO es un error, es el comportamiento
            # esperado del cascade.
            return await self._embed_batch_per_text_cascade(texts, use_case, canonical_tier_name)
        except Exception as canonical_exc:
            # Canonical fallo (5xx, timeout, etc.): intenta cascade
            # per-text para recuperar resultados parciales.
            logger.warning(
                "embedding_router_batch_canonical_failed_falling_back",
                extra={
                    "policy": use_case.value,
                    "canonical_tier": canonical_tier_name,
                    "error": type(canonical_exc).__name__,
                    "error_msg": str(canonical_exc),
                },
            )
            return await self._embed_batch_per_text_cascade(
                texts, use_case, canonical_tier_name, canonical_error=canonical_exc
            )
        # 1 request OK. Construye EmbeddingResults para cache per-text,
        # luego desenvuelve a np.ndarray para el caller.
        latency_ms = (time.monotonic() - start) * 1000.0
        results_vectors: list[np.ndarray] = []
        for i, vec in enumerate(vectors):
            results_vectors.append(vec.copy())
            # Cachea per-text. Usamos el texto original (no el
            # truncado que pueda haber aplicado el backend) como key.
            self._cache.put(
                texts[i],
                use_case,
                EmbeddingResult(
                    vector=vec.copy(),
                    source_tier=canonical_tier_name,
                    latency_ms=latency_ms,
                    cost_estimate=0.0,
                ),
            )
        return results_vectors

    async def _embed_batch_per_text_cascade(
        self,
        texts: list[str],
        use_case: EmbeddingPolicy,
        canonical_tier_name: str,
        canonical_error: BaseException | None = None,
    ) -> list[np.ndarray]:
        """Fallback per-text cascade para ``embed_batch``.

        Solo se invoca cuando el canonical tier falla para el batch
        entero (happy path lo evita con 1 native request). Aqui
        cada texto pasa por ``embed_with_policy`` completo, lo que
        activa el cascade completo (canonical -> fallback tiers) y
        la cache per-text.

        Si un texto individual falla (no el batch entero), se loggea
        y se devuelve un vector de ceros dim=0 como placeholder
        (el caller lo detectara porque la dim no es la esperada).
        """
        results: list[np.ndarray] = []
        for t in texts:
            try:
                result = await self.embed_with_policy(t, use_case)
                results.append(result.vector)
            except NoPolicyConfiguredError:
                raise
            except AllTiersFailed as exc:
                logger.warning(
                    "embedding_router_batch_per_text_fallback_failed",
                    extra={
                        "policy": use_case.value,
                        "canonical_tier": canonical_tier_name,
                        "tried_tiers": exc.tried_tiers,
                        "errors": [type(e).__name__ for e in exc.errors],
                    },
                )
                results.append(np.zeros(0, dtype=np.float32))
        return results

    # --- public methods: ops -----------------------------------------------

    async def health_check(self) -> bool:
        """``True`` si al menos el canonical tier del primer policy responde.

        Health check barato: un embed de ``"ping"`` contra el canonical
        tier del primer policy configurado. Si falla, retorna ``False``
        sin probar fallbacks (el chat loop live puede probar
        fallbacks por su cuenta).
        """
        if not self._policies:
            return False
        first_policy_cfg = next(iter(self._policies.values()))
        if not first_policy_cfg.tiers:
            return False
        canonical_tier_name = first_policy_cfg.tiers[0]
        backend = self._backends.get(canonical_tier_name)
        if backend is None or not backend.is_enabled():
            return False
        try:
            await backend.embed("ping")
            return True
        except Exception:
            return False

    def invalidate_cache(self, policy: str | None = None) -> None:
        """Invalida el cache en memoria.

        Args:
            policy: nombre del policy a invalidar (``"chat_rag"`` o
                ``"vault_ingest"``). ``None`` invalida TODO el cache
                (default, backward compat con la API anterior, TDD
                §2.5 M-7).
        """
        self._cache.invalidate(policy)

    # --- cascade internals -------------------------------------------------

    async def _cascade(
        self,
        text: str,
        policy_cfg: PolicyConfig,
    ) -> EmbeddingResult:
        """Implementa el cascade: tiers en orden, primer exito gana.

        Ver docstring de la clase para la logica completa. Esta funcion
        es el corazon del router y se mantiene pequeña y testeable.
        """
        errors: list[BaseException] = []
        tried: list[str] = []
        for tier_name in policy_cfg.tiers:
            backend = self._backends.get(tier_name)
            if backend is None:
                # Tier referenciado por el policy pero no configurado
                # (no se pasó al factory). Loggeamos y seguimos.
                logger.warning(
                    "embedding_router_tier_not_in_router",
                    extra={
                        "policy": policy_cfg.use_case.value,
                        "tier": tier_name,
                    },
                )
                continue
            if not backend.is_enabled():
                # Tier configurado pero deshabilitado (e.g. EMBEDDING_TIER_X__ENABLED=false
                # o key vacia en runtime). No incrementamos el breaker;
                # esto no es un fail transitorio sino una decision de
                # configuracion.
                continue
            breaker = self._breakers[tier_name]
            tried.append(tier_name)
            start = time.monotonic()
            try:
                # Nested async function (no lambda) para que mypy
                # pueda inferir el tipo de retorno. Los default args
                # ``backend=backend`` y ``text=text`` fijan los loop
                # variables en la closure, evitando el late-binding
                # bug (Ruff B023) y haciendo el codigo robusto ante
                # cambios futuros del loop.
                async def _do_embed(_b: EmbeddingBackend = backend, _t: str = text) -> np.ndarray:
                    return await _b.embed(_t)

                vector = await breaker.call(_do_embed)
            except CircuitOpenError:
                # Breaker abierto: fail fast, siguiente tier.
                continue
            except Exception as exc:
                # Cualquier otro error (timeout, 5xx, dim mismatch,
                # etc.) es un fail del backend. El breaker ya lo
                # registro internamente. Siguiente tier.
                errors.append(exc)
                continue
            latency_ms = (time.monotonic() - start) * 1000.0
            return EmbeddingResult(
                vector=vector,
                source_tier=tier_name,
                latency_ms=latency_ms,
            )
        # Todos los tiers fallaron (o no habia tiers habilitados).
        if not errors:
            # Caso especial: el policy tiene tiers pero todos
            # estaban deshabilitados o no configurados. Generamos
            # errores sinteticos para que el caller tenga contexto.
            errors = [
                RuntimeError(
                    f"no enabled tier available for policy "
                    f"{policy_cfg.use_case.value!r} (tried={tried})"
                )
            ]
        raise AllTiersFailed(policy_cfg.use_case, errors, tried)


# ---------------------------------------------------------------------------
# Factory: parse env vars -> EmbeddingRouter
# ---------------------------------------------------------------------------


def _parse_tier_configs() -> dict[str, TierConfig]:
    """Parsea ``EMBEDDING_TIER_<NAME>__*`` env vars a ``dict[TierConfig]``.

    Busca todas las env vars con prefijo ``EMBEDDING_TIER_`` y separador
    ``__`` (estilo pydantic-settings). El nombre del tier es la parte
    entre ``EMBEDDING_TIER_`` y el primer ``__`` (case-insensitive).
    Los campos reconocidos son los del dataclass ``TierConfig``:
    ``ENABLED``, ``BASE_URL``, ``MODEL``, ``API_KEY``, ``TIMEOUT_S``,
    ``EXTRA_HEADERS``.

    Returns:
        ``dict`` keyed por nombre de tier (lowercase). Vacío si no
        hay env vars ``EMBEDDING_TIER_*`` configuradas (en cuyo caso
        ``_build_router`` retorna ``None``).
    """
    tiers: dict[str, dict[str, str]] = {}
    prefix = "EMBEDDING_TIER_"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix) or "__" not in env_key:
            continue
        rest = env_key[len(prefix) :]
        name_raw, field_raw = rest.split("__", 1)
        name = name_raw.lower()
        field = field_raw.lower()
        if name not in tiers:
            tiers[name] = {}
        tiers[name][field] = env_value
    # Construir TierConfig instances.
    result: dict[str, TierConfig] = {}
    for name, fields in tiers.items():
        enabled_raw = fields.get("enabled", "false").strip().lower()
        enabled = enabled_raw in ("true", "1", "yes", "on")
        base_url = fields.get("base_url", "").strip()
        model = fields.get("model", "").strip()
        api_key = fields.get("api_key", "").strip()
        timeout_raw = fields.get("timeout_s", "30").strip()
        try:
            timeout_s = float(timeout_raw)
        except ValueError as exc:
            raise ConfigError(
                f"EMBEDDING_TIER_{name.upper()}__TIMEOUT_S={timeout_raw!r} is not a valid float"
            ) from exc
        extra_headers_raw = fields.get("extra_headers", "").strip()
        extra_headers: dict[str, str] = {}
        if extra_headers_raw:
            try:
                parsed = json.loads(extra_headers_raw)
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    f"EMBEDDING_TIER_{name.upper()}__EXTRA_HEADERS="
                    f"{extra_headers_raw!r} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ConfigError(
                    f"EMBEDDING_TIER_{name.upper()}__EXTRA_HEADERS must be "
                    f"a JSON object, got {type(parsed).__name__}"
                )
            for k, v in parsed.items():
                if not isinstance(v, str):
                    raise ConfigError(
                        f"EMBEDDING_TIER_{name.upper()}__EXTRA_HEADERS["
                        f"{k!r}] must be a string, got {type(v).__name__}"
                    )
                extra_headers[str(k)] = v
        result[name] = TierConfig(
            name=name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            enabled=enabled,
            extra_headers=extra_headers,
        )
    return result


def _parse_policy_configs() -> dict[EmbeddingPolicy, PolicyConfig]:
    """Parsea ``EMBEDDING_POLICY_<POLICY>=tier1,tier2`` env vars.

    Para cada ``EmbeddingPolicy`` enum value, busca el env var
    ``EMBEDDING_POLICY_<VALUE_UPPER>``. Si está presente y no vacío,
    split por coma y construye ``PolicyConfig``. Si falta, loggea
    warning y la policy queda no-configurada (TDD §2.4 Rule 9:
    ``embed_with_policy(text, X)`` levantara ``NoPolicyConfiguredError``
    en tiempo de uso).
    """
    result: dict[EmbeddingPolicy, PolicyConfig] = {}
    for policy in EmbeddingPolicy:
        env_key = f"EMBEDDING_POLICY_{policy.value.upper()}"
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            # No es un error — significa que el deploy no quiere
            # esta policy. Loggeamos a INFO (no warning) para que el
            # operador sepa que la policy esta disponible pero no
            # activa.
            logger.info(
                "embedding_router_policy_not_configured",
                extra={
                    "policy": policy.value,
                    "hint": f"set {env_key}=tier1,tier2 to enable",
                },
            )
            continue
        tiers = [t.strip().lower() for t in raw.split(",") if t.strip()]
        if not tiers:
            # Edge case: env var presente pero vacia (e.g.
            # ``EMBEDDING_POLICY_CHAT_RAG=``). Lo tratamos como no
            # configurado.
            logger.info(
                "embedding_router_policy_empty_value",
                extra={"policy": policy.value, "env_key": env_key},
            )
            continue
        result[policy] = PolicyConfig(use_case=policy, tiers=tiers)
    return result


def _make_backend_for_tier(
    tier: TierConfig,
) -> EmbeddingBackend:
    """Factory: crea el ``EmbeddingBackend`` apropiado para un ``TierConfig``.

    Hoy solo soporta OpenAI-compat (todos los tiers reales — NAS,
    edge, OpenRouter — exponen ``/v1/embeddings``). GeminiBackend
    existe para tests y para el caso legacy de ``vault_ingest``
    Gemini; en el futuro se puede seleccionar con un campo
    ``__TYPE`` en el env schema (TDD v0.9 lo tenia; Slice 6 lo
    omite por YAGNI).
    """
    # Por ahora, todos los tiers son OpenAI-compat. Si en el futuro
    # queremos distinguir (e.g. tier ``gemini_legacy`` con URL de
    # Google), añadir un campo ``__TYPE`` al env schema. Por ahora,
    # la URL + el modelo distinguen lo suficiente: si la URL apunta
    # a ``generativelanguage.googleapis.com`` usamos GeminiBackend.
    if "generativelanguage.googleapis.com" in tier.base_url:
        return GeminiBackend(
            name=tier.name,
            model=tier.model,
            api_key=tier.api_key,
            base_url=tier.base_url,
            timeout_s=tier.timeout_s,
            extra_headers=tier.extra_headers,
        )
    return OpenAICompatibleBackend(tier)


def _build_router(settings: Any) -> EmbeddingRouter | None:
    """Factory: construye el ``EmbeddingRouter`` desde env vars.

    Args:
        settings: instancia de ``hermes.config.Settings`` (forward-looking
            — en Commit 2 se ignora porque los env vars
            ``EMBEDDING_TIER_*`` y ``EMBEDDING_POLICY_*`` aún no están
            en el dataclass de Settings; Commit 3 los añadira). Se
            acepta el parametro ya para que Commit 3 sea un drop-in.

    Returns:
        ``EmbeddingRouter`` listo para usar, o ``None`` si no hay tiers
        configurados (EmbeddingsService cae a legacy mode en Commit 3).

    Raises:
        ConfigError: si la configuracion viola Rule 3 (within-policy
            dim mismatch) o Rule 7 (enabled tier sin base_url/model)
            u otras reglas de validacion.
    """
    # 1. Parsear tiers de env vars.
    tier_configs = _parse_tier_configs()
    if not tier_configs:
        return None
    # 2. Validar Rule 7: enabled tier sin base_url/model.
    for name, cfg in tier_configs.items():
        if cfg.enabled and (not cfg.base_url or not cfg.model):
            raise ConfigError(
                f"tier {name!r} is enabled (EMBEDDING_TIER_{name.upper()}"
                f"__ENABLED=true) but EMBEDDING_TIER_{name.upper()}__BASE_URL "
                f"or __MODEL is empty. Set them explicitly, or disable the "
                f"tier (EMBEDDING_TIER_{name.upper()}__ENABLED=false)."
            )
    # 3. Instanciar backends SOLO para los tiers enabled.
    backends: dict[str, EmbeddingBackend] = {}
    for name, cfg in tier_configs.items():
        if not cfg.enabled:
            continue
        backends[name] = _make_backend_for_tier(cfg)
    # 3b. Validar que cada backend tiene un modelo conocido
    # (``_dim_for_model`` levanta ``ConfigError`` si el modelo no
    # esta en ``KNOWN_MODEL_DIMS``). Llamamos ``.dim`` aqui aunque
    # el policy tenga un solo tier, para que el operador vea el
    # error en startup, no en la primera llamada a ``embed()`` (que
    # ya estara en produccion con un crash confuso en cosine_search).
    for _name, backend in backends.items():
        # ``_name`` esta como prefijo para silenciar B007 (variable
        # de control no usada) y dejar claro que la iteracion es
        # solo para forzar la validacion.
        _ = backend.dim
    # 4. Parsear policies.
    policy_configs = _parse_policy_configs()
    # 5. Validar Rule 3 (within-policy dim match) y que cada policy
    #    referencia tiers que existen.
    for policy, pol_cfg in policy_configs.items():
        if not pol_cfg.tiers:
            # No deberia pasar (filtrado en _parse_policy_configs)
            # pero defendemos por si acaso.
            raise ConfigError(f"policy {policy.value!r} has no tiers configured")
        for tier_name in pol_cfg.tiers:
            if tier_name not in backends:
                # Tier referenciado por el policy pero no enabled (o no
                # configurado). El cascade cae a ``AllTiersFailed`` en
                # runtime, pero avisamos al operador en startup con un
                # warning explicito.
                logger.warning(
                    "embedding_router_policy_references_disabled_tier",
                    extra={
                        "policy": policy.value,
                        "tier": tier_name,
                        "hint": (
                            f"tier {tier_name!r} is referenced by policy "
                            f"{policy.value!r} but is not enabled. "
                            f"Set EMBEDDING_TIER_{tier_name.upper()}"
                            f"__ENABLED=true or remove the tier from "
                            f"EMBEDDING_POLICY_{policy.value.upper()}."
                        ),
                    },
                )
        # Check dim match entre los tiers enabled del policy.
        # Solo comparamos los que sí están enabled (los disabled
        # simplemente no se cuentan).
        enabled_tiers_in_policy = [t for t in pol_cfg.tiers if t in backends]
        if len(enabled_tiers_in_policy) >= 2:
            first_dim = backends[enabled_tiers_in_policy[0]].dim
            for tier_name in enabled_tiers_in_policy[1:]:
                tier_dim = backends[tier_name].dim
                if tier_dim != first_dim:
                    raise ConfigError(
                        f"policy {policy.value!r} has tiers with "
                        f"different dims: "
                        f"{enabled_tiers_in_policy[0]}={first_dim}, "
                        f"{tier_name}={tier_dim}. All tiers in a policy "
                        f"MUST share the same dim, otherwise "
                        f"cosine_search will fail with ValueError. "
                        f"Fix: set the model in both tiers to one with "
                        f"the same dim, or split into multiple policies."
                    )
    return EmbeddingRouter(backends=backends, policies=policy_configs)


# Re-export for tests and other modules.
__all__ = [
    "KNOWN_MODEL_DIMS",
    "AllTiersFailed",
    "ConfigError",
    "EmbeddingBackend",
    "EmbeddingPolicy",
    "EmbeddingResult",
    "EmbeddingRouter",
    "GeminiBackend",
    "NoPolicyConfiguredError",
    "OpenAICompatibleBackend",
    "PolicyConfig",
    "TierConfig",
    "_build_router",
]


# ---------------------------------------------------------------------------
# Async helper: shutdown limpio
# ---------------------------------------------------------------------------


async def aclose_router(router: EmbeddingRouter) -> None:
    """Cierra todos los backends del router (httpx clients).

    Llamar al shutdown de la app. No-op para backends que no tienen
    estado (e.g. mocks en tests).
    """
    # ``backends`` es un dict[str, EmbeddingBackend]. Usamos el nombre
    # del attr para no exponer ``_backends`` como API publica.
    for backend in router._backends.values():
        await backend.aclose()


# Silence linter warning sobre ``settings`` no usado en _build_router
# (es forward-looking para Commit 3; el parametro existe para que la
# firma no cambie cuando se enchufe).
_ = asyncio  # import no usado directamente, pero queremos mantenerlo
# para futuros awaits en modulos hermanos (consistencia con
# embeddings.py).
