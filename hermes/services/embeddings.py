"""Sprint 9.1 (renamed in 9.2): Embeddings service para RAG via OpenRouter.

Arquitectura de backends intercambiables (Sprint 9.1.1, refactored en 9.2):
- **OpenRouterBackend** (default, ZDR): usa `openrouter_api_key`.
  Header `X-OpenRouter-ZDR: true` forzado en cada peticion para
  Zero Data Retention. Default: `qwen/qwen3-embedding-8b` (4096-dim,
  $0.01/M tokens, mejor precio/context/multilingual).
- **GeminiBackend** (opt-in, NO ZDR): usa `gemini_api_key`. Override
  via `EMBEDDING_PROVIDER=gemini`. **NO usar para datos sensibles**:
  el free tier de Gemini entrena con los datos enviados.

Auto-detect: si `openrouter_api_key` está set, usa OpenRouter. Si
no, fallback a Gemini si tiene key. Si ambos fallan, RAG se desactiva
graciosamente (EmbeddingsService cae en fallback NoOp, log warning).

Componentes:
- EmbeddingsCache: numpy array en memoria con los embeddings de la
  library. Invalidación explícita al añadir/borrar embeddings.
- EmbeddingsService: orquesta embed() (API call al backend activo),
  embed_and_store() (truncado a 24K chars), cosine_search() (cache +
  numpy).

P1-1 v1.2 fix (Gemini 3.5 Thinking ronda 8): np.frombuffer() retorna
array read-only. El cache hace .copy() explícito para evitar errores
de asignación in-place en normalizaciones futuras.

Fallback NoOp: si no hay key válido para ningún backend, el service
opera en modo disabled (cosine_search retorna [], embed_and_store
loggea warning). Esto permite que el resto del sistema funcione sin
RAG.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import httpx

    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.services.embedding_router import EmbeddingPolicy, EmbeddingRouter

logger = logging.getLogger(__name__)

# Límite defensivo para embed_and_store (TDD S9 §2.3 v1.3 Gemini fix).
# Default: Qwen 3 Embedding 8B (32K context = ~128K chars), pero
# truncamos a 32K chars (~8K tokens) por seguridad con cualquier
# modelo OpenAI-compatible. Si el user setea openai/text-embedding-3-small
# (8K context), 32K chars es seguro (8K tokens completos). Para
# Gemini text-embedding-004 (2K tokens = ~8K chars), truncamos a 8K.
MAX_EMBED_CHARS_GEMINI = 8_000
MAX_EMBED_CHARS_OPENAI = 32_000

# Dim por modelo. Qwen 3 Embedding 8B es 4096-dim (default). OpenAI
# text-embedding-3-small es 1536. Gemini text-embedding-004 es 768.
# Si el user cambia el modelo, debe actualizar manualmente el cache
# (los embeddings viejos son incompatibles con la nueva dim).
GEMINI_DEFAULT_DIM = 3072  # gemini-embedding-001 (was 768 for text-embedding-004)
OPENAI_DEFAULT_DIM = 4096  # qwen/qwen3-embedding-8b (default)

# Backward compat alias (tests antiguos usaban EMBEDDING_DIM_DEFAULT).
EMBEDDING_DIM_DEFAULT = OPENAI_DEFAULT_DIM
# Backward compat alias (MAX_EMBED_CHARS era el antiguo default OpenAI).
MAX_EMBED_CHARS = MAX_EMBED_CHARS_OPENAI

# Sprint 19.5 Slice 6 Commit 3: integration con EmbeddingRouter
# (hermes.services.embedding_router). Import lazy para evitar cycle:
# ``embedding_router`` importa ``hermes.llm.breaker`` y ``numpy``; este
# modulo es un leaf. El factory ``_build_router`` retorna ``None`` si
# no hay tiers configurados (legacy mode) o un ``EmbeddingRouter``
# listo. ``aclose_router`` cierra los clientes httpx de los backends.
# Imports absolute (no relativo) para que la lazy import funcione desde
# tests que importan ``hermes.services.embeddings`` directamente sin
# que el paquete padre se cargue primero.
from hermes.services.embedding_router import (  # noqa: E402
    AllTiersFailed,
    EmbeddingPolicy,
    EmbeddingResult,
    aclose_router,
)
from hermes.services.embedding_router import (  # noqa: E402
    _build_router as _build_embedding_router,
)


class EmbeddingsCache:
    """Cache in-memory de embeddings (numpy array 2D).

    Sprint 9.1 P1-1 v1.2 fix: sin cache, cada query RAG carga 60MB
    de embeddings desde SQLite (10K docs x 6KB). Con cache, las
    queries son ~50ms vs ~200-500ms sin cache. 4-10x mejora.

    Decisión arquitectónica: el cache es write-through pero lazy-load.
    - Al primer `get_all()`, lee de DB y guarda el array numpy.
    - `invalidate()` fuerza re-load en la próxima query.
    - `add_file_embedding` no actualiza el cache automáticamente
      (para no complicar concurrencia); el caller debe invalidar
      tras insertar.

    **Importante** (P0 fix Gemini 3.5 v1.2 round 8): np.frombuffer()
    por defecto retorna un array read-only que apunta al buffer
    subyacente. Sin .copy(), cualquier normalización in-place
    (e.g. `emb /= np.linalg.norm(emb)`) lanza ValueError. Por eso
    hacemos .copy() explícito al construir el cache.
    """

    def __init__(self) -> None:
        self._cache: np.ndarray | None = None  # shape: (N, dim) float32
        self._file_ids: list[str] = []
        self._loaded: bool = False
        self._loaded_policy: str | None = None
        self._dim: int | None = None  # dim del cache actual

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def get_all(
        self, db: Database, policy: str | None = None
    ) -> tuple[np.ndarray, list[str]]:
        """Retorna (numpy_array, file_ids). Lazy load desde DB.

        Args:
            db: Database instance para leer embeddings.

        Returns:
            (cache, file_ids) donde cache es np.ndarray shape (N, dim)
            y file_ids es la lista paralela de file_ids. Vacío si no
            hay embeddings.
        """
        if not self._loaded or self._loaded_policy != policy:
            await self._load_from_db(db, policy)
        assert self._cache is not None
        return self._cache, self._file_ids

    async def _load_from_db(self, db: Database, policy: str | None) -> None:
        """Carga embeddings desde DB. Idempotente.

        Detecta la dim del primer embedding no-vacio y la usa como
        referencia. Si hay embeddings con dims distintas, falla
        ruidosamente (defense in depth).
        """
        rows = await db.get_all_embeddings(policy=policy)
        if not rows:
            # Cache vacío: usamos dim 0 placeholder; se rellenará
            # en la primera inserción.
            self._cache = np.empty((0, 0), dtype=np.float32)
            self._file_ids = []
            self._loaded = True
            self._loaded_policy = policy
            self._dim = None
            return
        self._file_ids = [r[0] for r in rows]
        # v1.2 fix Gemini 3.5: .copy() obligatorio tras np.frombuffer.
        # Sin esto, los arrays son read-only y colapsan con cualquier
        # operación in-place (e.g. normalización).
        embeddings = [np.frombuffer(r[1], dtype=np.float32).copy() for r in rows]
        # Detectar dim (asumimos uniforme; si no, falla ruidosamente)
        first_dim = embeddings[0].shape[0]
        for i, emb in enumerate(embeddings):
            if emb.shape[0] != first_dim:
                raise RuntimeError(
                    f"EmbeddingsCache dim inconsistente: "
                    f"file_id={self._file_ids[i]} tiene dim={emb.shape[0]}, "
                    f"esperado {first_dim}. Probable bug: se mezclaron "
                    f"embeddings de modelos distintos en la misma DB."
                )
        self._dim = first_dim
        self._loaded_policy = policy
        self._cache = np.array(embeddings, dtype=np.float32)
        # Validación defensiva: shape debe ser 2D con N filas
        if self._cache.ndim != 2 or self._cache.shape[0] != len(self._file_ids):
            raise RuntimeError(
                f"EmbeddingsCache inconsistente: shape={self._cache.shape}, "
                f"file_ids={len(self._file_ids)}"
            )
        logger.info(
            "embeddings_cache_loaded",
            extra={
                "count": len(self._file_ids),
                "bytes": self._cache.nbytes,
                "dim": first_dim,
            },
        )

    def invalidate(self) -> None:
        """Fuerza re-load en la próxima query.

        Llamar tras:
        - add_file_embedding (nuevo file indexado)
        - delete_file (cascade limpia el embedding)
        - delete_file_embedding (re-embed)
        """
        self._cache = None
        self._file_ids = []
        self._loaded = False
        self._loaded_policy = None
        self._dim = None


class EmbeddingsBackend(ABC):
    """Interfaz abstracta para backends de embeddings.

    Cada backend encapsula la logica especifica del provider:
    - Gemini: httpx contra generativelanguage.googleapis.com
    - OpenAI / OpenRouter: SDK openai (OpenAI-compatible)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del provider para logging ('gemini' | 'openai')."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dim del embedding (768 para Gemini, 1536 para OpenAI)."""

    @property
    @abstractmethod
    def max_embed_chars(self) -> int:
        """Maximo de chars a enviar a la API (truncado defensivo)."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Genera el embedding de un texto."""


class GeminiBackend(EmbeddingsBackend):
    """Backend Gemini Embedding (free tier).

    API: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent
    Auth: API key en query param `?key=...` o header `x-goog-api-key`
    Response: `{"embedding": {"values": [0.1, 0.2, ...]}}`
    Modelos: text-embedding-004 (768-dim, default), embedding-001 (legacy)
    """

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._model = model
        # base_url override para testing (default: Google oficial)
        self._base_url = base_url or "https://generativelanguage.googleapis.com/v1beta"
        # Cliente httpx lazy (se crea en la primera llamada)
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def dim(self) -> int:
        # text-embedding-004 es 768-dim. Otros modelos pueden variar.
        return GEMINI_DEFAULT_DIM

    @property
    def max_embed_chars(self) -> int:
        # Gemini limita a ~2048 tokens ≈ 8K chars
        return MAX_EMBED_CHARS_GEMINI

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, text: str) -> np.ndarray:
        client = self._ensure_client()
        url = f"{self._base_url}/models/{self._model}:embedContent"
        # Gemini usa query param `key` para auth
        params = {"key": self._api_key}
        body = {
            "content": {
                "parts": [{"text": text}],
            },
        }
        response = await client.post(url, params=params, json=body)
        response.raise_for_status()
        data = response.json()
        # Response shape: {"embedding": {"values": [0.1, 0.2, ...]}}
        values = data["embedding"]["values"]
        return np.array(values, dtype=np.float32)


class OpenRouterBackend(EmbeddingsBackend):
    """Backend OpenAI / OpenRouter (SDK openai, OpenAI-compatible).

    Por defecto apunta a OpenAI direct. Override openrouter_api_base
    a https://openrouter.ai/api/v1 para usar OpenRouter (con naming
    prefijado: "openai/text-embedding-3-small", etc.).

    Zero Data Retention (ZDR): si el base_url apunta a OpenRouter,
    añadimos el header `X-OpenRouter-ZDR: true` a cada peticion
    para forzar zero data retention. OpenAI direct ya es ZDR por
    defecto para usuarios con ZDR habilitado en su cuenta.
    """

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        # Detectar si el base_url es OpenRouter (para ZDR header).
        # Matching simple por substring para evitar dependencia de urlparse.
        self._is_openrouter = "openrouter" in base_url.lower()
        self._client: Any = None
        try:
            import openai

            # openai SDK >=1.50: extra_headers via default_headers kwarg
            extra_headers: dict[str, str] = {}
            if self._is_openrouter:
                # Header ZDR: garantiza que OpenRouter no retiene los
                # datos del request (no logging, no training, no cache).
                # Ver https://openrouter.ai/docs/zero-data-retention
                extra_headers["X-OpenRouter-ZDR"] = "true"
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=extra_headers if extra_headers else None,
            )
        except ImportError:
            logger.warning(
                "openai_backend_disabled_no_package",
                extra={"hint": "pip install openai>=1.50.0"},
            )

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dim(self) -> int:
        return OPENAI_DEFAULT_DIM

    @property
    def max_embed_chars(self) -> int:
        return MAX_EMBED_CHARS_OPENAI

    @property
    def is_zdr(self) -> bool:
        """True si este backend garantiza Zero Data Retention.

        - OpenRouter: ZDR forzado via header X-OpenRouter-ZDR=true.
        - OpenAI direct: depende de la cuenta del user (ZDR por defecto
          para algunos tiers; no verificable desde el cliente).
        - Otros (e.g. Ollama, vLLM): ZDR por definicion (modelo local).
        """
        return self._is_openrouter  # otros casos: no podemos verificar

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()

    async def embed(self, text: str) -> np.ndarray:
        if self._client is None:
            raise RuntimeError("OpenAI client no inicializado (package missing)")
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        embedding_list = response.data[0].embedding
        return np.array(embedding_list, dtype=np.float32)

    async def embed_batch(self, texts: list[str], *, model: str | None = None) -> list[np.ndarray]:
        """OpenAI API: embeddings.create con `input` siendo una list."""
        if self._client is None:
            raise RuntimeError("OpenAI client no inicializado (package missing)")
        if not texts:
            return []
        # Truncado defensivo: el limite estricto de OpenAI es 8191 tokens
        # (~24K chars). Si un chunk > 24K, lo truncamos (pierde info pero
        # evita 400). Coherente con embed_and_store.
        truncated = [
            (t[: self.max_embed_chars] if len(t) > self.max_embed_chars else t) for t in texts
        ]
        response = await self._client.embeddings.create(
            model=model or self._model,
            input=truncated,
        )
        # OpenAI devuelve response.data como lista en el mismo orden que input.
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]


class EmbeddingsService:
    """Orquesta el backend activo + cache + cosine search.

    Modos de operación:
    1. **enabled** (key válido para Gemini o OpenAI): todas las ops
       funcionan. Llama a la API, persiste en DB, mantiene cache.
    2. **disabled** (ningún key disponible): RAG desactivado.
       - `embed()` retorna array de ceros
       - `embed_and_store()` loggea warning y no-op
       - `cosine_search()` retorna []

    Auto-detect: Gemini (free) > OpenAI (paid).
    Override manual con EMBEDDING_PROVIDER env var.
    """

    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db
        self._cache = EmbeddingsCache()
        self._backend: EmbeddingsBackend | None = None
        # Sprint 19.5 Slice 6 Commit 3: cuando se configura el router
        # multi-tier (env vars ``EMBEDDING_TIER_*__ENABLED=true``), el
        # service usa ``self._router`` en vez de ``self._backend``. Son
        # mutuamente excluyentes: el service tiene _router O _backend,
        # nunca ambos. ``_initialized`` evita reinicializaciones (init
        # es idempotente, per TDD §3 commit 3 lazy init).
        self._router: EmbeddingRouter | None = None
        self._initialized: bool = False
        # Legacy init lock; conservado por retrocompat con los tests
        # existentes que inspeccionan ``self._init_lock``.
        self._init_lock = False  # simple flag; init no es concurrente

    def _resolve_openai_key(self) -> str:
        """Resuelve la key de OpenAI/OpenRouter con backward compat.

        Prioridad: `openrouter_api_key` (S9.1+) > `openai_api_key`
        (legacy, pre-S9.1). Esto preserva el comportamiento de
        deployments existentes que aun usan `OPENAI_API_KEY` y los
        guia hacia el nuevo nombre sin romperse.

        Returns:
            la key efectiva (string vacio si ninguna está seteada).
        """
        return self._settings.openrouter_api_key.strip() or self._settings.openai_api_key.strip()

    def _resolve_openai_base(self) -> str:
        """Resuelve el base_url de OpenAI/OpenRouter con backward compat.

        Prioridad: `openrouter_api_base` (S9.1+) > `openai_api_base`
        (legacy, pre-S9.1). Si ambas vacias, retorna el default
        OpenRouter (ZDR).

        Returns:
            el base_url efectivo.
        """
        return self._settings.openrouter_api_base.strip() or self._settings.openai_api_base.strip()

    def _tier_multi_mode_enabled(self) -> bool:
        """Sprint 19.5 Slice 6 Commit 3: True si CUALQUIER tier
        ``EMBEDDING_TIER_<NAME>__ENABLED=true`` está seteado.

        Si True, ``ensure_initialized`` intenta construir el
        ``EmbeddingRouter``. Si False, cae al legacy single-backend
        (OpenRouter o Gemini). Es la primera discriminacion entre
        los dos modos — antes de gastar trabajo en ``_build_router``
        (que parsea env vars y puede lanzar ``ConfigError``).

        v0.16 R1 cycle 10 fix (c574379+1): leemos ``os.environ``
        directamente como source-of-truth defensivo. Settings se
        carga UNA vez al startup (en ``__main__.py``); si un
        operator cambia env vars DESPUES (reload, test mid-run,
        container restart parcial), ``self._settings`` se queda
        con el valor stale pero ``os.environ`` tiene el real. El
        factory ``_build_embedding_router`` también lee
        ``os.environ`` directamente, asi que este check es
        consistente con el factory.

        Acepta ``true|1|yes|on`` (case-insensitive) para
        ``ENABLED``, igual que el factory.
        """
        env = os.environ
        for name in ("NAS", "EDGE", "CLOUD"):
            val = env.get(f"EMBEDDING_TIER_{name}__ENABLED", "").lower().strip()
            if val in ("true", "1", "yes", "on"):
                return True
        return False

    def _try_build_router(self) -> EmbeddingRouter | None:
        """Sprint 19.5 Slice 6 Commit 3: factory wrapper para
        ``_build_embedding_router`` con deteccion del modo legacy.

        Returns:
            ``EmbeddingRouter`` listo, o ``None`` si no hay tiers
            configurados (legacy mode).

        Raises:
            ``ConfigError`` (de ``_build_embedding_router``): si la
                configuracion viola Rule 3 o Rule 7. NO capturamos:
                TDD §2.4 "fail loudly at startup". El caller
                (``__main__.py`` o el que sea) tiene la opcion de
                continuar en legacy o abortar.
        """
        if not self._tier_multi_mode_enabled():
            return None
        return _build_embedding_router(self._settings)

    def _init_backend_sync(self) -> None:
        """Inicializa el backend SIN health check (sync, sin red).

        Default (privacy-first): OpenAI/OpenRouter con ZDR. Gemini
        solo si EMBEDDING_PROVIDER=gemini explicitamente (NO ZDR,
        free tier entrena con tus datos).
        """
        provider = self._settings.embedding_provider.lower().strip()
        openai_key = self._resolve_openai_key()
        # Sprint 19.6+ Phase 5: gemini_api_key es opcional (str | None,
        # default None). Usar `or ""` para que `.strip()` no explote
        # si el campo es None. Patron consistente con `opencode_go_api_key`
        # y `telegram_bot_token` (Sprint 11 ADR-004).
        gemini_key = (self._settings.gemini_api_key or "").strip()

        # Aceptar 'openai' (legacy) y 'openrouter' (S9.1+) como sinonimos.
        if provider in ("openai", "openrouter") and openai_key:
            self._backend = OpenRouterBackend(
                api_key=openai_key,
                model=self._settings.embedding_model,
                base_url=self._resolve_openai_base(),
            )
            return
        if provider == "gemini":
            if not gemini_key:
                logger.warning(
                    "embeddings_provider_gemini_but_no_key",
                    extra={"hint": "Set GEMINI_API_KEY"},
                )
                return
            # Privacy warning: Gemini free tier NO es ZDR.
            logger.warning(
                "embeddings_using_gemini_no_zdr",
                extra={
                    "provider": "gemini",
                    "model": self._settings.gemini_embedding_model,
                    "warning": (
                        "Gemini free tier puede usar tus datos para "
                        "entrenar. Si tu library tiene contenido sensible, "
                        "usa EMBEDDING_PROVIDER=openai (OpenRouter con ZDR)."
                    ),
                },
            )
            self._backend = GeminiBackend(
                api_key=gemini_key,
                model=self._settings.gemini_embedding_model,
            )
            return
        # Default sin provider especifico: intentar OpenAI (ZDR)
        if openai_key:
            self._backend = OpenRouterBackend(
                api_key=openai_key,
                model=self._settings.embedding_model,
                base_url=self._resolve_openai_base(),
            )
            return
        self._backend = None

    async def ensure_initialized(self) -> None:
        """Inicializa el backend con health check + fallback.

        Default: OpenAI/OpenRouter (ZDR). Si primary falla, fallback
        al otro provider (Gemini o viceversa). Si ambos fallan, RAG
        disabled.

        Privacy: si el user setea EMBEDDING_PROVIDER=gemini explicitamente,
        forzamos Gemini aunque tenga OpenRouter key disponible.

        Sprint 19.5 Slice 6 Commit 3: si el operator configura
        ``EMBEDDING_TIER_*__ENABLED=true``, se instancia el
        ``EmbeddingRouter`` (``_build_embedding_router``) en vez del
        legacy single-backend. El router toma precedencia absoluta:
        si el factory retorna un router, NO se instancia el backend
        legacy. ``_initialized`` evita reinicializaciones (TDD §3
        commit 3 lazy init pattern + integration MAJOR M3).
        """
        if self._initialized:
            return
        if self._init_lock:
            return
        # --- Multi-tier mode (Sprint 19.5) ---
        # Si CUALQUIER tier está enabled, se intenta construir el
        # router. Si el factory retorna None (caso raro: tier enabled
        # pero invalido), cae al legacy. El factory puede lanzar
        # ``ConfigError`` (Rule 3 / Rule 7 violation) — dejamos que
        # propague (TDD §2.4: "fail loudly at startup").
        router = self._try_build_router()
        if router is not None:
            self._router = router
            # Reporta el canonical tier de cada policy configurado.
            policies_info: dict[str, str] = {}
            for policy_name, policy_cfg in router._policies.items():
                canonical = policy_cfg.tiers[0] if policy_cfg.tiers else "<none>"
                policies_info[policy_name.value] = canonical
            logger.info(
                "embeddings_router_initialized",
                extra={
                    "tiers": list(router._backends.keys()),
                    "policies": policies_info,
                    "hint": "multi-tier mode active (Sprint 19.5)",
                },
            )
            self._initialized = True
            self._init_lock = True
            return
        # --- Legacy mode (pre-Sprint-19.5) ---
        if self._backend is None:
            self._init_backend_sync()
        if self._backend is None:
            logger.warning(
                "embeddings_disabled_no_backend",
                extra={
                    "hint": (
                        "Set openrouter_api_key (OpenRouter con ZDR) o "
                        "GEMINI_API_KEY (NO ZDR) para activar RAG."
                    ),
                },
            )
            self._init_lock = True
            self._initialized = True
            return
        # Health check del backend primario
        primary_name = self._backend.name
        if await self._health_check_backend(self._backend):
            logger.info(
                "embeddings_backend_initialized",
                extra={
                    "provider": primary_name,
                    "dim": self._backend.dim,
                    "zdr": self._is_zdr_active(),
                },
            )
            self._init_lock = True
            self._initialized = True
            return
        # Fallback al otro provider
        logger.warning(
            "embeddings_primary_health_check_failed",
            extra={
                "primary": primary_name,
                "hint": "Intentando fallback al otro provider.",
            },
        )
        fallback_backend = self._build_fallback_backend(primary_name)
        if fallback_backend is not None and await self._health_check_backend(fallback_backend):
            self._backend = fallback_backend
            logger.info(
                "embeddings_backend_initialized",
                extra={
                    "provider": self._backend.name,
                    "dim": self._backend.dim,
                    "via_fallback": True,
                    "zdr": self._is_zdr_active(),
                },
            )
            self._init_lock = True
            self._initialized = True
            return
        # Ambos fallan
        logger.error(
            "embeddings_all_backends_failed",
            extra={
                "primary": primary_name,
                "hint": "RAG desactivado. Verifica keys y conectividad.",
            },
        )
        self._backend = None
        self._init_lock = True
        self._initialized = True

    def _is_zdr_active(self) -> bool:
        """True si el backend actual garantiza ZDR (solo OpenRouter)."""
        if isinstance(self._backend, OpenRouterBackend):
            return self._backend.is_zdr
        # Gemini: NUNCA ZDR en free tier
        return False

    def _build_fallback_backend(self, primary_name: str) -> EmbeddingsBackend | None:
        """Construye el backend de fallback (el que NO es primary)."""
        # Sprint 19.6+ Phase 5: gemini_api_key es opcional (str | None,
        # default None). Usar `or ""` para que `.strip()` no explote
        # si el campo es None. Patron consistente con `opencode_go_api_key`
        # y `telegram_bot_token` (Sprint 11 ADR-004).
        gemini_key = (self._settings.gemini_api_key or "").strip()
        openai_key = self._resolve_openai_key()
        if primary_name == "gemini" and openai_key:
            return OpenRouterBackend(
                api_key=openai_key,
                model=self._settings.embedding_model,
                base_url=self._resolve_openai_base(),
            )
        if primary_name == "openai" and gemini_key:
            return GeminiBackend(
                api_key=gemini_key,
                model=self._settings.gemini_embedding_model,
            )
        return None

    async def _init_backend(self) -> None:
        """Inicializa el backend segun EMBEDDING_PROVIDER y keys.

        Prioridad con fallback:
        1. Si EMBEDDING_PROVIDER='gemini' o (='auto' Y gemini_api_key
           está set): intenta GeminiBackend
        2. Si Gemini falla el health check (API caída, auth, rate limit):
           fallback automatico a OpenRouterBackend (si tiene key)
        3. Si EMBEDDING_PROVIDER='openai' o (='auto' Y openrouter_api_key
           está set): OpenRouterBackend directo
        4. Else: disabled

        Por qué fallback en startup (no per-request): los embeddings
        de providers distintos tienen dim diferente (Gemini 768 vs
        OpenAI 1536). Mezclarlos en el cache rompe cosine_search. La
        unica manera robusta es elegir UN provider por instancia.
        """
        provider = self._settings.embedding_provider.lower().strip()
        # Sprint 19.6+ Phase 5: gemini_api_key es opcional (str | None,
        # default None). Usar `or ""` para que `.strip()` no explote
        # si el campo es None. Patron consistente con `opencode_go_api_key`
        # y `telegram_bot_token` (Sprint 11 ADR-004).
        gemini_key = (self._settings.gemini_api_key or "").strip()
        openai_key = self._resolve_openai_key()

        # Intentar Gemini si el provider es 'gemini' o 'auto' con key
        if provider in ("gemini", "auto") and gemini_key:
            backend: EmbeddingsBackend = GeminiBackend(
                api_key=gemini_key,
                model=self._settings.gemini_embedding_model,
            )
            if await self._health_check_backend(backend):
                self._backend = backend
                logger.info(
                    "embeddings_backend_initialized",
                    extra={
                        "provider": "gemini",
                        "model": self._settings.gemini_embedding_model,
                        "dim": backend.dim,
                    },
                )
                return
            logger.warning(
                "embeddings_gemini_health_check_failed",
                extra={
                    "provider": "gemini",
                    "hint": "API caida, auth invalida, o rate limit. "
                    "Fallback a OpenRouter/OpenAI si está disponible.",
                },
            )
        # Intentar OpenAI/OpenRouter (fallback o provider directo).
        # Aceptar 'openai' (legacy) y 'openrouter' (S9.1+) como sinonimos.
        if provider in ("openai", "openrouter", "auto") and openai_key:
            backend = OpenRouterBackend(
                api_key=openai_key,
                model=self._settings.embedding_model,
                base_url=self._resolve_openai_base(),
            )
            if backend._client is not None and await self._health_check_backend(backend):
                self._backend = backend
                logger.info(
                    "embeddings_backend_initialized",
                    extra={
                        "provider": "openai",
                        "model": self._settings.embedding_model,
                        "base_url": self._resolve_openai_base(),
                        "dim": backend.dim,
                        "zdr": "openrouter" in self._resolve_openai_base(),
                    },
                )
                return
            logger.warning(
                "embeddings_openai_health_check_failed",
                extra={"provider": "openai"},
            )
        # Si llegamos aqui, ningun backend está disponible
        logger.warning(
            "embeddings_disabled_no_backend",
            extra={
                "provider": provider,
                "hint": (
                    "Set GEMINI_API_KEY (free) o openrouter_api_key (paid) "
                    "para activar RAG. Default: Gemini si está disponible."
                ),
            },
        )
        self._backend = None

    async def _health_check_backend(self, backend: EmbeddingsBackend) -> bool:
        """Health check barato: embed de 'ping' (1 token).

        Usado en _init_backend para decidir fallback. Si falla, NO
        usamos ese backend.
        """
        try:
            await backend.embed("ping")
            return True
        except Exception as exc:
            logger.debug(
                "embeddings_backend_health_check_failed",
                extra={
                    "provider": backend.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:200],
                },
            )
            return False

    @property
    def is_enabled(self) -> bool:
        """True si el service puede servir embeddings (router o legacy).

        Sprint 19.5 Slice 6 Commit 3: en multi-tier mode, basta con
        que el router esté instanciado. El router tiene su propia
        semantica de ``is_enabled`` (al menos un policy con al menos
        un tier enabled, TDD §2.5 backward-compat). Aqui tratamos
        "tengo algo configurado" como enabled, alineado con la
        semantica legacy ("hay backend, da igual si responde ahora").
        Callers que quieran granularidad fina usan ``health_check()``.
        """
        return bool(self._router) or bool(self._backend)

    @property
    def backend_name(self) -> str | None:
        """Nombre del backend activo (multi_tier | openai | gemini).

        Sprint 19.5 Slice 6 Commit 3: si el router está activo,
        retorna ``"multi_tier"`` (el router expone su propio
        ``backend_name()`` method, TDD §2.5). Para los callers
        existentes, ``backend_name`` sigue siendo property (no
        metodo) para no romper las 9 call sites que la usan sin
        parentesis (verificado 2026-07-14 en loop.py:790, http_api.py
        :1551, search_files.py:74, embed_vault.py:185, builtin.py:582).
        """
        if self._router is not None:
            return self._router.backend_name()
        if self._backend is not None:
            return self._backend.name
        return None

    async def aclose(self) -> None:
        """Cierra clientes HTTP (httpx, openai). Llamar al shutdown.

        Sprint 19.5 Slice 6 Commit 3: si el router está activo,
        cierra sus backends (cada uno con su ``aclose()`` para
        liberar clientes httpx). El router no tiene estado
        persistente más alla de la cache en memoria; eso se
        libera por GC.
        """
        if self._router is not None:
            await aclose_router(self._router)
            return
        if self._backend is not None and hasattr(self._backend, "aclose"):
            await self._backend.aclose()

    async def embed(self, text: str, *, use_case: EmbeddingPolicy | None = None) -> np.ndarray:
        """Genera el embedding de un texto (dim según backend).

        Returns:
            np.ndarray shape (dim,), dtype float32. Si RAG disabled,
            retorna array de ceros con dim por defecto.

        Sprint 19.5 Slice 6 Commit 3: en multi-tier mode, delega al
        router (``self._router.embed(text)``) que retorna np.ndarray
        (no ``EmbeddingResult``) por backward compat. El router
        internamente cachea el resultado en ``_RouterCache`` y cae
        al siguiente tier si el canonical falla. Si el router
        levanta ``AllTiersFailed`` (todos los tiers cayeron), se
        propaga al caller (que suele ser el chat loop y maneja la
        excepcion con log warning + chat sin RAG).
        """
        await self.ensure_initialized()
        if self._router is not None:
            if use_case is None:
                return await self._router.embed(text)
            return await self._router.embed(text, use_case=use_case)
        if self._backend is None:
            return np.zeros(OPENAI_DEFAULT_DIM, dtype=np.float32)
        return await self._backend.embed(text)

    async def embed_batch(
        self,
        texts: list[str],
        *,
        use_case: EmbeddingPolicy | None = None,
        model: str | None = None,
    ) -> list[np.ndarray]:
        """Genera embeddings para N textos en una sola llamada API.

        PR #113b Slice 2.5: VaultEmbedder.embed_file() necesita embeber
        N chunks (1 file = 1-N chunks). Llamar N veces a `embed()` hace
        N round-trips HTTP — para 50 chunks x 50ms = 2.5s extra por
        file. Un solo call con batched input baja eso a ~200-400ms
        (latency amortizada).

        Args:
            texts: lista de N textos. Empty list → empty list.
            model: override del modelo (default: self._settings.embedding_model).
                Slice 2.5 Embedder lo deja en None para usar el default
                (consistente con embed()).

        Returns:
            lista de N np.ndarray shape (dim,), dtype float32.
            Si RAG disabled, retorna N arrays de ceros con dim default.
            Si backend no soporta batch (e.g. Gemini), cae a N llamadas
            individuales via embed() y devuelve los resultados (best-effort).

        Raises:
            Lo que el backend lance. Slice 2.5 VaultEmbedder captura
            la excepción y aborta el embed_file (el chunker ya escribió
            a vault_chunks con la transaccionalidad BEGIN IMMEDIATE).

        Sprint 19.5 Slice 6 Commit 3: en multi-tier mode, delega al
        router (``self._router.embed_batch(texts, model=model)``) que
        hace 1 sola request HTTP al canonical tier con ``input=texts``
        (TDD §2.5 Gemini Blocker 2 fix). El kwarg ``model`` se acepta
        por backward compat pero el router lo ignora (single canonical
        por policy en v0.16).
        """
        if not texts:
            return []
        await self.ensure_initialized()
        if self._router is not None:
            if use_case is None:
                return await self._router.embed_batch(texts, model=model)
            return await self._router.embed_batch(texts, use_case=use_case, model=model)
        if self._backend is None:
            return [np.zeros(OPENAI_DEFAULT_DIM, dtype=np.float32) for _ in texts]
        # OpenAI + OpenRouter (httpx) backend soportan batch en una llamada.
        # Gemini no; cae a N individuales.
        backend = self._backend
        if hasattr(backend, "embed_batch"):
            return await backend.embed_batch(texts, model=model)
        # Fallback: N llamadas individuales.
        results: list[np.ndarray] = []
        for t in texts:
            results.append(await backend.embed(t))
        return results

    async def embed_and_store(self, file_id: str, text: str) -> bool:
        """Genera embedding y lo persiste (best-effort).

        Truncado defensivo a `backend.max_embed_chars` (8K para Gemini,
        24K para OpenAI). Si la API call falla, log warning y return
        False (NO rompe el upload).

        Sprint 19.5 Slice 6 Commit 4: en multi-tier mode, delega a
        ``_embed_and_store_router`` que usa el router para embedding
        (canonical tier de ``VAULT_INGEST`` policy) y luego persiste
        via ``db.upsert_embedding`` con ``policy='vault_ingest'`` y
        el ``dim`` extraido de ``result.vector.shape[0]``. En legacy
        mode sigue usando ``db.add_file_embedding`` (no toca
        ``dim``/``policy``). ``AllTiersFailed`` del router se traga
        (return False) para preservar el contrato bool.
        """
        await self.ensure_initialized()
        if self._router is not None:
            return await self._embed_and_store_router(file_id, text)
        if self._backend is None:
            return False
        if not text or not text.strip():
            logger.info(
                "embed_and_store_skipped_empty_text",
                extra={"file_id": file_id},
            )
            return False
        max_chars = self._backend.max_embed_chars
        truncated = text[:max_chars] if len(text) > max_chars else text
        try:
            embedding = await self._backend.embed(truncated)
        except Exception as exc:
            logger.warning(
                "embed_and_store_api_call_failed",
                extra={
                    "file_id": file_id,
                    "provider": self._backend.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:200],
                },
            )
            return False
        try:
            await self._db.add_file_embedding(
                file_id, embedding.tobytes(), self._settings.embedding_model
            )
        except Exception:
            logger.exception("embed_and_store_db_persist_failed", extra={"file_id": file_id})
            return False
        # Invalidar cache para forzar re-load en próxima query
        self._cache.invalidate()
        return True

    async def _embed_and_store_router(self, file_id: str, text: str) -> bool:
        """Sprint 19.5 Slice 6 Commit 4: path del router para
        ``embed_and_store``.

        1. Router embed con policy ``VAULT_INGEST`` (canonical tier,
           per TDD §2.1 "first-wins").
        2. Extrae vector + model name del EmbeddingResult.
        3. Persiste via ``db.upsert_embedding(file_id, vector, model, dim, policy)``
           con ``policy='vault_ingest'`` (hardcoded: este helper
           SOLO se llama para la policy VAULT_INGEST) y ``dim`` extraido
           de ``vector.shape[0]`` (canonical dim del embedding que se
           acaba de producir).
        4. Invalida cache legacy (la cache en memoria del router
           es independiente; el cache en DB-backed se hara en Commit 4).

        Returns:
            ``True`` si embedding + persist OK, ``False`` si el
            router no tiene policy VAULT_INGEST configurada, o si
            todos los tiers del cascade fallan (``AllTiersFailed``),
            o si la DB write falla.
        """
        if not text or not text.strip():
            logger.info(
                "embed_and_store_skipped_empty_text",
                extra={"file_id": file_id, "mode": "router"},
            )
            return False
        if self._router is None:  # pragma: no cover — caller checks
            return False
        try:
            result: EmbeddingResult = await self._router.embed_with_policy(
                text, EmbeddingPolicy.VAULT_INGEST
            )
        except AllTiersFailed as exc:
            logger.warning(
                "embed_and_store_router_all_tiers_failed",
                extra={
                    "file_id": file_id,
                    "tried_tiers": exc.tried_tiers,
                    "errors": [type(e).__name__ for e in exc.errors],
                },
            )
            return False
        vector = result.vector
        if vector.size == 0:
            # Fallback per-text del router retorna dim=0 si ese texto
            # individual fallo. Log + return False (no persistir basura).
            logger.warning(
                "embed_and_store_router_empty_vector",
                extra={"file_id": file_id, "source_tier": result.source_tier},
            )
            return False
        # Model name: el router expone el canonical tier (source_tier)
        # pero no el model name en el EmbeddingResult. Lo recuperamos
        # del backend concreto que sirvio la respuesta.
        source_backend = self._router._backends.get(result.source_tier)
        model_name: str
        if source_backend is not None and hasattr(source_backend, "_config"):
            model_name = source_backend._config.model
        elif source_backend is not None and hasattr(source_backend, "_model"):
            model_name = source_backend._model
        else:
            # Fallback extremo: el router cambio internals. Usar el
            # del settings (legacy embedding_model) para que el row
            # en DB tenga un model name plausible.
            model_name = self._settings.embedding_model
        # Dim: extraido del vector que se acaba de producir (no de
        # config). Si el router cae a un tier con dim diferente
        # (e.g. cascade cloud->edge, ambos 4096 hoy), el dim aqui
        # es el del tier que sirvio la respuesta. Esto es la
        # fuente de verdad para el dim persistido.
        canonical_dim = int(vector.shape[0])
        try:
            await self._db.upsert_embedding(
                file_id,
                vector.astype(np.float32).tobytes(),
                model_name,
                canonical_dim,
                policy="vault_ingest",
            )
        except Exception:
            logger.exception(
                "embed_and_store_router_db_persist_failed",
                extra={"file_id": file_id, "source_tier": result.source_tier},
            )
            return False
        # Invalida cache legacy. La cache en memoria del router NO
        # necesita invalidacion (es por texto, no por file_id).
        self._cache.invalidate()
        return True

    async def cosine_search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Búsqueda semántica: top-k files más similares al query.

        Args:
            query: texto natural.
            top_k: máximo de resultados.

        Returns:
            lista de (file_id, score) ordenada por score DESC.
            Vacía si RAG disabled, library vacía, o ningún match
            supera el threshold.

        Sprint 19.5 Slice 6 Commit 3: en multi-tier mode, delega a
        ``_cosine_search_router`` que embebe con policy ``VAULT_INGEST``
        (canonical tier) y luego busca en el cache legacy (DB-backed,
        via ``get_all_embeddings``). Para v0.16, los embeddings
        en DB estan todos en la tabla ``file_embeddings`` SIN policy
        column (v25 migration es Commit 4). Cuando Commit 4 añada
        la columna, ``_cosine_search_router`` filtrara por policy.
        """
        await self.ensure_initialized()
        if self._router is not None:
            return await self._cosine_search_router(query, top_k)
        if self._backend is None:
            return []
        if not query or not query.strip():
            return []
        # 1. Embed query
        try:
            query_emb = await self._backend.embed(query)
        except Exception as exc:
            logger.warning(
                "cosine_search_query_embed_failed",
                extra={
                    "provider": self._backend.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:200],
                },
            )
            return []
        # 2. Cargar cache
        cache, file_ids = await self._cache.get_all(self._db, policy="vault_ingest")
        if cache.shape[0] == 0:
            return []
        # 3. Cosine similarity
        query_norm = float(np.linalg.norm(query_emb))
        if query_norm == 0:
            return []
        cache_norms = np.linalg.norm(cache, axis=1)
        safe_cache_norms = np.where(cache_norms == 0, 1.0, cache_norms)
        scores = (cache @ query_emb) / (query_norm * safe_cache_norms)
        # 4. Filtrar por threshold
        threshold = self._settings.min_similarity_threshold
        mask = scores >= threshold
        if not np.any(mask):
            return []
        # 5. Top-k por score DESC
        valid_indices = np.where(mask)[0]
        valid_scores = scores[valid_indices]
        order = np.argsort(-valid_scores)[:top_k]
        return [(file_ids[valid_indices[i]], float(valid_scores[i])) for i in order]

    async def _cosine_search_router(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Sprint 19.5 Slice 6 Commit 3: path del router para
        ``cosine_search``.

        1. Router embed con policy ``VAULT_INGEST`` (the same space used by the producer).
        2. Carga todos los embeddings del cache legacy (DB-backed).
        3. Computa cosine similarity + threshold + top-k.

        Para v0.16 NO hay columna ``policy`` en ``file_embeddings``
        (esa es la migracion v25 de Commit 4). El cache del router
        es en memoria, keyed por ``(text, policy)`` — no es lo que
        queremos para retrieval (queremos match contra TODOS los
        files del vault, no solo los textos recientes embebidos
        con esta policy). Por eso leemos de la DB legacy.
        """
        if self._router is None:  # pragma: no cover — caller checks
            return []
        if not query or not query.strip():
            return []
        try:
            result = await self._router.embed_with_policy(query, EmbeddingPolicy.VAULT_INGEST)
        except AllTiersFailed as exc:
            logger.warning(
                "cosine_search_router_all_tiers_failed",
                extra={
                    "tried_tiers": exc.tried_tiers,
                    "errors": [type(e).__name__ for e in exc.errors],
                },
            )
            return []
        query_emb = result.vector
        if query_emb.size == 0:
            return []
        # Carga cache legacy (la cache del router es per-text y no
        # sirve para retrieval contra todos los files del vault).
        cache, file_ids = await self._cache.get_all(self._db, policy="vault_ingest")
        if cache.shape[0] == 0:
            return []
        # Cosine similarity
        query_norm = float(np.linalg.norm(query_emb))
        if query_norm == 0:
            return []
        cache_norms = np.linalg.norm(cache, axis=1)
        safe_cache_norms = np.where(cache_norms == 0, 1.0, cache_norms)
        scores = (cache @ query_emb) / (query_norm * safe_cache_norms)
        # Threshold + top-k
        threshold = self._settings.min_similarity_threshold
        mask = scores >= threshold
        if not np.any(mask):
            return []
        valid_indices = np.where(mask)[0]
        valid_scores = scores[valid_indices]
        order = np.argsort(-valid_scores)[:top_k]
        return [(file_ids[valid_indices[i]], float(valid_scores[i])) for i in order]

    def invalidate_cache(self, policy: str | None = None) -> None:
        """Invalida el cache manualmente.

        Args:
            policy: nombre del policy a invalidar (``"chat_rag"`` o
                ``"vault_ingest"``). ``None`` invalida TODO (default,
                backward compat con la firma pre-Sprint 19.5). TDD
                §2.5 M-7 fix (v0.16): specific policy invalida solo
                ese cache del router.

        En multi-tier mode, delega al router (``invalidate_cache``
        con el policy string). En legacy mode, invalida el cache
        en memoria (``EmbeddingsCache.invalidate``).
        """
        if self._router is not None:
            self._router.invalidate_cache(policy=policy)
            return
        self._cache.invalidate()

    async def health_check(self) -> bool:
        """Verifica que el backend responde a una API call minima.

        Sprint 19.5 Slice 6 Commit 3: en multi-tier mode, delega al
        router (``self._router.health_check()``) que hace ping al
        canonical tier del primer policy configurado.
        """
        await self.ensure_initialized()
        if self._router is not None:
            return await self._router.health_check()
        if self._backend is None:
            return False
        try:
            await self._backend.embed("ping")
            return True
        except Exception:
            return False
