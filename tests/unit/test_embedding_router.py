"""Sprint 19.5 Slice 6 Commit 2: tests para EmbeddingRouter.

Cubre (TDD §3 commit 2, ~10 tests):
- Cascade order respected (failures fall through)
- Circuit breaker per tier is independent (one tier's failure does
  NOT open another tier's breaker)
- Dim validation (Rule 3: 2+ tiers with different dims -> ConfigError
  al construir el router via _build_router)
- Canonical tier selection (first in list)
- AllTiersFailed raised when all tiers fail
- embed_with_policy uses the policy's tier list
- embed() (no policy) defaults to CHAT_RAG
- embed_batch native (single request al backend, no N sequential)
- _build_router factory: env-var -> TierConfig mapping
- invalidate_cache(policy=None) invalida todo; invalidate_cache(
  policy='X') invalida solo ese policy

Helpers:
- ``FakeBackend``: implementa ``EmbeddingBackend`` con comportamiento
  configurable por instancia (return vector / raise / count calls).
  No hace HTTP — los backends HTTP (OpenAICompatibleBackend /
  GeminiBackend) se testan con respx en un test dedicado
  (``test_openai_backend_embed_batch_native``).
- ``make_router(...)``: factory para construir routers con backends
  fake, parametrizable por tiers + policies.

Por que FakeBackend y no respx en todos los tests: el cascade, la
logica del breaker, la validacion de dim, etc. son logica PURA del
router. Mockear HTTP introduce ruido y acopla los tests a la
implementacion de OpenAI/HTTP. Solo testeamos el HTTP path donde
importa (batch nativo, auth header, error mapping).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import numpy as np
import pytest

from hermes.services.embedding_router import (
    AllTiersFailed,
    ConfigError,
    EmbeddingBackend,
    EmbeddingPolicy,
    EmbeddingResult,
    EmbeddingRouter,
    GeminiBackend,
    NoPolicyConfiguredError,
    OpenAICompatibleBackend,
    PolicyConfig,
    TierConfig,
    _build_router,
)

# ===========================================================================
# Test fixtures y helpers
# ===========================================================================


class FakeBackend(EmbeddingBackend):
    """Backend fake con comportamiento configurable por instancia.

    Atributos configurables en el constructor:
    - ``return_vector``: np.ndarray que devuelve ``embed()`` /
      ``embed_batch()``. Default vector de ceros de la dim dada.
    - ``raise_on_embed``: excepcion que ``embed()`` levanta. Si esta
      set y la llamada pasa por el circuit breaker, el breaker
      registra el fail. Default None (no raise).
    - ``enabled``: valor que devuelve ``is_enabled()``. Default True.
    - ``call_count``: contador de cuantas veces se llamo ``embed()``.
    - ``embed_call_log``: lista de textos recibidos en ``embed()``
      (para assertions de orden).

    Ejemplo:
        backend = FakeBackend(name='nas', dim=384, return_vector=vec)
        router = EmbeddingRouter(backends={'nas': backend}, policies=...)
    """

    def __init__(
        self,
        *,
        name: str = "fake",
        dim: int = 384,
        return_vector: np.ndarray | None = None,
        raise_on_embed: Exception | None = None,
        enabled: bool = True,
    ) -> None:
        self._name = name
        self._dim = dim
        self._return_vector = (
            return_vector if return_vector is not None else np.zeros(dim, dtype=np.float32)
        )
        self._raise_on_embed: Exception | None = raise_on_embed
        self._enabled = enabled
        self.call_count = 0
        self.embed_call_log: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def is_enabled(self) -> bool:
        return self._enabled

    async def embed(self, text: str) -> np.ndarray:
        self.call_count += 1
        self.embed_call_log.append(text)
        if self._raise_on_embed is not None:
            raise self._raise_on_embed
        return self._return_vector.copy()

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        # Para los tests de cascade/cache, embed_batch NO es el path
        # principal (el router llama embed() internamente para no
        # interferir con el circuit breaker per-call). Pero algunos
        # tests del backend (e.g. ``test_openai_backend_embed_batch_native``)
        # usan ``embed_batch`` directamente. Aqui delegamos a embed()
        # para mantener el comportamiento del FakeBackend consistente.
        return [await self.embed(t) for t in texts]

    async def aclose(self) -> None:
        return None


def make_router(
    *,
    backends: dict[str, EmbeddingBackend],
    policies: dict[EmbeddingPolicy, PolicyConfig] | None = None,
) -> EmbeddingRouter:
    """Construye un ``EmbeddingRouter`` con la signature minima usada en tests.

    Si ``policies`` es None, crea un default con ``CHAT_RAG`` mapeado
    a la primera key de ``backends`` y ``VAULT_INGEST`` mapeado a la
    segunda (si existe), o al primer tier de nuevo. Esto evita tener
    que repetir la config en cada test.
    """
    if policies is None:
        backend_names = list(backends.keys())
        if not backend_names:
            raise ValueError("backends dict must not be empty")
        policies = {
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=backend_names[:1],
            ),
        }
        if len(backend_names) >= 2:
            policies[EmbeddingPolicy.VAULT_INGEST] = PolicyConfig(
                use_case=EmbeddingPolicy.VAULT_INGEST,
                tiers=backend_names[1:2],
            )
    return EmbeddingRouter(backends=backends, policies=policies)


@pytest.fixture
def clean_embedding_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limpia env vars ``EMBEDDING_TIER_*`` y ``EMBEDDING_POLICY_*``.

    Necesario porque ``_build_router`` lee directamente de
    ``os.environ`` y tests anteriores (o un shell contaminado) pueden
    dejar vars seteadas. El monkeypatch se revierte al final del test.
    """
    for key in list(os.environ.keys()):
        if key.startswith("EMBEDDING_TIER_") or key.startswith("EMBEDDING_POLICY_"):
            monkeypatch.delenv(key, raising=False)


# ===========================================================================
# 1. Cascade order respected
# ===========================================================================


@pytest.mark.asyncio
async def test_cascade_order_respects_priority_and_falls_through_on_failure() -> None:
    """Si el primer tier falla, el router cae al segundo; primer exito gana.

    TDD §2.6 "Router pattern": cascade iterates tiers in priority
    order, first success wins, exceptions fall through. Verificamos
    que (a) el tier primario se intenta primero, (b) si falla, se
    intenta el siguiente, (c) el resultado viene del tier que tuvo
    exito.
    """
    primary_vec = np.full(384, 0.1, dtype=np.float32)
    fallback_vec = np.full(384, 0.9, dtype=np.float32)
    primary = FakeBackend(
        name="primary",
        dim=384,
        return_vector=primary_vec,
        raise_on_embed=RuntimeError("primary down"),
    )
    fallback = FakeBackend(name="fallback", dim=384, return_vector=fallback_vec)
    router = EmbeddingRouter(
        backends={"primary": primary, "fallback": fallback},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["primary", "fallback"],
            ),
        },
    )
    result = await router.embed("hello", use_case=EmbeddingPolicy.CHAT_RAG)
    # ``embed()`` retorna np.ndarray (drop-in compat con
    # EmbeddingsService). Para inspeccionar ``source_tier`` usamos
    # ``embed_with_policy()`` que sigue devolviendo ``EmbeddingResult``.
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, fallback_vec)
    result_full = await router.embed_with_policy("hello2", EmbeddingPolicy.CHAT_RAG)
    assert result_full.source_tier == "fallback"
    # Verifica que ambos tiers se intentaron. ``embed("hello")`` invoca
    # el cascade (primary fail + fallback success). El segundo
    # ``embed_with_policy("hello2")`` es texto distinto → cache miss →
    # cascade de nuevo (primary fail + fallback success). Resultado:
    # primary=2, fallback=2.
    assert primary.call_count == 2
    assert fallback.call_count == 2


# ===========================================================================
# 2. Circuit breakers per tier are independent
# ===========================================================================


@pytest.mark.asyncio
async def test_circuit_breakers_per_tier_are_independent() -> None:
    """Si el tier A abre su breaker, el tier B sigue operativo.

    Clave de arquitectura (TDD §2.6): un NAS que se va a OPEN no debe
    afectar al breaker del edge. Cada tier tiene su propio
    ``CircuitBreaker``. Forzamos al tier A a fallar ``fail_max`` veces
    (default 5) y verificamos que (a) el tier A's breaker esta OPEN,
    (b) el tier B sigue respondiendo OK.
    """
    # fail_max=2 para no tener que esperar 5 fails.
    failing = FakeBackend(
        name="failing",
        dim=384,
        return_vector=np.zeros(384, dtype=np.float32),
        raise_on_embed=ConnectionError("nas down"),
    )
    healthy = FakeBackend(
        name="healthy",
        dim=384,
        return_vector=np.full(384, 0.5, dtype=np.float32),
    )
    router = EmbeddingRouter(
        backends={"failing": failing, "healthy": healthy},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["failing", "healthy"],
            ),
        },
        breaker_fail_max=2,  # Agilizar el test (default 5).
    )
    # 1. Primer request: failing falla, healthy responde.
    #    ``embed()`` retorna np.ndarray. Para verificar
    #    ``source_tier`` usamos ``embed_with_policy()``.
    r1_vec = await router.embed("a", use_case=EmbeddingPolicy.CHAT_RAG)
    assert isinstance(r1_vec, np.ndarray)
    r1 = await router.embed_with_policy("a1", EmbeddingPolicy.CHAT_RAG)
    assert r1.source_tier == "healthy"
    # 2. Segundo request: failing falla OTRA vez, healthy responde.
    #    A partir de aqui el breaker de "failing" tiene 2 fails
    #    >= fail_max=2, asi que deberia abrirse.
    r2_vec = await router.embed("b", use_case=EmbeddingPolicy.CHAT_RAG)
    assert isinstance(r2_vec, np.ndarray)
    r2 = await router.embed_with_policy("b1", EmbeddingPolicy.CHAT_RAG)
    assert r2.source_tier == "healthy"
    # 3. Tercer request: el breaker de "failing" deberia estar OPEN.
    #    El cascade salta "failing" sin llamarlo (fail fast via
    #    CircuitOpenError), y va directo a "healthy".
    r3_vec = await router.embed("c", use_case=EmbeddingPolicy.CHAT_RAG)
    assert isinstance(r3_vec, np.ndarray)
    r3 = await router.embed_with_policy("c1", EmbeddingPolicy.CHAT_RAG)
    assert r3.source_tier == "healthy"
    # 4. Verificar que "failing" solo fue llamado 2 veces (las
    #    primeras 2 requests). El 3er request NO lo llama porque
    #    su breaker esta abierto.
    assert failing.call_count == 2
    # 5. El breaker de "healthy" sigue cerrado: healthy.call_count
    #    es 6 (3 embed() calls + 3 embed_with_policy() calls del test,
    #    todos hacia "healthy" despues del fail de "failing").
    assert healthy.call_count == 6
    # 6. Inspeccion directa: el breaker de "failing" esta OPEN;
    #    el de "healthy" sigue CLOSED.
    assert router._breakers["failing"].current_state == "open"
    assert router._breakers["healthy"].current_state == "closed"


# ===========================================================================
# 3. dim validation (Rule 3: within-policy dim mismatch -> ConfigError)
# ===========================================================================


@pytest.mark.asyncio
async def test_build_router_raises_config_error_on_within_policy_dim_mismatch(
    clean_embedding_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule 3 (TDD §2.4 #3): un policy con 2+ tiers de dim distinta -> ConfigError.

    Hard fail en startup: un fallback que devuelve un vector de dim
    distinta que el cache produce ValueError en cosine_search. Mejor
    fallar al arrancar que crashear en produccion.
    """
    # nas = 384-dim, cloud = 4096-dim
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://nas.local:8082/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")  # 384
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__API_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__MODEL", "qwen/qwen3-embedding-8b")  # 4096
    # Policy mezcla los dos tiers con dim distinta -> ConfigError
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas,cloud")
    with pytest.raises(ConfigError) as exc_info:
        _build_router(settings=None)
    msg = str(exc_info.value)
    assert "chat_rag" in msg
    assert "384" in msg or "4096" in msg  # menciona las dims en conflicto


@pytest.mark.asyncio
async def test_build_router_passes_when_all_tiers_in_policy_share_dim(
    clean_embedding_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path del Rule 3: 2+ tiers con la misma dim pasa la validacion."""
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__API_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__MODEL", "qwen/qwen3-embedding-8b")
    monkeypatch.setenv("EMBEDDING_TIER_EDGE__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_EDGE__BASE_URL", "http://edge.local:8800/v1")
    monkeypatch.setenv("EMBEDDING_TIER_EDGE__MODEL", "qwen3-embedding:8b")
    # Ambos 4096-dim.
    monkeypatch.setenv("EMBEDDING_POLICY_VAULT_INGEST", "cloud,edge")
    router = _build_router(settings=None)
    assert router is not None
    assert router._backends["cloud"].dim == 4096
    assert router._backends["edge"].dim == 4096


# ===========================================================================
# 4. Canonical tier selection (first in list)
# ===========================================================================


@pytest.mark.asyncio
async def test_canonical_tier_is_first_in_priority_list() -> None:
    """El primer tier en ``policy.tiers`` es el canonical.

    TDD §2.1 "first-wins" rule: el canonical se elige por posicion
    en la lista, sin campo ``__CANONICAL`` explicito. Verificamos
    que (a) con un cache vacio y todos los tiers healthy, el router
    elije el primero, (b) el resultado viene de ese tier.
    """
    canonical = FakeBackend(
        name="canonical",
        dim=384,
        return_vector=np.full(384, 0.1, dtype=np.float32),
    )
    secondary = FakeBackend(
        name="secondary",
        dim=384,
        return_vector=np.full(384, 0.9, dtype=np.float32),
    )
    router = EmbeddingRouter(
        backends={"canonical": canonical, "secondary": secondary},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["canonical", "secondary"],
            ),
        },
    )
    result = await router.embed("hi", use_case=EmbeddingPolicy.CHAT_RAG)
    # ``embed()`` retorna np.ndarray. Para inspeccionar ``source_tier``
    # usamos ``embed_with_policy()`` (lower-level).
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.full(384, 0.1, dtype=np.float32))
    result_full = await router.embed_with_policy("hi2", EmbeddingPolicy.CHAT_RAG)
    assert result_full.source_tier == "canonical"
    # El segundo tier NO se llamo (cascade gano en el primero).
    # ``canonical.call_count`` = 2 (1 del embed() + 1 del
    # embed_with_policy() — el cache no hit porque text es distinto).
    assert canonical.call_count == 2
    assert secondary.call_count == 0


# ===========================================================================
# 5. AllTiersFailed when all tiers fail
# ===========================================================================


@pytest.mark.asyncio
async def test_all_tiers_failed_raised_when_every_tier_fails() -> None:
    """Si TODOS los tiers fallan, el router lanza ``AllTiersFailed``
    con la lista de errores y los tiers intentados.

    El cascade NO aborta en el primer fail; recoge todos los errores
    y los reporta juntos para que el operador tenga contexto completo.
    """
    a = FakeBackend(
        name="a",
        dim=384,
        raise_on_embed=TimeoutError("a timeout"),
    )
    b = FakeBackend(
        name="b",
        dim=384,
        raise_on_embed=ConnectionError("b refused"),
    )
    c = FakeBackend(
        name="c",
        dim=384,
        raise_on_embed=ValueError("c broken"),
    )
    router = EmbeddingRouter(
        backends={"a": a, "b": b, "c": c},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a", "b", "c"],
            ),
        },
    )
    with pytest.raises(AllTiersFailed) as exc_info:
        await router.embed("hi", use_case=EmbeddingPolicy.CHAT_RAG)
    exc = exc_info.value
    assert exc.policy == EmbeddingPolicy.CHAT_RAG
    assert exc.tried_tiers == ["a", "b", "c"]
    assert len(exc.errors) == 3
    assert isinstance(exc.errors[0], TimeoutError)
    assert isinstance(exc.errors[1], ConnectionError)
    assert isinstance(exc.errors[2], ValueError)
    # El mensaje incluye el policy y los tipos de error (legible para
    # logs).
    assert "chat_rag" in str(exc)
    assert "TimeoutError" in str(exc)
    # Todos los tiers se intentaron (ninguno abrio breaker antes de
    # fail_max=5, asi que no se saltaron).
    assert a.call_count == 1
    assert b.call_count == 1
    assert c.call_count == 1


# ===========================================================================
# 6. embed_with_policy uses the policy's tier list
# ===========================================================================


@pytest.mark.asyncio
async def test_embed_with_policy_uses_assigned_tier_list() -> None:
    """``embed_with_policy(text, X)`` usa los tiers del policy X,
    NO los del policy Y.

    Aislamiento entre policies: el mismo router tiene dos policies
    configurados con tiers distintos, y ``embed_with_policy`` respeta
    cada uno.
    """
    chat_vec = np.full(384, 0.1, dtype=np.float32)
    vault_vec = np.full(4096, 0.5, dtype=np.float32)
    chat_backend = FakeBackend(name="nas", dim=384, return_vector=chat_vec)
    vault_backend = FakeBackend(name="cloud", dim=4096, return_vector=vault_vec)
    router = EmbeddingRouter(
        backends={"nas": chat_backend, "cloud": vault_backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["nas"],
            ),
            EmbeddingPolicy.VAULT_INGEST: PolicyConfig(
                use_case=EmbeddingPolicy.VAULT_INGEST,
                tiers=["cloud"],
            ),
        },
    )
    r_chat = await router.embed_with_policy("q", EmbeddingPolicy.CHAT_RAG)
    r_vault = await router.embed_with_policy("q", EmbeddingPolicy.VAULT_INGEST)
    assert r_chat.source_tier == "nas"
    assert r_chat.vector.shape == (384,)
    assert r_vault.source_tier == "cloud"
    assert r_vault.vector.shape == (4096,)


@pytest.mark.asyncio
async def test_embed_with_policy_raises_when_policy_not_configured() -> None:
    """Si el policy X no esta en self._policies, raise ``NoPolicyConfiguredError``.

    TDD §2.4 Rule 9: el sistema arranca aunque una policy no este
    configurada, pero usarla es un error explicito.
    """
    router = EmbeddingRouter(
        backends={"x": FakeBackend(name="x", dim=384)},
        policies={},  # Sin policies
    )
    with pytest.raises(NoPolicyConfiguredError) as exc_info:
        await router.embed_with_policy("q", EmbeddingPolicy.CHAT_RAG)
    assert exc_info.value.policy == EmbeddingPolicy.CHAT_RAG
    # Mensaje action able: incluye el nombre del env var a setear.
    assert "EMBEDDING_POLICY_CHAT_RAG" in str(exc_info.value)


# ===========================================================================
# 7. embed() default is CHAT_RAG
# ===========================================================================


@pytest.mark.asyncio
async def test_embed_default_policy_is_chat_rag() -> None:
    """``embed(text)`` sin ``use_case`` usa ``CHAT_RAG`` (TDD §2.5 M-8 fix).

    El default es CHAT_RAG porque el caller mas frecuente es el chat
    loop live (``hermes/agent/loop.py:790``). El caller que quiere
    vault_ingest debe ser explicito (``embed(text, use_case=
    EmbeddingPolicy.VAULT_INGEST)``).
    """
    chat_vec = np.full(384, 0.1, dtype=np.float32)
    vault_vec = np.full(4096, 0.7, dtype=np.float32)
    chat_backend = FakeBackend(name="nas", dim=384, return_vector=chat_vec)
    vault_backend = FakeBackend(name="cloud", dim=4096, return_vector=vault_vec)
    router = EmbeddingRouter(
        backends={"nas": chat_backend, "cloud": vault_backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["nas"],
            ),
            EmbeddingPolicy.VAULT_INGEST: PolicyConfig(
                use_case=EmbeddingPolicy.VAULT_INGEST,
                tiers=["cloud"],
            ),
        },
    )
    # Sin use_case -> CHAT_RAG -> nas. ``embed()`` retorna
    # ``np.ndarray`` (drop-in compat con EmbeddingsService).
    r_default = await router.embed("hi")
    assert isinstance(r_default, np.ndarray)
    assert r_default.shape == (384,)
    np.testing.assert_array_equal(r_default, chat_vec)
    # Con use_case explicito -> VAULT_INGEST -> cloud.
    r_explicit = await router.embed("hi2", use_case=EmbeddingPolicy.VAULT_INGEST)
    assert isinstance(r_explicit, np.ndarray)
    assert r_explicit.shape == (4096,)
    np.testing.assert_array_equal(r_explicit, vault_vec)
    # Para verificar ``source_tier`` usamos ``embed_with_policy()``.
    r_default_full = await router.embed_with_policy("hi3", EmbeddingPolicy.CHAT_RAG)
    assert r_default_full.source_tier == "nas"
    r_explicit_full = await router.embed_with_policy("hi4", EmbeddingPolicy.VAULT_INGEST)
    assert r_explicit_full.source_tier == "cloud"


# ===========================================================================
# 8. embed_batch native (single request, not N sequential)
# ===========================================================================


@pytest.mark.asyncio
async def test_openai_backend_embed_batch_uses_single_request(respx_mock: Any) -> None:
    """``OpenAICompatibleBackend.embed_batch`` hace UN solo request HTTP
    con ``input=texts`` (TDD §2.6 OpenAI row, batch nativo).

    Esto evita DDoS al edge daemon (N conexiones) y 429 en
    OpenRouter (rate limit por minute). Verificamos que con N=3
    textos, se hace exactamente 1 POST a ``/v1/embeddings`` y la
    respuesta tiene 3 embeddings.
    """
    cfg = TierConfig(
        name="cloud",
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3-embedding-8b",
        api_key="fake-key",
        timeout_s=30.0,
        enabled=True,
        extra_headers={"X-OpenRouter-ZDR": "true"},
    )
    backend = OpenAICompatibleBackend(cfg)
    # Mock del endpoint /v1/embeddings con respuesta de 3 embeddings.
    route = respx_mock.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1] * 4096},
                    {"embedding": [0.2] * 4096},
                    {"embedding": [0.3] * 4096},
                ],
            },
        )
    )
    results = await backend.embed_batch(["text-a", "text-b", "text-c"])
    # Un solo request, no N.
    assert route.call_count == 1
    assert len(results) == 3
    assert all(r.shape == (4096,) for r in results)
    assert results[0][0] == pytest.approx(0.1)
    assert results[1][0] == pytest.approx(0.2)
    assert results[2][0] == pytest.approx(0.3)
    # El header de auth y el extra_header se enviaron.
    request = route.calls[0].request
    assert request.headers.get("Authorization") == "Bearer fake-key"
    assert request.headers.get("X-OpenRouter-ZDR") == "true"
    await backend.aclose()


@pytest.mark.asyncio
async def test_gemini_backend_embed_batch_falls_back_to_n_sequential() -> None:
    """``GeminiBackend.embed_batch`` cae a N llamadas secuenciales
    (Gemini API no soporta batch nativo, TDD §2.6).

    No testeamos el HTTP path aqui (eso es para integration con
    Google); testeamos que ``embed_batch`` llama ``embed()`` N veces.
    Para eso usamos un ``FakeBackend``-style override: subclass que
    cuenta las llamadas a ``embed``.
    """
    call_count = 0

    class _Counting(GeminiBackend):
        async def embed(self, text: str) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.full(3072, 0.5, dtype=np.float32)

    backend = _Counting(
        name="gemini",
        model="gemini-embedding-001",
        api_key="fake-key",
    )
    results = await backend.embed_batch(["a", "b", "c", "d"])
    assert call_count == 4  # N llamadas, no 1 batch.
    assert len(results) == 4
    await backend.aclose()


# ===========================================================================
# 9. _build_router factory: env-var -> TierConfig mapping
# ===========================================================================


@pytest.mark.asyncio
async def test_build_router_factory_maps_env_vars_to_tier_configs(
    clean_embedding_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_build_router`` lee env vars ``EMBEDDING_TIER_*`` y
    ``EMBEDDING_POLICY_*`` y los mapea a ``TierConfig`` /
    ``PolicyConfig`` correctos.

    Cubre (a) enabled=True, (b) parsing de base_url, model, api_key,
    timeout_s, extra_headers, (c) que la policy de tiers se lee
    correctamente.
    """
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://nas.lan:8082/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__API_KEY", "")  # LAN, sin auth
    monkeypatch.setenv("EMBEDDING_TIER_NAS__TIMEOUT_S", "10")
    monkeypatch.setenv(
        "EMBEDDING_TIER_NAS__EXTRA_HEADERS",
        '{"X-Custom-Header": "test-value"}',
    )
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas")
    router = _build_router(settings=None)
    assert router is not None
    # Tier config parseada correctamente.
    nas_backend = router._backends["nas"]
    assert isinstance(nas_backend, OpenAICompatibleBackend)
    assert nas_backend.name == "nas"
    assert nas_backend.dim == 384  # granite-97m -> 384
    # El extra_header se aplico al cliente httpx (lo verificamos
    # inspeccionando el cliente lazy tras un primer embed).
    # No necesitamos network — solo necesitamos que el _ensure_client
    # se haya creado con el header.
    # Llamamos a un atributo lazy: forzar creacion.
    _ = nas_backend._ensure_client()
    assert nas_backend._client is not None
    assert nas_backend._client.headers.get("X-Custom-Header") == "test-value"
    # Sin auth header (api_key vacia).
    assert "Authorization" not in nas_backend._client.headers
    # Policy mapeada correctamente.
    assert EmbeddingPolicy.CHAT_RAG in router._policies
    assert router._policies[EmbeddingPolicy.CHAT_RAG].tiers == ["nas"]


@pytest.mark.asyncio
async def test_build_router_returns_none_when_no_tiers_configured(
    clean_embedding_env: None,
) -> None:
    """Si no hay env vars ``EMBEDDING_TIER_*``, retorna ``None``.

    ``EmbeddingsService`` (Commit 3) usa el ``None`` para decidir
    legacy mode (single-backend con OpenRouterBackend + GeminiBackend).
    """
    # No hay env vars (clean_embedding_env ya limpio).
    assert _build_router(settings=None) is None


@pytest.mark.asyncio
async def test_build_router_raises_config_error_on_enabled_tier_without_url(
    clean_embedding_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule 7 (TDD §2.4 #7): enabled=True pero sin base_url o model -> ConfigError.

    Defensa contra el bug del rewrite anterior (Appendix A #7): el
    rewrite usaba defaults hardcoded para base_url, lo cual era un
    leak del IP del NAS. El factory NO debe permitir arrancar con
    configuracion incompleta.
    """
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")
    # Sin BASE_URL.
    with pytest.raises(ConfigError) as exc_info:
        _build_router(settings=None)
    assert "nas" in str(exc_info.value)
    assert "BASE_URL" in str(exc_info.value) or "MODEL" in str(exc_info.value)


@pytest.mark.asyncio
async def test_build_router_raises_config_error_on_unknown_model(
    clean_embedding_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modelo desconocido (no en ``KNOWN_MODEL_DIMS``) -> ConfigError
    con hint explicito de como arreglarlo.

    Trade-off vs default magico: preferimos fail loud con accion
    concreta (agregar al registry) a un silent default que luego
    crashee en cosine_search con un ValueError confuso.
    """
    monkeypatch.setenv("EMBEDDING_TIER_CUSTOM__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_CUSTOM__BASE_URL", "http://custom.local:8080")
    monkeypatch.setenv("EMBEDDING_TIER_CUSTOM__MODEL", "unknown-model-xyz")
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "custom")
    with pytest.raises(ConfigError) as exc_info:
        _build_router(settings=None)
    msg = str(exc_info.value)
    assert "unknown-model-xyz" in msg
    assert "KNOWN_MODEL_DIMS" in msg


# ===========================================================================
def test_build_router_supports_local_qwen_embedding_model(
    clean_embedding_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public Compose embedding model has a known 1024 dimension."""
    monkeypatch.setenv("EMBEDDING_TIER_EDGE__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_EDGE__BASE_URL", "http://ollama:11434/v1")
    monkeypatch.setenv("EMBEDDING_TIER_EDGE__MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "edge")

    router = _build_router(settings=None)

    assert router is not None
    assert router._backends["edge"].dim == 1024


def test_build_router_supports_granite_311m(
    clean_embedding_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The low-power CPU profile uses Granite 311M at 768 dimensions."""
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://embedding.local:8083/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-311m")
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas")

    router = _build_router(settings=None)

    assert router is not None
    assert router._backends["nas"].dim == 768

# 10. invalidate_cache semantics (None vs specific policy)
# ===========================================================================


@pytest.mark.asyncio
async def test_invalidate_cache_with_none_clears_all_policies() -> None:
    """``invalidate_cache()`` (default) limpia el cache de TODOS los
    policies (TDD §2.5 M-7 fix v0.16).

    Backward compat con la API anterior de
    ``EmbeddingsService.invalidate_cache()`` que no tomaba args.
    """
    chat_vec = np.full(384, 0.1, dtype=np.float32)
    vault_vec = np.full(4096, 0.5, dtype=np.float32)
    chat_backend = FakeBackend(name="nas", dim=384, return_vector=chat_vec)
    vault_backend = FakeBackend(name="cloud", dim=4096, return_vector=vault_vec)
    router = EmbeddingRouter(
        backends={"nas": chat_backend, "cloud": vault_backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["nas"],
            ),
            EmbeddingPolicy.VAULT_INGEST: PolicyConfig(
                use_case=EmbeddingPolicy.VAULT_INGEST,
                tiers=["cloud"],
            ),
        },
    )
    # Llena el cache de ambos policies.
    r_chat_1 = await router.embed_with_policy("text-1", EmbeddingPolicy.CHAT_RAG)
    r_vault_1 = await router.embed_with_policy("text-1", EmbeddingPolicy.VAULT_INGEST)
    assert r_chat_1.cached is False
    assert r_vault_1.cached is False
    # Verifica que el cache tiene 2 entradas.
    assert len(router._cache._results) == 2
    # Invalida todo.
    router.invalidate_cache()
    assert len(router._cache._results) == 0
    # La proxima llamada NO viene de cache (cached=False).
    r_chat_2 = await router.embed_with_policy("text-1", EmbeddingPolicy.CHAT_RAG)
    r_vault_2 = await router.embed_with_policy("text-1", EmbeddingPolicy.VAULT_INGEST)
    assert r_chat_2.cached is False
    assert r_vault_2.cached is False


@pytest.mark.asyncio
async def test_invalidate_cache_with_specific_policy_clears_only_that_one() -> None:
    """``invalidate_cache(policy='chat_rag')`` invalida solo el cache
    de ese policy, NO los demas (TDD §2.5 M-7 fix v0.16).

    Caso de uso: re-embed batch solo de vault_ingest, sin afectar el
    cache de chat_rag. La operacion selectiva es un ahorro de tiempo
    (no releer la library de chat_rag de DB).
    """
    chat_vec = np.full(384, 0.1, dtype=np.float32)
    vault_vec = np.full(4096, 0.5, dtype=np.float32)
    chat_backend = FakeBackend(name="nas", dim=384, return_vector=chat_vec)
    vault_backend = FakeBackend(name="cloud", dim=4096, return_vector=vault_vec)
    router = EmbeddingRouter(
        backends={"nas": chat_backend, "cloud": vault_backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["nas"],
            ),
            EmbeddingPolicy.VAULT_INGEST: PolicyConfig(
                use_case=EmbeddingPolicy.VAULT_INGEST,
                tiers=["cloud"],
            ),
        },
    )
    # Llena el cache de ambos policies.
    await router.embed_with_policy("text-1", EmbeddingPolicy.CHAT_RAG)
    await router.embed_with_policy("text-1", EmbeddingPolicy.VAULT_INGEST)
    assert len(router._cache._results) == 2
    # Invalida SOLO chat_rag.
    router.invalidate_cache(policy="chat_rag")
    # Queda 1 entrada (vault_ingest).
    assert len(router._cache._results) == 1
    # El cache de vault_ingest sigue hit.
    r_vault = await router.embed_with_policy("text-1", EmbeddingPolicy.VAULT_INGEST)
    assert r_vault.cached is True
    # El cache de chat_rag NO hit (re-invalidado).
    r_chat = await router.embed_with_policy("text-1", EmbeddingPolicy.CHAT_RAG)
    assert r_chat.cached is False
    # Despues de la nueva llamada, ambos policies vuelven a estar en cache.
    assert len(router._cache._results) == 2


# ===========================================================================
# Tests adicionales (no en el spec inicial pero utiles)
# ===========================================================================


@pytest.mark.asyncio
async def test_router_is_enabled_true_when_at_least_one_tier_enabled() -> None:
    """``is_enabled`` (property) es True si al menos un tier de cualquier
    policy configurado responde ``is_enabled() == True``.

    R1 cycle 8 fix: ``is_enabled`` es ``@property`` (no metodo) para
    ser drop-in compatible con los 4 callers existentes
    (``loop.py``, ``http_api.py``, ``search_files.py``,
    ``embed_vault.py``) que la usan como propiedad. Ver
    ``test_is_enabled_is_property_not_method`` para el contrato
    completo.
    """
    a = FakeBackend(name="a", dim=384, enabled=False)  # disabled
    b = FakeBackend(name="b", dim=384, enabled=True)
    router = EmbeddingRouter(
        backends={"a": a, "b": b},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a", "b"],
            ),
        },
    )
    assert router.is_enabled is True


@pytest.mark.asyncio
async def test_router_is_enabled_false_when_all_tiers_disabled() -> None:
    """Si todos los tiers estan ``is_enabled() == False``, el router
    esta disabled. ``embed()`` igual intenta el cascade (y falla
    con ``AllTiersFailed`` o un error sintetico).
    """
    a = FakeBackend(name="a", dim=384, enabled=False)
    router = EmbeddingRouter(
        backends={"a": a},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a"],
            ),
        },
    )
    assert router.is_enabled is False


@pytest.mark.asyncio
async def test_router_backend_name_returns_multi_tier() -> None:
    """``backend_name()`` retorna literal ``\"multi_tier\"``.

    Mantiene el contrato de la API anterior de ``EmbeddingsService``
    (TDD §2.5 compat table). Commit 3 expone esto como el
    ``backend_name`` del service.
    """
    router = EmbeddingRouter(
        backends={"a": FakeBackend(name="a", dim=384)},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a"],
            ),
        },
    )
    assert router.backend_name() == "multi_tier"


@pytest.mark.asyncio
async def test_health_check_returns_true_when_canonical_responds() -> None:
    """``health_check()`` hace ping al canonical tier del primer
    policy; True si responde OK.
    """
    backend = FakeBackend(
        name="a",
        dim=384,
        return_vector=np.full(384, 0.1, dtype=np.float32),
    )
    router = EmbeddingRouter(
        backends={"a": backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a"],
            ),
        },
    )
    assert await router.health_check() is True


@pytest.mark.asyncio
async def test_health_check_returns_false_when_canonical_fails() -> None:
    """``health_check()`` retorna False si el canonical tier falla."""
    backend = FakeBackend(
        name="a",
        dim=384,
        raise_on_embed=ConnectionError("down"),
    )
    router = EmbeddingRouter(
        backends={"a": backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a"],
            ),
        },
    )
    assert await router.health_check() is False


@pytest.mark.asyncio
async def test_aclose_router_closes_all_backends() -> None:
    """``aclose_router`` cierra los clientes HTTP de todos los backends."""

    closed: list[str] = []

    class _TrackingBackend(FakeBackend):
        async def aclose(self) -> None:
            closed.append(self.name)

    router = EmbeddingRouter(
        backends={
            "a": _TrackingBackend(name="a", dim=384),
            "b": _TrackingBackend(name="b", dim=384),
        },
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a", "b"],
            ),
        },
    )
    from hermes.services.embedding_router import aclose_router

    await aclose_router(router)
    assert sorted(closed) == ["a", "b"]


@pytest.mark.asyncio
async def test_embedding_policy_does_not_include_facts() -> None:
    """``EmbeddingPolicy`` NO tiene el valor ``FACTS``.

    Defense-in-depth: el enum se itera en sitios que auto-generan
    env var parsing, builders, docs, etc. Si en algun commit futuro
    alguien re-añade FACTS sin cablearlo en los 9 callers (cycle 4
    B-4), este test cae y alerta.

    Si necesitas re-añadir FACTS, sigue el recipe del TDD §2.6 ciclo 5
    m27: 4 pasos coordinados (enum + env var + callers explicitos +
    validacion de cache per-policy).
    """
    values = {p.value for p in EmbeddingPolicy}
    assert "facts" not in values, (
        "FACTS policy was removed in cycle 4 B-4 (phantom-policy bug). "
        "If re-adding, follow TDD §2.6 m27 (4 steps) or this is the same bug."
    )
    # Verificamos que solo hay los 2 esperados.
    assert values == {"chat_rag", "vault_ingest"}


@pytest.mark.asyncio
async def test_embed_batch_returns_empty_for_empty_input() -> None:
    """``embed_batch([])`` retorna ``[]`` sin tocar el cascade."""
    router = EmbeddingRouter(
        backends={"a": FakeBackend(name="a", dim=384)},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a"],
            ),
        },
    )
    results = await router.embed_batch([], use_case=EmbeddingPolicy.CHAT_RAG)
    assert results == []
    # El backend NO se llamo. Cast a FakeBackend para acceder al
    # atributo ``call_count`` (no parte de la ABC EmbeddingBackend).
    backend = router._backends["a"]
    assert isinstance(backend, FakeBackend)
    assert backend.call_count == 0


# ===========================================================================
# R1 cycle 8 fix tests: drop-in compat with EmbeddingsService API.
#
# TDD §2.5 backward-compat table requires:
#   - embed()           -> np.ndarray                (caller: embedder.py:540)
#   - embed_batch()     -> list[np.ndarray]          (caller: embedder.py:377)
#   - embed_batch(...,model=...)  accepts model kwarg (caller: embedder.py:377)
#   - embed_batch(texts) 1 native HTTP request      (NOT a loop)
#   - is_enabled          @property                 (5 callers use as property)
#   - Cache vector.copy() on put (prevent mutation)
# ===========================================================================


@pytest.mark.asyncio
async def test_embed_returns_ndarray_not_embeddingresult() -> None:
    """``embed()`` retorna ``np.ndarray`` (drop-in compat con
    ``EmbeddingsService.embed``).

    Caller concreto que depende de este contrato:
    ``hermes/memory/embedder.py:540`` hace ``cache @ query_emb``
    (matmul numpy). Si ``embed()`` retornara ``EmbeddingResult``, ese
    matmul lanzaria ``TypeError``. R1 cycle 8 BLOCKING fix.
    """
    backend = FakeBackend(
        name="a",
        dim=384,
        return_vector=np.full(384, 0.42, dtype=np.float32),
    )
    router = EmbeddingRouter(
        backends={"a": backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a"],
            ),
        },
    )
    result = await router.embed("hello")
    # Contract: np.ndarray, NOT EmbeddingResult.
    assert isinstance(result, np.ndarray)
    assert not isinstance(result, EmbeddingResult)
    # Sanity: el vector tiene la dim correcta.
    assert result.shape == (384,)
    np.testing.assert_array_equal(result, np.full(384, 0.42, dtype=np.float32))
    # El caller hace matmul — verificar que funciona:
    cache = np.tile(result, (3, 1))  # shape (3, 384)
    scores = cache @ result
    assert scores.shape == (3,)
    assert np.all(scores > 0)  # dot product de vector consigo mismo


@pytest.mark.asyncio
async def test_embed_batch_returns_list_of_ndarrays() -> None:
    """``embed_batch()`` retorna ``list[np.ndarray]`` (drop-in compat).

    Caller concreto: ``hermes/memory/embedder.py:377`` y
    ``cosine_search`` en ``embeddings.py`` iteran la lista y hacen
    matmul sobre cada elemento. Si retornara ``list[EmbeddingResult]``,
    el matmul fallaria. R1 cycle 8 BLOCKING fix.
    """
    backend = FakeBackend(name="a", dim=384)
    router = EmbeddingRouter(
        backends={"a": backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a"],
            ),
        },
    )
    results = await router.embed_batch(["text-a", "text-b", "text-c"])
    # Contract: list[np.ndarray].
    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, np.ndarray)
        assert not isinstance(r, EmbeddingResult)
        assert r.shape == (384,)


@pytest.mark.asyncio
async def test_embed_batch_accepts_model_kwarg() -> None:
    """``embed_batch(texts, model=...)`` no falla (backward compat).

    Caller concreto: ``hermes/memory/embedder.py:377`` —
    ``await self._embeddings.embed_batch(chunk_texts, model=model_version)``.
    Si la firma no acepta ``model``, lanza ``TypeError: unexpected
    keyword argument 'model'``. R1 cycle 8 BLOCKING fix.

    v0.16: ``model`` se acepta y se ignora (single canonical model
    per policy). Verifica que NO rompe la llamada.
    """
    backend = FakeBackend(name="a", dim=384)
    router = EmbeddingRouter(
        backends={"a": backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a"],
            ),
        },
    )
    # Cualquier valor de model (incluso uno "raro") debe ser aceptado.
    results = await router.embed_batch(
        ["t1", "t2"], use_case=EmbeddingPolicy.CHAT_RAG, model="any-model-xyz"
    )
    assert len(results) == 2
    assert all(isinstance(r, np.ndarray) for r in results)
    # Tambien con model=None explicito (kwarg opcional).
    results_none = await router.embed_batch(
        ["t3", "t4"], use_case=EmbeddingPolicy.CHAT_RAG, model=None
    )
    assert len(results_none) == 2


@pytest.mark.asyncio
async def test_embed_batch_uses_single_http_request(respx_mock: Any) -> None:
    """``embed_batch`` con ``OpenAICompatibleBackend`` hace **1 sola**
    request HTTP, no N secuenciales.

    TDD §2.5: "Native batch, NOT a loop over ``embed_with_policy``".
    Esto evita DDoS al edge daemon y 429 en OpenRouter. R1 cycle 8
    BLOCKING fix.
    """
    cfg = TierConfig(
        name="cloud",
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3-embedding-8b",
        api_key="fake-key",
        timeout_s=30.0,
        enabled=True,
        extra_headers={"X-OpenRouter-ZDR": "true"},
    )
    backend = OpenAICompatibleBackend(cfg)
    router = EmbeddingRouter(
        backends={"cloud": backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["cloud"],
            ),
        },
    )
    # Mock del endpoint /v1/embeddings con respuesta de 5 embeddings.
    route = respx_mock.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1] * 4096},
                    {"embedding": [0.2] * 4096},
                    {"embedding": [0.3] * 4096},
                    {"embedding": [0.4] * 4096},
                    {"embedding": [0.5] * 4096},
                ],
            },
        )
    )
    results = await router.embed_batch(
        ["t1", "t2", "t3", "t4", "t5"],
        use_case=EmbeddingPolicy.CHAT_RAG,
    )
    # 1 sola request, NO 5.
    assert route.call_count == 1
    assert len(results) == 5
    for i, r in enumerate(results):
        assert isinstance(r, np.ndarray)
        assert r.shape == (4096,)
        assert r[0] == pytest.approx(0.1 * (i + 1))
    # El body de la request lleva los 5 textos juntos (no 1 por 1).
    request = route.calls[0].request
    body = request.content.decode("utf-8")
    assert '"input"' in body
    # El input es una list de 5 strings.
    import json as _json

    parsed = _json.loads(body)
    assert isinstance(parsed["input"], list)
    assert len(parsed["input"]) == 5
    await backend.aclose()


@pytest.mark.asyncio
async def test_embed_batch_falls_back_to_per_text_on_canonical_failure(
    respx_mock: Any,
) -> None:
    """Si el canonical tier falla (HTTP 500), ``embed_batch`` cae al
    cascade per-text ``embed_with_policy`` para recuperacion parcial.

    R1 cycle 8 BLOCKING fix: el fallback preserva la granularidad
    per-text del circuit breaker — si el 30% de los textos fallan en
    canonical, el 70% se recupera del cascade per-text.

    Aqui mockeamos: canonical falla (500), fallback (tambien OpenAI
    pero otro tier) responde OK. Verificamos que el router hace el
    fallback per-text y devuelve los vectores del fallback.
    """
    canonical_cfg = TierConfig(
        name="canonical",
        base_url="https://canonical.example.com/v1",
        model="qwen/qwen3-embedding-8b",
        api_key="fake-key",
        timeout_s=30.0,
        enabled=True,
    )
    fallback_cfg = TierConfig(
        name="fallback",
        base_url="https://fallback.example.com/v1",
        model="qwen/qwen3-embedding-8b",
        api_key="fake-key",
        timeout_s=30.0,
        enabled=True,
    )
    canonical_backend = OpenAICompatibleBackend(canonical_cfg)
    fallback_backend = OpenAICompatibleBackend(fallback_cfg)
    router = EmbeddingRouter(
        backends={"canonical": canonical_backend, "fallback": fallback_backend},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["canonical", "fallback"],
            ),
        },
    )
    # Canonical falla con 500.
    respx_mock.post("https://canonical.example.com/v1/embeddings").mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )
    # Fallback responde OK con 3 embeddings.
    fallback_route = respx_mock.post("https://fallback.example.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.7] * 4096},
                    {"embedding": [0.8] * 4096},
                    {"embedding": [0.9] * 4096},
                ],
            },
        )
    )
    results = await router.embed_batch(
        ["t1", "t2", "t3"],
        use_case=EmbeddingPolicy.CHAT_RAG,
    )
    # El router cae al cascade per-text. Cada texto llama canonical
    # (fail) + fallback (success). Asi que canonical recibe 3 calls
    # (respx_mock cuenta todas), y fallback recibe 3 calls.
    assert len(results) == 3
    for r in results:
        assert isinstance(r, np.ndarray)
        assert r.shape == (4096,)
    # Verificamos que se uso el fallback (no canonical).
    # El fallback_route.call_count >= 3 (una vez por cada texto).
    assert fallback_route.call_count >= 3
    await canonical_backend.aclose()
    await fallback_backend.aclose()


@pytest.mark.asyncio
async def test_is_enabled_is_property_not_method() -> None:
    """``is_enabled`` es ``@property`` (no metodo).

    Los 4 callers existentes la usan como propiedad (sin parentesis):
    - ``hermes/agent/loop.py:790``: ``if not self._embeddings_service.is_enabled``
    - ``hermes/receivers/http_api.py:1551``: ``getattr(..., "is_enabled", False)``
    - ``hermes/tools/search_files.py:74``: ``getattr(..., "is_enabled", False)``
    - ``hermes/services/embed_vault.py:185``: ``getattr(..., "is_enabled", False)``

    Si fuera metodo, ``getattr(..., "is_enabled", False)`` retornaria
    el bound method object (truthy) y el gate nunca bloquearia. R1
    cycle 8 BLOCKING fix.
    """
    router = EmbeddingRouter(
        backends={"a": FakeBackend(name="a", dim=384, enabled=True)},
        policies={
            EmbeddingPolicy.CHAT_RAG: PolicyConfig(
                use_case=EmbeddingPolicy.CHAT_RAG,
                tiers=["a"],
            ),
        },
    )
    # Acceder como property (sin parens) retorna bool.
    val = router.is_enabled
    assert isinstance(val, bool)
    assert val is True
    # Llamar como metodo (con parens) DEBE fallar — confirma que
    # es property, no metodo.
    with pytest.raises(TypeError):
        router.is_enabled()  # type: ignore[call-arg]
    # Tambien: getattr con el patron de los callers existentes
    # retorna bool, no un method object.
    got = getattr(router, "is_enabled", False)
    assert isinstance(got, bool)
    # Disabled: un router con 0 policies o todos los tiers disabled.
    empty_router = EmbeddingRouter(backends={}, policies={})
    assert empty_router.is_enabled is False
    # Tambien con `getattr(..., "is_enabled", False)` el default
    # es bool, no method.
    empty_default = getattr(empty_router, "is_enabled", False)
    assert isinstance(empty_default, bool)
    assert empty_default is False


@pytest.mark.asyncio
async def test_cache_vector_copied_on_put() -> None:
    """``_RouterCache.put`` copia el vector para evitar in-place mutation
    entre cache y caller (R1 cycle 8 X-2 MAJOR fix).

    Si el caller muta ``result.vector`` in-place (``result.vector /= norm``),
    el cache no debe verse afectado (y viceversa). Esto es un footgun
    latente — los callers actuales no mutan, pero futuros podrian.

    Demostramos el contrato verificando que el vector del cache es
    un objeto DISTINTO del original (identity check), no la misma
    referencia compartida. Esto previene cualquier in-place mutation
    de un lado u otro.
    """
    from hermes.services.embedding_router import _RouterCache

    original_vec = np.full(384, 0.5, dtype=np.float32)
    original = EmbeddingResult(
        vector=original_vec,
        source_tier="a",
        latency_ms=10.0,
        cost_estimate=0.0,
        cached=False,
    )
    cache = _RouterCache()
    cache.put("text-1", EmbeddingPolicy.CHAT_RAG, original)
    cached = cache.get("text-1", EmbeddingPolicy.CHAT_RAG)
    assert cached is not None
    # Sanity: ambos vectores tienen los mismos valores (deep equality).
    np.testing.assert_array_equal(cached.vector, original_vec)
    # El contrato clave: el vector del cache es un objeto DISTINTO
    # del original (no la misma referencia compartida). Esto es lo
    # que previene in-place mutation entre cache y caller.
    assert cached.vector is not original_vec
    # Verificacion adicional: ambos vectores pueden mutarse
    # independientemente. Como el array es mutable, mutamos el cache
    # in-place (cambia el contenido del array referenciado por
    # cached.vector) y comprobamos que el original NO se ve afectado
    # (porque apunta a OTRO array).
    cached.vector.fill(0.1)
    # El original sigue con sus valores originales (0.5), no 0.1.
    np.testing.assert_array_equal(original_vec, np.full(384, 0.5, dtype=np.float32))
    # El cache cambio.
    np.testing.assert_array_equal(cached.vector, np.full(384, 0.1, dtype=np.float32))
    # Y viceversa: mutamos el original, el cache NO se ve afectado.
    original_vec.fill(0.99)
    np.testing.assert_array_equal(cached.vector, np.full(384, 0.1, dtype=np.float32))
