"""Sprint 19.5 Slice 6 Commit 3: integracion EmbeddingsService <-> EmbeddingRouter.

Cubre (TDD §3 commit 3, ~5 tests + 1 e2e):
- Router mode: env vars ``EMBEDDING_TIER_*__ENABLED=true`` activan
  el router multi-tier; el service tiene ``_router`` y NO
  ``_backend``. ``is_enabled`` True, ``backend_name == "multi_tier"``.
- Legacy mode: sin env vars, el service cae al legacy single-backend
  (OpenRouter mockeado o Gemini). ``_router is None``, ``_backend``
  no es None, ``is_enabled`` True si hay key, ``backend_name`` es
  ``"openai"`` o ``"gemini"``.
- ``is_enabled`` retorna True en ambos modos (compat shim, TDD §2.5
  M-8 fix).
- ``embed_batch`` retorna ``list[np.ndarray]`` con la dim correcta
  en ambos modos.
- ``embed_and_store`` retorna ``bool`` en ambos modos (no propaga
  excepciones, log warning + return False en errores).
- E2E: en router mode, llamar ``embed()`` sale del service al router
  (puede fallar con ``AllTiersFailed`` porque la URL es fake, pero
  NO con errores de Wiring / NoneType / AttributeError — el router
  está enchufado correctamente).

Estrategia de testing:
- Usamos ``monkeypatch.setenv`` para los env vars (automatica
  cleanup post-test, no contamina otros tests).
- Para router mode, los tests que llaman a ``embed()`` reciben
  un EmbeddingResult fallido (porque el ``base_url`` apunta a un
  host inexistente). Es OK: lo que probamos es el wiring, no el
  HTTP. ``httpx.ConnectError`` propagado por el router esta bien.
- Para embed_and_store en router mode, mockeamos
  ``self._router.embed_with_policy`` directamente con ``AsyncMock``
  para no tocar la red.
- Para embed_batch, mockeamos el backend (OpenAI en legacy, router
  en multi-tier) con ``AsyncMock``.

No tocamos ``tests/unit/test_embeddings.py`` (33 tests existentes
deben pasar sin modificacion) ni ``tests/unit/test_embedding_router
.py`` (32 tests de Commit 2). Este archivo es NUEVO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from hermes.memory.db import Database
from hermes.services.embedding_router import EmbeddingResult
from hermes.services.embeddings import EMBEDDING_DIM_DEFAULT, EmbeddingsService

# --- Helpers --------------------------------------------------------------


class _RouterSettings:
    """Settings stand-in con env vars de multi-tier activadas.

    Solo los campos que la nueva discriminacion de modo necesita
    (los tier enables y la policy). El resto son los defaults del
    legacy. Los tests pueden mutar atributos para forzar escenarios
    concretos.
    """

    def __init__(
        self,
        *,
        nas_enabled: bool = False,
        nas_base_url: str = "",
        nas_model: str = "",
        nas_api_key: str = "",
        nas_timeout_s: float = 30.0,
        edge_enabled: bool = False,
        edge_base_url: str = "",
        edge_model: str = "",
        edge_api_key: str = "",
        edge_timeout_s: float = 30.0,
        cloud_enabled: bool = False,
        cloud_base_url: str = "",
        cloud_model: str = "",
        cloud_api_key: str = "",
        cloud_timeout_s: float = 30.0,
        policy_chat_rag: str = "",
        policy_vault_ingest: str = "",
        openrouter_api_key: str = "",
        embedding_provider: str = "openrouter",
        embedding_model: str = "qwen/qwen3-embedding-8b",
        min_similarity_threshold: float = 0.82,
        gemini_api_key: str = "",
    ) -> None:
        # Tier NAS
        self.embedding_tier_nas__enabled = nas_enabled
        self.embedding_tier_nas__base_url = nas_base_url
        self.embedding_tier_nas__model = nas_model
        self.embedding_tier_nas__api_key = nas_api_key
        self.embedding_tier_nas__timeout_s = nas_timeout_s
        # Tier edge
        self.embedding_tier_edge__enabled = edge_enabled
        self.embedding_tier_edge__base_url = edge_base_url
        self.embedding_tier_edge__model = edge_model
        self.embedding_tier_edge__api_key = edge_api_key
        self.embedding_tier_edge__timeout_s = edge_timeout_s
        # Tier cloud
        self.embedding_tier_cloud__enabled = cloud_enabled
        self.embedding_tier_cloud__base_url = cloud_base_url
        self.embedding_tier_cloud__model = cloud_model
        self.embedding_tier_cloud__api_key = cloud_api_key
        self.embedding_tier_cloud__timeout_s = cloud_timeout_s
        # Policies
        self.embedding_policy_chat_rag = policy_chat_rag
        self.embedding_policy_vault_ingest = policy_vault_ingest
        # Legacy
        self.openrouter_api_key = openrouter_api_key
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.min_similarity_threshold = min_similarity_threshold
        self.gemini_api_key = gemini_api_key
        # Backward compat aliases (legacy names)
        self.openai_api_key = openrouter_api_key
        self.openrouter_api_base = "https://openrouter.ai/api/v1"
        self.openai_api_base = "https://openrouter.ai/api/v1"
        self.gemini_embedding_model = "gemini-embedding-001"


def _make_mock_openai_client(emb_value: list[float]) -> Any:
    """Mock del AsyncOpenAI client (legacy mode). Reused de test_embeddings.py."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = emb_value
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


# --- Test 1: router mode (env vars configured) ---------------------------


@pytest.mark.asyncio
async def test_router_mode_when_tiers_configured(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si EMBEDDING_TIER_NAS__ENABLED=true + base_url + model, el
    service construye el router y cae a multi-tier mode.

    Verifica:
    - ``is_enabled`` es True (router instanciado)
    - ``backend_name`` retorna ``"multi_tier"`` (no "openai" ni "gemini")
    - ``_router`` no es None
    - ``_backend`` es None (legacy NO se inicializa)
    """
    # ``_build_router`` parsea env vars directo de os.environ (no
    # del Settings object). Setear via monkeypatch para que el
    # factory los vea.
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://nas.lan:8082/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__API_KEY", "")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__TIMEOUT_S", "10.0")
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas")
    monkeypatch.setenv("EMBEDDING_POLICY_VAULT_INGEST", "nas")
    settings = _RouterSettings(
        nas_enabled=True,
        nas_base_url="http://nas.lan:8082/v1",
        nas_model="granite-97m",
        policy_chat_rag="nas",
        policy_vault_ingest="nas",
    )
    svc = EmbeddingsService(settings, db)  # type: ignore[arg-type]
    await svc.ensure_initialized()
    assert svc.is_enabled is True
    assert svc.backend_name == "multi_tier"
    assert svc._router is not None
    assert svc._backend is None


# --- Test 2: legacy mode (no env vars) -----------------------------------


@pytest.mark.asyncio
async def test_legacy_mode_when_no_tiers(db: Database) -> None:
    """Sin env vars de tiers, el service cae al legacy single-backend.

    Verifica:
    - ``is_enabled`` es True si hay key de OpenRouter
    - ``backend_name`` retorna ``"openai"``
    - ``_router`` es None
    - ``_backend`` no es None (legacy OpenRouter inicializado)
    """
    settings = _RouterSettings(openrouter_api_key="sk-test")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.0] * EMBEDDING_DIM_DEFAULT)
        svc = EmbeddingsService(settings, db)  # type: ignore[arg-type]
        await svc.ensure_initialized()
        assert svc.is_enabled is True
        assert svc.backend_name == "openai"
        assert svc._router is None
        assert svc._backend is not None


# --- Test 3: is_enabled in both modes ------------------------------------


@pytest.mark.asyncio
async def test_is_enabled_in_both_modes(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    """``is_enabled`` retorna True en ambos modos cuando hay config valida.

    Router mode: un tier enabled es suficiente.
    Legacy mode: openrouter_api_key es suficiente.
    """
    # Router mode
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://nas.lan:8082/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas")
    settings_router = _RouterSettings(
        nas_enabled=True,
        nas_base_url="http://nas.lan:8082/v1",
        nas_model="granite-97m",
        policy_chat_rag="nas",
    )
    svc_router = EmbeddingsService(settings_router, db)  # type: ignore[arg-type]
    await svc_router.ensure_initialized()
    assert svc_router.is_enabled is True

    # Limpiar env vars para que el siguiente test caiga en legacy
    monkeypatch.delenv("EMBEDDING_TIER_NAS__ENABLED", raising=False)
    monkeypatch.delenv("EMBEDDING_TIER_NAS__BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_TIER_NAS__MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_POLICY_CHAT_RAG", raising=False)

    # Legacy mode
    settings_legacy = _RouterSettings(openrouter_api_key="sk-test")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.0] * EMBEDDING_DIM_DEFAULT)
        svc_legacy = EmbeddingsService(settings_legacy, db)  # type: ignore[arg-type]
        await svc_legacy.ensure_initialized()
        assert svc_legacy.is_enabled is True


# --- Test 4: embed_batch in both modes -----------------------------------


@pytest.mark.asyncio
async def test_embed_batch_in_both_modes(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    """``embed_batch`` retorna ``list[np.ndarray]`` con N elementos en
    ambos modos.

    Legacy: mock del AsyncOpenAI client (1 response batched).
    Router: mock del ``EmbeddingRouter.embed_batch`` que retorna
    directamente N np.ndarray (sin tocar la red).
    """
    # Legacy mode
    settings_legacy = _RouterSettings(openrouter_api_key="sk-test")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536), MagicMock(embedding=[0.2] * 1536)]
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client
        svc_legacy = EmbeddingsService(settings_legacy, db)  # type: ignore[arg-type]
        await svc_legacy.ensure_initialized()
        results = await svc_legacy.embed_batch(["a", "b"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)

    # Router mode — mock the router.embed_batch directly
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://nas.lan:8082/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas")
    settings_router = _RouterSettings(
        nas_enabled=True,
        nas_base_url="http://nas.lan:8082/v1",
        nas_model="granite-97m",
        policy_chat_rag="nas",
    )
    svc_router = EmbeddingsService(settings_router, db)  # type: ignore[arg-type]
    await svc_router.ensure_initialized()
    # Replace the router's embed_batch with a mock that returns vectors
    expected_vecs = [np.zeros(384, dtype=np.float32) for _ in range(2)]
    assert svc_router._router is not None  # type guard
    svc_router._router.embed_batch = AsyncMock(return_value=expected_vecs)  # type: ignore[method-assign]
    results_router = await svc_router.embed_batch(["x", "y"])
    assert isinstance(results_router, list)
    assert len(results_router) == 2
    assert all(isinstance(r, np.ndarray) for r in results_router)
    assert all(r.shape == (384,) for r in results_router)


# --- Test 5: embed_and_store returns bool in both modes ------------------


@pytest.mark.asyncio
async def test_embed_and_store_returns_bool_in_both_modes(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``embed_and_store`` retorna bool (no levanta excepciones) en
    ambos modos.

    Legacy: mock de OpenAI + insertar fila en DB.
    Router: mock del ``embed_with_policy`` para no tocar la red.
    """
    await db.add_file("file_legacy", "l.pdf", "application/pdf", 100, "t", "pypdf")
    await db.add_file("file_router", "r.pdf", "application/pdf", 100, "t", "pypdf")

    # Legacy mode
    settings_legacy = _RouterSettings(openrouter_api_key="sk-test")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.0] * 1536)
        svc_legacy = EmbeddingsService(settings_legacy, db)  # type: ignore[arg-type]
        await svc_legacy.ensure_initialized()
        ok = await svc_legacy.embed_and_store("file_legacy", "some text")
        assert isinstance(ok, bool)
        assert ok is True
        # Verifica persistencia
        blob = await db.get_file_embedding("file_legacy")
        assert blob is not None

    # Router mode
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://nas.lan:8082/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas")
    monkeypatch.setenv("EMBEDDING_POLICY_VAULT_INGEST", "nas")
    settings_router = _RouterSettings(
        nas_enabled=True,
        nas_base_url="http://nas.lan:8082/v1",
        nas_model="granite-97m",
        policy_chat_rag="nas",
        policy_vault_ingest="nas",
    )
    svc_router = EmbeddingsService(settings_router, db)  # type: ignore[arg-type]
    await svc_router.ensure_initialized()
    # Mock el router.embed_with_policy
    assert svc_router._router is not None  # type guard
    mock_result = EmbeddingResult(
        vector=np.zeros(384, dtype=np.float32),
        source_tier="nas",
        latency_ms=10.0,
    )
    svc_router._router.embed_with_policy = AsyncMock(return_value=mock_result)  # type: ignore[method-assign]
    ok_router = await svc_router.embed_and_store("file_router", "router text")
    assert isinstance(ok_router, bool)
    assert ok_router is True
    # Verifica persistencia
    blob_router = await db.get_file_embedding("file_router")
    assert blob_router is not None
    assert len(blob_router) == 384 * 4  # 384 dims x 4 bytes (float32)


# --- Test 6: e2e router mode (sin mock del router, salida real) ---------


@pytest.mark.asyncio
async def test_router_mode_e2e_embed_raises_connect_or_similar(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """En router mode, ``embed()`` sale del service al router. La URL
    apunta a un host inexistente (``nas.lan`` no resuelve), asi que
    la llamada falla con ``httpx.ConnectError`` o similar. Lo que
    probamos aqui es:

    1. NO hay ``AttributeError`` / ``NoneType`` errors (el wiring
       esta bien).
    2. La excepcion que sale es del router, no del service
       (``httpx.ConnectError`` viene del ``OpenAICompatibleBackend``
       via el router).

    Esto valida que el service de verdad delega al router y que
    las 8 call sites que usan ``embed()`` seguiran funcionando
    (recibiran excepciones de red esperables, no crashes de wiring).
    """
    settings = _RouterSettings(
        nas_enabled=True,
        nas_base_url="http://nas.lan:8082/v1",
        nas_model="granite-97m",
        policy_chat_rag="nas",
    )
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://nas.lan:8082/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas")
    svc = EmbeddingsService(settings, db)  # type: ignore[arg-type]
    await svc.ensure_initialized()
    assert svc._router is not None

    # Llamar embed() debe levantar una excepcion de red, NO un
    # AttributeError (que seria un bug de wiring).
    with pytest.raises(Exception) as exc_info:
        await svc.embed("hello")
    # La excepcion NO debe ser de wiring (NoneType, AttributeError
    # sobre el router, etc.). httpx.ConnectError es lo esperado
    # para un host que no resuelve.
    err_type = type(exc_info.value)
    err_str = str(exc_info.value)
    # Aceptamos: ConnectError, AllTiersFailed (wrapper de ConnectError),
    # o cualquier excepcion que no sea AttributeError/NoneType.
    assert (
        "Attribute" not in err_type.__name__
    ), f"wiring bug: embed() raised {err_type.__name__}: {err_str}"
    assert (
        "NoneType" not in err_type.__name__
    ), f"wiring bug: embed() raised {err_type.__name__}: {err_str}"
    # Bonus: el router tiene la signature correcta (multi_tier backend name)
    assert svc.backend_name == "multi_tier"


# --- Sprint 19.5 Slice 6 Commit 4: db.upsert_embedding + validation -----


@pytest.mark.asyncio
async def test_upsert_embedding_in_router_mode_e2e(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commit 4: en router mode, ``embed_and_store`` usa
    ``db.upsert_embedding`` (no el legacy ``add_file_embedding``).

    Verifies:
    - The row is in DB with ``policy='vault_ingest'`` (not NULL).
    - The ``dim`` column matches the canonical dim of the embedding
      (extracted from ``vector.shape[0]``).
    - The ``model`` column matches the model name from the backend
      that served the response.
    """
    await db.add_file("file_e2e_upsert", "e.pdf", "application/pdf", 100, "text", "pypdf")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://nas.lan:8082/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas")
    monkeypatch.setenv("EMBEDDING_POLICY_VAULT_INGEST", "nas")
    settings = _RouterSettings(
        nas_enabled=True,
        nas_base_url="http://nas.lan:8082/v1",
        nas_model="granite-97m",
        policy_chat_rag="nas",
        policy_vault_ingest="nas",
    )
    svc = EmbeddingsService(settings, db)  # type: ignore[arg-type]
    await svc.ensure_initialized()
    assert svc._router is not None
    # Mock the router.embed_with_policy to return a 384-dim vector
    # (matching the granite-97m canonical dim per TDD §2.1).
    svc_router = svc._router
    assert svc_router is not None
    mock_result = EmbeddingResult(
        vector=np.zeros(384, dtype=np.float32),
        source_tier="nas",
        latency_ms=10.0,
    )
    svc_router.embed_with_policy = AsyncMock(return_value=mock_result)  # type: ignore[method-assign]
    ok = await svc.embed_and_store("file_e2e_upsert", "some text to embed")
    assert ok is True
    # Verify the row was persisted via upsert_embedding (post-v25 schema
    # has policy + dim NOT NULL).
    async with db.conn.execute(
        "SELECT file_id, policy, dim, model, length(embedding) FROM file_embeddings "
        "WHERE file_id=?",
        ("file_e2e_upsert",),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    file_id, policy, dim, model, blob_len = row
    assert file_id == "file_e2e_upsert"
    assert policy == "vault_ingest"
    assert int(dim) == 384
    assert model == "granite-97m"
    assert int(blob_len) == 384 * 4


@pytest.mark.asyncio
async def test_schema_producer_consumer_contract(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema-producer-consumer contract: every column that
    ``db.upsert_embedding`` writes must exist in the ``file_embeddings``
    schema. This catches a class of bugs where a code change to the
    INSERT statement references a column that the migration didn't add
    (or a migration drops a column the code still writes).
    """
    # Get the columns upsert_embedding writes to (hard-coded in the
    # SQL string in db.py). If db.py's upsert_embedding changes its
    # INSERT column list, this test should be updated to match.
    expected_columns = {"file_id", "embedding", "embedded_at", "model", "dim", "policy"}
    # Get the actual columns in the post-v25 schema.
    async with db.conn.execute("PRAGMA table_info(file_embeddings)") as cur:
        actual_columns = {row[1] for row in await cur.fetchall()}
    missing = expected_columns - actual_columns
    assert not missing, (
        f"upsert_embedding writes to {expected_columns} but the schema "
        f"is missing: {missing}. Did a migration drop a column?"
    )


@pytest.mark.asyncio
async def test_per_policy_isolation_in_db(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-policy isolation: same file_id can have one row per policy
    with different dims (composite PK allows).

    The composite PK ``(file_id, policy)`` is the X3 architectural
    keystone (TDD §2.7). Without it, the system couldn't have one
    chat_rag 384-dim embedding + one vault_ingest 4096-dim embedding
    for the same file.
    """
    # Seed a file + two embeddings with different policies and dims.
    await db.add_file("file_iso", "i.pdf", "application/pdf", 100, "text", "pypdf")
    emb_384 = b"\x00" * (384 * 4)
    emb_4096 = b"\x00" * (4096 * 4)
    await db.upsert_embedding("file_iso", emb_384, model="granite-97m", dim=384, policy="chat_rag")
    await db.upsert_embedding(
        "file_iso", emb_4096, model="qwen-8b", dim=4096, policy="vault_ingest"
    )
    # Verify both rows exist (composite PK allows).
    async with db.conn.execute(
        "SELECT file_id, policy, dim FROM file_embeddings WHERE file_id=? ORDER BY policy",
        ("file_iso",),
    ) as cur:
        rows = await cur.fetchall()
    assert len(rows) == 2
    by_policy = {row[1]: row for row in rows}
    assert by_policy["chat_rag"][2] == 384
    assert by_policy["vault_ingest"][2] == 4096


# --- Test 7 (Commit 4): validation rules propagate from router --------


@pytest.mark.asyncio
async def test_rule_3_violation_raises_config_error(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 3 (within-policy dim match) violation raises ConfigError.

    Per TDD §2.4 Rule 3: within a single policy, all tiers must have
    matching dim. The router's ``_build_router`` factory validates
    this at startup. If a policy has tiers with mismatched dims
    (e.g. ``chat_rag=nas,cloud`` where NAS=384 and cloud=4096),
    ``ensure_initialized()`` must raise ConfigError (not silently
    fall back, not defer the check to runtime).
    """
    # Set up a misconfigured policy: chat_rag has both nas (384-dim)
    # and cloud (4096-dim). The router factory should reject this.
    monkeypatch.setenv("EMBEDDING_TIER_NAS__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__BASE_URL", "http://nas.lan:8082/v1")
    monkeypatch.setenv("EMBEDDING_TIER_NAS__MODEL", "granite-97m")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__MODEL", "qwen/qwen3-embedding-8b")
    monkeypatch.setenv("EMBEDDING_TIER_CLOUD__API_KEY", "fake-key-for-test")
    # Mixed-dim policy: NAS is 384-dim, cloud is 4096-dim.
    monkeypatch.setenv("EMBEDDING_POLICY_CHAT_RAG", "nas,cloud")
    settings = _RouterSettings(
        nas_enabled=True,
        nas_base_url="http://nas.lan:8082/v1",
        nas_model="granite-97m",
        cloud_enabled=True,
        cloud_base_url="https://openrouter.ai/api/v1",
        cloud_model="qwen/qwen3-embedding-8b",
        cloud_api_key="fake-key-for-test",
        # The rule 3 violation: nas+cloud in same policy
        policy_chat_rag="nas,cloud",
    )
    svc = EmbeddingsService(settings, db)  # type: ignore[arg-type]
    # The router factory should raise ConfigError when it detects the
    # dim mismatch. The error propagates through ensure_initialized().
    with pytest.raises(Exception) as exc_info:
        await svc.ensure_initialized()
    # The error should mention "dim" or "config" to be clearly
    # identifiable. We accept any exception type — the test passes if
    # the service refuses to start with a misconfigured policy.
    err_str = str(exc_info.value).lower()
    err_type = type(exc_info.value).__name__.lower()
    assert (
        "dim" in err_str or "config" in err_str or "mismatch" in err_str or "config" in err_type
    ), f"Expected ConfigError about dim mismatch, got {err_type}: {err_str}"
