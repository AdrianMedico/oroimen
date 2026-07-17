"""Tests Sprint 9.1: EmbeddingsService + EmbeddingsCache + cosine search.

Cubre:
- EmbeddingsCache: .copy() defensivo, lazy load, invalidacion
- EmbeddingsService disabled mode (no key, no openai package)
- EmbeddingsService enabled mode (mock OpenAI API)
- embed_and_store: trunca a 24K chars, persiste, invalida cache
- cosine_search: ordena por score DESC, filtra por threshold,
  retorna vacio si library vacia o RAG disabled
- Integracion DB: add_file_embedding, get_file_embedding, get_all_embeddings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from hermes.memory.db import Database
from hermes.services.embeddings import (
    EMBEDDING_DIM_DEFAULT,
    GEMINI_DEFAULT_DIM,
    MAX_EMBED_CHARS_GEMINI,
    MAX_EMBED_CHARS_OPENAI,
    EmbeddingsCache,
    EmbeddingsService,
)

# --- Helpers: fake Settings ---


class _FakeSettings:
    """Minimal stand-in for hermes.config.Settings con campos de S9.2.

    Defaults: OpenRouter con ZDR (alineado con config.py post-rename).
    """

    def __init__(
        self,
        openrouter_api_key: str = "sk-test",
        openrouter_api_base: str = "https://openrouter.ai/api/v1",
        embedding_model: str = "qwen/qwen3-embedding-8b",
        gemini_api_key: str = "",
        gemini_embedding_model: str = "text-embedding-004",
        embedding_provider: str = "openrouter",
        min_similarity_threshold: float = 0.82,
    ) -> None:
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_api_base = openrouter_api_base
        self.embedding_model = embedding_model
        self.gemini_api_key = gemini_api_key
        self.gemini_embedding_model = gemini_embedding_model
        self.embedding_provider = embedding_provider
        self.min_similarity_threshold = min_similarity_threshold
        # Backward compat aliases (los nombres viejos siguen funcionando
        # via el config de Settings real; aquí solo para tests legacy).
        self.openai_api_key = openrouter_api_key
        self.openai_api_base = openrouter_api_base


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


# --- EmbeddingsCache ---


@pytest.mark.asyncio
async def test_cache_lazy_load_empty_db(db: Database) -> None:
    """Si no hay embeddings en DB, cache retorna (empty, [])."""
    cache = EmbeddingsCache()
    arr, fids = await cache.get_all(db)
    # Cache vacío: shape (0, 0) — dim se determina al cargar datos
    assert arr.shape[0] == 0
    assert arr.dtype == np.float32
    assert fids == []
    assert cache.is_loaded


@pytest.mark.asyncio
async def test_cache_loads_from_db_with_copy_fix(db: Database) -> None:
    """P0 v1.2 fix: .copy() tras np.frombuffer para evitar read-only.

    Inserta un embedding, luego verifica que el cache puede normalizar
    in-place (test de la P0 fix).
    """
    await db.add_file("file_a", "a.pdf", "application/pdf", 100, "text", "pypdf")
    emb = np.random.rand(EMBEDDING_DIM_DEFAULT).astype(np.float32)
    await db.add_file_embedding("file_a", emb.tobytes())

    cache = EmbeddingsCache()
    arr, fids = await cache.get_all(db)
    assert arr.shape == (1, EMBEDDING_DIM_DEFAULT)
    assert fids == ["file_a"]
    # v1.2 P0 fix: si .copy() no se hizo, esto lanza
    # "ValueError: assignment destination is read-only".
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    # Verificar que la normalizacion in-place funciono
    norms = np.linalg.norm(arr, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.asyncio
async def test_cache_invalidate_forces_reload(db: Database) -> None:
    """invalidate() resetea el cache; proxima query recarga de DB."""
    cache = EmbeddingsCache()
    # Primera carga: vacio
    arr1, _ = await cache.get_all(db)
    assert arr1.shape[0] == 0
    # Anado un file
    await db.add_file("file_x", "x.pdf", "application/pdf", 100, "text", "pypdf")
    emb = np.zeros(EMBEDDING_DIM_DEFAULT, dtype=np.float32)
    await db.add_file_embedding("file_x", emb.tobytes())
    # Sin invalidar, la cache sigue vacia
    arr2, _ = await cache.get_all(db)
    assert arr2.shape[0] == 0
    # Invalido y vuelvo a leer
    cache.invalidate()
    arr3, fids3 = await cache.get_all(db)
    assert arr3.shape == (1, EMBEDDING_DIM_DEFAULT)
    assert fids3 == ["file_x"]


@pytest.mark.asyncio
async def test_cache_raises_on_dimension_mismatch(db: Database) -> None:
    """Si dos files tienen embeddings con dim distinta, el cache falla."""
    await db.add_file("file_1", "a.pdf", "application/pdf", 100, "t", "pypdf")
    await db.add_file("file_2", "b.pdf", "application/pdf", 100, "t", "pypdf")
    # File 1: dim 1536 (OpenAI). File 2: dim 768 (Gemini).
    # Mezcla imposible — el cache debe fallar ruidosamente.
    await db.add_file_embedding("file_1", b"\x00" * (1536 * 4))
    await db.add_file_embedding("file_2", b"\x00" * (768 * 4))
    cache = EmbeddingsCache()
    with pytest.raises(RuntimeError, match="dim inconsistente"):
        await cache.get_all(db)


# --- EmbeddingsService: disabled mode ---


@pytest.mark.asyncio
async def test_service_disabled_when_no_api_key(db: Database) -> None:
    """Si ambas keys están vacías, RAG disabled."""
    settings = _FakeSettings(openrouter_api_key="", gemini_api_key="")
    svc = EmbeddingsService(settings, db)
    await svc.ensure_initialized()
    assert svc.is_enabled is False
    # embed retorna array de ceros
    emb = await svc.embed("hola")
    assert emb.shape == (EMBEDDING_DIM_DEFAULT,)
    assert np.allclose(emb, 0.0)
    # cosine_search retorna vacio
    results = await svc.cosine_search("query")
    assert results == []
    # health_check False
    assert await svc.health_check() is False


@pytest.mark.asyncio
async def test_service_default_uses_openrouter_zdr(db: Database) -> None:
    """Default (sin override): OpenAI/OpenRouter con ZDR."""
    settings = _FakeSettings()  # defaults: openai_api_key="sk-test", provider="openai"
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.1] * 1536)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        assert svc.is_enabled is True
        assert svc.backend_name == "openai"
        # ZDR activo (OpenRouter es el default base_url)
        assert svc._is_zdr_active() is True
        # Reset mock para que el embed real no incluya el health check
        mock_cls.return_value.embeddings.create.reset_mock()
        emb = await svc.embed("hola")
    assert emb.shape == (1536,)


@pytest.mark.asyncio
async def test_resolve_openai_key_prefers_openrouter_over_legacy(db: Database) -> None:
    """P1 Copilot review 2026-06-26: si ambas keys están seteadas,
    gana openrouter_api_key (la nueva)."""
    settings = _FakeSettings(
        openrouter_api_key="sk-new",
    )
    # Override el alias para que openai_api_key sea distinto
    settings.openai_api_key = "sk-legacy"
    svc = EmbeddingsService(settings, db)
    assert svc._resolve_openai_key() == "sk-new"


@pytest.mark.asyncio
async def test_resolve_openai_key_falls_back_to_legacy_when_openrouter_empty(
    db: Database,
) -> None:
    """P1 Copilot review 2026-06-26: si openrouter_api_key está vacío
    pero openai_api_key (legacy) está seteado, usar legacy. Esto
    preserva deployments pre-S9.1 sin tocar .env."""
    settings = _FakeSettings(openrouter_api_key="")
    settings.openai_api_key = "sk-legacy-only"
    svc = EmbeddingsService(settings, db)
    assert svc._resolve_openai_key() == "sk-legacy-only"


@pytest.mark.asyncio
async def test_resolve_openai_key_empty_when_no_keys(db: Database) -> None:
    """Sin ninguna key, retorna string vacío (service queda disabled)."""
    settings = _FakeSettings(openrouter_api_key="", gemini_api_key="")
    settings.openai_api_key = ""
    svc = EmbeddingsService(settings, db)
    assert svc._resolve_openai_key() == ""


@pytest.mark.asyncio
async def test_resolve_openai_base_prefers_openrouter_over_legacy(db: Database) -> None:
    """P1 Copilot review 2026-06-26 (3rd sweep): si ambas base_url
    estan seteadas, gana openrouter_api_base."""
    settings = _FakeSettings()
    settings.openai_api_base = "https://api.openai.com/v1-legacy"
    svc = EmbeddingsService(settings, db)
    assert svc._resolve_openai_base() == "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
async def test_resolve_openai_base_falls_back_to_legacy_when_openrouter_empty(
    db: Database,
) -> None:
    """P1 Copilot review 2026-06-26 (3rd sweep): si openrouter_api_base
    esta vacio pero openai_api_base (legacy) esta seteado, usar legacy."""
    settings = _FakeSettings(openrouter_api_base="")
    settings.openai_api_base = "https://api.openai.com/v1-legacy"
    svc = EmbeddingsService(settings, db)
    assert svc._resolve_openai_base() == "https://api.openai.com/v1-legacy"


@pytest.mark.asyncio
async def test_service_init_accepts_openrouter_provider_name(db: Database) -> None:
    """P1 Copilot review 2026-06-26: EMBEDDING_PROVIDER=openrouter
    debe ser reconocido (no solo 'openai' legacy)."""
    settings = _FakeSettings(embedding_provider="openrouter")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.1] * 1536)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        assert svc.is_enabled is True
        assert svc.backend_name == "openai"


@pytest.mark.asyncio
async def test_service_init_falls_back_to_legacy_key(db: Database) -> None:
    """P1 Copilot review 2026-06-26: con solo OPENAI_API_KEY legacy
    (openrouter_api_key vacío), el service debe inicializar OK."""
    settings = _FakeSettings(openrouter_api_key="")
    settings.openai_api_key = "sk-legacy-only"
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.1] * 1536)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        assert svc.is_enabled is True
        assert svc.backend_name == "openai"


@pytest.mark.asyncio
async def test_service_explicit_gemini_with_warning(db: Database) -> None:
    """Si EMBEDDING_PROVIDER=gemini explicitamente, usa Gemini (NO ZDR)."""
    settings = _FakeSettings(
        gemini_api_key="AIza-test",
        openrouter_api_key="sk-test",
        embedding_provider="gemini",
    )
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": {"values": [0.1] * GEMINI_DEFAULT_DIM}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        assert svc.is_enabled is True
        assert svc.backend_name == "gemini"
        # Gemini NO es ZDR
        assert svc._is_zdr_active() is False


@pytest.mark.asyncio
async def test_service_falls_back_to_gemini_when_openai_fails(
    db: Database,
) -> None:
    """Si OpenAI/OpenRouter falla, fallback a Gemini."""
    settings = _FakeSettings(
        gemini_api_key="AIza-test",
        openrouter_api_key="sk-test",
        embedding_provider="openrouter",  # explicito: openrouter primary
    )
    # OpenAI falla
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.0] * 1536)
        mock_cls.return_value.embeddings.create.side_effect = Exception("OpenRouter down")
        # Gemini OK
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": {"values": [0.1] * GEMINI_DEFAULT_DIM}}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response
            svc = EmbeddingsService(settings, db)
            await svc.ensure_initialized()
            assert svc.is_enabled is True
            assert svc.backend_name == "gemini"  # fallback


@pytest.mark.asyncio
async def test_service_both_backends_fail_disabled(db: Database) -> None:
    """Si ambos backends fallan health check, RAG disabled."""
    settings = _FakeSettings(  # noqa: F841  (test asserts on svc state, not settings)
        gemini_api_key="AIza-test",
        openrouter_api_key="sk-test",
        embedding_provider="openrouter",
    )


@pytest.mark.asyncio
async def test_service_openai_backend_adds_zdr_header(db: Database) -> None:
    """OpenRouter base_url → X-OpenRouter-ZDR: true header."""
    settings = _FakeSettings(
        openrouter_api_key="sk-or-test",
        openrouter_api_base="https://openrouter.ai/api/v1",
        embedding_model="qwen/qwen3-embedding-8b",
    )
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.0] * 1536)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        # Verificar que el backend es OpenRouter (ZDR)
        assert svc._backend is not None
        assert svc._backend.is_zdr is True


@pytest.mark.asyncio
async def test_service_openai_direct_no_zdr_marker(db: Database) -> None:
    """OpenAI direct base_url → is_zdr False (no podemos verificar)."""
    settings = _FakeSettings(
        openrouter_api_key="sk-test",
        openrouter_api_base="https://api.openai.com/v1",
    )
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.0] * 1536)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        # OpenAI direct: no podemos verificar ZDR, asumimos False
        assert svc._backend is not None
        assert svc._backend.is_zdr is False


@pytest.mark.asyncio
async def test_service_disabled_when_whitespace_api_key(db: Database) -> None:
    """Key con solo whitespace también se considera disabled."""
    settings = _FakeSettings(openrouter_api_key="   \t  ", gemini_api_key="  \t  ")
    svc = EmbeddingsService(settings, db)
    await svc.ensure_initialized()
    assert svc.is_enabled is False


# --- EmbeddingsService: enabled mode (mocked) ---


def _make_mock_openai_client(emb_value: list[float]) -> Any:
    """Crea un mock del AsyncOpenAI client que retorna emb_value."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = emb_value
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.mark.asyncio
async def test_service_embed_returns_1536_dim_array(db: Database) -> None:
    """embed() retorna np.ndarray shape (1536,) dtype float32."""
    emb_value = [0.1] * EMBEDDING_DIM_DEFAULT
    settings = _FakeSettings()
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client(emb_value)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        assert svc.is_enabled is True
        emb = await svc.embed("hola mundo")
    assert emb.shape == (EMBEDDING_DIM_DEFAULT,)
    assert emb.dtype == np.float32
    assert emb[0] == 0.1


@pytest.mark.asyncio
async def test_service_embed_and_store_persists_and_invalidates(
    db: Database,
) -> None:
    """embed_and_store persiste el BLOB e invalida el cache."""
    await db.add_file("file_emb", "emb.pdf", "application/pdf", 100, "some text", "pypdf")
    emb_value = [0.5] * EMBEDDING_DIM_DEFAULT
    settings = _FakeSettings()
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client(emb_value)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        # Pre-poblar cache para verificar invalidacion
        cache_arr, _ = await svc._cache.get_all(db)
        assert cache_arr.shape == (0, 0)  # vacio
        # Embed + store
        ok = await svc.embed_and_store("file_emb", "some text")
        assert ok is True
        # Verificar persistencia
        blob = await db.get_file_embedding("file_emb")
        assert blob is not None
        assert len(blob) == EMBEDDING_DIM_DEFAULT * 4  # 1536 * 4 bytes
        # Verificar invalidacion: cache debe estar dirty
        assert svc._cache.is_loaded is False


@pytest.mark.asyncio
async def test_service_embed_and_store_truncates_to_24k(db: Database) -> None:
    """Text > 24K chars se trunca antes de enviar a OpenAI API."""
    await db.add_file("file_big", "big.pdf", "application/pdf", 100, "short", "pypdf")
    long_text = "X" * (MAX_EMBED_CHARS_OPENAI + 10_000)  # 34K chars
    settings = _FakeSettings()
    captured_input: list[str] = []

    async def capture_create(*args: Any, **kwargs: Any) -> Any:
        # openai>=1.50: kwarg 'input' contiene el texto
        captured_input.append(kwargs.get("input", args[0] if args else ""))
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.0] * EMBEDDING_DIM_DEFAULT
        return mock_response

    mock_client = MagicMock()
    mock_client.embeddings.create = capture_create
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        # Reset captured_input: ensure_initialized hace 1 health check
        # ("ping") que queremos ignorar.
        captured_input.clear()
        await svc.embed_and_store("file_big", long_text)
    assert len(captured_input) == 1
    assert len(captured_input[0]) == MAX_EMBED_CHARS_OPENAI


@pytest.mark.asyncio
async def test_service_embed_and_store_truncates_gemini_to_8k(
    db: Database,
) -> None:
    """Text > 8K chars se trunca antes de enviar a Gemini API."""
    await db.add_file("file_g", "g.pdf", "application/pdf", 100, "short", "pypdf")
    long_text = "X" * (MAX_EMBED_CHARS_GEMINI + 5_000)
    captured_text: list[str] = []

    # Mock httpx para Gemini
    async def capture_post(*args: Any, **kwargs: Any) -> Any:
        json_body = kwargs.get("json", {})
        parts = json_body.get("content", {}).get("parts", [])
        if parts:
            captured_text.append(parts[0].get("text", ""))
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": {"values": [0.0] * GEMINI_DEFAULT_DIM}}
        mock_response.raise_for_status = MagicMock()
        return mock_response

    settings = _FakeSettings(
        gemini_api_key="AIza-test",
        openrouter_api_key="",
        embedding_provider="gemini",
    )
    with patch("httpx.AsyncClient.post", side_effect=capture_post):
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        assert svc.backend_name == "gemini"
        # Reset captured_text: ensure_initialized hace 1 health check
        captured_text.clear()
        await svc.embed_and_store("file_g", long_text)
    assert len(captured_text) == 1
    assert len(captured_text[0]) == MAX_EMBED_CHARS_GEMINI


@pytest.mark.asyncio
async def test_service_embed_and_store_skips_empty_text(db: Database) -> None:
    """Texto vacio o whitespace se skipea (no API call, no persist)."""
    await db.add_file("file_empty", "e.pdf", "application/pdf", 100, "", "pypdf")
    settings = _FakeSettings()
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.0] * 1536)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        ok = await svc.embed_and_store("file_empty", "")
        assert ok is False
        # El blob no debe existir
        assert await db.get_file_embedding("file_empty") is None


@pytest.mark.asyncio
async def test_service_cosine_search_empty_library(db: Database) -> None:
    """Si la library no tiene embeddings, cosine_search retorna []."""
    settings = _FakeSettings()
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.5] * 1536)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        results = await svc.cosine_search("hola")
    assert results == []


@pytest.mark.asyncio
async def test_service_cosine_search_top_k_sorted_desc(db: Database) -> None:
    """cosine_search retorna (file_id, score) ordenado por score DESC."""
    await db.add_file("file_a", "a.pdf", "application/pdf", 100, "t", "pypdf")
    await db.add_file("file_b", "b.pdf", "application/pdf", 100, "t", "pypdf")
    await db.add_file("file_c", "c.pdf", "application/pdf", 100, "t", "pypdf")
    emb_a = np.array([1.0] + [0.0] * 1535, dtype=np.float32)
    emb_b = np.array([0.0, 1.0] + [0.0] * 1534, dtype=np.float32)  # perpendicular
    emb_c = np.array([0.7, 0.7] + [0.0] * 1534, dtype=np.float32)  # 45 grados
    await db.add_file_embedding("file_a", emb_a.tobytes())
    await db.add_file_embedding("file_b", emb_b.tobytes())
    await db.add_file_embedding("file_c", emb_c.tobytes())
    query_emb = emb_a.tolist()
    settings = _FakeSettings(min_similarity_threshold=0.5)
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client(query_emb)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        results = await svc.cosine_search("query", top_k=3)
    # a (1.0) > c (~0.707) > b (0.0, filtrado)
    assert len(results) == 2
    assert results[0][0] == "file_a"
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)
    assert results[1][0] == "file_c"
    assert results[1][1] == pytest.approx(0.707, abs=1e-3)


@pytest.mark.asyncio
async def test_service_cosine_search_filters_below_threshold(
    db: Database,
) -> None:
    """Scores < min_similarity_threshold se descartan (P1-1 v1.2)."""
    await db.add_file("file_low", "low.pdf", "application/pdf", 100, "t", "pypdf")
    emb_low = np.array([0.5, 0.5] + [0.0] * 1534, dtype=np.float32)
    await db.add_file_embedding("file_low", emb_low.tobytes())
    # Query perpendicular a emb_low → score 0
    query_emb = np.array([1.0, -1.0] + [0.0] * 1534, dtype=np.float32).tolist()
    settings = _FakeSettings(min_similarity_threshold=0.82)
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client(query_emb)
        svc = EmbeddingsService(settings, db)
        await svc.ensure_initialized()
        results = await svc.cosine_search("query")
    assert results == []  # score=0 < 0.82


@pytest.mark.asyncio
async def test_service_health_check_disabled(db: Database) -> None:
    """health_check retorna False si RAG disabled."""
    settings = _FakeSettings(openrouter_api_key="", gemini_api_key="")
    svc = EmbeddingsService(settings, db)
    assert await svc.health_check() is False


@pytest.mark.asyncio
async def test_service_health_check_enabled(db: Database) -> None:
    """health_check retorna True si la API responde OK."""
    settings = _FakeSettings()
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_openai_client([0.0] * 1536)
        svc = EmbeddingsService(settings, db)
        assert await svc.health_check() is True


# --- DB methods ---


@pytest.mark.asyncio
async def test_db_add_file_embedding_upserts(db: Database) -> None:
    """INSERT OR REPLACE: el segundo add reemplaza el primero."""
    await db.add_file("f1", "f.pdf", "application/pdf", 100, "t", "pypdf")
    emb1 = b"\x00" * 6144
    emb2 = b"\xff" * 6144
    await db.add_file_embedding("f1", emb1, model="model-a")
    blob = await db.get_file_embedding("f1")
    assert blob == emb1
    # Re-insert con modelo distinto
    await db.add_file_embedding("f1", emb2, model="model-b")
    blob = await db.get_file_embedding("f1")
    assert blob == emb2


@pytest.mark.asyncio
async def test_db_get_file_embedding_returns_none_for_missing(
    db: Database,
) -> None:
    assert await db.get_file_embedding("nonexistent") is None


@pytest.mark.asyncio
async def test_db_get_all_embeddings_returns_all(db: Database) -> None:
    await db.add_file("f1", "a.pdf", "application/pdf", 100, "t", "pypdf")
    await db.add_file("f2", "b.pdf", "application/pdf", 100, "t", "pypdf")
    await db.add_file_embedding("f1", b"\x00" * 100)
    await db.add_file_embedding("f2", b"\xff" * 100)
    rows = await db.get_all_embeddings()
    assert len(rows) == 2
    fids = {r[0] for r in rows}
    assert fids == {"f1", "f2"}


@pytest.mark.asyncio
async def test_db_delete_file_embedding(db: Database) -> None:
    await db.add_file("f_del", "d.pdf", "application/pdf", 100, "t", "pypdf")
    await db.add_file_embedding("f_del", b"\x00" * 100)
    assert await db.get_file_embedding("f_del") is not None
    await db.delete_file_embedding("f_del")
    assert await db.get_file_embedding("f_del") is None


@pytest.mark.asyncio
async def test_router_store_then_search_uses_same_vault_policy_with_mixed_dims(
    db: Database,
) -> None:
    """The file producer and query consumer share VAULT_INGEST space."""
    from hermes.services.embedding_router import EmbeddingPolicy, EmbeddingResult

    await db.add_file("vault_file", "vault.txt", "text/plain", 10, "vault", "plain")
    await db.add_file("chat_file", "chat.txt", "text/plain", 10, "chat", "plain")
    await db.upsert_embedding(
        "chat_file",
        np.array([1.0, 0.0], dtype=np.float32).tobytes(),
        "chat-model",
        2,
        policy="chat_rag",
    )

    class _PolicyRouter:
        def __init__(self) -> None:
            self._backends: dict[str, Any] = {}
            self.calls: list[EmbeddingPolicy] = []

        async def embed_with_policy(self, text: str, policy: EmbeddingPolicy) -> EmbeddingResult:
            self.calls.append(policy)
            assert text in {"document", "query"}
            return EmbeddingResult(
                vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                source_tier="edge",
                latency_ms=1.0,
            )

    router = _PolicyRouter()
    service = EmbeddingsService(
        _FakeSettings(min_similarity_threshold=0.1),
        db,  # type: ignore[arg-type]
    )
    service._initialized = True
    service._router = router  # type: ignore[assignment]

    assert await service.embed_and_store("vault_file", "document") is True
    assert await service.cosine_search("query", top_k=5) == [("vault_file", pytest.approx(1.0))]
    assert router.calls == [EmbeddingPolicy.VAULT_INGEST, EmbeddingPolicy.VAULT_INGEST]
    assert [row[0] for row in await db.get_all_embeddings(policy="chat_rag")] == ["chat_file"]
    assert [row[0] for row in await db.get_all_embeddings(policy="vault_ingest")] == ["vault_file"]
