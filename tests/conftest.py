"""Pytest configuration and shared fixtures.

Contiene:
- Hook `pytest_sessionfinish` para evitar hang de `threading._shutdown()`
  en GitHub Actions (ver `docs/POSTMORTEM_CI_HANG.md`).
- Fixtures compartidas para tests de handlers de Telegram: `bot`,
  `settings`, `db`, `telemetry`.
- Helpers para construir mocks de `aiogram.Message` (la clase real es
  pydantic frozen, lo que impide reasignar `answer`).
- Helper para extraer el callable de un Router de aiogram.
- Markers: `@pytest.mark.slow` para tests >1s (deselect en CI con
  `-m "not slow"` para mantener la suite <60s; correlos con
  `pytest -m slow` o `pytest --runslow` cuando necesites validar
  esos flujos end-to-end con tiempos reales).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from aiogram import Bot
from aiogram.types import Chat, User

from hermes.config import Settings
from hermes.memory.db import Database
from hermes.telemetry import Telemetry

# ---------------------------------------------------------------------------
# CI hang workaround
# ---------------------------------------------------------------------------

# Detect if running in GitHub Actions
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def pytest_sessionfinish(session, exitstatus):
    """Force immediate process exit to avoid threading._shutdown() hang.

    GitHub Actions runners hang in threading._shutdown() when asyncio libraries
    (aiogram, aiohttp, httpx) create daemon threads that don't close cleanly.
    Pytest finishes successfully (2-3s, all tests pass) but the Python process
    can't exit because _shutdown() waits for daemon threads.

    os._exit(0) bypasses cleanup and threading._shutdown, allowing the job
    to terminate cleanly.

    Reference: https://github.com/actions/runner/issues/3535
    """
    if IN_GITHUB_ACTIONS:
        os._exit(0)


# ---------------------------------------------------------------------------
# Constantes de test compartidas
# ---------------------------------------------------------------------------

TEST_USER_ID = 12345
TEST_CHAT_ID = 67890
TEST_BOT_TOKEN = "9999999999:AAFakeTestTokenForUnitTests12345"


# ---------------------------------------------------------------------------
# Fixtures compartidas (handlers, db, settings, etc.)
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Settings con credenciales fake y DB en tmp_path.

    Reutilizada por `test_messages.py`, `test_commands.py`, `test_main.py`,
    `test_router.py` y demás.

    El smart routing por defecto es OpenAI-compatible (deepseek-v4-flash →
    minimax-m3) para que los tests que validan paths concretos
    (OpenAI/Anthropic) no se vean afectados por la inversión de prioridad
    de v1.2 (que es de producción). Tests específicos del nuevo
    comportamiento v1.2 usan el fixture `settings_v12`.

    LLM_ALLOWED_MODELS override (Sprint 12+): superset con modelos legacy
    + MiniMax para que tests que usan nombres concretos no se rompan al
    cambiar la lista oficial.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TEST_BOT_TOKEN)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    # Superset: legacy OpenCode-Go + MiniMax Sprint 12+
    monkeypatch.setenv(
        "LLM_ALLOWED_MODELS",
        '["deepseek-v4-flash","minimax-m3","minimax-m2",'
        '"MiniMax-M3","MiniMax-M2.7","MiniMax-M2.7-highspeed",'
        '"kimi-k2.6","qwen3.7-plus","mimo-v2-omni"]',
    )
    # v1.0 chain (OpenAI-first) para compatibilidad con tests pre-existentes
    monkeypatch.setenv("LLM_TEXT_PRIMARY", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_TEXT_FALLBACK", "minimax-m3")
    monkeypatch.setenv("LLM_VOICE_PRIMARY", "deepseek-v4-flash")
    # Sprint 19.6+ Phase 5: this fixture is for tests that exercise
    # the LEGACY MiniMax / OpenCode-Go path. The default
    # `LLM_TEXT_PRIMARY_PROVIDER` is now "ollama" (local-first);
    # these tests need to explicitly pin it to "minimax" so the
    # router dispatches to the main httpx client (talking to
    # api.minimax.io) instead of the Ollama client. Without this
    # override, the tests' `deepseek-v4-flash` and `minimax-m3`
    # primary models would be routed to a non-existent local
    # Ollama server, breaking the test mocks.
    monkeypatch.setenv("LLM_TEXT_PRIMARY_PROVIDER", "minimax")
    return Settings(_env_file=None)


@pytest.fixture
def settings_v12(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Settings con el smart routing unificado de v1.2.

    Chain: minimax-m3 (primary) → deepseek-v4-flash (fallback).
    voice_chain == text_chain (unificado tras bug #30389).
    Usado por tests específicos del nuevo comportamiento.

    LLM_ALLOWED_MODELS override: superset legacy + MiniMax para no
    depender de la lista oficial.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TEST_BOT_TOKEN)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv(
        "LLM_ALLOWED_MODELS",
        '["deepseek-v4-flash","minimax-m3","minimax-m2",'
        '"MiniMax-M3","MiniMax-M2.7","MiniMax-M2.7-highspeed",'
        '"kimi-k2.6","qwen3.7-plus","mimo-v2-omni"]',
    )
    monkeypatch.setenv("LLM_TEXT_PRIMARY", "minimax-m3")
    monkeypatch.setenv("LLM_TEXT_FALLBACK", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_VOICE_PRIMARY", "minimax-m3")
    # Sprint 19.6+ Phase 5: pin provider to "minimax" so the v1.2
    # primary (`minimax-m3`) is dispatched to the main httpx
    # client, not the Ollama client. See comment in the `settings`
    # fixture above for the full rationale.
    monkeypatch.setenv("LLM_TEXT_PRIMARY_PROVIDER", "minimax")
    return Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Performance: backoff casi cero en TODOS los tests
# ---------------------------------------------------------------------------
#
# En producción el router espera 0.5s/1.0s/2.0s entre reintentos para no
# martillear un provider caido. En tests, esos sleeps suman ~3s por test
# que falla con retry (hay 5+ tests asi). Bajamos el backoff a 1ms en
# tests via este fixture autouse. NO afecta a producción: solo cambia
# el default class-level de Settings, y los tests crean Settings dentro
# de su scope (el monkeypatch se revierte al final del test).
#
# Si en el futuro hay tests que validan el backoff en sí (timing), pueden
# sobreescribir el setting con un fixture explicito o restaurar el
# default manualmente.


@pytest.fixture(autouse=True)
def _fast_retry_backoff_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce el backoff de retry a 1ms en todos los tests (autouse).

    Speedup estimado: -2 a -3s en los 5+ tests de TestRetryLogic y
    TestFallbackChain que actualmente gastan tiempo en sleeps reales.

    Implementacion: setea LLM_RETRY_BACKOFF_BASE=0.001 como env var.
    Pydantic-settings lo lee al instanciar Settings, asi que cualquier
    Settings() creado en el test tendra el backoff reducido. El
    monkeypatch.setattr no funciona porque pydantic v2 no expone los
    fields como atributos class-level mutables.
    """
    monkeypatch.setenv("LLM_RETRY_BACKOFF_BASE", "0.001")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Añade flag --runslow para correr tests @pytest.mark.slow.

    Por defecto, los tests @slow NO se corren (es el comportamiento
    estándar de pytest para markers custom). Con --runslow o -m slow
    sí se incluyen. Esto evita gastar 5s+ en tests con esperas reales
    en CI por defecto.

    Tambien añade --runnetwork para tests @pytest.mark.network (smoke
    tests contra hermes-deploy en LAN). Sprint 15 issue #74 (Nemotron
    review de PR #73, SUGGESTION #5): el marker estaba documentado en
    pytest.ini pero el flag no estaba registrado, por lo que
    `pytest --runnetwork` fallaba con "unrecognized arguments".
    """
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (with real waits >1s)",
    )
    parser.addoption(
        "--runnetwork",
        action="store_true",
        default=False,
        help="Run tests marked as network (HTTP against real services)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Si NO se pasa --runslow/--runnetwork, deselecciona esos tests."""
    skip_slow = pytest.mark.skip(reason="slow test, use --runslow or -m slow to run")
    skip_network = pytest.mark.skip(reason="network test, use --runnetwork or -m network to run")
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "network" in item.keywords and not config.getoption("--runnetwork"):
            item.add_marker(skip_network)


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Database real con schema inicializado (sqlite en tmp_path)."""
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


@pytest.fixture
def set_conv_updated_at(db: Database):
    """Fixture-factory para forzar updated_at en convs de test.

    Reemplaza `await asyncio.sleep(1.1)` que era el anti-pattern previo
    (SQLite CURRENT_TIMESTAMP tiene granularidad de 1s, asi que para
    que dos convs tengan updated_at distintos habia que esperar >1s).
    Cada test S12.1 que necesite modificar el cursor de un chat usa:

    ```python
    await set_conv_updated_at(conv_id, "2026-07-01 12:34:56")
    ```

    El formato es 'YYYY-MM-DD HH:MM:SS' UTC zero-padded (TDD trampa #1).

    Esto es ~1000x mas rapido que `asyncio.sleep(1.1)` y permite a los
    tests sync testear paginacion con timestamps arbitrarios sin esperar
    tiempo real.
    """
    from datetime import UTC, datetime

    async def _set(conv_id: int, dt: str | datetime) -> None:
        if isinstance(dt, datetime):
            dt_str = dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
        else:
            dt_str = dt
        async with db.conn.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (dt_str, conv_id),
        ) as cur:
            await cur.fetchall()
        await db.conn.commit()

    return _set


@pytest.fixture
def set_conv_field(db: Database):
    """Fixture-factory para forzar cualquier campo de timestamp de conversations.

    Usado por tests que necesitan manipular `deleted_at` o `encrypted_at`
    (no solo updated_at). Ejemplo:

    ```python
    await set_conv_field("deleted_at", conv_id, "2026-07-01 12:34:56")
    ```

    Por seguridad, valida que el field sea uno de los whitelisted. Esto
    evita SQL injection en tests.
    """
    from datetime import UTC, datetime

    ALLOWED_FIELDS = {"updated_at", "deleted_at", "encrypted_at", "purge_at"}

    async def _set(field: str, conv_id: int, dt: str | datetime) -> None:
        if field not in ALLOWED_FIELDS:
            raise ValueError(f"field must be one of {ALLOWED_FIELDS}, got {field!r}")
        if isinstance(dt, datetime):
            dt_str = dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
        else:
            dt_str = dt
        async with db.conn.execute(
            f"UPDATE conversations SET {field}=? WHERE id=?",
            (dt_str, conv_id),
        ) as cur:
            await cur.fetchall()
        await db.conn.commit()

    return _set


@pytest.fixture
def telemetry(settings: Settings) -> Telemetry:
    """Telemetry deshabilitada (sin InfluxDB) — no hace red."""
    return Telemetry(settings)


@pytest.fixture
def embeddings_mock(db: Database):
    """EmbeddingsService falso: embed() devuelve vector fijo sin red.

    Sprint 15 (US-3.1 §4 PR #69): todos los tests que ejerciten el flujo
    de embeddings (search_files, embed_vault, integration e2e) usan
    este fixture en lugar de un backend real. Razon:

    - Coste: OpenRouter free tiene rate limit (~60 RPM). Tests
      repetidos harian irrelevable el rate budget.
    - Determinismo: vector fijo -> cosine similarity reproducible,
      search_files tests assertable sin flakes.
    - Velocidad: 0ms vs 150ms por embed.

    Shape del vector: 4096 dims float32 (mismo shape que
    qwen/qwen3-embedding-8b en OpenRouter, el modelo que usaremos
    en prod). Los tests no asumen valores especificos del vector,
    solo que (a) tiene el shape correcto y (b) embed_and_store
    persiste.

    Args:
        db: Database fixture del conftest (inyectado para que
            embed_and_store / cosine_search puedan persistir en DB
            real sin tocar la API de embeddings).

    Returns:
        callable factory: `make(db)` o instancia directa? Una instancia
        lista para usar, con `_db_ref` ya apuntando al db del test.
    """
    import numpy as np

    from hermes.services.embeddings import EmbeddingsService

    class _FakeBackend:
        name = "fake"
        dim = 4096

        @property
        def is_enabled(self) -> bool:
            return True

        async def ensure_initialized(self) -> None:
            """Production search tools may initialize the service lazily."""

        async def embed(self, text: str) -> np.ndarray:
            # Vector fijo de 4096 dims. El valor concreto (0.5) no
            # importa — los tests solo verifican shape y persistencia.
            return np.full(self.dim, 0.5, dtype=np.float32)

        async def aclose(self) -> None:
            pass

    class _FakeService(EmbeddingsService):
        """EmbeddingsService con backend fake.

        Sobreescribe `is_enabled` y `embed_and_store` para no tocar
        la API real. Mantiene la misma interfaz que el servicio real
        para que el código de producción no cambie.
        """

        def __init__(self, db: Database) -> None:
            # Skip parent __init__ (que abre backend real); construimos
            # lo minimo para que los métodos overriden funcionen.
            self._backend: Any = _FakeBackend()
            self._db_ref = db
            self._cache: Any = None

        @property
        def is_enabled(self) -> bool:
            return True

        async def ensure_initialized(self) -> None:
            """Production search tools may initialize the service lazily."""

        async def embed(self, text: str) -> np.ndarray:
            return await self._backend.embed(text)

        async def embed_and_store(self, file_id: str, text: str) -> bool:
            """Genera embedding fake y lo persiste en la DB real.

            NO toca OpenRouter. Persiste el vector fijo en
            `file_embeddings` para que cosine_search tests puedan
            verificar el flujo completo.
            """
            vec = await self._backend.embed(text)
            await self._db_ref.add_file_embedding(file_id, vec.tobytes(), model="fake-test")
            return True

        async def cosine_search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
            """Cosine similarity sobre los embeddings fake en DB.

            Como todos los embeddings son vector constante (0.5),
            la cosine similarity es 1.0 para cualquier par. Devolvemos
            todos los file_ids con embedding, ordenados por file_id
            (orden arbitrario pero determinista para assertions).

            Chore 2026-07-05 (Nemotron 3 Ultra 550B review): usamos
            `list_embedded_file_ids(top_k)` en vez de
            `get_all_embeddings()` para no descargar los 60MB de blobs
            en libraries grandes. Como el mock es determinista
            (score=1.0 siempre), basta con los IDs.
            """
            fids = await self._db_ref.list_embedded_file_ids(limit=top_k)
            return [(fid, 1.0) for fid in fids]

    return _FakeService(db=db)


@pytest.fixture
def bot() -> Bot:
    """Bot real (aiogram valida el formato del token al construirlo)."""
    return Bot(token=TEST_BOT_TOKEN)


# ---------------------------------------------------------------------------
# Helpers para tests de handlers de Telegram
# ---------------------------------------------------------------------------


class _AnswerCapture:
    """Mock del método `Message.answer`. Registra cada llamada.

    `aiogram.Message` es pydantic frozen, lo que impide reasignar
    `answer` directamente. Usamos un MagicMock con `spec` limitado y le
    enchufamos este callable como atributo `answer` (MagicMock permite
    setear atributos incluso si no están en el spec, si la spec no es
    muy estricta).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, text: str, **kwargs: Any) -> None:
        # kwargs ignorados (parse_mode, reply_markup, etc.) — son
        # opcionales y el mock solo registra el texto.
        self.calls.append(text)

    def last(self) -> str:
        assert self.calls, "answer() nunca fue llamado"
        return self.calls[-1]

    def count(self) -> int:
        return len(self.calls)


def make_text_message(
    text: str,
    *,
    user_id: int = TEST_USER_ID,
    chat_id: int = TEST_CHAT_ID,
    thread_id: int | None = None,
) -> tuple[Any, _AnswerCapture]:
    """Mensaje de texto con `from_user`, `chat` reales y `answer` mockeado.

    Devuelve (msg, capture) donde `capture.calls` contiene todos los
    textos pasados a `msg.answer(...)` en orden.
    """
    user = User(id=user_id, is_bot=False, first_name="Test")
    chat = Chat(id=chat_id, type="private")
    capture = _AnswerCapture()
    msg = MagicMock(spec=["from_user", "chat", "text", "voice", "message_thread_id", "answer"])
    msg.from_user = user
    msg.chat = chat
    msg.text = text
    msg.voice = None
    msg.message_thread_id = thread_id
    msg.answer = capture  # async callable que registra
    return msg, capture


def get_handler(router) -> Any:
    """Extrae el callable registrado en `router.message.handlers[0].callback`.

    Esto evita tener que levantar un Dispatcher completo: invocamos la misma
    función que se ejecutaría en producción (con la closure ya capturada).
    Devuelve el primer handler de mensajes del router.

    Para routers con múltiples handlers (ej. command router con 4 comandos),
    usa `get_all_handlers` o `get_handler_at(idx)`.
    """
    observers = router.message.handlers
    assert len(observers) >= 1, "El router no tiene handler de mensaje"
    return observers[0].callback


def get_handler_at(router, index: int) -> Any:
    """Extrae el callable del handler en `router.message.handlers[index]`.

    Útil para routers con múltiples handlers (ej. `build_command_router`
    registra 4: /start, /help, /clear, /status). En ese caso el índice
    corresponde al orden de registro (0=/start, 1=/help, 2=/clear, 3=/status).
    """
    handlers = router.message.handlers
    assert (
        0 <= index < len(handlers)
    ), f"Índice {index} fuera de rango (router tiene {len(handlers)} handlers)"
    return handlers[index].callback


def get_all_handlers(router) -> list[Any]:
    """Devuelve todos los callables registrados como handlers de mensajes."""
    return [h.callback for h in router.message.handlers]
