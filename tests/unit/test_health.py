"""Tests industriales para `hermes.health.HealthServer`.

Cubre:
- Endpoint `/health`:
  - 200 OK con DB up
  - 503 Service Unavailable con DB down
  - Content-Type application/json
  - Estructura del body
- Múltiples health checks concurrentes
- Lifecycle: start / stop / restart

Estrategia:
- `aiohttp.test_utils.TestServer` no se usa porque añade complejidad.
  En su lugar, levantamos el server en un puerto aleatorio y usamos
  `aiohttp.ClientSession` para hacer requests reales. Más fiel al
  comportamiento de producción.
- `Database` real (sqlite en tmp_path).
- Cada test usa un puerto diferente para evitar conflictos si los
  tests corren en paralelo.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import aiohttp
import pytest

from hermes.health import HealthServer
from hermes.memory.db import Database

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Encuentra un puerto TCP libre. Útil para tests en paralelo."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Database real en tmp_path.

    Se inicializa en un event loop separado del test (porque pytest-asyncio
    con scope function crea uno nuevo por test).
    """
    d = Database(tmp_path / "test.db")

    async def _setup() -> None:
        await d.initialize()

    await _setup()
    yield d
    await d.close()


@pytest.fixture
def health_port() -> int:
    """Puerto libre aleatorio para evitar conflictos en tests paralelos."""
    return _find_free_port()


@pytest.fixture
async def health_server(db: Database, health_port: int):
    """HealthServer real arrancado en localhost:health_port."""
    server = HealthServer(host="127.0.0.1", port=health_port, db=db)
    await server.start()
    yield server
    await server.stop()


# ---------------------------------------------------------------------------
# Tests del endpoint /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests de la ruta principal `/health`."""

    @pytest.mark.asyncio
    async def test_returns_200_when_db_up(self, health_server: HealthServer) -> None:
        """Con DB funcionando, devuelve 200."""
        url = f"http://127.0.0.1:{health_server.port}/health"
        async with (
            aiohttp.ClientSession() as session,
            session.get(url) as resp,
        ):
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["db"] == "up"

    @pytest.mark.asyncio
    async def test_returns_503_when_db_down(self, db: Database, health_port: int) -> None:
        """Con DB caída, devuelve 503 y status 'degraded'."""
        server = HealthServer(host="127.0.0.1", port=health_port, db=db)
        await server.start()
        try:
            # Forzamos DB no disponible cerrando la conexión
            await db.close()

            url = f"http://127.0.0.1:{server.port}/health"
            async with (
                aiohttp.ClientSession() as session,
                session.get(url) as resp,
            ):
                assert resp.status == 503
                data = await resp.json()
                assert data["status"] == "degraded"
                assert data["db"] == "down"
                # El error concreto se incluye
                assert "error" in data
                assert isinstance(data["error"], str)
        finally:
            # server.stop no fallará porque el cleanup maneja db=None
            await server.stop()

    @pytest.mark.asyncio
    async def test_response_content_type_is_json(self, health_server: HealthServer) -> None:
        """El response lleva Content-Type application/json."""
        url = f"http://127.0.0.1:{health_server.port}/health"
        async with (
            aiohttp.ClientSession() as session,
            session.get(url) as resp,
        ):
            # aiohttp normaliza el content-type
            assert "json" in resp.headers.get("Content-Type", "").lower()

    @pytest.mark.asyncio
    async def test_health_body_shape(self, health_server: HealthServer) -> None:
        """El body del response tiene exactamente los campos esperados."""
        url = f"http://127.0.0.1:{health_server.port}/health"
        async with (
            aiohttp.ClientSession() as session,
            session.get(url) as resp,
        ):
            data = await resp.json()
        assert set(data.keys()) == {"status", "db"}

    @pytest.mark.asyncio
    async def test_health_degraded_body_shape(self, db: Database, health_port: int) -> None:
        """El body degraded tiene status, db, error."""
        server = HealthServer(host="127.0.0.1", port=health_port, db=db)
        await server.start()
        try:
            await db.close()
            url = f"http://127.0.0.1:{server.port}/health"
            async with (
                aiohttp.ClientSession() as session,
                session.get(url) as resp,
            ):
                data = await resp.json()
            assert set(data.keys()) == {"status", "db", "error"}
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, health_server: HealthServer) -> None:
        """Múltiples requests concurrentes al endpoint funcionan."""
        url = f"http://127.0.0.1:{health_server.port}/health"

        async def make_request() -> tuple[int, dict]:
            async with (
                aiohttp.ClientSession() as session,
                session.get(url) as resp,
            ):
                return resp.status, await resp.json()

        # 10 requests concurrentes
        results = await asyncio.gather(*[make_request() for _ in range(10)])
        for status, data in results:
            assert status == 200
            assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Tests del lifecycle del server
# ---------------------------------------------------------------------------


class TestHealthServerLifecycle:
    """Tests de start/stop/restart del HealthServer."""

    @pytest.mark.asyncio
    async def test_server_can_be_stopped_and_restarted(
        self, db: Database, health_port: int
    ) -> None:
        """El server puede pararse y rearrancar sin leaks."""
        server = HealthServer(host="127.0.0.1", port=health_port, db=db)

        # 1ª vez: start, check, stop
        await server.start()
        url = f"http://127.0.0.1:{server.port}/health"
        async with (
            aiohttp.ClientSession() as session,
            session.get(url) as resp,
        ):
            assert resp.status == 200
        await server.stop()

        # 2ª vez: start de nuevo, check
        await server.start()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(url) as resp,
            ):
                assert resp.status == 200
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, db: Database, health_port: int) -> None:
        """Llamar stop() varias veces no falla."""
        server = HealthServer(host="127.0.0.1", port=health_port, db=db)
        await server.start()
        await server.stop()
        # Segunda llamada: no debería fallar
        await server.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self, db: Database, health_port: int) -> None:
        """Llamar stop() sin haber llamado start() no falla (estado limpio)."""
        server = HealthServer(host="127.0.0.1", port=health_port, db=db)
        # No llamamos start
        await server.stop()
        # El server queda en estado "nunca arrancado", stop() es no-op
        assert server._runner is None


# ---------------------------------------------------------------------------
# Tests de paths no encontrados
# ---------------------------------------------------------------------------


class TestHealthServerRouting:
    """Tests de routing del HealthServer."""

    @pytest.mark.asyncio
    async def test_404_on_unknown_path(self, health_server: HealthServer) -> None:
        """Paths distintos de /health devuelven 404."""
        url = f"http://127.0.0.1:{health_server.port}/notfound"
        async with (
            aiohttp.ClientSession() as session,
            session.get(url) as resp,
        ):
            assert resp.status == 404
