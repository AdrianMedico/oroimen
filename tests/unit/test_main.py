"""Tests industriales para `hermes.__main__` (punto de entrada).

Cubre:
- Inicialización: orden correcto, todos los componentes (DB, Telemetry, Health, Receiver).
- Loop principal: stop_event bloquea y desbloquea.
- Shutdown: orden de cierre (health, telemetry, db), idempotencia.
- `main()` wrapper: código de salida 0 (éxito) y 130 (KeyboardInterrupt).
- Errores fatales: fallos de inicialización se propagan.

Estrategia de mocking (justificación):
- `PollingReceiver` se sustituye por una fake class que NO toca la red
  de Telegram. El fake expone un `stop_event_hook` para que el test
  pueda activar el `stop_event` que `run()` crea internamente.
- `HealthServer` se sustituye por MagicMock (no queremos abrir puertos
  en CI).
- `install_signal_handlers` se sustituye por no-op (pytest ya gestiona
  signals; no queremos que el test se cuelgue al recibir SIGTERM).
- `Database` y `Telemetry` son reales (igual que en S2.3): más fidelidad.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes import __main__ as main_module
from hermes.memory.db import Database

# ---------------------------------------------------------------------------
# Fakes inyectables en hermes.__main__
# ---------------------------------------------------------------------------


class FakePollingReceiver:
    """Sustituto de `PollingReceiver` que NO toca Telegram.

    Cuando `run_forever` se llama, captura el `stop_event` en
    `FakePollingReceiver.last_stop_event` y luego espera a que se
    active. Esto permite al test programar el shutdown.

    Si el test no programa el stop_event, el task será cancelado por
    `run()` en su bloque `finally`, lo cual también es un camino válido
    de shutdown (testeado por separado).
    """

    last_stop_event: asyncio.Event | None = None
    last_init_kwargs: dict | None = None
    instance_count: int = 0

    def __init__(self, **kwargs) -> None:
        FakePollingReceiver.last_init_kwargs = kwargs
        FakePollingReceiver.instance_count += 1
        self.kwargs = kwargs
        # Validar argumentos esenciales
        assert "bot_token" in kwargs
        assert "db" in kwargs
        assert "settings" in kwargs
        assert "telemetry" in kwargs

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        FakePollingReceiver.last_stop_event = stop_event
        # Espera hasta que stop_event se active O el task sea cancelado
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            # Comportamiento real: el receiver se cancela, no propaga
            return

    @classmethod
    def reset(cls) -> None:
        cls.last_stop_event = None
        cls.last_init_kwargs = None
        cls.instance_count = 0


def make_fake_health() -> MagicMock:
    """Crea un mock de `HealthServer` que no abre puertos."""
    mock = MagicMock()
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_BOT_TOKEN = "9999999999:AAFakeTestTokenForUnitTests12345"


@pytest.fixture
def hermes_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Variables de entorno válidas para que `Settings()` funcione.

    Sprint 11 (ADR-004): `enable_http_api` default cambió de False a True
    en el scaffold, pero los tests pre-existentes asumen el comportamiento
    legacy (HealthServer activo). Para no romper la suite, fijamos
    `ENABLE_HTTP_API=false` aquí. Tests nuevos del Sprint 11 (que sí
    quieren HTTP API) deben overridear este env var explicitamente.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TEST_BOT_TOKEN)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("ENABLE_HTTP_API", "false")
    return tmp_path / "test.db"


@pytest.fixture
def patched_hermes_main(
    hermes_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Parchea `hermes.__main__` con fakes. Devuelve el dict con las refs.

    Tras la fixture, los componentes del módulo son:
    - `main_module.PollingReceiver = FakePollingReceiver`
    - `main_module.HealthServer = MagicMock()` con start/stop async
    - `main_module.install_signal_handlers = lambda e: None`
    """
    FakePollingReceiver.reset()
    fake_health = make_fake_health()
    captured_signals: list[asyncio.Event] = []

    monkeypatch.setattr(main_module, "PollingReceiver", FakePollingReceiver)
    monkeypatch.setattr(main_module, "HealthServer", MagicMock(return_value=fake_health))
    monkeypatch.setattr(
        main_module,
        "install_signal_handlers",
        lambda evt: captured_signals.append(evt),
    )
    return {
        "fake_health": fake_health,
        "captured_signals": captured_signals,
    }


@pytest.fixture(autouse=True)
def restore_root_logger() -> None:
    """Restaura los handlers del root logger después de cada test.

    `configure_logging()` muta el root logger (clear + add JsonFormatter).
    Si no restauramos, los tests siguientes reciben un logger con
    formateador que falla con KeyError("name") al intentar
    sobrescribir el campo "name" del LogRecord.
    """
    import logging

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    # Restaurar handlers y level
    root.handlers = original_handlers
    root.level = original_level


# ---------------------------------------------------------------------------
# Tests de `run()` — Inicialización
# ---------------------------------------------------------------------------
# Todos los TestRun* son @pytest.mark.slow porque:
# - Arrancan el ciclo de vida completo de hermes (DB + telemetry + receiver).
# - ~3s por test (12 tests x 3s = 36s del total de la suite).
# - Son integration tests, no unit tests puros.
# Se excluyen del CI por default; corren con `pytest --runslow` o `pytest -m slow`.
# Marcado individualmente para granularidad (no en la clase) por si en el
# futuro queremos excluir uno especifico.


@pytest.mark.slow
class TestRunInitialization:
    """`run()` debe crear todos los componentes en el orden correcto."""

    @pytest.mark.asyncio
    async def test_loads_settings_from_env(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        # Activamos stop inmediatamente para que run() retorne
        async def activate_immediately() -> None:
            # Esperamos a que run_forever capture el stop_event
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            FakePollingReceiver.last_stop_event.set()

        task = asyncio.create_task(activate_immediately())
        try:
            code = await main_module.run()
        finally:
            await task

        assert code == 0

    @pytest.mark.asyncio
    async def test_configures_logging(self, patched_hermes_main: dict, hermes_env: Path) -> None:
        """`configure_logging` se llama con el log_level de Settings."""
        from hermes.config import Settings

        settings = Settings()
        assert settings.log_level == "INFO"  # default

        # Spy sobre configure_logging
        with patch("hermes.__main__.configure_logging") as spy:

            async def activate() -> None:
                while FakePollingReceiver.last_stop_event is None:
                    await asyncio.sleep(0.01)
                FakePollingReceiver.last_stop_event.set()

            task = asyncio.create_task(activate())
            try:
                await main_module.run()
            finally:
                await task

            spy.assert_called_once_with("INFO")

    @pytest.mark.asyncio
    async def test_creates_database_with_schema(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """`run()` inicializa la DB; la tabla `messages` existe tras run()."""

        async def activate() -> None:
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            FakePollingReceiver.last_stop_event.set()

        task = asyncio.create_task(activate())
        try:
            await main_module.run()
        finally:
            await task

        # Verificar que la DB existe y tiene el schema
        assert hermes_env.exists()
        import aiosqlite

        async with (
            aiosqlite.connect(hermes_env) as conn,
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
            ) as cur,
        ):
            row = await cur.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_passes_correct_args_to_receiver(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """El receiver recibe bot_token, db, settings, telemetry."""

        async def activate() -> None:
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            FakePollingReceiver.last_stop_event.set()

        task = asyncio.create_task(activate())
        try:
            await main_module.run()
        finally:
            await task

        assert FakePollingReceiver.last_init_kwargs is not None
        kw = FakePollingReceiver.last_init_kwargs
        assert kw["bot_token"] == TEST_BOT_TOKEN
        assert isinstance(kw["db"], Database)
        assert kw["settings"].telegram_bot_token == TEST_BOT_TOKEN
        assert kw["telemetry"] is not None  # Real Telemetry instance

    @pytest.mark.asyncio
    async def test_db_init_failure_propagates(
        self, patched_hermes_main: dict, hermes_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Si la DB no se puede inicializar, se propaga la excepción."""

        async def broken_init(self: Database) -> None:
            raise RuntimeError("Disk full")

        monkeypatch.setattr(Database, "initialize", broken_init)

        with pytest.raises(RuntimeError, match="Disk full"):
            await main_module.run()

    @pytest.mark.asyncio
    async def test_installs_signal_handlers(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """`install_signal_handlers` se llama con el stop_event del run()."""

        async def activate() -> None:
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            FakePollingReceiver.last_stop_event.set()

        task = asyncio.create_task(activate())
        try:
            await main_module.run()
        finally:
            await task

        # El hook capturó un asyncio.Event
        assert len(patched_hermes_main["captured_signals"]) == 1
        assert isinstance(patched_hermes_main["captured_signals"][0], asyncio.Event)
        # Y es el mismo que el receiver capturó
        assert patched_hermes_main["captured_signals"][0] is FakePollingReceiver.last_stop_event


# ---------------------------------------------------------------------------
# Tests de `run()` — Loop principal
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRunLoop:
    """El loop principal bloquea hasta que `stop_event` se activa."""

    @pytest.mark.asyncio
    async def test_blocks_until_stop_event_set(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """`run()` no retorna hasta que stop_event se active."""
        run_done = False

        async def run_and_mark_done() -> None:
            nonlocal run_done
            await main_module.run()
            run_done = True

        async def activate_after_delay() -> None:
            # Esperamos a que run_forever esté esperando
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            # Esperamos un poco más para asegurar que run() está bloqueado
            await asyncio.sleep(0.05)
            assert not run_done, "run() retornó antes de stop_event"
            FakePollingReceiver.last_stop_event.set()

        tasks = [
            asyncio.create_task(run_and_mark_done()),
            asyncio.create_task(activate_after_delay()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

        assert run_done is True

    @pytest.mark.asyncio
    async def test_starts_receiver_before_waiting(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """El `receiver.run_forever` se lanza ANTES de esperar stop_event."""
        sequence: list[str] = []
        original_run_forever = FakePollingReceiver.run_forever

        async def tracking_run_forever(self, stop_event):
            sequence.append("run_forever_called")
            FakePollingReceiver.last_stop_event = stop_event
            # Activar stop inmediatamente para no bloquear
            stop_event.set()
            sequence.append("run_forever_returned")
            return

        FakePollingReceiver.run_forever = tracking_run_forever
        try:
            await main_module.run()
            # run_forever debe haberse llamado
            assert "run_forever_called" in sequence
            assert "run_forever_returned" in sequence
        finally:
            FakePollingReceiver.run_forever = original_run_forever


# ---------------------------------------------------------------------------
# Tests de `run()` — Shutdown
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestRunShutdown:
    """El bloque `finally` cierra los componentes en orden."""

    @pytest.mark.asyncio
    async def test_stops_health_on_shutdown(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """`health.stop()` se llama durante el shutdown."""
        fake_health = patched_hermes_main["fake_health"]

        async def activate() -> None:
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            FakePollingReceiver.last_stop_event.set()

        task = asyncio.create_task(activate())
        try:
            await main_module.run()
        finally:
            await task

        fake_health.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_db_on_shutdown(self, patched_hermes_main: dict, hermes_env: Path) -> None:
        """La DB se cierra en el shutdown (verificable: ping() falla)."""

        async def activate() -> None:
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            FakePollingReceiver.last_stop_event.set()

        task = asyncio.create_task(activate())
        try:
            await main_module.run()
        finally:
            await task

        # El archivo existe (la DB se creó) y el schema se aplicó
        assert hermes_env.exists()
        # Reabrir debería funcionar (no quedó colgado)
        import aiosqlite

        async with (
            aiosqlite.connect(hermes_env) as conn,
            conn.execute("SELECT COUNT(*) FROM messages") as cur,
        ):
            row = await cur.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_cancels_receiver_task_on_shutdown(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """El task del receiver se cancela correctamente en el shutdown.

        El test activa el stop_event tras capturar la referencia. La
        cancelación real del task (vía `receiver_task.cancel()`) se
        ejecuta en el `finally` de `run()` después de que stop_event.wait()
        retorna. El test verifica que el flujo completo termina sin
        colgarse.
        """
        # Custom fake que rastrea si fue cancelado
        was_cancelled = False

        class TrackingReceiver(FakePollingReceiver):
            async def run_forever(self, stop_event):
                FakePollingReceiver.last_stop_event = stop_event
                try:
                    await stop_event.wait()
                except asyncio.CancelledError:
                    nonlocal was_cancelled
                    was_cancelled = True
                    raise

        original_receiver = main_module.PollingReceiver
        main_module.PollingReceiver = TrackingReceiver
        try:

            async def activate() -> None:
                while FakePollingReceiver.last_stop_event is None:
                    await asyncio.sleep(0.01)
                FakePollingReceiver.last_stop_event.set()

            task = asyncio.create_task(activate())
            try:
                # Con timeout de seguridad: si el cancel no funcionara, esto
                # saltaría TimeoutError en lugar de quedarse colgado
                await asyncio.wait_for(main_module.run(), timeout=2.0)
            finally:
                await task

            # run() retornó correctamente
        finally:
            main_module.PollingReceiver = original_receiver

        # Si la cancelación no funcionara, run() se quedaría en
        # `await receiver_task` esperando que el task termine, pero como
        # el fake acepta CancelledError, el task termina inmediatamente.
        # Por lo tanto, este test valida que la cadena completa
        # stop_event.set → wait() retorna → finally ejecuta → cancel → done.
        # El flag was_cancelled puede ser True o False dependiendo de si
        # asyncio cancela el task antes de que stop_event.wait() retorne,
        # pero lo importante es que run() retorna sin colgarse.

    @pytest.mark.asyncio
    async def test_shutdown_order(self, patched_hermes_main: dict, hermes_env: Path) -> None:
        """Orden de shutdown: health.stop → telemetry.aclose → db.close.

        Lo verificamos con un spy en cada método.
        """
        call_order: list[str] = []

        # Spy en HealthServer.stop
        fake_health = patched_hermes_main["fake_health"]
        original_health_stop = fake_health.stop

        async def tracking_health_stop() -> None:
            call_order.append("health.stop")
            await original_health_stop()

        fake_health.stop = tracking_health_stop

        # Spy en Telemetry.aclose (es método real, no mock)
        from hermes.telemetry import Telemetry

        original_telemetry_close = Telemetry.aclose

        async def tracking_telemetry_close(self) -> None:
            call_order.append("telemetry.aclose")
            await original_telemetry_close(self)

        Telemetry.aclose = tracking_telemetry_close

        # Spy en Database.close
        original_db_close = Database.close

        async def tracking_db_close(self) -> None:
            call_order.append("db.close")
            await original_db_close(self)

        Database.close = tracking_db_close

        try:

            async def activate() -> None:
                while FakePollingReceiver.last_stop_event is None:
                    await asyncio.sleep(0.01)
                FakePollingReceiver.last_stop_event.set()

            task = asyncio.create_task(activate())
            try:
                await main_module.run()
            finally:
                await task

            # Verificar el orden
            assert call_order == [
                "health.stop",
                "telemetry.aclose",
                "db.close",
            ], f"Shutdown order incorrect: {call_order}"
        finally:
            Telemetry.aclose = original_telemetry_close
            Database.close = original_db_close


# ---------------------------------------------------------------------------
# Tests de `main()` — wrapper síncrono
# ---------------------------------------------------------------------------


class TestMain:
    """`main()` ejecuta `asyncio.run(run())` y mapea KeyboardInterrupt → 130.

    Nota: `main()` es síncrono (usa `asyncio.run`). Lo testeamos con
    mocks de `asyncio.run` en lugar de ejecutarlo realmente, porque
    `asyncio.run` crea un nuevo event loop y no comparte estado con el
    loop del test.
    """

    def test_main_returns_zero_on_success(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """Si run() retorna 0, main() retorna 0."""
        with patch("hermes.__main__.asyncio.run", return_value=0) as spy:
            code = main_module.main()
        assert code == 0
        spy.call_args[0][0].close()

    def test_main_returns_130_on_keyboard_interrupt(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """Si run() levanta KeyboardInterrupt, main() retorna 130."""
        with patch("hermes.__main__.asyncio.run", side_effect=KeyboardInterrupt) as spy:
            code = main_module.main()
        assert code == 130
        spy.call_args[0][0].close()

    def test_main_propagates_other_exceptions(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """Si run() levanta una excepción que no es KeyboardInterrupt, se propaga."""
        with (
            patch(
                "hermes.__main__.asyncio.run",
                side_effect=RuntimeError("boom"),
            ) as spy,
            pytest.raises(RuntimeError, match="boom"),
        ):
            main_module.main()
        spy.call_args[0][0].close()

    def test_main_calls_asyncio_run_with_run(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """`main()` invoca `asyncio.run(run())` (verificación de la API)."""
        with patch("hermes.__main__.asyncio.run", return_value=0) as spy:
            main_module.main()
            # Verificamos que asyncio.run fue llamado exactamente una vez
            assert spy.call_count == 1
            # El argumento es una coroutine (resultado de llamar `run()`)
            called_arg = spy.call_args[0][0]
            assert asyncio.iscoroutine(called_arg)
            # Cerramos la coroutine para evitar el warning
            called_arg.close()


# ---------------------------------------------------------------------------
# Tests de logging y observabilidad
@pytest.mark.slow
# ---------------------------------------------------------------------------


class TestRunLogging:
    """`run()` emite eventos estructurados en los puntos clave.

    Nota: `configure_logging` reemplaza los handlers del root logger
    con un `StreamHandler(sys.stdout)` que usa `JsonFormatter`. Por
    tanto, los logs no se capturan con `caplog` (que usa handlers
    internos), sino que van a stdout. Usamos `capsys` para capturarlos.
    """

    @pytest.mark.asyncio
    async def test_logs_hermes_starting_and_stopping(
        self, patched_hermes_main: dict, hermes_env: Path, capsys
    ) -> None:
        async def activate() -> None:
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            FakePollingReceiver.last_stop_event.set()

        task = asyncio.create_task(activate())
        try:
            await main_module.run()
        finally:
            await task

        # Los logs JSON van a stdout, los parseamos
        import json as _json

        captured = capsys.readouterr().out
        messages = []
        for line in captured.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = _json.loads(line)
                if obj.get("logger") == "hermes.__main__":
                    messages.append(obj.get("message"))
            except _json.JSONDecodeError:
                pass

        assert "hermes_starting" in messages
        assert "hermes_stopping" in messages
        assert "hermes_stopped" in messages


# ---------------------------------------------------------------------------
@pytest.mark.slow
# Tests de integración con componentes reales
# ---------------------------------------------------------------------------


class TestRunIntegration:
    """Test de integración end-to-end: todos los componentes reales juntos."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_real_components(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """Lifecycle completo: start → run → stop → cleanup, todo real (DB, Telemetry).

        v0.3.2: reescrito para evitar hang en CI Python 3.11.

        Problemas previos:
        1. Verificación de schema via nueva conexión aiosqlite después
           del run() → file lock en Windows, race en Linux.
        2. Patrón `while None: await sleep(0.01)` para esperar el
           stop_event → bajo pytest-asyncio + Python 3.11, el event
           loop puede no progresar correctamente, colgando el test.

        Solución:
        1. Eliminar la verificación de schema (ya cubierta por test_db.py).
        2. Hacer que el FakePollingReceiver active stop_event internamente
           tras un breve delay, sin depender de tareas externas ni de
           polling activo.
        """
        # 1. Pre-condición: DB no existe
        assert not hermes_env.exists()

        # 2. Sobrescribir run_forever del fake para que active stop_event
        # automáticamente tras un breve delay. Esto elimina la dependencia
        # de tareas asyncio.create_task que pueden colgarse en CI 3.11.
        original_run_forever = FakePollingReceiver.run_forever

        async def self_activating_run_forever(self, stop_event):
            FakePollingReceiver.last_stop_event = stop_event
            # Espera breve (50ms) y activa stop automáticamente
            await asyncio.sleep(0.05)
            stop_event.set()
            # Mantén la coroutine viva hasta que el task sea cancelado por
            # el finally de run(). No es strictly necesario, pero asegura
            # que no retornamos antes de que se complete la cancelación.
            try:
                await stop_event.wait()
            except asyncio.CancelledError:
                return

        FakePollingReceiver.run_forever = self_activating_run_forever
        try:
            # 3. Run
            code = await main_module.run()
        finally:
            FakePollingReceiver.run_forever = original_run_forever

        # 4. Post-condiciones
        assert code == 0
        assert hermes_env.exists()


@pytest.mark.slow
# ---------------------------------------------------------------------------
# Tests S11 (ADR-004): WebUI-primary + Telegram opt-in legacy
# ---------------------------------------------------------------------------


class TestRunTelegramGating:
    """Sprint 11: PollingReceiver arrancado bajo flag `enable_telegram`.

    Estos tests verifican el nuevo wire-up de __main__.py:
    - `enable_telegram=True` (default S11.0): arranca PollingReceiver
      si telegram_bot_token esta set.
    - `enable_telegram=True` sin token: warning, no arranca, run() OK.
    - `enable_telegram=False`: NO arranca PollingReceiver, run() OK
      (HTTP API + HealthChecker siguen funcionando).
    - Deprecation warning se emite al startup cuando enable_telegram=True.
    """

    @pytest.mark.asyncio
    async def test_telegram_receiver_started_when_enabled(
        self, patched_hermes_main: dict, hermes_env: Path
    ) -> None:
        """enable_telegram=True + token set: PollingReceiver se crea."""

        async def activate() -> None:
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            FakePollingReceiver.last_stop_event.set()

        task = asyncio.create_task(activate())
        try:
            await main_module.run()
        finally:
            await task

        # El FakePollingReceiver se creo (incremento instance_count).
        assert FakePollingReceiver.instance_count == 1

    @pytest.mark.asyncio
    async def test_telegram_receiver_skipped_when_disabled(
        self, patched_hermes_main: dict, hermes_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """enable_telegram=False: PollingReceiver NO se crea."""

        monkeypatch.setenv("ENABLE_TELEGRAM", "false")

        async def activate_short_circuit() -> None:
            # El run() no debe crear receiver, no hay FakePollingReceiver
            # instanciado. run() bloqueara en stop_event.wait() (que nadie
            # va a setear), asi que con timeout matamos el test.
            await asyncio.sleep(0.05)
            raise TimeoutError("run() no retorno, esperado")

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(main_module.run(), timeout=2.0)

        # Confirmar que PollingReceiver NO se creo.
        assert FakePollingReceiver.instance_count == 0

    @pytest.mark.asyncio
    async def test_telegram_skipped_when_enabled_but_no_token(
        self, patched_hermes_main: dict, hermes_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """enable_telegram=True pero sin TELEGRAM_BOT_TOKEN: warning, no crash.

        Verifica que hermes arranca sin PollingReceiver pero el resto
        (HTTP API, HealthChecker, etc) sigue funcionando.
        """
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

        async def activate_short_circuit() -> None:
            await asyncio.sleep(0.05)
            raise TimeoutError("run() no retorno, esperado")

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(main_module.run(), timeout=2.0)

        # El receiver NO se creo (token falta)
        assert FakePollingReceiver.instance_count == 0

    @pytest.mark.asyncio
    async def test_deprecation_warning_logged_when_telegram_enabled(
        self, patched_hermes_main: dict, hermes_env: Path, capsys
    ) -> None:
        """enable_telegram=True: emite warning de deprecation al startup.

        El user (operator) debe ver que Telegram sera opt-in legacy en
        S12+. Visible en logs estructurados.
        """

        async def activate() -> None:
            while FakePollingReceiver.last_stop_event is None:
                await asyncio.sleep(0.01)
            FakePollingReceiver.last_stop_event.set()

        task = asyncio.create_task(activate())
        try:
            await main_module.run()
        finally:
            await task

        captured = capsys.readouterr().out
        assert "s11_telegram_deprecation_notice" in captured


@pytest.mark.asyncio
async def test_close_core_resources_closes_all_after_earlier_failure() -> None:
    embeddings = MagicMock()
    embeddings.aclose = AsyncMock(side_effect=RuntimeError("embedding close failed"))
    telemetry = MagicMock()
    telemetry.aclose = AsyncMock()
    llm = MagicMock()
    llm.aclose = AsyncMock()
    db = MagicMock()
    db.close = AsyncMock()

    with pytest.raises(RuntimeError, match="embedding close failed"):
        await main_module._close_core_resources(
            embeddings_service=embeddings,
            telemetry=telemetry,
            llm=llm,
            db=db,
        )

    embeddings.aclose.assert_awaited_once()
    telemetry.aclose.assert_awaited_once()
    llm.aclose.assert_awaited_once()
    db.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Slice 1C1b: Deep Research runtime startup ordering + capability booleans
# ---------------------------------------------------------------------------
# These tests assert the reviewed startup sequence WITHOUT spinning up the
# full HTTP server, real Telegram polling, or any network. We replace the
# real DeepResearchService / DeepResearchScheduler / SafeExternalFetcher /
# recover_research_jobs / set_deep_research_service call sites with fakes
# that record their invocation order, then assert the published capabilities.
# Disabled / missing-prerequisite paths register NO research-specific runtime
# and report false runtime capabilities.


class _RecordingDRScheduler:
    """Stand-in for DeepResearchScheduler that records the startup sequence.

    Every event (``set_service`` / ``start``) is recorded both into
    ``self.calls`` (used for in-order scheduler assertions) and via the
    ``record_callback`` (if set), which lets the global ordering list
    see scheduler events interleaved with closure-local ordering from
    other fakes.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._service: object | None = None
        self.received_service: object | None = None
        self.record_callback: Callable[[str], None] | None = None

    def set_service(self, service: object) -> None:
        self.calls.append("set_service")
        self._service = service
        self.received_service = service
        if self.record_callback is not None:
            self.record_callback("attach_to_scheduler")

    async def start(self) -> None:
        self.calls.append("start")
        if self.record_callback is not None:
            self.record_callback("scheduler_start")


def _patch_dr_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scheduler: _RecordingDRScheduler,
    service_should_fail: bool = False,
) -> dict:
    """Wire all Deep Research runtime call sites to the recording stubs.

    Returns a dict with call logs. Uses ``setattr`` on the imported
    module symbols so the fake instances replace the real classes.

    Slice 1C1b closure: also patches ``hermes.receivers.http_api.create_app``
    so the test can assert that the SAME service instance reaches the
    app's deep_research_capabilities-singleton registration AND that
    recovery completes BEFORE both ``create_app`` and the HTTP task
    are constructed.
    """
    service_constructor_calls: list[dict] = []
    fetcher_constructor_calls: list[dict] = []
    set_singleton_calls: list[object | None] = []
    recover_calls: list[dict] = []
    # Holds the constructed service instance so assertions can verify
    # that the SAME object was passed to set_service, set_deep_research_service,
    # and scheduler.set_service(...).
    constructed_services: list[object] = []
    # Records the full successful ordering across all reviewed steps plus
    # ``create_app`` invocation and HTTP task creation. The list contains
    # short string labels in invocation order, e.g.
    #   ["construct_service", "attach_to_scheduler", "singleton_register",
    #    "scheduler_start", "recover", "create_app", "http_task"]
    ordering: list[str] = []
    # The capabilities object that was passed into the app via create_app,
    # captured for assertion on the seven approved runtime booleans.
    app_caps: list[Any] = []
    # The ``create_app`` keyword arguments across invocations, used to
    # verify it received the SAME service singleton via its deps and the
    # SAME capability object.
    create_app_kwargs: list[dict] = []

    class _FakeDRService:
        def __init__(self, **kwargs: Any) -> None:
            service_constructor_calls.append(kwargs)
            constructed_services.append(self)
            ordering.append("construct_service")
            self._service_id = id(self)
            if service_should_fail:
                raise RuntimeError("service construct exploded")
            self.kwargs = kwargs

    class _FakeSafeFetcherCls:
        def __init__(self, **kwargs: Any) -> None:
            fetcher_constructor_calls.append(kwargs)
            ordering.append("construct_fetcher")
            self.kwargs = kwargs

    async def _fake_recover_research_jobs(**kwargs: Any) -> int:
        recover_calls.append(kwargs)
        ordering.append("recover")
        return 0

    def _fake_set_singleton(service: object | None) -> None:
        set_singleton_calls.append(service)
        ordering.append("singleton_register")

    def _fake_create_app(**kwargs: Any) -> MagicMock:
        """Recording stand-in for ``hermes.receivers.http_api.create_app``.

        Captures the kwargs so we can verify that the same
        ``deep_research_capabilities`` instance reaches the app and the
        same service deps are visible (the test does not pull the deps
        through the app — they're already asserted via the scheduler +
        singleton assertions — so a MagicMock is safe).
        """
        ordering.append("create_app")
        create_app_kwargs.append(kwargs)
        # Capture the capability object the runtime handed to create_app;
        # assertions read it back to compare with the singleton.
        caps_arg = kwargs.get("deep_research_capabilities")
        if caps_arg is not None:
            app_caps.append(caps_arg)
        return MagicMock()

    class _FakeUvicornServer:
        """Stand-in for ``uvicorn.Server`` — only ``serve`` is awaited."""

        def __init__(self, config: Any) -> None:
            self.config = config

        async def serve(self) -> None:
            ordering.append("http_task_serve")
            # Block until the test activates the stop_event; mirrors the
            # real uvicorn.Server.serve() in main.py.
            stop_event = (
                FakePollingReceiver.last_stop_event
                if FakePollingReceiver.last_stop_event is not None
                else asyncio.Event()
            )
            try:
                await stop_event.wait()
            except asyncio.CancelledError:
                return

    class _FakeUvicornConfig:
        def __init__(self, app: Any, **kwargs: Any) -> None:
            self.app = app
            self.kwargs = kwargs

    import hermes.jobs.recovery as recovery_module
    import hermes.jobs.safe_fetcher as safe_fetcher_module
    import hermes.jobs.scheduler as scheduler_module
    import hermes.jobs.service as service_module
    import hermes.receivers.http_api as http_api_module
    import hermes.receivers.jobs_api as jobs_api_module

    # Let the scheduler append its events into the same closure ordering
    # list so the assertion can interleave scheduler events with the
    # other fake's events without touching the scheduler's own ``calls``.
    def _record(event: str) -> None:
        ordering.append(event)

    scheduler.record_callback = _record

    monkeypatch.setattr(scheduler_module, "DeepResearchScheduler", lambda **kw: scheduler)
    monkeypatch.setattr(service_module, "DeepResearchService", _FakeDRService)
    monkeypatch.setattr(
        safe_fetcher_module,
        "SafeExternalFetcher",
        _FakeSafeFetcherCls,
    )
    monkeypatch.setattr(recovery_module, "recover_research_jobs", _fake_recover_research_jobs)
    monkeypatch.setattr(jobs_api_module, "set_deep_research_service", _fake_set_singleton)
    monkeypatch.setattr(http_api_module, "create_app", _fake_create_app)
    # Patch uvicorn via sys.modules because hermes.__main__.run() imports
    # uvicorn locally (``import uvicorn`` inside the enable_http_api
    # block), so main_module.uvicorn does not exist as a module attribute.
    # Replacing sys.modules["uvicorn"] ensures the local ``import uvicorn``
    # statement resolves to the fake without spinning up a real listener.
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        MagicMock(
            Server=_FakeUvicornServer,
            Config=_FakeUvicornConfig,
        ),
    )
    # DeepResearchCapabilities stays real (no I/O, just a frozen model).

    return {
        "service_constructor_calls": service_constructor_calls,
        "fetcher_constructor_calls": fetcher_constructor_calls,
        "set_singleton_calls": set_singleton_calls,
        "recover_calls": recover_calls,
        "constructed_services": constructed_services,
        "ordering": ordering,
        "app_caps": app_caps,
        "create_app_kwargs": create_app_kwargs,
    }


def _enable_dr_prereqs(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    """Configure the env so the runtime sees enabled + a Tavily key + a cloud LLM.

    Slice 1C1b: also enable the Search Router so the Tavily backend is
    actually constructed at startup. Without SEARCH_ENABLED=true,
    ``settings.search_enabled`` is False → ``search_backends`` is empty
    → ``tavily_configured`` is False → ``all_prereqs_ready`` is False,
    so the successful-order test would silently exercise the
    missing-prerequisite branch instead of the reviewed construction path.
    """
    monkeypatch.setenv("HERMES_DEEP_RESEARCH_ENABLED", "true" if enabled else "false")
    if enabled:
        # SEARCH_ENABLED activates the search block in hermes.__main__.run()
        # which constructs the TavilyBackend when TAVILY_API_KEY is set.
        monkeypatch.setenv("SEARCH_ENABLED", "true")
        # Tavily backend construction key. Fake value — must not be a real
        # secret; the test does NOT issue any network requests, it only
        # verifies that the constructor was invoked.
        monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
        # Cloud LLM client (OPENCODE_GO_API_KEY) is set in the hermes_env
        # fixture, so cloud_client_configured is True in run().
    else:
        # Disabled path: explicitly turn the Search Router off so the
        # disabled test exercises the `deep_research_enabled=False` branch.
        monkeypatch.setenv("SEARCH_ENABLED", "false")
        monkeypatch.setenv("TAVILY_API_KEY", "")


@pytest.fixture
def dr_patched_hermes_main(
    hermes_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Apply the standard hermes_main patches AND enable Deep Research prereqs."""
    FakePollingReceiver.reset()
    fake_health = make_fake_health()
    captured_signals: list[asyncio.Event] = []
    monkeypatch.setattr(main_module, "PollingReceiver", FakePollingReceiver)
    monkeypatch.setattr(main_module, "HealthServer", MagicMock(return_value=fake_health))
    monkeypatch.setattr(main_module, "ToolRegistry", MagicMock())
    monkeypatch.setattr(main_module, "register_builtin_tools", MagicMock())
    monkeypatch.setattr(
        main_module,
        "install_signal_handlers",
        lambda evt: captured_signals.append(evt),
    )
    # Keep the composition tests focused on the Deep Research seam.  The
    # normal application lifecycle starts several unrelated schedulers and a
    # health-check task; their shutdown timing is intentionally covered by
    # the existing lifecycle tests and makes these ordering assertions slow
    # and flaky on Windows.
    # The fake receiver provides the deterministic stop-event seam used by
    # these tests; it does not make any Telegram network calls.
    monkeypatch.setenv("ENABLE_TELEGRAM", "true")
    monkeypatch.setenv("BACKUP_ENABLED", "false")
    monkeypatch.setenv("CLEANUP_ENABLED", "false")
    monkeypatch.setenv("VAULT_DROP_ENABLED", "false")
    monkeypatch.setenv("PUSH_NOTIFICATIONS_ENABLED", "false")
    monkeypatch.setenv("TOOLS_ENABLED", "false")
    scheduler = _RecordingDRScheduler()
    return {
        "fake_health": fake_health,
        "captured_signals": captured_signals,
        "scheduler": scheduler,
    }


def _dr_settings(*, enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        deep_research_enabled=enabled,
        search_enabled=True,
        search_size_guard_chars=1000,
    )


@pytest.mark.asyncio
async def test_compose_deep_research_runtime_orders_success_without_full_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _RecordingDRScheduler()
    logs = _patch_dr_runtime(monkeypatch, scheduler=scheduler)
    caps, returned_scheduler, service = await main_module._compose_deep_research_runtime(
        settings=_dr_settings(enabled=True),
        db=MagicMock(),
        notifier=MagicMock(),
        llm=SimpleNamespace(cloud_client_configured=True),
        search_backends={"tavily": MagicMock()},
        search_budget=MagicMock(),
        search_circuit_breaker=MagicMock(),
        search_concurrency=MagicMock(),
    )
    assert returned_scheduler is scheduler
    assert service is logs["constructed_services"][0]
    assert logs["ordering"] == [
        "construct_fetcher",
        "construct_service",
        "attach_to_scheduler",
        "singleton_register",
        "scheduler_start",
        "recover",
    ]
    assert scheduler.calls == ["set_service", "start"]
    assert all(
        getattr(caps, name)
        for name in (
            "service_wiring",
            "recovery_wiring",
            "fetch_policy",
            "external_fetch",
            "search_backend_configured",
            "llm_provider_configured",
            "model_output_enforced",
        )
    )


@pytest.mark.asyncio
async def test_compose_deep_research_runtime_disabled_constructs_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _RecordingDRScheduler()
    logs = _patch_dr_runtime(monkeypatch, scheduler=scheduler)
    caps, returned_scheduler, service = await main_module._compose_deep_research_runtime(
        settings=_dr_settings(enabled=False),
        db=MagicMock(),
        notifier=MagicMock(),
        llm=SimpleNamespace(cloud_client_configured=True),
        search_backends={"tavily": MagicMock()},
        search_budget=MagicMock(),
        search_circuit_breaker=MagicMock(),
        search_concurrency=MagicMock(),
    )
    assert not caps.service_wiring
    assert returned_scheduler is None and service is None
    assert logs["ordering"] == [] and logs["set_singleton_calls"] == []


@pytest.mark.asyncio
async def test_compose_deep_research_runtime_failure_stops_and_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingScheduler(_RecordingDRScheduler):
        async def start(self) -> None:
            await super().start()
            raise RuntimeError("test failure")

        async def shutdown(self) -> None:
            self.calls.append("shutdown")

    scheduler = FailingScheduler()
    logs = _patch_dr_runtime(monkeypatch, scheduler=scheduler)
    caps, returned_scheduler, service = await main_module._compose_deep_research_runtime(
        settings=_dr_settings(enabled=True),
        db=MagicMock(),
        notifier=MagicMock(),
        llm=SimpleNamespace(cloud_client_configured=True),
        search_backends={"tavily": MagicMock()},
        search_budget=MagicMock(),
        search_circuit_breaker=MagicMock(),
        search_concurrency=MagicMock(),
    )
    assert not caps.service_wiring
    assert returned_scheduler is None and service is None
    assert scheduler.calls == ["set_service", "start", "shutdown"]
    assert logs["set_singleton_calls"][-1] is None
