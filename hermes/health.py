"""Healthcheck HTTP + S10.4 periodic health checks.

Dos componentes:
- `HealthServer`: servidor HTTP /health (aiohttp). Usado por el container
  healthcheck de Docker y por HealthChecker para self-ping.
- `HealthChecker` (S10.4): loop periodico que ejecuta health checks
  (HTTP API self-ping, disk space, DB integrity) y dispara push
  notifications via TelegramNotifier cuando falla alguno.

Las metricas se exportan a InfluxDB (ver telemetry.py).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from hermes.handlers.notifications import TelegramNotifier
    from hermes.memory.db import Database

logger = logging.getLogger(__name__)


class HealthServer:
    def __init__(self, host: str, port: int, db: Database) -> None:
        self.host = host
        self.port = port
        self.db = db
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info("health_server_started", extra={"host": self.host, "port": self.port})

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            logger.info("health_server_stopped")

    async def _health(self, _request: web.Request) -> web.Response:
        # Sprint 6 T53 v3.1: ping() ahora retorna bool en vez de lanzar.
        # Mantenemos el try/except por defensa en profundidad (por si una
        # implementacion futura de Database.ping cambia de contrato).
        try:
            db_ok = await self.db.ping()
        except Exception as exc:
            return web.json_response(
                {"status": "degraded", "db": "down", "error": str(exc)},
                status=503,
            )
        if not db_ok:
            return web.json_response(
                {"status": "degraded", "db": "down", "error": "ping returned False"},
                status=503,
            )
        return web.json_response({"status": "ok", "db": "up"})


# Sprint 10.4: HealthChecker (periodic checks con push notifications)
# ============================================================================
# Loop periodico que corre cada `check_interval_seconds` (default 60s) y
# ejecuta checks:
# 1. HTTP API self-ping: el container hace GET a su propio /health
# 2. Disk space: chequea que el data dir tenga > 1GB libre
# 3. DB integrity: ping() a la DB (ya existe, lo reusamos)
#
# Cada check que falla dispara un Telegram push via TelegramNotifier.
# TelegramNotifier tiene deduplicacion 1/hora por alert_type.
#
# Sprint 11 dependency: este modulo es REQUISITO para que Sprint 11
# pueda mitigar el WebUI SPOF (ver ADR-004 §5). Sin push notifications,
# el user no se entera de que Oroimen esta caido hasta mirar el PC.

# Thresholds
_DISK_LOW_THRESHOLD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
_DEFAULT_CHECK_INTERVAL_SECONDS = 60
_DEFAULT_HTTP_TIMEOUT_SECONDS = 5.0


class HealthChecker:
    """S10.4: periodic health checks con push notifications via Telegram.

    Uso:
        notifier = TelegramNotifier()  # lee de env TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
        checker = HealthChecker(settings, notifier, db)
        await checker.start()  # corre hasta stop()

    Si TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no estan en env, notifier
    es no-op (log warning al instanciar). HealthChecker sigue corriendo
    pero los alerts no se envian.
    """

    def __init__(
        self,
        settings: Any,  # hermes.config.Settings (Any para evitar import circular)
        notifier: TelegramNotifier,
        db: Database,
        *,
        check_interval_seconds: int = _DEFAULT_CHECK_INTERVAL_SECONDS,
        http_timeout_seconds: float = _DEFAULT_HTTP_TIMEOUT_SECONDS,
        data_dir: str = "/app/data",  # hermes_data volume mount en container
    ) -> None:
        self._settings = settings
        self._notifier = notifier
        self._db = db
        self._check_interval = check_interval_seconds
        self._http_timeout = http_timeout_seconds
        self._data_dir = data_dir
        self._stop_event = asyncio.Event()
        self._http_api_url = (
            f"http://localhost:{settings.hermes_api_port}/health"
            if hasattr(settings, "hermes_api_port")
            else "http://localhost:8000/health"
        )
        # Estado: alerta ya enviada para este estado (true = DOWN ya reportado).
        # Solo re-enviamos cuando hay RECOVERY (api_recovered). Asi evitamos
        # spam en outages largos y notificamos el alta.
        self._api_was_down: bool = False

    def stop(self) -> None:
        """Signal al loop para terminar (no bloqueante)."""
        self._stop_event.set()

    async def start(self) -> None:
        """Loop principal. Cancela con stop()."""
        logger.info(
            "health_checker_started",
            extra={
                "interval_seconds": self._check_interval,
                "http_api_url": self._http_api_url,
                "data_dir": self._data_dir,
                "telegram_enabled": self._notifier.enabled,
            },
        )
        while not self._stop_event.is_set():
            try:
                await self._check_once()
            except Exception:
                # Si un check lanza excepcion, log y continua. No queremos
                # que un check bug rompa el loop.
                logger.exception("health_check_iteration_error")
            # asyncio.wait_for permite cancelar el sleep cuando stop().
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._check_interval,
                )
                # Si wait() retorna (no timeout), stop fue llamado.
                break
            except TimeoutError:
                # Normal: timeout del sleep, siguiente iteracion.
                pass
        logger.info("health_checker_stopped")

    async def _check_once(self) -> None:
        """Ejecuta todos los checks. Cada uno es independiente."""
        await self._check_http_api()
        await self._check_disk_space()
        await self._check_db()

    async def _check_http_api(self) -> None:
        """Self-ping al HTTP API. Alerta si no responde 200 o noreachable."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                r = await client.get(self._http_api_url)
            if r.status_code == 200:
                # Si estaba DOWN antes, ahora esta UP: alerta de recovery.
                if self._api_was_down:
                    await self._notifier.send_health_alert(
                        alert_type="api_recovered",
                        message=f"HTTP API back online (status {r.status_code})",
                        severity="info",
                    )
                    self._api_was_down = False
                return
            # Status != 200: alertar
            await self._notifier.send_health_alert(
                alert_type="api_down",
                message=f"HTTP API returned {r.status_code}",
                severity="critical",
            )
            self._api_was_down = True
        except Exception as exc:
            await self._notifier.send_health_alert(
                alert_type="api_down",
                message=f"HTTP API unreachable: {type(exc).__name__}: {str(exc)[:200]}",
                severity="critical",
            )
            self._api_was_down = True

    async def _check_disk_space(self) -> None:
        """Alerta si /app/data (volume hermes_data) tiene < 1GB libre."""
        try:
            usage = shutil.disk_usage(self._data_dir)
            if usage.free < _DISK_LOW_THRESHOLD_BYTES:
                free_gb = usage.free / (1024**3)
                total_gb = usage.total / (1024**3)
                await self._notifier.send_health_alert(
                    alert_type="disk_low",
                    message=f"Only {free_gb:.1f}GB free of {total_gb:.1f}GB on {self._data_dir}",
                    severity="critical",
                )
        except FileNotFoundError:
            # data_dir no existe (raro en container, deberia estar mount).
            # No alertamos: es un config issue, no un health issue.
            logger.warning("health_check_disk_path_not_found", extra={"path": self._data_dir})
        except Exception:
            logger.exception("health_check_disk_error")

    async def _check_db(self) -> None:
        """Ping a la DB. Alerta si falla o retorna False."""
        try:
            db_ok = await self._db.ping()
            if not db_ok:
                await self._notifier.send_health_alert(
                    alert_type="db_error",
                    message="DB ping returned False",
                    severity="critical",
                )
        except Exception as exc:
            await self._notifier.send_health_alert(
                alert_type="db_error",
                message=f"DB ping exception: {type(exc).__name__}: {str(exc)[:200]}",
                severity="critical",
            )
