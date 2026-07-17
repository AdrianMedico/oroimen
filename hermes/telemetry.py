"""Telemetría: push de métricas a InfluxDB.

Las métricas se modelan como `point` de InfluxDB con measurement `hermes_*`
y tags para filtrado (model, tool, result). El cliente InfluxDB hace
batching automático cada 1s; al cerrar se fuerza flush.

Sprint 17 (F2-1): consecutive-failure counter + escalation. Si el write
falla N veces seguidas dentro de T segundos, se escala a ERROR y se
registra un `telemetry_escalated` event. Esto evita el patron actual de
silent fail de 5+ dias (5 dias sin saber que INFLUX_TOKEN estaba
revocado). Threshold por defecto: 5 fallos / 600s. Ver
test_telemetry.py para casos.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

if TYPE_CHECKING:
    from hermes.config import Settings

logger = logging.getLogger(__name__)


class Telemetry:
    # Sprint 17 (F2-1): consecutive-failure threshold. Mantenemos
    # defaults conservativos para no escalar en blips transitorios
    # (red lenta, InfluxDB con un GC). 5 fallos / 600s = ~2 min de
    # outage antes de alertar, razonable.
    _FAILURE_ESCALATION_THRESHOLD = 5
    _FAILURE_ESCALATION_WINDOW_S = 600.0

    def __init__(self, settings: Settings) -> None:
        self.enabled = bool(settings.influx_url and settings.influx_token)
        if not self.enabled:
            logger.warning(
                "telemetry_disabled", extra={"reason": "missing influx_url or influx_token"}
            )
            self._client = None
            self._write_api = None
            return
        self._client = InfluxDBClient(
            url=settings.influx_url,
            token=settings.influx_token,
            org=settings.influx_org,
            timeout=10_000,
        )
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
        self._bucket = settings.influx_bucket
        # Sprint 17 (F2-1): failure tracking state
        self._consecutive_failures = 0
        self._first_failure_at: float | None = None
        self._escalated = False
        logger.info(
            "telemetry_initialized",
            extra={"url": settings.influx_url, "bucket": self._bucket},
        )

    async def aclose(self) -> None:
        if self._client is not None:
            self._client.close()
            logger.info("telemetry_closed")

    def record_message(self, *, result: str) -> None:
        if not self.enabled:
            return
        self._write(Point("hermes_messages").tag("result", result).field("count", 1))

    def record_llm_call(
        self, *, model: str, status: str, latency_ms: int, tokens_in: int, tokens_out: int
    ) -> None:
        if not self.enabled:
            return
        self._write(
            Point("hermes_llm_call")
            .tag("model", model)
            .tag("status", status)
            .field("latency_ms", latency_ms)
            .field("tokens_in", tokens_in)
            .field("tokens_out", tokens_out)
            .field("count", 1)
        )

    def record_tool_call(self, *, tool: str, status: str, latency_ms: int) -> None:
        if not self.enabled:
            return
        self._write(
            Point("hermes_tool_call")
            .tag("tool", tool)
            .tag("status", status)
            .field("latency_ms", latency_ms)
            .field("count", 1)
        )

    def record_circuit_breaker_state(self, *, model: str, state: str) -> None:
        if not self.enabled:
            return
        state_value = {"closed": 0, "half-open": 1, "open": 2}.get(state, -1)
        self._write(Point("hermes_circuit_breaker").tag("model", model).field("state", state_value))

    def record_heartbeat(self) -> None:
        if not self.enabled:
            return
        self._write(Point("hermes_heartbeat").field("uptime_s", int(time.time())))

    def _write(self, point: Point) -> None:
        if self._write_api is None:
            return
        try:
            self._write_api.write(
                bucket=self._bucket,
                record=point,
                write_precision=WritePrecision.S,  # type: ignore[arg-type]
            )
            # Sprint 17 (F2-1): write succeeded -> reset failure tracking.
            # Esto es importante: si el write ahora funciona (ej. tras
            # rotar el token), el operador no recibe un error de
            # escalation stale. El log de INFO deja trail del recovery.
            if self._consecutive_failures > 0 or self._escalated:
                logger.info(
                    "telemetry_recovered",
                    extra={
                        "consecutive_failures": self._consecutive_failures,
                        "was_escalated": self._escalated,
                    },
                )
                self._consecutive_failures = 0
                self._first_failure_at = None
                self._escalated = False
        except Exception as exc:
            self._handle_write_failure(point, exc)

    def _handle_write_failure(self, point: Point, exc: Exception) -> None:
        """Sprint 17 (F2-1): record failure, escalate if threshold hit.

        Failure tracking:
        - First failure: log WARNING, set _first_failure_at = now
        - Subsequent failures: increment _consecutive_failures
        - If counter >= threshold AND elapsed < window: escalate to
          ERROR with telemetry_escalated event + reset state (to
          avoid spamming ERROR on every subsequent write)
        - After window expires, reset (so a new outage gets a fresh
          escalation cycle)
        """
        now = time.time()
        if self._first_failure_at is None:
            self._first_failure_at = now
        self._consecutive_failures += 1
        elapsed = now - self._first_failure_at

        # Log WARNING siempre (trail per-call, no escalado hasta threshold)
        logger.warning(
            "telemetry_write_failed",
            extra={
                "error": str(exc),
                "consecutive_failures": self._consecutive_failures,
                "elapsed_s": round(elapsed, 1),
            },
        )

        # Si ya escalamos, no re-escalamos (un solo ERROR por outage)
        if self._escalated:
            return

        # Threshold: N failures dentro de la ventana -> escalate
        if (
            self._consecutive_failures >= self._FAILURE_ESCALATION_THRESHOLD
            and elapsed <= self._FAILURE_ESCALATION_WINDOW_S
        ):
            self._escalated = True
            logger.error(
                "telemetry_escalated",
                extra={
                    "consecutive_failures": self._consecutive_failures,
                    "elapsed_s": round(elapsed, 1),
                    "threshold": self._FAILURE_ESCALATION_THRESHOLD,
                    "window_s": self._FAILURE_ESCALATION_WINDOW_S,
                    "last_error": str(exc),
                    "action_required": (
                        "Check INFLUX_TOKEN validity. Use "
                        "'docker exec <INFLUXDB_CONTAINER> influx auth list' "
                        "to see active tokens."
                    ),
                },
            )
            # Reset failure counter para no spammear, pero mantener
            # _escalated=True hasta que un write funcione.
            self._consecutive_failures = 0
            self._first_failure_at = None
        elif elapsed > self._FAILURE_ESCALATION_WINDOW_S:
            # Ventana expirada sin suficientes failures -> reset y
            # empezar fresh
            self._first_failure_at = now
            self._consecutive_failures = 1
