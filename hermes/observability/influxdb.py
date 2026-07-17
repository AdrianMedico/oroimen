"""InfluxDB line protocol para métricas de research jobs.

Ver TDD_S14_DEEP_RESEARCH.md §13.2.

Por qué InfluxDB y NO Prometheus (post-pregunta del user, TDD round 3 Q3):
- Stack existente del proyecto: InfluxDB v2.x + Grafana como capa de
  observabilidad. El codebase ya tiene `hermes/telemetry.py` con este stack.
- Prometheus añadiría infra redundante (otro exporter, otro puerto) y tiene
  el problema multi-worker: si uvicorn arranca >1 worker, los descriptores
  /metrics colisionan. InfluxDB es append-only (tags indexed), no colisiona.

> Phase 2 audit: la URL concreta de InfluxDB (host/port) y el nombre de la
> org NO son valores por defecto genericos. Cada deployment los configura
> via env vars (INFLUXDB_URL, INFLUXDB_ORG); ver seccion env-vars abajo.

Diseño:
- Singleton `_client`/`_write_api` inicializado en startup() desde __main__.py.
- `init_influx()` lee env vars (INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG,
  INFLUXDB_BUCKET_RESEARCH). Si falta URL/token, no-op graceful (dev/CI).
- `write_research_metric()` fire-and-forget: si InfluxDB no inicializado
  o write falla, log warning, NO raise. Las métricas son best-effort; un
  fallo de InfluxDB no debe tumbar un research job.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import WriteApi

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Singleton — inicializado en hermes/__main__.py:startup(). None hasta entonces.
_client: InfluxDBClient | None = None
_write_api: WriteApi | None = None
_bucket: str = "mnemosyne_research"
_org: str = ""
_initialized: bool = False


def init_influx() -> None:
    """Inicializa cliente InfluxDB. Llamado desde hermes/__main__.py:startup().

    Si INFLUXDB_URL o INFLUXDB_TOKEN no están set, no-op graceful (log
    warning, no crash). Esto permite que S14 funcione en dev sin InfluxDB
    (ej. tests en CI).

    Env vars leídas:
        INFLUXDB_URL (required para activar)
        INFLUXDB_TOKEN (required para activar)
        INFLUXDB_ORG (required para activar, sin default - log warning
            y degradamos a no-op si falta)
        INFLUXDB_BUCKET_RESEARCH (default "mnemosyne_research")
    """
    global _client, _write_api, _bucket, _org, _initialized

    url = os.environ.get("INFLUXDB_URL")
    token = os.environ.get("INFLUXDB_TOKEN")
    _org = os.environ.get("INFLUXDB_ORG", "")
    _bucket = os.environ.get("INFLUXDB_BUCKET_RESEARCH", "mnemosyne_research")

    if not url or not token:
        logger.warning(
            "influxdb_disabled",
            extra={
                "reason": "missing_url_or_token",
                "has_url": bool(url),
                "has_token": bool(token),
            },
        )
        _initialized = False
        return

    _client = InfluxDBClient(url=url, token=token, org=_org, timeout=5000)
    # Batch 50 puntos o flush cada 10s (BATCH_SIZE + FLUSH_INTERVAL) — reduce
    # round-trips sin perder datos en crash corto.
    _write_api = _client.write_api(
        write_options=__import__(
            "influxdb_client.client.write_api", fromlist=["WriteOptions"]
        ).WriteOptions(batch_size=50, flush_interval=10_000),
    )
    _initialized = True
    logger.info(
        "influxdb_initialized",
        extra={"url": url, "bucket": _bucket, "org": _org},
    )


def write_research_metric(
    measurement: str,
    tags: dict[str, str | int | bool] | None = None,
    fields: dict[str, str | int | float | bool] | None = None,
) -> None:
    """Helper de escritura fire-and-forget.

    Args:
        measurement: nombre del measurement (e.g. 'research_job_created').
        tags: dict de tag key → value (string-coercible). Tags indexados.
        fields: dict de field key → value (numeric o string).

    Behavior:
        - Si InfluxDB no inicializado (dev/CI sin env vars), no-op silencioso.
        - Si write falla (network, auth), log warning, NO raise. Métricas son
          best-effort; un fallo de InfluxDB no debe tumbar un research job.
    """
    if not _initialized or _write_api is None:
        return  # dev/CI: InfluxDB disabled
    if not measurement:
        return
    try:
        point = Point(measurement)
        for k, v in (tags or {}).items():
            point = point.tag(k, str(v))
        for k, v in (fields or {}).items():  # type: ignore[assignment]
            point = point.field(k, v)
        _write_api.write(bucket=_bucket, org=_org, record=point)
    except Exception as exc:
        logger.warning(
            "influxdb_write_failed",
            extra={"measurement": measurement, "error": str(exc)},
        )


def write_research_metrics_batch(
    points: list[dict[str, object]],
) -> None:
    """Escribe varios points a la vez (más eficiente que N calls individuales).

    Cada point es un dict con keys: measurement (str), tags (dict, opcional),
    fields (dict).

    Mismo manejo de errores fire-and-forget que write_research_metric.
    """
    if not _initialized or _write_api is None:
        return
    if not points:
        return
    try:
        batch: list[Point] = []
        for spec in points:
            measurement = spec.get("measurement")
            if not isinstance(measurement, str):
                continue
            point = Point(measurement)
            tags_dict = spec.get("tags") or {}
            if isinstance(tags_dict, dict):
                for k, v in tags_dict.items():
                    point = point.tag(k, str(v))
            fields_dict = spec.get("fields") or {}
            if isinstance(fields_dict, dict):
                for k, v in fields_dict.items():
                    point = point.field(k, v)
            batch.append(point)
        if batch:
            _write_api.write(bucket=_bucket, org=_org, record=batch)
    except Exception as exc:
        logger.warning(
            "influxdb_batch_write_failed",
            extra={"count": len(points), "error": str(exc)},
        )


def close_influx() -> None:
    """Cleanup en shutdown. Cierra el write_api y el cliente. Idempotent."""
    global _client, _write_api, _initialized
    if _write_api is not None:
        with contextlib.suppress(Exception):
            _write_api.close()
        _write_api = None
    if _client is not None:
        with contextlib.suppress(Exception):
            _client.close()
        _client = None
    _initialized = False


def is_initialized() -> bool:
    """Helper de test: True si init_influx() fue exitoso."""
    return _initialized
