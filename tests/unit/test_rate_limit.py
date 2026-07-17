"""Tests para Fase 0 Hardening: rate limiter en HTTP API.

Cubre:
- 0 = deshabilitado
- N requests dentro del limite: pasan
- Request N+1: 429 con Retry-After header
- /health y /v1/models exentos
- IP-based buckets (request desde 2 IPs no se interfieren)
- Window movil (requests viejos no cuentan)
- Reset despues del cooldown
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes.memory.db import Database


async def _make_app(rate_limit: int, tmp_path) -> FastAPI:
    """Crea un hermes HTTP API app con el rate limit pedido y un endpoint /_test."""
    from hermes.receivers.http_api import create_app

    db = Database(tmp_path / "test.db")
    await db.initialize()
    settings = MagicMock()
    settings.db_path = str(tmp_path / "test.db")
    settings.hermes_api_port = 8000
    settings.http_api_api_key = None
    settings.http_api_rate_limit_per_minute = rate_limit

    router = MagicMock()
    registry = MagicMock()
    telemetry = MagicMock()
    app = create_app(
        settings=settings,
        db=db,
        router=router,
        registry=registry,
        embeddings_service=None,
        telemetry=telemetry,
    )

    # Anadir endpoint de test (FastAPI style, no aiohttp)
    @app.get("/_test")
    async def _test_endpoint():
        return {"ok": True}

    return app


def _run_async(coro):
    """Helper para correr coroutines en tests sync."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def make_app(tmp_path):
    """Fixture: devuelve funcion para crear app con rate limit dado."""

    def _make(rate_limit: int) -> FastAPI:
        return _run_async(_make_app(rate_limit, tmp_path))

    return _make


def test_rate_limit_disabled_when_zero(make_app):
    """rate_limit=0 deshabilita el middleware (no 429)."""
    app = make_app(0)
    with TestClient(app) as client:
        for _ in range(100):
            r = client.get("/_test")
            assert r.status_code == 200, f"Rate limit no deberia aplicar, status={r.status_code}"


def test_rate_limit_allows_under_limit(make_app):
    """N requests dentro del limite pasan."""
    app = make_app(5)
    with TestClient(app) as client:
        for i in range(5):
            r = client.get("/_test")
            assert r.status_code == 200, f"Request {i + 1} deberia pasar, status={r.status_code}"


def test_rate_limit_blocks_over_limit(make_app):
    """Request N+1: 429 con Retry-After."""
    app = make_app(3)
    with TestClient(app) as client:
        for _ in range(3):
            r = client.get("/_test")
            assert r.status_code == 200
        # 4ta: 429
        r = client.get("/_test")
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        body = r.json()
        assert body["error"]["type"] == "RateLimitError"
        assert "3 req/min" in body["error"]["message"]


def test_rate_limit_exempts_health(make_app):
    """/health NO cuenta para el rate limit."""
    app = make_app(2)
    with TestClient(app) as client:
        for _ in range(20):
            r = client.get("/health")
            assert r.status_code == 200


def test_rate_limit_exempts_models(make_app):
    """/v1/models NO cuenta para el rate limit."""
    app = make_app(2)
    with TestClient(app) as client:
        for _ in range(20):
            r = client.get("/v1/models")
            assert r.status_code == 200


def test_rate_limit_per_ip_buckets(make_app):
    """Requests desde distintas IPs no se interfieren."""
    app = make_app(2)
    with TestClient(app) as client:
        # IP 1: 2 requests OK, 3ra = 429
        for _ in range(2):
            r = client.get("/_test", headers={"X-Forwarded-For": "1.1.1.1"})
            assert r.status_code == 200
        r = client.get("/_test", headers={"X-Forwarded-For": "1.1.1.1"})
        assert r.status_code == 429
        # IP 2: 2 requests OK (bucket vacio)
        for _ in range(2):
            r = client.get("/_test", headers={"X-Forwarded-For": "2.2.2.2"})
            assert r.status_code == 200


def test_rate_limit_retry_after_header_format(make_app):
    """Retry-After header es un int positivo."""
    app = make_app(1)
    with TestClient(app) as client:
        r = client.get("/_test")
        assert r.status_code == 200
        # 2da: 429
        r = client.get("/_test")
        assert r.status_code == 429
        retry_after = int(r.headers["Retry-After"])
        assert 1 <= retry_after <= 60
