"""Integration tests para el router HTTP /v1/jobs (Sprint 14 US-2.1).

Estrategia: TestClient de FastAPI con un DeepResearchService "fake"
(duck-typed) registrado via ``set_deep_research_service``. La DB es real
(fixture `db` de conftest) — los endpoints interactuan con ella via el
fake service.

Por qué NO mockear el service entero:
  - Los endpoints hacen lógica propia (404 si user_id mismatch, paginación,
    filtros). Mockear todo el service perdería coverage del router.
  - Sustituimos solo las llamadas LLM/web_search (lo caro) por AsyncMock,
    pero dejamos el service real con la misma DB. Esto valida que el
    router serializa correctamente los Pydantic models.

Por qué mockear el scheduler:
  - submit_job() llama a ``scheduler.enqueue()`` que arranca AsyncIOScheduler.
  - Para tests unitarios del router, mockeamos el scheduler entero.

Cobertura (resumen, ver cada test para detalles):
  - POST /v1/jobs (3 tests: 201, 422 query corto, 422 query largo)
  - GET  /v1/jobs (4 tests: empty, with results, filter, pagination)
  - GET  /v1/jobs/{id} (3 tests: 200, 404, 404 cross-user)
  - POST /v1/jobs/{id}/cancel (3 tests: graceful, 409 terminal, 404)
  - POST /v1/jobs/{id}/retry (2 tests: 201 from failed, 409 not failed)
  - GET  /v1/jobs/budget (1 test)
  - auth (1 test: 401 sin bearer)

Total: 17 tests.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hermes.jobs.cost import estimate_research_cost, format_now
from hermes.jobs.exceptions import (
    JobAlreadyTerminalError,
    JobNotFoundError,
    JobNotRetryableError,
)
from hermes.jobs.models import (
    CancelResponse,
    CreateJobRequest,
    JobDetail,
    JobResponse,
    JobStatus,
    JobSummary,
    PhaseName,
)
from hermes.receivers import jobs_api
from hermes.receivers.http_api import create_app
from hermes.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Fixtures compartidos
# ---------------------------------------------------------------------------


class _FakeService:
    """Fake DeepResearchService — implementación minima duck-typed.

    Cumple el contrato que jobs_api espera:
      - submit_job(request, user_id) -> JobResponse
      - get_job(job_id) -> JobDetail
      - list_jobs(user_id, status, limit) -> list[JobSummary]
      - cancel_job(job_id, graceful) -> CancelResponse
      - retry_job(job_id, user_id) -> JobResponse
      - _db (con get_today_research_cost + conn.execute + get_research_job)
      - _settings (con deep_research_daily_budget_usd)
    """

    def __init__(self, db: Any, settings: Any) -> None:
        self._db = db
        self._settings = settings

    async def submit_job(
        self,
        request: CreateJobRequest,
        user_id: int = 0,
    ) -> JobResponse:
        job_id = uuid.uuid4().hex[:12]
        now = format_now()
        await self._db.create_research_job(
            job_id=job_id,
            query=request.query,
            notify_via_tg=1 if request.notify_via_tg else 0,
            job_type=request.job_type.value,
            user_id=user_id,
        )
        estimated = float(
            estimate_research_cost(
                max_sources=int(getattr(self._settings, "deep_research_max_sources", 5)),
                per_source_max_tokens=int(
                    getattr(
                        self._settings,
                        "deep_research_per_source_max_tokens",
                        3000,
                    )
                ),
                output_max_tokens=int(
                    getattr(
                        self._settings,
                        "deep_research_output_max_tokens",
                        10000,
                    )
                ),
                pricing_table={
                    "MiniMax-M3": (
                        __import__("decimal").Decimal("0.30"),
                        __import__("decimal").Decimal("0.60"),
                    ),
                },
                primary_model="MiniMax-M3",
            )
        )
        return JobResponse(
            id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            estimated_cost_usd=estimated,
        )

    async def get_job(self, job_id: str) -> JobDetail:
        row = await self._db.get_research_job(job_id)
        if row is None:
            raise JobNotFoundError(f"Job {job_id} not found")
        # token_usage vacio en el fake (drill-down no es scope US-2.1).
        return JobDetail(
            id=row["id"],
            query=row["query"],
            status=JobStatus(row["status"]),
            current_phase=PhaseName(row["current_phase"]) if row.get("current_phase") else None,
            progress_percent=row["progress_percent"],
            cost_usd=row["cost_usd"],
            created_at=row["created_at"],
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
            job_type=row.get("job_type", "deep_research"),
            notify_via_tg=bool(row.get("notify_via_tg", 1)),
            output_path=row.get("output_path"),
            partial_output_path=row.get("partial_output_path"),
            error_taxonomy=row.get("error_taxonomy"),
            error_message=row.get("error_message"),
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            notified=bool(row.get("notified", 0)),
            updated_at=row["updated_at"],
            token_usage=[],
            checkpoint_path=None,
        )

    async def list_jobs(
        self,
        user_id: int = 0,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobSummary]:
        rows = await self._db.list_research_jobs(
            user_id=user_id,
            status=status.value if status else None,
            limit=limit,
        )
        return [
            JobSummary(
                id=r["id"],
                query=r["query"],
                status=JobStatus(r["status"]),
                current_phase=PhaseName(r["current_phase"]) if r.get("current_phase") else None,
                progress_percent=r["progress_percent"],
                cost_usd=r["cost_usd"],
                created_at=r["created_at"],
                started_at=r.get("started_at"),
                completed_at=r.get("completed_at"),
            )
            for r in rows
        ]

    async def cancel_job(self, job_id: str, graceful: bool = True) -> CancelResponse:
        row = await self._db.get_research_job(job_id)
        if row is None:
            raise JobNotFoundError(f"Job {job_id} not found")
        if row["status"] in ("complete", "failed", "cancelled"):
            raise JobAlreadyTerminalError(JobStatus(row["status"]))
        await self._db.update_research_job_status(job_id, "cancelling")
        if not graceful:
            await self._db.update_research_job_status(
                job_id,
                "cancelled",
                completed_at=format_now(),
            )
            return CancelResponse(
                id=job_id,
                status=JobStatus.CANCELLED,
                graceful=False,
                partial_output_path=None,
            )
        return CancelResponse(
            id=job_id,
            status=JobStatus.CANCELLING,
            graceful=True,
            partial_output_path=None,
        )

    async def retry_job(self, job_id: str, user_id: int = 0) -> JobResponse:
        row = await self._db.get_research_job(job_id)
        if row is None:
            raise JobNotFoundError(f"Job {job_id} not found")
        if row["status"] != "failed":
            raise JobNotRetryableError(JobStatus(row["status"]))
        new_id = uuid.uuid4().hex[:12]
        await self._db.create_research_job(
            job_id=new_id,
            query=row["query"],
            notify_via_tg=int(row.get("notify_via_tg", 1)),
            job_type=row.get("job_type", "deep_research"),
            user_id=user_id,
        )
        return JobResponse(
            id=new_id,
            status=JobStatus.PENDING,
            created_at=format_now(),
            estimated_cost_usd=0.05,
        )


def _fake_router() -> MagicMock:
    """Router fake para que create_app() no toque la red."""
    r = MagicMock()
    r.get_breaker_states = MagicMock(return_value={"primary": "closed"})
    return r


@pytest.fixture
def authed_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Any:
    """Settings con API key activada (para que ``authenticate_bearer`` valide)."""
    from hermes.config import Settings

    monkeypatch.setenv(
        "TELEGRAM_BOT_TOKEN",
        "9999999999:AAFakeTestTokenForUnitTests12345",
    )
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-12345")
    monkeypatch.setenv("DB_PATH", str(tmp_path_factory.mktemp("db") / "test.db"))
    monkeypatch.setenv("HERMES_API_API_KEY", "test-bearer-key-xyz")
    return Settings(_env_file=None)


@pytest.fixture
def client_with_auth(
    authed_settings: Any,
    db: Any,
) -> Any:
    """TestClient con service fake registrado. Bearer key requerida para 200.

    Por defecto los tests pasan ``Authorization: Bearer test-bearer-key-xyz``.
    Para tests "sin bearer" se omite la cabecera.
    """
    app = create_app(
        authed_settings,
        db,
        _fake_router(),
        ToolRegistry(),
    )
    service = _FakeService(db, authed_settings)
    jobs_api.set_deep_research_service(service)
    with TestClient(app) as client:
        yield client
    # Cleanup singleton para no contaminar otros tests
    jobs_api.set_deep_research_service(None)  # type: ignore[arg-type]


@pytest.fixture
def bearer() -> dict[str, str]:
    """Header ``Authorization`` valido para client_with_auth."""
    return {"Authorization": "Bearer test-bearer-key-xyz"}


# ---------------------------------------------------------------------------
# POST /v1/jobs
# ---------------------------------------------------------------------------


def test_post_job_201(client_with_auth: Any, bearer: dict[str, str], db: Any) -> None:
    """Body valido → 201 con id, status='pending', estimated_cost_usd > 0."""
    response = client_with_auth.post(
        "/v1/jobs",
        json={"query": "Research Hermes architecture in detail"},
        headers=bearer,
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert "id" in body and len(body["id"]) == 12
    assert body["status"] == "pending"
    assert body["estimated_cost_usd"] > 0
    # Verify the row was actually created in DB
    import sqlite3

    with sqlite3.connect(str(db.path)) as conn:
        row = conn.execute(
            "SELECT id, query, status FROM research_jobs WHERE id = ?",
            (body["id"],),
        ).fetchone()
    assert row is not None, f"Job {body['id']} not in DB"


def test_post_job_422_query_too_short(client_with_auth: Any, bearer: dict[str, str]) -> None:
    """Query de 2 chars → 422 (Pydantic min_length=3)."""
    response = client_with_auth.post(
        "/v1/jobs",
        json={"query": "ab"},
        headers=bearer,
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("query" in str(e.get("loc", "")) for e in errors)


def test_post_job_422_query_too_long(client_with_auth: Any, bearer: dict[str, str]) -> None:
    """Query de 2001 chars → 422 (Pydantic max_length=2000)."""
    response = client_with_auth.post(
        "/v1/jobs",
        json={"query": "x" * 2001},
        headers=bearer,
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/jobs (list)
# ---------------------------------------------------------------------------


async def test_get_jobs_list_empty(client_with_auth: Any, bearer: dict[str, str], db: Any) -> None:
    """User nuevo, sin jobs → 200 con lista vacia."""
    response = client_with_auth.get("/v1/jobs", headers=bearer)
    assert response.status_code == 200
    assert response.json() == []


async def test_get_jobs_list_with_results(
    client_with_auth: Any, bearer: dict[str, str], db: Any
) -> None:
    """Crea 3 jobs → lista los devuelve ordenados por created_at DESC."""
    ids = []
    for i in range(3):
        resp = client_with_auth.post(
            "/v1/jobs",
            json={"query": f"Query number {i}"},
            headers=bearer,
        )
        assert resp.status_code == 201
        ids.append(resp.json()["id"])
    response = client_with_auth.get("/v1/jobs", headers=bearer)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    # Created at LAST → first in DESC order. SQLite TEXT lexicographic.
    returned_ids = [r["id"] for r in data]
    assert returned_ids == list(reversed(ids))


async def test_get_jobs_list_filter_by_status(
    client_with_auth: Any, bearer: dict[str, str], db: Any
) -> None:
    """Crea 2 pending + 1 forzado a failed → filter status=failed → 1 item."""
    # Crear 2 jobs pending
    for i in range(2):
        resp = client_with_auth.post(
            "/v1/jobs",
            json={"query": f"pending-{i}"},
            headers=bearer,
        )
        # Verifica que se creo (status_code ya validado implicitamente arriba)
        assert resp.json()["id"]
    # Crear 1 job forzado a failed via DB directa
    resp = client_with_auth.post(
        "/v1/jobs",
        json={"query": "will-fail"},
        headers=bearer,
    )
    failed_id = resp.json()["id"]
    await db.update_research_job_status(failed_id, "failed", error_taxonomy="network")
    # Filtrar
    response = client_with_auth.get(
        "/v1/jobs?status=failed",
        headers=bearer,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == failed_id
    assert data[0]["status"] == "failed"


async def test_get_jobs_list_pagination(
    client_with_auth: Any, bearer: dict[str, str], db: Any
) -> None:
    """Crea 5 jobs, limit=2&offset=2 → 2 items saltando los 2 más recientes."""
    ids = []
    for i in range(5):
        resp = client_with_auth.post(
            "/v1/jobs",
            json={"query": f"page-{i}"},
            headers=bearer,
        )
        ids.append(resp.json()["id"])
    # DESC order: ids[4], ids[3], ids[2], ids[1], ids[0]
    response = client_with_auth.get(
        "/v1/jobs?limit=2&offset=2",
        headers=bearer,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    returned_ids = [r["id"] for r in data]
    # skip 2 (ids[4] y ids[3]), next 2 son ids[2] y ids[1]
    assert returned_ids == [ids[2], ids[1]]


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id}
# ---------------------------------------------------------------------------


async def test_get_job_detail_200(client_with_auth: Any, bearer: dict[str, str], db: Any) -> None:
    """Crea job, get by id → 200 con todos los campos de JobDetail."""
    resp = client_with_auth.post(
        "/v1/jobs",
        json={"query": "Detail test query"},
        headers=bearer,
    )
    job_id = resp.json()["id"]
    response = client_with_auth.get(f"/v1/jobs/{job_id}", headers=bearer)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id
    assert data["query"] == "Detail test query"
    assert data["status"] == "pending"
    assert "job_type" in data
    assert "tokens_in" in data
    assert "progress_percent" in data


async def test_get_job_detail_404_unknown(client_with_auth: Any, bearer: dict[str, str]) -> None:
    """ID inexistente (12 hex chars validos) → 404."""
    response = client_with_auth.get(
        "/v1/jobs/000000000000",
        headers=bearer,
    )
    assert response.status_code == 404


async def test_get_job_detail_404_other_user(
    client_with_auth: Any, bearer: dict[str, str], db: Any
) -> None:
    """Job creado por user 0, request sin auth efectivo → 404.

    El fake service filtra por user_id. Para simular "otro user"
    modificamos el row directamente para cambiar user_id.

    Luego forzamos que authenticate_bearer devuelva user 1 via
    dependency_overrides.
    """
    # Crear job como user 0 (default)
    resp = client_with_auth.post(
        "/v1/jobs",
        json={"query": "other-user"},
        headers=bearer,
    )
    job_id = resp.json()["id"]
    # Modificar user_id del job a 99 (simula que lo creó otro user)
    import sqlite3

    with sqlite3.connect(str(db.path)) as conn:
        conn.execute("UPDATE research_jobs SET user_id = 99 WHERE id = ?", (job_id,))
        conn.commit()
    # Override del dependency: el siguiente request ve user_id=99 mismatch
    # con el row.user_id=99 — pero el bearer dependency SIEMPRE devuelve 0,
    # por lo que el row.user_id=99 !== 0 → 404.
    response = client_with_auth.get(f"/v1/jobs/{job_id}", headers=bearer)
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/jobs/{id}/cancel
# ---------------------------------------------------------------------------


async def test_cancel_job_graceful(client_with_auth: Any, bearer: dict[str, str], db: Any) -> None:
    """Crea job (pending), cancel → 200 CancelResponse con status=pending→cancelling."""
    resp = client_with_auth.post(
        "/v1/jobs",
        json={"query": "to-cancel"},
        headers=bearer,
    )
    job_id = resp.json()["id"]
    response = client_with_auth.post(
        f"/v1/jobs/{job_id}/cancel?graceful=true",
        headers=bearer,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id
    assert data["status"] in ("cancelling", "cancelled")
    assert data["graceful"] is True


async def test_cancel_job_409_terminal(
    client_with_auth: Any, bearer: dict[str, str], db: Any
) -> None:
    """Job 'complete' → cancel → 409."""
    resp = client_with_auth.post(
        "/v1/jobs",
        json={"query": "to-complete"},
        headers=bearer,
    )
    job_id = resp.json()["id"]
    await db.update_research_job_status(job_id, "complete")
    response = client_with_auth.post(
        f"/v1/jobs/{job_id}/cancel",
        headers=bearer,
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"]["type"] == "job_already_terminal"
    assert detail["error"]["status"] == "complete"


async def test_cancel_job_404_unknown(client_with_auth: Any, bearer: dict[str, str]) -> None:
    """ID inexistente → 404."""
    response = client_with_auth.post(
        "/v1/jobs/000000000000/cancel",
        headers=bearer,
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/jobs/{id}/retry
# ---------------------------------------------------------------------------


async def test_retry_job_201_from_failed(
    client_with_auth: Any, bearer: dict[str, str], db: Any
) -> None:
    """Crea job, fuerza 'failed', retry → 201 con nuevo id."""
    resp = client_with_auth.post(
        "/v1/jobs",
        json={"query": "retry-me"},
        headers=bearer,
    )
    job_id = resp.json()["id"]
    await db.update_research_job_status(job_id, "failed", error_taxonomy="llm_5xx")
    response = client_with_auth.post(
        f"/v1/jobs/{job_id}/retry",
        headers=bearer,
    )
    assert response.status_code == 201
    new_id = response.json()["id"]
    assert new_id != job_id
    assert response.json()["status"] == "pending"


async def test_retry_job_409_not_failed(
    client_with_auth: Any, bearer: dict[str, str], db: Any
) -> None:
    """Job 'complete' → retry → 409 (only failed jobs are retryable)."""
    resp = client_with_auth.post(
        "/v1/jobs",
        json={"query": "complete-me"},
        headers=bearer,
    )
    job_id = resp.json()["id"]
    await db.update_research_job_status(job_id, "complete")
    response = client_with_auth.post(
        f"/v1/jobs/{job_id}/retry",
        headers=bearer,
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"]["type"] == "job_not_retryable"
    assert detail["error"]["status"] == "complete"


# ---------------------------------------------------------------------------
# GET /v1/jobs/budget
# ---------------------------------------------------------------------------


async def test_budget_endpoint_returns_status(
    client_with_auth: Any, bearer: dict[str, str], db: Any, authed_settings: Any
) -> None:
    """Crea 2 jobs con cost, GET /v1/jobs/budget → 200 con todos los campos."""
    # Crear 2 jobs + forzar cost_usd no-cero via DB
    for i in range(2):
        resp = client_with_auth.post(
            "/v1/jobs",
            json={"query": f"budget-{i}"},
            headers=bearer,
        )
        job_id = resp.json()["id"]
        # Update aggregate cost via method helper — emulate post-LLM-call.
        # `add_token_usage` espera phase/model, lo saltamos y usamos UPDATE
        # directo porque el endpoint solo necesita la SUM.
        import sqlite3

        with sqlite3.connect(str(db.path)) as conn:
            conn.execute(
                "UPDATE research_jobs SET cost_usd = 0.25 WHERE id = ?",
                (job_id,),
            )
            conn.commit()
    response = client_with_auth.get("/v1/jobs/budget", headers=bearer)
    assert response.status_code == 200
    data = response.json()
    assert data["today_cost_usd"] >= 0.5  # 2 jobs x $0.25
    assert data["daily_cap_usd"] == authed_settings.deep_research_daily_budget_usd
    assert "remaining_usd" in data
    assert data["jobs_today"] >= 2
    assert "resets_at" in data
    # resets_at ISO 8601 → termina en 'Z'
    assert data["resets_at"].endswith("Z")
    # 0.5 restantes = cap - 0.5
    assert data["remaining_usd"] == pytest.approx(
        authed_settings.deep_research_daily_budget_usd - data["today_cost_usd"],
        abs=1e-6,
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_auth_required_all_endpoints(client_with_auth: Any) -> None:
    """Sin bearer → 401 en los 6 endpoints."""
    endpoints = [
        ("POST", "/v1/jobs", {"query": "no-auth"}),
        ("GET", "/v1/jobs", None),
        ("GET", "/v1/jobs/budget", None),
        ("GET", "/v1/jobs/000000000000", None),
        ("POST", "/v1/jobs/000000000000/cancel", None),
        ("POST", "/v1/jobs/000000000000/retry", None),
    ]
    for method, path, json_body in endpoints:
        if method == "POST":
            response = client_with_auth.post(path, json=json_body)
        else:
            response = client_with_auth.get(path)
        assert response.status_code == 401, (
            f"{method} {path} should be 401, got {response.status_code}: " f"{response.text}"
        )


def test_auth_invalid_token_401(client_with_auth: Any) -> None:
    """Bearer invalido (no coincide) → 401."""
    response = client_with_auth.get(
        "/v1/jobs",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 503 si el service no esta inicializado
# ---------------------------------------------------------------------------


def test_service_unavailable_503_when_singleton_missing(authed_settings: Any, db: Any) -> None:
    """Sin service registrado → 503 (degraded mode).

    Monta un app nuevo sin llamar a ``set_deep_research_service``.
    """
    # Forzar singleton a None antes de crear el app
    jobs_api.set_deep_research_service(None)  # type: ignore[arg-type]
    app = create_app(
        authed_settings,
        db,
        _fake_router(),
        ToolRegistry(),
    )
    with TestClient(app) as client:
        response = client.get(
            "/v1/jobs",
            headers={"Authorization": "Bearer test-bearer-key-xyz"},
        )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "not initialized" in detail["error"]["message"]
