"""HTTP API router para deep research jobs (Sprint 14, US-2.1).

Ver EPIC2_SYNTHESIS.md §2.4 + TDD_S14_DEEP_RESEARCH.md §10.

Endpoints:
  POST   /v1/jobs                  bearer auth → 201 JobResponse
  GET    /v1/jobs                  bearer auth → 200 [JobSummary]
  GET    /v1/jobs/budget           bearer auth → 200 DailyBudgetStatus
  GET    /v1/jobs/{job_id}         bearer auth → 200 JobDetail | 404
  POST   /v1/jobs/{job_id}/cancel  bearer auth → 200 CancelResponse | 404 | 409
  POST   /v1/jobs/{job_id}/retry   bearer auth → 201 JobResponse | 404 | 409

Todos los endpoints requieren bearer auth via `Depends(authenticate_bearer)`.
El service singleton se obtiene via `Depends(get_deep_research_service_dep)`
que 503ea si el service no está inicializado (degraded mode).

Wiring: `create_app()` en hermes/receivers/http_api.py monta este router
y registra el singleton via `set_deep_research_service(service)` ANTES
de aceptar requests. Patron deliberadamente minimal: este modulo NO
inicializa el service (responsabilidad del startup lifecycle).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from hermes.jobs.exceptions import (
    BudgetExceededError,
    JobAlreadyTerminalError,
    JobNotFoundError,
    JobNotRetryableError,
    SchedulerUnavailableError,
)
from hermes.jobs.models import (
    CancelResponse,
    CreateJobRequest,
    DailyBudgetStatus,
    JobDetail,
    JobResponse,
    JobStatus,
    JobSummary,
)
from hermes.jobs.preflight import (
    DeepResearchPreflight,
    evaluate_deep_research_preflight,
)
from hermes.receivers.auth import authenticate_bearer

logger = logging.getLogger(__name__)

# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/v1", tags=["jobs"])

# ============================================================================
# Singleton service cache
# ============================================================================
# El service se inicializa en hermes/__main__.py:startup() (Track 1 + 1.5 scope,
# ya cerrado). Aqui solo cacheamos referencia para que las routes lo lean
# sin acoplar create_app() al ciclo de vida del AsyncIOScheduler.
#
# Patron deliberado: NO initializar lazy aqui. Si el service no está listo,
# devolvemos 503 (degraded mode) — el cliente decide si reintenta o muestra
# "feature unavailable" en la UI. Esto es mejor que auto-spinner el scheduler
# desde un request HTTP, que mezclaria lifetimes de event loops.

_service_singleton: Any | None = None


def set_deep_research_service(service: Any) -> None:
    """Registra el singleton del service (llamado desde hermes/__main__.py:startup).

    Args:
        service: instancia de ``DeepResearchService``.
    """
    global _service_singleton
    _service_singleton = service
    logger.info("jobs_api_service_registered")


def get_deep_research_service_dep() -> Any:
    """FastAPI dependency que devuelve el service singleton.

    Raises:
        HTTPException 503 si el service no está inicializado.
    """
    if _service_singleton is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "type": "service_unavailable",
                    "message": (
                        "DeepResearchService not initialized. "
                        "Wait for startup to complete or check container logs."
                    ),
                }
            },
        )
    return _service_singleton


# ============================================================================
# Helpers de paginación
# ============================================================================
# db.list_research_jobs() (Track 1) solo soporta `limit`, no `offset`.
# Hacer offset en la API layer con un sub-select `LIMIT (offset+limit)` y
# slice en Python es correcto: SQL `WHERE user_id=? AND status=?` es
# estable bajo ORDER BY created_at DESC (índice idx_research_jobs_user_status_created).
# Memory cost: limit+offset filas en memoria, capado por el max(200) de limit.
# No tocamos db.py (es scope Track 1 / 1.5, ya cerrado).

_OFFSET_SLICE_CAP = 1000  # defensa: offset > 1000 probablemente es bug del cliente


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_job(
    body: CreateJobRequest,
    user_id: Annotated[int, Depends(authenticate_bearer)],
    service: Annotated[Any, Depends(get_deep_research_service_dep)],
) -> JobResponse:
    """POST /v1/jobs — crea un nuevo deep research job.

    201 con JobResponse si OK.
    422 si el body es inválido (Pydantic auto).
    429 si el daily budget está agotado (BudgetExceededError).
    503 si el service no está inicializado o el scheduler no arrancó.
    """
    try:
        return await service.submit_job(
            request=body,
            user_id=user_id,
        )
    except BudgetExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "type": "budget_exceeded",
                    "message": str(exc),
                }
            },
        ) from exc
    except SchedulerUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "type": "scheduler_unavailable",
                    "message": str(exc),
                }
            },
        ) from exc


@router.get("/jobs/budget", status_code=status.HTTP_200_OK)
async def get_budget(
    user_id: Annotated[int, Depends(authenticate_bearer)],
    service: Annotated[Any, Depends(get_deep_research_service_dep)],
) -> DailyBudgetStatus:
    """GET /v1/jobs/budget — daily budget status helper.

    Implementación inline (no service.get_daily_budget_status existe en Track 1).
    Fuente: db.get_today_research_cost() + settings.deep_research_daily_budget_usd
    + COUNT(*) directo (jobs_today).

    200 con DailyBudgetStatus.
    503 si el service no está inicializado.
    """
    db = service._db
    settings = service._settings

    cap = float(getattr(settings, "deep_research_daily_budget_usd", 3.0))
    today_cost = await db.get_today_research_cost(user_id=user_id)

    # `today` start = 00:00 UTC. format_now() es UTC, así que basta
    # con un cutoff de medianoche UTC en formato SQLite TEXT.
    now_utc = datetime.now(UTC)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_str = today_start.strftime("%Y-%m-%d %H:%M:%S.000")
    # ISO 8601 reset (próximo 00:00 UTC) para el campo `resets_at`.
    next_reset = (today_start + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    # COUNT(*) directo: jobs del user desde las 00:00 UTC. No expone
    # cancelled (count > 0 incluye cualquier status post-creation,
    # pero el cliente lo quiere como "how many have I started today",
    # no como "how many are actionable").
    count_sql, count_params = _build_count_jobs_sql(user_id, today_start_str)
    async with db.conn.execute(count_sql, count_params) as cur:
        row = await cur.fetchone()
    # ``row`` puede ser ``sqlite3.Row`` (dict-like) o tuple segun backend.
    jobs_today = int(row[0]) if row is not None else 0

    return DailyBudgetStatus(
        today_cost_usd=today_cost,
        daily_cap_usd=cap,
        remaining_usd=max(0.0, cap - today_cost),
        jobs_today=jobs_today,
        resets_at=next_reset,
    )


@router.get("/jobs/preflight", status_code=status.HTTP_200_OK)
async def get_deep_research_preflight(
    request: Request,
    _user_id: Annotated[int, Depends(authenticate_bearer)],
) -> DeepResearchPreflight:
    """Return deterministic offline readiness without requiring the service."""

    settings = request.app.state.settings
    capabilities = request.app.state.deep_research_capabilities
    return evaluate_deep_research_preflight(settings, capabilities)


@router.get("/jobs/{job_id}", status_code=status.HTTP_200_OK)
async def get_job_detail(
    job_id: Annotated[str, Path(min_length=12, max_length=12, pattern=r"^[0-9a-f]{12}$")],
    user_id: Annotated[int, Depends(authenticate_bearer)],
    service: Annotated[Any, Depends(get_deep_research_service_dep)],
) -> JobDetail:
    """GET /v1/jobs/{job_id} — detalle completo de un job.

    200 con JobDetail si el job existe Y pertenece al user_id del token.
    404 si no existe o pertenece a otro user (no leak info cross-user).
    503 si el service no está inicializado.
    """
    try:
        job_detail = await service.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "type": "job_not_found",
                    "message": f"Job {job_id} not found.",
                }
            },
        ) from exc
    await _assert_owner(job_detail, user_id, service)
    return job_detail


@router.post("/jobs/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_job(
    job_id: Annotated[str, Path(min_length=12, max_length=12, pattern=r"^[0-9a-f]{12}$")],
    user_id: Annotated[int, Depends(authenticate_bearer)],
    service: Annotated[Any, Depends(get_deep_research_service_dep)],
    graceful: bool = Query(
        True,
        description=(
            "Si True (default), cancela tras finalizar la phase actual. "
            "Si False, hard cancel inmediato."
        ),
    ),
) -> CancelResponse:
    """POST /v1/jobs/{job_id}/cancel — cancela un job.

    200 con CancelResponse (graceful, partial_output_path si existía).
    404 si no existe o pertenece a otro user.
    409 si ya está en estado terminal (complete/failed/cancelled).
    503 si el service no está inicializado.
    """
    try:
        await _assert_owner_id(job_id, user_id, service)
        return await service.cancel_job(job_id=job_id, graceful=graceful)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "type": "job_not_found",
                    "message": f"Job {job_id} not found.",
                }
            },
        ) from exc
    except JobAlreadyTerminalError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "type": "job_already_terminal",
                    "status": exc.status.value,
                    "message": str(exc),
                }
            },
        ) from exc


@router.post("/jobs/{job_id}/retry", status_code=status.HTTP_201_CREATED)
async def retry_job(
    job_id: Annotated[str, Path(min_length=12, max_length=12, pattern=r"^[0-9a-f]{12}$")],
    user_id: Annotated[int, Depends(authenticate_bearer)],
    service: Annotated[Any, Depends(get_deep_research_service_dep)],
) -> JobResponse:
    """POST /v1/jobs/{job_id}/retry — crea nuevo job desde checkpoint.

    201 con JobResponse (nuevo id) si OK.
    404 si el original no existe o pertenece a otro user.
    409 si el original no está en `failed`.
    503 si el service no está inicializado o scheduler no listo.
    """
    try:
        await _assert_owner_id(job_id, user_id, service)
        return await service.retry_job(job_id=job_id, user_id=user_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "type": "job_not_found",
                    "message": f"Job {job_id} not found.",
                }
            },
        ) from exc
    except JobNotRetryableError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "type": "job_not_retryable",
                    "status": exc.status.value,
                    "message": str(exc),
                }
            },
        ) from exc
    except SchedulerUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "type": "scheduler_unavailable",
                    "message": str(exc),
                }
            },
        ) from exc


@router.get("/jobs", status_code=status.HTTP_200_OK)
async def list_jobs(
    user_id: Annotated[int, Depends(authenticate_bearer)],
    service: Annotated[Any, Depends(get_deep_research_service_dep)],
    status_filter: Annotated[
        JobStatus | None,
        Query(
            alias="status",
            description="Filtra por status exacto del job.",
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=200,
            description="Maximo de items (default 50, cap 200).",
        ),
    ] = 50,
    offset: Annotated[
        int,
        Query(
            ge=0,
            description="Salta los primeros N items (paginacion forward).",
        ),
    ] = 0,
) -> list[JobSummary]:
    """GET /v1/jobs — lista jobs del user autenticado.

    Orden estable: ORDER BY created_at DESC (índice en DB).
    Filtrado por user_id del token (no leak cross-user).
    Paginación: limit (default 50, max 200) + offset (>= 0).

    200 con [JobSummary] (puede ser lista vacía).
    503 si el service no está inicializado.
    """
    db = service._db
    status_value = status_filter.value if status_filter else None

    if offset > _OFFSET_SLICE_CAP:
        # Defensa anti-paginacion patologica (cliente con bug). 200 vacia
        # mejor que 500 (que es lo que pasaria si dejamos que el DB
        # escanee millones de rows para devolver 0).
        logger.warning(
            "jobs_list_offset_too_large",
            extra={"user_id": user_id, "offset": offset},
        )
        return []

    # Fetch limit+offset rows y slicemos en Python (db.list_research_jobs
    # no soporta offset nativo para no tocar Track 1). Memory cost <= 1200.
    rows = await db.list_research_jobs(user_id=user_id, status=status_value, limit=limit + offset)
    page = rows[offset : offset + limit]

    return [
        JobSummary(
            id=r["id"],
            query=r["query"],
            status=JobStatus(r["status"]),
            current_phase=_safe_phase(r.get("current_phase")),
            progress_percent=r["progress_percent"],
            cost_usd=r["cost_usd"],
            created_at=r["created_at"],
            started_at=r.get("started_at"),
            completed_at=r.get("completed_at"),
        )
        for r in page
    ]


# ============================================================================
# Internal helpers
# ============================================================================


async def _assert_owner(
    job_detail: JobDetail,
    user_id: int,
    service: Any,
) -> None:
    """Verifica que el job pertenece al user_id del token.

    Si no, devuelve 404 (no leak info — no distinguimos "no existe" de
    "no es tuyo"; evita enumeration de IDs cross-user).

    Raises:
        HTTPException 404.
    """
    db = service._db
    row = await db.get_research_job(job_detail.id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "type": "job_not_found",
                    "message": f"Job {job_detail.id} not found.",
                }
            },
        )
    if int(row.get("user_id", 0)) != int(user_id):
        # Same response shape as not-found: no info leak cross-user.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "type": "job_not_found",
                    "message": f"Job {job_detail.id} not found.",
                }
            },
        )


async def _assert_owner_id(
    job_id: str,
    user_id: int,
    service: Any,
) -> None:
    """Como _assert_owner pero para paths donde aún no tenemos el JobDetail.

    Hace solo el SELECT del row (más barato que get_job() que también
    carga token_usage drill-down).
    """
    db = service._db
    row = await db.get_research_job(job_id)
    if row is None or int(row.get("user_id", 0)) != int(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "type": "job_not_found",
                    "message": f"Job {job_id} not found.",
                }
            },
        )


def _safe_phase(value: Any) -> Any:
    """Convierte row['current_phase'] (puede ser None) a PhaseName o None.

    Wraps el lookup para tolerar valores legacy/extraños en DB.
    """
    if value is None:
        return None
    from hermes.jobs.models import PhaseName

    try:
        return PhaseName(value)
    except ValueError:
        logger.warning("jobs_api_unknown_phase", extra={"phase": str(value)})
        return None


def _build_count_jobs_sql(user_id: int, today_start_str: str) -> tuple[str, list]:
    """SQL parametrizado para COUNT(*) de jobs del user desde hoy UTC.

    Inlining: SQLite acepta `?` params; nada de string-format con user_id.
    """
    return (
        "SELECT COUNT(*) FROM research_jobs WHERE user_id = ? AND created_at >= ?",
        [user_id, today_start_str],
    )
