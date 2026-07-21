"""HTTP API router para deep research jobs (Sprint 14, US-2.1).

Ver EPIC2_SYNTHESIS.md §2.4 + TDD_S14_DEEP_RESEARCH.md §10.

Endpoints:
  POST   /v1/jobs                  bearer auth → 201 JobResponse
  GET    /v1/jobs                  bearer auth → 200 [JobSummary]
  GET    /v1/jobs/budget           bearer auth → 200 DailyBudgetStatus
  GET    /v1/jobs/preflight        bearer auth → 200 DeepResearchPreflight
  GET    /v1/jobs/{job_id}         bearer auth → 200 JobDetail | 404
  POST   /v1/jobs/{job_id}/cancel  bearer auth → 200 CancelResponse | 404 | 409
  POST   /v1/jobs/{job_id}/retry   bearer auth → 201 JobResponse | 404 | 409
  GET    /v1/jobs/{job_id}/report  bearer auth → 200 markdown | 404 | 409 | 500

Todos los endpoints requieren bearer auth via `Depends(authenticate_bearer)`.
El service singleton se obtiene via `Depends(get_deep_research_service_dep)`
que 503ea si el service no está inicializado (degraded mode).

Wiring: `create_app()` en hermes/receivers/http_api.py monta este router
y registra el singleton via `set_deep_research_service(service)` ANTES
de aceptar requests. Patron deliberadamente minimal: este modulo NO
inicializa el service (responsabilidad del startup lifecycle).

Slice 1C2: added ``GET /v1/jobs/{job_id}/report``. The read path is
derived from ``settings.deep_research_data_root + job_id`` (NEVER from
the DB ``output_path`` column). The route translates internal
``LocalReportStore`` exceptions into a single 500 ``report_unavailable``
envelope — the public response never exposes filesystem paths, byte
limits, decoder text, raw exceptions, symlink targets, or OS details.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response, status

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


def clear_deep_research_service() -> bool:
    """Clear the singleton. Idempotent lifecycle seam (Slice 1C1c).

    Called from the centralized shutdown helper BEFORE shared DB /
    provider resources are closed so that no stale dependency can
    route a late request into a service that is being torn down.

    Returns:
        True if a singleton was cleared, False if no singleton was set.

    Backward compatible: the existing ``set_deep_research_service``
    setter-based tests continue to work. ``get_deep_research_service_dep``
    keeps its 503 behavior after clearing.
    """
    global _service_singleton
    had_value = _service_singleton is not None
    if had_value:
        _service_singleton = None
        logger.info("jobs_api_service_cleared")
    return had_value


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
                    "message": "Job not found.",
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
            "True (default): bounded-wait for the local asyncio task "
            "to acknowledge cancellation, up to "
            "``deep_research_cancel_wait_s`` seconds, then return. "
            "False: signal the cancellation and return immediately; "
            "the task's finalizer completes the row transition. Both "
            "values request real local cancellation. ``graceful`` "
            "controls bounded waiting only, not cancellation strength. "
            "An already-received provider request may still be "
            "processed or counted by the provider; cancellation does "
            "NOT claim quota reversal, refund, or reversal of billed "
            "tokens."
        ),
    ),
) -> CancelResponse:
    """POST /v1/jobs/{job_id}/cancel — cancela un job (DR-Q1A-PRE1B).

    200 con ``CancelResponse`` describiendo la transición aplicada.
    La respuesta expone únicamente los campos públicos
    (``id``, ``status``, ``graceful``).

    Estados:
      - 200 + status=cancelled: no queda ejecución local activa.
      - 200 + status=cancelling: cancelación solicitada; la tarea
        asyncio está siendo señalada o el timeout de wait expiró.
      - 200 idempotente: cancelar un job ya ``cancelled`` devuelve
        200 con status=cancelled.
      - 404 si no existe o pertenece a otro user.
      - 409 si ya está en ``complete`` o ``failed`` (terminal real).
      - 503 si el service no está inicializado.

    Contrato:
      Cuando el owner cancela un job de Deep Research, Oroimen deja
      de ejecutar ese job localmente. La cancelación se propaga por
      la coroutine del cliente (search, fetch, LLM) mediante
      ``asyncio``; un request ya recibido por el provider puede
      seguir siendo procesado o contado.
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
                    "message": "Job not found.",
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
                    "message": "Job not found.",
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


@router.get("/jobs/{job_id}/report", status_code=status.HTTP_200_OK)
async def get_job_report(
    job_id: Annotated[str, Path(min_length=12, max_length=12, pattern=r"^[0-9a-f]{12}$")],
    user_id: Annotated[int, Depends(authenticate_bearer)],
    service: Annotated[Any, Depends(get_deep_research_service_dep)],
) -> Response:
    """GET /v1/jobs/{job_id}/report — return the final markdown report.

    Slice 1C2 contract (owner-adjudicated):

    - 200 OK with markdown body, headers:
        Content-Type: text/markdown; charset=utf-8
        Content-Disposition: inline; filename="research-{job_id}.md"
        Cache-Control: private, no-store
        X-Content-Type-Options: nosniff
    - 401 — missing or invalid bearer token (existing).
    - 404 ``job_not_found`` — owner check fails (missing OR foreign-owned,
      byte-identical body).
    - 409 ``report_not_ready`` — owner job in {pending, running, cancelling}.
    - 409 ``report_unavailable`` — owner job in {failed, cancelled} (no
      final report).
    - 500 ``report_unavailable`` — owner job in {complete} but the file is
      missing, escaped, symlink-denied, oversize, invalid UTF-8, or
      otherwise unreadable.
    - 503 — service singleton not initialized (existing).

    The read path is derived from ``settings.deep_research_data_root +
    job_id`` (NEVER from the DB ``output_path`` column). The internal
    column stays as a completion/recovery marker only.
    """
    # (1) Ownership check first — same 404 shape as missing/foreign.
    try:
        await _assert_owner_id(job_id, user_id, service)
    except HTTPException:
        # Re-raise as-is (the 404 shape is already correct).
        raise

    # (2) Look up the job row to learn the status. Same path as
    # get_job_detail; we re-fetch instead of calling get_job() to avoid
    # loading the token_usage drill-down we do not need.
    db = service._db
    row = await db.get_research_job(job_id)
    if row is None:
        # Should not happen — _assert_owner_id already confirmed the
        # row exists and is owned. But defense in depth: if a race
        # deletes the row between the owner check and the status read,
        # the route should still return 404 (same shape).
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "type": "job_not_found",
                    "message": "Job not found.",
                }
            },
        )
    job_status = JobStatus(row["status"])

    # (3) Status-driven dispatch. Status is the source of truth — the
    # FILE EXISTING ON DISK does NOT authorize a 200 if the DB says
    # the job is still pending/running/cancelling.
    if job_status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.CANCELLING):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "type": "report_not_ready",
                    "message": "Report is not ready yet.",
                }
            },
        )
    if job_status in (JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "type": "report_unavailable",
                    "message": "Report is not available for this job.",
                }
            },
        )
    # job_status == COMPLETE → attempt the read. Any internal error
    # (missing, escaped, symlink, oversize, invalid UTF-8) is mapped
    # to 500 ``report_unavailable`` by the translation block below.

    # (4) Resolve the report store from the wired service. The store
    # is REQUIRED: production composition is fail-closed and never
    # publishes a service without a real LocalReportStore. The
    # ``None`` branch is a defensive guard for tests or future
    # refactors that might bypass the composition root; in that
    # case we return the same 503 contract that
    # ``get_deep_research_service_dep`` returns for an uninitialized
    # singleton. We do NOT return 500 ``report_unavailable`` here
    # because the route is the read path, and an uninitialized
    # reader is a service-level concern, not a per-job concern.
    report_store = getattr(service, "_report_store", None)
    if report_store is None:
        logger.warning(
            "report_unavailable_no_store",
            extra={"job_id": job_id},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "type": "service_unavailable",
                    "message": (
                        "DeepResearchService has no report reader wired. "
                        "The composition root is responsible for the "
                        "fail-closed invariant; this is a misconfiguration."
                    ),
                }
            },
        )

    # (5) Run the read in a thread so the event loop is not blocked.
    # The store is sync and bounded by max_bytes (default 5 MiB).
    try:
        markdown_text = await asyncio.to_thread(report_store.read, job_id)
    except FileNotFoundError:
        logger.warning(
            "report_missing",
            extra={"job_id": job_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "type": "report_unavailable",
                    "message": "Report is not available for this job.",
                }
            },
        ) from None
    except Exception as exc:
        # Map ALL report-store errors to 500 ``report_unavailable``.
        # We log the internal category for ops/observability but never
        # include the type text in the response body.
        from hermes.jobs.exceptions import (
            InvalidJobIdError,
            InvalidUTF8Error,
            PathEscapeError,
            ReportTooLargeError,
            SymlinkEscapeError,
        )

        if isinstance(exc, InvalidJobIdError):
            category = "report_invalid_job_id"
        elif isinstance(exc, PathEscapeError):
            category = "report_path_escape"
        elif isinstance(exc, SymlinkEscapeError):
            category = "report_symlink_denied"
        elif isinstance(exc, ReportTooLargeError):
            category = "report_size_limit_exceeded"
        elif isinstance(exc, InvalidUTF8Error):
            category = "report_invalid_utf8"
        else:
            category = "report_read_failed"
        logger.warning(
            category,
            extra={"job_id": job_id, "error": exc.__class__.__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "type": "report_unavailable",
                    "message": "Report is not available for this job.",
                }
            },
        ) from exc

    # (6) Successful read. Encode the markdown as UTF-8 bytes and
    # return with the documented headers. The filename is
    # ``research-{job_id}.md`` — exactly reproducible from the URL
    # parameter, no path or extension disclosure.
    markdown_bytes = markdown_text.encode("utf-8")
    return Response(
        content=markdown_bytes,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Type": "text/markdown; charset=utf-8",
            "Content-Disposition": f'inline; filename="research-{job_id}.md"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


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
                    "message": "Job not found.",
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
                    "message": "Job not found.",
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
                    "message": "Job not found.",
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
