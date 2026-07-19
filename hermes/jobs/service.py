"""DeepResearchService: 5-phase pipeline async, state machine, recovery.

Componente principal de Sprint 14. Ver TDD_S14_DEEP_RESEARCH.md §6.

Pipeline:
  Phase 1: search        — Tavily web search via hermes_search(intent='deep_research')
  Phase 2: scrape        — HTTP fetch + selectolax HTML-to-text per URL
  Phase 3: per_source_synthesis — 1 LLM call per source (5 calls en paralelo)
  Phase 4: final_synthesis      — 1 LLM call con todos los summaries
  Phase 5: write         — atomic write data/jobs/{id}.md

Cost tracking:
  - PRICING_TABLE en cost.py
  - calculate_cost() con Decimal + quantize a 4 decimales
  - _record_token_usage: checkpoint PRIMERO → DB después (anti-drift)
  - reconcile_cost(): max(checkpoint, token_usage_sum, aggregate)

Budget:
  - _check_daily_budget() (TDD §8.2): pre-check en submit
  - _check_per_job_budget() (TDD §8.3): soft alert per job

Observability:
  - write_research_metric() en cada evento (job_created, phase_completed,
    job_completed, budget_drift, threadpool_saturation)

ContextVar pattern (TDD §1.5.2): cada LLM call envuelve
`llm.chat()` con `job_id_var.set()` + `try/finally: var.reset()`.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from hermes.jobs.cost import (
    PRICING_TABLE,
    calculate_cost,
    estimate_research_cost,
    format_now,
)
from hermes.jobs.exceptions import (
    BudgetExceededError,
    JobAlreadyTerminalError,
    JobNotFoundError,
    JobNotRetryableError,
    PhaseError,
    SchedulerUnavailableError,
)
from hermes.jobs.models import (
    CancelResponse,
    CreateJobRequest,
    ErrorTaxonomy,
    JobDetail,
    JobResponse,
    JobStatus,
    JobSummary,
    JobType,
    PhaseName,
    TokenUsageEntry,
)
from hermes.jobs.prompts import (
    FINAL_SYNTH_PROMPT,
    PER_SOURCE_PROMPT,
    sanitize_summary,
)
from hermes.observability.influxdb import write_research_metric

logger = logging.getLogger(__name__)


# Retryable errors (TDD §6.7). 4xx / cancelled / budget / oom NO retry.
RETRYABLE_ERRORS = frozenset(
    {
        "search_5xx",  # Tavily transient
        "llm_5xx",  # LLM provider transient
        "timeout",  # asyncio.wait_for agotó
        "network",  # DNS, connection refused
        # NO retryable (explícito):
        # 'search_4xx' (API key, quota — manual fix needed)
        # 'llm_4xx' (content policy, context length — won't fix itself)
        # 'cancelled' (user action)
        # 'budget_exceeded' (need user action o esperar al día)
        # 'oom' (NAS host saturado — retry solo empeora)
        # 'checkpoint_corrupt' (data corruption — re-init manual)
    }
)

# Backoff schedule (seconds): 3 attempts total.
_RETRY_BACKOFF_SCHEDULE = (1, 4, 16)

# HTML size guard (TDD §6.3): ANTES de to_thread truncar a 2MB.
_HTML_SIZE_GUARD_BYTES = 2_000_000

# Default model for LLM phases (TDD §6.4 §6.5).
_DEFAULT_LLM_MODEL = "MiniMax-M3"


# ContextVar pattern (TDD §1.5.2): cada LLM call envuelve
# `llm.chat()` con `job_id_var.set()` + `try/finally: var.reset()`.
# Esto permite que un middleware del LLMRouter pueda inyectar cost
# tracking con el job_id correcto sin acoplar el router a este módulo.
job_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("research_job_id", default="")
phase_var: contextvars.ContextVar[str] = contextvars.ContextVar("research_phase", default="")


def html_to_text_selectolax(html: str) -> str:
    """HTML → texto plano usando selectolax (C-extension, memory-safe).

    Args:
        html: HTML crudo (potencialmente >2MB — caller aplica Size Guard).

    Returns:
        texto plano con whitespace normalizado.

    Note:
        Parser robusto contra HTML malformado. NO usar html2text
        (Python regex, vulnerable a ReDoS en HTML patológico — TDD §6.3.1).
    """
    try:
        from selectolax.parser import HTMLParser
    except ImportError:
        # Fallback defensivo si selectolax no está instalado (CI/dev).
        # No debería pasar en producción (requirements.txt lo pinne).
        import re

        return re.sub(r"<[^>]+>", " ", html)

    tree = HTMLParser(html)
    # Extraer texto de body o root
    body = tree.body
    if body is None:
        text = tree.text(separator="\n", strip=True)
    else:
        text = body.text(separator="\n", strip=True)
    # Normalizar whitespace
    import re as _re

    return _re.sub(r"\n{3,}", "\n\n", text).strip()


class DeepResearchService:
    """Service de investigación profunda. Sprint 14, ÉPICA 2.

    Inyectado con db, notifier, llm_router, web_search, settings, scheduler.

    Uso:
        service = DeepResearchService(
            db=db, notifier=notifier, llm_router=llm,
            web_search=search_fn, settings=settings, scheduler=scheduler,
        )
        response = await service.submit_job(CreateJobRequest(query="..."), user_id=0)
    """

    def __init__(
        self,
        *,
        db: Any,
        notifier: Any,
        llm_router: Any,
        web_search: Any,
        settings: Any,
        scheduler: Any,
    ) -> None:
        self._db = db
        self._notifier = notifier
        self._llm = llm_router
        self._search = web_search
        self._settings = settings
        self._scheduler = scheduler

        # ThreadPoolExecutor custom para HTML parsing (TDD §6.3.1).
        # 4 workers (NAS host 2 vCPU: 4 threads = uso racional).
        # thread_name_prefix="scrape" para distinguir en logs/metrics.
        self._scrape_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="scrape",
        )
        # NB1 verifier finding: counter explícito de workers activos.
        # Reemplaza el acceso a self._scrape_pool._idle_semaphore._value
        # (estado interno de CPython, frágil entre versiones). Se incrementa
        # en _run_in_scrape_pool() y se decrementa en finally. NO usa
        # threading.enumerate() porque enumera también el thread principal
        # y los de pytest/pytest-asyncio en tests → ruidoso.
        self._scrape_active = 0

        # Path raíz para outputs. Default: data/jobs/ relative a cwd.
        # Tests inyectan via settings.output_dir si necesitan tmp_path.
        self._data_root = Path(getattr(settings, "deep_research_data_root", None) or "data/jobs")

    async def _run_in_scrape_pool(self, fn: Any, *args: Any) -> Any:
        """Ejecuta ``fn`` en el threadpool ``_scrape_pool`` con counter explícito.

        Wrapper sobre ``loop.run_in_executor`` que mantiene
        ``self._scrape_active`` sincronizado. Usado por Phase 2 (HTML parsing)
        para que la métrica de saturación refleje workers realmente ocupados
        en vez de inspeccionar ``_idle_semaphore._value`` (interno de CPython).

        El counter se incrementa ANTES de submit (evita race: si el thread
        arranca y decrementa antes de que incrementemos, veríamos negativo)
        y se decrementa en finally (cubre excepciones y cancels).
        """
        self._scrape_active += 1
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._scrape_pool, fn, *args)
        finally:
            self._scrape_active -= 1

    # =====================================================================
    # Public API (HTTP-facing, retorna modelos Pydantic)
    # =====================================================================

    async def submit_job(
        self,
        request: CreateJobRequest,
        user_id: int = 0,
    ) -> JobResponse:
        """Crea job, valida daily budget, enqueue en AsyncIOScheduler. <100ms.

        Raises:
            BudgetExceededError: si daily cap reached.
            SchedulerUnavailableError: si el scheduler no está inicializado.
        """
        import uuid

        # Pre-check 1 (TDD §10.2): budget rápido para UX (fail-fast con 429).
        # Check 2 (atómico en _run_research) captura el TOCTOU race.
        can_submit, remaining = await self._check_daily_budget(user_id=user_id)
        if not can_submit:
            raise BudgetExceededError(
                f"Daily budget exceeded. Remaining: ${float(remaining):.4f}. "
                "Retry tomorrow or raise cap via HERMES_DEEP_RESEARCH_DAILY_BUDGET_USD."
            )

        # UUID 12-char hex (TDD §1.5): uuid4().hex[:12]
        job_id = uuid.uuid4().hex[:12]

        notify_int = 1 if request.notify_via_tg else 0
        await self._db.create_research_job(
            job_id=job_id,
            query=request.query,
            notify_via_tg=notify_int,
            job_type=request.job_type.value,
            user_id=user_id,
        )

        # Estimación heurística (Q3 verifier finding): ahora computada de
        # settings reales en lugar de Decimal("0.05") hardcoded. Si usuario
        # sube max_sources o output_max_tokens, la estimación escala.
        # Ver cost.estimate_research_cost() para la fórmula.
        estimated_cost = estimate_research_cost(
            max_sources=int(getattr(self._settings, "deep_research_max_sources", 5)),
            per_source_max_tokens=int(
                getattr(self._settings, "deep_research_per_source_max_tokens", 3000)
            ),
            output_max_tokens=int(
                getattr(self._settings, "deep_research_output_max_tokens", 10000)
            ),
            pricing_table=PRICING_TABLE,
            primary_model=_DEFAULT_LLM_MODEL,
        )

        # Enqueue en el scheduler (TDD §4.2).
        if self._scheduler is None:
            raise SchedulerUnavailableError("Scheduler not initialized")
        try:
            await self._scheduler.enqueue(job_id, run_date=self._now_dt())
        except Exception as exc:
            logger.exception("submit_job_enqueue_failed", extra={"job_id": job_id})
            raise SchedulerUnavailableError(f"Enqueue failed: {exc}") from exc

        # Observability event
        write_research_metric(
            "research_job_created",
            tags={
                "job_type": request.job_type.value,
                "notify_via_tg": str(bool(request.notify_via_tg)),
            },
            fields={"count": 1},
        )

        logger.info(
            "research_job_created",
            extra={
                "job_id": job_id,
                "user_id": user_id,
                "query_length": len(request.query),
                "job_type": request.job_type.value,
                "notify_via_tg": bool(request.notify_via_tg),
                "estimated_cost_usd": float(estimated_cost),
            },
        )

        created_row = await self._db.get_research_job(job_id)
        return JobResponse(
            id=job_id,
            status=JobStatus.PENDING,
            created_at=created_row["created_at"] if created_row else format_now(),
            estimated_cost_usd=float(estimated_cost),
        )

    async def get_job(self, job_id: str) -> JobDetail:
        """Lee job + token usage. Single DB query + join.

        Raises:
            JobNotFoundError: si id no existe.
        """
        job_row = await self._db.get_research_job(job_id)
        if job_row is None:
            raise JobNotFoundError(f"Job {job_id} not found")

        # Drill-down token_usage
        token_rows = await self._db.list_token_usage_for_job(job_id)
        token_usage = [
            TokenUsageEntry(
                phase=PhaseName(r["phase"])
                if r["phase"] in {p.value for p in PhaseName}
                else PhaseName.SEARCH,
                model=r["model"],
                tokens_in=r["tokens_in"],
                tokens_out=r["tokens_out"],
                cost_usd=r["cost_usd"],
                created_at=r["created_at"],
            )
            for r in token_rows
        ]

        # checkpoint_path si existe
        ckpt_path = self._data_root / job_id / "checkpoint.json"
        checkpoint_path = str(ckpt_path) if ckpt_path.exists() else None

        return JobDetail(
            id=job_row["id"],
            query=job_row["query"],
            status=JobStatus(job_row["status"]),
            current_phase=(
                PhaseName(job_row["current_phase"]) if job_row.get("current_phase") else None
            ),
            progress_percent=job_row["progress_percent"],
            cost_usd=job_row["cost_usd"],
            created_at=job_row["created_at"],
            started_at=job_row.get("started_at"),
            completed_at=job_row.get("completed_at"),
            job_type=JobType(job_row.get("job_type", "deep_research")),
            notify_via_tg=bool(job_row.get("notify_via_tg", 1)),
            output_path=job_row.get("output_path"),
            partial_output_path=job_row.get("partial_output_path"),
            error_taxonomy=(
                ErrorTaxonomy(job_row["error_taxonomy"]) if job_row.get("error_taxonomy") else None
            ),
            error_message=job_row.get("error_message"),
            tokens_in=job_row["tokens_in"],
            tokens_out=job_row["tokens_out"],
            notified=bool(job_row.get("notified", 0)),
            updated_at=job_row["updated_at"],
            token_usage=token_usage,
            checkpoint_path=checkpoint_path,
        )

    async def list_jobs(
        self,
        user_id: int = 0,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobSummary]:
        """Lista jobs del user, ordenado por created_at DESC."""
        status_value = status.value if status else None
        rows = await self._db.list_research_jobs(user_id=user_id, status=status_value, limit=limit)
        return [
            JobSummary(
                id=r["id"],
                query=r["query"],
                status=JobStatus(r["status"]),
                current_phase=(PhaseName(r["current_phase"]) if r.get("current_phase") else None),
                progress_percent=r["progress_percent"],
                cost_usd=r["cost_usd"],
                created_at=r["created_at"],
                started_at=r.get("started_at"),
                completed_at=r.get("completed_at"),
            )
            for r in rows
        ]

    async def cancel_job(self, job_id: str, graceful: bool = True) -> CancelResponse:
        """Marca cancelling/cancelled.

        Si graceful, await current phase finish (max 10s). Si no, hard cancel.
        Raises:
            JobNotFoundError: si id no existe.
            JobAlreadyTerminalError: si ya está en complete/failed/cancelled.
        """
        job_row = await self._db.get_research_job(job_id)
        if job_row is None:
            raise JobNotFoundError(f"Job {job_id} not found")

        current_status = JobStatus(job_row["status"])
        if current_status in (
            JobStatus.COMPLETE,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            raise JobAlreadyTerminalError(current_status)

        # Marcar cancelling inmediatamente
        await self._db.update_research_job_status(job_id, "cancelling")

        if not graceful:
            # Hard cancel: marcar cancelled ahora
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

        # Graceful: la próxima vez que _run_research poll el status, verá
        # 'cancelling' y marcará cancelled. Para el response, devolvemos
        # 'cancelling' (estado transitorio).
        return CancelResponse(
            id=job_id,
            status=JobStatus.CANCELLING,
            graceful=True,
            partial_output_path=None,
        )

    async def retry_job(self, job_id: str, user_id: int = 0) -> JobResponse:
        """Crea nuevo job copiando checkpoint del original. NO re-scrape.

        Raises:
            JobNotFoundError: si original no existe.
            JobNotRetryableError: si original NO está en 'failed'.
        """
        import uuid

        original = await self._db.get_research_job(job_id)
        if original is None:
            raise JobNotFoundError(f"Job {job_id} not found")
        if original["status"] != "failed":
            raise JobNotRetryableError(JobStatus(original["status"]))

        # Nuevo job con el mismo query, copia checkpoint si existe
        new_job_id = uuid.uuid4().hex[:12]
        notify_int = int(original.get("notify_via_tg", 1))
        await self._db.create_research_job(
            job_id=new_job_id,
            query=original["query"],
            notify_via_tg=notify_int,
            job_type=original.get("job_type", "deep_research"),
            user_id=user_id,
        )

        # Encolar
        if self._scheduler is None:
            raise SchedulerUnavailableError("Scheduler not initialized")
        try:
            await self._scheduler.enqueue(new_job_id, run_date=self._now_dt())
        except Exception as exc:
            raise SchedulerUnavailableError(f"Enqueue failed: {exc}") from exc

        created_row = await self._db.get_research_job(new_job_id)
        # Estimación heurística dinámica (Q3 verifier finding): misma fórmula
        # que submit_job — escala con max_sources/output_max_tokens de settings.
        estimated_cost = estimate_research_cost(
            max_sources=int(getattr(self._settings, "deep_research_max_sources", 5)),
            per_source_max_tokens=int(
                getattr(self._settings, "deep_research_per_source_max_tokens", 3000)
            ),
            output_max_tokens=int(
                getattr(self._settings, "deep_research_output_max_tokens", 10000)
            ),
            pricing_table=PRICING_TABLE,
            primary_model=_DEFAULT_LLM_MODEL,
        )
        return JobResponse(
            id=new_job_id,
            status=JobStatus.PENDING,
            created_at=created_row["created_at"] if created_row else format_now(),
            estimated_cost_usd=float(estimated_cost),
        )

    # =====================================================================
    # Internal: main run loop
    # =====================================================================

    async def _run_research(self, job_id: str) -> None:
        """Main loop. Llamado por AsyncIOScheduler.

        Idempotente: si status != pending, no-op (recovery hook ya maneja
        transiciones; este método asume 'pending' como estado de entrada).

        Transiciones:
          pending → running → (complete | failed | cancelled)
        """
        start_time = time.monotonic()

        # TOCTOU check 2 atómico (TDD §10.2): verificar budget DENTRO de lock.
        # Si 2 jobs queued y el primero agotó el budget, este falla limpio.
        try:
            await self._db.conn.execute("BEGIN IMMEDIATE")
            async with self._db.conn.execute(
                "SELECT status FROM research_jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                await self._db.conn.execute("COMMIT")
                return
            status_now = row["status"] if isinstance(row, dict) else row[0]
            if status_now != "pending":
                await self._db.conn.execute("COMMIT")
                logger.info(
                    "run_research_skip_non_pending",
                    extra={"job_id": job_id, "status": status_now},
                )
                return
            # Check 2 budget
            today_cost = await self._db.get_today_research_cost(user_id=0)
            cap = Decimal(str(getattr(self._settings, "deep_research_daily_budget_usd", 3.0)))
            # Estimación dinámica (Q3 verifier finding): si el user subió
            # max_sources/output_max_tokens, este gate budget-aware escala en
            # lugar de asumir $0.05 fijo.
            estimated = estimate_research_cost(
                max_sources=int(getattr(self._settings, "deep_research_max_sources", 5)),
                per_source_max_tokens=int(
                    getattr(self._settings, "deep_research_per_source_max_tokens", 3000)
                ),
                output_max_tokens=int(
                    getattr(self._settings, "deep_research_output_max_tokens", 10000)
                ),
                pricing_table=PRICING_TABLE,
                primary_model=_DEFAULT_LLM_MODEL,
            )
            if Decimal(str(today_cost)) + estimated > cap:
                # Budget exhausted entre submit y run
                await self._db.conn.execute(
                    "UPDATE research_jobs SET status='failed', "
                    "error_taxonomy='budget_exceeded', "
                    "error_message='budget_exhausted_in_queue', "
                    "completed_at=?, updated_at=? WHERE id=?",
                    (format_now(), format_now(), job_id),
                )
                await self._db.conn.execute("COMMIT")
                logger.warning(
                    "research_budget_exhausted_in_queue",
                    extra={"job_id": job_id},
                )
                return
            # OK, transicionar a running
            now = format_now()
            await self._db.conn.execute(
                "UPDATE research_jobs SET status='running', started_at=?, updated_at=? "
                "WHERE id = ?",
                (now, now, job_id),
            )
            await self._db.conn.execute("COMMIT")
        except Exception:
            with contextlib.suppress(Exception):
                await self._db.conn.execute("ROLLBACK")
            logger.exception("run_research_lock_error", extra={"job_id": job_id})
            return

        # === Ejecutar las 5 phases ===
        try:
            # Phase 1
            await self._update_phase(job_id, PhaseName.SEARCH, progress=10)
            urls = await self._run_phase_with_retry(
                job_id,
                PhaseName.SEARCH,
                lambda: self._phase_search(job_id),
            )
            await self._write_checkpoint_phase(job_id, PhaseName.SEARCH, {"urls": urls})
            await self._update_phase(job_id, PhaseName.SEARCH, progress=20)

            # Phase 2
            await self._update_phase(job_id, PhaseName.SCRAPE, progress=25)
            sources = await self._run_phase_with_retry(
                job_id,
                PhaseName.SCRAPE,
                lambda: self._phase_scrape(job_id, urls),
            )
            await self._write_checkpoint_phase(
                job_id, PhaseName.SCRAPE, {"sources": self._sources_summary(sources)}
            )
            await self._update_phase(job_id, PhaseName.SCRAPE, progress=50)

            # Phase 3
            await self._update_phase(job_id, PhaseName.PER_SOURCE_SYNTHESIS, progress=55)
            summaries = await self._run_phase_with_retry(
                job_id,
                PhaseName.PER_SOURCE_SYNTHESIS,
                lambda: self._phase_per_source_synthesis(job_id, sources),
            )
            await self._write_checkpoint_phase(
                job_id,
                PhaseName.PER_SOURCE_SYNTHESIS,
                {"summaries": summaries},
            )
            await self._update_phase(job_id, PhaseName.PER_SOURCE_SYNTHESIS, progress=75)

            # Phase 4
            await self._update_phase(job_id, PhaseName.FINAL_SYNTHESIS, progress=80)
            report = await self._run_phase_with_retry(
                job_id,
                PhaseName.FINAL_SYNTHESIS,
                lambda: self._phase_final_synthesis(job_id, summaries),
            )
            # Sanitize final output (defense in depth)
            report = sanitize_summary(report)
            await self._write_checkpoint_phase(
                job_id, PhaseName.FINAL_SYNTHESIS, {"report": report}
            )
            await self._update_phase(job_id, PhaseName.FINAL_SYNTHESIS, progress=90)

            # Phase 5
            await self._update_phase(job_id, PhaseName.WRITE, progress=95)
            await self._phase_write(job_id, report)

            duration = time.monotonic() - start_time
            total_cost = await self._db.get_research_job_cost(job_id)
            logger.info(
                "research_job_completed",
                extra={
                    "job_id": job_id,
                    "total_duration_s": duration,
                    "total_cost_usd": total_cost,
                },
            )
            write_research_metric(
                "research_job_completed",
                tags={
                    "status": "complete",
                    "job_type": "deep_research",
                    "error_taxonomy": "none",
                },
                fields={
                    "count": 1,
                    "total_duration_s": duration,
                    "total_cost_usd": total_cost,
                    "phases_completed": 5,
                },
            )
        except PhaseError as phase_err:
            duration = time.monotonic() - start_time
            total_cost = await self._db.get_research_job_cost(job_id)
            await self._db.update_research_job_status(
                job_id,
                "failed",
                error_taxonomy=phase_err.taxonomy,
                error_message=phase_err.message[:500],
                completed_at=format_now(),
            )
            logger.error(
                "research_job_failed",
                extra={
                    "job_id": job_id,
                    "error_taxonomy": phase_err.taxonomy,
                    "error_message": phase_err.message,
                    "total_cost_usd": total_cost,
                    "total_duration_s": duration,
                },
            )
            write_research_metric(
                "research_job_completed",
                tags={
                    "status": "failed",
                    "job_type": "deep_research",
                    "error_taxonomy": phase_err.taxonomy,
                },
                fields={
                    "count": 1,
                    "total_duration_s": duration,
                    "total_cost_usd": total_cost,
                },
            )
            # Notifier: failed
            if hasattr(self._notifier, "send_research_failed"):
                try:
                    await self._notifier.send_research_failed(
                        job_id=job_id,
                        error_taxonomy=phase_err.taxonomy,
                        error_message=phase_err.message,
                        retryable=phase_err.retryable,
                    )
                    await self._db.mark_research_job_notified(job_id)
                except Exception:
                    logger.exception("research_notif_failed", extra={"job_id": job_id})
        except Exception as exc:
            duration = time.monotonic() - start_time
            total_cost = await self._db.get_research_job_cost(job_id)
            await self._db.update_research_job_status(
                job_id,
                "failed",
                error_taxonomy="network",
                error_message=f"unhandled:{exc!s}"[:500],
                completed_at=format_now(),
            )
            logger.exception(
                "research_job_unhandled_error",
                extra={"job_id": job_id, "total_cost_usd": total_cost},
            )

    # =====================================================================
    # Internal: 5 phases
    # =====================================================================

    async def _phase_search(self, job_id: str) -> list[str]:
        """Phase 1: web search via hermes_search(intent='deep_research')."""
        query = await self._db.get_research_job_query(job_id)
        if not query:
            raise PhaseError("search_5xx", "job_query_missing", retryable=False)
        timeout = int(getattr(self._settings, "deep_research_phase1_timeout_s", 30))
        max_sources = int(getattr(self._settings, "deep_research_max_sources", 5))
        try:
            # hermes_search returns SearchResult (dataclass).
            # Si el caller mockea `web_search`, podría devolver ya una lista
            # de URLs (test convenience). Aceptamos ambas formas.
            result = await asyncio.wait_for(
                self._search(
                    query=query,
                    intent="deep_research",
                    content="snippet",
                    num_results=max_sources,
                ),
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise PhaseError("timeout", "search_timeout", retryable=True) from exc
        except Exception as exc:
            err_msg = str(exc).lower()
            if "401" in err_msg or "403" in err_msg or "api key" in err_msg:
                raise PhaseError("search_4xx", f"search_auth:{exc!s}", retryable=False) from exc
            raise PhaseError("search_5xx", f"search_error:{exc!s}", retryable=True) from exc

        # Extraer URLs del SearchResult (duck-typed) o si es ya list[str].
        urls: list[str] = []
        if isinstance(result, list):
            urls = [str(u) for u in result]
        elif hasattr(result, "results"):
            urls = [r["url"] for r in result.results if r.get("url")]
        else:
            raise PhaseError(
                "search_5xx",
                f"unexpected_search_result:{type(result).__name__}",
                retryable=True,
            )

        if not urls:
            raise PhaseError("search_5xx", "no_results", retryable=True)
        return urls[:max_sources]

    async def _phase_scrape(self, job_id: str, urls: list[str]) -> list[dict]:
        """Phase 2: HTTP fetch + selectolax html_to_text. Size Guard 2MB.

        Para cada URL:
          1. fetch raw con httpx (timeout 30s)
          2. Size Guard: si raw > 2MB → truncate ANTES de to_thread
          3. html_to_text via custom ThreadPoolExecutor (4 workers)
          4. si clean_text < 100 chars → mark success=False, error='too_short'

        Output: list of dicts con keys: url, success, clean_text?, error?
        """
        import httpx

        timeout = int(getattr(self._settings, "deep_research_phase2_timeout_s", 30))

        # Observability: threadpool saturation al inicio de phase 2
        # NB1 verifier finding: usa el counter explícito ``self._scrape_active``
        # en vez de inspeccionar ``self._scrape_pool._idle_semaphore._value``
        # (estado interno de CPython, frágil entre versiones). Al inicio de
        # phase 2 el counter es 0 (no hay fetches corriendo todavía); la
        # observación útil sucede durante el gather() — ver _run_in_scrape_pool.
        max_workers = self._scrape_pool._max_workers
        active = self._scrape_active
        saturation_pct = (active / max(max_workers, 1)) * 100.0
        write_research_metric(
            "research_threadpool_saturation",
            tags={"pool_name": "scrape"},
            fields={
                "active_threads": float(active),
                "max_workers": float(max_workers),
                "saturation_pct": saturation_pct,
            },
        )

        async def fetch_one(url: str) -> dict:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                raw = resp.text
                # Size Guard 2MB ANTES de to_thread (P0-1 v1.3 Gemini)
                if len(raw) > _HTML_SIZE_GUARD_BYTES:
                    raw = raw[:_HTML_SIZE_GUARD_BYTES]
                # HTML parse en thread pool dedicado (no default executor).
                # NB1: usamos _run_in_scrape_pool para mantener
                # self._scrape_active sincronizado (saturación métrica).
                clean = await self._run_in_scrape_pool(html_to_text_selectolax, raw)
                if len(clean) < 100:
                    return {"url": url, "success": False, "error": "too_short"}
                return {"url": url, "success": True, "clean_text": clean}
            except TimeoutError:
                return {"url": url, "success": False, "error": "timeout"}
            except httpx.HTTPStatusError as exc:
                return {
                    "url": url,
                    "success": False,
                    "error": f"http_{exc.response.status_code}",
                }
            except Exception as exc:
                return {"url": url, "success": False, "error": f"parse_error:{exc!s}"}

        results = await asyncio.gather(*[fetch_one(u) for u in urls], return_exceptions=False)
        return results

    async def _phase_per_source_synthesis(self, job_id: str, sources: list[dict]) -> list[str]:
        """Phase 3: 1 LLM call por source (success=True).

        ContextVar pattern (TDD §1.5.2): cada synth_one envuelve
        `llm.chat()` con job_id_var.set() + try/finally: var.reset().
        """
        valid = [s for s in sources if s.get("success")]
        if not valid:
            raise PhaseError("llm_5xx", "no_valid_sources", retryable=False)

        query = await self._db.get_research_job_query(job_id) or ""

        async def synth_one(source: dict) -> str:
            t_jid = job_id_var.set(job_id)
            t_phase = phase_var.set(PhaseName.PER_SOURCE_SYNTHESIS.value)
            try:
                prompt = PER_SOURCE_PROMPT.substitute(
                    query=query,
                    url=source.get("url", ""),
                    source=source.get("clean_text", "")[:50000],
                )
                # LLMRouter.chat(): usa chain_override=[model] para forzar
                # un único modelo en lugar del text_chain.
                try:
                    response = await asyncio.wait_for(
                        self._llm.chat(
                            messages=[{"role": "user", "content": prompt}],
                            chain_override=[_DEFAULT_LLM_MODEL],
                            max_tokens=int(
                                getattr(
                                    self._settings,
                                    "deep_research_per_source_max_tokens",
                                    3000,
                                )
                            ),
                        ),
                        timeout=int(
                            getattr(
                                self._settings,
                                "deep_research_phase3_timeout_s",
                                90,
                            )
                        ),
                    )
                except TimeoutError:
                    return "SOURCE_FAILED: timeout"
                except Exception as exc:
                    logger.warning(
                        "per_source_synth_failed",
                        extra={"job_id": job_id, "url": source.get("url"), "error": str(exc)},
                    )
                    return f"SOURCE_FAILED: {exc!s}"

                # Sanitize output (defense in depth)
                clean_text = sanitize_summary(response.content)

                # Token usage + cost tracking
                cost = calculate_cost(
                    _DEFAULT_LLM_MODEL,
                    response.tokens_in,
                    response.tokens_out,
                )
                # Checkpoint PRIMERO (anti-drift), DB después
                await self._update_checkpoint_cost(
                    job_id, cost, response.tokens_in, response.tokens_out
                )
                await self._record_token_usage(
                    job_id,
                    PhaseName.PER_SOURCE_SYNTHESIS,
                    _DEFAULT_LLM_MODEL,
                    response.tokens_in,
                    response.tokens_out,
                    cost,
                )
                # Metric
                write_research_metric(
                    "research_phase_completed",
                    tags={
                        "phase": PhaseName.PER_SOURCE_SYNTHESIS.value,
                        "model": _DEFAULT_LLM_MODEL,
                    },
                    fields={
                        "count": 1,
                        "duration_s": float(response.latency_ms) / 1000.0,
                        "tokens_in": response.tokens_in,
                        "tokens_out": response.tokens_out,
                        "cost_usd": float(cost),
                    },
                )
                return clean_text
            finally:
                job_id_var.reset(t_jid)
                phase_var.reset(t_phase)

        return await asyncio.gather(*[synth_one(s) for s in valid])

    async def _phase_final_synthesis(self, job_id: str, summaries: list[str]) -> str:
        """Phase 4: 1 LLM call con summaries concatenadas. Max 10K tokens output."""
        # Filter out SOURCE_FAILED summaries (defense in depth)
        valid = [s for s in summaries if not (isinstance(s, str) and s.startswith("SOURCE_FAILED"))]
        if not valid:
            raise PhaseError("llm_5xx", "no_valid_summaries", retryable=False)

        # Sanitize each summary BEFORE injecting en el prompt
        valid_clean = [sanitize_summary(s) for s in valid]

        query = await self._db.get_research_job_query(job_id) or ""

        t_jid = job_id_var.set(job_id)
        t_phase = phase_var.set(PhaseName.FINAL_SYNTHESIS.value)
        try:
            summaries_block = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(valid_clean))
            prompt = FINAL_SYNTH_PROMPT.substitute(query=query, summaries=summaries_block)
            try:
                response = await asyncio.wait_for(
                    self._llm.chat(
                        messages=[{"role": "user", "content": prompt}],
                        chain_override=[_DEFAULT_LLM_MODEL],
                        max_tokens=int(
                            getattr(
                                self._settings,
                                "deep_research_output_max_tokens",
                                10000,
                            )
                        ),
                    ),
                    timeout=int(
                        getattr(
                            self._settings,
                            "deep_research_phase4_timeout_s",
                            120,
                        )
                    ),
                )
            except TimeoutError as exc:
                raise PhaseError("timeout", "final_synthesis_timeout", retryable=True) from exc
            except Exception as exc:
                raise PhaseError("llm_5xx", f"final_synth_error:{exc!s}", retryable=True) from exc

            cost = calculate_cost(
                _DEFAULT_LLM_MODEL,
                response.tokens_in,
                response.tokens_out,
            )
            await self._update_checkpoint_cost(
                job_id, cost, response.tokens_in, response.tokens_out
            )
            await self._record_token_usage(
                job_id,
                PhaseName.FINAL_SYNTHESIS,
                _DEFAULT_LLM_MODEL,
                response.tokens_in,
                response.tokens_out,
                cost,
            )
            write_research_metric(
                "research_phase_completed",
                tags={
                    "phase": PhaseName.FINAL_SYNTHESIS.value,
                    "model": _DEFAULT_LLM_MODEL,
                },
                fields={
                    "count": 1,
                    "duration_s": float(response.latency_ms) / 1000.0,
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "cost_usd": float(cost),
                },
            )
            return response.content
        finally:
            job_id_var.reset(t_jid)
            phase_var.reset(t_phase)

    async def _phase_write(self, job_id: str, report: str) -> Path:
        """Phase 5: atomic write data/jobs/{id}.md + DB update + notifier.

        Atomic write: tmp + fsync + os.replace (P0 mitigation de critique §4.2).
        """
        final_path = self._data_root / f"{job_id}.md"
        tmp_path = final_path.with_suffix(".md.tmp")

        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(report)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, final_path)
        except OSError as exc:
            raise PhaseError("oom", f"disk_write_failed:{exc!s}", retryable=False) from exc

        # Reconciliación (TDD §6.8): max(checkpoint, db_sum, aggregate).
        # Si divergen, usa el más alto (asume infra sub-reporta).
        await self.reconcile_cost(job_id)

        # DB update
        cost = await self._db.get_research_job_cost(job_id)
        await self._db.update_research_job_status(
            job_id,
            "complete",
            progress_percent=100,
            output_path=str(final_path),
            completed_at=format_now(),
        )

        # Notifier (best-effort)
        if hasattr(self._notifier, "send_research_complete"):
            job_row = await self._db.get_research_job(job_id)
            notify_via_tg = bool(job_row.get("notify_via_tg", 1)) if job_row else True
            if notify_via_tg:
                try:
                    sent_ok = await self._notifier.send_research_complete(
                        job_id=job_id,
                        output_path=str(final_path),
                        cost_usd=cost,
                    )
                    if sent_ok:
                        await self._db.mark_research_job_notified(job_id)
                except Exception:
                    logger.exception(
                        "research_notif_complete_failed",
                        extra={"job_id": job_id},
                    )

        # Cleanup checkpoint
        ckpt_path = self._data_root / job_id / "checkpoint.json"
        try:
            if ckpt_path.exists():
                ckpt_path.unlink()
        except OSError:
            pass
        # Cleanup tmp dir si quedó vacío
        try:
            job_dir = self._data_root / job_id
            if job_dir.is_dir() and not any(job_dir.iterdir()):
                job_dir.rmdir()
        except OSError:
            pass

        return final_path

    # =====================================================================
    # Internal: retry helper
    # =====================================================================

    async def _run_phase_with_retry(
        self,
        job_id: str,
        phase_name: PhaseName,
        phase_fn,
    ) -> Any:
        """Wrapper: ejecuta phase_fn con retry exp backoff (1s, 4s, 16s), max 3 attempts.

        Backoff: 1s, 4s, 16s. Si el job total excede ~30min, recovery hook lo detecta
        y re-enqueua (con checkpoint de última phase exitosa).
        """
        last_error: PhaseError | None = None
        for attempt in range(3):
            try:
                result = await phase_fn()
                return result
            except PhaseError as exc:
                last_error = exc
                if exc.taxonomy not in RETRYABLE_ERRORS:
                    logger.warning(
                        "phase_non_retryable_error",
                        extra={
                            "job_id": job_id,
                            "phase": phase_name.value,
                            "error_taxonomy": exc.taxonomy,
                        },
                    )
                    raise
                if attempt < 2:
                    backoff = _RETRY_BACKOFF_SCHEDULE[attempt]
                    logger.warning(
                        "phase_retry",
                        extra={
                            "job_id": job_id,
                            "phase": phase_name.value,
                            "attempt": attempt + 1,
                            "error_taxonomy": exc.taxonomy,
                            "next_retry_in_s": backoff,
                        },
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "phase_retries_exhausted",
                        extra={
                            "job_id": job_id,
                            "phase": phase_name.value,
                            "error_taxonomy": exc.taxonomy,
                        },
                    )
                    raise
        # unreachable (raise in loop) pero mypy necesita
        assert last_error is not None
        raise last_error

    # =====================================================================
    # Internal: budget + cost
    # =====================================================================

    async def _check_daily_budget(self, user_id: int = 0) -> tuple[bool, Decimal]:
        """Retorna (can_submit, remaining_usd).

        Pre-check (TDD §8.2): cap rápido para UX (fail-fast 429).
        Check 2 atómico en _run_research captura TOCTOU race.
        """
        today_cost = await self._db.get_today_research_cost(user_id=user_id)
        cap = Decimal(str(getattr(self._settings, "deep_research_daily_budget_usd", 3.0)))
        # Estimación dinámica (Q3 verifier finding): ver cost.estimate_research_cost.
        estimated = estimate_research_cost(
            max_sources=int(getattr(self._settings, "deep_research_max_sources", 5)),
            per_source_max_tokens=int(
                getattr(self._settings, "deep_research_per_source_max_tokens", 3000)
            ),
            output_max_tokens=int(
                getattr(self._settings, "deep_research_output_max_tokens", 10000)
            ),
            pricing_table=PRICING_TABLE,
            primary_model=_DEFAULT_LLM_MODEL,
        )
        remaining = cap - Decimal(str(today_cost))
        can_submit = Decimal(str(today_cost)) + estimated <= cap
        return (can_submit, remaining)

    async def _check_per_job_budget(self, job_id: str, cost_so_far: Decimal) -> None:
        """Si cost_so_far > per_job_budget_usd, emite log warning (no cancela)."""
        cap = Decimal(str(getattr(self._settings, "deep_research_per_job_budget_usd", 5.0)))
        if cost_so_far > cap:
            logger.warning(
                "research_per_job_budget_alert",
                extra={
                    "job_id": job_id,
                    "cost_usd": float(cost_so_far),
                    "cap_usd": float(cap),
                },
            )

    async def _record_token_usage(
        self,
        job_id: str,
        phase: PhaseName,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: Decimal,
    ) -> None:
        """INSERT en research_job_token_usage + UPDATE aggregates en research_jobs.

        **Orden estricto (TDD §8.4 anti-drift)**:
          1. UPDATE checkpoint file con cost_accumulated += cost_usd
             (atómico: tmp+fsync+rename) — YA HECHO en caller vía
             _update_checkpoint_cost. Este método solo hace DB writes.
          2. DB write (best-effort post-checkpoint). Si falla: log warning,
             continue. El checkpoint ya tiene el dato; reconcile_cost()
             en _phase_write / recovery lo arreglará.
          3. Per-job soft alert check (post-commit).
        """
        try:
            await self._db.add_token_usage(
                job_id=job_id,
                phase=phase.value,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=float(cost_usd),
            )
        except Exception as exc:
            logger.warning(
                "token_usage_db_write_failed",
                extra={
                    "job_id": job_id,
                    "phase": phase.value,
                    "error": str(exc),
                    "checkpoint_has_data": True,
                },
            )
            return

        # Per-job soft alert (post-commit)
        new_total = await self._db.get_research_job_cost(job_id)
        await self._check_per_job_budget(job_id, Decimal(str(new_total)))

        # Daily cost gauge
        today_cost = await self._db.get_today_research_cost(user_id=0)
        write_research_metric(
            "research_daily_cost",
            tags={},
            fields={"cost_usd": float(today_cost), "jobs_count": 0},
        )

    async def _update_checkpoint_cost(
        self,
        job_id: str,
        cost_delta: Decimal,
        tokens_in_delta: int,
        tokens_out_delta: int,
    ) -> None:
        """Actualiza cost_accumulated en checkpoint.json. Atómico: tmp+fsync+rename.

        Raise si falla (no se puede garantizar source of truth).
        """
        ckpt_path = self._data_root / job_id / "checkpoint.json"
        tmp_path = ckpt_path.with_suffix(".json.tmp")

        # Read existing or init
        ckpt: dict
        if ckpt_path.exists():
            try:
                ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                ckpt = {
                    "job_id": job_id,
                    "completed_phases": [],
                    "phase_data": {},
                }
        else:
            ckpt = {
                "job_id": job_id,
                "completed_phases": [],
                "phase_data": {},
            }

        # Update accumulators
        prev_cost = Decimal(str(ckpt.get("cost_accumulated_usd", 0)))
        ckpt["cost_accumulated_usd"] = float(prev_cost + cost_delta)
        ckpt["tokens_in_accumulated"] = ckpt.get("tokens_in_accumulated", 0) + tokens_in_delta
        ckpt["tokens_out_accumulated"] = ckpt.get("tokens_out_accumulated", 0) + tokens_out_delta
        ckpt["updated_at"] = format_now()

        # Atomic write (tmp + fsync + rename)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ckpt_path)

    async def reconcile_cost(self, job_id: str) -> Decimal:
        """Source-of-truth priority: checkpoint > token_usage > aggregate.

        Por qué checkpoint primero: es lo ÚNICO que se fsyncea inmediatamente
        tras cada LLM call. La DB write puede perderse por busy_timeout,
        async cancel, container kill. El checkpoint no (write atómico a
        tmp + rename).
        """
        # Source 1: checkpoint file
        ckpt_cost = await self._read_checkpoint_cost(job_id)

        # Source 2: SUM of token_usage table
        db_sum = await self._db.query_scalar(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM research_job_token_usage WHERE job_id = ?",
            job_id,
        )
        db_sum_dec = Decimal(str(db_sum or 0.0))

        # Source 3: aggregate en research_jobs
        agg = await self._db.get_research_job_cost(job_id)
        agg_dec = Decimal(str(agg))

        reconciled = max(ckpt_cost, db_sum_dec, agg_dec)

        if reconciled != agg_dec:
            logger.warning(
                "cost_reconciliation_drift",
                extra={
                    "job_id": job_id,
                    "checkpoint": float(ckpt_cost),
                    "token_usage_sum": float(db_sum_dec),
                    "aggregate": float(agg_dec),
                    "reconciled": float(reconciled),
                },
            )
            write_research_metric(
                "research_budget_drift",
                tags={"source": "reconcile_cost"},
                fields={
                    "count": 1,
                    "drift_usd": float(abs(reconciled - agg_dec)),
                    "checkpoint_usd": float(ckpt_cost),
                    "aggregate_usd": float(agg_dec),
                },
            )
        return reconciled

    async def _read_checkpoint_cost(self, job_id: str) -> Decimal:
        """Lee cost_accumulated_usd del checkpoint.json. 0 si no existe."""
        ckpt_path = self._data_root / job_id / "checkpoint.json"
        if not ckpt_path.exists():
            return Decimal("0")
        try:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            return Decimal(str(ckpt.get("cost_accumulated_usd", 0)))
        except (json.JSONDecodeError, OSError):
            return Decimal("0")

    # =====================================================================
    # Internal helpers
    # =====================================================================

    async def _write_checkpoint_phase(
        self, job_id: str, phase: PhaseName, phase_data: dict
    ) -> None:
        """Persiste phase data en checkpoint.json (atomic write via tmp+rename)."""
        ckpt_path = self._data_root / job_id / "checkpoint.json"
        tmp_path = ckpt_path.with_suffix(".json.tmp")

        if ckpt_path.exists():
            try:
                ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                ckpt = {"job_id": job_id, "completed_phases": [], "phase_data": {}}
        else:
            ckpt = {"job_id": job_id, "completed_phases": [], "phase_data": {}}

        if phase.value not in ckpt.get("completed_phases", []):
            ckpt.setdefault("completed_phases", []).append(phase.value)
        ckpt["current_phase"] = phase.value
        ckpt.setdefault("phase_data", {})[phase.value] = {
            "completed_at": format_now(),
            **phase_data,
        }
        ckpt["updated_at"] = format_now()

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ckpt_path)

    async def _update_phase(
        self,
        job_id: str,
        phase: PhaseName,
        *,
        progress: int | None = None,
    ) -> None:
        """Actualiza current_phase (+progress opcional) en DB."""
        await self._db.update_research_job_status(
            job_id,
            "running",
            current_phase=phase.value,
            progress_percent=progress if progress is not None else 0,
        )

    def _sources_summary(self, sources: list[dict]) -> list[dict]:
        """Compact sources para checkpoint (sin clean_text por tamaño)."""
        return [
            {"url": s.get("url"), "success": s.get("success"), "error": s.get("error")}
            for s in sources
        ]

    @staticmethod
    def _now_dt() -> datetime:
        """Helper: datetime UTC ahora (para run_date del scheduler)."""
        return datetime.now(UTC)
