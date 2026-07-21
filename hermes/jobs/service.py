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
        fetcher: Any,
        settings: Any,
        scheduler: Any,
        report_store: Any | None = None,
    ) -> None:
        self._db = db
        self._notifier = notifier
        self._llm = llm_router
        self._search = web_search
        # Slice 1B+1C1b: external HTTP is funneled through the
        # reviewed safe fetcher boundary. The service no longer owns
        # any direct HTTP transport, AsyncClient, or fallback path.
        # The fetcher must be supplied — there is no default and no
        # optional fallback because that would defeat the boundary.
        self._fetcher = fetcher
        self._settings = settings
        self._scheduler = scheduler
        # Slice 1C2: the read path for ``GET /v1/jobs/{id}/report`` is
        # delegated to ``LocalReportStore``. The service does NOT use
        # this for the WRITE path (``_phase_write`` still writes via
        # ``tmp + fsync + os.replace``) and does NOT use it for any
        # internal read — the route is the sole reader. ``None`` is
        # the legitimate value when the composition root could not
        # construct a report store; the route then returns 500
        # ``report_unavailable`` for complete jobs.
        self._report_store = report_store

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

        # Path raíz para outputs. The writer (this class) and the
        # reader (LocalReportStore) MUST use the SAME canonical path
        # for the full process lifetime. The composition root in
        # ``hermes.__main__._compose_deep_research_runtime`` resolves
        # ``settings.deep_research_data_root`` against the current
        # working directory and passes the resolved absolute path to
        # both the LocalReportStore and this service. If a report
        # store is wired, we use ITS root (the canonical path
        # already resolved at startup); otherwise we fall back to
        # the raw setting so the writer still works in tests that
        # bypass the composition root.
        _store_root = (
            self._report_store.root
            if self._report_store is not None
            else None
        )
        if _store_root is not None:
            self._data_root = _store_root
        else:
            self._data_root = Path(
                getattr(settings, "deep_research_data_root", None) or "data/jobs"
            )

        # =====================================================================
        # Slice 1C1c: explicit stopping / closed lifecycle state.
        #
        # Lifecycle invariants (no exception text is exposed):
        # - ``_stopping`` is set first (synchronous, idempotent). Any new
        #   submit / enqueue is rejected immediately afterwards.
        # - ``_closed`` flips on only AFTER ``aclose`` has drained the
        #   scrape pool, so in-flight workers can still observe
        #   ``_stopping`` while they run.
        # - ``_scrape_active`` is NEVER driven negative: ``_run_in_scrape_pool``
        #   fails closed (raises after closing) BEFORE incrementing the
        #   counter, so the increment/decrement pair remains balanced.
        # =====================================================================
        self._stopping: bool = False
        self._closed: bool = False
        self._aclose_lock: asyncio.Lock = asyncio.Lock()

        # =====================================================================
        # DR-Q1A-PRE1B: real Deep Research cancellation contract.
        #
        # Per-job active-task registry + per-job terminal-state seam + a
        # per-job mutex around the registry. The registry is intentionally
        # ``dict[str, asyncio.Task]`` (NOT a ``set``): the cancel endpoint
        # needs to obtain the exact task to call ``.cancel()`` on it, and
        # the registry-unregister step in the research task's ``finally``
        # block must verify the task is still the same one that registered
        # (a newer attempt must NOT be evicted by an older one's teardown).
        #
        # The user-cancel intent set distinguishes user-requested
        # cancellation from process-level / scheduler-teardown
        # cancellation. A ``CancelledError`` raised by asyncio without
        # user intent and without persistence in ``cancelling`` /
        # ``cancelled`` is treated as an infra shutdown — the running
        # state is preserved for recovery instead of being finalized as
        # user-cancelled.
        #
        # Both the registry and the intent set are process-local. They
        # are guarded by the same ``_cancel_lock`` because they are
        # always touched together (register+intent-mark, or
        # unregister+intent-check).
        # =====================================================================
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._user_cancel_intent: set[str] = set()
        self._terminal_locks: dict[str, asyncio.Lock] = {}
        self._cancel_lock: asyncio.Lock = asyncio.Lock()

    # =====================================================================
    # Slice 1C1c: explicit stopping / closed lifecycle seams
    # =====================================================================

    def stop_accepting(self) -> bool:
        """Mark the service as no longer accepting submissions. Idempotent.

        Synchronous seam callable from any thread or shutdown hook.
        Subsequent ``submit_job`` / ``retry_job`` invocations will reject
        with ``SchedulerUnavailableError`` *before* any budget check,
        DB write, or scheduler enqueue. Subsequent calls to this method
        are no-ops and return the already-stopping value.

        Returns:
            True if this call flipped the state (no prior stop), False
            if the service was already in stopping mode.
        """
        if self._stopping:
            return False
        self._stopping = True
        return True

    @property
    def accepting(self) -> bool:
        """True when ``submit_job`` / ``retry_job`` can still be called."""
        return not self._stopping

    @property
    def closed(self) -> bool:
        """True once ``aclose`` has finished (Drain-safe)."""
        return self._closed

    # =====================================================================
    # DR-Q1A-PRE1B: per-job task registry + user-cancel intent + terminal seam
    # =====================================================================

    def _get_terminal_lock(self, job_id: str) -> asyncio.Lock:
        """Lazy per-job terminal-state lock (linearizes cancel-vs-complete).

        Both ``cancel_job`` and the final phase-5 ``_phase_write``
        completion path must serialize through this lock. It is
        acquired only across the DB CAS + a single await, and is
        NEVER held while waiting for the research task to
        acknowledge cancellation.
        """
        lock = self._terminal_locks.get(job_id)
        if lock is None:
            lock = asyncio.Lock()
            self._terminal_locks[job_id] = lock
        return lock

    async def _register_active_task(
        self, job_id: str, task: asyncio.Task[Any]
    ) -> None:
        """Register the research asyncio.Task for ``job_id`` (replace any older).

        Called by ``_run_research`` at the very beginning, before
        the pending -> running CAS. A second register for the
        same ``job_id`` (e.g. recovery re-runs the coroutine) MUST
        replace the prior task. The terminal lock and the cancel
        intent set are NOT replaced: they reflect the user-facing
        intent and survive across attempts.
        """
        async with self._cancel_lock:
            self._active_tasks[job_id] = task

    async def _unregister_active_task(
        self, job_id: str, expected: asyncio.Task[Any]
    ) -> None:
        """Remove the registry entry only if it still points to ``expected``.

        A newer attempt's task must NOT be evicted by an older
        attempt's finally block. The ``is`` identity check is the
        contract.
        """
        async with self._cancel_lock:
            current = self._active_tasks.get(job_id)
            if current is expected:
                del self._active_tasks[job_id]

    def _peek_active_task(self, job_id: str) -> asyncio.Task[Any] | None:
        """Read the registry WITHOUT taking the lock.

        Used by the cancel endpoint to obtain the exact task to
        ``.cancel()`` so the cancellation can propagate. The
        registry may legitimately be empty (the task has already
        finished), in which case the cancel endpoint treats the
        job as already-finalized and falls back to the DB state.
        """
        return self._active_tasks.get(job_id)

    def _mark_user_cancel_intent(self, job_id: str) -> None:
        """Mark that the user requested cancellation for this job.

        Survives across attempts (a recovery re-run will see the
        intent and finalizes as user-cancelled, not infra-shutdown).
        The intent is removed only when the row transitions to a
        terminal state.
        """
        self._user_cancel_intent.add(job_id)

    def _user_cancel_intended(self, job_id: str) -> bool:
        """Whether the user requested cancellation for this job."""
        return job_id in self._user_cancel_intent

    def _clear_user_cancel_intent(self, job_id: str) -> None:
        """Drop the intent after the job reaches a terminal state."""
        self._user_cancel_intent.discard(job_id)

    async def _run_in_scrape_pool(self, fn: Any, *args: Any) -> Any:
        """Ejecuta ``fn`` en el threadpool ``_scrape_pool`` con counter explícito.

        Wrapper sobre ``loop.run_in_executor`` que mantiene
        ``self._scrape_active`` sincronizado. Usado por Phase 2 (HTML parsing)
        para que la métrica de saturación refleje workers realmente ocupados
        en vez de inspeccionar ``_idle_semaphore._value`` (interno de CPython).

        El counter se incrementa ANTES de submit (evita race: si el thread
        arranca y decrementa antes de que incrementemos, veríamos negativo)
        y se decrementa en finally (cubre excepciones y cancels).

        Slice 1C1c: failure-closed after ``aclose`` begins. Once
        ``self._closed`` is set, this method raises ``SchedulerUnavailableError``
        BEFORE incrementing the counter so the in-flight accounting never
        goes negative. In-flight workers that have already incremented are
        free to finish or be cancelled; only NEW submissions are rejected.
        """
        if self._closed:
            raise SchedulerUnavailableError("Service is closing")
        self._scrape_active += 1
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._scrape_pool, fn, *args)
        finally:
            self._scrape_active -= 1

    async def aclose(self, timeout_s: float = 10.0) -> bool:
        """Stop the service deterministically. Idempotent and deadline-bounded.

        Sequence (intentional order):
        1. ``stop_accepting()`` — flips ``_stopping`` synchronously so
           concurrent ``submit_job`` / ``retry_job`` reject immediately.
        2. Fail-closed the scrape pool: any *new* ``_run_in_scrape_pool``
           call raises ``SchedulerUnavailableError`` (counter is NEVER
           driven negative — see that method for the guarantee).
        3. Cancel in-flight executor work and wait for the scrape pool
           to drain. We use a bounded ``run_in_executor`` awaiting pattern
           scheduled on the loop so the deadline is honored even if a
           worker hangs on a ``to_thread`` call.
        4. Mark ``_closed`` so further ``aclose`` calls are idempotent
           no-ops. The original executor reference is released (no
           replacement executor is created — would defeat shutdown).

        No exception text is exposed. The boolean honestly distinguishes
        graceful drain (``True``) from deadline expiry (``False``).

        Returns:
            True if the scrape pool drained within ``timeout_s`` and the
            service is fully closed; False if the deadline was hit (in
            which case the executor is cancelled and the service remains
            in ``_closed=True`` anyway).
        """
        # Idempotency guard. ``_closed`` is set on the first call's
        # success OR deadline path; subsequent calls return its outcome
        # without re-running the lifecycle.
        if self._closed:
            return self._scrape_active == 0

        async with self._aclose_lock:
            if self._closed:
                return self._scrape_active == 0

            # (1) Stop accepting first — synchronous and immediate.
            self.stop_accepting()

            drained = True
            loop = asyncio.get_running_loop()
            deadline = loop.time() + max(timeout_s, 0.0)

            # (2)+(3) Drain or cancel the scrape pool. We do NOT wait
            # on ``self._scrape_pool.shutdown(wait=True)`` because that
            # blocks the asyncio event-loop thread; we instead probe the
            # counter under the deadline and explicitly cancel stuck
            # workers via the loop's ``run_in_executor`` integration.
            try:
                remaining_sleep = max(0.0, deadline - loop.time())
                while self._scrape_active > 0 and loop.time() < deadline:
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(asyncio.sleep(min(0.05, remaining_sleep))),
                            timeout=remaining_sleep,
                        )
                    except TimeoutError:
                        break
                    remaining_sleep = max(0.0, deadline - loop.time())
                if self._scrape_active > 0:
                    drained = False
            except Exception:
                logger.exception("deep_research_service_drain_error")
                drained = False

            # (4) Finalize: release the executor. We do NOT recreate it.
            # If drain wasn't complete we still mark ``_closed`` so any
            # later ``aclose`` call is a no-op instead of racing.
            self._closed = True
            try:
                self._scrape_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                # Best-effort: the pool is going out of scope either way.
                logger.exception("deep_research_service_executor_shutdown_error")

            return drained

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
            SchedulerUnavailableError: si el scheduler no está inicializado
                OR ``stop_accepting()`` has been called OR ``aclose`` has
                started. The rejection happens BEFORE budget / DB /
                scheduler enqueue so the service can drain safely.
        """
        import uuid

        # Slice 1C1c: reject submissions the moment stopping begins, BEFORE
        # any budget check, DB write, or scheduler enqueue. This is the
        # "fail closed" guarantee — once stop_accepting() flips the
        # flag (synchronously), no new research work enters the pipeline.
        if self._stopping or self._closed:
            raise SchedulerUnavailableError("Service is no longer accepting submissions")

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

        # Slice 1C2: JobDetail no longer exposes filesystem paths. The
        # internal DB ``output_path`` / ``partial_output_path`` /
        # ``checkpoint_path`` columns stay in the schema (no migration
        # in 1C2) but are NOT part of the public DTO. Status is the
        # source of truth; the client calls
        # ``GET /v1/jobs/{id}/report`` to retrieve the markdown.

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
            error_taxonomy=(
                ErrorTaxonomy(job_row["error_taxonomy"]) if job_row.get("error_taxonomy") else None
            ),
            error_message=job_row.get("error_message"),
            tokens_in=job_row["tokens_in"],
            tokens_out=job_row["tokens_out"],
            notified=bool(job_row.get("notified", 0)),
            updated_at=job_row["updated_at"],
            token_usage=token_usage,
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
        """Real Deep Research cancellation: request immediate local cancellation.

        DR-Q1A-PRE1B. The previous PRE1A behavior was DB-only: the
        endpoint set ``status='cancelling'`` and the running task
        was not signalled. PRE1B makes the cancellation real for
        every state:

          pending / scheduled:
            atomically transition pending|running -> cancelling;
            remove the scheduler entry best-effort;
            finalize as cancelled (set completed_at, set
            error_taxonomy='cancelled', clear pending state).
            No provider call may start after this point — the
            startup CAS in ``_run_research`` will see a non-pending
            state and exit without external work.

          running:
            atomically transition running -> cancelling;
            mark user-cancel intent;
            obtain the registered asyncio Task and call
            ``task.cancel()``;
            for ``graceful=True``, bounded wait for the task to
            acknowledge; if it does, the row is already cancelled
            and we return ``status=cancelled``; otherwise we
            return ``status=cancelling`` (the task finalizer will
            complete the transition when the asyncio cancellation
            propagates);
            for ``graceful=False``, return immediately with
            ``status=cancelling`` (the task finalizer will produce
            the row's ``cancelled`` transition when propagation
            completes).

          already cancelling:
            idempotently re-signal cancellation if a registered
            task is still present; apply the requested wait mode.

          already cancelled:
            return 200 with status=cancelled; the cancellation is
            idempotent.

          complete / failed:
            return 409 ``JobAlreadyTerminalError`` (the previous
            PRE1A contract; the post-merge slice refuses to
            resurrect terminal state).

        Product contract honored:

          When the owner cancels a Deep Research job, Oroimen
          stops executing that job locally.

        The ``graceful`` parameter is a WAIT MODE, not a
        cancellation-strength mode. Both values request real
        local cancellation. ``graceful=True`` waits up to
        ``deep_research_cancel_wait_s`` for the asyncio task to
        acknowledge; ``graceful=False`` returns as soon as the
        cancellation request has been signalled.

        Provider-side truth:

          Oroimen requests immediate local cancellation and
          propagates asyncio cancellation through the awaited
          client coroutine. An already-received provider request
          may still be processed or counted by the provider.
          Cancellation does NOT claim quota reversal, refund, or
          reversal of billed tokens.

        Raises:
            JobNotFoundError: if the id does not exist.
            JobAlreadyTerminalError: if the job is in ``complete``
                or ``failed`` (the contract keeps the 409).
        """
        job_row = await self._db.get_research_job(job_id)
        if job_row is None:
            raise JobNotFoundError(f"Job {job_id} not found")

        current_status = JobStatus(job_row["status"])

        # 409 contract: complete and failed are terminal, cannot be
        # cancelled. The existing JobAlreadyTerminalError is reused.
        if current_status in (JobStatus.COMPLETE, JobStatus.FAILED):
            raise JobAlreadyTerminalError(current_status)

        # Idempotent: already cancelled -> 200 with the existing
        # status. No work to do.
        if current_status is JobStatus.CANCELLED:
            return CancelResponse(
                id=job_id,
                status=JobStatus.CANCELLED,
                graceful=graceful,
            )

        # Mark user-cancel intent BEFORE the CAS. A recovery re-run
        # that observes a CancelledError will see the intent and
        # treat it as a user cancellation, not infra shutdown.
        self._mark_user_cancel_intent(job_id)

        # Linearize the transition through the per-job terminal
        # seam so a parallel _phase_write completion cannot win
        # before we have applied the cancelling transition. NB: we
        # do NOT hold this lock while waiting for the research
        # task to acknowledge (the task's finalizer needs the
        # same lock to flip the row to cancelled).
        term_lock = self._get_terminal_lock(job_id)
        async with term_lock:
            # pending|running -> cancelling. If the row is in
            # another state (e.g. a parallel completion won the
            # race and the row is already complete), the predicate
            # does not match and the function returns the canonical
            # 200 + status from the row.
            transitioned = await self._db.transition_research_job_status(
                job_id,
                from_states=(JobStatus.PENDING.value, JobStatus.RUNNING.value),
                to_state=JobStatus.CANCELLING.value,
            )
            if not transitioned:
                # Re-read the row to see the actual state.
                job_row = await self._db.get_research_job(job_id)
                actual = JobStatus(job_row["status"]) if job_row else JobStatus.PENDING
                if actual in (JobStatus.COMPLETE, JobStatus.FAILED):
                    raise JobAlreadyTerminalError(actual)
                if actual is JobStatus.CANCELLED:
                    return CancelResponse(
                        id=job_id,
                        status=JobStatus.CANCELLED,
                        graceful=graceful,
                    )
                # cancelling: idempotent; fall through to wait.
                current_status = actual

        # At this point: the row is in 'cancelling' (either because
        # we just transitioned it, or it was already there and we
        # re-marked user intent).

        # pending: there is no active task. Remove the scheduler
        # entry best-effort and finalize the row as cancelled
        # immediately. The startup CAS will reject the (now-cancelled)
        # row before any external call begins.
        if current_status is JobStatus.PENDING:
            try:
                self._scheduler.cancel_scheduled(job_id)
            except Exception:
                logger.exception(
                    "cancel_scheduler_remove_failed",
                    extra={"job_id": job_id},
                )
            # Finalize as cancelled synchronously: the row never
            # transitioned through 'running', so no provider
            # work happened.
            await self._db.transition_research_job_status(
                job_id,
                from_states=(JobStatus.CANCELLING.value,),
                to_state=JobStatus.CANCELLED.value,
                completed_at=format_now(),
                error_taxonomy="cancelled",
                error_message="cancelled_before_start",
            )
            self._clear_user_cancel_intent(job_id)
            return CancelResponse(
                id=job_id,
                status=JobStatus.CANCELLED,
                graceful=graceful,
            )

        # current_status is CANCELLING (either just transitioned
        # from running, or already cancelling on entry).
        # If a task is registered, signal it.
        active = self._peek_active_task(job_id)
        if active is not None and not active.done():
            # The cancel handler must not be the research task
            # itself (we do not call cancel on ourselves).
            try:
                if active is not asyncio.current_task():
                    active.cancel()
            except Exception:
                logger.exception(
                    "cancel_task_signal_failed",
                    extra={"job_id": job_id},
                )

        if not graceful:
            # Return immediately. The task's finalizer will flip
            # the row to cancelled when the asyncio cancellation
            # propagates. The HTTP caller does not wait.
            return CancelResponse(
                id=job_id,
                status=JobStatus.CANCELLING,
                graceful=False,
            )

        # graceful=True: bounded wait for the task to acknowledge
        # cancellation. The wait is bounded by
        # ``deep_research_cancel_wait_s`` and is implemented with
        # ``asyncio.wait_for`` so a stuck finalizer cannot hang the
        # HTTP handler.
        wait_s = float(
            getattr(self._settings, "deep_research_cancel_wait_s", 5.0)
        )
        active = self._peek_active_task(job_id)
        if active is None or active.done():
            # No active task or already done — the finalizer has
            # either not yet run, or already finalized. Re-read the
            # row: if it is cancelled, return cancelled; otherwise
            # the task has terminated but the finalizer is racing
            # — surface cancelling.
            job_row = await self._db.get_research_job(job_id)
            actual = JobStatus(job_row["status"]) if job_row else JobStatus.CANCELLING
            return CancelResponse(
                id=job_id,
                status=(
                    JobStatus.CANCELLED
                    if actual is JobStatus.CANCELLED
                    else JobStatus.CANCELLING
                ),
                graceful=True,
            )
        try:
            await asyncio.wait_for(asyncio.shield(active), timeout=wait_s)
        except TimeoutError:
            # Acknowledgement did not arrive inside the bounded
            # wait. The task is still alive but the row is in
            # 'cancelling' and the finalizer will complete the
            # transition when propagation finishes.
            logger.warning(
                "cancel_graceful_timeout",
                extra={"job_id": job_id, "wait_s": wait_s},
            )
            return CancelResponse(
                id=job_id,
                status=JobStatus.CANCELLING,
                graceful=True,
            )
        except asyncio.CancelledError:
            # The research task raised CancelledError (the
            # asyncio cancellation propagated through its
            # awaits). The shield does not protect against the
            # inner task raising CancelledError on its own;
            # ``wait_for`` propagates the exception. The HTTP
            # handler's own task is NOT cancelled here (the
            # handler is awaiting ``wait_for``, not a direct
            # cancel target). Treat this as success: the
            # cancellation was acknowledged. Re-read the row
            # for the final status. The finalizer has likely
            # already run (or is about to) and the row may be
            # ``cancelling`` or ``cancelled``.
            logger.info(
                "cancel_graceful_acknowledged",
                extra={"job_id": job_id},
            )
            job_row = await self._db.get_research_job(job_id)
            actual = (
                JobStatus(job_row["status"]) if job_row else JobStatus.CANCELLING
            )
            return CancelResponse(
                id=job_id,
                status=(
                    JobStatus.CANCELLED
                    if actual is JobStatus.CANCELLED
                    else JobStatus.CANCELLING
                ),
                graceful=True,
            )
        except Exception:
            logger.exception(
                "cancel_graceful_wait_failed",
                extra={"job_id": job_id},
            )
            return CancelResponse(
                id=job_id,
                status=JobStatus.CANCELLING,
                graceful=True,
            )

        # Acknowledgement received. Re-read the row for the final
        # status. If the finalizer has already flipped to cancelled,
        # we return cancelled; otherwise we surface cancelling
        # (rare: the task ended but the finalizer is still mid-DB).
        job_row = await self._db.get_research_job(job_id)
        actual = JobStatus(job_row["status"]) if job_row else JobStatus.CANCELLING
        return CancelResponse(
            id=job_id,
            status=(
                JobStatus.CANCELLED
                if actual is JobStatus.CANCELLED
                else JobStatus.CANCELLING
            ),
            graceful=True,
        )

    async def retry_job(self, job_id: str, user_id: int = 0) -> JobResponse:
        """Crea nuevo job copiando checkpoint del original. NO re-scrape.

        Raises:
            JobNotFoundError: si original no existe.
            JobNotRetryableError: si original NO está en 'failed'.
            SchedulerUnavailableError: si ``stop_accepting()`` /
                ``aclose`` ha begun — antes de cualquier DB write o
                scheduler enqueue (Slice 1C1c fail-closed contract).
        """
        import uuid

        # Slice 1C1c: same fail-closed guard as submit_job — applied
        # before the original-row SELECT so we never touch DB rows when
        # the service is no longer accepting work.
        if self._stopping or self._closed:
            raise SchedulerUnavailableError("Service is no longer accepting submissions")

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

        DR-Q1A-PRE1B: registers the active asyncio.Task at the very
        start, before any state transition, so the cancel endpoint
        can signal it. The startup CAS (pending -> running) is
        conditional: a cancellation that won first flips the row
        to ``cancelling`` and the task exits without any external
        call. A ``CancelledError`` is interpreted against the
        user-cancel intent marker and the persisted status: a
        user-requested cancellation runs a finalizer that
        reconciles cost, preserves token usage, and transitions
        the row to ``cancelled``; an infrastructure-level
        cancellation without user intent preserves the running
        state for the recovery contract.

        Transiciones:
          pending → running → (complete | failed | cancelled)
        """
        start_time = time.monotonic()

        # DR-Q1A-PRE1B: register the asyncio.Task at the very
        # beginning, before any state transition. This guarantees
        # the cancel endpoint can find the task to call
        # ``.cancel()`` on. A second register for the same job
        # (recovery re-run) replaces the prior task.
        current_task = asyncio.current_task()
        if current_task is None:
            # ``_run_research`` is always invoked from inside
            # the asyncio event loop (via AsyncIOScheduler), so
            # ``current_task`` is non-None in practice. The guard
            # exists to satisfy the type checker and the rare
            # edge case where a unit test invokes this method
            # without an event loop.
            logger.error(
                "run_research_no_current_task",
                extra={"job_id": job_id},
            )
            return
        await self._register_active_task(job_id, current_task)

        try:
            await self._run_research_inner(job_id, start_time)
        finally:
            # Idempotent: only the task that registered itself
            # removes itself. A newer attempt cannot be evicted
            # by an older one's finally block.
            await self._unregister_active_task(job_id, current_task)
            # NB: the cancel intent set is NOT cleared here. It
            # is cleared when the row reaches a terminal state
            # (cancelled, complete, failed) — the next run, if
            # any, starts without user-cancel intent. A recovery
            # re-run will see the intent and treat the CancelledError
            # as a user cancellation, not infra shutdown.

    async def _run_research_inner(self, job_id: str, start_time: float) -> None:
        """The actual research loop. Separated from ``_run_research``
        so the outer method owns the active-task registry + the
        finalizer.
        """
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
            # OK, transicionar a running.
            # DR-Q1A-PRE1B: atomic CAS. The pending -> running
            # transition is conditional on the row still being in
            # 'pending'. If a cancellation won the race and
            # transitioned the row to 'cancelling', the CAS does
            # not match and the task exits. The startup CAS is
            # the only correct linearization point: a cancel that
            # arrives even one asyncio tick later is honored by
            # the phase guards in the inner loop and by the
            # CancelledError handler.
            now = format_now()
            await self._db.conn.execute(
                "UPDATE research_jobs SET status='running', started_at=?, updated_at=? "
                "WHERE id = ? AND status = 'pending'",
                (now, now, job_id),
            )
            # If the rowcount is 0 the row is no longer 'pending'
            # (e.g. cancelled). Read the row to confirm.
            cur = await self._db.conn.execute(
                "SELECT status FROM research_jobs WHERE id = ?", (job_id,)
            )
            status_row = await cur.fetchone()
            await self._db.conn.execute("COMMIT")
            actual_status = (
                status_row["status"] if isinstance(status_row, dict) else status_row[0]
            )
            if actual_status != "running":
                logger.info(
                    "run_research_skip_after_cas",
                    extra={"job_id": job_id, "status": actual_status},
                )
                return
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
        except asyncio.CancelledError:
            # DR-Q1A-PRE1B: real Deep Research cancellation. The
            # ``asyncio.CancelledError`` is propagated by an
            # active ``task.cancel()`` call from ``cancel_job`` or
            # by the asyncio loop itself (infrastructure shutdown).
            # The two cases are distinguished by:
            #   - the user-cancel intent marker (set by
            #     ``cancel_job`` BEFORE the CAS);
            #   - the persisted status (if the row is in
            #     ``cancelling`` or ``cancelled`` the row was
            #     already moved by the cancel endpoint).
            # If the cancellation was user-initiated (or the row
            # already shows cancelling/cancelled) the row is
            # finalized as ``cancelled``: cost is reconciled, the
            # checkpoint is removed, transient artifacts (the
            # ``.md.tmp`` and the per-attempt checkpoint dir) are
            # cleaned, no notifier is sent. If the cancellation
            # was NOT user-initiated and the row is not in
            # ``cancelling``/``cancelled``, the running state is
            # preserved for the recovery contract.
            await self._handle_cancellation(job_id, start_time)
            raise
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
    # Internal: cancellation finalizer (DR-Q1A-PRE1B)
    # =====================================================================

    async def _handle_cancellation(self, job_id: str, start_time: float) -> None:
        """Finalize a cancelled job (cancelling -> cancelled).

        Invoked from the ``asyncio.CancelledError`` branch in
        ``_run_research_inner``. Decides whether the cancellation
        was user-initiated (intent marker or persisted status) and:

          - user-cancelled: atomic CAS cancelling -> cancelled, set
            ``error_taxonomy='cancelled'``, reconcile the
            checkpoint cost, remove the per-attempt checkpoint
            after successful reconciliation, clean the ``.md.tmp``
            transient report, do not call the complete or the
            failed notifier.
          - not user-cancelled: preserve the running state for
            recovery. The recovery hook (recover_research_jobs)
            will see the running row and reset it to pending on
            the next startup. The persisted cost and token usage
            are left intact.

        The finalizer is cancellation-safe: it is itself a
        coroutine and the caller's ``except CancelledError``
        re-raises the original after the finalizer completes. A
        second ``CancelledError`` raised inside the finalizer is
        caught and logged so the finalizer itself is not
        abandoned.

        Does NOT close shared provider clients. Does NOT issue
        a live cancellation probe. Does NOT claim quota reversal.
        """
        try:
            # Decide: user-cancelled or infra shutdown?
            row = await self._db.get_research_job(job_id)
            current = JobStatus(row["status"]) if row else JobStatus.PENDING
            user_intended = self._user_cancel_intended(job_id)
            persisted_cancel = current in (
                JobStatus.CANCELLING,
                JobStatus.CANCELLED,
            )
            if not (user_intended or persisted_cancel):
                # Infrastructure shutdown. Preserve running state
                # for the recovery contract. The task is being
                # cancelled by the asyncio loop, NOT by the user
                # cancel endpoint. The DB row is still 'running'
                # (or whatever the current phase left it as).
                # No DB writes; no notifier; no cleanup. The
                # recovery hook on the next startup will see the
                # running row and reset it to pending.
                logger.info(
                    "cancel_infra_preserved",
                    extra={"job_id": job_id, "status": current.value},
                )
                return

            # User-cancelled: finalize the row.
            # 1. Reconcile the cost. The reconcile is monotonic;
            #    the cost includes any in-flight checkpoint
            #    exposure that the token-usage DB write may not
            #    have captured.
            try:
                await self.reconcile_cost(job_id)
            except Exception:
                logger.exception(
                    "cancel_reconcile_failed",
                    extra={"job_id": job_id},
                )

            # 2. Atomic CAS cancelling -> cancelled. The status
            #    predicate accepts both 'cancelling' (the user
            #    cancel path) and 'running' (a race where the
            #    cancel arrived between phase boundaries; the row
            #    was still 'running' at the last phase update but
            #    the user intent was already marked). 'cancelling'
            #    is the documented prior state.
            term_lock = self._get_terminal_lock(job_id)
            async with term_lock:
                finalized = await self._db.transition_research_job_status(
                    job_id,
                    from_states=(
                        JobStatus.CANCELLING.value,
                        JobStatus.RUNNING.value,
                    ),
                    to_state=JobStatus.CANCELLED.value,
                    completed_at=format_now(),
                    error_taxonomy="cancelled",
                    error_message="cancelled_by_user",
                )
                if not finalized:
                    # Another path (e.g. recovery) already
                    # finalized the row. Log and exit.
                    logger.info(
                        "cancel_already_finalized",
                        extra={"job_id": job_id},
                    )
            # 3. Remove transient artifacts: the per-attempt
            #    checkpoint file and the .md.tmp draft. The
            #    permanent report file (job_id.md) is NOT
            #    written for a cancelled attempt. The DB job row
            #    and the recorded token_usage rows are kept so
            #    the cancelled job remains inspectable through
            #    JobDetail.
            try:
                ckpt_path = self._data_root / job_id / "checkpoint.json"
                if ckpt_path.exists():
                    ckpt_path.unlink()
            except OSError:
                pass
            try:
                job_dir = self._data_root / job_id
                if job_dir.is_dir() and not any(job_dir.iterdir()):
                    job_dir.rmdir()
            except OSError:
                pass
            try:
                tmp_path = self._data_root / f"{job_id}.md.tmp"
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

            # 4. Log + metric. No notifier (cancelled jobs are
            #    not reported as failed or complete; the cancel
            #    response already acknowledged the cancellation).
            duration = time.monotonic() - start_time
            total_cost = await self._db.get_research_job_cost(job_id)
            logger.info(
                "research_job_cancelled",
                extra={
                    "job_id": job_id,
                    "total_cost_usd": total_cost,
                    "total_duration_s": duration,
                },
            )
            write_research_metric(
                "research_job_completed",
                tags={
                    "status": "cancelled",
                    "job_type": "deep_research",
                    "error_taxonomy": "cancelled",
                },
                fields={
                    "count": 1,
                    "total_duration_s": duration,
                    "total_cost_usd": total_cost,
                },
            )
        except asyncio.CancelledError:
            # The finalizer itself was cancelled. Log and
            # swallow so the outer ``except CancelledError`` can
            # re-raise. The recovery contract will see the row
            # in 'cancelling' and finalize it on the next
            # startup.
            logger.exception(
                "cancel_finalizer_cancelled",
                extra={"job_id": job_id},
            )
        finally:
            # Clear the user-cancel intent for this job; the row
            # is now terminal.
            self._clear_user_cancel_intent(job_id)

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
        """Phase 2: safe-fetch external URL + selectolax html_to_text.

        Slice 1B+1C1b: HTTP fetching is funneled exclusively through the
        reviewed SafeExternalFetcher boundary. There is no direct httpx,
        no AsyncClient, and no fallback transport — the fetcher must be
        supplied at construction time.

        Para cada URL:
          1. fetch bounded bytes via ``await self._fetcher.fetch(url)``
          2. Size Guard: si body > 2MB → truncate ANTES de to_thread
          3. html_to_text via custom ThreadPoolExecutor (4 workers)
          4. si clean_text < 100 chars → mark success=False, error='too_short'

        Output: list of dicts con keys: url, success, clean_text?, error?

        Failure handling: any safe-fetch failure (including
        ``SafeFetchError`` and unexpected exceptions) is collapsed into a
        stable redacted source failure. The URL, hostname, body bytes,
        exception text, and FetchErrorCode value are NEVER returned to
        the caller nor logged here — only a generic ``safe_fetch_failed``
        marker. This preserves the privacy guarantee from the fetcher
        boundary.
        """
        # Observability: threadpool saturation al inicio de phase 2.
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
                # SafeExternalFetcher.fetch returns a FetchResult with bounded bytes.
                # The fetcher boundary is the ONLY place where the URL is resolved.
                result = await self._fetcher.fetch(url)
                raw = result.body
                # Size Guard 2MB ANTES de to_thread (P0-1 v1.3 Gemini).
                # Operates on local bytes only — no additional fetch.
                if len(raw) > _HTML_SIZE_GUARD_BYTES:
                    raw = raw[:_HTML_SIZE_GUARD_BYTES]
                # Deterministic, bounded UTF-8 decode (errors='replace') BEFORE
                # dispatching to the thread pool. This is the ONLY network → text
                # decode seam: no second fetch, no fallback transport, no httpx
                # access here. selectolax and the regex fallback both expect str,
                # so we MUST convert bytes → str before to_thread.
                html_text = raw.decode("utf-8", errors="replace")
                # HTML parse en thread pool dedicado (no default executor).
                # NB1: usamos _run_in_scrape_pool para mantener
                # self._scrape_active sincronizado (saturación métrica).
                clean = await self._run_in_scrape_pool(html_to_text_selectolax, html_text)
                if len(clean) < 100:
                    return {"url": url, "success": False, "error": "too_short"}
                return {"url": url, "success": True, "clean_text": clean}
            except Exception:
                # Stable redacted source failure. Do NOT include URL,
                # hostname, exception text, exception type, or any
                # underlying fetcher code in the returned marker. The
                # fetcher boundary already enforces the same redaction.
                return {"success": False, "error": "safe_fetch_failed"}

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

        DR-Q1A-PRE1B: cancel-vs-complete linearization. The completion
        path acquires the per-job terminal seam, verifies the row is
        still in ``running`` via a conditional CAS, writes the
        report atomically, reconciles cost, conditionally transitions
        running -> complete, releases the seam, then sends the
        completion notifier. If a cancellation won the race the
        CAS fails and the function exits without publishing the
        report and without sending the notifier; the temporary
        ``.md.tmp`` draft is cleaned.
        """
        final_path = self._data_root / f"{job_id}.md"
        tmp_path = final_path.with_suffix(".md.tmp")

        # Step 1: write the report to a temporary file. This is
        # safe to do before the terminal CAS because the
        # ``.md.tmp`` is local and is either atomically replaced
        # to ``.md`` (commit) or unlinked (cancel won).
        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(report)
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise PhaseError("oom", f"disk_write_failed:{exc!s}", retryable=False) from exc

        # Step 2: acquire the per-job terminal seam and verify the
        # row is still in 'running'. If a cancellation won the
        # race the row is in 'cancelling' or 'cancelled' and the
        # function returns without publishing the report or
        # sending the notifier. The terminal seam is NOT held
        # while waiting for the research task; it only serializes
        # the cancel-vs-complete decision and the final DB write.
        term_lock = self._get_terminal_lock(job_id)
        async with term_lock:
            # Conditional transition: running -> complete. If the
            # predicate does not match, cancellation won the race
            # and the function returns without publishing.
            # progress_percent=100 is set atomically with the
            # status transition so a subsequent reader always
            # sees the same 100% / complete state (PRE1B bug fix:
            # the previous conditional ``_update_phase(WRITE, 100)``
            # would skip because the status had already flipped
            # out of 'running').
            cost = await self.reconcile_cost(job_id)
            transitioned = await self._db.transition_research_job_status(
                job_id,
                from_states=(JobStatus.RUNNING.value,),
                to_state=JobStatus.COMPLETE.value,
                completed_at=format_now(),
                progress_percent=100,
            )
            if not transitioned:
                # Cancellation won the race. Clean the temp file
                # and return without publishing.
                logger.info(
                    "phase_write_cancel_won",
                    extra={"job_id": job_id},
                )
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
                # Re-raise so the caller (the research task)
                # unwinds through the CancelledError handler. The
                # task has likely already been signalled by
                # cancel_job; the finalizer will then run.
                raise asyncio.CancelledError()

            # CAS succeeded. The row is now 'complete'. We may
            # now publish the report file and proceed.
            # Mark the output_path on the row (separate update;
            # the CAS already wrote status='complete').
            try:
                await self._db.conn.execute(
                    "UPDATE research_jobs SET output_path = ?, updated_at = ? "
                    "WHERE id = ?",
                    (str(final_path), format_now(), job_id),
                )
                await self._db.conn.commit()
            except Exception:
                logger.exception(
                    "phase_write_output_path_failed",
                    extra={"job_id": job_id},
                )
            # Publish the report file atomically. tmp + os.replace
            # is atomic at the POSIX/NTFS level for the same
            # filesystem.
            try:
                os.replace(tmp_path, final_path)
            except OSError as exc:
                # The row is already 'complete' but the report
                # file is not. The next /report GET will 500
                # ``report_unavailable`` defensively.
                logger.exception(
                    "phase_write_publish_failed",
                    extra={"job_id": job_id, "error": str(exc)},
                )

            # Notifier (best-effort). Sent only after the
            # report is published; the notifier reads the row
            # state, not the file.
            if hasattr(self._notifier, "send_research_complete"):
                job_row = await self._db.get_research_job(job_id)
                notify_via_tg = (
                    bool(job_row.get("notify_via_tg", 1)) if job_row else True
                )
                if notify_via_tg:
                    try:
                        sent_ok = await self._notifier.send_research_complete(
                            job_id=job_id,
                            cost_usd=cost,
                        )
                        if sent_ok:
                            await self._db.mark_research_job_notified(job_id)
                    except Exception:
                        logger.exception(
                            "research_notif_complete_failed",
                            extra={"job_id": job_id},
                        )

        # Step 3: terminal-state seam released. Clean up the
        # checkpoint and the per-attempt job dir.
        ckpt_path = self._data_root / job_id / "checkpoint.json"
        try:
            if ckpt_path.exists():
                ckpt_path.unlink()
        except OSError:
            pass
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
        """Wrapper: ejecuta phase_fn con retry de 3 intentos totales.

        Comportamiento actual (DR-Q1A-PRE1A, sin cambios de comportamiento):
          - Máximo de 3 intentos totales para errores retryables
            (``RETRYABLE_ERRORS``: search_5xx, llm_5xx, timeout, network).
          - Backoff efectivo: 1 segundo después del primer fallo,
            4 segundos después del segundo fallo.
          - El tercer fallo TERMINA la fase (no se reintenta);
            el ``PhaseError`` se relanza.
          - El valor 16 en ``_RETRY_BACKOFF_SCHEDULE = (1, 4, 16)``
            EXISTE en la tupla pero NO se consume en el bucle actual
            (el bucle itera ``for attempt in range(3)`` y solo lee
            ``_RETRY_BACKOFF_SCHEDULE[attempt]`` cuando
            ``attempt < 2``). El valor 16 es un residuo histórico y
            no afecta el comportamiento observable.

        El docstring anterior ("exp backoff 1s, 4s, 16s, max 3
        attempts") era incorrecto: las esperas efectivas son 1 y 4
        segundos, no 1, 4 y 16. El cambio es de documentación
        únicamente.

        Si el job total excede ~30 minutos, el recovery hook lo
        detecta y re-enqueua (con el checkpoint de la última fase
        exitosa).
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

        DR-Q1A-PRE1A cost-reconciliation fix: this method now
        also PERSISTS the reconciled maximum back to
        ``research_jobs.cost_usd`` via
        ``_db.set_research_job_cost_monotonic`` (atomic
        ``MAX(cost_usd, reconciled)`` write) so that
        subsequent reads of the aggregate (e.g. from
        ``_phase_write`` or the completion notifier) observe
        the same reconciled value. Previously, the
        reconciliation was computed but not persisted, and
        the subsequent ``get_research_job_cost`` read could
        return a stale aggregate when the checkpoint or the
        token-usage sum legitimately exceeded it.

        The method is idempotent: re-running it with the
        same three sources returns the same persisted
        value (the aggregate is monotonically non-decreasing,
        so a second run cannot lower the first run's result).
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

        # Persist the reconciled maximum back to the aggregate.
        # The DB op is atomic MAX(cost_usd, reconciled) so this
        # is monotonic and idempotent: a second call cannot
        # lower the value set by the first. The method returns
        # the post-update aggregate value, which is the value
        # the completion notifier and ``JobDetail.cost_usd``
        # should expose.
        persisted = await self._db.set_research_job_cost_monotonic(
            job_id, float(reconciled)
        )
        return Decimal(str(persisted))

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
        """Actualiza current_phase (+progress opcional) en DB.

        DR-Q1A-PRE1B: phase updates are CONDITIONAL on the row
        still being in 'running'. A cancellation that wins
        between phases flips the row to 'cancelling'; the next
        phase's ``_update_phase`` call then sees a non-running
        status and refuses to write the phase. This is the
        phase-guard invariant: a cancelled job cannot advance to
        the next phase.
        """
        if progress is None:
            progress = 0
        applied = await self._db.update_research_job_phase(
            job_id,
            phase.value,
            int(progress),
        )
        if not applied:
            logger.info(
                "phase_update_skipped_status_changed",
                extra={"job_id": job_id, "phase": phase.value, "progress": progress},
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
