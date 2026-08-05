"""Custom exceptions del dominio research jobs.

Ver TDD_S14_DEEP_RESEARCH.md §10.3. Separadas por HTTP status code que
mapean (404 / 409 / 429 / 503). PhaseError es interno al service.

Slice 1C2: añadidas 5 excepciones para el read path de
``LocalReportStore`` (``report_paths`` / ``report_store``). Cada una
lleva ``job_id`` (str, sin path) y un atributo específico que el route
HTTP traduce a 500 ``report_unavailable``. El route no expone el tipo
interno al cliente.
"""

from __future__ import annotations

from hermes.jobs.models import JobStatus


class JobNotFoundError(Exception):
    """Raised when job_id doesn't exist in DB. HTTP 404."""


class JobAlreadyTerminalError(Exception):
    """Raised when trying to cancel a job that's already complete/failed/cancelled. HTTP 409."""

    def __init__(self, status: JobStatus) -> None:
        self.status = status
        super().__init__(f"Job is {status.value}, cannot cancel")


class JobNotRetryableError(Exception):
    """Raised when trying to retry a job that's not in 'failed' state. HTTP 409."""

    def __init__(self, status: JobStatus) -> None:
        self.status = status
        super().__init__(f"Job is {status.value}, only failed jobs can be retried")


class BudgetExceededError(Exception):
    """Raised when daily budget cap is reached. HTTP 429."""


class SchedulerUnavailableError(Exception):
    """Raised when DeepResearchScheduler.start() hasn't completed yet. HTTP 503."""


class SchedulerEnqueueError(Exception):
    """Raised when ``enqueue`` could not persist the job to the jobstore.

    Distinct from ``SchedulerUnavailableError`` (which is a
    transient "try later" condition). ``SchedulerEnqueueError`` is
    a "this enqueue will not succeed unless the operator
    intervenes" condition: the jobstore refused the row, the
    serialization failed, or the post-add lookup returned ``None``.

    The HTTP layer maps this to a 503 (NOT a 201). The submit
    path compensates the row in the DB to a terminal ``failed``
    state so the row does not stay in ``pending`` indefinitely.
    """


class JobStateInvalid(Exception):
    """Internal exception: a phase guard observed a row in an unexpected
    non-terminal state that prevents the phase from continuing safely.

    DR-Q1A-PRE1B remediation. Raised by ``_update_phase`` when the
    conditional ``update_research_job_phase`` returns False AND the row
    is in neither ``cancelling`` nor ``cancelled``. The research task
    treats this as an internal invariant violation: a recovery re-run
    can reset the row to ``pending`` on the next startup, but the
    current run must NOT continue into the phase body. The exception
    is caught by the generic ``except Exception`` branch in
    ``_run_research_inner`` and a conditional ``running -> failed``
    transition is attempted (which will fail if the row is not in
    ``running``; the task simply exits in that case).
    """

    def __init__(self, job_id: str, observed_status: str, phase: str) -> None:
        self.job_id = job_id
        self.observed_status = observed_status
        self.phase = phase
        super().__init__(
            f"job {job_id} phase {phase!r} guard failed; "
            f"row is in unexpected state {observed_status!r}"
        )


class PhaseError(Exception):
    """Internal error durante ejecución de una phase.

    Args:
        taxonomy: ErrorTaxonomy value (string). Determina la taxonomía
            amplia persistida (search_4xx, search_5xx, timeout, network,
            llm_5xx, llm_4xx, etc.). El nombre de la taxonomía ya NO
            decide el retry: el campo ``retryable`` es autoritativo
            desde PRE2-A1.
        message: human-readable detalle. PRE2-A1: cuando viene de un
            search failure, debe ser un mensaje estático seguro
            (sin ``str(exc)``).
        retryable: PRE2-A1. Si False, el retry loop NO reintenta esta
            phase. Si True, se aplican los retries acotados (3 intentos
            totales, backoff efectivo 1s y 4s).
        search_error_code: PRE2-A1. SearchErrorCode.value cuando el
            error viene del search router. None en otros casos.
        search_backend: PRE2-A1. Backend que produjo el search failure.
        search_breaker_relevant: PRE2-A1. Indica si el error de search
            debe contar contra el circuit breaker del backend.
        search_http_status: PRE2-A1. HTTP status code del search
            failure si estaba disponible.
        search_diagnostic_category: PRE2-A1. SearchDiagnosticCategory
            value del search failure (estable y finita).

    PRE2-A1 note: el retry loop (``_run_phase_with_retry``) decide
    basado en ``retryable``, NO en pertenencia a un set de taxonomías.
    Esto permite reportar search_4xx con retryable=True (caso 429) y
    search_5xx con retryable=False (caso específico) sin contradicción
    con la taxonomía persistida.
    """

    def __init__(
        self,
        taxonomy: str,
        message: str,
        retryable: bool = True,
        *,
        search_error_code: str | None = None,
        search_backend: str | None = None,
        search_breaker_relevant: bool | None = None,
        search_http_status: int | None = None,
        search_diagnostic_category: str | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.message = message
        self.retryable = retryable
        # PRE2-A1: in-memory structured bridge from search failure
        # to phase error. Not persisted; the broad taxonomy is the
        # only persisted identifier.
        self.search_error_code = search_error_code
        self.search_backend = search_backend
        self.search_breaker_relevant = search_breaker_relevant
        self.search_http_status = search_http_status
        self.search_diagnostic_category = search_diagnostic_category
        super().__init__(f"{taxonomy}: {message}")


# =====================================================================
# Slice 1C2: report-content retrieval exceptions
# =====================================================================
# These are raised by ``hermes.jobs.report_paths`` and
# ``hermes.jobs.report_store`` while reading a job's report. The HTTP
# route in ``hermes.receivers.jobs_api`` translates ALL of them into
# 500 ``report_unavailable`` (a single response shape) — the exception
# class is for internal classification only. Logs may distinguish the
# stable internal category:
#   - report_invalid_utf8   (InvalidUTF8Error)
#   - report_size_limit_exceeded (ReportTooLargeError)
#   - report_symlink_denied (SymlinkEscapeError)
#   - report_path_escape    (PathEscapeError)
#   - report_invalid_job_id (InvalidJobIdError)
# Public responses never expose filesystem paths, byte limits, decoder
# text, raw exceptions, symlink targets, or OS details.


class InvalidJobIdError(Exception):
    """Raised when the job_id fails validation (UUID12 hex, no NUL, no '..')."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"invalid_job_id:{job_id!r}")


class PathEscapeError(Exception):
    """Raised when the canonical path resolves outside the configured root."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"path_escape:{job_id!r}")


class SymlinkEscapeError(Exception):
    """Raised when a symlink target resolves outside the configured root."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"symlink_escape:{job_id!r}")


class ReportTooLargeError(Exception):
    """Raised when the report's size exceeds the configured max_bytes."""

    def __init__(self, job_id: str, size_bytes: int) -> None:
        self.job_id = job_id
        self.size_bytes = int(size_bytes)
        super().__init__(f"report_too_large:{job_id!r}")


class InvalidUTF8Error(Exception):
    """Raised when the report bytes are not valid UTF-8."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"invalid_utf8:{job_id!r}")


# Public mapping used by the route to decide the HTTP status. Keep the
# set closed — only the report-content read path may classify these
# into the single 500 ``report_unavailable`` envelope. The route is
# the SOLE place that decides 200 vs 500 for the report path.
REPORT_UNAVAILABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    InvalidJobIdError,
    PathEscapeError,
    SymlinkEscapeError,
    ReportTooLargeError,
    InvalidUTF8Error,
)
