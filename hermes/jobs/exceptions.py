"""Custom exceptions del dominio research jobs.

Ver TDD_S14_DEEP_RESEARCH.md §10.3. Separadas por HTTP status code que
mapean (404 / 409 / 429 / 503). PhaseError es interno al service.
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


class PhaseError(Exception):
    """Internal error durante ejecución de una phase.

    Args:
        taxonomy: ErrorTaxonomy value (string). Determina retry behavior
            (RETRYABLE_ERRORS set en service.py).
        message: human-readable detalle.
        retryable: si False, el retry loop NO reintenta esta phase.
    """

    def __init__(self, taxonomy: str, message: str, retryable: bool = True) -> None:
        self.taxonomy = taxonomy
        self.message = message
        self.retryable = retryable
        super().__init__(f"{taxonomy}: {message}")
