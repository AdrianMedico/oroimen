"""Deep Research jobs (Sprint 14, ÉPICA 2).

Subpaquete con:
- models: Pydantic models (request/response + enums)
- exceptions: custom exceptions del dominio
- cost: PRICING_TABLE, calculate_cost, format_now helpers
- prompts: PER_SOURCE_PROMPT + FINAL_SYNTH_PROMPT + sanitize_summary
- scheduler: AsyncIOScheduler wrapper con SQLAlchemyJobStore persistent
- recovery: recovery hook para reconciliación post-restart
- service: DeepResearchService con 5-phase pipeline
"""

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
    DailyBudgetStatus,
    ErrorTaxonomy,
    JobDetail,
    JobResponse,
    JobStatus,
    JobSummary,
    JobType,
    PhaseName,
    TokenUsageEntry,
)

__all__ = [
    "BudgetExceededError",
    "CancelResponse",
    "CreateJobRequest",
    "DailyBudgetStatus",
    "ErrorTaxonomy",
    "JobAlreadyTerminalError",
    "JobDetail",
    "JobNotFoundError",
    "JobNotRetryableError",
    "JobResponse",
    "JobStatus",
    "JobSummary",
    "JobType",
    "PhaseError",
    "PhaseName",
    "SchedulerUnavailableError",
    "TokenUsageEntry",
]
