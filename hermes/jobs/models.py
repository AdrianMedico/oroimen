"""Pydantic models para el dominio research jobs.

Ver TDD_S14_DEEP_RESEARCH.md §3. Validación centralizada: HTTP API,
scheduler, service y tests consumen estos modelos. NO dicts anónimos.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class JobStatus(StrEnum):
    """State machine del job. Ver TDD §2.1 CHECK constraint."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class JobType(StrEnum):
    """Tipo de job. S14 implementa solo DEEP_RESEARCH; S15 añadirá más."""

    DEEP_RESEARCH = "deep_research"  # S14
    REMINDER = "reminder"  # S15 US-2.3
    EMBED_VAULT = "embed_vault"  # S15 US-2.4


class PhaseName(StrEnum):
    """Las 5 phases del pipeline."""

    SEARCH = "search"  # Phase 1: Tavily search
    SCRAPE = "scrape"  # Phase 2: HTTP fetch + html_to_text
    PER_SOURCE_SYNTHESIS = "per_source_synthesis"  # Phase 3: per-source LLM
    FINAL_SYNTHESIS = "final_synthesis"  # Phase 4: final LLM
    WRITE = "write"  # Phase 5: write .md


class ErrorTaxonomy(StrEnum):
    """Taxonomía de errores (TDD §2.1 CHECK constraint)."""

    SEARCH_5XX = "search_5xx"
    SEARCH_4XX = "search_4xx"
    LLM_5XX = "llm_5xx"
    LLM_4XX = "llm_4xx"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    OOM = "oom"
    NETWORK = "network"
    CHECKPOINT_CORRUPT = "checkpoint_corrupt"


class CreateJobRequest(BaseModel):
    """POST /v1/jobs body."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Research query, natural language",
    )
    job_type: JobType = Field(default=JobType.DEEP_RESEARCH)
    notify_via_tg: bool = Field(
        default=True,
        description="Opt-in TG push al completar (default retrocompat S10.4).",
    )

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        """Strip whitespace; queries con solo whitespace → falla en min_length."""
        return v.strip()


class JobResponse(BaseModel):
    """POST /v1/jobs response (201 Created)."""

    id: str = Field(..., description="UUID 12-char hex")
    status: JobStatus
    created_at: str  # formato 'YYYY-MM-DD HH:MM:SS.sss'
    estimated_cost_usd: float = Field(..., description="Heurística previa, NO garantía")


class JobSummary(BaseModel):
    """GET /v1/jobs list item."""

    id: str
    query: str
    status: JobStatus
    current_phase: PhaseName | None = None
    progress_percent: int
    cost_usd: float
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class JobDetail(JobSummary):
    """GET /v1/jobs/{id} response — extiende JobSummary con más campos."""

    job_type: JobType
    notify_via_tg: bool
    output_path: str | None = None
    partial_output_path: str | None = None
    error_taxonomy: ErrorTaxonomy | None = None
    error_message: str | None = None
    tokens_in: int
    tokens_out: int
    notified: bool
    updated_at: str
    token_usage: list[TokenUsageEntry] = Field(default_factory=list)
    checkpoint_path: str | None = Field(None, description="Path al .json si existe")


class CancelResponse(BaseModel):
    """POST /v1/jobs/{id}/cancel response."""

    id: str
    status: JobStatus  # 'cancelling' o 'cancelled' si ya estaba finished
    graceful: bool  # True si partial output guardado, False si hard cancel
    partial_output_path: str | None = None


class TokenUsageEntry(BaseModel):
    """Drill-down per-LLM-call (sub-entry de JobDetail.token_usage)."""

    phase: PhaseName
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    created_at: str


class DailyBudgetStatus(BaseModel):
    """GET /v1/jobs/budget (helper endpoint, no en §10 HTTP API pero útil)."""

    today_cost_usd: float
    daily_cap_usd: float
    remaining_usd: float
    jobs_today: int
    resets_at: str  # ISO8601, próximo 00:00 UTC


# Resolver forward-ref de JobDetail.token_usage -> TokenUsageEntry
JobDetail.model_rebuild()
