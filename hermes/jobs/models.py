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
    estimated_cost_usd: float = Field(
        ...,
        description=(
            "Estimated official pay-as-you-go-equivalent amount at the "
            "verified PRICING_TABLE rates (PRICING_BASIS = "
            "'official_paygo_equivalent'); not actual provider billing, "
            "not actual subscription or quota-plan spend, not remaining "
            "account balance, not invoice truth. Operators using a "
            "subscription or quota-backed plan should treat this value "
            "as a relative cost proxy, not as a spend figure. The "
            "estimate is a pre-submit heuristic (per-source + final "
            "synth + 1.30 safety margin) at the official standard tier; "
            "actual billed tokens may differ if the operator's plan "
            "uses a different rate or if a provider call times out "
            "without returning."
        ),
    )


class JobSummary(BaseModel):
    """GET /v1/jobs list item."""

    id: str
    query: str
    status: JobStatus
    current_phase: PhaseName | None = None
    progress_percent: int
    cost_usd: float = Field(
        ...,
        description=(
            "Accumulated estimated pay-as-you-go-equivalent amount "
            "derived from the recorded token_usage rows at the verified "
            "PRICING_TABLE rates (PRICING_BASIS = 'official_paygo_equivalent'). "
            "Not actual provider billing. The aggregate is the sum of "
            "the per-call cost_usd entries in token_usage. The "
            "RECORDED token usage may understate provider-billed usage "
            "when a dispatched call times out without returning a "
            "response; in that case the recorded usage row is missing "
            "and the corresponding cost_usd contribution is also "
            "missing. The cost_usd field may therefore understate the "
            "official paygo-equivalent cost of dispatched calls. Actual "
            "provider billing remains unknown and is not represented "
            "by cost_usd."
        ),
    )
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class JobDetail(JobSummary):
    """GET /v1/jobs/{id} response — extiende JobSummary con más campos.

    Slice 1C2: ``output_path`` / ``partial_output_path`` / ``checkpoint_path``
    have been REMOVED from the public DTO. The internal DB columns stay
    (no migration in 1C2) but the HTTP response no longer leaks filesystem
    paths. Clients that previously called ``GET /v1/jobs/{id}`` and read
    ``output_path`` should now call ``GET /v1/jobs/{id}/report`` and
    interpret the HTTP status code per the Slice 1C2 contract.

    NO ``report_available`` flag is added. The status field is the
    source of truth for "is the report ready".

    NOTE (DR-Q1A-PRE1A): ``cost_usd`` is the estimated
    pay-as-you-go-equivalent amount (see ``cost_usd`` description on
    ``JobSummary``). It is NOT automatically embedded in the final
    Markdown report returned by ``GET /v1/jobs/{id}/report``; the
    report body is the LLM-generated content only.
    """

    job_type: JobType
    notify_via_tg: bool
    error_taxonomy: ErrorTaxonomy | None = None
    error_message: str | None = None
    tokens_in: int
    tokens_out: int
    notified: bool
    updated_at: str
    token_usage: list[TokenUsageEntry] = Field(default_factory=list)


class CancelResponse(BaseModel):
    """POST /v1/jobs/{id}/cancel response.

    Slice 1C2: ``partial_output_path`` REMOVED. The DB column stays (no
    migration in 1C2) but the field was always ``None`` in the wire
    response because every service path hardcoded it. The
    ``graceful`` / ``status`` semantics are preserved unchanged.

    DR-Q1A-PRE1A truth: ``graceful`` describes the persistence
    transition requested, not what was actually stored or whether
    any task or provider request was cancelled. Neither value proves
    running-task cancellation. Neither value implies that a partial
    output was stored. See ``hermes/jobs/service.py::cancel_job`` for
    the actual current behavior.
    """

    id: str
    status: JobStatus = Field(
        ...,
        description=(
            "``cancelling`` when graceful=True (the cancelling "
            "transition was applied; the persistence will be moved "
            "to ``cancelled`` the next time the running task polls). "
            "``cancelled`` when graceful=False (the persistence was "
            "moved directly to ``cancelled``). In both cases the "
            "current code does not prove cancellation of the running "
            "asyncio task or any in-flight provider request."
        ),
    )
    graceful: bool = Field(
        ...,
        description=(
            "``True``: the cancelling transition was applied "
            "(persistence marked 'cancelling'); the response returns "
            "immediately without waiting. ``False``: the persistence "
            "was moved directly to 'cancelled'. Neither value proves "
            "that the running asyncio task was cancelled. Neither "
            "value proves that any in-flight provider request was "
            "cancelled. Neither value implies anything about whether "
            "an intermediate run result was stored."
        ),
    )


class TokenUsageEntry(BaseModel):
    """Drill-down per-LLM-call (sub-entry de JobDetail.token_usage)."""

    phase: PhaseName
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float = Field(
        ...,
        description=(
            "Per-returned-call estimated pay-as-you-go-equivalent "
            "amount at the verified PRICING_TABLE rates "
            "(PRICING_BASIS = 'official_paygo_equivalent'). Computed "
            "from the tokens_in / tokens_out the client observed. "
            "Calls that time out without returning a response are NOT "
            "represented in this table: the corresponding RECORDED "
            "token usage is missing, and the corresponding cost_usd "
            "is therefore also missing. The cost_usd field may "
            "understate the official paygo-equivalent cost of "
            "dispatched calls. Actual provider billing remains "
            "unknown and is not represented by cost_usd."
        ),
    )
    created_at: str


class DailyBudgetStatus(BaseModel):
    """GET /v1/jobs/budget (helper endpoint, no en §10 HTTP API pero útil)."""

    today_cost_usd: float = Field(
        ...,
        description=(
            "Sum of estimated pay-as-you-go-equivalent cost_usd across "
            "all jobs created today (UTC), at the verified PRICING_TABLE "
            "rates (PRICING_BASIS = 'official_paygo_equivalent'). "
            "Internal admission-control value, NOT the operator's "
            "actual account balance."
        ),
    )
    daily_cap_usd: float = Field(
        ...,
        description=(
            "Configured daily spend cap (settings.deep_research_daily_budget_usd). "
            "Internal admission-control value, NOT a hard provider-side "
            "monetary limit; the per-job cap is a soft warning and the "
            "daily cap is a pre-submit check (the cap is re-checked at "
            "job start, but the service does not cancel running jobs)."
        ),
    )
    remaining_usd: float = Field(
        ...,
        description=(
            "``daily_cap_usd - today_cost_usd`` at admission time. "
            "Internal admission-control value, NOT a hard provider-side "
            "monetary limit."
        ),
    )
    jobs_today: int
    resets_at: str  # ISO8601, próximo 00:00 UTC


# Resolver forward-ref de JobDetail.token_usage -> TokenUsageEntry
JobDetail.model_rebuild()
