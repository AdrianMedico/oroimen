"""Unit tests for hermes.jobs.models — Pydantic validation.

Anti-regression checks (TDD §1.5 / §3):
- Empty/whitespace-only query → ValidationError
- Query > 2000 chars → ValidationError
- All enums (JobStatus, JobType, PhaseName, ErrorTaxonomy) instantiate.

Verifies also that `_strip_query` validator trims whitespace.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hermes.jobs.models import (
    CreateJobRequest,
    ErrorTaxonomy,
    JobStatus,
    JobType,
    PhaseName,
)


def test_create_job_request_empty_query() -> None:
    """Empty query string fails validation (min_length=3)."""
    with pytest.raises(ValidationError) as exc_info:
        CreateJobRequest(query="")
    # Verify the error is on the query field
    errors = exc_info.value.errors()
    assert any(
        err["loc"] == ("query",) for err in errors
    ), f"Expected error on 'query' field, got: {errors}"


def test_create_job_request_query_too_long() -> None:
    """Query > 2000 chars fails validation (max_length=2000)."""
    long_query = "x" * 2001
    with pytest.raises(ValidationError) as exc_info:
        CreateJobRequest(query=long_query)
    errors = exc_info.value.errors()
    assert any(
        err["loc"] == ("query",) for err in errors
    ), f"Expected error on 'query' field for 2001-char query, got: {errors}"


def test_create_job_request_strips_whitespace() -> None:
    """Strip validator: '  hi  ' becomes 'hi' (passes min_length)."""
    req = CreateJobRequest(query="  hello world  ")
    assert req.query == "hello world"


def test_create_job_request_defaults() -> None:
    """Defaults: job_type=DEEP_RESEARCH, notify_via_tg=True."""
    req = CreateJobRequest(query="test query")
    assert req.job_type == JobType.DEEP_RESEARCH
    assert req.notify_via_tg is True


def test_all_enums_instantiate() -> None:
    """Smoke check: every enum value in §3 enumerations exists."""
    # JobStatus: 6 states
    assert {s.value for s in JobStatus} == {
        "pending",
        "running",
        "complete",
        "failed",
        "cancelling",
        "cancelled",
    }
    # JobType: S14 implements deep_research only
    assert JobType.DEEP_RESEARCH.value == "deep_research"
    # PhaseName: 5 phases
    assert {p.value for p in PhaseName} == {
        "search",
        "scrape",
        "per_source_synthesis",
        "final_synthesis",
        "write",
    }
    # ErrorTaxonomy: 10 categories
    assert {e.value for e in ErrorTaxonomy} == {
        "search_5xx",
        "search_4xx",
        "llm_5xx",
        "llm_4xx",
        "timeout",
        "cancelled",
        "budget_exceeded",
        "oom",
        "network",
        "checkpoint_corrupt",
    }
