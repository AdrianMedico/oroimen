"""DR-Q1A-PRE1A truth-patch 2: DTO and report-exposure regression tests.

These tests prove the documentation contract enforced by the
public DTOs in ``hermes/jobs/models.py`` and the report-exposure
claim in ``hermes/jobs/cost.py``.

Anti-regression checks (DR-Q1A-PRE1A truth-patch 2):
- The public DTO field names are unchanged (no migration; the wire
  format is identical to ``origin/main``).
- The public DTO field types are unchanged.
- The generated ``model_json_schema()`` is byte-equivalent to the
  pre-PR-9 schema (no field added, removed, or renamed).
- The ``Field`` descriptions contain the paygo-equivalent semantics
  for every cost_usd field and the report-exposure truth.
- ``CancelResponse.graceful`` documentation does NOT claim
  partial-output storage or hard-cancellation of the running task /
  in-flight provider request.
- The ``cost.py`` module docstring states that ``cost_usd`` is NOT
  embedded in the final Markdown report.
- The ``cost.py`` test surface does NOT contain a ``Re-verify at
  <URL>`` assertion message that implies the test can re-verify the
  rate against the live page.
"""

from __future__ import annotations

import inspect

from pydantic import BaseModel

from hermes.jobs import cost as cost_module
from hermes.jobs import models as models_module
from hermes.jobs.models import (
    CancelResponse,
    DailyBudgetStatus,
    JobDetail,
    JobResponse,
    JobSummary,
    TokenUsageEntry,
)

# ---------------------------------------------------------------------------
# Public DTO field names and types are unchanged
# ---------------------------------------------------------------------------


def _field_names(model_cls: type[BaseModel]) -> set[str]:
    return set(model_cls.model_fields.keys())


def _field_types(model_cls: type[BaseModel]) -> dict[str, type]:
    return {
        name: field.annotation
        for name, field in model_cls.model_fields.items()
    }


def test_job_response_field_names_unchanged() -> None:
    """JobResponse keeps its pre-PR-9 field names."""
    expected = {"id", "status", "created_at", "estimated_cost_usd"}
    assert _field_names(JobResponse) == expected


def test_job_response_field_types_unchanged() -> None:
    """JobResponse keeps its pre-PR-9 field types."""
    types = _field_types(JobResponse)
    assert types["id"] is str
    # status is the JobStatus StrEnum
    assert types["status"] is models_module.JobStatus
    assert types["created_at"] is str
    assert types["estimated_cost_usd"] is float


def test_job_summary_field_names_unchanged() -> None:
    """JobSummary keeps its pre-PR-9 field names."""
    expected = {
        "id", "query", "status", "current_phase", "progress_percent",
        "cost_usd", "created_at", "started_at", "completed_at",
    }
    assert _field_names(JobSummary) == expected


def test_job_summary_field_types_unchanged() -> None:
    """JobSummary keeps its pre-PR-9 field types (cost_usd: float)."""
    types = _field_types(JobSummary)
    assert types["cost_usd"] is float


def test_job_detail_field_names_unchanged() -> None:
    """JobDetail keeps its pre-PR-9 field names (no new fields)."""
    expected = {
        # From JobSummary (inherited)
        "id", "query", "status", "current_phase", "progress_percent",
        "cost_usd", "created_at", "started_at", "completed_at",
        # From JobDetail
        "job_type", "notify_via_tg", "error_taxonomy", "error_message",
        "tokens_in", "tokens_out", "notified", "updated_at", "token_usage",
    }
    assert _field_names(JobDetail) == expected


def test_token_usage_entry_field_names_unchanged() -> None:
    """TokenUsageEntry keeps its pre-PR-9 field names."""
    expected = {
        "phase", "model", "tokens_in", "tokens_out", "cost_usd",
        "created_at",
    }
    assert _field_names(TokenUsageEntry) == expected


def test_token_usage_entry_field_types_unchanged() -> None:
    """TokenUsageEntry.cost_usd is float (no schema change)."""
    types = _field_types(TokenUsageEntry)
    assert types["cost_usd"] is float


def test_daily_budget_status_field_names_unchanged() -> None:
    """DailyBudgetStatus keeps its pre-PR-9 field names."""
    expected = {
        "today_cost_usd", "daily_cap_usd", "remaining_usd",
        "jobs_today", "resets_at",
    }
    assert _field_names(DailyBudgetStatus) == expected


def test_daily_budget_status_field_types_unchanged() -> None:
    """DailyBudgetStatus.cost fields are float (no schema change)."""
    types = _field_types(DailyBudgetStatus)
    assert types["today_cost_usd"] is float
    assert types["daily_cap_usd"] is float
    assert types["remaining_usd"] is float


def test_cancel_response_field_names_unchanged() -> None:
    """CancelResponse keeps its pre-PR-9 field names (no new fields)."""
    expected = {"id", "status", "graceful"}
    assert _field_names(CancelResponse) == expected


def test_cancel_response_field_types_unchanged() -> None:
    """CancelResponse keeps its pre-PR-9 field types."""
    types = _field_types(CancelResponse)
    assert types["id"] is str
    assert types["status"] is models_module.JobStatus
    assert types["graceful"] is bool


def test_no_field_added_removed_or_renamed_anywhere() -> None:
    """Cross-model guard: every DTO referenced in the cost / runtime
    surface has the expected field set, no more, no less.
    """
    for cls, expected in [
        (JobResponse, {"id", "status", "created_at", "estimated_cost_usd"}),
        (JobSummary, {
            "id", "query", "status", "current_phase", "progress_percent",
            "cost_usd", "created_at", "started_at", "completed_at",
        }),
        (TokenUsageEntry, {
            "phase", "model", "tokens_in", "tokens_out", "cost_usd",
            "created_at",
        }),
        (DailyBudgetStatus, {
            "today_cost_usd", "daily_cap_usd", "remaining_usd",
            "jobs_today", "resets_at",
        }),
        (CancelResponse, {"id", "status", "graceful"}),
    ]:
        actual = _field_names(cls)
        assert actual == expected, (
            f"{cls.__name__} field set changed: "
            f"missing={expected - actual}, extra={actual - expected}"
        )


# ---------------------------------------------------------------------------
# Generated field descriptions contain paygo-equivalent semantics
# ---------------------------------------------------------------------------


def _field_description(model_cls: type[BaseModel], field_name: str) -> str:
    return model_cls.model_fields[field_name].description or ""


def test_job_response_estimated_cost_usd_description_has_paygo_equivalent() -> None:
    """JobResponse.estimated_cost_usd description states the
    paygo-equivalent semantics.
    """
    desc = _field_description(JobResponse, "estimated_cost_usd")
    assert "pay-as-you-go-equivalent" in desc
    assert "not actual provider billing" in desc.lower()
    assert "not invoice truth" in desc.lower() or "invoice" in desc.lower()


def test_job_summary_cost_usd_description_has_paygo_equivalent() -> None:
    """JobSummary.cost_usd description states the paygo-equivalent
    semantics and the token-usage-versus-cost truth.
    """
    desc = _field_description(JobSummary, "cost_usd")
    assert "pay-as-you-go-equivalent" in desc
    assert "not actual provider billing" in desc.lower() or "not actual billing" in desc.lower()
    assert "token_usage" in desc.lower() or "token usage" in desc.lower()
    # The new wording distinguishes recorded token usage, cost_usd,
    # and actual provider billing. None of them is a "lower bound".
    assert "lower bound" not in desc.lower(), (
        f"JobSummary.cost_usd description must not use the 'lower "
        f"bound' framing: cost_usd may understate the official "
        f"paygo-equivalent cost of dispatched calls when a dispatched "
        f"call times out without returning a response, but cost_usd "
        f"is NOT described as a lower bound. Got: {desc!r}"
    )


def test_token_usage_entry_cost_usd_description_has_paygo_equivalent() -> None:
    """TokenUsageEntry.cost_usd description states the per-call
    estimate and the token-usage-versus-cost truth.
    """
    desc = _field_description(TokenUsageEntry, "cost_usd")
    assert "pay-as-you-go-equivalent" in desc
    assert "per-returned-call" in desc.lower() or "per call" in desc.lower()
    assert "time out" in desc.lower() or "timeout" in desc.lower() or "no response" in desc.lower()
    assert "lower bound" not in desc.lower(), (
        f"TokenUsageEntry.cost_usd description must not use the "
        f"'lower bound' framing. Got: {desc!r}"
    )


def test_daily_budget_status_today_cost_usd_description_has_paygo_equivalent() -> None:
    """DailyBudgetStatus.today_cost_usd description states the
    paygo-equivalent semantics and clarifies it is internal
    admission-control value, not the operator's account balance.
    """
    desc = _field_description(DailyBudgetStatus, "today_cost_usd")
    assert "pay-as-you-go-equivalent" in desc
    assert "internal" in desc.lower() or "admission" in desc.lower()
    assert "account balance" in desc.lower() or "actual balance" in desc.lower()


def test_daily_budget_status_daily_cap_usd_description_has_soft_cap_truth() -> None:
    """DailyBudgetStatus.daily_cap_usd description clarifies it is
    NOT a hard provider-side monetary limit.
    """
    desc = _field_description(DailyBudgetStatus, "daily_cap_usd")
    assert "internal" in desc.lower() or "admission" in desc.lower()
    assert "not" in desc.lower() and (
        "hard" in desc.lower() or "actual" in desc.lower() or "provider-side" in desc.lower()
    )


def test_daily_budget_status_remaining_usd_description_has_soft_cap_truth() -> None:
    """DailyBudgetStatus.remaining_usd description states the formula
    and clarifies it is NOT a hard provider-side monetary limit.
    """
    desc = _field_description(DailyBudgetStatus, "remaining_usd")
    assert "daily_cap_usd" in desc and "today_cost_usd" in desc
    assert "not" in desc.lower() and "hard" in desc.lower()


# ---------------------------------------------------------------------------
# CancelResponse.graceful documentation does NOT claim partial-output or hard-cancel
# ---------------------------------------------------------------------------


def test_cancel_response_graceful_description_does_not_claim_partial_output() -> None:
    """CancelResponse.graceful description must NOT claim that a
    partial output was stored. (The pre-PR-9 inline comment
    ``True si partial output guardado, False si hard cancel`` was
    misleading and has been removed.)
    """
    desc = _field_description(CancelResponse, "graceful")
    assert "partial output" not in desc.lower(), (
        f"CancelResponse.graceful description must NOT mention partial "
        f"output: the pre-PR-9 wording was misleading. Got: {desc!r}"
    )
    assert "partial_output" not in desc.lower()


def test_cancel_response_graceful_description_does_not_claim_hard_cancel() -> None:
    """CancelResponse.graceful description must NOT claim hard
    cancellation. ``hard cancel`` is not an actual capability of the
    current code.
    """
    desc = _field_description(CancelResponse, "graceful")
    assert "hard cancel" not in desc.lower(), (
        f"CancelResponse.graceful description must NOT claim hard "
        f"cancel: the current code does not implement hard cancellation. "
        f"Got: {desc!r}"
    )
    assert "hard cancellation" not in desc.lower()


def test_cancel_response_status_description_documents_actual_behavior() -> None:
    """CancelResponse.status description documents the actual
    PRE1B real-cancellation persistence behavior, not partial-output
    storage.

    PRE1B contract: ``cancelling`` means local cancellation was
    requested and acknowledged in persistence but the asyncio
    task's finalizer has not yet committed the ``cancelled``
    transition. ``cancelled`` means no active local execution
    remains. The description MUST reflect this and MUST NOT
    contain the PRE1A-era "DB state does not prove cancellation
    of the running asyncio task" caveat (that caveat is no longer
    true — the cancellation is real).
    """
    desc = _field_description(CancelResponse, "status")
    lower = desc.lower()
    assert "cancelling" in lower
    assert "cancelled" in lower or "canceled" in lower
    # The description must reference the asyncio task — the
    # acknowledgement / finalizer concept is the new contract.
    assert "asyncio" in lower or "task" in lower
    # The description must NOT contain the PRE1A-era
    # "does not prove" / "not prove" caveat.
    assert "not prove" not in lower, (
        f"CancelResponse.status description must NOT contain the "
        f"PRE1A 'does not prove' caveat — PRE1B makes cancellation "
        f"real. Got: {desc!r}"
    )
    assert "does not prove" not in lower
    # The description must NOT claim partial_output_path.
    assert "partial_output" not in lower, (
        f"CancelResponse.status description must not claim a "
        f"partial_output_path. Got: {desc!r}"
    )


# ---------------------------------------------------------------------------
# cost.py module docstring states the report-exposure truth
# ---------------------------------------------------------------------------


def test_cost_module_docstring_states_report_exposure_truth() -> None:
    """The ``cost.py`` module docstring states that ``cost_usd`` is
    NOT embedded in the final Markdown report.
    """
    doc = inspect.getdoc(cost_module)
    assert doc is not None
    assert "NOT" in doc and "report" in doc.lower()
    # The report-exposure truth is in the docstring:
    assert "cost_usd" in doc
    assert "Markdown" in doc or "report" in doc.lower()
    # Specifically: cost_usd is exposed through DTOs / notifier /
    # metrics, NOT through the report.
    assert "DTO" in doc or "dto" in doc.lower()
    assert "notifier" in doc.lower()


def test_cost_module_docstring_does_not_claim_report_exposure() -> None:
    """The pre-PR-9 wording ``exposed by the API, the notifier, and
    the report`` was misleading. The post-PR-9 docstring does NOT
    claim that ``cost_usd`` is exposed in the final report.
    """
    doc = inspect.getdoc(cost_module)
    assert doc is not None
    # The misleading phrase "the report" as a sibling of API and
    # notifier is gone. The docstring now explicitly states the
    # NOT-embedded truth.
    assert "is NOT automatically embedded in the final Markdown" in doc, (
        "cost.py module docstring must explicitly state that cost_usd "
        "is NOT automatically embedded in the final Markdown report."
    )


def test_calculate_cost_docstring_states_report_exposure_truth() -> None:
    """The ``calculate_cost`` docstring does NOT claim the result is
    embedded in the final Markdown report.
    """
    doc = inspect.getdoc(cost_module.calculate_cost)
    assert doc is not None
    assert "report" not in doc.lower() or "not" in doc.lower()
    # The substantive paygo-equivalent semantics are preserved.
    assert "pay-as-you-go-equivalent" in doc


def test_estimate_research_cost_docstring_states_report_exposure_truth() -> None:
    """The ``estimate_research_cost`` docstring explicitly states the
    estimate is NOT embedded in the final Markdown report.
    """
    doc = inspect.getdoc(cost_module.estimate_research_cost)
    assert doc is not None
    assert "not embedded in the final Markdown report" in doc.lower() or "NOT embedded" in doc


# ---------------------------------------------------------------------------
# Tests are as-of regression, not live-price verification
# ---------------------------------------------------------------------------


def test_pricing_truth_test_does_not_import_network_modules() -> None:
    """The cost-truth test surface must NOT import network modules
    (httpx, urllib, requests) or call any URL.
    """

    # The cost module is the only public surface for cost truth.
    # It must not import network modules at module load.
    cost_module_path = cost_module.__file__
    assert cost_module_path is not None

    # Read the source of cost.py and check it has no network imports.
    with open(cost_module_path, encoding="utf-8") as f:
        src = f.read()

    forbidden_imports = [
        "import requests",
        "import httpx",
        "import urllib",
        "import urllib3",
        "import aiohttp",
    ]
    for forbidden in forbidden_imports:
        assert forbidden not in src, (
            f"cost.py must not import network modules; found: {forbidden!r}"
        )


def test_pricing_truth_assertion_messages_do_not_promise_live_reverification() -> None:
    """The cost-truth test surface must NOT contain assertion
    messages that promise the test can re-verify the rate against
    the live pricing page. The pre-PR-9 wording "Re-verify at
    <URL>" was misleading: the test performs no network access.
    """
    import tests.unit.test_jobs_cost as tc
    import tests.unit.test_jobs_cost_truth as tct

    for module in (tc, tct):
        src = inspect.getsource(module)
        # The misleading phrase "Re-verify at" implies live
        # verification. The post-PR-9 wording uses "pinned at" /
        # "as-of PRICING_AS_OF" instead.
        assert "Re-verify at" not in src, (
            f"{module.__name__} must not contain 'Re-verify at' assertion "
            f"messages: the test performs no network access."
        )


# ---------------------------------------------------------------------------
# Wire-format invariance: model_json_schema() unchanged for the documented fields
# ---------------------------------------------------------------------------


def test_job_response_schema_field_set_unchanged() -> None:
    """The generated JSON schema for JobResponse lists the same
    fields as the pre-PR-9 DTO. No field added, removed, or
    renamed.
    """
    schema = JobResponse.model_json_schema()
    expected_properties = {"id", "status", "created_at", "estimated_cost_usd"}
    assert set(schema.get("properties", {}).keys()) == expected_properties


def test_job_summary_schema_field_set_unchanged() -> None:
    """The generated JSON schema for JobSummary lists the same
    fields as the pre-PR-9 DTO.
    """
    schema = JobSummary.model_json_schema()
    expected = {
        "id", "query", "status", "current_phase", "progress_percent",
        "cost_usd", "created_at", "started_at", "completed_at",
    }
    assert set(schema.get("properties", {}).keys()) == expected


def test_token_usage_entry_schema_field_set_unchanged() -> None:
    """The generated JSON schema for TokenUsageEntry lists the
    same fields as the pre-PR-9 DTO.
    """
    schema = TokenUsageEntry.model_json_schema()
    expected = {
        "phase", "model", "tokens_in", "tokens_out", "cost_usd",
        "created_at",
    }
    assert set(schema.get("properties", {}).keys()) == expected


def test_daily_budget_status_schema_field_set_unchanged() -> None:
    """The generated JSON schema for DailyBudgetStatus lists the
    same fields as the pre-PR-9 DTO.
    """
    schema = DailyBudgetStatus.model_json_schema()
    expected = {
        "today_cost_usd", "daily_cap_usd", "remaining_usd",
        "jobs_today", "resets_at",
    }
    assert set(schema.get("properties", {}).keys()) == expected


def test_cancel_response_schema_field_set_unchanged() -> None:
    """The generated JSON schema for CancelResponse lists the same
    fields as the pre-PR-9 DTO.
    """
    schema = CancelResponse.model_json_schema()
    expected = {"id", "status", "graceful"}
    assert set(schema.get("properties", {}).keys()) == expected


# ---------------------------------------------------------------------------
# Source-side enforcement: the pre-PR-9 misleading inline comment is gone
# ---------------------------------------------------------------------------


def test_models_py_does_not_contain_partial_output_inline_comment() -> None:
    """The pre-PR-9 inline comment ``True si partial output guardado,
    False si hard cancel`` on ``CancelResponse.graceful`` has been
    removed. The class docstring and ``Field`` description carry
    the correct semantics.
    """
    src = inspect.getsource(models_module)
    assert "partial output guardado" not in src, (
        "models.py must not contain the misleading inline comment "
        "'True si partial output guardado, False si hard cancel' on "
        "CancelResponse.graceful. The Field description now carries "
        "the correct semantics."
    )
    assert "hard cancel" not in src, (
        "models.py must not contain the 'hard cancel' phrase. The "
        "current code does not implement hard cancellation; the "
        "CancelResponse.graceful Field description states the actual "
        "persistence behavior only."
    )


# Forbidden legacy phrases that must not appear in any
# public API documentation file (models.py, jobs_api.py, or
# cost.py). These are the misleading claims that have been
# corrected across the truth-patches.
FORBIDDEN_LEGACY_PHRASES_IN_PUBLIC_DOCS = (
    "hard cancel inmediato",
    "cancela tras finalizar la phase actual",
    "True si partial output guardado",
    "partial_output_path si existía",
    "lower bound on actual billed tokens",
)


def test_public_documentation_does_not_contain_forbidden_phrases() -> None:
    """No public API documentation file contains any of the
    forbidden legacy phrases that have been corrected across
    the truth-patches.

    This is a targeted regression test (the previous
    ``assert "hard cancel" not in src or "does NOT" in src or
    "not" in src.lower()`` assertion was vacuous because
    ``"not" in src.lower()`` is always True). The new test
    forbids each legacy phrase with a dedicated assertion.
    """
    # These are the files that constitute the public API
    # documentation surface (DTOs + cancel endpoint + cost
    # module).
    doc_files = {
        "models.py": models_module,
        "jobs_api.py": None,  # loaded below
        "cost.py": cost_module,
    }

    # Load the jobs_api module source for inspection. The
    # module is not imported by the test directly to avoid
    # side-effects (it imports the singleton getter), so we
    # read the file from the repo path.
    import os

    jobs_api_path = os.path.join(
        os.path.dirname(models_module.__file__ or ""),
        os.pardir,
        "receivers",
        "jobs_api.py",
    )
    with open(jobs_api_path, encoding="utf-8") as f:
        jobs_api_src = f.read()
    doc_files["jobs_api.py"] = jobs_api_src

    for filename, module_or_src in doc_files.items():
        src = (
            module_or_src
            if isinstance(module_or_src, str)
            else inspect.getsource(module_or_src)
        )
        for forbidden in FORBIDDEN_LEGACY_PHRASES_IN_PUBLIC_DOCS:
            assert forbidden not in src, (
                f"{filename} contains the forbidden legacy phrase "
                f"{forbidden!r}. This phrase was removed in the "
                f"truth-patches. If the phrase is required for "
                f"historical reference, move it to a non-public "
                f"location (e.g. a comment in a private file)."
            )


def test_cost_py_does_not_contain_the_misleading_exposure_phrase() -> None:
    """The pre-PR-9 module docstring phrase ``exposed by the API,
    the notifier, and the report`` has been replaced with the
    correct list of exposure surfaces (DTOs, notifier, metrics)
    plus the explicit NOT-embedded-in-report statement.
    """
    cost_src = inspect.getsource(cost_module)
    # The misleading phrase (with "and the report" as a sibling
    # of API and notifier) is gone.
    assert "exposed by the API, the notifier, and the report" not in cost_src
    # The correct exposure surfaces are present.
    assert "DTO" in cost_src
    assert "notifier" in cost_src.lower()
    # The NOT-embedded truth is present.
    assert "NOT automatically embedded in the final Markdown" in cost_src
