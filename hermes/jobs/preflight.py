"""Pure offline readiness evaluation for Deep Research.

The evaluator accepts validated settings and closed capability flags. It does
not perform DNS, HTTP, search, fetch, filesystem, database, or LLM operations.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from hermes.config import Settings


class PreflightStatus(StrEnum):
    DISABLED = "disabled"
    BLOCKED = "blocked"
    DEGRADED = "degraded"
    READY = "ready"


class PreflightCheckState(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class LimitClassification(StrEnum):
    HARD = "hard"
    ADMISSION_CONTROL = "admission_control"
    SOFT_WARNING = "soft_warning"
    ESTIMATE = "estimate"


class PreflightCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    state: PreflightCheckState
    required: bool
    message: str
    remediation: str


class PreflightLimit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    classification: LimitClassification
    configured: bool
    unit: str | None = None


class DeepResearchPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    status: PreflightStatus
    mode: Literal["offline"] = "offline"
    checks: tuple[PreflightCheck, ...]
    limits: tuple[PreflightLimit, ...]


class DeepResearchCapabilities(BaseModel):
    """Closed, value-free facts supplied by reviewed runtime wiring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    service_wiring: bool = False
    recovery_wiring: bool = False
    search_backend_configured: bool = False
    llm_provider_configured: bool = False
    fetch_policy: bool = False
    external_fetch: bool = False
    report_retrieval: bool = False
    model_output_enforced: bool = False
    egress_firewall: bool = False
    query_decomposition: bool = False


def _result(
    *,
    code: str,
    available: bool,
    required: bool,
    available_message: str,
    unavailable_message: str,
    remediation: str,
) -> PreflightCheck:
    return PreflightCheck(
        code=code,
        state=PreflightCheckState.PASS if available else PreflightCheckState.FAIL,
        required=required,
        message=available_message if available else unavailable_message,
        remediation="" if available else remediation,
    )


def _disabled_check(code: str, required: bool) -> PreflightCheck:
    return PreflightCheck(
        code=code,
        state=PreflightCheckState.SKIP,
        required=required,
        message="Deep Research is disabled.",
        remediation="Enable Deep Research only after the required safety gates pass.",
    )


def _derive_status(
    *,
    enabled: bool,
    checks: tuple[PreflightCheck, ...],
) -> PreflightStatus:
    if not enabled:
        return PreflightStatus.DISABLED
    if any(check.required and check.state is PreflightCheckState.FAIL for check in checks):
        return PreflightStatus.BLOCKED
    if any(check.state is PreflightCheckState.WARN for check in checks):
        return PreflightStatus.DEGRADED
    return PreflightStatus.READY


def evaluate_deep_research_preflight(
    settings: Settings,
    capabilities: DeepResearchCapabilities | None = None,
) -> DeepResearchPreflight:
    """Evaluate deterministic offline readiness without performing I/O."""

    caps = capabilities or DeepResearchCapabilities()
    enabled = bool(settings.deep_research_enabled)

    check_specs: list[PreflightCheck] = [
        PreflightCheck(
            code="dr.feature.opt_in",
            state=PreflightCheckState.PASS if enabled else PreflightCheckState.SKIP,
            required=True,
            message="Deep Research is enabled." if enabled else "Deep Research is disabled.",
            remediation="" if enabled else "Set the explicit opt-in after safety review.",
        ),
        _result(
            code="dr.runtime.service_wiring",
            available=caps.service_wiring,
            required=True,
            available_message="Runtime service wiring is available.",
            unavailable_message="Runtime service wiring is unavailable.",
            remediation="Install the reviewed service and scheduler wiring.",
        ),
        _result(
            code="dr.runtime.recovery_wiring",
            available=caps.recovery_wiring,
            required=True,
            available_message="Startup recovery wiring is available.",
            unavailable_message="Startup recovery wiring is unavailable.",
            remediation="Install the reviewed startup recovery integration.",
        ),
        _result(
            code="dr.search.backend_configured",
            available=caps.search_backend_configured,
            required=True,
            available_message="A Deep Research search backend is configured.",
            unavailable_message="A Deep Research search backend is not configured.",
            remediation="Configure the supported search backend without exposing its value.",
        ),
        PreflightCheck(
            code="dr.search.backend_reachable",
            state=PreflightCheckState.SKIP,
            required=False,
            message="Live reachability is not evaluated in offline mode.",
            remediation="Use a future explicit live preflight.",
        ),
        PreflightCheck(
            code="dr.search.backend_authorized",
            state=PreflightCheckState.SKIP,
            required=False,
            message="Live authorization is not evaluated in offline mode.",
            remediation="Use a future explicit live preflight.",
        ),
        _result(
            code="dr.fetch.policy_available",
            available=caps.fetch_policy,
            required=True,
            available_message="The reviewed external-fetch policy is available.",
            unavailable_message="The reviewed external-fetch policy is unavailable.",
            remediation="Install and validate the safe external fetcher.",
        ),
        _result(
            code="dr.fetch.external_enabled",
            available=caps.external_fetch,
            required=True,
            available_message="External source fetching is explicitly enabled.",
            unavailable_message="External source fetching is disabled.",
            remediation="Enable external fetching only after the fetch-policy gate passes.",
        ),
        _result(
            code="dr.llm.provider_configured",
            available=caps.llm_provider_configured,
            required=True,
            available_message="The research model provider is configured.",
            unavailable_message="The research model provider is not configured.",
            remediation="Configure the supported provider without exposing its value.",
        ),
        PreflightCheck(
            code="dr.llm.provider_reachable",
            state=PreflightCheckState.SKIP,
            required=False,
            message="Live reachability is not evaluated in offline mode.",
            remediation="Use a future explicit live preflight.",
        ),
        PreflightCheck(
            code="dr.llm.provider_authorized",
            state=PreflightCheckState.SKIP,
            required=False,
            message="Live authorization is not evaluated in offline mode.",
            remediation="Use a future explicit live preflight.",
        ),
        _result(
            code="dr.report.retrieval_available",
            available=caps.report_retrieval,
            required=True,
            available_message="Authenticated report retrieval is available.",
            unavailable_message="Authenticated report retrieval is unavailable.",
            remediation="Add owner-scoped report-content retrieval.",
        ),
        PreflightCheck(
            code="dr.egress.firewall_enabled",
            state=(PreflightCheckState.PASS if caps.egress_firewall else PreflightCheckState.WARN),
            required=False,
            message=(
                "Container egress defense is enabled."
                if caps.egress_firewall
                else "Container egress defense is not confirmed."
            ),
            remediation=(
                ""
                if caps.egress_firewall
                else "Review optional container egress defense separately."
            ),
        ),
        _result(
            code="dr.limits.model_output_enforced",
            available=caps.model_output_enforced,
            required=True,
            available_message="Research model-output bounds are enforced.",
            unavailable_message="Research model-output bounds are not enforced.",
            remediation="Pass hard model-output bounds to both research model calls.",
        ),
        PreflightCheck(
            code="dr.limits.job_budget_enforced",
            state=PreflightCheckState.PASS,
            required=False,
            message="Daily admission control and per-job warning are configured.",
            remediation="",
        ),
        PreflightCheck(
            code="dr.architecture.query_decomposition",
            state=(
                PreflightCheckState.PASS if caps.query_decomposition else PreflightCheckState.WARN
            ),
            required=False,
            message=(
                "Query decomposition is available."
                if caps.query_decomposition
                else "Query decomposition is not implemented."
            ),
            remediation=(
                ""
                if caps.query_decomposition
                else "Keep query decomposition deferred for the supported first slice."
            ),
        ),
    ]

    checks = tuple(sorted(check_specs, key=lambda check: check.code))
    if not enabled:
        checks = tuple(
            check
            if check.code.startswith("dr.egress.")
            or check.code.startswith("dr.architecture.")
            or check.code.startswith("dr.limits.job_budget")
            or check.code.startswith("dr.search.backend_reachable")
            or check.code.startswith("dr.search.backend_authorized")
            or check.code.startswith("dr.llm.provider_reachable")
            or check.code.startswith("dr.llm.provider_authorized")
            else _disabled_check(check.code, check.required)
            for check in checks
        )
    status = _derive_status(enabled=enabled, checks=checks)

    limits = tuple(
        sorted(
            (
                PreflightLimit(
                    name="daily_job_budget",
                    classification=LimitClassification.ADMISSION_CONTROL,
                    configured=True,
                    unit="usd",
                ),
                PreflightLimit(
                    name="final_model_output",
                    classification=LimitClassification.ESTIMATE,
                    configured=True,
                    unit="model_output_units",
                ),
                PreflightLimit(
                    name="per_job_budget",
                    classification=LimitClassification.SOFT_WARNING,
                    configured=True,
                    unit="usd",
                ),
                PreflightLimit(
                    name="per_source_model_output",
                    classification=LimitClassification.ESTIMATE,
                    configured=True,
                    unit="model_output_units",
                ),
            ),
            key=lambda limit: limit.name,
        )
    )
    return DeepResearchPreflight(status=status, checks=checks, limits=limits)


__all__ = [
    "DeepResearchCapabilities",
    "DeepResearchPreflight",
    "LimitClassification",
    "PreflightCheck",
    "PreflightCheckState",
    "PreflightLimit",
    "PreflightStatus",
    "evaluate_deep_research_preflight",
]
