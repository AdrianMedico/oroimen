"""Offline Deep Research preflight contract tests."""

from __future__ import annotations

import socket
import subprocess
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from hermes.config import Settings
from hermes.jobs.preflight import (
    DeepResearchCapabilities,
    LimitClassification,
    PreflightCheck,
    PreflightCheckState,
    PreflightStatus,
    _derive_status,
    evaluate_deep_research_preflight,
)

EXPECTED_CODES = (
    "dr.architecture.query_decomposition",
    "dr.egress.firewall_enabled",
    "dr.feature.opt_in",
    "dr.fetch.external_enabled",
    "dr.fetch.policy_available",
    "dr.limits.job_budget_enforced",
    "dr.limits.model_output_enforced",
    "dr.llm.provider_authorized",
    "dr.llm.provider_configured",
    "dr.llm.provider_reachable",
    "dr.report.retrieval_available",
    "dr.runtime.recovery_wiring",
    "dr.runtime.service_wiring",
    "dr.search.backend_authorized",
    "dr.search.backend_configured",
    "dr.search.backend_reachable",
)


def _settings(
    *,
    enabled: bool = False,
) -> Any:
    return SimpleNamespace(deep_research_enabled=enabled)


def _by_code(report: Any) -> dict[str, Any]:
    return {check.code: check for check in report.checks}


def test_default_disabled_skips_required_execution_checks() -> None:
    report = evaluate_deep_research_preflight(_settings())

    assert report.status is PreflightStatus.DISABLED
    assert report.mode == "offline"
    assert report.schema_version == 1
    assert tuple(check.code for check in report.checks) == EXPECTED_CODES
    for check in report.checks:
        if check.required:
            assert check.state is PreflightCheckState.SKIP


def test_enabled_baseline_is_blocked() -> None:
    report = evaluate_deep_research_preflight(
        _settings(enabled=True),
        DeepResearchCapabilities(
            search_backend_configured=True,
            llm_provider_configured=True,
        ),
    )
    checks = _by_code(report)

    assert report.status is PreflightStatus.BLOCKED
    for code in (
        "dr.runtime.service_wiring",
        "dr.runtime.recovery_wiring",
        "dr.fetch.policy_available",
        "dr.fetch.external_enabled",
        "dr.report.retrieval_available",
        "dr.limits.model_output_enforced",
    ):
        assert checks[code].state is PreflightCheckState.FAIL


def test_existing_credentials_cannot_make_slice_1a_ready() -> None:
    report = evaluate_deep_research_preflight(
        _settings(enabled=True),
        DeepResearchCapabilities(
            search_backend_configured=True,
            llm_provider_configured=True,
        ),
    )

    assert report.status is PreflightStatus.BLOCKED


def test_future_complete_capabilities_can_reach_ready() -> None:
    report = evaluate_deep_research_preflight(
        _settings(enabled=True),
        DeepResearchCapabilities(
            service_wiring=True,
            recovery_wiring=True,
            search_backend_configured=True,
            llm_provider_configured=True,
            fetch_policy=True,
            external_fetch=True,
            report_retrieval=True,
            model_output_enforced=True,
            egress_firewall=True,
            query_decomposition=True,
        ),
    )

    assert report.status is PreflightStatus.READY


def test_optional_warnings_produce_degraded() -> None:
    report = evaluate_deep_research_preflight(
        _settings(enabled=True),
        DeepResearchCapabilities(
            service_wiring=True,
            recovery_wiring=True,
            search_backend_configured=True,
            llm_provider_configured=True,
            fetch_policy=True,
            external_fetch=True,
            report_retrieval=True,
            model_output_enforced=True,
        ),
    )

    assert report.status is PreflightStatus.DEGRADED


def test_presence_checks_do_not_expose_configuration_values() -> None:
    marker = "runtime-value-must-not-appear"
    settings = SimpleNamespace(
        deep_research_enabled=True,
        unrelated_runtime_value=marker,
    )

    payload = evaluate_deep_research_preflight(settings).model_dump_json()

    assert marker not in payload
    assert "http://" not in payload
    assert "https://" not in payload


def test_limit_classifications_are_truthful() -> None:
    report = evaluate_deep_research_preflight(_settings())
    limits = {limit.name: limit for limit in report.limits}

    assert limits["daily_job_budget"].classification is LimitClassification.ADMISSION_CONTROL
    assert limits["per_job_budget"].classification is LimitClassification.SOFT_WARNING
    assert limits["final_model_output"].classification is LimitClassification.ESTIMATE
    assert limits["per_source_model_output"].classification is LimitClassification.ESTIMATE


def test_evaluator_performs_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("offline preflight attempted I/O")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(httpx, "Client", fail)
    monkeypatch.setattr(httpx, "AsyncClient", fail)

    report = evaluate_deep_research_preflight(_settings(enabled=True))

    assert report.mode == "offline"


def test_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PreflightCheck(
            code="dr.test",
            state=PreflightCheckState.PASS,
            required=False,
            message="ok",
            remediation="",
            unexpected="not allowed",  # type: ignore[call-arg]
        )


def test_any_required_failure_blocks_status() -> None:
    checks = (
        PreflightCheck(
            code="dr.future.required_gate",
            state=PreflightCheckState.FAIL,
            required=True,
            message="A future required gate is unavailable.",
            remediation="Complete the required gate.",
        ),
    )

    assert _derive_status(enabled=True, checks=checks) is PreflightStatus.BLOCKED


def test_setting_defaults_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERMES_DEEP_RESEARCH_ENABLED", raising=False)

    settings = Settings(_env_file=None)

    assert settings.deep_research_enabled is False


def test_setting_accepts_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_DEEP_RESEARCH_ENABLED", "true")

    settings = Settings(_env_file=None)

    assert settings.deep_research_enabled is True
