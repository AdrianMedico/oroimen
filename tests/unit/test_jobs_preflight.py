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


# ---------------------------------------------------------------------------
# Slice 1C2: report_retrieval capability gate
# ---------------------------------------------------------------------------
# The brief mandates the EXACT rule: ``report_retrieval`` flips to True
# ONLY when ``LocalReportStore`` has been successfully constructed in
# the current process. A class existing in the codebase or a configured
# ``deep_research_data_root`` is NOT sufficient. The composition root
# is the single authority over this flip; the preflight gate is a
# pure consumer of the ``capabilities.report_retrieval`` field.
#
# These tests prove:
# 1. Default value of ``capabilities.report_retrieval`` is False.
# 2. Setting it to True unblocks the gate.
# 3. Setting it to False keeps the gate blocked even when other
#    capabilities are True (defense in depth: a partially-wired
#    runtime cannot pass preflight by accident).
# 4. The composition test in ``test_main.py`` proves the wiring step
#    is the source of the flag (a class existing on disk is not
#    sufficient; only a constructed ``LocalReportStore`` counts).


def test_report_retrieval_default_false() -> None:
    """``DeepResearchCapabilities()`` defaults ``report_retrieval`` to False.

    The composition root MUST explicitly set it to True after a
    successful ``LocalReportStore`` construction. A class existing in
    the codebase or a configured ``deep_research_data_root`` is NOT
    sufficient — see ``test_main.py::test_compose_*`` for the wiring
    proof.
    """
    caps = DeepResearchCapabilities()
    assert caps.report_retrieval is False


def test_preflight_honors_report_reader_wired_flag() -> None:
    """The preflight gate honors ``capabilities.report_retrieval`` exactly.

    When the composition root successfully constructs a
    ``LocalReportStore``, it MUST set ``capabilities.report_retrieval
    = True``. The preflight gate consumes that field — the test
    proves the gate transitions from BLOCKED to DEGRADED/READY when
    the flag flips, and stays BLOCKED when it does not.
    """
    # (1) Without report_retrieval: preflight is BLOCKED (required
    # gate fails). Construct a minimal settings + capability set.
    base_caps = DeepResearchCapabilities(
        service_wiring=True,
        recovery_wiring=True,
        search_backend_configured=True,
        llm_provider_configured=True,
        fetch_policy=True,
        external_fetch=True,
        model_output_enforced=True,
        # report_retrieval: explicitly NOT set (defaults to False).
    )
    settings = SimpleNamespace(deep_research_enabled=True)
    report_blocked = evaluate_deep_research_preflight(settings, base_caps)
    assert report_blocked.status is PreflightStatus.BLOCKED
    report_check = next(
        c for c in report_blocked.checks if c.code == "dr.report.retrieval_available"
    )
    assert report_check.state is PreflightCheckState.FAIL

    # (2) With report_retrieval=True: preflight is at least DEGRADED
    # (the optional egress / decomposition checks are still WARN
    # unless explicitly set). The required report_retrieval check
    # transitions to PASS.
    report_wired_caps = DeepResearchCapabilities(
        service_wiring=True,
        recovery_wiring=True,
        search_backend_configured=True,
        llm_provider_configured=True,
        fetch_policy=True,
        external_fetch=True,
        model_output_enforced=True,
        report_retrieval=True,  # the flag the composition root flips
    )
    report_ok = evaluate_deep_research_preflight(settings, report_wired_caps)
    report_check = next(c for c in report_ok.checks if c.code == "dr.report.retrieval_available")
    assert report_check.state is PreflightCheckState.PASS
    # No required gate failed → status is DEGRADED (because the
    # optional egress / decomposition checks are WARN) or READY (if
    # those are also PASS). Either way: NOT BLOCKED.
    assert report_ok.status in (PreflightStatus.DEGRADED, PreflightStatus.READY)
