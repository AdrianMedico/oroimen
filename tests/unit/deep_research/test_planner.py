"""Deterministic tests for the C1B planner seams."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from hermes.deep_research.planner import (
    DIRECT_PLANNER_KIND,
    STRUCTURED_LLM_PLANNER_KIND,
    HybridPlanner,
    PlannerOutputError,
    PlannerRequest,
    StructuredPlanParser,
    is_direct_safe,
)
from hermes.deep_research.planning import (
    CapabilitySnapshot,
    PlanningLimits,
    compute_research_brief_sha256,
)

LIMITS = PlanningLimits(max_queries_per_wave=4, max_query_chars=399)
DIRECT_SNAPSHOT = CapabilitySnapshot(
    planner_kind=DIRECT_PLANNER_KIND,
    planner_version="1.0.0",
    max_queries_per_wave=4,
    max_query_chars=399,
    planner_provenance={
        "provider": "local",
        "model": "deterministic-direct",
        "version": "1.0.0",
    },
)
LLM_SNAPSHOT = CapabilitySnapshot(
    planner_kind=STRUCTURED_LLM_PLANNER_KIND,
    planner_version="1.0.0",
    max_queries_per_wave=4,
    max_query_chars=399,
    planner_provenance={
        "provider": "test-adapter",
        "model": "fake-structured",
        "version": "0.1.0",
    },
)


def _request(brief: str) -> PlannerRequest:
    return PlannerRequest(
        research_brief=brief,
        planning_limits=LIMITS,
        capability_snapshot=DIRECT_SNAPSHOT,
        created_at="2026-08-07T00:00:00Z",
    )


class RecordingStructuredPlanner:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.requests: list[PlannerRequest] = []

    async def __call__(self, *, request: PlannerRequest) -> Any:
        self.requests.append(request)
        return self.payload


@pytest.mark.asyncio
async def test_hybrid_planner_uses_direct_path_without_structured_call() -> None:
    fake = RecordingStructuredPlanner({"queries": []})
    result = await HybridPlanner(
        fake,
        direct_capability_snapshot=DIRECT_SNAPSHOT,
        structured_capability_snapshot=LLM_SNAPSHOT,
    ).plan(_request("What is the durable answer?"))

    assert result.local_usage == {"planner_calls": 0}
    assert result.plan.planner_kind == DIRECT_PLANNER_KIND
    assert result.plan.queries[0].text == "What is the durable answer?"
    assert fake.requests == []


@pytest.mark.asyncio
async def test_hybrid_planner_routes_composite_brief_once_to_structured_boundary() -> None:
    fake = RecordingStructuredPlanner(
        {
            "queries": [
                {
                    "text": "primary source for regulation A",
                    "purpose": "authority",
                    "dimension_ids": ["authority"],
                },
                {
                    "text": "timeline and jurisdiction for regulation A",
                    "purpose": "scope",
                    "dimension_ids": ["timeline", "jurisdiction"],
                },
            ]
        }
    )
    brief = "Compare regulation A across jurisdiction and timeline."
    result = await HybridPlanner(
        fake,
        direct_capability_snapshot=DIRECT_SNAPSHOT,
        structured_capability_snapshot=LLM_SNAPSHOT,
    ).plan(_request(brief))

    assert result.local_usage == {"planner_calls": 1}
    assert result.plan.planner_kind == STRUCTURED_LLM_PLANNER_KIND
    assert result.plan.research_brief_sha256 == compute_research_brief_sha256(brief)
    assert result.plan.capability_snapshot.to_dict()["planner_provenance"] == {
        "model": "fake-structured",
        "provider": "test-adapter",
        "version": "0.1.0",
    }
    assert fake.requests[0].capability_snapshot == LLM_SNAPSHOT
    assert all(query.query_id.startswith("q-") for query in result.plan.queries)


def test_structured_parser_generates_ids_and_rejects_model_ids() -> None:
    request = PlannerRequest(
        research_brief="A long internal brief that requires decomposition.",
        planning_limits=LIMITS,
        capability_snapshot=LLM_SNAPSHOT,
        created_at="2026-08-07T00:00:00Z",
    )
    plan = StructuredPlanParser.parse(
        '{"queries":[{"text":"one","purpose":"p","dimension_ids":["d"]}]}',
        request,
    )
    assert plan.queries[0].query_id.startswith("q-")
    with pytest.raises(PlannerOutputError, match="invalid shape"):
        StructuredPlanParser.parse(
            {
                "queries": [
                    {
                        "query_id": "model-controlled",
                        "text": "one",
                        "purpose": "p",
                        "dimension_ids": ["d"],
                    }
                ]
            },
            request,
        )


def test_structured_parser_fails_closed_without_echoing_payload() -> None:
    request = PlannerRequest(
        research_brief="A composite brief.",
        planning_limits=LIMITS,
        capability_snapshot=LLM_SNAPSHOT,
        created_at="2026-08-07T00:00:00Z",
    )
    secret = "provider-secret-not-for-error"
    with pytest.raises(PlannerOutputError) as exc_info:
        StructuredPlanParser.parse(
            {"queries": [{"text": secret, "purpose": "", "dimension_ids": ["d"]}]},
            request,
        )
    assert secret not in str(exc_info.value)


def test_direct_routing_boundary_is_deterministic_and_request_is_immutable() -> None:
    simple = _request("single safe question")
    composite = _request("Compare two jurisdictions")
    assert is_direct_safe(simple) is True
    assert is_direct_safe(composite) is False
    with pytest.raises(FrozenInstanceError):
        simple.research_brief = "changed"  # type: ignore[misc]
