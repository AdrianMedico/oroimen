"""Regression tests for the replaceable semantic planner boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from hermes.deep_research.planner import (
    DIRECT_PLANNER_KIND,
    STRUCTURED_LLM_PLANNER_KIND,
    DirectSearchPlanner,
    PlannerOutputError,
    PlannerRequest,
    PlanningDecisionKind,
    SemanticPlanner,
    StructuredPlanParser,
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


def _request(brief: str, snapshot: CapabilitySnapshot = LLM_SNAPSHOT) -> PlannerRequest:
    return PlannerRequest(
        research_brief=brief,
        planning_limits=LIMITS,
        capability_snapshot=snapshot,
        created_at="2026-08-07T00:00:00Z",
    )


class RecordingSemanticPlanner:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.requests: list[PlannerRequest] = []

    async def __call__(self, *, request: PlannerRequest) -> Any:
        self.requests.append(request)
        return self.payload


def _query(text: str, purpose: str, dimension: str) -> dict[str, Any]:
    return {
        "text": text,
        "purpose": purpose,
        "dimension_ids": [dimension],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("brief", "decision", "query_count"),
    [
        (
            "What is the durable answer?",
            "DIRECT",
            1,
        ),
        (
            "¿Cuál es la respuesta vigente sobre la norma?",
            "DIRECT",
            1,
        ),
        (
            "Welche aktuelle Antwort gilt für die Regelung?",
            "DIRECT",
            1,
        ),
        (
            "Quina és la resposta vigent sobre la norma?",
            "DIRECT",
            1,
        ),
        (
            "Compara dues normes segons abast i efectes, sense limitar-te al resum.",
            "DECOMPOSE",
            2,
        ),
        (
            "Traça l'evolució de la norma i identifica els canvis per període.",
            "DECOMPOSE",
            2,
        ),
        (
            "Welche Rechtsordnung gilt und welche Behörde ist zuständig?",
            "DECOMPOSE",
            2,
        ),
        (
            "Analitza la norma, compara els efectes i explica l'evolució temporal.",
            "DECOMPOSE",
            2,
        ),
    ],
)
async def test_every_language_and_failure_class_reaches_semantic_planner(
    brief: str, decision: str, query_count: int
) -> None:
    queries = [
        _query(f"evidence query {index}", f"dimension {index}", f"d{index}")
        for index in range(query_count)
    ]
    fake = RecordingSemanticPlanner({"decision": decision, "queries": queries})
    result = await SemanticPlanner(
        fake,
        capability_snapshot=LLM_SNAPSHOT,
    ).plan(_request(brief, DIRECT_SNAPSHOT))

    assert len(fake.requests) == 1
    assert fake.requests[0].research_brief == brief
    assert fake.requests[0].capability_snapshot == LLM_SNAPSHOT
    assert result.local_usage == {"planner_calls": 1}
    assert result.decision.kind is PlanningDecisionKind(decision)
    assert result.plan.planning_decision == decision
    assert result.plan.research_brief_sha256 == compute_research_brief_sha256(brief)
    assert result.plan.capability_snapshot.to_dict()["planner_provenance"] == {
        "model": "fake-structured",
        "provider": "test-adapter",
        "version": "0.1.0",
    }
    assert all(query.query_id.startswith("q-") for query in result.plan.queries)


def test_structured_parser_generates_local_ids_and_rejects_model_ids() -> None:
    request = _request("A brief that requires decomposition.")
    decision = StructuredPlanParser.parse(
        {
            "decision": "DECOMPOSE",
            "queries": [
                _query("one", "first dimension", "d1"),
                _query("two", "second dimension", "d2"),
            ],
        },
        request,
    )
    assert decision.kind is PlanningDecisionKind.DECOMPOSE
    assert decision.plan.queries[0].query_id.startswith("q-")

    with pytest.raises(PlannerOutputError, match="invalid shape"):
        StructuredPlanParser.parse(
            {
                "decision": "DIRECT",
                "queries": [
                    {
                        "query_id": "model-controlled",
                        "text": "one",
                        "purpose": "p",
                        "dimension_ids": ["d"],
                    }
                ],
            },
            request,
        )


def test_structured_parser_requires_decision_and_cardinality() -> None:
    request = _request("A brief.")
    with pytest.raises(PlannerOutputError, match="invalid shape"):
        StructuredPlanParser.parse(
            {"queries": [_query("one", "p", "d")]},
            request,
        )
    with pytest.raises(PlannerOutputError, match="DIRECT"):
        StructuredPlanParser.parse(
            {
                "decision": "DIRECT",
                "queries": [
                    _query("one", "p", "d"),
                    _query("two", "p", "e"),
                ],
            },
            request,
        )
    with pytest.raises(PlannerOutputError, match="DECOMPOSE"):
        StructuredPlanParser.parse(
            {"decision": "DECOMPOSE", "queries": [_query("one", "p", "d")]},
            request,
        )


def test_structured_parser_rejects_duplicate_queries_deterministically() -> None:
    request = _request("A comparison brief.")
    duplicate = _query("same query", "same purpose", "same")
    with pytest.raises(PlannerOutputError, match="deterministic validation"):
        StructuredPlanParser.parse(
            {"decision": "DECOMPOSE", "queries": [duplicate, duplicate]},
            request,
        )


def test_structured_parser_fails_closed_without_echoing_payload() -> None:
    request = _request("A composite brief.")
    secret = "provider-secret-not-for-error"
    with pytest.raises(PlannerOutputError) as exc_info:
        StructuredPlanParser.parse(
            {
                "decision": "DIRECT",
                "queries": [_query(secret, "", "d")],
            },
            request,
        )
    assert secret not in str(exc_info.value)


@pytest.mark.asyncio
async def test_planner_unavailable_does_not_fallback_to_direct_search() -> None:
    class Unavailable:
        calls = 0

        async def __call__(self, *, request: PlannerRequest) -> Any:
            self.calls += 1
            raise RuntimeError("planner unavailable")

    unavailable = Unavailable()
    with pytest.raises(RuntimeError, match="planner unavailable"):
        await SemanticPlanner(
            unavailable,
            capability_snapshot=LLM_SNAPSHOT,
        ).plan(_request("A short brief", DIRECT_SNAPSHOT))
    assert unavailable.calls == 1


@pytest.mark.asyncio
async def test_invalid_planner_output_does_not_fallback_to_direct_search() -> None:
    fake = RecordingSemanticPlanner(
        {"decision": "UNKNOWN", "queries": [_query("one", "p", "d")]}
    )
    with pytest.raises(PlannerOutputError, match="decision is invalid"):
        await SemanticPlanner(
            fake,
            capability_snapshot=LLM_SNAPSHOT,
        ).plan(_request("A short brief", DIRECT_SNAPSHOT))
    assert len(fake.requests) == 1


def test_direct_planner_is_only_a_control_primitive() -> None:
    request = _request(
        "Compara dos jurisdicciones y explica su evolución temporal.",
        DIRECT_SNAPSHOT,
    )
    plan = DirectSearchPlanner().plan(request)
    assert plan.planner_kind == DIRECT_PLANNER_KIND
    assert plan.planning_decision == "DIRECT"
    assert len(plan.queries) == 1


def test_semantic_planner_requires_complete_provenance() -> None:
    incomplete = CapabilitySnapshot(
        planner_kind=STRUCTURED_LLM_PLANNER_KIND,
        planner_version="1.0.0",
        max_queries_per_wave=4,
        max_query_chars=399,
        planner_provenance={"provider": "test"},
    )
    with pytest.raises(ValueError, match="provider/model/version"):
        SemanticPlanner(RecordingSemanticPlanner({}), capability_snapshot=incomplete)


def test_planner_request_is_immutable() -> None:
    request = _request("single safe question")
    with pytest.raises(FrozenInstanceError):
        request.research_brief = "changed"  # type: ignore[misc]
