"""Provider-neutral seams for semantic planning of one SearchWave.

The semantic planner proposes a structured decision.  Local code generates
query IDs, binds the plan to the Research Brief hash, and validates every
bound.  This module does not invoke providers or execute searches.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from hermes.deep_research.planning import (
    ALLOWED_WAVE_INDICES,
    SCHEMA_VERSION,
    CapabilitySnapshot,
    PlannedSearchQuery,
    PlanningLimits,
    PlanningValidationError,
    SearchPlan,
    build_search_plan,
    compute_query_id,
    compute_research_brief_sha256,
    normalize_query_text,
)

DIRECT_PLANNER_KIND = "c1b-direct"
STRUCTURED_LLM_PLANNER_KIND = "c1b-llm-structured"
PLANNER_VERSION = "1.0.0"
MAX_STRUCTURED_OUTPUT_CHARS = 20_000

_STRUCTURED_TOP_LEVEL_KEYS = frozenset({"decision", "queries"})
_STRUCTURED_QUERY_KEYS = frozenset({"text", "purpose", "dimension_ids"})


class PlannerOutputError(ValueError):
    """Raised when structured planner output cannot be accepted safely."""


@dataclass(frozen=True)
class PlannerRequest:
    """Immutable planner input bound to the original Research Brief hash."""

    research_brief: str
    planning_limits: PlanningLimits
    capability_snapshot: CapabilitySnapshot
    created_at: str
    wave_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.research_brief, str) or not self.research_brief.strip():
            raise ValueError("research_brief must be a non-empty string")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("created_at must be a non-empty string")
        if self.wave_index not in ALLOWED_WAVE_INDICES:
            raise ValueError(f"wave_index must be in {sorted(ALLOWED_WAVE_INDICES)}")
        if (
            self.planning_limits.max_queries_per_wave
            != self.capability_snapshot.max_queries_per_wave
            or self.planning_limits.max_query_chars
            != self.capability_snapshot.max_query_chars
        ):
            raise ValueError("planning limits and capability snapshot must match")

    @property
    def research_brief_sha256(self) -> str:
        return compute_research_brief_sha256(self.research_brief)


class SemanticPlannerCallable(Protocol):
    """Replaceable async semantic-planner boundary."""

    async def __call__(self, *, request: PlannerRequest) -> Any: ...


# Compatibility name for the existing provider-adapter seam.  Both names
# describe the same replaceable boundary; only ``SemanticPlanner`` owns the
# production routing decision.
StructuredPlannerCallable = SemanticPlannerCallable


class PlanningDecisionKind(StrEnum):
    """The only semantic routing decisions accepted from the planner."""

    DIRECT = "DIRECT"
    DECOMPOSE = "DECOMPOSE"


@dataclass(frozen=True)
class PlanningDecision:
    """Structured semantic decision bound to its validated SearchPlan."""

    kind: PlanningDecisionKind
    plan: SearchPlan

    def __post_init__(self) -> None:
        if self.plan.planning_decision != self.kind.value:
            raise ValueError("planning decision and SearchPlan decision must match")


@dataclass(frozen=True)
class PlannerResult:
    """Validated decision plus planner-local accounting, never invoice truth."""

    decision: PlanningDecision
    local_usage: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_usage", MappingProxyType(dict(self.local_usage)))

    @property
    def plan(self) -> SearchPlan:
        """Convenience view for the wave runner boundary."""

        return self.decision.plan


class DirectSearchPlanner:
    """Build one direct query as a deterministic primitive/control.

    Deep Research production routing does not call this class to decide
    whether a brief is simple.  The replaceable semantic planner makes that
    decision; this primitive is retained for controls and isolated tests.
    """

    def plan(self, request: PlannerRequest) -> SearchPlan:
        if request.capability_snapshot.planner_kind != DIRECT_PLANNER_KIND:
            raise PlannerOutputError("direct planner requires c1b-direct capability")

        text = normalize_query_text(request.research_brief)
        if not text or len(text) > request.planning_limits.max_query_chars:
            raise PlannerOutputError(
                "direct planner query violates deterministic limits"
            )
        query = PlannedSearchQuery(
            query_id=compute_query_id(
                schema_version=SCHEMA_VERSION,
                wave_index=request.wave_index,
                ordinal=0,
                normalized_text=text,
                research_brief_sha256=request.research_brief_sha256,
            ),
            text=text,
            purpose="direct retrieval",
            dimension_ids=("direct",),
            ordinal=0,
        )
        return build_search_plan(
            planner_kind=DIRECT_PLANNER_KIND,
            planner_version=request.capability_snapshot.planner_version,
            research_brief_sha256=request.research_brief_sha256,
            wave_index=request.wave_index,
            queries=(query,),
            planning_limits=request.planning_limits,
            capability_snapshot=request.capability_snapshot,
            created_at=request.created_at,
            expected_research_brief_sha256=request.research_brief_sha256,
            planning_decision=PlanningDecisionKind.DIRECT.value,
        )


class StructuredPlanParser:
    """Parse and locally validate the semantic planner schema."""

    @classmethod
    def parse(cls, raw: Any, request: PlannerRequest) -> PlanningDecision:
        if request.capability_snapshot.planner_kind != STRUCTURED_LLM_PLANNER_KIND:
            raise PlannerOutputError(
                "structured parser requires c1b-llm-structured capability"
            )
        payload = cls._payload(raw)
        if frozenset(payload) != _STRUCTURED_TOP_LEVEL_KEYS:
            raise PlannerOutputError("structured planner output has an invalid shape")

        raw_decision = payload.get("decision")
        if not isinstance(raw_decision, str):
            raise PlannerOutputError("structured planner decision is invalid")
        try:
            decision = PlanningDecisionKind(raw_decision)
        except ValueError:
            raise PlannerOutputError("structured planner decision is invalid") from None

        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            raise PlannerOutputError("structured planner queries must be a list")
        if decision is PlanningDecisionKind.DIRECT and len(raw_queries) != 1:
            raise PlannerOutputError("DIRECT requires exactly one query")
        if decision is PlanningDecisionKind.DECOMPOSE and not (
            2
            <= len(raw_queries)
            <= request.planning_limits.max_queries_per_wave
        ):
            raise PlannerOutputError("DECOMPOSE requires 2..4 queries")

        queries: list[PlannedSearchQuery] = []
        for ordinal, raw_query in enumerate(raw_queries):
            if not isinstance(raw_query, Mapping):
                raise PlannerOutputError("structured planner query must be an object")
            if frozenset(raw_query) != _STRUCTURED_QUERY_KEYS:
                raise PlannerOutputError("structured planner query has an invalid shape")
            text = raw_query.get("text")
            purpose = raw_query.get("purpose")
            dimensions = raw_query.get("dimension_ids")
            if not isinstance(text, str) or not isinstance(purpose, str):
                raise PlannerOutputError(
                    "structured planner text and purpose must be strings"
                )
            if not isinstance(dimensions, list) or not dimensions:
                raise PlannerOutputError(
                    "structured planner dimension_ids must be non-empty"
                )
            if not all(isinstance(item, str) and item for item in dimensions):
                raise PlannerOutputError(
                    "structured planner dimensions must be strings"
                )
            normalized_text = normalize_query_text(text)
            if not normalized_text or not purpose.strip():
                raise PlannerOutputError(
                    "structured planner text and purpose must be non-empty"
                )
            queries.append(
                PlannedSearchQuery(
                    query_id=compute_query_id(
                        schema_version=SCHEMA_VERSION,
                        wave_index=request.wave_index,
                        ordinal=ordinal,
                        normalized_text=normalized_text,
                        research_brief_sha256=request.research_brief_sha256,
                    ),
                    text=normalized_text,
                    purpose=purpose.strip(),
                    dimension_ids=tuple(dimensions),
                    ordinal=ordinal,
                )
            )

        try:
            plan = build_search_plan(
                planner_kind=STRUCTURED_LLM_PLANNER_KIND,
                planner_version=request.capability_snapshot.planner_version,
                research_brief_sha256=request.research_brief_sha256,
                wave_index=request.wave_index,
                queries=tuple(queries),
                planning_limits=request.planning_limits,
                capability_snapshot=request.capability_snapshot,
                created_at=request.created_at,
                expected_research_brief_sha256=request.research_brief_sha256,
                planning_decision=decision.value,
            )
        except (PlanningValidationError, ValueError, TypeError):
            raise PlannerOutputError(
                "structured planner output failed deterministic validation"
            ) from None
        return PlanningDecision(kind=decision, plan=plan)

    @staticmethod
    def _payload(raw: Any) -> Mapping[str, Any]:
        if isinstance(raw, str):
            if len(raw) > MAX_STRUCTURED_OUTPUT_CHARS:
                raise PlannerOutputError("structured planner output exceeds the size cap")
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raise PlannerOutputError(
                    "structured planner output is not valid JSON"
                ) from None
        if not isinstance(raw, Mapping):
            raise PlannerOutputError("structured planner output must be an object")
        return raw


class SemanticPlanner:
    """Send every Research Brief through the semantic planner exactly once."""

    def __init__(
        self,
        semantic_planner: SemanticPlannerCallable,
        *,
        capability_snapshot: CapabilitySnapshot,
    ) -> None:
        if capability_snapshot.planner_kind != STRUCTURED_LLM_PLANNER_KIND:
            raise ValueError(
                "semantic planner capability snapshot must use c1b-llm-structured"
            )
        if set(capability_snapshot.planner_provenance) != {
            "provider",
            "model",
            "version",
        }:
            raise ValueError(
                "semantic planner capability snapshot requires provider/model/version"
            )
        self._capability_snapshot = capability_snapshot
        self._semantic_planner = semantic_planner

    async def plan(self, request: PlannerRequest) -> PlannerResult:
        semantic_request = replace(
            request,
            capability_snapshot=self._capability_snapshot,
        )
        raw = await self._semantic_planner(request=semantic_request)
        return PlannerResult(
            decision=StructuredPlanParser.parse(raw, semantic_request),
            local_usage={"planner_calls": 1},
        )


__all__ = [
    "DIRECT_PLANNER_KIND",
    "MAX_STRUCTURED_OUTPUT_CHARS",
    "PLANNER_VERSION",
    "STRUCTURED_LLM_PLANNER_KIND",
    "DirectSearchPlanner",
    "PlannerOutputError",
    "PlannerRequest",
    "PlannerResult",
    "PlanningDecision",
    "PlanningDecisionKind",
    "SemanticPlanner",
    "SemanticPlannerCallable",
    "StructuredPlanParser",
    "StructuredPlannerCallable",
]
