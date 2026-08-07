"""Provider-neutral, deterministic planning seams for one SearchWave.

The module deliberately stops at plan creation.  It does not invoke a model,
select a provider, or iterate across waves.  A caller supplies an optional
structured-planner boundary and receives a validated, provenance-bound
``SearchPlan`` suitable for ``SearchWaveRunner``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol

from hermes.deep_research.planning import (
    ALLOWED_WAVE_INDICES,
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

_STRUCTURED_TOP_LEVEL_KEYS = frozenset({"queries"})
_STRUCTURED_QUERY_KEYS = frozenset({"text", "purpose", "dimension_ids"})
_DIRECT_COMPLEXITY_MARKERS = (
    "\n",
    "\r",
    "compare",
    "comparison",
    "versus",
    " vs ",
    "jurisdiction",
    "timeline",
    "across ",
    "multiple ",
    "list ",
)


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


class StructuredPlannerCallable(Protocol):
    """Replaceable async boundary for a provider/model adapter."""

    async def __call__(self, *, request: PlannerRequest) -> Any: ...


@dataclass(frozen=True)
class PlannerResult:
    """Validated plan plus planner-local accounting, never invoice truth."""

    plan: SearchPlan
    local_usage: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_usage", MappingProxyType(dict(self.local_usage)))


def is_direct_safe(request: PlannerRequest) -> bool:
    """Return the deterministic hybrid-routing decision for a brief.

    The boundary is intentionally conservative and inspectable: a single-line
    brief within the derived-query cap and without known composite markers can
    use the direct path.  Everything else goes through the structured seam.
    """

    normalized = normalize_query_text(request.research_brief)
    if not normalized or len(normalized) > request.planning_limits.max_query_chars:
        return False
    lowered = normalized.lower()
    return not any(marker in lowered for marker in _DIRECT_COMPLEXITY_MARKERS)


class DirectSearchPlanner:
    """Build one direct query without a model/provider call."""

    def plan(self, request: PlannerRequest) -> SearchPlan:
        if request.capability_snapshot.planner_kind != DIRECT_PLANNER_KIND:
            raise PlannerOutputError("direct planner requires c1b-direct capability")
        if not is_direct_safe(request):
            raise PlannerOutputError("brief is not eligible for the direct planner")

        text = normalize_query_text(request.research_brief)
        query = PlannedSearchQuery(
            query_id=compute_query_id(
                schema_version=1,
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
        )


class StructuredPlanParser:
    """Parse and locally validate the minimal structured planner schema."""

    @classmethod
    def parse(cls, raw: Any, request: PlannerRequest) -> SearchPlan:
        if request.capability_snapshot.planner_kind != STRUCTURED_LLM_PLANNER_KIND:
            raise PlannerOutputError(
                "structured parser requires c1b-llm-structured capability"
            )
        payload = cls._payload(raw)
        if frozenset(payload) != _STRUCTURED_TOP_LEVEL_KEYS:
            raise PlannerOutputError("structured planner output has an invalid shape")
        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            raise PlannerOutputError("structured planner queries must be a list")

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
                raise PlannerOutputError("structured planner text and purpose must be strings")
            if not isinstance(dimensions, list) or not dimensions:
                raise PlannerOutputError("structured planner dimension_ids must be non-empty")
            if not all(isinstance(item, str) and item for item in dimensions):
                raise PlannerOutputError("structured planner dimensions must be strings")
            normalized_text = normalize_query_text(text)
            if not normalized_text or not purpose.strip():
                raise PlannerOutputError("structured planner text and purpose must be non-empty")
            queries.append(
                PlannedSearchQuery(
                    query_id=compute_query_id(
                        schema_version=1,
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
            return build_search_plan(
                planner_kind=STRUCTURED_LLM_PLANNER_KIND,
                planner_version=request.capability_snapshot.planner_version,
                research_brief_sha256=request.research_brief_sha256,
                wave_index=request.wave_index,
                queries=tuple(queries),
                planning_limits=request.planning_limits,
                capability_snapshot=request.capability_snapshot,
                created_at=request.created_at,
                expected_research_brief_sha256=request.research_brief_sha256,
            )
        except (PlanningValidationError, ValueError, TypeError):
            raise PlannerOutputError(
                "structured planner output failed deterministic validation"
            ) from None

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


class HybridPlanner:
    """Route simple briefs directly and composite briefs once to the seam."""

    def __init__(
        self,
        structured_planner: StructuredPlannerCallable,
        *,
        direct_capability_snapshot: CapabilitySnapshot,
        structured_capability_snapshot: CapabilitySnapshot,
    ) -> None:
        if direct_capability_snapshot.planner_kind != DIRECT_PLANNER_KIND:
            raise ValueError("direct capability snapshot must use c1b-direct")
        if structured_capability_snapshot.planner_kind != STRUCTURED_LLM_PLANNER_KIND:
            raise ValueError(
                "structured capability snapshot must use c1b-llm-structured"
            )
        self._direct_capability_snapshot = direct_capability_snapshot
        self._structured_capability_snapshot = structured_capability_snapshot
        self._structured_planner = structured_planner

    async def plan(self, request: PlannerRequest) -> PlannerResult:
        if is_direct_safe(request):
            direct_request = replace(
                request,
                capability_snapshot=self._direct_capability_snapshot,
            )
            return PlannerResult(
                plan=DirectSearchPlanner().plan(direct_request),
                local_usage={"planner_calls": 0},
            )
        structured_request = replace(
            request,
            capability_snapshot=self._structured_capability_snapshot,
        )
        raw = await self._structured_planner(request=structured_request)
        return PlannerResult(
            plan=StructuredPlanParser.parse(raw, structured_request),
            local_usage={"planner_calls": 1},
        )


__all__ = [
    "DIRECT_PLANNER_KIND",
    "MAX_STRUCTURED_OUTPUT_CHARS",
    "PLANNER_VERSION",
    "STRUCTURED_LLM_PLANNER_KIND",
    "DirectSearchPlanner",
    "HybridPlanner",
    "PlannerOutputError",
    "PlannerRequest",
    "PlannerResult",
    "StructuredPlanParser",
    "StructuredPlannerCallable",
    "is_direct_safe",
]
