"""One-wave orchestration with durable plan reuse.

This module composes the provider-free plan store and wave executor without
creating a ResearchController or changing the live Deep Research service.
It is deliberately bounded: one call loads or creates one SearchPlan and
executes exactly that plan once.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from hermes.deep_research.plan_store import (
    LocalPlanStore,
    PlanCapabilitySnapshotMismatchError,
    PlanNotFoundError,
)
from hermes.deep_research.planner import STRUCTURED_LLM_PLANNER_KIND
from hermes.deep_research.planning import (
    KNOWN_PLANNING_DECISIONS,
    CapabilitySnapshot,
    SearchPlan,
    validate_search_plan,
)
from hermes.deep_research.wave_execution import (
    SearchWaveExecutor,
    WaveExecutionResult,
)


class SearchPlanFactory(Protocol):
    """Provider-replaceable seam for creating one bounded plan."""

    def __call__(self) -> SearchPlan | Awaitable[SearchPlan]: ...


@dataclass(frozen=True)
class WaveRunResult:
    """The reused-or-created plan and its single execution result."""

    plan: SearchPlan
    execution: WaveExecutionResult
    plan_reused: bool


class SearchWaveRunner:
    """Load or create one plan, then execute it once.

    A persisted plan is always preferred.  A missing plan invokes the factory
    exactly once and writes the validated result before any search dispatch.
    Corrupt, incompatible, or conflicting artifacts fail closed and are not
    silently replaced.
    """

    def __init__(
        self,
        *,
        plan_store: LocalPlanStore,
        executor: SearchWaveExecutor,
    ) -> None:
        self._plan_store = plan_store
        self._executor = executor

    async def run(
        self,
        job_id: str,
        *,
        expected_research_brief_sha256: str,
        expected_capability_snapshot: CapabilitySnapshot,
        plan_factory: SearchPlanFactory | Callable[[], SearchPlan | Awaitable[SearchPlan]],
    ) -> WaveRunResult:
        """Run one bounded wave, reusing a valid persisted plan when present."""

        try:
            plan = self._plan_store.load(
                job_id,
                expected_research_brief_sha256=expected_research_brief_sha256,
                expected_capability_snapshot=expected_capability_snapshot,
            )
            plan_reused = True
        except PlanNotFoundError:
            candidate = plan_factory()
            plan = await candidate if inspect.isawaitable(candidate) else candidate
            self._validate_factory_plan(
                job_id=job_id,
                plan=plan,
                expected_research_brief_sha256=expected_research_brief_sha256,
                expected_capability_snapshot=expected_capability_snapshot,
            )
            self._plan_store.write(job_id, plan)
            plan_reused = False

        self._require_semantic_plan(
            job_id=job_id,
            plan=plan,
            expected_capability_snapshot=expected_capability_snapshot,
        )
        execution = await self._executor.execute(plan)
        return WaveRunResult(
            plan=plan,
            execution=execution,
            plan_reused=plan_reused,
        )

    @staticmethod
    def _validate_factory_plan(
        *,
        job_id: str,
        plan: SearchPlan,
        expected_research_brief_sha256: str,
        expected_capability_snapshot: CapabilitySnapshot,
    ) -> None:
        if not isinstance(plan, SearchPlan):
            raise TypeError("plan_factory must return a SearchPlan")
        validate_search_plan(
            plan,
            expected_research_brief_sha256=expected_research_brief_sha256,
        )
        if plan.capability_snapshot.to_dict() != expected_capability_snapshot.to_dict():
            raise PlanCapabilitySnapshotMismatchError(job_id)
        SearchWaveRunner._require_semantic_plan(
            job_id=job_id,
            plan=plan,
            expected_capability_snapshot=expected_capability_snapshot,
        )

    @staticmethod
    def _require_semantic_plan(
        *,
        job_id: str,
        plan: SearchPlan,
        expected_capability_snapshot: CapabilitySnapshot,
    ) -> None:
        if (
            expected_capability_snapshot.planner_kind != STRUCTURED_LLM_PLANNER_KIND
            or plan.planner_kind != STRUCTURED_LLM_PLANNER_KIND
            or plan.planning_decision not in KNOWN_PLANNING_DECISIONS
        ):
            raise PlanCapabilitySnapshotMismatchError(job_id)


__all__ = ["SearchPlanFactory", "SearchWaveRunner", "WaveRunResult"]
