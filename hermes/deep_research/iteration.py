"""Bounded, recoverable multi-wave Deep Research coordination.

This is an internal C2 domain seam.  It coordinates the already validated
semantic planner and one-wave executor, but it does not select providers,
invoke live services, or expose a public Research Brief API.

The controller owns deterministic authorization of continuation.  An
injected planner and gap assessor may propose; the controller owns limits,
brief-hash binding, exhausted-search rules, evidence gain, cancellation,
idempotent checkpoints, and terminal STOP reasons.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from hermes.deep_research.iteration_assessment import (
    apply_assessment,
    validate_assessment,
)
from hermes.deep_research.iteration_evidence import (
    count_search_calls,
    has_new_evidence,
    record_wave,
    source_evidence,
)
from hermes.deep_research.iteration_state import (
    MAX_LOCAL_CALL_UNITS,
    MAX_SEARCHES_PER_JOB,
    MAX_SOURCES_PER_JOB,
    ContinuationDecision,
    GapAssessment,
    IterationInvariantError,
    IterationLimits,
    IterationPhase,
    ResearchAccounting,
    ResearchIterationState,
    StopReason,
    WaveRecord,
)
from hermes.deep_research.planner import (
    STRUCTURED_LLM_PLANNER_KIND,
    PlannerRequest,
    PlannerResult,
)
from hermes.deep_research.planning import (
    CapabilitySnapshot,
    PlanningLimits,
    SearchObservation,
    compute_research_brief_sha256,
    normalize_query_text,
    validate_search_plan,
)
from hermes.deep_research.wave_execution import (
    SearchWaveExecutor,
    WaveExecutionCancelled,
    WaveExecutionResult,
)


class IterationStateStore(Protocol):
    """Atomic state checkpoint boundary."""

    def load(self, job_id: str) -> ResearchIterationState | None: ...

    def write(self, job_id: str, state: ResearchIterationState) -> None: ...

    def claim(self, job_id: str) -> None: ...

    def release(self, job_id: str) -> None: ...


class CancellationProbe(Protocol):
    def is_cancelled(self) -> bool: ...


class SemanticWavePlanner(Protocol):
    async def plan(self, request: PlannerRequest) -> PlannerResult: ...


class GapAssessor(Protocol):
    async def assess(
        self,
        *,
        research_brief: str,
        state: ResearchIterationState,
        wave: WaveRecord,
    ) -> GapAssessment: ...


@dataclass(frozen=True)
class ResearchRunResult:
    state: ResearchIterationState


class ResearchController:
    """Run bounded waves until a deterministic terminal STOP is reached."""

    def __init__(
        self,
        *,
        planner: SemanticWavePlanner,
        executor: SearchWaveExecutor,
        assessor: GapAssessor,
        state_store: IterationStateStore,
        planning_limits: PlanningLimits,
        capability_snapshot: CapabilitySnapshot,
        clock_ms: Callable[[], int] | None = None,
        created_at_factory: Callable[[], str] | None = None,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._assessor = assessor
        self._state_store = state_store
        self._planning_limits = planning_limits
        self._capability_snapshot = capability_snapshot
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._created_at_factory = created_at_factory or (
            lambda: str(self._clock_ms())
        )

    async def run(
        self,
        job_id: str,
        research_brief: str,
        *,
        limits: IterationLimits,
        cancellation: CancellationProbe | None = None,
    ) -> ResearchRunResult:
        self._state_store.claim(job_id)
        try:
            return await self._run_claimed(
                job_id,
                research_brief,
                limits=limits,
                cancellation=cancellation,
            )
        finally:
            self._state_store.release(job_id)

    async def _run_claimed(
        self,
        job_id: str,
        research_brief: str,
        *,
        limits: IterationLimits,
        cancellation: CancellationProbe | None = None,
    ) -> ResearchRunResult:
        brief_hash = compute_research_brief_sha256(research_brief)
        state = self._state_store.load(job_id)
        if state is None:
            state = ResearchIterationState.new(
                job_id=job_id,
                research_brief_sha256=brief_hash,
                limits=limits,
                planning_limits=self._planning_limits,
                capability_snapshot=self._capability_snapshot,
                started_at_ms=self._clock_ms(),
            )
            self._state_store.write(job_id, state)
        else:
            self._validate_loaded_state(state, job_id, brief_hash, limits)

        if state.phase is IterationPhase.STOPPED:
            return ResearchRunResult(state=state)

        # No provider idempotency key exists at this internal seam.  An
        # interrupted dispatched call is therefore never replayed silently;
        # recovery records a truthful cancellation STOP instead.
        if (
            state.planning_inflight
            or state.assessment_inflight
            or state.active_inflight_query_id is not None
        ):
            state = self._stop(state, StopReason.CANCELLED)
            return ResearchRunResult(state=state)

        while state.phase is not IterationPhase.STOPPED:
            if self._cancelled(cancellation):
                state = self._stop(state, StopReason.CANCELLED)
                break
            if self._elapsed(state) >= limits.max_elapsed_ms:
                state = self._stop(state, StopReason.BUDGET_EXHAUSTED)
                break

            if state.phase is IterationPhase.READY_TO_PLAN:
                if not self._can_start_wave(state):
                    state = self._stop(state, StopReason.BUDGET_EXHAUSTED)
                    break
                request = PlannerRequest(
                    research_brief=research_brief,
                    planning_limits=self._planning_limits,
                    capability_snapshot=self._capability_snapshot,
                    created_at=self._created_at_factory(),
                    wave_index=state.next_wave_index,
                    prior_source_refs=state.source_refs,
                    open_gaps=state.open_gaps,
                    exhausted_query_ids=state.exhausted_query_ids,
                    exhausted_source_refs=state.exhausted_source_refs,
                )
                planner_state = self._replace_state(
                    state,
                    planning_inflight=True,
                    accounting=ResearchAccounting(
                        planner_calls=state.accounting.planner_calls + 1,
                        search_calls=state.accounting.search_calls,
                        assessment_calls=state.accounting.assessment_calls,
                    ),
                )
                self._state_store.write(job_id, planner_state)
                try:
                    planned = await asyncio.wait_for(
                        self._planner.plan(request),
                        timeout=self._remaining_seconds(planner_state, limits),
                    )
                except TimeoutError:
                    state = self._stop(planner_state, StopReason.BUDGET_EXHAUSTED)
                    break
                self._validate_planner_result(
                    planned,
                    planner_state,
                    brief_hash,
                    self._capability_snapshot,
                )
                state = self._replace_state(
                    planner_state,
                    phase=IterationPhase.PLAN_PERSISTED,
                    active_plan=planned.plan,
                    planning_inflight=False,
                )
                self._state_store.write(job_id, state)
                continue

            if state.phase is IterationPhase.PLAN_PERSISTED:
                plan = state.active_plan
                if plan is None:
                    raise IterationInvariantError("plan_persisted state has no active plan")
                if self._cancelled(cancellation):
                    state = self._stop(state, StopReason.CANCELLED)
                    break
                if not self._can_execute_plan(state):
                    state = self._stop(state, StopReason.BUDGET_EXHAUSTED)
                    break
                assert state is not None
                state_for_execution: ResearchIterationState = state

                def checkpoint_observation(observation: SearchObservation) -> None:
                    nonlocal state_for_execution
                    state_for_execution = self._checkpoint_observation(
                        job_id=job_id,
                        state=state_for_execution,
                        observation=observation,
                        limits=limits,
                    )

                def checkpoint_dispatch(query: Any) -> None:
                    nonlocal state_for_execution
                    state_for_execution = self._checkpoint_dispatch(
                        job_id=job_id,
                        state=state_for_execution,
                        query=query,
                        limits=limits,
                    )

                try:
                    execution = await asyncio.wait_for(
                        self._executor.execute(
                            plan,
                            completed_observations=state_for_execution.active_observations,
                            on_observation=checkpoint_observation,
                            on_dispatch=checkpoint_dispatch,
                            cancellation=(
                                None
                                if cancellation is None
                                else cancellation.is_cancelled
                            ),
                        ),
                        timeout=self._remaining_seconds(state, limits),
                    )
                except WaveExecutionCancelled:
                    state = self._stop(state_for_execution, StopReason.CANCELLED)
                    break
                except TimeoutError:
                    state = state_for_execution
                    state = self._stop(state, StopReason.BUDGET_EXHAUSTED)
                    break
                state = state_for_execution
                search_calls = self._count_search_calls(execution)
                if (
                    state.active_inflight_query_id is not None
                    or len(state.active_observations) != len(plan.queries)
                ):
                    raise IterationInvariantError(
                        "executor completed without durable observation checkpoints"
                    )
                wave = WaveRecord.from_result(execution)
                if state.accounting.search_calls + search_calls - len(
                    state.active_observations
                ) > limits.max_searches:
                    raise IterationInvariantError("search accounting exceeded its reservation")
                state = self._record_wave(state, wave)
                self._state_store.write(job_id, state)
                continue

            if state.phase is IterationPhase.ASSESSMENT_PENDING:
                if not state.waves:
                    raise IterationInvariantError("assessment_pending state has no waves")
                if state.accounting.total_local_call_units >= limits.max_local_call_units:
                    state = self._stop(state, StopReason.BUDGET_EXHAUSTED)
                    break
                wave = state.waves[-1]
                assessment_state = self._replace_state(
                    state,
                    assessment_inflight=True,
                    accounting=ResearchAccounting(
                        planner_calls=state.accounting.planner_calls,
                        search_calls=state.accounting.search_calls,
                        assessment_calls=state.accounting.assessment_calls + 1,
                    ),
                )
                self._state_store.write(job_id, assessment_state)
                try:
                    proposal = await asyncio.wait_for(
                        self._assessor.assess(
                            research_brief=research_brief,
                            state=assessment_state,
                            wave=wave,
                        ),
                        timeout=self._remaining_seconds(assessment_state, limits),
                    )
                except TimeoutError:
                    state = self._stop(assessment_state, StopReason.BUDGET_EXHAUSTED)
                    break
                if self._cancelled(cancellation):
                    state = self._stop(assessment_state, StopReason.CANCELLED)
                    break
                completed_assessment_state = self._replace_state(
                    assessment_state,
                    phase=IterationPhase.READY_TO_PLAN,
                    assessment_inflight=False,
                )
                self._validate_assessment(proposal, completed_assessment_state, wave)
                state = self._apply_assessment(
                    completed_assessment_state,
                    proposal,
                    completed_assessment_state.accounting,
                )
                self._state_store.write(job_id, state)
                continue

            raise IterationInvariantError(f"unsupported phase: {state.phase}")

        return ResearchRunResult(state=state)

    def _validate_loaded_state(
        self,
        state: ResearchIterationState,
        job_id: str,
        brief_hash: str,
        limits: IterationLimits,
    ) -> None:
        if state.job_id != job_id or state.research_brief_sha256 != brief_hash:
            raise IterationInvariantError("iteration state is bound to a different brief/job")
        if state.limits != limits:
            raise IterationInvariantError("iteration limits changed during recovery")
        if state.planning_limits != self._planning_limits:
            raise IterationInvariantError("planning limits changed during recovery")
        if state.capability_snapshot.to_dict() != self._capability_snapshot.to_dict():
            raise IterationInvariantError("planner capability changed during recovery")

    def _elapsed(self, state: ResearchIterationState) -> int:
        return max(0, self._clock_ms() - state.started_at_ms)

    def _remaining_seconds(
        self,
        state: ResearchIterationState,
        limits: IterationLimits,
    ) -> float:
        remaining_ms = limits.max_elapsed_ms - self._elapsed(state)
        if remaining_ms <= 0:
            return 0.001
        return remaining_ms / 1000

    @staticmethod
    def _cancelled(cancellation: CancellationProbe | None) -> bool:
        return cancellation is not None and cancellation.is_cancelled()

    @staticmethod
    def _replace_state(state: ResearchIterationState, **changes: Any) -> ResearchIterationState:
        return replace(state, **changes)

    def _stop(self, state: ResearchIterationState, reason: StopReason) -> ResearchIterationState:
        stopped = self._replace_state(
            state,
            phase=IterationPhase.STOPPED,
            stop_reason=reason,
        )
        self._state_store.write(state.job_id, stopped)
        return stopped

    def _can_start_wave(self, state: ResearchIterationState) -> bool:
        if state.next_wave_index >= state.limits.max_waves:
            return False
        if state.accounting.search_calls >= state.limits.max_searches:
            return False
        return state.accounting.total_local_call_units + 2 <= state.limits.max_local_call_units

    def _can_execute_plan(self, state: ResearchIterationState) -> bool:
        assert state.active_plan is not None
        query_count = len(state.active_plan.queries)
        remaining_queries = query_count - len(state.active_observations)
        already_reserved = 1 if state.active_inflight_query_id is not None else 0
        return (
            state.accounting.search_calls + remaining_queries - already_reserved
            <= state.limits.max_searches
            and state.accounting.total_local_call_units + remaining_queries - already_reserved
            <= state.limits.max_local_call_units
        )

    def _checkpoint_dispatch(
        self,
        *,
        job_id: str,
        state: ResearchIterationState,
        query: Any,
        limits: IterationLimits,
    ) -> ResearchIterationState:
        if state.active_plan is None:
            raise IterationInvariantError("query dispatch requires an active plan")
        if state.active_inflight_query_id is not None:
            raise IterationInvariantError("a query is already in flight")
        ordered_queries = sorted(state.active_plan.queries, key=lambda item: item.ordinal)
        next_ordinal = len(state.active_observations)
        if next_ordinal >= len(ordered_queries):
            raise IterationInvariantError("executor dispatched too many queries")
        if query.query_id != ordered_queries[next_ordinal].query_id:
            raise IterationInvariantError("executor dispatch order drifted")
        if state.accounting.search_calls + 1 > limits.max_searches:
            raise IterationInvariantError("executor exceeded the search-call budget")
        if state.accounting.total_local_call_units + 1 > limits.max_local_call_units:
            raise IterationInvariantError("executor exceeded the local-call budget")
        updated = self._replace_state(
            state,
            active_inflight_query_id=query.query_id,
            accounting=ResearchAccounting(
                planner_calls=state.accounting.planner_calls,
                search_calls=state.accounting.search_calls + 1,
                assessment_calls=state.accounting.assessment_calls,
            ),
        )
        self._state_store.write(job_id, updated)
        return updated

    def _checkpoint_observation(
        self,
        *,
        job_id: str,
        state: ResearchIterationState,
        observation: SearchObservation,
        limits: IterationLimits,
    ) -> ResearchIterationState:
        if state.active_plan is None:
            raise IterationInvariantError("partial observation requires an active plan")
        ordered_queries = sorted(
            state.active_plan.queries,
            key=lambda query: query.ordinal,
        )
        next_ordinal = len(state.active_observations)
        if next_ordinal >= len(ordered_queries):
            raise IterationInvariantError("executor returned too many observations")
        if observation.query_id != ordered_queries[next_ordinal].query_id:
            raise IterationInvariantError("executor observation order drifted")
        if dict(observation.local_usage) != {"search_calls": 1}:
            raise IterationInvariantError("executor observation accounting drifted")
        if state.active_inflight_query_id != observation.query_id:
            raise IterationInvariantError("observation has no matching dispatched query")
        active_observations = (*state.active_observations, observation)
        active_source_refs, active_source_query_ids = self._source_evidence(
            active_observations
        )
        updated = self._replace_state(
            state,
            active_observations=active_observations,
            active_source_refs=active_source_refs,
            active_source_query_ids=active_source_query_ids,
            active_inflight_query_id=None,
        )
        self._state_store.write(job_id, updated)
        return updated

    @staticmethod
    def _source_evidence(
        observations: tuple[SearchObservation, ...],
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        return source_evidence(observations)

    @staticmethod
    def _validate_planner_result(
        planned: PlannerResult,
        state: ResearchIterationState,
        brief_hash: str,
        capability_snapshot: CapabilitySnapshot,
    ) -> None:
        if not isinstance(planned, PlannerResult):
            raise IterationInvariantError("semantic planner must return PlannerResult")
        if dict(planned.local_usage) != {"planner_calls": 1}:
            raise IterationInvariantError("planner accounting must report exactly one call")
        plan = planned.plan
        if plan.wave_index != state.next_wave_index:
            raise IterationInvariantError("planner returned an unexpected wave index")
        if plan.research_brief_sha256 != brief_hash:
            raise IterationInvariantError("planner returned a plan for another brief")
        if plan.capability_snapshot.to_dict() != capability_snapshot.to_dict():
            raise IterationInvariantError("planner capability provenance drifted")
        validate_search_plan(plan, expected_research_brief_sha256=brief_hash)
        exhausted = set(state.exhausted_query_texts)
        for query in plan.queries:
            if normalize_query_text(query.text) in exhausted:
                raise IterationInvariantError("planner repeated an exhausted query")
            if normalize_query_text(query.text) in state.searched_query_texts:
                raise IterationInvariantError("planner repeated a searched query")
        if plan.planner_kind != STRUCTURED_LLM_PLANNER_KIND:
            raise IterationInvariantError(
                "iterative research requires the semantic structured planner"
            )

    @staticmethod
    def _count_search_calls(execution: WaveExecutionResult) -> int:
        return count_search_calls(execution)

    @staticmethod
    def _record_wave(
        state: ResearchIterationState,
        wave: WaveRecord,
    ) -> ResearchIterationState:
        return record_wave(state, wave)

    @staticmethod
    def _validate_assessment(
        assessment: GapAssessment,
        state: ResearchIterationState,
        wave: WaveRecord,
    ) -> None:
        validate_assessment(assessment, state, wave)

    @staticmethod
    def _has_new_evidence(
        state: ResearchIterationState,
        wave: WaveRecord,
    ) -> bool:
        return has_new_evidence(state, wave)

    @staticmethod
    def _apply_assessment(
        state: ResearchIterationState,
        assessment: GapAssessment,
        accounting: ResearchAccounting,
    ) -> ResearchIterationState:
        return apply_assessment(state, assessment, accounting)



__all__ = [
    "MAX_LOCAL_CALL_UNITS",
        "MAX_SEARCHES_PER_JOB",
        "MAX_SOURCES_PER_JOB",
    "ContinuationDecision",
    "GapAssessment",
    "GapAssessor",
    "IterationInvariantError",
    "IterationLimits",
    "IterationPhase",
    "IterationStateStore",
    "ResearchAccounting",
    "ResearchController",
    "ResearchIterationState",
    "ResearchRunResult",
    "SemanticWavePlanner",
    "StopReason",
    "WaveRecord",
]
