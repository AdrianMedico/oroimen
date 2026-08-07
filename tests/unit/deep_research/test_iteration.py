"""Deterministic PRE2-C2 tests for bounded iterative research."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from hermes.deep_research.iteration import (
    ContinuationDecision,
    GapAssessment,
    IterationInvariantError,
    IterationLimits,
    IterationPhase,
    ResearchController,
    StopReason,
)
from hermes.deep_research.iteration_store import (
    IterationStateCorruptError,
    LocalIterationStateStore,
)
from hermes.deep_research.planner import (
    STRUCTURED_LLM_PLANNER_KIND,
    PlannerRequest,
    SemanticPlanner,
)
from hermes.deep_research.planning import (
    CapabilitySnapshot,
    PlanningLimits,
    compute_research_brief_sha256,
)
from hermes.deep_research.wave_execution import SearchWaveExecutor
from hermes.services.search.protocol import SearchResult

PLANNING_LIMITS = PlanningLimits(max_queries_per_wave=4, max_query_chars=399)
CAPABILITY = CapabilitySnapshot(
    planner_kind=STRUCTURED_LLM_PLANNER_KIND,
    planner_version="1.0.0",
    max_queries_per_wave=4,
    max_query_chars=399,
    planner_provenance={
        "provider": "fake-provider",
        "model": "fake-planner",
        "version": "0.1.0",
    },
)
JOB_ID = "abcdef123456"
BRIEF = "Vergleiche die Rechtslage in Deutschland und Österreich über mehrere Zeiträume."


def _payload(decision: str, prefix: str) -> dict[str, Any]:
    count = 1 if decision == "DIRECT" else 2
    return {
        "decision": decision,
        "queries": [
            {
                "text": f"{prefix} retrieval question {ordinal}",
                "purpose": f"cover dimension {ordinal}",
                "dimension_ids": [f"dimension-{ordinal}"],
            }
            for ordinal in range(count)
        ],
    }


class _RawPlanner:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[PlannerRequest] = []

    async def __call__(self, *, request: PlannerRequest) -> dict[str, Any]:
        self.requests.append(request)
        return self.responses.pop(0)


class _FakeSearch:
    def __init__(self, *, same_source: bool = False, empty: bool = False) -> None:
        self.same_source = same_source
        self.empty = empty
        self.calls: list[str] = []

    async def __call__(
        self,
        *,
        query: str,
        intent: str,
        content: str,
        num_results: int,
    ) -> SearchResult:
        del intent, content, num_results
        self.calls.append(query)
        ordinal = 0 if self.same_source else len(self.calls)
        return SearchResult(
            results=(
                []
                if self.empty
                else [{"url": f"https://sources.test/evidence/{ordinal}"}]
            ),
            backend_used="fake",
            query=query,
            content_mode="snippet",
            original_content_mode="snippet",
            format_fallback=False,
            size_guard_chars=200_000,
            truncated=False,
        )


class _ScriptedAssessor:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.calls: list[Any] = []

    def assess(self, *, research_brief: str, state: Any, wave: Any) -> GapAssessment:
        del research_brief
        self.calls.append((state, wave))
        return self.callback(len(self.calls), state, wave)


class _Cancellation:
    def __init__(self, cancelled: bool = False) -> None:
        self.cancelled = cancelled

    def is_cancelled(self) -> bool:
        return self.cancelled


def _controller(
    *,
    raw_planner: _RawPlanner,
    assessor: _ScriptedAssessor,
    store: LocalIterationStateStore,
    search: _FakeSearch,
    clock: Any = None,
) -> ResearchController:
    return ResearchController(
        planner=SemanticPlanner(raw_planner, capability_snapshot=CAPABILITY),
        executor=SearchWaveExecutor(search),
        assessor=assessor,
        state_store=store,
        planning_limits=PLANNING_LIMITS,
        capability_snapshot=CAPABILITY,
        clock_ms=clock or (lambda: 0),
        created_at_factory=lambda: "2026-08-07T00:00:00Z",
    )


def _limits(
    *,
    max_waves: int = 3,
    max_searches: int = 4,
    max_elapsed_ms: int = 10_000,
    max_local_call_units: int = 12,
) -> IterationLimits:
    return IterationLimits(
        max_waves=max_waves,
        max_searches=max_searches,
        max_elapsed_ms=max_elapsed_ms,
        max_local_call_units=max_local_call_units,
    )


@pytest.mark.asyncio
async def test_controller_runs_multiple_waves_and_preserves_original_brief(
    tmp_path: Path,
) -> None:
    raw_planner = _RawPlanner(
        [_payload("DIRECT", "first-wave"), _payload("DECOMPOSE", "second-wave")]
    )

    def assess(call_number: int, _state: Any, wave: Any) -> GapAssessment:
        if call_number == 1:
            return GapAssessment(
                decision=ContinuationDecision.CONTINUE,
                remaining_gaps=("jurisdictional comparison",),
                material_gain=True,
                exhausted_query_ids=(wave.plan.queries[0].query_id,),
            )
        return GapAssessment(
            decision=ContinuationDecision.STOP_COVERED,
            material_gain=True,
        )

    assessor = _ScriptedAssessor(assess)
    search = _FakeSearch()
    store = LocalIterationStateStore(tmp_path)
    result = await _controller(
        raw_planner=raw_planner,
        assessor=assessor,
        store=store,
        search=search,
    ).run(JOB_ID, BRIEF, limits=_limits())

    assert result.state.phase is IterationPhase.STOPPED
    assert result.state.stop_reason is StopReason.OBJECTIVE_COVERED
    assert [wave.wave_index for wave in result.state.waves] == [0, 1]
    assert [wave.plan.planning_decision for wave in result.state.waves] == [
        "DIRECT",
        "DECOMPOSE",
    ]
    assert [request.research_brief for request in raw_planner.requests] == [BRIEF, BRIEF]
    assert raw_planner.requests[1].prior_source_refs
    assert raw_planner.requests[1].open_gaps == ("jurisdictional comparison",)
    assert raw_planner.requests[1].exhausted_query_ids
    assert result.state.accounting.planner_calls == 2
    assert result.state.accounting.search_calls == 3
    assert result.state.accounting.assessment_calls == 2

    persisted = (tmp_path / "research_iterations" / f"{JOB_ID}.iteration.json").read_text(
        encoding="utf-8"
    )
    assert BRIEF not in persisted
    assert compute_research_brief_sha256(BRIEF) in persisted


@pytest.mark.asyncio
async def test_recovery_reuses_plan_persisted_before_cancellation(tmp_path: Path) -> None:
    raw_planner = _RawPlanner([_payload("DIRECT", "recoverable")])

    class _CancellingExecutor:
        async def execute(self, plan: Any) -> Any:
            del plan
            raise asyncio.CancelledError

    assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: GapAssessment(
            decision=ContinuationDecision.STOP_COVERED,
            material_gain=True,
        )
    )
    store = LocalIterationStateStore(tmp_path)
    first = ResearchController(
        planner=SemanticPlanner(raw_planner, capability_snapshot=CAPABILITY),
        executor=_CancellingExecutor(),
        assessor=assessor,
        state_store=store,
        planning_limits=PLANNING_LIMITS,
        capability_snapshot=CAPABILITY,
        clock_ms=lambda: 0,
        created_at_factory=lambda: "2026-08-07T00:00:00Z",
    )

    with pytest.raises(asyncio.CancelledError):
        await first.run(JOB_ID, BRIEF, limits=_limits(max_waves=1, max_searches=1))

    checkpoint = store.load(JOB_ID)
    assert checkpoint is not None
    assert checkpoint.phase is IterationPhase.PLAN_PERSISTED
    assert checkpoint.active_plan is not None
    assert checkpoint.accounting.planner_calls == 1
    assert checkpoint.accounting.search_calls == 0

    class _PlannerMustNotRun:
        async def plan(self, request: PlannerRequest) -> Any:
            del request
            raise AssertionError("recovery must reuse the persisted plan")

    search = _FakeSearch()
    recovered = ResearchController(
        planner=_PlannerMustNotRun(),
        executor=SearchWaveExecutor(search),
        assessor=assessor,
        state_store=store,
        planning_limits=PLANNING_LIMITS,
        capability_snapshot=CAPABILITY,
        clock_ms=lambda: 0,
        created_at_factory=lambda: "2026-08-07T00:00:00Z",
    )
    result = await recovered.run(JOB_ID, BRIEF, limits=_limits(max_waves=1, max_searches=1))

    assert result.state.stop_reason is StopReason.OBJECTIVE_COVERED
    assert result.state.accounting.planner_calls == 1
    assert result.state.accounting.search_calls == 1
    assert len(search.calls) == 1


@pytest.mark.asyncio
async def test_stop_reasons_cover_budget_no_progress_and_cooperative_cancel(
    tmp_path: Path,
) -> None:
    budget_assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: GapAssessment(
            decision=ContinuationDecision.CONTINUE,
            remaining_gaps=("still open",),
            material_gain=True,
        )
    )
    budget_search = _FakeSearch()
    budget_result = await _controller(
        raw_planner=_RawPlanner([_payload("DIRECT", "budget")]),
        assessor=budget_assessor,
        store=LocalIterationStateStore(tmp_path / "budget"),
        search=budget_search,
    ).run(
        JOB_ID,
        BRIEF,
        limits=_limits(max_waves=1, max_searches=1, max_local_call_units=3),
    )
    assert budget_result.state.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert budget_result.state.accounting.assessment_calls == 1

    no_progress_assessor = _ScriptedAssessor(
        lambda call_number, _state, _wave: GapAssessment(
            decision=ContinuationDecision.CONTINUE,
            remaining_gaps=("still open",),
            material_gain=call_number == 1,
        )
    )
    no_progress_result = await _controller(
        raw_planner=_RawPlanner(
            [_payload("DIRECT", "first unique"), _payload("DIRECT", "new query same source")]
        ),
        assessor=no_progress_assessor,
        store=LocalIterationStateStore(tmp_path / "no-progress"),
        search=_FakeSearch(same_source=True),
    ).run(JOB_ID, BRIEF, limits=_limits(max_waves=2, max_searches=2, max_local_call_units=6))
    assert no_progress_result.state.stop_reason is StopReason.NO_MATERIAL_GAIN
    assert no_progress_result.state.accounting.assessment_calls == 2

    empty_assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: GapAssessment(
            decision=ContinuationDecision.STOP_NO_MATERIAL_GAIN,
            material_gain=False,
        )
    )
    empty_result = await _controller(
        raw_planner=_RawPlanner([_payload("DIRECT", "empty evidence")]),
        assessor=empty_assessor,
        store=LocalIterationStateStore(tmp_path / "empty"),
        search=_FakeSearch(empty=True),
    ).run(JOB_ID, BRIEF, limits=_limits(max_waves=1, max_searches=1))
    assert empty_result.state.stop_reason is StopReason.NO_MATERIAL_GAIN

    elapsed_calls = iter((0, 100))
    elapsed_assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: pytest.fail("assessor must not run")
    )
    elapsed_result = await _controller(
        raw_planner=_RawPlanner([_payload("DIRECT", "elapsed")]),
        assessor=elapsed_assessor,
        store=LocalIterationStateStore(tmp_path / "elapsed"),
        search=_FakeSearch(),
        clock=lambda: next(elapsed_calls),
    ).run(
        JOB_ID,
        BRIEF,
        limits=_limits(max_elapsed_ms=50),
    )
    assert elapsed_result.state.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert elapsed_result.state.accounting.total_local_call_units == 0

    local_budget_assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: pytest.fail("assessment must not run")
    )
    local_budget_result = await _controller(
        raw_planner=_RawPlanner([_payload("DIRECT", "local budget")]),
        assessor=local_budget_assessor,
        store=LocalIterationStateStore(tmp_path / "local-budget"),
        search=_FakeSearch(),
    ).run(
        JOB_ID,
        BRIEF,
        limits=_limits(max_waves=1, max_searches=1, max_local_call_units=2),
    )
    assert local_budget_result.state.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert local_budget_result.state.accounting.planner_calls == 1
    assert local_budget_result.state.accounting.search_calls == 1
    assert local_budget_result.state.accounting.assessment_calls == 0

    cancelled = _Cancellation(cancelled=True)
    cancel_assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: pytest.fail("assessor must not run")
    )
    cancel_result = await _controller(
        raw_planner=_RawPlanner([_payload("DIRECT", "never run")]),
        assessor=cancel_assessor,
        store=LocalIterationStateStore(tmp_path / "cancelled"),
        search=_FakeSearch(),
    ).run(JOB_ID, BRIEF, limits=_limits(), cancellation=cancelled)
    assert cancel_result.state.stop_reason is StopReason.CANCELLED
    assert cancel_result.state.accounting.total_local_call_units == 0


@pytest.mark.asyncio
async def test_controller_rejects_non_semantic_plan_and_repeated_query(
    tmp_path: Path,
) -> None:
    class _InvalidPlanner:
        async def plan(self, request: PlannerRequest) -> Any:
            del request
            return object()

    assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: pytest.fail("assessor must not run")
    )
    controller = ResearchController(
        planner=_InvalidPlanner(),
        executor=SearchWaveExecutor(_FakeSearch()),
        assessor=assessor,
        state_store=LocalIterationStateStore(tmp_path / "invalid"),
        planning_limits=PLANNING_LIMITS,
        capability_snapshot=CAPABILITY,
        clock_ms=lambda: 0,
        created_at_factory=lambda: "2026-08-07T00:00:00Z",
    )
    with pytest.raises(IterationInvariantError, match="PlannerResult"):
        await controller.run(JOB_ID, BRIEF, limits=_limits())

    repeated_assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: GapAssessment(
            decision=ContinuationDecision.CONTINUE,
            remaining_gaps=("still open",),
            material_gain=True,
        )
    )
    repeated_controller = ResearchController(
        planner=SemanticPlanner(
            _RawPlanner(
                [_payload("DIRECT", "same query"), _payload("DIRECT", "same query")]
            ),
            capability_snapshot=CAPABILITY,
        ),
        executor=SearchWaveExecutor(_FakeSearch()),
        assessor=repeated_assessor,
        state_store=LocalIterationStateStore(tmp_path / "repeated"),
        planning_limits=PLANNING_LIMITS,
        capability_snapshot=CAPABILITY,
        clock_ms=lambda: 0,
        created_at_factory=lambda: "2026-08-07T00:00:00Z",
    )
    with pytest.raises(IterationInvariantError, match="searched query"):
        await repeated_controller.run(
            JOB_ID,
            BRIEF,
            limits=_limits(max_waves=2, max_searches=2, max_local_call_units=6),
        )


def test_iteration_state_store_is_atomic_and_fail_closed(tmp_path: Path) -> None:
    store = LocalIterationStateStore(tmp_path)
    path = tmp_path / "research_iterations" / f"{JOB_ID}.iteration.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(IterationStateCorruptError):
        store.load(JOB_ID)
