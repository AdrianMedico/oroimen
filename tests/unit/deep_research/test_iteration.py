"""Deterministic PRE2-C2 tests for bounded iterative research."""

from __future__ import annotations

import asyncio
import json
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
    ResearchIterationState,
    StopReason,
)
from hermes.deep_research.iteration_store import (
    IterationStateBusyError,
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
                else [
                    {
                        "url": f"https://sources.test/evidence/{ordinal}",
                        "title": f"Evidence title {ordinal}",
                        "snippet": f"Bounded evidence snippet for {query}",
                    }
                ]
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

    async def assess(
        self, *, research_brief: str, state: Any, wave: Any
    ) -> GapAssessment:
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

    persisted_payload = json.loads(persisted)
    persisted_payload["waves"][0]["source_query_ids"] = {}
    (tmp_path / "research_iterations" / f"{JOB_ID}.iteration.json").write_text(
        json.dumps(persisted_payload),
        encoding="utf-8",
    )
    with pytest.raises(IterationStateCorruptError):
        store.load(JOB_ID)

    store.write(JOB_ID, result.state)
    persisted_payload = json.loads(persisted)
    persisted_payload["source_query_ids"] = {}
    (tmp_path / "research_iterations" / f"{JOB_ID}.iteration.json").write_text(
        json.dumps(persisted_payload),
        encoding="utf-8",
    )
    with pytest.raises(IterationStateCorruptError):
        store.load(JOB_ID)


@pytest.mark.asyncio
async def test_cancellation_between_queries_stops_before_next_dispatch(
    tmp_path: Path,
) -> None:
    cancellation = _Cancellation()

    class _CancelAfterFirstSearch(_FakeSearch):
        async def __call__(self, **kwargs: Any) -> SearchResult:
            result = await super().__call__(**kwargs)
            cancellation.cancelled = True
            return result

    assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: pytest.fail("assessment must not run")
    )
    search = _CancelAfterFirstSearch()
    result = await _controller(
        raw_planner=_RawPlanner([_payload("DECOMPOSE", "cancel-between")]),
        assessor=assessor,
        store=LocalIterationStateStore(tmp_path),
        search=search,
    ).run(
        JOB_ID,
        BRIEF,
        limits=_limits(max_waves=1, max_searches=2, max_local_call_units=4),
        cancellation=cancellation,
    )

    assert result.state.phase is IterationPhase.STOPPED
    assert result.state.stop_reason is StopReason.CANCELLED
    assert len(search.calls) == 1
    assert len(result.state.active_observations) == 1


@pytest.mark.asyncio
async def test_recovery_does_not_replay_uncertain_dispatch(tmp_path: Path) -> None:
    raw_planner = _RawPlanner([_payload("DECOMPOSE", "recoverable")])

    class _CancelOnSecondSearch(_FakeSearch):
        async def __call__(self, **kwargs: Any) -> SearchResult:
            if len(self.calls) == 1:
                self.calls.append(kwargs["query"])
                raise asyncio.CancelledError
            return await super().__call__(**kwargs)

    assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: GapAssessment(
            decision=ContinuationDecision.STOP_COVERED,
            material_gain=True,
        )
    )
    store = LocalIterationStateStore(tmp_path)
    first = ResearchController(
        planner=SemanticPlanner(raw_planner, capability_snapshot=CAPABILITY),
        executor=SearchWaveExecutor(_CancelOnSecondSearch()),
        assessor=assessor,
        state_store=store,
        planning_limits=PLANNING_LIMITS,
        capability_snapshot=CAPABILITY,
        clock_ms=lambda: 0,
        created_at_factory=lambda: "2026-08-07T00:00:00Z",
    )

    with pytest.raises(asyncio.CancelledError):
        await first.run(
            JOB_ID,
            BRIEF,
            limits=_limits(max_waves=1, max_searches=2, max_local_call_units=4),
        )

    checkpoint = store.load(JOB_ID)
    assert checkpoint is not None
    assert checkpoint.phase is IterationPhase.PLAN_PERSISTED
    assert checkpoint.active_plan is not None
    assert len(checkpoint.active_observations) == 1
    assert checkpoint.accounting.planner_calls == 1
    assert checkpoint.accounting.search_calls == 2
    assert checkpoint.active_inflight_query_id == checkpoint.active_plan.queries[1].query_id

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
    result = await recovered.run(
        JOB_ID,
        BRIEF,
        limits=_limits(max_waves=1, max_searches=2, max_local_call_units=4),
    )

    assert result.state.stop_reason is StopReason.CANCELLED
    assert result.state.accounting.planner_calls == 1
    assert result.state.accounting.search_calls == 2
    assert search.calls == []


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

    invalid_coverage = _ScriptedAssessor(
        lambda _call, _state, _wave: GapAssessment(
            decision=ContinuationDecision.STOP_COVERED,
            material_gain=True,
        )
    )
    with pytest.raises(IterationInvariantError, match="new bounded evidence"):
        await _controller(
            raw_planner=_RawPlanner([_payload("DIRECT", "false coverage")]),
            assessor=invalid_coverage,
            store=LocalIterationStateStore(tmp_path / "false-coverage"),
            search=_FakeSearch(empty=True),
        ).run(JOB_ID, BRIEF, limits=_limits(max_waves=1, max_searches=1))

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

    exact_assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: GapAssessment(
            decision=ContinuationDecision.STOP_COVERED,
            material_gain=True,
        )
    )
    exact_result = await _controller(
        raw_planner=_RawPlanner([_payload("DECOMPOSE", "exact budget")]),
        assessor=exact_assessor,
        store=LocalIterationStateStore(tmp_path / "exact-budget"),
        search=_FakeSearch(),
    ).run(
        JOB_ID,
        BRIEF,
        limits=_limits(max_waves=1, max_searches=2, max_local_call_units=4),
    )
    assert exact_result.state.stop_reason is StopReason.OBJECTIVE_COVERED
    assert exact_result.state.accounting.search_calls == 2

    under_assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: pytest.fail("assessment must not run")
    )
    under_result = await _controller(
        raw_planner=_RawPlanner([_payload("DECOMPOSE", "under budget")]),
        assessor=under_assessor,
        store=LocalIterationStateStore(tmp_path / "under-budget"),
        search=_FakeSearch(),
    ).run(
        JOB_ID,
        BRIEF,
        limits=_limits(max_waves=1, max_searches=2, max_local_call_units=3),
    )
    assert under_result.state.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert under_result.state.accounting.planner_calls == 1
    assert under_result.state.accounting.search_calls == 2
    assert under_result.state.accounting.assessment_calls == 0

    class _SlowSearch(_FakeSearch):
        async def __call__(self, **kwargs: Any) -> SearchResult:
            await asyncio.sleep(0.05)
            return await super().__call__(**kwargs)

    timed_executor_assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: pytest.fail("assessment must not run")
    )
    timed_result = await _controller(
        raw_planner=_RawPlanner([_payload("DIRECT", "deadline")]),
        assessor=timed_executor_assessor,
        store=LocalIterationStateStore(tmp_path / "deadline"),
        search=_SlowSearch(),
    ).run(
        JOB_ID,
        BRIEF,
        limits=_limits(max_waves=1, max_searches=1, max_elapsed_ms=5),
    )
    assert timed_result.state.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert timed_result.state.active_plan is not None
    assert timed_result.state.accounting.planner_calls == 1
    assert timed_result.state.accounting.search_calls == 1
    assert timed_result.state.active_inflight_query_id is not None

    class _SlowAssessor:
        async def assess(self, **_: Any) -> GapAssessment:
            await asyncio.sleep(0.1)
            return GapAssessment(
                decision=ContinuationDecision.STOP_NO_MATERIAL_GAIN,
                material_gain=False,
            )

    slow_assessor_result = await ResearchController(
        planner=SemanticPlanner(
            _RawPlanner([_payload("DIRECT", "slow assessor")]),
            capability_snapshot=CAPABILITY,
        ),
        executor=SearchWaveExecutor(_FakeSearch()),
        assessor=_SlowAssessor(),
        state_store=LocalIterationStateStore(tmp_path / "slow-assessor"),
        planning_limits=PLANNING_LIMITS,
        capability_snapshot=CAPABILITY,
        clock_ms=lambda: 0,
        created_at_factory=lambda: "2026-08-07T00:00:00Z",
    ).run(
        JOB_ID,
        BRIEF,
        limits=_limits(max_waves=1, max_searches=1, max_elapsed_ms=20),
    )
    assert slow_assessor_result.state.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert slow_assessor_result.state.accounting.assessment_calls == 1
    assert slow_assessor_result.state.assessment_inflight is True

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


@pytest.mark.asyncio
async def test_signed_source_is_sanitized_before_durable_checkpoint(tmp_path: Path) -> None:
    class _SignedSearch(_FakeSearch):
        async def __call__(self, **kwargs: Any) -> SearchResult:
            del kwargs
            return SearchResult(
                results=[
                    {"url": "https://sources.test/report?X-Amz-Signature=secret-value"}
                ],
                backend_used="fake",
                query="signed",
                content_mode="snippet",
                original_content_mode="snippet",
                format_fallback=False,
                size_guard_chars=200_000,
                truncated=False,
            )

    store = LocalIterationStateStore(tmp_path)
    assessor = _ScriptedAssessor(
        lambda _call, _state, _wave: GapAssessment(
            decision=ContinuationDecision.STOP_NO_MATERIAL_GAIN,
            material_gain=False,
        )
    )
    result = await _controller(
        raw_planner=_RawPlanner([_payload("DIRECT", "signed")]),
        assessor=assessor,
        store=store,
        search=_SignedSearch(),
    ).run(JOB_ID, BRIEF, limits=_limits(max_waves=1, max_searches=1))

    checkpoint = store.load(JOB_ID)
    assert checkpoint is not None
    assert result.state.stop_reason is StopReason.NO_MATERIAL_GAIN
    assert checkpoint.waves[0].unique_source_refs == ()
    assert "secret-value" not in (
        tmp_path / "research_iterations" / f"{JOB_ID}.iteration.json"
    ).read_text(encoding="utf-8")


def test_iteration_state_store_is_atomic_and_fail_closed(tmp_path: Path) -> None:
    store = LocalIterationStateStore(tmp_path)
    path = tmp_path / "research_iterations" / f"{JOB_ID}.iteration.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(IterationStateCorruptError):
        store.load(JOB_ID)

    valid_state = ResearchIterationState.new(
        job_id=JOB_ID,
        research_brief_sha256=compute_research_brief_sha256(BRIEF),
        limits=_limits(),
        planning_limits=PLANNING_LIMITS,
        capability_snapshot=CAPABILITY,
        started_at_ms=0,
    )
    store.write(JOB_ID, valid_state)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["next_wave_index"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IterationStateCorruptError):
        store.load(JOB_ID)

    store.write(JOB_ID, valid_state)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phase"] = "stopped"
    payload["stop_reason"] = "cancelled"
    payload["assessment_inflight"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IterationStateCorruptError):
        store.load(JOB_ID)

    store.write(JOB_ID, valid_state)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phase"] = "stopped"
    payload["stop_reason"] = "objective_covered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IterationStateCorruptError):
        store.load(JOB_ID)


def test_iteration_state_store_claim_prevents_concurrent_coordinators(
    tmp_path: Path,
) -> None:
    first = LocalIterationStateStore(tmp_path)
    second = LocalIterationStateStore(tmp_path)
    first.claim(JOB_ID)
    with pytest.raises(IterationStateBusyError):
        second.claim(JOB_ID)
    first.release(JOB_ID)
    second.claim(JOB_ID)
    second.release(JOB_ID)
