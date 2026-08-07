"""Tests for one-wave plan reuse and execution orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.deep_research.plan_store import (
    LocalPlanStore,
    PlanBriefHashMismatchError,
)
from hermes.deep_research.planner import (
    STRUCTURED_LLM_PLANNER_KIND,
    PlannerRequest,
    SemanticPlanner,
)
from hermes.deep_research.planning import (
    CapabilitySnapshot,
    PlannedSearchQuery,
    PlanningLimits,
    SearchPlan,
    build_search_plan,
    compute_query_id,
    compute_research_brief_sha256,
)
from hermes.deep_research.wave_execution import SearchWaveExecutor
from hermes.deep_research.wave_runner import SearchWaveRunner
from hermes.services.search.protocol import SearchResult

JOB_ID = "0123456789ab"
BRIEF = "Find the durable answer."
BRIEF_HASH = compute_research_brief_sha256(BRIEF)
SNAPSHOT = CapabilitySnapshot(
    planner_kind="c1a-deterministic-stub",
    planner_version="1.0.0",
    max_queries_per_wave=4,
    max_query_chars=399,
)


def _plan() -> SearchPlan:
    query_text = "durable answer"
    query = PlannedSearchQuery(
        query_id=compute_query_id(
            schema_version=1,
            wave_index=0,
            ordinal=0,
            normalized_text=query_text,
            research_brief_sha256=BRIEF_HASH,
        ),
        text=query_text,
        purpose="answer",
        dimension_ids=("answer",),
        ordinal=0,
    )
    return build_search_plan(
        planner_kind=SNAPSHOT.planner_kind,
        planner_version=SNAPSHOT.planner_version,
        research_brief_sha256=BRIEF_HASH,
        wave_index=0,
        queries=(query,),
        planning_limits=PlanningLimits(
            max_queries_per_wave=4,
            max_query_chars=399,
        ),
        capability_snapshot=SNAPSHOT,
        created_at="2026-08-07T00:00:00Z",
    )


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(
        self,
        *,
        query: str,
        intent: str,
        content: str,
        num_results: int,
    ) -> SearchResult:
        self.calls.append(query)
        assert intent == "deep_research"
        return SearchResult(
            results=[{"url": "https://example.com/source"}],
            backend_used="fake",
            query=query,
            content_mode=content,
            original_content_mode=content,
            format_fallback=False,
            size_guard_chars=100,
            truncated=False,
        )


@pytest.mark.asyncio
async def test_missing_plan_is_built_persisted_before_search(tmp_path: Path) -> None:
    search = RecordingSearch()
    store = LocalPlanStore(tmp_path)
    runner = SearchWaveRunner(
        plan_store=store,
        executor=SearchWaveExecutor(search, max_unique_sources=2),
    )
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        assert not store.exists(JOB_ID)
        return _plan()

    result = await runner.run(
        JOB_ID,
        expected_research_brief_sha256=BRIEF_HASH,
        expected_capability_snapshot=SNAPSHOT,
        plan_factory=factory,
    )

    assert result.plan_reused is False
    assert factory_calls == 1
    assert store.exists(JOB_ID)
    assert search.calls == ["durable answer"]


@pytest.mark.asyncio
async def test_existing_plan_is_reused_without_factory_call(tmp_path: Path) -> None:
    search = RecordingSearch()
    store = LocalPlanStore(tmp_path)
    store.write(JOB_ID, _plan())
    runner = SearchWaveRunner(
        plan_store=store,
        executor=SearchWaveExecutor(search, max_unique_sources=2),
    )

    def must_not_plan():
        raise AssertionError("persisted plan must be reused")

    result = await runner.run(
        JOB_ID,
        expected_research_brief_sha256=BRIEF_HASH,
        expected_capability_snapshot=SNAPSHOT,
        plan_factory=must_not_plan,
    )

    assert result.plan_reused is True
    assert search.calls == ["durable answer"]


@pytest.mark.asyncio
async def test_incompatible_persisted_plan_fails_before_search(tmp_path: Path) -> None:
    search = RecordingSearch()
    store = LocalPlanStore(tmp_path)
    store.write(JOB_ID, _plan())
    runner = SearchWaveRunner(
        plan_store=store,
        executor=SearchWaveExecutor(search, max_unique_sources=2),
    )

    with pytest.raises(PlanBriefHashMismatchError, match="plan_brief_hash_mismatch"):
        await runner.run(
            JOB_ID,
            expected_research_brief_sha256=compute_research_brief_sha256(
                "different brief"
            ),
            expected_capability_snapshot=SNAPSHOT,
            plan_factory=must_not_plan,
        )

    assert search.calls == []


@pytest.mark.asyncio
async def test_semantic_planner_runs_through_persisted_wave_boundary(tmp_path: Path) -> None:
    brief = "Compare regulation A across jurisdiction and timeline."
    llm_snapshot = CapabilitySnapshot(
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
    class FakeStructuredPlanner:
        calls = 0

        async def __call__(self, *, request: PlannerRequest):
            self.calls += 1
            assert request.capability_snapshot == llm_snapshot
            return {
                "decision": "DIRECT",
                "queries": [
                    {
                        "text": "primary regulation A source",
                        "purpose": "authority",
                        "dimension_ids": ["authority"],
                    }
                ]
            }

    structured = FakeStructuredPlanner()
    semantic = SemanticPlanner(
        structured,
        capability_snapshot=llm_snapshot,
    )
    request = PlannerRequest(
        research_brief=brief,
        planning_limits=PlanningLimits(max_queries_per_wave=4, max_query_chars=399),
        capability_snapshot=llm_snapshot,
        created_at="2026-08-07T00:00:00Z",
    )
    search = RecordingSearch()
    runner = SearchWaveRunner(
        plan_store=LocalPlanStore(tmp_path),
        executor=SearchWaveExecutor(search, max_unique_sources=2),
    )
    usage: list[dict[str, int]] = []

    async def factory():
        planned = await semantic.plan(request)
        usage.append(dict(planned.local_usage))
        return planned.plan

    first = await runner.run(
        JOB_ID,
        expected_research_brief_sha256=compute_research_brief_sha256(brief),
        expected_capability_snapshot=llm_snapshot,
        plan_factory=factory,
    )

    async def must_not_replan():
        raise AssertionError("persisted plan must be reused")

    second = await runner.run(
        JOB_ID,
        expected_research_brief_sha256=compute_research_brief_sha256(brief),
        expected_capability_snapshot=llm_snapshot,
        plan_factory=must_not_replan,
    )

    assert first.plan_reused is False
    assert second.plan_reused is True
    assert usage == [{"planner_calls": 1}]
    assert structured.calls == 1
    assert len(search.calls) == 2


def must_not_plan():
    raise AssertionError("the planner must not be called")
