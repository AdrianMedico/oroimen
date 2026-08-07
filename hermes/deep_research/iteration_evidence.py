"""Evidence and provenance helpers for iterative Deep Research."""

from __future__ import annotations

from hermes.deep_research.iteration_state import (
    IterationInvariantError,
    IterationPhase,
    ResearchAccounting,
    ResearchIterationState,
    WaveRecord,
)
from hermes.deep_research.planning import (
    SearchObservation,
    normalize_query_text,
)
from hermes.deep_research.wave_execution import WaveExecutionResult


def source_evidence(
    observations: tuple[SearchObservation, ...],
) -> tuple[tuple[str, ...], dict[str, str]]:
    refs: list[str] = []
    provenance: dict[str, str] = {}
    for observation in observations:
        for source_ref in observation.result_refs:
            if source_ref in provenance:
                continue
            refs.append(source_ref)
            provenance[source_ref] = observation.query_id
    return tuple(refs), provenance


def count_search_calls(execution: WaveExecutionResult) -> int:
    calls = 0
    for observation in execution.observations:
        raw = observation.local_usage.get("search_calls")
        if not isinstance(raw, int) or isinstance(raw, bool) or raw != 1:
            raise IterationInvariantError(
                "each observation must report exactly one search call"
            )
        calls += raw
    if calls != len(execution.plan.queries):
        raise IterationInvariantError("search accounting does not match plan queries")
    return calls


def record_wave(
    state: ResearchIterationState,
    wave: WaveRecord,
) -> ResearchIterationState:
    query_ids = tuple(query.query_id for query in wave.plan.queries)
    query_texts = tuple(normalize_query_text(query.text) for query in wave.plan.queries)
    source_refs = tuple(dict.fromkeys((*state.source_refs, *wave.unique_source_refs)))
    source_query_ids = dict(state.source_query_ids)
    for source_ref, query_id in wave.source_query_ids.items():
        source_query_ids.setdefault(source_ref, query_id)
    return ResearchIterationState(
        job_id=state.job_id,
        research_brief_sha256=state.research_brief_sha256,
        limits=state.limits,
        planning_limits=state.planning_limits,
        capability_snapshot=state.capability_snapshot,
        phase=IterationPhase.ASSESSMENT_PENDING,
        next_wave_index=state.next_wave_index + 1,
        started_at_ms=state.started_at_ms,
        active_plan=None,
        active_observations=(),
        active_source_refs=(),
        active_source_query_ids={},
        waves=(*state.waves, wave),
        source_refs=source_refs,
        source_query_ids=source_query_ids,
        searched_query_ids=(*state.searched_query_ids, *query_ids),
        searched_query_texts=(*state.searched_query_texts, *query_texts),
        exhausted_query_ids=state.exhausted_query_ids,
        exhausted_query_texts=state.exhausted_query_texts,
        exhausted_source_refs=state.exhausted_source_refs,
        open_gaps=state.open_gaps,
        accounting=ResearchAccounting(
            planner_calls=state.accounting.planner_calls,
            search_calls=state.accounting.search_calls,
            assessment_calls=state.accounting.assessment_calls,
        ),
        stop_reason=None,
    )


def has_new_evidence(state: ResearchIterationState, wave: WaveRecord) -> bool:
    prior_digests = {
        digest
        for previous_wave in state.waves[:-1]
        for digest in previous_wave.evidence_digests
    }
    return any(
        item.digest not in prior_digests and (item.title or item.snippet)
        for observation in wave.observations
        for item in observation.evidence_items
    )


__all__ = [
    "count_search_calls",
    "has_new_evidence",
    "record_wave",
    "source_evidence",
]
