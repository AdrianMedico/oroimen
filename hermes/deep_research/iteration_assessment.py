"""Continuation-policy validation and state application for iterative research."""

from __future__ import annotations

from dataclasses import replace

from hermes.deep_research.iteration_evidence import has_new_evidence
from hermes.deep_research.iteration_state import (
    ContinuationDecision,
    GapAssessment,
    IterationInvariantError,
    IterationPhase,
    ResearchAccounting,
    ResearchIterationState,
    StopReason,
    WaveRecord,
)
from hermes.deep_research.planning import normalize_query_text


def validate_assessment(
    assessment: GapAssessment,
    state: ResearchIterationState,
    wave: WaveRecord,
) -> None:
    if not isinstance(assessment, GapAssessment):
        raise IterationInvariantError("gap assessor must return GapAssessment")
    if not set(assessment.exhausted_query_ids).issubset(state.searched_query_ids):
        raise IterationInvariantError("assessor exhausted an unknown query")
    if not set(assessment.exhausted_source_refs).issubset(state.source_refs):
        raise IterationInvariantError("assessor exhausted an unknown source")
    if assessment.decision is ContinuationDecision.STOP_COVERED and assessment.remaining_gaps:
        raise IterationInvariantError("STOP_COVERED requires no remaining gaps")
    if assessment.decision is ContinuationDecision.CONTINUE and not assessment.remaining_gaps:
        raise IterationInvariantError("CONTINUE requires remaining gaps")
    evidence_is_new = has_new_evidence(state, wave)
    if (
        assessment.decision is ContinuationDecision.STOP_COVERED
        and (not assessment.material_gain or not evidence_is_new)
    ):
        raise IterationInvariantError(
            "STOP_COVERED requires material gain and new bounded evidence"
        )
    if (
        assessment.decision is ContinuationDecision.STOP_NO_MATERIAL_GAIN
        and assessment.material_gain
    ):
        raise IterationInvariantError("STOP_NO_MATERIAL_GAIN cannot report material gain")


def apply_assessment(
    state: ResearchIterationState,
    assessment: GapAssessment,
    accounting: ResearchAccounting,
) -> ResearchIterationState:
    query_text_by_id = {
        query.query_id: normalize_query_text(query.text)
        for wave in state.waves
        for query in wave.plan.queries
    }
    exhausted_query_texts = tuple(
        dict.fromkeys(
            (
                *state.exhausted_query_texts,
                *(
                    query_text_by_id[query_id]
                    for query_id in assessment.exhausted_query_ids
                ),
            )
        )
    )
    updated = ResearchIterationState(
        job_id=state.job_id,
        research_brief_sha256=state.research_brief_sha256,
        limits=state.limits,
        planning_limits=state.planning_limits,
        capability_snapshot=state.capability_snapshot,
        phase=IterationPhase.READY_TO_PLAN,
        next_wave_index=state.next_wave_index,
        started_at_ms=state.started_at_ms,
        active_plan=None,
        waves=state.waves,
        source_refs=state.source_refs,
        source_query_ids=state.source_query_ids,
        searched_query_ids=state.searched_query_ids,
        searched_query_texts=state.searched_query_texts,
        exhausted_query_ids=tuple(
            dict.fromkeys((*state.exhausted_query_ids, *assessment.exhausted_query_ids))
        ),
        exhausted_query_texts=exhausted_query_texts,
        exhausted_source_refs=tuple(
            dict.fromkeys((*state.exhausted_source_refs, *assessment.exhausted_source_refs))
        ),
        open_gaps=assessment.remaining_gaps,
        accounting=accounting,
        stop_reason=None,
    )
    if assessment.decision is ContinuationDecision.STOP_COVERED:
        return replace(
            updated,
            phase=IterationPhase.STOPPED,
            stop_reason=StopReason.OBJECTIVE_COVERED,
        )
    if assessment.decision is ContinuationDecision.STOP_NO_MATERIAL_GAIN:
        return replace(
            updated,
            phase=IterationPhase.STOPPED,
            stop_reason=StopReason.NO_MATERIAL_GAIN,
        )
    if not assessment.material_gain or not has_new_evidence(state, state.waves[-1]):
        return replace(
            updated,
            phase=IterationPhase.STOPPED,
            stop_reason=StopReason.NO_MATERIAL_GAIN,
        )
    return updated


__all__ = ["apply_assessment", "validate_assessment"]
