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
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlsplit

from hermes.deep_research.planner import (
    STRUCTURED_LLM_PLANNER_KIND,
    PlannerRequest,
    PlannerResult,
)
from hermes.deep_research.planning import (
    MAX_QUERIES_PER_WAVE,
    MAX_QUERY_CHARS,
    MAX_WAVES_PER_JOB,
    CapabilitySnapshot,
    PlanningLimits,
    SearchObservation,
    SearchPlan,
    compute_research_brief_sha256,
    deserialize_search_plan,
    normalize_query_text,
    serialize_search_plan,
    validate_search_plan,
)
from hermes.deep_research.wave_execution import (
    SearchWaveExecutor,
    WaveExecutionCancelled,
    WaveExecutionOutcome,
    WaveExecutionResult,
)

ITERATION_SCHEMA_VERSION = 1
MAX_SEARCHES_PER_JOB = MAX_WAVES_PER_JOB * MAX_QUERIES_PER_WAVE
MAX_LOCAL_CALL_UNITS = MAX_SEARCHES_PER_JOB + (2 * MAX_WAVES_PER_JOB)
MAX_GAPS = 32
MAX_CONTEXT_ITEM_CHARS = MAX_QUERY_CHARS
MAX_SOURCES_PER_JOB = MAX_SEARCHES_PER_JOB * 5
MAX_SOURCE_REF_CHARS = 2_048
MAX_RESULT_REFS_PER_OBSERVATION = 32
_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ITERATION_STATE_KEYS = frozenset(
    {
        "schema_version",
        "job_id",
        "research_brief_sha256",
        "limits",
        "planning_limits",
        "capability_snapshot",
        "phase",
        "next_wave_index",
        "started_at_ms",
        "active_plan",
        "planning_inflight",
        "assessment_inflight",
        "active_inflight_query_id",
        "active_observations",
        "active_source_refs",
        "active_source_query_ids",
        "waves",
        "source_refs",
        "source_query_ids",
        "searched_query_ids",
        "searched_query_texts",
        "exhausted_query_ids",
        "exhausted_query_texts",
        "exhausted_source_refs",
        "open_gaps",
        "accounting",
        "stop_reason",
    }
)
_SENSITIVE_URL_KEY_PARTS = frozenset(
    {"token", "key", "sig", "signature", "secret", "auth", "credential"}
)


class IterationPhase(StrEnum):
    READY_TO_PLAN = "ready_to_plan"
    PLAN_PERSISTED = "plan_persisted"
    ASSESSMENT_PENDING = "assessment_pending"
    STOPPED = "stopped"


class StopReason(StrEnum):
    OBJECTIVE_COVERED = "objective_covered"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_MATERIAL_GAIN = "no_material_gain"
    CANCELLED = "cancelled"


class ContinuationDecision(StrEnum):
    CONTINUE = "continue"
    STOP_COVERED = "stop_covered"
    STOP_NO_MATERIAL_GAIN = "stop_no_material_gain"


class IterationInvariantError(ValueError):
    """Raised when an injected C2 proposal violates deterministic rules."""


@dataclass(frozen=True)
class IterationLimits:
    """Job-level safety limits; query limits remain per SearchWave."""

    max_waves: int
    max_searches: int
    max_elapsed_ms: int
    max_local_call_units: int

    def __post_init__(self) -> None:
        if not isinstance(self.max_waves, int) or isinstance(self.max_waves, bool):
            raise ValueError("max_waves must be an int")
        if not 1 <= self.max_waves <= MAX_WAVES_PER_JOB:
            raise ValueError(
                f"max_waves must be in [1, {MAX_WAVES_PER_JOB}]"
            )
        if not isinstance(self.max_searches, int) or isinstance(
            self.max_searches, bool
        ):
            raise ValueError("max_searches must be an int")
        if not 1 <= self.max_searches <= MAX_SEARCHES_PER_JOB:
            raise ValueError(
                f"max_searches must be in [1, {MAX_SEARCHES_PER_JOB}]"
            )
        if self.max_searches < self.max_waves:
            raise ValueError("max_searches must allow at least one query per wave")
        if not isinstance(self.max_elapsed_ms, int) or isinstance(
            self.max_elapsed_ms, bool
        ):
            raise ValueError("max_elapsed_ms must be an int")
        if self.max_elapsed_ms <= 0:
            raise ValueError("max_elapsed_ms must be positive")
        if not isinstance(self.max_local_call_units, int) or isinstance(
            self.max_local_call_units, bool
        ):
            raise ValueError("max_local_call_units must be an int")
        if not 1 <= self.max_local_call_units <= MAX_LOCAL_CALL_UNITS:
            raise ValueError(
                f"max_local_call_units must be in [1, {MAX_LOCAL_CALL_UNITS}]"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_waves": self.max_waves,
            "max_searches": self.max_searches,
            "max_elapsed_ms": self.max_elapsed_ms,
            "max_local_call_units": self.max_local_call_units,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IterationLimits:
        _require_keys(
            payload,
            {"max_waves", "max_searches", "max_elapsed_ms", "max_local_call_units"},
            "iteration limits",
        )
        return cls(
            max_waves=_strict_int(payload, "max_waves"),
            max_searches=_strict_int(payload, "max_searches"),
            max_elapsed_ms=_strict_int(payload, "max_elapsed_ms"),
            max_local_call_units=_strict_int(payload, "max_local_call_units"),
        )


@dataclass(frozen=True)
class ResearchAccounting:
    """Provider-independent call accounting for one iterative job."""

    planner_calls: int = 0
    search_calls: int = 0
    assessment_calls: int = 0

    def __post_init__(self) -> None:
        for name in ("planner_calls", "search_calls", "assessment_calls"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")

    @property
    def total_local_call_units(self) -> int:
        return self.planner_calls + self.search_calls + self.assessment_calls

    def to_dict(self) -> dict[str, int]:
        return {
            "planner_calls": self.planner_calls,
            "search_calls": self.search_calls,
            "assessment_calls": self.assessment_calls,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResearchAccounting:
        _require_keys(
            payload,
            {"planner_calls", "search_calls", "assessment_calls"},
            "research accounting",
        )
        return cls(
            planner_calls=_strict_int(payload, "planner_calls"),
            search_calls=_strict_int(payload, "search_calls"),
            assessment_calls=_strict_int(payload, "assessment_calls"),
        )


def _bounded_items(
    name: str,
    values: tuple[str, ...],
    *,
    max_items: int = MAX_GAPS,
    max_chars: int = MAX_CONTEXT_ITEM_CHARS,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    if len(values) > max_items:
        raise ValueError(f"{name} exceeds the {max_items}-item cap")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    for value in values:
        if not isinstance(value, str) or not value or len(value) > max_chars:
            raise ValueError(f"{name} contains an invalid bounded item")
        if "\n" in value or "\r" in value:
            raise ValueError(f"{name} items must be single-line")
    return values


def _require_keys(
    payload: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    name: str,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != set(expected):
        raise ValueError(f"{name} has an invalid field set")


def _strict_int(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    return value


def _strict_str(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _strict_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _strict_capability_snapshot(payload: Mapping[str, Any]) -> CapabilitySnapshot:
    allowed = {
        "planner_kind",
        "planner_version",
        "max_queries_per_wave",
        "max_query_chars",
        "planner_provenance",
    }
    if not set(payload).issubset(allowed) or not {
        "planner_kind",
        "planner_version",
        "max_queries_per_wave",
        "max_query_chars",
    }.issubset(payload):
        raise ValueError("capability snapshot has an invalid field set")
    provenance_raw = payload.get("planner_provenance", {})
    if not isinstance(provenance_raw, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in provenance_raw.items()
    ):
        raise ValueError("planner provenance must be a string mapping")
    return CapabilitySnapshot(
        planner_kind=_strict_str(payload, "planner_kind"),
        planner_version=_strict_str(payload, "planner_version"),
        max_queries_per_wave=_strict_int(payload, "max_queries_per_wave"),
        max_query_chars=_strict_int(payload, "max_query_chars"),
        planner_provenance=dict(provenance_raw),
    )


@dataclass(frozen=True)
class GapAssessment:
    """Proposal from a deterministic/fake or future model-backed assessor."""

    decision: ContinuationDecision
    remaining_gaps: tuple[str, ...] = ()
    material_gain: bool = False
    exhausted_query_ids: tuple[str, ...] = ()
    exhausted_source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            decision = ContinuationDecision(self.decision)
        except ValueError as exc:
            raise ValueError("invalid continuation decision") from exc
        object.__setattr__(self, "decision", decision)
        if not isinstance(self.material_gain, bool):
            raise ValueError("material_gain must be a bool")
        _bounded_items("remaining_gaps", self.remaining_gaps)
        _bounded_items("exhausted_query_ids", self.exhausted_query_ids)
        _bounded_items(
            "exhausted_source_refs",
            self.exhausted_source_refs,
            max_items=MAX_SOURCES_PER_JOB,
            max_chars=MAX_SOURCE_REF_CHARS,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "remaining_gaps": list(self.remaining_gaps),
            "material_gain": self.material_gain,
            "exhausted_query_ids": list(self.exhausted_query_ids),
            "exhausted_source_refs": list(self.exhausted_source_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GapAssessment:
        material_gain = payload.get("material_gain", False)
        if not isinstance(material_gain, bool):
            raise ValueError("material_gain must be a bool")
        return cls(
            decision=ContinuationDecision(str(payload["decision"])),
            remaining_gaps=_strict_string_tuple(payload, "remaining_gaps"),
            material_gain=material_gain,
            exhausted_query_ids=_strict_string_tuple(payload, "exhausted_query_ids"),
            exhausted_source_refs=_strict_string_tuple(payload, "exhausted_source_refs"),
        )


def _strict_string_tuple(payload: Mapping[str, Any], name: str) -> tuple[str, ...]:
    raw = payload.get(name, [])
    if not isinstance(raw, list):
        raise ValueError(f"{name} must be a list")
    if not all(isinstance(value, str) for value in raw):
        raise ValueError(f"{name} must contain strings")
    return tuple(raw)


def _safe_source_ref(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    try:
        query_keys = {
            key.lower()
            for key, _value in parse_qsl(
                parsed.query,
                keep_blank_values=True,
                max_num_fields=64,
            )
        }
    except ValueError:
        return False
    if any(
        any(part in key for part in _SENSITIVE_URL_KEY_PARTS)
        for key in query_keys
    ):
        return False
    return (
        parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


@dataclass(frozen=True)
class WaveRecord:
    """Durable, provider-neutral evidence from one completed SearchWave."""

    wave_index: int
    plan: SearchPlan
    observations: tuple[SearchObservation, ...]
    unique_source_refs: tuple[str, ...]
    source_query_ids: Mapping[str, str]
    outcome: WaveExecutionOutcome
    unique_source_cap: int

    def __post_init__(self) -> None:
        if self.wave_index != self.plan.wave_index:
            raise ValueError("wave record and plan wave index must match")
        if len(self.observations) != len(self.plan.queries):
            raise ValueError("wave record must contain one observation per query")
        if len(set(self.unique_source_refs)) != len(self.unique_source_refs):
            raise ValueError("wave source refs must be unique")
        if (
            not isinstance(self.unique_source_cap, int)
            or isinstance(self.unique_source_cap, bool)
            or self.unique_source_cap <= 0
        ):
            raise ValueError("unique_source_cap must be a positive int")
        if len(self.unique_source_refs) > self.unique_source_cap:
            raise ValueError("wave source refs exceed the executor cap")
        if not all(
            isinstance(source_ref, str)
            and len(source_ref) <= MAX_SOURCE_REF_CHARS
            and _safe_source_ref(source_ref)
            for source_ref in self.unique_source_refs
        ):
            raise ValueError("wave source refs must be safe HTTP(S) URLs")
        query_ids = {query.query_id for query in self.plan.queries}
        observation_ids = [observation.query_id for observation in self.observations]
        if any(observation.wave_index != self.wave_index for observation in self.observations):
            raise ValueError("wave observations must bind to the wave")
        if len(set(observation_ids)) != len(observation_ids) or set(observation_ids) != query_ids:
            raise ValueError("wave observations must cover each planned query once")
        for observation in self.observations:
            if len(observation.result_refs) > MAX_RESULT_REFS_PER_OBSERVATION:
                raise ValueError("observation result refs exceed the per-query cap")
            if len(observation.evidence_digests) != len(observation.result_refs):
                raise ValueError("observation evidence must align with source refs")
            if dict(observation.local_usage) != {"search_calls": 1}:
                raise ValueError("observation accounting must report one search call")
            if not all(
                isinstance(source_ref, str)
                and len(source_ref) <= MAX_SOURCE_REF_CHARS
                and _safe_source_ref(source_ref)
                for source_ref in observation.result_refs
            ):
                raise ValueError("observation result refs must be safe HTTP(S) URLs")
        if not set(self.source_query_ids).issubset(self.unique_source_refs):
            raise ValueError("source provenance must point to wave evidence")
        expected_refs: list[str] = []
        expected_provenance: dict[str, str] = {}
        for observation in self.observations:
            for source_ref in observation.result_refs:
                if source_ref in expected_provenance:
                    continue
                if len(expected_refs) >= self.unique_source_cap:
                    break
                expected_refs.append(source_ref)
                expected_provenance[source_ref] = observation.query_id
        if tuple(expected_refs) != self.unique_source_refs:
            raise ValueError("wave sources must match first-seen observation evidence")
        if dict(self.source_query_ids) != expected_provenance:
            raise ValueError("wave source provenance must match first-seen evidence")
        if set(self.source_query_ids.values()) - query_ids:
            raise ValueError("source provenance must point to wave queries")
        failures = sum(
            observation.structured_error is not None for observation in self.observations
        )
        if failures == len(self.observations):
            expected_outcome = WaveExecutionOutcome.ALL_FAILED
        elif not self.unique_source_refs:
            expected_outcome = (
                WaveExecutionOutcome.PARTIAL_NO_EVIDENCE
                if failures
                else WaveExecutionOutcome.ALL_EMPTY
            )
        elif failures:
            expected_outcome = WaveExecutionOutcome.PARTIAL_SUCCESS
        else:
            expected_outcome = WaveExecutionOutcome.ALL_SUCCESS
        if self.outcome != expected_outcome:
            raise ValueError("wave outcome contradicts observations and evidence")
        if not isinstance(self.outcome, WaveExecutionOutcome):
            object.__setattr__(self, "outcome", WaveExecutionOutcome(self.outcome))
        object.__setattr__(
            self,
            "source_query_ids",
            MappingProxyType(dict(self.source_query_ids)),
        )

    @property
    def evidence_digests(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                digest
                for observation in self.observations
                for digest in observation.evidence_digests
            )
        )

    @classmethod
    def from_result(cls, result: WaveExecutionResult) -> WaveRecord:
        return cls(
            wave_index=result.plan.wave_index,
            plan=result.plan,
            observations=result.observations,
            unique_source_refs=result.unique_source_refs,
            source_query_ids=result.source_query_ids,
            outcome=result.outcome,
            unique_source_cap=result.unique_source_cap,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_index": self.wave_index,
            "plan": json.loads(serialize_search_plan(self.plan).decode("utf-8")),
            "observations": [observation.to_dict() for observation in self.observations],
            "unique_source_refs": list(self.unique_source_refs),
            "source_query_ids": dict(self.source_query_ids),
            "outcome": self.outcome.value,
            "unique_source_cap": self.unique_source_cap,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WaveRecord:
        _require_keys(
            payload,
            {
                "wave_index",
                "plan",
                "observations",
                "unique_source_refs",
                "source_query_ids",
                "outcome",
                "unique_source_cap",
            },
            "wave record",
        )
        observations_raw = payload["observations"]
        source_refs_raw = payload["unique_source_refs"]
        if not isinstance(observations_raw, list) or not all(
            isinstance(item, Mapping) for item in observations_raw
        ):
            raise ValueError("wave observations must be a list of objects")
        if not isinstance(source_refs_raw, list) or not all(
            isinstance(value, str) for value in source_refs_raw
        ):
            raise ValueError("wave source refs must be a list of strings")
        source_query_ids_raw = _strict_mapping(payload, "source_query_ids")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in source_query_ids_raw.items()
        ):
            raise ValueError("wave source provenance must contain strings")
        outcome_raw = _strict_str(payload, "outcome")
        return cls(
            wave_index=_strict_int(payload, "wave_index"),
            plan=deserialize_search_plan(
                json.dumps(payload["plan"], sort_keys=True).encode("utf-8")
            ),
            observations=tuple(
                SearchObservation.from_dict(item)
                for item in observations_raw
            ),
            unique_source_refs=tuple(source_refs_raw),
            source_query_ids=dict(source_query_ids_raw),
            outcome=WaveExecutionOutcome(outcome_raw),
            unique_source_cap=_strict_int(payload, "unique_source_cap"),
        )


@dataclass(frozen=True)
class ResearchIterationState:
    """Checkpointed state for one bounded Research Brief iteration."""

    job_id: str
    research_brief_sha256: str
    limits: IterationLimits
    planning_limits: PlanningLimits
    capability_snapshot: CapabilitySnapshot
    phase: IterationPhase
    next_wave_index: int
    started_at_ms: int
    active_plan: SearchPlan | None = None
    planning_inflight: bool = False
    assessment_inflight: bool = False
    active_inflight_query_id: str | None = None
    active_observations: tuple[SearchObservation, ...] = ()
    active_source_refs: tuple[str, ...] = ()
    active_source_query_ids: Mapping[str, str] = field(default_factory=dict)
    waves: tuple[WaveRecord, ...] = ()
    source_refs: tuple[str, ...] = ()
    source_query_ids: Mapping[str, str] = field(default_factory=dict)
    searched_query_ids: tuple[str, ...] = ()
    searched_query_texts: tuple[str, ...] = ()
    exhausted_query_ids: tuple[str, ...] = ()
    exhausted_query_texts: tuple[str, ...] = ()
    exhausted_source_refs: tuple[str, ...] = ()
    open_gaps: tuple[str, ...] = ()
    accounting: ResearchAccounting = field(default_factory=ResearchAccounting)
    stop_reason: StopReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not _JOB_ID_RE.fullmatch(self.job_id):
            raise ValueError("job_id must be a lowercase hex UUID12 token")
        if not isinstance(self.research_brief_sha256, str) or not _SHA256_RE.fullmatch(
            self.research_brief_sha256
        ):
            raise ValueError("research_brief_sha256 must be a lowercase SHA-256")
        if not isinstance(self.next_wave_index, int) or not 0 <= self.next_wave_index <= MAX_WAVES_PER_JOB:
            raise ValueError("next_wave_index is outside the hard wave bound")
        if not isinstance(self.started_at_ms, int) or self.started_at_ms < 0:
            raise ValueError("started_at_ms must be a non-negative int")
        try:
            phase = IterationPhase(self.phase)
        except ValueError as exc:
            raise ValueError("invalid iteration phase") from exc
        object.__setattr__(self, "phase", phase)
        for name in ("planning_inflight", "assessment_inflight"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.active_inflight_query_id is not None and (
            not isinstance(self.active_inflight_query_id, str)
            or not self.active_inflight_query_id
        ):
            raise ValueError("active_inflight_query_id must be a string or null")
        if self.planning_inflight and self.assessment_inflight:
            raise ValueError("only one local call may be in flight")
        if self.planning_inflight and phase not in {
            IterationPhase.READY_TO_PLAN,
            IterationPhase.STOPPED,
        }:
            raise ValueError("planning_inflight requires ready_to_plan")
        if self.assessment_inflight and phase not in {
            IterationPhase.ASSESSMENT_PENDING,
            IterationPhase.STOPPED,
        }:
            raise ValueError("assessment_inflight requires assessment_pending")
        if self.active_inflight_query_id is not None and phase not in {
            IterationPhase.PLAN_PERSISTED,
            IterationPhase.STOPPED,
        }:
            raise ValueError("active_inflight_query_id requires plan_persisted")
        if self.active_plan is not None and self.active_plan.wave_index != self.next_wave_index:
            raise ValueError("active plan must target next_wave_index")
        plans = [wave.plan for wave in self.waves]
        if self.active_plan is not None:
            plans.append(self.active_plan)
        for plan in plans:
            validate_search_plan(
                plan,
                expected_research_brief_sha256=self.research_brief_sha256,
            )
            if plan.planning_limits != self.planning_limits:
                raise ValueError("wave plan planning limits drifted from iteration state")
            if plan.capability_snapshot.to_dict() != self.capability_snapshot.to_dict():
                raise ValueError("wave plan capability snapshot drifted from iteration state")
        if len(self.waves) > self.limits.max_waves:
            raise ValueError("wave records exceed max_waves")
        if self.next_wave_index != len(self.waves):
            raise ValueError("next_wave_index must equal the completed wave count")
        for expected_index, wave in enumerate(self.waves):
            if wave.wave_index != expected_index:
                raise ValueError("wave records must be contiguous from zero")
        if self.active_plan is None:
            if (
                self.active_observations
                or self.active_source_refs
                or self.active_source_query_ids
            ):
                raise ValueError("partial wave evidence requires an active plan")
            if self.active_inflight_query_id is not None:
                raise ValueError("in-flight query requires an active plan")
        else:
            ordered_queries = sorted(
                self.active_plan.queries,
                key=lambda query: query.ordinal,
            )
            if len(self.active_observations) > len(ordered_queries):
                raise ValueError("partial observations exceed the active plan")
            if any(
                dict(observation.local_usage) != {"search_calls": 1}
                for observation in self.active_observations
            ):
                raise ValueError("partial observation accounting is not exact")
            expected_active_ids = tuple(
                query.query_id
                for query in ordered_queries[: len(self.active_observations)]
            )
            actual_active_ids = tuple(
                observation.query_id for observation in self.active_observations
            )
            if actual_active_ids != expected_active_ids:
                raise ValueError("partial observations must be an ordinal plan prefix")
            expected_active_refs: list[str] = []
            expected_active_provenance: dict[str, str] = {}
            for observation in self.active_observations:
                if observation.wave_index != self.active_plan.wave_index:
                    raise ValueError("partial observations must bind to the active wave")
                for source_ref in observation.result_refs:
                    if source_ref in expected_active_provenance:
                        continue
                    expected_active_refs.append(source_ref)
                    expected_active_provenance[source_ref] = observation.query_id
            _bounded_items(
                "active_source_refs",
                self.active_source_refs,
                max_items=MAX_SOURCES_PER_JOB,
                max_chars=MAX_SOURCE_REF_CHARS,
            )
            if not all(_safe_source_ref(source_ref) for source_ref in self.active_source_refs):
                raise ValueError("active source refs must be safe HTTP(S) URLs")
            if tuple(expected_active_refs) != self.active_source_refs:
                raise ValueError("active sources must match partial observations")
            if dict(self.active_source_query_ids) != expected_active_provenance:
                raise ValueError("active source provenance must match partial observations")
            next_query_id = (
                ordered_queries[len(self.active_observations)].query_id
                if len(self.active_observations) < len(ordered_queries)
                else None
            )
            if self.active_inflight_query_id not in {None, next_query_id}:
                raise ValueError("in-flight query must be the next ordinal query")
        expected_query_ids = tuple(
            query.query_id for wave in self.waves for query in wave.plan.queries
        )
        expected_query_texts = tuple(
            normalize_query_text(query.text)
            for wave in self.waves
            for query in wave.plan.queries
        )
        if self.searched_query_ids != expected_query_ids:
            raise ValueError("searched query ids must match completed wave plans")
        if self.searched_query_texts != expected_query_texts:
            raise ValueError("searched query texts must match completed wave plans")
        expected_source_refs = tuple(
            dict.fromkeys(
                source_ref
                for wave in self.waves
                for source_ref in wave.unique_source_refs
            )
        )
        if self.source_refs != expected_source_refs:
            raise ValueError("source refs must match completed wave evidence")
        for name, values in (
            ("searched_query_ids", self.searched_query_ids),
            ("searched_query_texts", self.searched_query_texts),
            ("exhausted_query_ids", self.exhausted_query_ids),
            ("exhausted_query_texts", self.exhausted_query_texts),
            ("open_gaps", self.open_gaps),
        ):
            _bounded_items(name, values)
        _bounded_items(
            "source_refs",
            self.source_refs,
            max_items=MAX_SOURCES_PER_JOB,
            max_chars=MAX_SOURCE_REF_CHARS,
        )
        if not all(_safe_source_ref(source_ref) for source_ref in self.source_refs):
            raise ValueError("source_refs must be safe HTTP(S) URLs")
        if not set(self.exhausted_query_ids).issubset(self.searched_query_ids):
            raise ValueError("exhausted query ids must have been searched")
        if not set(self.exhausted_query_texts).issubset(self.searched_query_texts):
            raise ValueError("exhausted query texts must have been searched")
        if not set(self.exhausted_source_refs).issubset(self.source_refs):
            raise ValueError("exhausted sources must be present in evidence")
        expected_source_query_ids: dict[str, str] = {}
        for wave in self.waves:
            for source_ref, query_id in wave.source_query_ids.items():
                expected_source_query_ids.setdefault(source_ref, query_id)
        if dict(self.source_query_ids) != expected_source_query_ids:
            raise ValueError("source provenance must match every completed wave")
        if not all(
            isinstance(source_ref, str)
            and isinstance(query_id, str)
            and query_id
            for source_ref, query_id in self.source_query_ids.items()
        ):
            raise ValueError("source provenance must contain non-empty strings")
        expected_search_calls = (
            len(expected_query_ids)
            + len(self.active_observations)
            + (1 if self.active_inflight_query_id is not None else 0)
        )
        if self.accounting.search_calls != expected_search_calls:
            raise ValueError("search accounting must match dispatched query calls")
        all_query_ids = set(expected_query_ids)
        if not set(self.source_query_ids.values()).issubset(all_query_ids):
            raise ValueError("source provenance must match completed wave queries")
        if (
            self.accounting.planner_calls > self.limits.max_waves
            or self.accounting.search_calls > self.limits.max_searches
            or self.accounting.assessment_calls > self.limits.max_waves
            or self.accounting.total_local_call_units > self.limits.max_local_call_units
        ):
            raise ValueError("accounting exceeds iteration limits")
        if self.phase is IterationPhase.STOPPED and self.stop_reason is None:
            raise ValueError("stopped state requires a stop reason")
        if self.phase is not IterationPhase.STOPPED and self.stop_reason is not None:
            raise ValueError("non-stopped state cannot have a stop reason")
        if self.phase is IterationPhase.READY_TO_PLAN and self.active_plan is not None:
            raise ValueError("ready_to_plan state cannot contain an active plan")
        if self.phase is IterationPhase.ASSESSMENT_PENDING and self.active_plan is not None:
            raise ValueError("assessment_pending state cannot contain an active plan")
        if self.phase is IterationPhase.PLAN_PERSISTED and self.active_plan is None:
            raise ValueError("plan_persisted state requires an active plan")
        completed_waves = len(self.waves)
        if self.phase is IterationPhase.READY_TO_PLAN:
            expected_planner_calls = completed_waves + (1 if self.planning_inflight else 0)
            expected_assessment_calls = completed_waves
        elif self.phase is IterationPhase.PLAN_PERSISTED:
            expected_planner_calls = completed_waves + 1
            expected_assessment_calls = completed_waves
        elif self.phase is IterationPhase.ASSESSMENT_PENDING:
            expected_planner_calls = completed_waves
            expected_assessment_calls = completed_waves - 1 + (
                1 if self.assessment_inflight else 0
            )
        else:
            expected_planner_calls = completed_waves + (
                1 if self.planning_inflight or self.active_plan is not None else 0
            )
            expected_assessment_calls = completed_waves
            if not self.assessment_inflight and not self.active_plan:
                expected_assessment_calls -= 1 if self.accounting.assessment_calls == completed_waves - 1 else 0
        if self.accounting.planner_calls != expected_planner_calls:
            raise ValueError("planner accounting is incompatible with iteration phase")
        if self.accounting.assessment_calls != expected_assessment_calls:
            raise ValueError("assessment accounting is incompatible with iteration phase")
        if self.phase is IterationPhase.STOPPED:
            if self.stop_reason in {
                StopReason.OBJECTIVE_COVERED,
                StopReason.NO_MATERIAL_GAIN,
            } and (
                not self.waves
                or self.active_plan is not None
                or self.active_observations
                or self.active_source_refs
                or self.active_source_query_ids
                or self.planning_inflight
                or self.assessment_inflight
                or self.active_inflight_query_id is not None
                or self.accounting.planner_calls != completed_waves
                or self.accounting.assessment_calls != completed_waves
            ):
                raise ValueError("completed assessment STOP requires no in-flight work")
            in_flight_modes = sum(
                (
                    self.planning_inflight,
                    self.assessment_inflight,
                    self.active_inflight_query_id is not None,
                )
            )
            if in_flight_modes > 1:
                raise ValueError("stopped state cannot contain multiple in-flight calls")
            if self.planning_inflight and self.active_plan is not None:
                raise ValueError("planner in-flight state cannot contain an active plan")
            if self.assessment_inflight and (
                not self.waves
                or self.active_plan is not None
                or self.active_observations
                or self.active_source_refs
                or self.active_source_query_ids
            ):
                raise ValueError("assessment in-flight state cannot lack a completed wave")
            if self.active_inflight_query_id is not None and (
                self.active_plan is None
                or self.planning_inflight
                or self.assessment_inflight
            ):
                raise ValueError("query in-flight state must own the active plan")
        object.__setattr__(self, "source_query_ids", MappingProxyType(dict(self.source_query_ids)))
        object.__setattr__(
            self,
            "active_source_query_ids",
            MappingProxyType(dict(self.active_source_query_ids)),
        )

    @classmethod
    def new(
        cls,
        *,
        job_id: str,
        research_brief_sha256: str,
        limits: IterationLimits,
        planning_limits: PlanningLimits,
        capability_snapshot: CapabilitySnapshot,
        started_at_ms: int,
    ) -> ResearchIterationState:
        return cls(
            job_id=job_id,
            research_brief_sha256=research_brief_sha256,
            limits=limits,
            planning_limits=planning_limits,
            capability_snapshot=capability_snapshot,
            phase=IterationPhase.READY_TO_PLAN,
            next_wave_index=0,
            started_at_ms=started_at_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ITERATION_SCHEMA_VERSION,
            "job_id": self.job_id,
            "research_brief_sha256": self.research_brief_sha256,
            "limits": self.limits.to_dict(),
            "planning_limits": self.planning_limits.to_dict(),
            "capability_snapshot": self.capability_snapshot.to_dict(),
            "phase": self.phase.value,
            "next_wave_index": self.next_wave_index,
            "started_at_ms": self.started_at_ms,
            "active_plan": (
                None
                if self.active_plan is None
                else json.loads(serialize_search_plan(self.active_plan).decode("utf-8"))
            ),
            "planning_inflight": self.planning_inflight,
            "assessment_inflight": self.assessment_inflight,
            "active_inflight_query_id": self.active_inflight_query_id,
            "active_observations": [
                observation.to_dict() for observation in self.active_observations
            ],
            "active_source_refs": list(self.active_source_refs),
            "active_source_query_ids": dict(self.active_source_query_ids),
            "waves": [wave.to_dict() for wave in self.waves],
            "source_refs": list(self.source_refs),
            "source_query_ids": dict(self.source_query_ids),
            "searched_query_ids": list(self.searched_query_ids),
            "searched_query_texts": list(self.searched_query_texts),
            "exhausted_query_ids": list(self.exhausted_query_ids),
            "exhausted_query_texts": list(self.exhausted_query_texts),
            "exhausted_source_refs": list(self.exhausted_source_refs),
            "open_gaps": list(self.open_gaps),
            "accounting": self.accounting.to_dict(),
            "stop_reason": None if self.stop_reason is None else self.stop_reason.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResearchIterationState:
        _require_keys(payload, _ITERATION_STATE_KEYS, "iteration state")
        if _strict_int(payload, "schema_version") != ITERATION_SCHEMA_VERSION:
            raise ValueError("unsupported iteration state schema")
        active_raw = payload.get("active_plan")
        stop_raw = payload.get("stop_reason")
        active_inflight_raw = payload.get("active_inflight_query_id")
        if active_inflight_raw is not None and not isinstance(active_inflight_raw, str):
            raise ValueError("active_inflight_query_id must be a string or null")
        planning_inflight_raw = payload.get("planning_inflight")
        assessment_inflight_raw = payload.get("assessment_inflight")
        if not isinstance(planning_inflight_raw, bool) or not isinstance(
            assessment_inflight_raw, bool
        ):
            raise ValueError("in-flight flags must be bools")
        active_observations_raw = payload.get("active_observations", [])
        if not isinstance(active_observations_raw, list):
            raise ValueError("active_observations must be a list")
        waves_raw = payload.get("waves")
        if not isinstance(waves_raw, list):
            raise ValueError("waves must be a list")
        active_source_query_ids_raw = _strict_mapping(
            payload,
            "active_source_query_ids",
        )
        source_query_ids_raw = _strict_mapping(payload, "source_query_ids")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in (
                *active_source_query_ids_raw.items(),
                *source_query_ids_raw.items(),
            )
        ):
            raise ValueError("source provenance mappings must contain strings")
        if stop_raw is not None and not isinstance(stop_raw, str):
            raise ValueError("stop_reason must be a string or null")
        planning_limits_raw = _strict_mapping(payload, "planning_limits")
        _require_keys(
            planning_limits_raw,
            {"max_queries_per_wave", "max_query_chars"},
            "planning limits",
        )
        return cls(
            job_id=_strict_str(payload, "job_id"),
            research_brief_sha256=_strict_str(payload, "research_brief_sha256"),
            limits=IterationLimits.from_dict(_strict_mapping(payload, "limits")),
            planning_limits=PlanningLimits(
                max_queries_per_wave=_strict_int(
                    planning_limits_raw,
                    "max_queries_per_wave",
                ),
                max_query_chars=_strict_int(
                    planning_limits_raw,
                    "max_query_chars",
                ),
            ),
            capability_snapshot=_strict_capability_snapshot(
                _strict_mapping(payload, "capability_snapshot")
            ),
            phase=IterationPhase(_strict_str(payload, "phase")),
            next_wave_index=_strict_int(payload, "next_wave_index"),
            started_at_ms=_strict_int(payload, "started_at_ms"),
            active_plan=(
                None
                if active_raw is None
                else deserialize_search_plan(
                    json.dumps(active_raw, sort_keys=True).encode("utf-8")
                )
            ),
            planning_inflight=planning_inflight_raw,
            assessment_inflight=assessment_inflight_raw,
            active_inflight_query_id=active_inflight_raw,
            active_observations=tuple(
                SearchObservation.from_dict(item) for item in active_observations_raw
            ),
            active_source_refs=_strict_string_tuple(payload, "active_source_refs"),
            active_source_query_ids=dict(active_source_query_ids_raw),
            waves=tuple(WaveRecord.from_dict(item) for item in waves_raw),
            source_refs=_strict_string_tuple(payload, "source_refs"),
            source_query_ids=dict(source_query_ids_raw),
            searched_query_ids=_strict_string_tuple(payload, "searched_query_ids"),
            searched_query_texts=_strict_string_tuple(payload, "searched_query_texts"),
            exhausted_query_ids=_strict_string_tuple(payload, "exhausted_query_ids"),
            exhausted_query_texts=_strict_string_tuple(payload, "exhausted_query_texts"),
            exhausted_source_refs=_strict_string_tuple(payload, "exhausted_source_refs"),
            open_gaps=_strict_string_tuple(payload, "open_gaps"),
            accounting=ResearchAccounting.from_dict(
                _strict_mapping(payload, "accounting")
            ),
            stop_reason=(None if stop_raw is None else StopReason(stop_raw)),
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
        refs: list[str] = []
        provenance: dict[str, str] = {}
        for observation in observations:
            for source_ref in observation.result_refs:
                if source_ref in provenance:
                    continue
                refs.append(source_ref)
                provenance[source_ref] = observation.query_id
        return tuple(refs), provenance

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

    @staticmethod
    def _record_wave(
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

    @staticmethod
    def _validate_assessment(
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
        has_new_evidence = ResearchController._has_new_evidence(state, wave)
        if (
            assessment.decision is ContinuationDecision.STOP_COVERED
            and (not assessment.material_gain or not has_new_evidence)
        ):
            raise IterationInvariantError(
                "STOP_COVERED requires material gain and new bounded evidence"
            )
        if (
            assessment.decision is ContinuationDecision.STOP_NO_MATERIAL_GAIN
            and assessment.material_gain
        ):
            raise IterationInvariantError("STOP_NO_MATERIAL_GAIN cannot report material gain")

    @staticmethod
    def _has_new_evidence(state: ResearchIterationState, wave: WaveRecord) -> bool:
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

    @staticmethod
    def _apply_assessment(
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
                (*state.exhausted_query_texts,
                 *(query_text_by_id[query_id] for query_id in assessment.exhausted_query_ids))
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
            return ResearchController._replace_state(
                updated,
                phase=IterationPhase.STOPPED,
                stop_reason=StopReason.OBJECTIVE_COVERED,
            )
        if assessment.decision is ContinuationDecision.STOP_NO_MATERIAL_GAIN:
            return ResearchController._replace_state(
                updated,
                phase=IterationPhase.STOPPED,
                stop_reason=StopReason.NO_MATERIAL_GAIN,
            )
        if not assessment.material_gain or not ResearchController._has_new_evidence(
            state, state.waves[-1]
        ):
            return ResearchController._replace_state(
                updated,
                phase=IterationPhase.STOPPED,
                stop_reason=StopReason.NO_MATERIAL_GAIN,
            )
        return updated


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
