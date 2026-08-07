"""PRE2-C1A: wave-capable planning foundation (domain contracts + validator).

This module is the smallest production-quality, provider-free foundation
that makes a bounded SearchWave a first-class internal contract.

It is intentionally a FOUNDATION slice:

- It defines the typed planning domain (SearchPlan / PlannedSearchQuery /
  SearchObservation) and a deterministic structural validator.
- It defines stable, locally-recomputable ``query_id`` values (so that
  later runtime / replanner code can address queries without trusting
  model-supplied IDs).
- It binds every plan to the SHA-256 of the durable original Research
  Brief (the original brief itself is NOT copied into the plan artifact
  — the public persisted ``research_jobs.query`` remains the source of
  original intent).
- It performs NO search dispatch and makes NO LLM / provider call.
- It does NOT introduce a whole-job ``max_total_queries`` field. The
  active limit is ``max_queries_per_wave`` (per bounded SearchWave), so
  a future ResearchController can request additional waves without
  changing the C1A contract.
- It does NOT include a ResearchController, a GapAssessment, or a
  replanner. Those belong to PRE2-C1B and PRE2-C2.

Epistemic rule (enforced in code and tests)
-------------------------------------------
The validator proves STRUCTURAL correctness and DECLARED-COVERAGE
consistency of a candidate plan. It does NOT prove that an LLM
discovered every semantic research need in the Research Brief. That
distinction is intentional and is asserted in the public docstrings so
that later review (and later runtime code) does not accidentally treat
"structurally valid" as "semantically complete".
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final

# ---------------------------------------------------------------------------
# C1A constants (public for tests and downstream slices)
# ---------------------------------------------------------------------------

#: Supported schema version for a C1A SearchPlan. Bumped on breaking changes.
SCHEMA_VERSION: Final[int] = 1

#: Minimum queries per wave. The validator rejects 0-query waves.
MIN_QUERIES_PER_WAVE: Final[int] = 1

#: Maximum queries per wave. Per-wave bound, NOT a whole-job bound.
#: 1..MAX_QUERIES_PER_WAVE is the search-wave cardinality; a later
#: ResearchController may request additional waves (PRE2-C2) without
#: changing the meaning of this constant.
MAX_QUERIES_PER_WAVE: Final[int] = 4

#: Per-query text cap. Matches the Oroimen conservative operational cap
#: for Tavily introduced in PRE2-A2; tracked here as a planning-domain
#: fact so the validator can enforce it deterministically without
#: importing the search router. C1A's brief text is NOT subject to this
#: cap; only the DERIVED queries are.
MAX_QUERY_CHARS: Final[int] = 399

#: Bounded wave-index domain for the iterative C2 foundation. C1A/C1B use
#: wave zero; C2 may authorize later indices through its smaller per-job
#: ``IterationLimits.max_waves`` value.
MAX_WAVES_PER_JOB: Final[int] = 8
ALLOWED_WAVE_INDICES: Final[frozenset[int]] = frozenset(range(MAX_WAVES_PER_JOB))

#: Planner kinds known to the planning validator. C1A's deterministic
#: foundation remains available as a control; C1B adds provider-neutral
#: direct and structured-planner seams without selecting a provider.
KNOWN_PLANNER_KINDS: Final[frozenset[str]] = frozenset(
    {
        "c1a-deterministic-stub",
        "c1b-direct",
        "c1b-llm-structured",
    }
)

#: Semantic planner decisions are persisted with C1B plans.  The planner
#: proposes one of these values; the deterministic validator owns the bounds.
KNOWN_PLANNING_DECISIONS: Final[frozenset[str]] = frozenset(
    {"DIRECT", "DECOMPOSE"}
)

#: Regex for a bounded planner version string. Allows ``MAJOR.MINOR.PATCH``
#: with optional pre-release tag (lowercase alnum + dot + hyphen). Kept
#: strict so the validator can reject garbage without ambiguity.
_PLANNER_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.]+)?$"
)

#: Regex for a SHA-256 hex digest (64 lowercase hex chars).
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Regex for a dimension identifier. Non-empty, lowercase alnum + hyphen
#: + underscore, max 64 chars. Dimension ids appear in the plan artifact
#: and are exposed in observations; a tight regex keeps them safe to log.
_DIMENSION_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

#: Length of the hex prefix used for query_id and plan_id. 16 hex chars
#: (64 bits) is enough for collision avoidance within a single research
#: job while keeping ids short enough to embed in URLs and logs.
_ID_HEX_PREFIX_LEN: Final[int] = 16


# ---------------------------------------------------------------------------
# Custom exceptions (kept local; not exposed in hermes.jobs.__init__ yet)
# ---------------------------------------------------------------------------


class PlanningValidationError(ValueError):
    """Raised by the planning validator on a structural defect.

    The message is short, stable, and contains the failing check name
    (a single identifier) so logs and tests can match on it. The
    failing payload is exposed via ``violations`` (a tuple of
    ``(check_name, detail)``).
    """

    def __init__(self, violations: tuple[tuple[str, str], ...]) -> None:
        self.violations: tuple[tuple[str, str], ...] = violations
        joined = "; ".join(f"{name}:{detail}" for name, detail in violations)
        super().__init__(f"SearchPlan failed {len(violations)} check(s): {joined}")


def _require_exact_keys(
    payload: Mapping[str, Any],
    required: set[str],
    name: str,
    optional: set[str] | None = None,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    allowed = required | (optional or set())
    keys = set(payload)
    if not required.issubset(keys) or not keys.issubset(allowed):
        raise ValueError(f"{name} has an invalid field set")


def _strict_int(payload: Mapping[str, Any], name: str) -> int:
    value = payload[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    return value


def _strict_str(payload: Mapping[str, Any], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanningLimits:
    """Effective planning limits for ONE bounded SearchWave.

    This is a per-wave snapshot of the limits the validator enforces.
    It is intentionally NAMED ``PlanningLimits`` (not ``ResearchLimits``
    or ``JobLimits``) to make explicit that the bound is per wave, not
    per whole research job.

    A plan that names a ``PlanningLimits`` value with a different
    ``max_queries_per_wave`` from the contract constant would be
    rejected; the field is closed in C1A.
    """

    max_queries_per_wave: int
    max_query_chars: int

    def __post_init__(self) -> None:
        if not isinstance(self.max_queries_per_wave, int) or isinstance(
            self.max_queries_per_wave, bool
        ):
            raise ValueError("max_queries_per_wave must be an int")
        if not isinstance(self.max_query_chars, int) or isinstance(
            self.max_query_chars, bool
        ):
            raise ValueError("max_query_chars must be an int")
        if self.max_queries_per_wave < MIN_QUERIES_PER_WAVE:
            raise ValueError(
                f"max_queries_per_wave must be >= {MIN_QUERIES_PER_WAVE} "
                f"(got {self.max_queries_per_wave})"
            )
        if self.max_queries_per_wave > MAX_QUERIES_PER_WAVE:
            # The closed bound is part of the C1A contract; raising at
            # construction (not just at validation) keeps the type
            # self-consistent.
            raise ValueError(
                f"max_queries_per_wave must be <= {MAX_QUERIES_PER_WAVE} "
                f"(per-wave bound; whole-job limits are not a C1A concern) "
                f"(got {self.max_queries_per_wave})"
            )
        if self.max_query_chars < 1:
            raise ValueError("max_query_chars must be >= 1")
        if self.max_query_chars > MAX_QUERY_CHARS:
            raise ValueError(
                f"max_query_chars must be <= {MAX_QUERY_CHARS} "
                f"(got {self.max_query_chars})"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_queries_per_wave": self.max_queries_per_wave,
            "max_query_chars": self.max_query_chars,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlanningLimits:
        _require_exact_keys(
            payload,
            {"max_queries_per_wave", "max_query_chars"},
            "planning limits",
        )
        return cls(
            max_queries_per_wave=_strict_int(payload, "max_queries_per_wave"),
            max_query_chars=_strict_int(payload, "max_query_chars"),
        )


@dataclass(frozen=True)
class PlannedSearchQuery:
    """One planned derived search query inside a bounded SearchWave.

    ``query_id`` is generated by ``compute_query_id`` from deterministic
    local inputs. The validator re-derives ``query_id`` from the
    remaining fields and rejects the plan if the stored id does not
    match; this prevents a model or a buggy caller from spoofing ids.

    ``text`` is the provider-facing retrieval instruction (NOT the
    original Research Brief). It is bounded to ``MAX_QUERY_CHARS`` per
    the Oroimen conservative operational cap; the original brief has
    no such cap and is referenced only by hash.

    ``purpose`` is a short non-empty human-readable rationale.

    ``dimension_ids`` is a non-empty tuple of dimension identifiers the
    query is intended to cover. In C1A every query is a "decomposed"
    query (the "direct" plan shape is a future C1B concern), so an
    empty dimension list is rejected at validation time.

    ``ordinal`` is the deterministic 0-based position of the query
    inside the wave. The validator requires ordinals to be contiguous
    starting at 0.
    """

    query_id: str
    text: str
    purpose: str
    dimension_ids: tuple[str, ...]
    ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "purpose": self.purpose,
            "dimension_ids": list(self.dimension_ids),
            "ordinal": self.ordinal,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannedSearchQuery:
        _require_exact_keys(
            payload,
            {"query_id", "text", "purpose", "dimension_ids", "ordinal"},
            "planned search query",
        )
        dim_ids_raw = payload["dimension_ids"]
        if not isinstance(dim_ids_raw, list):
            raise ValueError("dimension_ids must be a list")
        if not all(isinstance(value, str) for value in dim_ids_raw):
            raise ValueError("dimension_ids must contain strings")
        return cls(
            query_id=_strict_str(payload, "query_id"),
            text=_strict_str(payload, "text"),
            purpose=_strict_str(payload, "purpose"),
            dimension_ids=tuple(dim_ids_raw),
            ordinal=_strict_int(payload, "ordinal"),
        )


@dataclass(frozen=True)
class EvidenceItem:
    """Bounded, sanitized evidence metadata retained for gap assessment."""

    query_id: str
    source_ref: str
    title: str = ""
    snippet: str = ""
    digest: str = ""

    def __post_init__(self) -> None:
        for name, value, max_chars in (
            ("query_id", self.query_id, 128),
            ("source_ref", self.source_ref, 2_048),
            ("title", self.title, 512),
            ("snippet", self.snippet, 1_024),
        ):
            if (
                not isinstance(value, str)
                or len(value) > max_chars
                or "\n" in value
                or "\r" in value
            ):
                raise ValueError(f"{name} is not a bounded single-line string")
        if not isinstance(self.digest, str) or len(self.digest) != 64 or not all(
            char in "0123456789abcdef" for char in self.digest
        ):
            raise ValueError("digest must be a lowercase SHA-256 value")
        if self.digest != compute_evidence_digest(
            self.source_ref, self.title, self.snippet
        ):
            raise ValueError("digest does not match persisted evidence")

    def to_dict(self) -> dict[str, str]:
        return {
            "query_id": self.query_id,
            "source_ref": self.source_ref,
            "title": self.title,
            "snippet": self.snippet,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceItem:
        _require_exact_keys(
            payload,
            {"query_id", "source_ref", "title", "snippet", "digest"},
            "evidence item",
        )
        return cls(
            query_id=_strict_str(payload, "query_id"),
            source_ref=_strict_str(payload, "source_ref"),
            title=_strict_str(payload, "title"),
            snippet=_strict_str(payload, "snippet"),
            digest=_strict_str(payload, "digest"),
        )


@dataclass(frozen=True)
class SearchObservation:
    """Minimal observation shape for a planned query's execution.

    C1A does NOT execute searches and does NOT fill real observations.
    The shape is defined here so C1B and later runtime code can produce
    observations that the validator and store accept without a contract
    change.

    Where reasonable, fields are composed with the existing
    ``hermes.services.search`` types (``backend``, ``result_refs``,
    ``structured_error``) rather than redefined; this keeps the
    cross-slice vocabulary consistent.
    """

    wave_index: int
    query_id: str
    backend: str | None = None
    result_refs: tuple[str, ...] = ()
    structured_error: str | None = None
    attempt_count: int = 1
    duration_ms: int | None = None
    evidence_items: tuple[EvidenceItem, ...] = ()
    # Bounded digests prove that evidence survived materialization without
    # persisting provider snippets/content or other untrusted payloads.
    evidence_digests: tuple[str, ...] = ()
    # ``local_usage`` is intentionally a small open-ended mapping
    # (provider/model tokens, byte counts, etc.). It is NOT validated
    # by the structural validator; runtime code is responsible for
    # not stuffing large or sensitive data into it.
    local_usage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.wave_index, int) or isinstance(self.wave_index, bool):
            raise ValueError("wave_index must be an int")
        if self.wave_index not in ALLOWED_WAVE_INDICES:
            raise ValueError(
                f"wave_index must be in {sorted(ALLOWED_WAVE_INDICES)} "
                f"(got {self.wave_index})"
            )
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(self.attempt_count, int) or isinstance(self.attempt_count, bool):
            raise ValueError("attempt_count must be an int")
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be >= 1")
        if not isinstance(self.local_usage, Mapping):
            raise ValueError("local_usage must be a mapping")
        if not isinstance(self.result_refs, tuple) or len(self.result_refs) > 32:
            raise ValueError("result_refs must be a bounded tuple")
        if not all(
            type(source_ref) is str and 0 < len(source_ref) <= 2_048
            for source_ref in self.result_refs
        ):
            raise ValueError("result_refs must be bounded strings")
        if not isinstance(self.evidence_digests, tuple):
            raise ValueError("evidence_digests must be a tuple")
        if len(self.evidence_digests) > 32:
            raise ValueError("evidence_digests exceeds the per-query cap")
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in self.evidence_digests
        ):
            raise ValueError("evidence_digests must be lowercase SHA-256 values")
        if not isinstance(self.evidence_items, tuple) or len(self.evidence_items) > 32:
            raise ValueError("evidence_items must be a bounded tuple")
        if not all(type(item) is EvidenceItem for item in self.evidence_items):
            raise ValueError("evidence_items must contain EvidenceItem values")
        if len(self.evidence_items) != len(self.result_refs):
            raise ValueError("evidence_items must align with result refs")
        if len(self.evidence_digests) != len(self.evidence_items):
            raise ValueError("evidence_digests must align with evidence items")
        if self.evidence_items or self.evidence_digests:
            if any(item.query_id != self.query_id for item in self.evidence_items):
                raise ValueError("evidence item query provenance drifted")
            if any(
                item.source_ref != source_ref or item.digest != digest
                for item, source_ref, digest in zip(
                    self.evidence_items,
                    self.result_refs,
                    self.evidence_digests,
                    strict=True,
                )
            ):
                raise ValueError("evidence item provenance drifted")
        if self.structured_error is not None and (
            self.result_refs or self.evidence_items or self.evidence_digests
        ):
            raise ValueError("structured errors cannot carry evidence")

    def to_dict(self) -> dict[str, Any]:
        # ``local_usage`` may be a dict or a MappingProxyType. We
        # convert to a sorted list of (key, value) pairs for stable
        # JSON serialization (dict iteration order is insertion order
        # in CPython 3.7+ but we make the stability explicit).
        usage_items = sorted(
            (str(k), v) for k, v in self.local_usage.items()
        )
        return {
            "wave_index": self.wave_index,
            "query_id": self.query_id,
            "backend": self.backend,
            "result_refs": list(self.result_refs),
            "structured_error": self.structured_error,
            "attempt_count": self.attempt_count,
            "duration_ms": self.duration_ms,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "evidence_digests": list(self.evidence_digests),
            "local_usage": usage_items,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SearchObservation:
        _require_exact_keys(
            payload,
            {
                "wave_index",
                "query_id",
                "backend",
                "result_refs",
                "structured_error",
                "attempt_count",
                "duration_ms",
                "evidence_items",
                "evidence_digests",
                "local_usage",
            },
            "search observation",
        )
        result_refs = payload["result_refs"]
        evidence_digests = payload["evidence_digests"]
        evidence_items_raw = payload["evidence_items"]
        usage_items = payload["local_usage"]
        if not isinstance(result_refs, list) or not all(
            isinstance(value, str) for value in result_refs
        ):
            raise ValueError("result_refs must be a list of strings")
        if not isinstance(evidence_digests, list) or not all(
            isinstance(value, str) for value in evidence_digests
        ):
            raise ValueError("evidence_digests must be a list of strings")
        if not isinstance(evidence_items_raw, list) or not all(
            isinstance(value, Mapping) for value in evidence_items_raw
        ):
            raise ValueError("evidence_items must be a list of objects")
        if not isinstance(usage_items, list):
            raise ValueError("local_usage must be a list")
        usage: dict[str, Any] = {}
        for item in usage_items:
            if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[0], str):
                raise ValueError("local_usage must contain [string, value] pairs")
            if item[0] in usage:
                raise ValueError("local_usage must not contain duplicate keys")
            usage[item[0]] = item[1]
        backend = payload["backend"]
        structured_error = payload["structured_error"]
        duration_ms = payload["duration_ms"]
        if backend is not None and not isinstance(backend, str):
            raise ValueError("backend must be a string or null")
        if structured_error is not None and not isinstance(structured_error, str):
            raise ValueError("structured_error must be a string or null")
        if duration_ms is not None and (
            not isinstance(duration_ms, int) or isinstance(duration_ms, bool)
        ):
            raise ValueError("duration_ms must be an int or null")
        return cls(
            wave_index=_strict_int(payload, "wave_index"),
            query_id=_strict_str(payload, "query_id"),
            backend=backend,
            result_refs=tuple(result_refs),
            structured_error=structured_error,
            attempt_count=_strict_int(payload, "attempt_count"),
            duration_ms=duration_ms,
            evidence_items=tuple(
                EvidenceItem.from_dict(item) for item in evidence_items_raw
            ),
            evidence_digests=tuple(evidence_digests),
            local_usage=usage,
        )


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Capability / limit snapshot captured at planning time.

    The snapshot is a stable, JSON-serializable view of the runtime
    capabilities that affected the plan's limits. The plan store
    refuses to load a plan whose ``capability_snapshot`` does not
    byte-equal the snapshot the caller expects (fail-closed on
    capability drift).

    The shape is intentionally narrow: planner kind/version and planning
    limits, plus an optional bounded provider/model/version descriptor.
    The provenance field is omitted from C1A serialization when empty so
    existing C1A artifacts retain their byte shape.
    """

    planner_kind: str
    planner_version: str
    max_queries_per_wave: int
    max_query_chars: int
    planner_provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.planner_kind, str) or not self.planner_kind:
            raise ValueError("planner_kind must be a non-empty string")
        if self.planner_kind not in KNOWN_PLANNER_KINDS:
            raise ValueError(
                f"planner_kind must be one of {sorted(KNOWN_PLANNER_KINDS)} "
                f"(got {self.planner_kind!r})"
            )
        if not isinstance(self.planner_version, str) or not _PLANNER_VERSION_RE.match(
            self.planner_version
        ):
            raise ValueError(
                f"planner_version must match {_PLANNER_VERSION_RE.pattern!r} "
                f"(got {self.planner_version!r})"
            )
        if (
            not isinstance(self.max_queries_per_wave, int)
            or isinstance(self.max_queries_per_wave, bool)
            or self.max_queries_per_wave < MIN_QUERIES_PER_WAVE
            or self.max_queries_per_wave > MAX_QUERIES_PER_WAVE
        ):
            raise ValueError(
                f"max_queries_per_wave must be in "
                f"[{MIN_QUERIES_PER_WAVE}, {MAX_QUERIES_PER_WAVE}] "
                f"(got {self.max_queries_per_wave})"
            )
        if (
            not isinstance(self.max_query_chars, int)
            or isinstance(self.max_query_chars, bool)
            or self.max_query_chars < 1
            or self.max_query_chars > MAX_QUERY_CHARS
        ):
            raise ValueError(
                f"max_query_chars must be in [1, {MAX_QUERY_CHARS}]"
            )
        if not isinstance(self.planner_provenance, Mapping):
            raise ValueError("planner_provenance must be a mapping")
        allowed_keys = {"provider", "model", "version"}
        unknown_keys = set(self.planner_provenance) - allowed_keys
        if unknown_keys:
            raise ValueError(
                "planner_provenance contains unsupported keys: "
                + ",".join(sorted(str(key) for key in unknown_keys))
            )
        for key, value in self.planner_provenance.items():
            if not isinstance(key, str) or not key:
                raise ValueError("planner_provenance keys must be non-empty strings")
            if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
                raise ValueError(
                    "planner_provenance values must be non-empty single-line strings"
                )
        object.__setattr__(
            self,
            "planner_provenance",
            MappingProxyType(dict(self.planner_provenance)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "planner_kind": self.planner_kind,
            "planner_version": self.planner_version,
            "max_queries_per_wave": self.max_queries_per_wave,
            "max_query_chars": self.max_query_chars,
        }
        if self.planner_provenance:
            payload["planner_provenance"] = {
                key: self.planner_provenance[key]
                for key in sorted(self.planner_provenance)
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CapabilitySnapshot:
        allowed = {
            "planner_kind",
            "planner_version",
            "max_queries_per_wave",
            "max_query_chars",
            "planner_provenance",
        }
        required = allowed - {"planner_provenance"}
        _require_exact_keys(payload, required, "capability snapshot", allowed - required)
        provenance_raw = payload.get("planner_provenance", {})
        if not isinstance(provenance_raw, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in provenance_raw.items()
        ):
            raise ValueError("planner_provenance must be a string mapping")
        return cls(
            planner_kind=_strict_str(payload, "planner_kind"),
            planner_version=_strict_str(payload, "planner_version"),
            max_queries_per_wave=_strict_int(payload, "max_queries_per_wave"),
            max_query_chars=_strict_int(payload, "max_query_chars"),
            planner_provenance=dict(provenance_raw),
        )


@dataclass(frozen=True)
class SearchPlan:
    """A SearchPlan represents ONE bounded SearchWave.

    The plan binds the original Research Brief by SHA-256 only; the
    brief text itself is NOT copied into the plan artifact (the
    public persisted ``research_jobs.query`` is the source of
    original intent).

    ``wave_index`` is the position of the wave in the bounded research loop.
    C1A/C1B use wave zero; C2 may produce later waves within the hard
    ``MAX_WAVES_PER_JOB`` safety bound.

    ``queries`` is an ordered tuple of ``PlannedSearchQuery``. The
    tuple length is bounded to ``[MIN_QUERIES_PER_WAVE,
    MAX_QUERIES_PER_WAVE]`` (per wave, not per job).
    """

    schema_version: int
    planner_kind: str
    planner_version: str
    research_brief_sha256: str
    wave_index: int
    queries: tuple[PlannedSearchQuery, ...]
    planning_limits: PlanningLimits
    capability_snapshot: CapabilitySnapshot
    created_at: str  # ISO 8601 UTC, e.g. "2026-08-07T12:00:00Z"
    # C1A artifacts omit this optional C1B field.  C1B persists the semantic
    # planner's decision so recovery does not infer it from query count.
    planning_decision: str | None = None


# ---------------------------------------------------------------------------
# Stable id helpers
# ---------------------------------------------------------------------------


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Serialize a mapping deterministically for hashing purposes.

    Uses ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` so
    the output is stable across runs and platforms. UTF-8 is used
    so non-ASCII brief text hash-equal across encodings.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_evidence_digest(source_ref: str, title: str, snippet: str) -> str:
    """Hash exactly the bounded evidence fields retained in a checkpoint."""

    return hashlib.sha256(
        _canonical_json(
            {"source_ref": source_ref, "title": title, "snippet": snippet}
        )
    ).hexdigest()


def compute_research_brief_sha256(brief_text: str) -> str:
    """Return the lowercase hex SHA-256 of the Research Brief text.

    The brief is hashed as UTF-8 bytes. The function accepts any
    non-empty string; the validator enforces the hex format on
    ``SearchPlan.research_brief_sha256``, not on this helper.
    """
    if not isinstance(brief_text, str):
        raise TypeError("brief_text must be a str")
    return hashlib.sha256(brief_text.encode("utf-8")).hexdigest()


def compute_query_id(
    *,
    schema_version: int,
    wave_index: int,
    ordinal: int,
    normalized_text: str,
    research_brief_sha256: str,
) -> str:
    """Return a stable, locally-recomputable id for a planned query.

    Inputs are the minimal deterministic local context: schema version,
    wave index, ordinal, the query's normalized text, and the SHA-256
    of the Research Brief the query was derived from. The function
    makes no network / model call; running it twice on the same
    inputs yields the same id.

    The returned id is ``"q-" + <16 hex chars>``. The 64-bit prefix is
    derived from SHA-256 over a canonical JSON encoding of the inputs;
    a future slice may move to 128 bits by raising
    ``_ID_HEX_PREFIX_LEN`` without breaking the prefix tag.
    """
    if not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a str")
    if not isinstance(research_brief_sha256, str):
        raise TypeError("research_brief_sha256 must be a str")
    payload = {
        "schema_version": schema_version,
        "wave_index": wave_index,
        "ordinal": ordinal,
        "normalized_text": normalized_text,
        "research_brief_sha256": research_brief_sha256,
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return "q-" + digest[:_ID_HEX_PREFIX_LEN]


def normalize_query_text(text: str) -> str:
    """Deterministic normalization for duplicate-detection (whitespace only).

    Per the C1A contract, duplicate detection is performed after a
    documented deterministic normalization LIMITED to whitespace
    trimming. This function is the single source of truth for that
    normalization; it is intentionally narrow (no Unicode folding, no
    case folding, no punctuation normalization) so the comparison
    result is easy to reason about and audit.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return text.strip()


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _violation(name: str, detail: str) -> tuple[str, str]:
    return (name, detail)


def validate_search_plan(
    plan: SearchPlan,
    *,
    expected_research_brief_sha256: str | None = None,
) -> None:
    """Validate a candidate SearchPlan against the C1A hard rules.

    The validator proves structural correctness and declared-coverage
    consistency. It does NOT claim to prove that an LLM discovered
    every semantic research need in the Research Brief. That
    distinction is asserted in the module docstring and exercised
    by ``test_planning.py``.

    When ``expected_research_brief_sha256`` is provided, the
    validator also enforces exact match (byte-for-byte lowercase
    hex equality). When it is ``None``, the validator only checks
    that the plan's bound hash is a well-formed SHA-256 string.
    """
    violations: list[tuple[str, str]] = []

    # (1) schema version
    if plan.schema_version != SCHEMA_VERSION:
        violations.append(
            _violation(
                "schema_version",
                f"must be {SCHEMA_VERSION}, got {plan.schema_version}",
            )
        )

    # (2) planner kind
    if plan.planner_kind not in KNOWN_PLANNER_KINDS:
        violations.append(
            _violation(
                "planner_kind",
                f"must be one of {sorted(KNOWN_PLANNER_KINDS)}, "
                f"got {plan.planner_kind!r}",
            )
        )

    # (3) planner version
    if not _PLANNER_VERSION_RE.match(plan.planner_version):
        violations.append(
            _violation(
                "planner_version",
                f"must match {_PLANNER_VERSION_RE.pattern!r}, "
                f"got {plan.planner_version!r}",
            )
        )

    # (4) brief hash format
    if not _SHA256_RE.match(plan.research_brief_sha256):
        violations.append(
            _violation(
                "research_brief_sha256",
                "must be a 64-char lowercase hex SHA-256 digest",
            )
        )

    # (5) optional exact brief hash binding
    if (
        expected_research_brief_sha256 is not None
        and expected_research_brief_sha256 != plan.research_brief_sha256
    ):
        violations.append(
            _violation(
                "research_brief_sha256_mismatch",
                "plan's bound hash does not match the expected brief hash",
            )
        )

    # (6) wave index
    if plan.wave_index not in ALLOWED_WAVE_INDICES:
        violations.append(
            _violation(
                "wave_index",
                f"must be in {sorted(ALLOWED_WAVE_INDICES)} for the bounded "
                "wave domain, "
                f"got {plan.wave_index}",
            )
        )

    # (7) queries cardinality (per-wave, not whole-job)
    n = len(plan.queries)
    if n < MIN_QUERIES_PER_WAVE:
        violations.append(
            _violation(
                "queries_count",
                f"wave must have >= {MIN_QUERIES_PER_WAVE} queries, got {n}",
            )
        )
    if n > MAX_QUERIES_PER_WAVE:
        violations.append(
            _violation(
                "queries_count",
                f"wave must have <= {MAX_QUERIES_PER_WAVE} queries "
                f"(per-wave bound; whole-job limits are a future concern), "
                f"got {n}",
            )
        )

    if n > plan.planning_limits.max_queries_per_wave:
        violations.append(
            _violation(
                "queries_count_planning_limit",
                "wave query count exceeds planning_limits.max_queries_per_wave",
            )
        )

    # (8) per-query checks
    seen_normalized: dict[str, int] = {}
    seen_ordinals: set[int] = set()
    expected_ordinals = set(range(n))
    for q in plan.queries:
        # (8.a) ordinal contiguous and 0-based
        if q.ordinal not in expected_ordinals:
            violations.append(
                _violation(
                    "query_ordinals",
                    f"query id {q.query_id!r} has ordinal {q.ordinal} "
                    f"outside 0..{n - 1} or duplicated",
                )
            )
        if q.ordinal in seen_ordinals:
            violations.append(
                _violation(
                    "query_ordinals",
                    f"duplicate ordinal {q.ordinal} for query id {q.query_id!r}",
                )
            )
        seen_ordinals.add(q.ordinal)

        # (8.b) non-empty text
        if not isinstance(q.text, str) or not q.text:
            violations.append(
                _violation(
                    "query_text_empty",
                    f"query id {q.query_id!r} has empty text",
                )
            )
        elif q.text.strip() == "":
            violations.append(
                _violation(
                    "query_text_whitespace",
                    f"query id {q.query_id!r} has whitespace-only text",
                )
            )

        # (8.c) per-query text cap
        if isinstance(q.text, str) and len(q.text) > MAX_QUERY_CHARS:
            violations.append(
                _violation(
                    "query_text_too_long",
                    f"query id {q.query_id!r} has len {len(q.text)} > "
                    f"{MAX_QUERY_CHARS}",
                )
            )

        if (
            isinstance(q.text, str)
            and len(q.text) > plan.planning_limits.max_query_chars
        ):
            violations.append(
                _violation(
                    "query_text_planning_limit",
                    f"query id {q.query_id!r} exceeds the effective query cap",
                )
            )

        # (8.d) non-empty purpose
        if not isinstance(q.purpose, str) or not q.purpose.strip():
            violations.append(
                _violation(
                    "query_purpose_empty",
                    f"query id {q.query_id!r} has empty purpose",
                )
            )

        # (8.e) non-empty dimension list (C1A: all queries are decomposed)
        if not q.dimension_ids:
            violations.append(
                _violation(
                    "query_dimensions_empty",
                    f"query id {q.query_id!r} has empty dimension_ids "
                    f"(C1A requires a non-empty declared dimension list)",
                )
            )
        else:
            for did in q.dimension_ids:
                if not _DIMENSION_ID_RE.match(did):
                    violations.append(
                        _violation(
                            "query_dimension_id_format",
                            f"query id {q.query_id!r} has invalid dimension_id "
                            f"{did!r} (must match {_DIMENSION_ID_RE.pattern!r})",
                        )
                    )

        # (8.f) query id locally recomputable and matching
        expected_qid = compute_query_id(
            schema_version=plan.schema_version,
            wave_index=plan.wave_index,
            ordinal=q.ordinal,
            normalized_text=normalize_query_text(q.text),
            research_brief_sha256=plan.research_brief_sha256,
        )
        if q.query_id != expected_qid:
            violations.append(
                _violation(
                    "query_id_mismatch",
                    f"query id {q.query_id!r} does not match locally "
                    f"recomputed {expected_qid!r}",
                )
            )

        # (8.g) duplicate text after deterministic whitespace/trim
        norm = normalize_query_text(q.text)
        if norm in seen_normalized:
            other_ord = seen_normalized[norm]
            violations.append(
                _violation(
                    "duplicate_query",
                    f"query id {q.query_id!r} (ordinal {q.ordinal}) duplicates "
                    f"query at ordinal {other_ord} after whitespace/trim "
                    f"normalization",
                )
            )
        else:
            seen_normalized[norm] = q.ordinal

    # (9) planning limits present and bounded
    if plan.planning_limits.max_queries_per_wave != plan.capability_snapshot.max_queries_per_wave:
        violations.append(
            _violation(
                "planning_limits_consistency",
                "planning_limits.max_queries_per_wave must equal "
                "capability_snapshot.max_queries_per_wave",
            )
        )
    if plan.planning_limits.max_query_chars != plan.capability_snapshot.max_query_chars:
        violations.append(
            _violation(
                "planning_limits_consistency",
                "planning_limits.max_query_chars must equal "
                "capability_snapshot.max_query_chars",
            )
        )

    # (10) planner kind/version in capability snapshot match the plan
    if plan.capability_snapshot.planner_kind != plan.planner_kind:
        violations.append(
            _violation(
                "capability_planner_kind",
                "capability_snapshot.planner_kind must equal plan.planner_kind",
            )
        )
    if plan.capability_snapshot.planner_version != plan.planner_version:
        violations.append(
            _violation(
                "capability_planner_version",
                "capability_snapshot.planner_version must equal plan.planner_version",
            )
        )
    if plan.planner_kind == "c1b-llm-structured" and set(
        plan.capability_snapshot.planner_provenance
    ) != {"provider", "model", "version"}:
        violations.append(
            _violation(
                "planner_provenance_missing",
                "structured C1B plans must persist provider/model/version provenance",
            )
        )

    # (11) C1B semantic decision, when present
    if (
        plan.planner_kind in {"c1b-direct", "c1b-llm-structured"}
        and plan.planning_decision is None
    ):
        violations.append(
            _violation(
                "planning_decision_missing",
                "C1B plans must persist the semantic planner decision",
            )
        )
    if plan.planning_decision is not None:
        if plan.planning_decision not in KNOWN_PLANNING_DECISIONS:
            violations.append(
                _violation(
                    "planning_decision",
                    "must be DIRECT or DECOMPOSE",
                )
            )
        elif plan.planning_decision == "DIRECT" and len(plan.queries) != 1:
            violations.append(
                _violation(
                    "planning_decision_cardinality",
                    "DIRECT requires exactly one query",
                )
            )
        elif plan.planning_decision == "DECOMPOSE" and not (
            2
            <= len(plan.queries)
            <= min(
                MAX_QUERIES_PER_WAVE,
                plan.planning_limits.max_queries_per_wave,
            )
        ):
            violations.append(
                _violation(
                    "planning_decision_cardinality",
                    "DECOMPOSE requires 2..4 queries",
                )
            )
    # (12) created_at is a non-empty string (the format is left to
    # the caller; C1A does not need to lock the format because the
    # store does not parse the timestamp).
    if not isinstance(plan.created_at, str) or not plan.created_at:
        violations.append(
            _violation(
                "created_at_empty",
                "created_at must be a non-empty string",
            )
        )

    # (13) no extra recursive / nested plan structure. The frozen
    # dataclass forbids nested SearchPlan / wave objects by typing,
    # so this is enforced statically; we still verify by reflection
    # that no field of the plan is itself a SearchPlan / tuple of
    # SearchPlan.
    for fname in plan.__dataclass_fields__:
        fvalue = getattr(plan, fname)
        if isinstance(fvalue, SearchPlan):
            violations.append(
                _violation(
                    "nested_plan_structure",
                    f"field {fname!r} is a SearchPlan (C1A forbids nesting)",
                )
            )
        if isinstance(fvalue, tuple):
            for inner in fvalue:
                if isinstance(inner, SearchPlan):
                    violations.append(
                        _violation(
                            "nested_plan_structure",
                            f"field {fname!r} contains a nested SearchPlan "
                            f"(C1A forbids nesting)",
                        )
                    )

    if violations:
        raise PlanningValidationError(tuple(violations))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_search_plan(
    *,
    planner_kind: str,
    planner_version: str,
    research_brief_sha256: str,
    wave_index: int,
    queries: tuple[PlannedSearchQuery, ...]
    | list[PlannedSearchQuery],
    planning_limits: PlanningLimits | None = None,
    capability_snapshot: CapabilitySnapshot | None = None,
    created_at: str,
    schema_version: int = SCHEMA_VERSION,
    expected_research_brief_sha256: str | None = None,
    planning_decision: str | None = None,
) -> SearchPlan:
    """Build and validate a SearchPlan in one call.

    The builder accepts the same field set as ``SearchPlan`` plus
    optional ``PlanningLimits`` and ``CapabilitySnapshot`` defaults.
    When the caller omits them, the builder uses the C1A defaults
    (``MIN/MAX_QUERIES_PER_WAVE`` and ``MAX_QUERY_CHARS`` plus the
    closed ``c1a-deterministic-stub`` planner). The builder runs
    ``validate_search_plan`` and raises ``PlanningValidationError``
    on any structural defect.
    """
    if planning_limits is None:
        planning_limits = PlanningLimits(
            max_queries_per_wave=MAX_QUERIES_PER_WAVE,
            max_query_chars=MAX_QUERY_CHARS,
        )
    if capability_snapshot is None:
        capability_snapshot = CapabilitySnapshot(
            planner_kind=planner_kind,
            planner_version=planner_version,
            max_queries_per_wave=planning_limits.max_queries_per_wave,
            max_query_chars=planning_limits.max_query_chars,
        )

    # Normalize the queries tuple so the typed contract is stable
    # regardless of whether the caller passed a list or a tuple.
    queries_tuple = tuple(queries)

    plan = SearchPlan(
        schema_version=schema_version,
        planner_kind=planner_kind,
        planner_version=planner_version,
        research_brief_sha256=research_brief_sha256,
        wave_index=wave_index,
        queries=queries_tuple,
        planning_limits=planning_limits,
        capability_snapshot=capability_snapshot,
        created_at=created_at,
        planning_decision=planning_decision,
    )
    validate_search_plan(
        plan,
        expected_research_brief_sha256=expected_research_brief_sha256,
    )
    return plan


# ---------------------------------------------------------------------------
# Serialization (used by the plan store)
# ---------------------------------------------------------------------------


def serialize_search_plan(plan: SearchPlan) -> bytes:
    """Serialize a SearchPlan to deterministic UTF-8 JSON bytes.

    Field order is fixed: schema_version, planner_kind, planner_version,
    research_brief_sha256, wave_index, queries, planning_limits,
    capability_snapshot, created_at, and optional planning_decision.
    ``sort_keys=True`` provides a
    second layer of stability for any nested mappings we might add in
    the future. ``ensure_ascii=False`` keeps the output compact while
    staying strict UTF-8.

    The store wraps this in a versioned envelope (see
    ``hermes.deep_research.plan_store``).
    """
    payload = {
        "schema_version": plan.schema_version,
        "planner_kind": plan.planner_kind,
        "planner_version": plan.planner_version,
        "research_brief_sha256": plan.research_brief_sha256,
        "wave_index": plan.wave_index,
        "queries": [q.to_dict() for q in plan.queries],
        "planning_limits": plan.planning_limits.to_dict(),
        "capability_snapshot": plan.capability_snapshot.to_dict(),
        "created_at": plan.created_at,
    }
    if plan.planning_decision is not None:
        payload["planning_decision"] = plan.planning_decision
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def deserialize_search_plan(blob: bytes) -> SearchPlan:
    """Parse the JSON bytes produced by ``serialize_search_plan``.

    Strict UTF-8 decode + strict JSON parse. The returned plan is
    NOT yet validated; the store validates before returning a
    reusable plan.
    """
    if isinstance(blob, (memoryview, bytearray)):
        blob = bytes(blob)
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError("blob must be bytes")
    try:
        text = bytes(blob).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("plan blob is not valid UTF-8") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"plan blob is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("plan JSON must be a top-level object")

    _require_exact_keys(
        payload,
        {
            "schema_version",
            "planner_kind",
            "planner_version",
            "research_brief_sha256",
            "wave_index",
            "queries",
            "planning_limits",
            "capability_snapshot",
            "created_at",
        },
        "search plan",
        {"planning_decision"},
    )

    queries_raw = payload["queries"]
    if not isinstance(queries_raw, list):
        raise ValueError("queries must be a list")
    if not all(isinstance(item, Mapping) for item in queries_raw):
        raise ValueError("queries must contain objects")
    queries = tuple(PlannedSearchQuery.from_dict(q) for q in queries_raw)

    planning_decision = payload.get("planning_decision")
    if planning_decision is not None and not isinstance(planning_decision, str):
        raise ValueError("planning_decision must be a string or null")
    planning_limits_raw = payload["planning_limits"]
    capability_raw = payload["capability_snapshot"]
    if not isinstance(planning_limits_raw, Mapping):
        raise ValueError("planning_limits must be an object")
    if not isinstance(capability_raw, Mapping):
        raise ValueError("capability_snapshot must be an object")

    return SearchPlan(
        schema_version=_strict_int(payload, "schema_version"),
        planner_kind=_strict_str(payload, "planner_kind"),
        planner_version=_strict_str(payload, "planner_version"),
        research_brief_sha256=_strict_str(payload, "research_brief_sha256"),
        wave_index=_strict_int(payload, "wave_index"),
        queries=queries,
        planning_limits=PlanningLimits.from_dict(planning_limits_raw),
        capability_snapshot=CapabilitySnapshot.from_dict(capability_raw),
        created_at=_strict_str(payload, "created_at"),
        planning_decision=planning_decision,
    )


__all__ = [
    "ALLOWED_WAVE_INDICES",
    "KNOWN_PLANNER_KINDS",
    "KNOWN_PLANNING_DECISIONS",
    "MAX_QUERIES_PER_WAVE",
    "MAX_QUERY_CHARS",
    "MAX_WAVES_PER_JOB",
    "MIN_QUERIES_PER_WAVE",
    "SCHEMA_VERSION",
    "CapabilitySnapshot",
    "EvidenceItem",
    "PlannedSearchQuery",
    "PlanningLimits",
    "PlanningValidationError",
    "SearchObservation",
    "SearchPlan",
    "build_search_plan",
    "compute_evidence_digest",
    "compute_query_id",
    "compute_research_brief_sha256",
    "deserialize_search_plan",
    "normalize_query_text",
    "replace",  # re-exported because callers may need to build a derived plan
    "serialize_search_plan",
    "validate_search_plan",
]
