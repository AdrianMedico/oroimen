"""Tests for the PRE2-C1A planning domain (hermes.jobs.planning).

These tests cover the public surface of the planning foundation:

- Hard structural validation (16 cases from the C1A mission spec).
- Long-Research-Brief acceptance without applying the public 2,000
  character cap (the cap is a public-API concern; the planning
  domain is a private internal contract).
- Deterministic query-id / serialization stability.

The tests use small in-process fakes (no LLM, no network, no DB).
They are pure-Python and run under the standard ``pytest`` runner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from hermes.jobs.planning import (
    ALLOWED_WAVE_INDICES,
    KNOWN_PLANNER_KINDS,
    MAX_QUERIES_PER_WAVE,
    MAX_QUERY_CHARS,
    SCHEMA_VERSION,
    CapabilitySnapshot,
    PlannedSearchQuery,
    PlanningLimits,
    PlanningValidationError,
    SearchObservation,
    SearchPlan,
    build_search_plan,
    compute_query_id,
    compute_research_brief_sha256,
    deserialize_search_plan,
    normalize_query_text,
    serialize_search_plan,
    validate_search_plan,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


PLANNER_KIND = "c1a-deterministic-stub"
PLANNER_VERSION = "0.1.0"
CREATED_AT = "2026-08-07T12:00:00Z"


def _brief_sha(*parts: str) -> str:
    """Return a deterministic SHA-256 for a brief made of ``parts``."""
    return compute_research_brief_sha256("\n\n".join(parts))


def _planning_limits() -> PlanningLimits:
    return PlanningLimits(
        max_queries_per_wave=MAX_QUERIES_PER_WAVE,
        max_query_chars=MAX_QUERY_CHARS,
    )


def _capability_snapshot(
    planner_kind: str = PLANNER_KIND,
    planner_version: str = PLANNER_VERSION,
) -> CapabilitySnapshot:
    return CapabilitySnapshot(
        planner_kind=planner_kind,
        planner_version=planner_version,
        max_queries_per_wave=MAX_QUERIES_PER_WAVE,
        max_query_chars=MAX_QUERY_CHARS,
    )


def _make_query(
    ordinal: int,
    text: str = "sample search query",
    *,
    brief_sha: str = "f" * 64,
    wave_index: int = 0,
    purpose: str = "covers the canonical example",
    dimension_ids: tuple[str, ...] = ("coverage",),
    schema_version: int = SCHEMA_VERSION,
) -> PlannedSearchQuery:
    return PlannedSearchQuery(
        query_id=compute_query_id(
            schema_version=schema_version,
            wave_index=wave_index,
            ordinal=ordinal,
            normalized_text=normalize_query_text(text),
            research_brief_sha256=brief_sha,
        ),
        text=text,
        purpose=purpose,
        dimension_ids=dimension_ids,
        ordinal=ordinal,
    )


def _make_plan(
    queries: tuple[PlannedSearchQuery, ...],
    *,
    brief_sha: str = "f" * 64,
    wave_index: int = 0,
    schema_version: int = SCHEMA_VERSION,
    planner_kind: str = PLANNER_KIND,
    planner_version: str = PLANNER_VERSION,
    created_at: str = CREATED_AT,
) -> SearchPlan:
    planning_limits = _planning_limits()
    capability_snapshot = CapabilitySnapshot(
        planner_kind=planner_kind,
        planner_version=planner_version,
        max_queries_per_wave=planning_limits.max_queries_per_wave,
        max_query_chars=planning_limits.max_query_chars,
    )
    return SearchPlan(
        schema_version=schema_version,
        planner_kind=planner_kind,
        planner_version=planner_version,
        research_brief_sha256=brief_sha,
        wave_index=wave_index,
        queries=queries,
        planning_limits=planning_limits,
        capability_snapshot=capability_snapshot,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# (1) one-query wave valid
# ---------------------------------------------------------------------------


def test_one_query_wave_is_valid() -> None:
    q = _make_query(0)
    plan = _make_plan((q,))
    # Does not raise.
    validate_search_plan(plan, expected_research_brief_sha256="f" * 64)


# ---------------------------------------------------------------------------
# (2) four-query wave valid
# ---------------------------------------------------------------------------


def test_four_query_wave_is_valid() -> None:
    qs = tuple(_make_query(i, text=f"query number {i}") for i in range(4))
    plan = _make_plan(qs)
    validate_search_plan(plan, expected_research_brief_sha256="f" * 64)


# ---------------------------------------------------------------------------
# (3) zero-query invalid
# ---------------------------------------------------------------------------


def test_zero_query_wave_is_invalid() -> None:
    plan = _make_plan(())
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "queries_count" in names


# ---------------------------------------------------------------------------
# (4) five-query invalid
# ---------------------------------------------------------------------------


def test_five_query_wave_is_invalid() -> None:
    qs = tuple(_make_query(i, text=f"query number {i}") for i in range(5))
    plan = _make_plan(qs)
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "queries_count" in names


# ---------------------------------------------------------------------------
# (5) 399-char query valid
# ---------------------------------------------------------------------------


def test_query_at_399_chars_is_valid() -> None:
    text = "a" * MAX_QUERY_CHARS  # exactly 399
    assert len(text) == MAX_QUERY_CHARS
    q = _make_query(0, text=text)
    plan = _make_plan((q,))
    validate_search_plan(plan, expected_research_brief_sha256="f" * 64)


# ---------------------------------------------------------------------------
# (6) 400-char query invalid
# ---------------------------------------------------------------------------


def test_query_at_400_chars_is_invalid() -> None:
    text = "a" * (MAX_QUERY_CHARS + 1)
    assert len(text) == 400
    q = _make_query(0, text=text)
    plan = _make_plan((q,))
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "query_text_too_long" in names


# ---------------------------------------------------------------------------
# (7) whitespace-empty invalid
# ---------------------------------------------------------------------------


def test_whitespace_only_query_is_invalid() -> None:
    q = _make_query(0, text="   \t  ")
    plan = _make_plan((q,))
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    # Either ``query_text_whitespace`` or ``query_text_empty`` is
    # acceptable; the spec mentions whitespace-empty, so we accept
    # the more specific check name.
    assert "query_text_whitespace" in names or "query_text_empty" in names


# ---------------------------------------------------------------------------
# (8) duplicate queries invalid
# ---------------------------------------------------------------------------


def test_duplicate_queries_are_invalid() -> None:
    q0 = _make_query(0, text="  same text  ")
    q1 = _make_query(1, text="same text")
    plan = _make_plan((q0, q1))
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "duplicate_query" in names


# ---------------------------------------------------------------------------
# (9) non-contiguous ordinal invalid
# ---------------------------------------------------------------------------


def test_non_contiguous_ordinals_are_invalid() -> None:
    q0 = _make_query(0, text="first")
    # Skip ordinal 1, jump to 2: ordinals are 0, 2 (not contiguous).
    q2 = _make_query(2, text="second")
    plan = _make_plan((q0, q2))
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "query_ordinals" in names


def test_duplicate_ordinals_are_invalid() -> None:
    q0 = _make_query(0, text="first")
    q0_dup = _make_query(0, text="second")
    plan = _make_plan((q0, q0_dup))
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "query_ordinals" in names


# ---------------------------------------------------------------------------
# (10) tampered query ID invalid
# ---------------------------------------------------------------------------


def test_tampered_query_id_is_invalid() -> None:
    q = _make_query(0)
    # Build a copy with a different (non-recomputable) id.
    tampered = PlannedSearchQuery(
        query_id="q-deadbeefdeadbeef",
        text=q.text,
        purpose=q.purpose,
        dimension_ids=q.dimension_ids,
        ordinal=q.ordinal,
    )
    plan = _make_plan((tampered,))
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "query_id_mismatch" in names


# ---------------------------------------------------------------------------
# (11) brief hash mismatch invalid
# ---------------------------------------------------------------------------


def test_brief_hash_mismatch_is_invalid() -> None:
    q = _make_query(0)
    plan = _make_plan((q,), brief_sha="a" * 64)
    # The plan was built with brief hash "aaa...". Calling validate
    # with the expected hash "bbb..." must fail.
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="b" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "research_brief_sha256_mismatch" in names


# ---------------------------------------------------------------------------
# (12) unsupported schema version invalid
# ---------------------------------------------------------------------------


def test_unsupported_schema_version_is_invalid() -> None:
    q = _make_query(0)
    plan = _make_plan((q,), schema_version=SCHEMA_VERSION + 999)
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "schema_version" in names


# ---------------------------------------------------------------------------
# (13) unsupported C1A wave index invalid
# ---------------------------------------------------------------------------


def test_unsupported_wave_index_is_invalid() -> None:
    q = _make_query(0, wave_index=1)
    plan = _make_plan((q,), wave_index=1)
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "wave_index" in names


def test_wave_index_constants_match_c1a_contract() -> None:
    # C1A only permits wave_index == 0. This test documents that
    # contract; a future slice that adds wave_index > 0 must update
    # both the constant and this test.
    assert frozenset({0}) == ALLOWED_WAVE_INDICES


# ---------------------------------------------------------------------------
# (14) long Research Brief >=10k chars accepted by planning domain
# ---------------------------------------------------------------------------


def test_long_research_brief_is_accepted_by_planning_domain() -> None:
    # A 12,000-character brief, well above the public 2,000 char cap.
    long_brief = "lorem ipsum " * 1100  # 12100 chars
    assert len(long_brief) >= 10_000
    brief_sha = compute_research_brief_sha256(long_brief)

    qs = tuple(
        _make_query(
            i,
            text=f"derived query {i}",
            brief_sha=brief_sha,
        )
        for i in range(2)
    )
    plan = _make_plan(qs, brief_sha=brief_sha)

    # Plan validation must succeed; the brief is referenced by hash
    # only and the per-query text cap (399) is enforced.
    validate_search_plan(plan, expected_research_brief_sha256=brief_sha)

    # The derived query texts are bounded to 399 chars and have
    # nothing to do with the brief's length.
    for q in plan.queries:
        assert len(q.text) <= MAX_QUERY_CHARS


# ---------------------------------------------------------------------------
# (15) declared dimension not covered invalid
# ---------------------------------------------------------------------------


def test_c1a_has_no_plan_level_dimension_registry() -> None:
    """C1A intentionally has NO plan-level dimension registry.

    The mission rule "every declared plan dimension is referenced by
    at least one query" only applies WHEN a plan-level dimension
    registry is used. C1A keeps dimensions per-query only; this
    test documents the choice. A future slice that adds a plan-level
    registry must extend the validator and add a corresponding test.
    """
    # Sanity: build a plan with no plan-level dimension registry and
    # confirm it is valid. The validator must not reference any
    # plan-level dimension field (it does not, by design).
    q0 = _make_query(0, text="alpha", dimension_ids=("alpha", "beta"))
    q1 = _make_query(1, text="beta", dimension_ids=("gamma",))
    plan = _make_plan((q0, q1))
    validate_search_plan(plan, expected_research_brief_sha256="f" * 64)


def test_empty_dimensions_per_query_is_invalid() -> None:
    # Even without a plan-level registry, per-query dimensions must
    # be non-empty (C1A contract: every query is decomposed).
    q = _make_query(0, dimension_ids=())
    plan = _make_plan((q,))
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(plan, expected_research_brief_sha256="f" * 64)
    names = {name for name, _ in excinfo.value.violations}
    assert "query_dimensions_empty" in names


# ---------------------------------------------------------------------------
# (16) deterministic serialization / ID stability
# ---------------------------------------------------------------------------


def test_serialization_roundtrip_is_byte_stable() -> None:
    q0 = _make_query(0, text="alpha")
    q1 = _make_query(1, text="beta")
    plan = _make_plan((q0, q1))

    blob1 = serialize_search_plan(plan)
    blob2 = serialize_search_plan(plan)
    assert blob1 == blob2

    reloaded = deserialize_search_plan(blob1)
    assert serialize_search_plan(reloaded) == blob1


def test_query_id_is_stable_across_fresh_constructions() -> None:
    # Two fresh PlannedSearchQuery instances with the same
    # deterministic inputs must produce the same id.
    args = dict(
        schema_version=SCHEMA_VERSION,
        wave_index=0,
        ordinal=2,
        normalized_text="hello world",
        research_brief_sha256="a" * 64,
    )
    id_a = compute_query_id(**args)
    id_b = compute_query_id(**args)
    assert id_a == id_b
    assert id_a.startswith("q-")
    # 16 hex chars after the "q-" prefix.
    assert len(id_a) == 2 + 16


def test_query_id_changes_when_text_changes() -> None:
    base = dict(
        schema_version=SCHEMA_VERSION,
        wave_index=0,
        ordinal=0,
        research_brief_sha256="a" * 64,
    )
    id_a = compute_query_id(normalized_text="alpha", **base)
    id_b = compute_query_id(normalized_text="beta", **base)
    assert id_a != id_b


def test_query_id_changes_when_ordinal_changes() -> None:
    base = dict(
        schema_version=SCHEMA_VERSION,
        wave_index=0,
        normalized_text="alpha",
        research_brief_sha256="a" * 64,
    )
    id_0 = compute_query_id(ordinal=0, **base)
    id_1 = compute_query_id(ordinal=1, **base)
    assert id_0 != id_1


def test_query_id_changes_when_brief_hash_changes() -> None:
    base = dict(
        schema_version=SCHEMA_VERSION,
        wave_index=0,
        ordinal=0,
        normalized_text="alpha",
    )
    id_a = compute_query_id(research_brief_sha256="a" * 64, **base)
    id_b = compute_query_id(research_brief_sha256="b" * 64, **base)
    assert id_a != id_b


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


def test_planning_limits_cannot_encode_a_whole_job_max() -> None:
    """The C1A contract has NO whole-job ``max_total_queries`` field.

    The only limits are per-wave. This test asserts that the
    ``PlanningLimits`` type does not expose any whole-job field by
    enumerating its fields.
    """
    field_names = set(PlanningLimits.__dataclass_fields__)  # type: ignore[attr-defined]
    assert "max_queries_per_wave" in field_names
    assert "max_query_chars" in field_names
    assert "max_total_queries" not in field_names
    assert "max_job_queries" not in field_names


def test_search_plan_has_no_nested_plan_field() -> None:
    """C1A forbids nested plan / wave structure (frozen dataclass)."""
    field_names = set(SearchPlan.__dataclass_fields__)  # type: ignore[attr-defined]
    assert "parent_plan" not in field_names
    assert "nested_plans" not in field_names
    assert "sub_waves" not in field_names


def test_planner_kind_constant_is_closed_for_c1a() -> None:
    # C1A ships the deterministic stub only. A future slice that
    # adds LLM-backed planner kinds must update both the constant
    # and this test.
    assert frozenset({"c1a-deterministic-stub"}) == KNOWN_PLANNER_KINDS


def test_planning_limits_rejects_out_of_band_values() -> None:
    # max_queries_per_wave > 4 must be rejected at construction.
    with pytest.raises(ValueError):
        PlanningLimits(max_queries_per_wave=5, max_query_chars=399)
    # max_queries_per_wave < 1 must be rejected.
    with pytest.raises(ValueError):
        PlanningLimits(max_queries_per_wave=0, max_query_chars=399)
    # max_query_chars < 1 must be rejected.
    with pytest.raises(ValueError):
        PlanningLimits(max_queries_per_wave=4, max_query_chars=0)


def test_capability_snapshot_rejects_unknown_planner_kind() -> None:
    with pytest.raises(ValueError):
        CapabilitySnapshot(
            planner_kind="gpt-future-planner",
            planner_version="0.1.0",
            max_queries_per_wave=4,
            max_query_chars=399,
        )


def test_capability_snapshot_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        CapabilitySnapshot(
            planner_kind="c1a-deterministic-stub",
            planner_version="not-a-version",
            max_queries_per_wave=4,
            max_query_chars=399,
        )


def test_observation_post_init_validates_wave_index() -> None:
    obs = SearchObservation(wave_index=0, query_id="q-abc")
    assert obs.wave_index == 0
    with pytest.raises(ValueError):
        SearchObservation(wave_index=5, query_id="q-abc")


def test_observation_serialization_roundtrip() -> None:
    obs = SearchObservation(
        wave_index=0,
        query_id="q-abc",
        backend="tavily",
        result_refs=("https://a", "https://b"),
        structured_error=None,
        attempt_count=1,
        duration_ms=250,
        local_usage={"tokens": 100, "model": "tavily-fast"},
    )
    payload = obs.to_dict()
    # local_usage must be a list of (key, value) pairs, sorted.
    assert isinstance(payload["local_usage"], list)
    assert payload["local_usage"] == sorted(payload["local_usage"])
    reloaded = SearchObservation.from_dict(payload)
    assert reloaded == obs


# ---------------------------------------------------------------------------
# build_search_plan helper
# ---------------------------------------------------------------------------


def test_build_search_plan_uses_c1a_defaults() -> None:
    brief = "short canonical brief"
    brief_sha = compute_research_brief_sha256(brief)
    qs = tuple(
        _make_query(i, text=f"derived {i}", brief_sha=brief_sha)
        for i in range(2)
    )
    plan = build_search_plan(
        planner_kind=PLANNER_KIND,
        planner_version=PLANNER_VERSION,
        research_brief_sha256=brief_sha,
        wave_index=0,
        queries=qs,
        created_at=CREATED_AT,
    )
    assert plan.planning_limits.max_queries_per_wave == MAX_QUERIES_PER_WAVE
    assert plan.planning_limits.max_query_chars == MAX_QUERY_CHARS
    assert plan.capability_snapshot.planner_kind == PLANNER_KIND
    assert plan.capability_snapshot.planner_version == PLANNER_VERSION
    # No whole-job "max_total_queries" field at all.
    assert not hasattr(plan, "max_total_queries")


def test_build_search_plan_validates_immediately() -> None:
    # An obviously bad plan (zero queries) must raise at build time.
    with pytest.raises(PlanningValidationError):
        build_search_plan(
            planner_kind=PLANNER_KIND,
            planner_version=PLANNER_VERSION,
            research_brief_sha256="f" * 64,
            wave_index=0,
            queries=(),
            created_at=CREATED_AT,
        )


def test_validator_uses_canonical_json_for_id_stability() -> None:
    # The validator's recomputed id must be identical to what
    # ``compute_query_id`` returns for the same inputs. This is a
    # meta-test that locks the canonical-JSON contract.
    brief_sha = "1" * 64
    q = _make_query(0, text="hello", brief_sha=brief_sha)
    plan = _make_plan((q,), brief_sha=brief_sha)
    validate_search_plan(plan, expected_research_brief_sha256=brief_sha)
    # Now tamper with the id and re-validate; the recomputed id
    # must differ from the tampered one.
    tampered = replace_for_test(q, query_id="q-aaaaaaaaaaaaaaaa")
    tampered_plan = _make_plan((tampered,), brief_sha=brief_sha)
    with pytest.raises(PlanningValidationError) as excinfo:
        validate_search_plan(
            tampered_plan, expected_research_brief_sha256=brief_sha
        )
    detail = next(d for n, d in excinfo.value.violations if n == "query_id_mismatch")
    assert compute_query_id(
        schema_version=SCHEMA_VERSION,
        wave_index=0,
        ordinal=0,
        normalized_text=normalize_query_text("hello"),
        research_brief_sha256=brief_sha,
    ) in detail


# ---------------------------------------------------------------------------
# Helpers (local)
# ---------------------------------------------------------------------------


def replace_for_test(
    query: PlannedSearchQuery, **overrides: object
) -> PlannedSearchQuery:
    """Dataclass ``replace``-style helper for a frozen PlannedSearchQuery.

    We do not import ``dataclasses.replace`` at the top of the file
    to keep the imports section focused on the production API.
    """
    import dataclasses

    return dataclasses.replace(query, **overrides)


def _has_only_expected_keys(payload: Mapping[str, object], expected: set[str]) -> bool:
    return set(payload) == expected


# ---------------------------------------------------------------------------
# Deterministic payload shape
# ---------------------------------------------------------------------------


def test_serialization_payload_has_no_extra_fields() -> None:
    q0 = _make_query(0, text="alpha")
    q1 = _make_query(1, text="beta")
    plan = _make_plan((q0, q1))
    blob = serialize_search_plan(plan)
    payload = json.loads(blob)
    assert _has_only_expected_keys(
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
    )
