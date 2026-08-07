"""Tests for the PRE2-C1A plan store (hermes.jobs.plan_store).

These tests cover the public surface of the local plan store:

- Atomic write / load round-trip.
- Idempotent rewrite of byte-equivalent plans.
- Conflicting overwrite rejection.
- Fail-closed corruption / version / brief-hash / capability-snapshot
  mismatches.
- Unsafe job-id / path handling.

The store is constructed against a per-test ``tmp_path`` so no
test ever touches the real Deep Research data root.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hermes.jobs.plan_store import (
    ENVELOPE_VERSION,
    PLAN_FILE_SUFFIX,
    PLANS_SUBDIR,
    LocalPlanStore,
    PlanBriefHashMismatchError,
    PlanCapabilitySnapshotMismatchError,
    PlanConflictError,
    PlanCorruptError,
    PlanInvalidJobIdError,
    PlanNotFoundError,
    PlanSchemaMismatchError,
)
from hermes.jobs.planning import (
    MAX_QUERIES_PER_WAVE,
    MAX_QUERY_CHARS,
    SCHEMA_VERSION,
    CapabilitySnapshot,
    PlannedSearchQuery,
    PlanningLimits,
    SearchPlan,
    build_search_plan,
    compute_query_id,
    compute_research_brief_sha256,
    normalize_query_text,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


PLANNER_KIND = "c1a-deterministic-stub"
PLANNER_VERSION = "0.1.0"
CREATED_AT = "2026-08-07T12:00:00Z"
VALID_JOB_ID = "0123456789ab"


@pytest.fixture
def store(tmp_path: Path) -> LocalPlanStore:
    """A LocalPlanStore rooted in a per-test tmp directory."""
    return LocalPlanStore(tmp_path)


@pytest.fixture
def brief_sha() -> str:
    return compute_research_brief_sha256("canonical research brief for tests")


def _capability_snapshot() -> CapabilitySnapshot:
    return CapabilitySnapshot(
        planner_kind=PLANNER_KIND,
        planner_version=PLANNER_VERSION,
        max_queries_per_wave=MAX_QUERIES_PER_WAVE,
        max_query_chars=MAX_QUERY_CHARS,
    )


def _make_plan(
    brief_sha: str,
    *,
    text: str = "alpha query",
    planner_kind: str = PLANNER_KIND,
    planner_version: str = PLANNER_VERSION,
    schema_version: int = SCHEMA_VERSION,
    wave_index: int = 0,
) -> SearchPlan:
    q0 = PlannedSearchQuery(
        query_id=compute_query_id(
            schema_version=schema_version,
            wave_index=wave_index,
            ordinal=0,
            normalized_text=normalize_query_text(text),
            research_brief_sha256=brief_sha,
        ),
        text=text,
        purpose="canonical example purpose",
        dimension_ids=("coverage",),
        ordinal=0,
    )
    return build_search_plan(
        planner_kind=planner_kind,
        planner_version=planner_version,
        research_brief_sha256=brief_sha,
        wave_index=wave_index,
        queries=(q0,),
        created_at=CREATED_AT,
        schema_version=schema_version,
    )


# ---------------------------------------------------------------------------
# (17) atomic round-trip
# ---------------------------------------------------------------------------


def test_atomic_round_trip_preserves_plan(store: LocalPlanStore, brief_sha: str) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)

    reloaded = store.load(
        VALID_JOB_ID,
        expected_research_brief_sha256=brief_sha,
        expected_capability_snapshot=_capability_snapshot(),
    )
    assert reloaded == plan
    # Serialization must be byte-stable after a round trip.
    assert store.exists(VALID_JOB_ID)


def test_write_creates_plans_subdir(store: LocalPlanStore, brief_sha: str) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)
    expected = store.plans_root / f"{VALID_JOB_ID}{PLAN_FILE_SUFFIX}"
    assert expected.is_file()
    # And the parent Deep Research data root must NOT have a
    # stray report-style file at the top level (we keep plans
    # confined to research_plans/).
    assert not (store.root / f"{VALID_JOB_ID}.md").exists()


def test_write_uses_atomic_replace_no_tmp_left_behind(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)
    # No leftover ``.tmp.json`` files in the plans directory.
    leftover = list(store.plans_root.glob(f".{VALID_JOB_ID}.plan.*.tmp.json"))
    assert leftover == []


# ---------------------------------------------------------------------------
# (18) equivalent rewrite idempotent
# ---------------------------------------------------------------------------


def test_equivalent_rewrite_is_idempotent(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)
    first_path = store.derive_path(VALID_JOB_ID)
    first_bytes = first_path.read_bytes()
    first_mtime = first_path.stat().st_mtime_ns

    # Re-write the same plan. The store must NOT raise and must
    # not change the file content. We do not assert mtime
    # stability (mtime is filesystem-dependent on Windows when no
    # actual write happens); we assert the bytes are unchanged.
    store.write(VALID_JOB_ID, plan)
    second_bytes = first_path.read_bytes()
    assert first_bytes == second_bytes
    # Sanity: the mtime did not change because no write happened.
    assert first_path.stat().st_mtime_ns == first_mtime


# ---------------------------------------------------------------------------
# (19) conflicting overwrite rejected
# ---------------------------------------------------------------------------


def test_conflicting_overwrite_is_rejected(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan_a = _make_plan(brief_sha, text="alpha query")
    plan_b = _make_plan(brief_sha, text="beta query")
    store.write(VALID_JOB_ID, plan_a)

    # Writing a different plan with the same job_id must raise
    # ``PlanConflictError`` (no silent replacement of an in-use plan).
    with pytest.raises(PlanConflictError):
        store.write(VALID_JOB_ID, plan_b)

    # The on-disk plan must still be plan_a.
    reloaded = store.load(
        VALID_JOB_ID,
        expected_research_brief_sha256=brief_sha,
        expected_capability_snapshot=_capability_snapshot(),
    )
    assert reloaded == plan_a


def test_conflicting_overwrite_with_force_overwrites(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan_a = _make_plan(brief_sha, text="alpha query")
    plan_b = _make_plan(brief_sha, text="beta query")
    store.write(VALID_JOB_ID, plan_a)
    store.write(VALID_JOB_ID, plan_b, force=True)
    reloaded = store.load(
        VALID_JOB_ID,
        expected_research_brief_sha256=brief_sha,
        expected_capability_snapshot=_capability_snapshot(),
    )
    assert reloaded == plan_b


# ---------------------------------------------------------------------------
# (20) corrupt JSON fail closed
# ---------------------------------------------------------------------------


def test_corrupt_json_fails_closed(store: LocalPlanStore, brief_sha: str) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)
    target = store.derive_path(VALID_JOB_ID)
    # Overwrite the plan file with garbage that is not valid JSON.
    target.write_bytes(b"\x00\x01\x02 not json \xff\xfe")
    with pytest.raises(PlanCorruptError):
        store.load(VALID_JOB_ID, expected_research_brief_sha256=brief_sha)


def test_non_utf8_bytes_fail_closed(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)
    target = store.derive_path(VALID_JOB_ID)
    # Bytes that are not valid UTF-8 (lone continuation byte).
    target.write_bytes(b"\x80\x81\x82\x83")
    with pytest.raises(PlanCorruptError):
        store.load(VALID_JOB_ID, expected_research_brief_sha256=brief_sha)


# ---------------------------------------------------------------------------
# (21) version mismatch fail closed
# ---------------------------------------------------------------------------


def test_unsupported_envelope_version_fails_closed(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)
    target = store.derive_path(VALID_JOB_ID)
    payload = json.loads(target.read_bytes())
    payload["envelope_version"] = ENVELOPE_VERSION + 999
    target.write_bytes(json.dumps(payload).encode("utf-8"))

    with pytest.raises(PlanSchemaMismatchError) as excinfo:
        store.load(VALID_JOB_ID, expected_research_brief_sha256=brief_sha)
    assert excinfo.value.found == ENVELOPE_VERSION + 999


def test_unsupported_inner_schema_version_fails_closed(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)
    target = store.derive_path(VALID_JOB_ID)
    payload = json.loads(target.read_bytes())
    payload["schema_version"] = SCHEMA_VERSION + 999
    target.write_bytes(json.dumps(payload).encode("utf-8"))

    with pytest.raises(PlanSchemaMismatchError):
        store.load(VALID_JOB_ID, expected_research_brief_sha256=brief_sha)


# ---------------------------------------------------------------------------
# (22) brief-hash mismatch fail closed
# ---------------------------------------------------------------------------


def test_brief_hash_mismatch_fails_closed(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)

    with pytest.raises(PlanBriefHashMismatchError):
        store.load(
            VALID_JOB_ID,
            expected_research_brief_sha256="0" * 64,
            expected_capability_snapshot=_capability_snapshot(),
        )


def test_no_expected_brief_hash_skips_binding_check(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)

    # No expected brief hash: load returns the plan as long as the
    # envelope and structure pass.
    reloaded = store.load(VALID_JOB_ID)
    assert reloaded == plan


# ---------------------------------------------------------------------------
# (23) capability / limit snapshot mismatch fail closed
# ---------------------------------------------------------------------------


def test_capability_snapshot_mismatch_fails_closed(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)

    other_snapshot = CapabilitySnapshot(
        planner_kind=PLANNER_KIND,
        planner_version="9.9.9-future",  # different version
        max_queries_per_wave=MAX_QUERIES_PER_WAVE,
        max_query_chars=MAX_QUERY_CHARS,
    )
    with pytest.raises(PlanCapabilitySnapshotMismatchError):
        store.load(
            VALID_JOB_ID,
            expected_research_brief_sha256=brief_sha,
            expected_capability_snapshot=other_snapshot,
        )


def test_no_expected_capability_snapshot_skips_binding_check(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)

    reloaded = store.load(VALID_JOB_ID, expected_research_brief_sha256=brief_sha)
    assert reloaded == plan


# ---------------------------------------------------------------------------
# (24) unsafe job-id / path handling rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_job_id",
    [
        "",
        " ",
        "..",
        "../escape",
        "0123456789ab/../escape",
        "0123456789ab\0",
        "UPPER12345678",  # uppercase not allowed
        "12345",          # too short
        "x" * 12,         # not hex
        "0" * 13,         # too long
        "0123456789ab/cd",  # path separator
    ],
)
def test_unsafe_job_id_is_rejected(
    store: LocalPlanStore, bad_job_id: str
) -> None:
    with pytest.raises(PlanInvalidJobIdError):
        store.derive_path(bad_job_id)
    with pytest.raises(PlanInvalidJobIdError):
        store.write(bad_job_id, _make_plan("f" * 64))
    # ``exists`` returns False instead of raising for an invalid id.
    assert store.exists(bad_job_id) is False


def test_load_missing_file_raises_not_found(
    store: LocalPlanStore, brief_sha: str
) -> None:
    with pytest.raises(PlanNotFoundError):
        store.load(VALID_JOB_ID, expected_research_brief_sha256=brief_sha)


def test_store_creates_root_even_if_parent_missing(tmp_path: Path) -> None:
    nested = tmp_path / "fresh" / "data"
    assert not nested.exists()
    store = LocalPlanStore(nested)
    assert store.plans_root.is_dir()
    plan = _make_plan("f" * 64)
    store.write(VALID_JOB_ID, plan)
    assert store.exists(VALID_JOB_ID)


def test_derive_path_is_inside_root(store: LocalPlanStore) -> None:
    p = store.derive_path(VALID_JOB_ID)
    # The path must be inside the plans_root.
    assert p.is_relative_to(store.plans_root)
    assert p.name == f"{VALID_JOB_ID}{PLAN_FILE_SUFFIX}"
    # And the plans_root must itself be inside the configured root.
    assert store.plans_root.is_relative_to(store.root)
    assert store.plans_root.name == PLANS_SUBDIR


# ---------------------------------------------------------------------------
# Epistemic separation: structural validation != semantic completeness
# ---------------------------------------------------------------------------


def test_structural_validation_does_not_claim_semantic_completeness(
    store: LocalPlanStore, brief_sha: str
) -> None:
    # A plan that passes all C1A structural checks is NOT
    # guaranteed to be a good Deep Research plan. This test
    # documents the epistemic rule: structural validity is a
    # necessary, not sufficient, condition.
    plan = _make_plan(brief_sha, text="a" * 10)  # tiny, content-free query
    store.write(VALID_JOB_ID, plan)
    reloaded = store.load(
        VALID_JOB_ID,
        expected_research_brief_sha256=brief_sha,
        expected_capability_snapshot=_capability_snapshot(),
    )
    # The plan loads cleanly: structural validation passed.
    assert reloaded == plan
    # But the query text is content-free; the validator is
    # correctly silent about semantic quality. This test does
    # not assert anything about ``quality``; it only asserts that
    # the structural contract is satisfied.
    assert reloaded.queries[0].text == "a" * 10


# ---------------------------------------------------------------------------
# No DB / public API surface changes
# ---------------------------------------------------------------------------


def test_store_does_not_import_orm_or_db_modules() -> None:
    # The store is filesystem-only. Catching accidental imports
    # of the SQLAlchemy / DB / HTTP / LLM-router modules is a
    # cheap regression guard.
    import hermes.jobs.plan_store as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = [
        "from hermes.memory",
        "import hermes.memory",
        "from sqlalchemy",
        "import sqlalchemy",
        "from hermes.receivers",
        "import hermes.receivers",
        "from hermes.services.search.router",
        "from hermes.llm",
        "import hermes.llm",
    ]
    for needle in forbidden:
        assert needle not in src, (
            f"plan_store must not import {needle!r} (C1A is filesystem-only)"
        )


def test_store_root_does_not_silently_share_state_with_report_store(
    tmp_path: Path, brief_sha: str
) -> None:
    # Constructing a plan store on a directory that ALSO hosts
    # report files must not affect the plan store's own files
    # (and vice-versa). The plan store keeps its subdirectory
    # isolated.
    root = tmp_path / "shared_data_root"
    root.mkdir()
    # Place a fake report file at the root.
    (root / "0123456789ab.md").write_text("fake report", encoding="utf-8")

    store = LocalPlanStore(root)
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)
    assert store.exists(VALID_JOB_ID)
    # The report file at the root must remain untouched.
    assert (root / "0123456789ab.md").read_text(encoding="utf-8") == "fake report"
    # And the plan file lives under research_plans/, not at the root.
    assert (root / f"{VALID_JOB_ID}{PLAN_FILE_SUFFIX}").exists() is False
    assert (root / PLANS_SUBDIR / f"{VALID_JOB_ID}{PLAN_FILE_SUFFIX}").is_file()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_serialization_is_deterministic_across_writes(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    store.write(VALID_JOB_ID, plan)
    first = store.derive_path(VALID_JOB_ID).read_bytes()
    # Wipe and rewrite to ensure a fresh serialization produces the
    # same bytes (the writer's content does not depend on
    # incidental state like mtime or filesystem).
    store.derive_path(VALID_JOB_ID).unlink()
    store.write(VALID_JOB_ID, plan)
    second = store.derive_path(VALID_JOB_ID).read_bytes()
    assert first == second


# ---------------------------------------------------------------------------
# Dataclass-level invariants (imported here to keep the imports focused)
# ---------------------------------------------------------------------------


def test_planning_limits_invariants_via_dataclasses(
    store: LocalPlanStore, brief_sha: str
) -> None:
    plan = _make_plan(brief_sha)
    # ``PlanningLimits`` only exposes per-wave fields.
    limits_dict = plan.planning_limits.to_dict()
    assert set(limits_dict) == {"max_queries_per_wave", "max_query_chars"}
    # And the dataclass must NOT carry a whole-job field even via
    # ``dataclasses.fields``.
    field_names = {
        f.name for f in dataclasses.fields(PlanningLimits)
    }
    assert "max_total_queries" not in field_names
    assert "max_job_queries" not in field_names


def test_capability_snapshot_field_shape() -> None:
    # The C1A capability snapshot is a narrow, closed shape.
    snap = _capability_snapshot()
    assert set(snap.to_dict()) == {
        "planner_kind",
        "planner_version",
        "max_queries_per_wave",
        "max_query_chars",
    }
