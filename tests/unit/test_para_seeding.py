"""Tests para Sprint 19 Slice 6 — PARA seeding.

TDD-VAULT-COLLECTIONS §5 (lines 1355-1379). 4 default collections
(PARA) se crean en el primer startup de Hermes, idempotente.

TDD spec tests (lines 2002-2004) + their actual implementations here:
  - test_seed_para_creates_four_collections
  - test_seed_para_idempotent_no_duplicates
  - test_seed_para_skips_user_deleted_collection
    → see test_seed_para_recreates_admin_hard_deleted_collection +
      test_seed_para_does_not_recreate_archived_collection
    (TDD name preserved semantically: "user deleted" is archive in
    the production API; admin SQL DELETE is the hard-delete path
    that the seed cannot distinguish from "never seeded" and so
    re-creates on next startup)

Plus 1 integration test for the __main__ wiring (TDD §5 says
"in `hermes/__main__.py` after DB open + before scheduler start").
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hermes.memory.collections import VaultCollectionsRepo
from hermes.memory.db import Database
from hermes.memory.seed import (
    PARA_DEFAULT_COLLECTIONS,
    migrate_legacy_para_names,
    seed_para_collections,
)

# Async tests use @pytest.mark.asyncio per test (not a global mark)
# to avoid the warning on the sync test below.


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Initialized empty Database at tmp_path (applies all migrations)."""
    database = Database(tmp_path / "test_para_seeding.db")
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
def collections_repo(db: Database) -> VaultCollectionsRepo:
    return VaultCollectionsRepo(db)


# ---------------------------------------------------------------------------
# TDD §5 contract: PARA_DEFAULT_COLLECTIONS is a list of (name, desc, sort_order)
# ---------------------------------------------------------------------------


def test_para_default_collections_is_list_of_4_tuples() -> None:
    """The 4 PARA defaults are 01_Proyectos, 02_Areas, 03_Recursos, 04_Archivo.

    Per TDD §5, the 4 collection names are hardcoded in Spanish.
    Sprint 19 retro (2026-07-11): names are ASCII-only to match
    filesystem directory conventions and avoid the dual-naming bug
    with M6 Phase 1 (which creates collections from filesystem
    dirs without accents).
    """
    assert len(PARA_DEFAULT_COLLECTIONS) == 4
    names = [c[0] for c in PARA_DEFAULT_COLLECTIONS]
    assert names == [
        "01_Proyectos_Activos",
        "02_Areas_de_Responsabilidad",
        "03_Recursos_y_Conocimiento",
        "04_Archivo",
    ]
    # ASCII-only: ensures the names can match filesystem directory
    # conventions and don't trigger the dual-naming bug.
    for entry in PARA_DEFAULT_COLLECTIONS:
        assert entry[0].isascii(), f"Collection name {entry[0]!r} contains non-ASCII chars"
    # Each tuple has 3 elements: (name, description, sort_order)
    for entry in PARA_DEFAULT_COLLECTIONS:
        assert len(entry) == 3, f"Expected 3-tuple, got {len(entry)}: {entry}"
        name, desc, sort_order = entry
        assert isinstance(name, str) and name
        assert isinstance(desc, str) and desc
        assert isinstance(sort_order, int) and 0 <= sort_order < 100


# ---------------------------------------------------------------------------
# TDD §5: test_seed_para_creates_four_collections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_para_creates_four_collections(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """First startup: seed_para_collections (with PARA_DEFAULT_COLLECTIONS
    passed explicitly) creates the 4 PARA defaults.

    Per TDD §5: "Crea las 4 collections PARA default si no existen."
    Per Sprint 19 followup: seed_para_collections(defaults=None) is
    now a true opt-in no-op. Tests that want the legacy 4 PARA
    behavior must pass PARA_DEFAULT_COLLECTIONS explicitly.
    """
    # Sanity: no collections yet
    assert await collections_repo.list_collections() == []

    seeded = await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)

    # All 4 created
    assert len(seeded) == 4
    assert set(seeded) == {
        "01_Proyectos_Activos",
        "02_Areas_de_Responsabilidad",
        "03_Recursos_y_Conocimiento",
        "04_Archivo",
    }

    # All 4 are in the DB
    all_colls = await collections_repo.list_collections(include_archived=True)
    assert len(all_colls) == 4
    db_names = {c.name for c in all_colls}
    assert db_names == {
        "01_Proyectos_Activos",
        "02_Areas_de_Responsabilidad",
        "03_Recursos_y_Conocimiento",
        "04_Archivo",
    }


@pytest.mark.asyncio
async def test_seed_para_assigns_descriptions_and_sort_order(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """Seeded collections carry the spec'd descriptions and sort_order.

    Per TDD §5, each default has a description (for the agent to
    surface to the user) and a sort_order (for stable display order
    in the /v1/collections API).
    """
    await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)

    for name, expected_desc, expected_sort in PARA_DEFAULT_COLLECTIONS:
        c = await collections_repo.get_collection_by_name(name)
        assert c is not None, f"Collection {name} not created"
        assert (
            c.description == expected_desc
        ), f"Description mismatch for {name}: got {c.description!r}, expected {expected_desc!r}"
        assert (
            c.sort_order == expected_sort
        ), f"sort_order mismatch for {name}: got {c.sort_order}, expected {expected_sort}"


# ---------------------------------------------------------------------------
# TDD §5: test_seed_para_idempotent_no_duplicates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_para_idempotent_no_duplicates(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """Second startup: seed_para_collections is a no-op (idempotent).

    Per TDD §5: "Idempotente: si ya existen (startup subsecuente), skip
    con log. Nunca duplica."
    """
    # First seed
    first = await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)
    assert len(first) == 4
    first_count = len(await collections_repo.list_collections(include_archived=True))
    assert first_count == 4

    # Second seed (simulates Hermes restart)
    second = await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)
    assert second == [], f"Second seed should be no-op, got: {second}"

    # No duplicates
    second_count = len(await collections_repo.list_collections(include_archived=True))
    assert second_count == 4

    # Original sort_order and IDs preserved (not recreated)
    all_colls = await collections_repo.list_collections(include_archived=True)
    for c in all_colls:
        # Find the matching default
        matching = [d for d in PARA_DEFAULT_COLLECTIONS if d[0] == c.name]
        assert len(matching) == 1
        assert c.sort_order == matching[0][2]


@pytest.mark.asyncio
async def test_seed_para_idempotent_under_repeated_calls(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """Repeated seeding (3+ times) never duplicates, even if user runs
    a manual 'seed again' command or the system restarts many times.
    """
    for i in range(5):
        seeded = await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)
        # First call creates 4, rest are no-ops
        if i == 0:
            assert len(seeded) == 4
        else:
            assert seeded == []
        # DB always has exactly 4
        count = len(await collections_repo.list_collections(include_archived=True))
        assert count == 4, f"Iteration {i}: expected 4, got {count}"


# ---------------------------------------------------------------------------
# TDD §5: test_seed_para_skips_user_deleted_collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_para_recreates_admin_hard_deleted_collection(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """Admin/operator hard-deletes a PARA collection via direct SQL —
    seed re-creates it on next startup (cannot distinguish from
    "never seeded").

    Rationale: `VaultCollectionsRepo` has no hard-delete method (only
    archive + restore). The only way to hard-delete is direct SQL,
    which is the admin/operator disaster-recovery path. From the
    seed's perspective, a hard-deleted row is identical to a never-
    seeded row — both are missing. The seed re-creates the missing
    one, which matches the user's intent of "I want a clean slate
    with all 4 PARA defaults again".

    For the user-facing "I deleted this" path, see
    `test_seed_para_does_not_recreate_archived_collection` (the
    archive path is the documented way for users to "remove" a PARA
    collection without losing the data).
    """
    # 1st seed: 4 created
    await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)
    assert len(await collections_repo.list_collections(include_archived=True)) == 4

    # Admin/operator HARD-DELETES 01_Proyectos_Activos via direct SQL
    target = await collections_repo.get_collection_by_name("01_Proyectos_Activos")
    assert target is not None
    await db.conn.execute(
        "DELETE FROM vault_collections WHERE collection_id = ?",
        (target.collection_id,),
    )
    await db.conn.commit()

    # Sanity: 3 left
    assert len(await collections_repo.list_collections(include_archived=True)) == 3

    # 2nd seed: DOES recreate the missing one (admin disaster recovery)
    seeded = await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)
    assert seeded == [
        "01_Proyectos_Activos"
    ], f"Expected re-creation of hard-deleted collection, got: {seeded}"

    # 4 again
    final = await collections_repo.list_collections(include_archived=True)
    assert len(final) == 4
    assert {c.name for c in final} == {
        "01_Proyectos_Activos",
        "02_Areas_de_Responsabilidad",
        "03_Recursos_y_Conocimiento",
        "04_Archivo",
    }


@pytest.mark.asyncio
async def test_seed_para_does_not_recreate_archived_collection(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """User can archive a PARA collection; seed leaves it archived.

    Same as delete but using the soft-delete (archive) path. This
    is the documented user workflow: "I don't use 04_Archivo, let me
    archive it" — and Hermes shouldn't keep recreating it as active.
    """
    # 1st seed: 4 created
    await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)

    # User archives 04_Archivo
    target = await collections_repo.get_collection_by_name("04_Archivo")
    assert target is not None
    await collections_repo.archive_collection(target.collection_id)

    # Sanity: 3 active + 1 archived = 4 total
    all_5 = await collections_repo.list_collections(include_archived=True)
    assert len(all_5) == 4
    assert len(await collections_repo.list_collections(include_archived=False)) == 3

    # 2nd seed: should NOT unarchive 04_Archivo
    seeded = await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)
    assert seeded == []

    # 04_Archivo stays archived
    archived_target = await collections_repo.get_collection_by_name("04_Archivo")
    assert archived_target is not None
    assert (
        archived_target.archived == 1
    ), f"04_Archivo was unarchived by seed, got archived={archived_target.archived}"


# ---------------------------------------------------------------------------
# Integration with __main__.py (TDD §5: "after DB open + before scheduler start")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_para_is_referenced_in_main_run() -> None:
    """`hermes/__main__.py:run()` must call seed_para_collections after
    DB open + before scheduler start (per TDD §5).

    This is a static analysis test (reads the source) — it doesn't
    actually run the full main loop. The full main loop is tested in
    test_main.py with mocked components.
    """
    import hermes.__main__ as main_mod

    source = inspect.getsource(main_mod.run)

    # Sanity: seed_para_collections is imported in __main__.py
    assert "seed_para_collections" in source, (
        "hermes/__main__.py:run() must call seed_para_collections "
        "per TDD §5 (after DB open + before scheduler start)"
    )

    # seed call comes AFTER db.initialize and BEFORE schedulers.
    # Sprint 19 Slice 4d v2 (commit 3): main.run() calls seed_all() (the
    # orchestrator) which internally calls seed_para_collections() +
    # seed_inbox_collections(). The test checks for seed_all().
    db_init_pos = source.find("await db.initialize")
    seed_pos = source.find("await seed_all(")
    if seed_pos < 0:
        # Backward compat: also accept the old direct call
        seed_pos = source.find("await seed_para_collections(")
    # Find the first Scheduler( instantiation after the seed
    scheduler_pos = source.find("Scheduler(", seed_pos)

    assert db_init_pos > 0, "db.initialize() not found in main.run()"
    assert seed_pos > 0, "seed_all() not found in main.run()"
    assert (
        db_init_pos < seed_pos
    ), f"seed_all must run AFTER db.initialize() (db_init at {db_init_pos}, seed at {seed_pos})"
    assert scheduler_pos == -1 or seed_pos < scheduler_pos, (
        f"seed_all must run BEFORE any Scheduler( "
        f"instantiation (seed at {seed_pos}, first scheduler at {scheduler_pos})"
    )


# ---------------------------------------------------------------------------
# Sprint 19 retro (2026-07-11): migrate_legacy_para_names()
#
# One-shot migration for installs that ran the OLD seed (with accented
# names). The migration renames accented PARA collections to the new
# ASCII names. Idempotent: re-running is a no-op.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migrate_legacy_para_names_no_op_on_fresh_db(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """Fresh install: no legacy names exist, migration is a no-op."""
    renamed = await migrate_legacy_para_names(collections_repo)
    assert renamed == []


@pytest.mark.asyncio
async def test_migrate_legacy_para_names_renames_legacy(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """Install with legacy accented name: migration renames to ASCII."""
    legacy = await collections_repo.create_collection(
        name="02_Áreas_de_Responsabilidad", description="legacy"
    )

    renamed = await migrate_legacy_para_names(collections_repo)
    assert renamed == ["02_Áreas_de_Responsabilidad -> 02_Areas_de_Responsabilidad"]

    # Legacy gone, new exists, same collection_id
    assert await collections_repo.get_collection_by_name("02_Áreas_de_Responsabilidad") is None
    new = await collections_repo.get_collection_by_name("02_Areas_de_Responsabilidad")
    assert new is not None
    assert new.collection_id == legacy.collection_id


@pytest.mark.asyncio
async def test_migrate_legacy_para_names_preserves_bridge_links(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """Migration must NOT break vault_file_collections (FK by id)."""
    import time as _time

    legacy = await collections_repo.create_collection(
        name="02_Áreas_de_Responsabilidad", description="legacy"
    )
    file_id = "test-file-bridge-001"
    now_iso = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime())
    await db.conn.execute(
        "INSERT INTO vault_files "
        "(file_id, source_path, content_sha256, mtime, size_bytes, added_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (file_id, "/legacy.md", "h1", 1.0, 10, now_iso),
    )
    await db.conn.execute(
        "INSERT INTO vault_file_collections (file_id, collection_id, added_at) VALUES (?, ?, ?)",
        (file_id, legacy.collection_id, now_iso),
    )
    await db.conn.commit()

    await migrate_legacy_para_names(collections_repo)

    # Bridge link still present (FK is on collection_id, not name)
    cur = await db.conn.execute(
        "SELECT file_id, collection_id FROM vault_file_collections "
        "WHERE file_id = ? AND collection_id = ?",
        (file_id, legacy.collection_id),
    )
    row = await cur.fetchone()
    assert row is not None, "Bridge link lost during migration"


@pytest.mark.asyncio
async def test_migrate_legacy_para_names_collision_skips(
    db: Database, collections_repo: VaultCollectionsRepo, caplog: pytest.LogCaptureFixture
) -> None:
    """Edge case: operator manually created the new ASCII name BEFORE
    the migration ran. Both rows coexist → log warning + skip.
    """
    import logging

    await collections_repo.create_collection(
        name="02_Áreas_de_Responsabilidad", description="legacy"
    )
    await collections_repo.create_collection(
        name="02_Areas_de_Responsabilidad", description="manual"
    )

    with caplog.at_level(logging.WARNING, logger="hermes.memory.seed"):
        renamed = await migrate_legacy_para_names(collections_repo)
    assert renamed == []
    collision = [r for r in caplog.records if r.message == "para_legacy_migration_collision"]
    assert len(collision) == 1


@pytest.mark.asyncio
async def test_seed_after_migrate_does_not_duplicate(
    db: Database, collections_repo: VaultCollectionsRepo
) -> None:
    """After migrate, the seed should NOT recreate the renamed one.

    End-to-end: simulate an existing install that has the legacy
    accented name (only 02_Áreas was accented; 01/03/04 were already
    ASCII), run migrate + seed, and verify no duplicates.
    """
    # Legacy install had the accented 02 + the 3 ASCII defaults
    await collections_repo.create_collection(name="01_Proyectos_Activos", description="legacy")
    await collections_repo.create_collection(
        name="02_Áreas_de_Responsabilidad", description="legacy"
    )
    await collections_repo.create_collection(
        name="03_Recursos_y_Conocimiento", description="legacy"
    )
    await collections_repo.create_collection(name="04_Archivo", description="legacy")

    # Migrate: only 02_Áreas needs renaming
    renamed = await migrate_legacy_para_names(collections_repo)
    assert renamed == ["02_Áreas_de_Responsabilidad -> 02_Areas_de_Responsabilidad"]

    # Now seed: all 4 already exist, seed is no-op
    created = await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)
    assert created == []

    # Total = 4, all with ASCII names
    all_colls = await collections_repo.list_collections(include_archived=True)
    assert len(all_colls) == 4
    assert all(c.name.isascii() for c in all_colls)
