"""Tests for VaultCollectionsRepo (Sprint 19 Slice 1)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from uuid import UUID

import pytest

from hermes.memory.collections import (
    CollectionNotFoundError,
    DuplicateCollectionError,
    VaultCollectionsRepo,
)
from hermes.memory.db import Database

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def db(tmp_path) -> AsyncGenerator[Database, None]:
    """Initialized empty Database at tmp_path (applies all migrations)."""
    db = Database(tmp_path / "test_collections.db")
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
def repo(db: Database) -> VaultCollectionsRepo:
    return VaultCollectionsRepo(db)


def _is_valid_uuid4_hex(s: str) -> bool:
    """UUID4 hex: 32 lowercase hex chars, version=4."""
    try:
        u = UUID(s)
    except ValueError:
        return False
    return len(s) == 32 and u.version == 4


async def _insert_vault_file(
    db: Database,
    file_id: str,
    *,
    source_path: str = "/tmp/test.pdf",
    sha: str | None = None,
    mtime: float = 100.0,
    size: int = 100,
) -> None:
    """Insert a minimal vault_files row for FK tests."""
    if sha is None:
        sha = (file_id * 2)[:64]
    await db.conn.execute(
        "INSERT INTO vault_files "
        "(file_id, source_path, content_sha256, mtime, size_bytes) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_id, source_path, sha, mtime, size),
    )
    await db.conn.commit()


# --- Creation ---------------------------------------------------------------


async def test_create_collection_assigns_uuid4_hex(repo: VaultCollectionsRepo) -> None:
    c = await repo.create_collection("01_Proyectos_Activos")
    assert _is_valid_uuid4_hex(c.collection_id)


async def test_create_collection_strips_name_whitespace(repo: VaultCollectionsRepo) -> None:
    c = await repo.create_collection("  foo  ")
    assert c.name == "foo"


async def test_create_collection_rejects_empty_name(repo: VaultCollectionsRepo) -> None:
    with pytest.raises(ValueError, match="empty"):
        await repo.create_collection("")


async def test_create_collection_rejects_whitespace_only_name(
    repo: VaultCollectionsRepo,
) -> None:
    with pytest.raises(ValueError, match="empty"):
        await repo.create_collection("   ")


async def test_create_collection_default_metadata(repo: VaultCollectionsRepo) -> None:
    c = await repo.create_collection("active")
    assert c.archived is False
    assert c.archived_at is None
    assert c.sort_order == 0
    assert c.parent_collection_id is None
    assert c.description is None
    assert c.created_at.endswith("Z")


async def test_create_collection_with_parent(repo: VaultCollectionsRepo) -> None:
    parent = await repo.create_collection("01_Proyectos")
    child = await repo.create_collection(
        "01_Proyectos_Alpha",
        parent_collection_id=parent.collection_id,
    )
    assert child.parent_collection_id == parent.collection_id


async def test_create_collection_raises_on_missing_parent(
    repo: VaultCollectionsRepo,
) -> None:
    with pytest.raises(CollectionNotFoundError):
        await repo.create_collection("orphan", parent_collection_id="0" * 32)


async def test_create_collection_raises_on_duplicate_name(
    repo: VaultCollectionsRepo,
) -> None:
    await repo.create_collection("01_Proyectos")
    with pytest.raises(DuplicateCollectionError):
        await repo.create_collection("01_Proyectos")


async def test_create_collection_duplicate_name_is_case_insensitive(
    repo: VaultCollectionsRepo,
) -> None:
    """Sprint 19 Slice 4d v2: v23 migration changed UNIQUE to COLLATE NOCASE
    so "Foo" and "foo" are equivalent at the SQL level. v0.1 was case-sensitive.
    """
    await repo.create_collection("Foo")
    with pytest.raises(DuplicateCollectionError):
        await repo.create_collection("foo")


# --- find_by_name_and_parent (Sprint 19 Slice 4d v2 commit 2) ---------------


async def test_find_by_name_and_parent_basic(
    repo: VaultCollectionsRepo,
) -> None:
    """Find collection by name at root level (parent=None)."""
    await repo.create_collection("01_Proyectos")
    found = await repo.find_by_name_and_parent("01_Proyectos", None)
    assert found is not None
    assert found.name == "01_Proyectos"
    assert found.parent_collection_id is None


async def test_find_by_name_and_parent_returns_none_if_not_found(
    repo: VaultCollectionsRepo,
) -> None:
    """Returns None when no match."""
    found = await repo.find_by_name_and_parent("nonexistent", None)
    assert found is None


async def test_find_by_name_and_parent_hierarchical(
    repo: VaultCollectionsRepo,
) -> None:
    """Same name at different parent levels is OK after v23 composite UNIQUE."""
    parent_root = await repo.create_collection("01_Proyectos")
    parent_archive = await repo.create_collection("archive")

    child_at_root = await repo.create_collection(
        "Active", parent_collection_id=parent_root.collection_id
    )
    child_at_archive = await repo.create_collection(
        "Active", parent_collection_id=parent_archive.collection_id
    )

    # Both children exist, find_by_name_and_parent disambiguates by parent
    found_root = await repo.find_by_name_and_parent("Active", parent_root.collection_id)
    found_archive = await repo.find_by_name_and_parent("Active", parent_archive.collection_id)

    assert found_root is not None
    assert found_archive is not None
    assert found_root.collection_id == child_at_root.collection_id
    assert found_archive.collection_id == child_at_archive.collection_id
    assert found_root.collection_id != found_archive.collection_id


async def test_find_by_name_and_parent_case_insensitive(
    repo: VaultCollectionsRepo,
) -> None:
    """`.casefold()` match: 'Foo' matches 'foo', 'FOO', 'fOo', etc.

    Unicode correctness: handles German ß→ss, Spanish accents, etc.
    """
    await repo.create_collection("01_Proyectos")
    # Query with different case — should find it
    found = await repo.find_by_name_and_parent("01_PROYECTOS", None)
    assert found is not None
    assert found.name == "01_Proyectos"  # original case preserved

    found = await repo.find_by_name_and_parent("01_proyectos", None)
    assert found is not None
    assert found.name == "01_Proyectos"


async def test_find_by_name_and_parent_unicode_casefold(
    repo: VaultCollectionsRepo,
) -> None:
    """Unicode casefold correctness: 'Áreas' matches 'áreas' (Spanish acute)."""
    await repo.create_collection("02_Áreas_de_Responsabilidad")
    found = await repo.find_by_name_and_parent("02_áreas_de_responsabilidad", None)
    assert found is not None
    assert found.name == "02_Áreas_de_Responsabilidad"  # original case


async def test_find_by_name_and_parent_case_sensitive_param(
    repo: VaultCollectionsRepo,
) -> None:
    """case_insensitive=False requires exact match (rare use case)."""
    await repo.create_collection("Foo")
    # Case-insensitive default: matches
    found = await repo.find_by_name_and_parent("foo", None)
    assert found is not None
    # Explicit case-sensitive: no match
    found = await repo.find_by_name_and_parent("foo", None, case_insensitive=False)
    assert found is None
    # Exact case match
    found = await repo.find_by_name_and_parent("Foo", None, case_insensitive=False)
    assert found is not None


async def test_find_by_name_and_parent_returns_archived(
    repo: VaultCollectionsRepo,
) -> None:
    """find_by_name_and_parent does NOT filter archived (caller's job).

    This is intentional: get_active_collection_by_name already exists for
    the archived=0 case. find_by_name_and_parent returns the row
    regardless of archived state — caller decides.
    """
    coll = await repo.create_collection("01_Proyectos")
    await repo.archive_collection(coll.collection_id, cascade=False)
    found = await repo.find_by_name_and_parent("01_Proyectos", None)
    assert found is not None
    assert found.archived is True


# --- _move_bridge (Sprint 19 Slice 4d v2 commit 2) ---------------------------


async def test_move_bridge_basic(
    repo: VaultCollectionsRepo,
) -> None:
    """Soft-delete old bridge + insert new bridge (v0.4 B4 + v0.6 §1)."""
    coll_a = await repo.create_collection("01_Proyectos")
    coll_b = await repo.create_collection("02_Áreas")

    # Setup: file linked to coll_a
    file_id = "f1" * 16  # 32 hex chars
    await repo._db.conn.execute(
        "INSERT INTO vault_files (file_id, source_path, content_sha256, mtime, added_at, size_bytes, text_version, text_at) "
        "VALUES (?, ?, ?, 1000, '2026-07-12 10:00:00', 100, 'v1', '2026-07-12 10:00:00')",
        (file_id, "/test/foo.pdf", "abc123"),
    )
    await repo._db.conn.execute(
        "INSERT INTO vault_file_collections (file_id, collection_id, added_at, superseded_at) "
        "VALUES (?, ?, '2026-07-12 10:00:00', NULL)",
        (file_id, coll_a.collection_id),
    )

    # Move from A to B
    result = await repo._move_bridge(
        file_id=file_id,
        old_collection_id=coll_a.collection_id,
        new_collection_id=coll_b.collection_id,
    )
    assert result == {"old_superseded": True, "new_inserted": True}

    # Verify: old bridge is superseded, new bridge is active
    async with repo._db.conn.execute(
        "SELECT superseded_at FROM vault_file_collections "
        "WHERE file_id = ? AND collection_id = ?",
        (file_id, coll_a.collection_id),
    ) as cur:
        old = await cur.fetchone()
    assert old[0] is not None, "old bridge should be superseded"
    async with repo._db.conn.execute(
        "SELECT superseded_at FROM vault_file_collections "
        "WHERE file_id = ? AND collection_id = ?",
        (file_id, coll_b.collection_id),
    ) as cur:
        new = await cur.fetchone()
    assert new[0] is None, "new bridge should be active"


async def test_move_bridge_same_collection_noop(
    repo: VaultCollectionsRepo,
) -> None:
    """old == new is a no-op (no UPDATE, no INSERT)."""
    coll = await repo.create_collection("01_Proyectos")
    result = await repo._move_bridge(
        file_id="f1" * 16,
        old_collection_id=coll.collection_id,
        new_collection_id=coll.collection_id,
    )
    assert result == {"old_superseded": False, "new_inserted": False}


async def test_move_bridge_no_old_idempotent_insert(
    repo: VaultCollectionsRepo,
) -> None:
    """old=None just inserts new bridge (case for new file)."""
    coll = await repo.create_collection("01_Proyectos")
    file_id = "f1" * 16
    # _move_bridge only handles the bridge; the file row must exist
    # (caller responsibility, per the spec — DropWatcher creates the
    # vault_files row before calling _move_bridge).
    await repo._db.conn.execute(
        "INSERT INTO vault_files (file_id, source_path, content_sha256, mtime, added_at, size_bytes, text_version, text_at) "
        "VALUES (?, ?, ?, 1000, '2026-07-12 10:00:00', 100, 'v1', '2026-07-12 10:00:00')",
        (file_id, "/test/foo.pdf", "abc123"),
    )
    result = await repo._move_bridge(
        file_id=file_id,
        old_collection_id=None,
        new_collection_id=coll.collection_id,
    )
    assert result == {"old_superseded": False, "new_inserted": True}


async def test_move_bridge_idempotent_on_repeat(
    repo: VaultCollectionsRepo,
) -> None:
    """Calling _move_bridge twice with the same args is idempotent."""
    coll_a = await repo.create_collection("01_Proyectos")
    coll_b = await repo.create_collection("02_Áreas")
    file_id = "f1" * 16

    await repo._db.conn.execute(
        "INSERT INTO vault_files (file_id, source_path, content_sha256, mtime, added_at, size_bytes, text_version, text_at) "
        "VALUES (?, ?, ?, 1000, '2026-07-12 10:00:00', 100, 'v1', '2026-07-12 10:00:00')",
        (file_id, "/test/foo.pdf", "abc123"),
    )

    # First call: insert + soft-delete
    r1 = await repo._move_bridge(
        file_id=file_id,
        old_collection_id=coll_a.collection_id,
        new_collection_id=coll_b.collection_id,
    )
    assert r1 == {"old_superseded": False, "new_inserted": True}

    # Second call: old bridge already superseded (no-op), new bridge already exists (IntegrityError caught, no-op)
    r2 = await repo._move_bridge(
        file_id=file_id,
        old_collection_id=coll_a.collection_id,
        new_collection_id=coll_b.collection_id,
    )
    assert r2 == {"old_superseded": False, "new_inserted": False}


# --- Phantom files regression (Gemini EC-5 warning) -------------------------


async def test_list_files_in_collection_filters_superseded_phantom(
    repo: VaultCollectionsRepo,
) -> None:
    """Gemini EC-5: read queries MUST filter `vfc.superseded_at IS NULL`.

    Scenario: file_id is moved from coll_a to coll_b. list_files_in_collection
    on coll_a should NOT return the file (the bridge is superseded).
    Without the filter, the user would see "phantom" files.
    """
    coll_a = await repo.create_collection("01_Proyectos")
    coll_b = await repo.create_collection("02_Áreas")
    file_id = "f1" * 16

    await repo._db.conn.execute(
        "INSERT INTO vault_files (file_id, source_path, content_sha256, mtime, added_at, size_bytes, text_version, text_at) "
        "VALUES (?, ?, ?, 1000, '2026-07-12 10:00:00', 100, 'v1', '2026-07-12 10:00:00')",
        (file_id, "/test/foo.pdf", "abc123"),
    )
    # Link to A
    await repo._db.conn.execute(
        "INSERT INTO vault_file_collections (file_id, collection_id, added_at, superseded_at) "
        "VALUES (?, ?, '2026-07-12 10:00:00', NULL)",
        (file_id, coll_a.collection_id),
    )

    # Before move: file is in A
    files_in_a = await repo.list_files_in_collection(coll_a.collection_id)
    assert files_in_a == [file_id]
    files_in_b = await repo.list_files_in_collection(coll_b.collection_id)
    assert files_in_b == []

    # Move
    await repo._move_bridge(
        file_id=file_id,
        old_collection_id=coll_a.collection_id,
        new_collection_id=coll_b.collection_id,
    )

    # After move: file is in B, NOT in A
    files_in_a = await repo.list_files_in_collection(coll_a.collection_id)
    assert files_in_a == [], f"phantom file in A after move: {files_in_a}"
    files_in_b = await repo.list_files_in_collection(coll_b.collection_id)
    assert files_in_b == [file_id]


async def test_list_collections_for_file_filters_superseded_phantom(
    repo: VaultCollectionsRepo,
) -> None:
    """list_collections_for_file MUST also filter superseded."""
    coll_a = await repo.create_collection("01_Proyectos")
    coll_b = await repo.create_collection("02_Áreas")
    file_id = "f1" * 16

    await repo._db.conn.execute(
        "INSERT INTO vault_files (file_id, source_path, content_sha256, mtime, added_at, size_bytes, text_version, text_at) "
        "VALUES (?, ?, ?, 1000, '2026-07-12 10:00:00', 100, 'v1', '2026-07-12 10:00:00')",
        (file_id, "/test/foo.pdf", "abc123"),
    )
    # Initial state: file linked to A only
    await repo._db.conn.execute(
        "INSERT INTO vault_file_collections (file_id, collection_id, added_at, superseded_at) "
        "VALUES (?, ?, '2026-07-12 10:00:00', NULL)",
        (file_id, coll_a.collection_id),
    )

    # Before move: file in A only
    colls = await repo.list_collections_for_file(file_id)
    assert {c.collection_id for c in colls} == {coll_a.collection_id}

    # Move: supersede A, insert B
    await repo._move_bridge(
        file_id=file_id,
        old_collection_id=coll_a.collection_id,
        new_collection_id=coll_b.collection_id,
    )

    # After move: only B is active (A is superseded)
    colls = await repo.list_collections_for_file(file_id)
    assert len(colls) == 1
    assert colls[0].collection_id == coll_b.collection_id


# --- Get / List -------------------------------------------------------------


async def test_get_collection_returns_none_for_missing(
    repo: VaultCollectionsRepo,
) -> None:
    assert await repo.get_collection("0" * 32) is None


async def test_get_collection_roundtrip(repo: VaultCollectionsRepo) -> None:
    c1 = await repo.create_collection("foo")
    c2 = await repo.get_collection(c1.collection_id)
    assert c2 == c1


async def test_get_collection_by_name(repo: VaultCollectionsRepo) -> None:
    await repo.create_collection("01_Proyectos_Activos")
    c = await repo.get_collection_by_name("01_Proyectos_Activos")
    assert c is not None
    assert c.name == "01_Proyectos_Activos"


async def test_get_collection_by_name_returns_none_when_missing(
    repo: VaultCollectionsRepo,
) -> None:
    assert await repo.get_collection_by_name("ghost") is None


async def test_list_collections_excludes_archived_by_default(
    repo: VaultCollectionsRepo,
) -> None:
    a = await repo.create_collection("a")
    b_id = (await repo.create_collection("b")).collection_id
    await repo.archive_collection(a.collection_id)

    listed = await repo.list_collections()
    names = [c.name for c in listed]
    assert names == ["b"]
    # b still active
    assert (await repo.get_collection(b_id)).archived is False


async def test_list_collections_includes_archived_when_requested(
    repo: VaultCollectionsRepo,
) -> None:
    a = await repo.create_collection("a")
    b_id = (await repo.create_collection("b")).collection_id
    await repo.archive_collection(a.collection_id)

    listed = await repo.list_collections(include_archived=True)
    names = sorted(c.name for c in listed)
    assert names == ["a", "b"]
    assert (await repo.get_collection(b_id)).archived is False


async def test_list_collections_filter_by_parent(
    repo: VaultCollectionsRepo,
) -> None:
    parent = await repo.create_collection("p")
    child1 = await repo.create_collection("c1", parent_collection_id=parent.collection_id)
    child2 = await repo.create_collection("c2", parent_collection_id=parent.collection_id)
    other = await repo.create_collection("other")

    children = await repo.list_collections(parent_collection_id=parent.collection_id)
    assert {c.collection_id for c in children} == {
        child1.collection_id,
        child2.collection_id,
    }
    assert other.collection_id not in {c.collection_id for c in children}


async def test_list_collections_ordered_by_sort_order_then_name(
    repo: VaultCollectionsRepo,
) -> None:
    await repo.create_collection("b", sort_order=1)
    await repo.create_collection("a", sort_order=0)
    await repo.create_collection("c", sort_order=1)

    listed = await repo.list_collections()
    assert [c.name for c in listed] == ["a", "b", "c"]


# --- Archive / Restore (with cascade) ---------------------------------------


async def test_archive_collection_sets_flag_and_timestamp(
    repo: VaultCollectionsRepo,
) -> None:
    c = await repo.create_collection("foo")
    n = await repo.archive_collection(c.collection_id, cascade=False)
    assert n == 1
    after = await repo.get_collection(c.collection_id)
    assert after is not None
    assert after.archived is True
    assert after.archived_at is not None
    assert after.archived_at.endswith("Z")


async def test_archive_collection_idempotent(repo: VaultCollectionsRepo) -> None:
    c = await repo.create_collection("foo")
    n1 = await repo.archive_collection(c.collection_id)
    n2 = await repo.archive_collection(c.collection_id)
    assert n1 == 1
    assert n2 == 0


async def test_archive_collection_raises_on_missing(
    repo: VaultCollectionsRepo,
) -> None:
    with pytest.raises(CollectionNotFoundError):
        await repo.archive_collection("f" * 32)


async def test_archive_collection_cascades_to_descendants(
    repo: VaultCollectionsRepo,
) -> None:
    """Option A (Gemini Sprint 19 EC-5): archiving parent cascades to descendants."""
    grandparent = await repo.create_collection("gp")
    parent = await repo.create_collection("p", parent_collection_id=grandparent.collection_id)
    child = await repo.create_collection("c", parent_collection_id=parent.collection_id)
    unrelated = await repo.create_collection("ur")

    n = await repo.archive_collection(parent.collection_id, cascade=True)
    assert n == 2

    assert (await repo.get_collection(parent.collection_id)).archived is True
    assert (await repo.get_collection(child.collection_id)).archived is True
    assert (await repo.get_collection(grandparent.collection_id)).archived is False
    assert (await repo.get_collection(unrelated.collection_id)).archived is False


async def test_archive_collection_cascade_depth_20_cap(
    repo: VaultCollectionsRepo,
) -> None:
    """Recursive CTE stops at total depth 20 (parent + 19 descendants)."""
    head = await repo.create_collection("L0")
    cur = head
    for i in range(1, 25):
        nxt = await repo.create_collection(f"L{i}", parent_collection_id=cur.collection_id)
        cur = nxt

    n = await repo.archive_collection(head.collection_id, cascade=True)
    assert n == 20


async def test_restore_collection_unarchives_single(
    repo: VaultCollectionsRepo,
) -> None:
    """restore_collection unarchives a single collection. NO cascade unarchive."""
    parent = await repo.create_collection("p")
    child = await repo.create_collection("c", parent_collection_id=parent.collection_id)
    await repo.archive_collection(parent.collection_id, cascade=True)

    n = await repo.restore_collection(parent.collection_id)
    assert n == 1

    parent_after = await repo.get_collection(parent.collection_id)
    child_after = await repo.get_collection(child.collection_id)
    assert parent_after.archived is False
    assert parent_after.archived_at is None
    assert child_after.archived is True


async def test_restore_collection_on_active_is_noop(
    repo: VaultCollectionsRepo,
) -> None:
    c = await repo.create_collection("foo")
    n = await repo.restore_collection(c.collection_id)
    assert n == 0


# --- Bridge: vault_file_collections -----------------------------------------


async def test_add_file_to_collection_inserts_bridge_row(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    c = await repo.create_collection("foo")
    file_id = "deadbeef" * 4
    await _insert_vault_file(db, file_id)

    inserted = await repo.add_file_to_collection(file_id, c.collection_id)
    assert inserted is True


async def test_add_file_to_collection_raises_on_missing_collection(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    file_id = "deadbeef" * 4
    await _insert_vault_file(db, file_id)

    with pytest.raises(CollectionNotFoundError):
        await repo.add_file_to_collection(file_id, "0" * 32)


async def test_add_file_to_collection_idempotent(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    c = await repo.create_collection("foo")
    file_id = "feedface" * 4
    await _insert_vault_file(db, file_id)

    first = await repo.add_file_to_collection(file_id, c.collection_id)
    second = await repo.add_file_to_collection(file_id, c.collection_id)
    assert first is True
    assert second is False


async def test_remove_file_from_collection(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    c = await repo.create_collection("foo")
    file_id = "cafebabe" * 4
    await _insert_vault_file(db, file_id)

    await repo.add_file_to_collection(file_id, c.collection_id)
    removed = await repo.remove_file_from_collection(file_id, c.collection_id)
    assert removed is True
    assert await repo.remove_file_from_collection(file_id, c.collection_id) is False


async def test_list_files_in_collection_excludes_archived_collections(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    """Archiving hides files from list, but bridge row remains (append-only)."""
    c = await repo.create_collection("foo")
    file_id = "12345678" * 4
    await _insert_vault_file(db, file_id)

    await repo.add_file_to_collection(file_id, c.collection_id)
    assert await repo.list_files_in_collection(c.collection_id) == [file_id]

    await repo.archive_collection(c.collection_id)
    assert await repo.list_files_in_collection(c.collection_id) == []
    cursor = await db.conn.execute(
        "SELECT COUNT(*) FROM vault_file_collections " "WHERE file_id = ? AND collection_id = ?",
        (file_id, c.collection_id),
    )
    row = await cursor.fetchone()
    assert row[0] == 1


async def test_list_files_in_collection_excludes_orphans(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    c = await repo.create_collection("foo")
    file_id = "abcd1234" * 4
    await _insert_vault_file(db, file_id)

    await repo.add_file_to_collection(file_id, c.collection_id)
    assert file_id in await repo.list_files_in_collection(c.collection_id)

    await repo.set_file_orphaned(file_id)
    assert file_id not in await repo.list_files_in_collection(c.collection_id)
    assert file_id in await repo.list_files_in_collection(
        c.collection_id,
        include_orphans=True,
    )


async def test_list_collections_for_file_excludes_archived(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    c1 = await repo.create_collection("c1")
    c2 = await repo.create_collection("c2")
    file_id = "feedbeef" * 4
    await _insert_vault_file(db, file_id)

    await repo.add_file_to_collection(file_id, c1.collection_id)
    await repo.add_file_to_collection(file_id, c2.collection_id)
    await repo.archive_collection(c1.collection_id)

    active = await repo.list_collections_for_file(file_id)
    assert {c.collection_id for c in active} == {c2.collection_id}


# --- Orphan tracking on vault_files -----------------------------------------


async def test_set_file_orphaned_sets_timestamp(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    file_id = "abcdef01" * 4
    await _insert_vault_file(db, file_id)

    ok = await repo.set_file_orphaned(file_id)
    assert ok is True

    cursor = await db.conn.execute(
        "SELECT orphaned_at FROM vault_files WHERE file_id = ?",
        (file_id,),
    )
    row = await cursor.fetchone()
    assert row[0] is not None
    assert row[0].endswith("Z")


async def test_set_file_orphaned_returns_false_for_missing_file(
    repo: VaultCollectionsRepo,
) -> None:
    assert await repo.set_file_orphaned("0" * 32) is False


async def test_clear_file_orphaned_resets_to_null(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    file_id = "abcdef02" * 4
    await _insert_vault_file(db, file_id)

    await repo.set_file_orphaned(file_id)
    ok = await repo.clear_file_orphaned(file_id)
    assert ok is True

    cursor = await db.conn.execute(
        "SELECT orphaned_at FROM vault_files WHERE file_id = ?",
        (file_id,),
    )
    row = await cursor.fetchone()
    assert row[0] is None


async def test_clear_file_orphaned_returns_false_for_missing(
    repo: VaultCollectionsRepo,
) -> None:
    assert await repo.clear_file_orphaned("0" * 32) is False


async def test_list_orphaned_files_returns_only_orphans_oldest_first(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    for i in range(3):
        file_id = f"{i:08d}" * 4
        await _insert_vault_file(db, file_id, sha=f"{(i+0xa0):02x}" * 32)

    await repo.set_file_orphaned(f"{0:08d}" * 4, orphaned_at="2026-07-01T00:00:00Z")
    await repo.set_file_orphaned(f"{1:08d}" * 4, orphaned_at="2026-07-05T00:00:00Z")

    orphans = await repo.list_orphaned_files()
    assert orphans == [f"{0:08d}" * 4, f"{1:08d}" * 4]


async def test_list_orphaned_files_respects_limit(
    repo: VaultCollectionsRepo,
    db: Database,
) -> None:
    for i in range(5):
        file_id = f"{i+10:08d}" * 4
        await _insert_vault_file(db, file_id, sha=f"{(i+0xb0):02x}" * 32)

    for i in range(5):
        await repo.set_file_orphaned(f"{i+10:08d}" * 4)

    orphans = await repo.list_orphaned_files(limit=2)
    assert len(orphans) == 2


# --- Concurrency ------------------------------------------------------------


async def test_concurrent_create_collection_no_deadlock(
    repo: VaultCollectionsRepo,
) -> None:
    names = [f"collection_{i:02d}" for i in range(10)]
    results = await asyncio.gather(*[repo.create_collection(n) for n in names])
    ids = {c.collection_id for c in results}
    assert len(ids) == 10
    assert {c.name for c in results} == set(names)
