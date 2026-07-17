"""Tests for Sprint 19 Slice 3 — agent tools for vault collections."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncGenerator

import pytest

from hermes.memory.collections import (
    CollectionNotFoundError,
    DuplicateCollectionError,
    VaultCollectionsRepo,
)
from hermes.memory.db import Database
from hermes.tools.collections import (
    add_file_to_collection,
    create_collection,
    list_collections,
    move_file_to_collection,
    register_collections_tools,
    remove_file_from_collection,
)
from hermes.tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def db(tmp_path) -> AsyncGenerator[Database, None]:
    """Initialized empty Database at tmp_path (applies all migrations)."""
    db = Database(tmp_path / "test_collections_tools.db")
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
def repo(db: Database) -> VaultCollectionsRepo:
    return VaultCollectionsRepo(db)


async def _insert_vault_file(
    db: Database,
    file_id: str,
    *,
    source_path: str = "/tmp/test.pdf",
    mtime: float = 100.0,
    size: int = 100,
) -> None:
    """Insert a minimal vault_files row for FK tests."""
    sha = (file_id * 2)[:64]
    await db.conn.execute(
        "INSERT INTO vault_files "
        "(file_id, source_path, content_sha256, mtime, size_bytes) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_id, source_path, sha, mtime, size),
    )
    await db.conn.commit()


# ===========================================================================
# list_collections
# ===========================================================================


async def test_list_collections_returns_empty_when_no_collections(db: Database) -> None:
    result = await list_collections(db=db)
    assert result == {"collections": []}


async def test_list_collections_returns_flat_at_root(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    await repo.create_collection("01_Proyectos_Activos")
    await repo.create_collection("02_Areas")
    result = await list_collections(db=db)
    assert len(result["collections"]) == 2
    names = {c["name"] for c in result["collections"]}
    assert names == {"01_Proyectos_Activos", "02_Areas"}


async def test_list_collections_returns_tree_with_hierarchy(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    root = await repo.create_collection("01_Proyectos_Activos")
    await repo.create_collection("Alfa_Q3", parent_collection_id=root.collection_id)
    await repo.create_collection("Bravo_Q4", parent_collection_id=root.collection_id)
    result = await list_collections(db=db)
    assert len(result["collections"]) == 1
    assert result["collections"][0]["name"] == "01_Proyectos_Activos"
    children = result["collections"][0]["children"]
    assert {c["name"] for c in children} == {"Alfa_Q3", "Bravo_Q4"}
    assert children[0]["children"] == []


async def test_list_collections_excludes_archived_by_default(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    a = await repo.create_collection("A")
    await repo.create_collection("B")
    await repo.archive_collection(a.collection_id)
    result = await list_collections(db=db)
    assert [c["name"] for c in result["collections"]] == ["B"]


async def test_list_collections_includes_archived_when_requested(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    a = await repo.create_collection("A")
    await repo.create_collection("B")
    await repo.archive_collection(a.collection_id)
    result = await list_collections(include_archived=True, db=db)
    assert {c["name"] for c in result["collections"]} == {"A", "B"}
    assert all(c["archived"] in (True, False) for c in result["collections"])


async def test_list_collections_respects_depth_limit(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    root = await repo.create_collection("L0")
    l1 = await repo.create_collection("L1", parent_collection_id=root.collection_id)
    await repo.create_collection("L2", parent_collection_id=l1.collection_id)
    # depth=1 -> only root, no children
    result_d1 = await list_collections(depth=1, db=db)
    assert result_d1["collections"][0]["children"] == []
    # depth=2 -> root + L1 (but not L2)
    result_d2 = await list_collections(depth=2, db=db)
    assert len(result_d2["collections"][0]["children"]) == 1
    assert result_d2["collections"][0]["children"][0]["children"] == []
    # depth=3 -> root + L1 + L2
    result_d3 = await list_collections(depth=3, db=db)
    assert len(result_d3["collections"][0]["children"][0]["children"]) == 1


async def test_list_collections_raises_for_invalid_depth(db: Database) -> None:
    with pytest.raises(ValueError, match="depth must be 1-10"):
        await list_collections(depth=0, db=db)
    with pytest.raises(ValueError, match="depth must be 1-10"):
        await list_collections(depth=11, db=db)


# ===========================================================================
# create_collection
# ===========================================================================


async def test_create_collection_returns_dict_with_required_fields(
    db: Database,
) -> None:
    result = await create_collection(name="01_Proyectos_Activos", db=db)
    assert "collection_id" in result
    assert result["name"] == "01_Proyectos_Activos"
    assert result["parent_collection_id"] is None
    assert result["description"] is None
    assert result["archived"] is False


async def test_create_collection_with_parent_and_description(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    parent = await repo.create_collection("Proyectos")
    result = await create_collection(
        name="Alfa_Q3",
        parent_collection_id=parent.collection_id,
        description="Q3 2026 sprint",
        db=db,
    )
    assert result["parent_collection_id"] == parent.collection_id
    assert result["description"] == "Q3 2026 sprint"


async def test_create_collection_raises_duplicate_name(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    await repo.create_collection("Foo")
    with pytest.raises(DuplicateCollectionError):
        await create_collection(name="Foo", db=db)


async def test_create_collection_raises_for_missing_parent(db: Database) -> None:
    with pytest.raises(CollectionNotFoundError):
        await create_collection(
            name="orphan",
            parent_collection_id="00000000000000000000000000000000",
            db=db,
        )


# ===========================================================================
# add_file_to_collection
# ===========================================================================


async def test_add_file_to_collection_returns_added_true_first_time(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    coll = await repo.create_collection("C")
    await _insert_vault_file(db, "f" * 32)
    result = await add_file_to_collection(file_id="f" * 32, collection_id=coll.collection_id, db=db)
    assert result == {
        "collection_id": coll.collection_id,
        "file_id": "f" * 32,
        "added": True,
    }


async def test_add_file_to_collection_returns_added_false_if_already_linked(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    coll = await repo.create_collection("C")
    await _insert_vault_file(db, "f" * 32)
    await add_file_to_collection(file_id="f" * 32, collection_id=coll.collection_id, db=db)
    result = await add_file_to_collection(file_id="f" * 32, collection_id=coll.collection_id, db=db)
    assert result["added"] is False


async def test_add_file_to_collection_raises_for_missing_collection(db: Database) -> None:
    await _insert_vault_file(db, "f" * 32)
    with pytest.raises(CollectionNotFoundError):
        await add_file_to_collection(
            file_id="f" * 32,
            collection_id="00000000000000000000000000000000",
            db=db,
        )


# ===========================================================================
# remove_file_from_collection
# ===========================================================================


async def test_remove_file_from_collection_returns_removed_true(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    a = await repo.create_collection("A")
    b = await repo.create_collection("B")
    await _insert_vault_file(db, "f" * 32)
    await add_file_to_collection(file_id="f" * 32, collection_id=a.collection_id, db=db)
    await add_file_to_collection(file_id="f" * 32, collection_id=b.collection_id, db=db)
    result = await remove_file_from_collection(
        file_id="f" * 32, collection_id=a.collection_id, db=db
    )
    assert result["removed"] is True
    # Verify: file still in B, NOT in A
    in_a = await repo.list_files_in_collection(a.collection_id)
    in_b = await repo.list_files_in_collection(b.collection_id)
    assert "f" * 32 not in in_a
    assert "f" * 32 in in_b


async def test_remove_file_from_collection_returns_removed_false_if_not_linked(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    # File has 1 link (in B), not in A. Remove from A is no-op idempotent.
    a = await repo.create_collection("A")
    b = await repo.create_collection("B")
    await _insert_vault_file(db, "f" * 32)
    await add_file_to_collection(file_id="f" * 32, collection_id=b.collection_id, db=db)
    result = await remove_file_from_collection(
        file_id="f" * 32, collection_id=a.collection_id, db=db
    )
    assert result["removed"] is False
    # File still in B
    assert "f" * 32 in await repo.list_files_in_collection(b.collection_id)


async def test_remove_file_from_collection_raises_if_last_link_without_allow_last(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    coll = await repo.create_collection("C")
    await _insert_vault_file(db, "f" * 32)
    await add_file_to_collection(file_id="f" * 32, collection_id=coll.collection_id, db=db)
    with pytest.raises(ValueError, match="only 1 collection link"):
        await remove_file_from_collection(file_id="f" * 32, collection_id=coll.collection_id, db=db)


async def test_remove_file_from_collection_succeeds_with_allow_last(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    coll = await repo.create_collection("C")
    await _insert_vault_file(db, "f" * 32)
    await add_file_to_collection(file_id="f" * 32, collection_id=coll.collection_id, db=db)
    result = await remove_file_from_collection(
        file_id="f" * 32,
        collection_id=coll.collection_id,
        allow_last=True,
        db=db,
    )
    assert result["removed"] is True


# ===========================================================================
# move_file_to_collection
# ===========================================================================


async def test_move_file_atomic_add_first(db: Database, repo: VaultCollectionsRepo) -> None:
    a = await repo.create_collection("A")
    b = await repo.create_collection("B")
    await _insert_vault_file(db, "f" * 32)
    await add_file_to_collection(file_id="f" * 32, collection_id=a.collection_id, db=db)
    result = await move_file_to_collection(
        file_id="f" * 32,
        from_collection_id=a.collection_id,
        to_collection_id=b.collection_id,
        db=db,
    )
    assert result["moved"] is True
    assert result["added"] is True
    assert result["removed"] is True
    # Verify: file is in B, NOT in A
    in_a = await repo.list_files_in_collection(a.collection_id)
    in_b = await repo.list_files_in_collection(b.collection_id)
    assert "f" * 32 not in in_a
    assert "f" * 32 in in_b


async def test_move_file_no_op_when_from_equals_to(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    a = await repo.create_collection("A")
    result = await move_file_to_collection(
        file_id="f" * 32,
        from_collection_id=a.collection_id,
        to_collection_id=a.collection_id,
        db=db,
    )
    assert result["moved"] is False
    assert result["added"] is False
    assert result["removed"] is False


async def test_move_file_raises_when_from_missing(db: Database) -> None:
    with pytest.raises(CollectionNotFoundError):
        await move_file_to_collection(
            file_id="f" * 32,
            from_collection_id="0" * 32,  # nonexistent
            to_collection_id="1" * 32,  # also nonexistent, but different
            db=db,
        )


async def test_move_file_raises_when_to_missing(db: Database, repo: VaultCollectionsRepo) -> None:
    a = await repo.create_collection("A")
    with pytest.raises(CollectionNotFoundError):
        await move_file_to_collection(
            file_id="f" * 32,
            from_collection_id=a.collection_id,
            to_collection_id="0" * 32,
            db=db,
        )


# ===========================================================================
# Round 2 fixes (adversarial review Round 1)
# ===========================================================================


async def test_move_file_atomicity_on_add_failure(db: Database, repo: VaultCollectionsRepo) -> None:
    """ADD failure must not touch source. Verify atomicity claim."""
    a = await repo.create_collection("A")
    b = await repo.create_collection("B")
    await _insert_vault_file(db, "f" * 32)
    await add_file_to_collection(file_id="f" * 32, collection_id=a.collection_id, db=db)

    with pytest.raises(sqlite3.IntegrityError):
        await move_file_to_collection(
            file_id="deadbeef" * 8,
            from_collection_id=a.collection_id,
            to_collection_id=b.collection_id,
            db=db,
        )

    assert "f" * 32 in await repo.list_files_in_collection(a.collection_id)
    assert "f" * 32 not in await repo.list_files_in_collection(b.collection_id)


async def test_move_file_no_op_when_from_equals_to_with_file_in_source(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    """from==to early-return cuando el file YA esta en source (no double-action)."""
    a = await repo.create_collection("A")
    await _insert_vault_file(db, "f" * 32)
    await add_file_to_collection(file_id="f" * 32, collection_id=a.collection_id, db=db)
    result = await move_file_to_collection(
        file_id="f" * 32,
        from_collection_id=a.collection_id,
        to_collection_id=a.collection_id,
        db=db,
    )
    assert result["added"] is False
    assert result["removed"] is False
    assert result["moved"] is False
    assert "f" * 32 in await repo.list_files_in_collection(a.collection_id)


async def test_create_collection_rejects_name_over_200_chars(db: Database) -> None:
    """Tool layer enforces maxLength=200 even when LLM bypasses JSON Schema."""
    with pytest.raises(ValueError, match="1-200 chars"):
        await create_collection(name="x" * 201, db=db)


async def test_create_collection_accepts_name_at_200_chars(db: Database) -> None:
    """Boundary: 200 chars accepted (maxLength inclusive)."""
    result = await create_collection(name="x" * 200, db=db)
    assert result["name"] == "x" * 200


async def test_create_collection_strips_whitespace_before_length_check(
    db: Database,
) -> None:
    """Whitespace-only 200-char input rejected at tool layer (Round 3 MINOR-1)."""
    # 200 spaces: passes len() check but strip() yields empty -> rejected
    with pytest.raises(ValueError, match="1-200 chars"):
        await create_collection(name=" " * 200, db=db)


async def test_create_collection_strips_trailing_only_whitespace_at_200(
    db: Database,
) -> None:
    """Strip is applied to count length; trailing-only whitespace is rejected at boundary.

    99 a's + 101 spaces = 200 chars total; after strip becomes 99 chars (OK).
    Internal whitespace would be preserved (verified by separate probe).
    """
    name_input = "a" * 99 + " " * 101
    result = await create_collection(name=name_input, db=db)
    # Repo stores the stripped name (memory/collections.py:159-160 strips)
    assert len(result["name"]) == 99
    assert result["name"].strip() == "a" * 99


async def test_create_collection_preserves_internal_whitespace(db: Database) -> None:
    """Internal whitespace in name is preserved (only edge whitespace stripped)."""
    result = await create_collection(name="foo  bar", db=db)
    assert result["name"] == "foo  bar"


async def test_list_collections_truncates_over_200_nodes(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    """200+ nodes trigger truncation signal so LLM can re-query with depth=1."""
    for i in range(210):
        await repo.create_collection(f"Collection_{i:03d}")
    result = await list_collections(db=db)
    assert result.get("truncated") is True
    assert result["total_node_count"] == 210
    assert result["returned_node_count"] == 200
    assert len(result["collections"]) == 200


async def test_list_collections_default_depth_5_reaches_deep_chains(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    """Default depth=5 reaches a 5-level chain."""
    cur_parent: str | None = None
    for i in range(5):
        c = await repo.create_collection(f"L{i}", parent_collection_id=cur_parent)
        cur_parent = c.collection_id
    result = await list_collections(db=db)
    chain = result["collections"][0]
    for _i in range(4):
        chain = chain["children"][0]
    assert chain["name"] == "L4"
    assert chain["children"] == []  # depth limit stops recursion at L4


async def test_list_collections_truncation_preserves_roots_when_children_sort_first(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    """Regression: tree cap must NOT eat all roots when children sort first.

    Pre-fix bug: with parents P_* and children C_* (alphabetical sort puts
    all 500 C_* before 100 P_*), the flat cap at 200 retained only children.
    LLM saw empty collections list and depth=1 retry also returned 0 roots.
    Fix: ORDER BY roots-first in memory/collections.py list_collections.
    """
    # 100 parents "P_000..P_099" + 5 children each "C_P000_000..C_P099_004"
    # Total = 600 nodes. Alphabetical sort puts all 500 C_* before all 100 P_*.
    for j in range(100):
        parent = await repo.create_collection(f"P_{j:03d}")
        for k in range(5):
            await repo.create_collection(
                f"C_P{j:03d}_{k}",
                parent_collection_id=parent.collection_id,
            )

    result = await list_collections(db=db)
    # Cap kicks in
    assert result.get("truncated") is True
    assert result["total_node_count"] == 600
    assert result["returned_node_count"] == 200
    # But ALL 100 roots must be in the response (they sort first now)
    assert len(result["collections"]) == 100
    # depth=1 retry also works
    result_d1 = await list_collections(depth=1, db=db)
    assert len(result_d1["collections"]) == 100
    assert all(c["children"] == [] for c in result_d1["collections"])


# ===========================================================================
# register helper
# ===========================================================================


async def test_register_collections_tools_registers_5_tools_with_schemas(
    db: Database,
) -> None:
    registry = ToolRegistry()
    register_collections_tools(registry, db=db)
    expected_tools = {
        "list_collections",
        "create_collection",
        "add_file_to_collection",
        "remove_file_from_collection",
        "move_file_to_collection",
    }
    assert set(registry.list_tools()) == expected_tools
    # All 5 schemas present in OpenAI Chat Completions format
    schemas = registry.tool_schemas()
    assert len(schemas) == 5
    schema_names = {s["function"]["name"] for s in schemas}
    assert schema_names == expected_tools


async def test_register_collections_tools_registered_callables_work(
    db: Database, repo: VaultCollectionsRepo
) -> None:
    """Smoke test: las 5 closures registradas funcionan end-to-end via execute()."""
    registry = ToolRegistry()
    register_collections_tools(registry, db=db)

    # 1. list_collections (empty)
    result = await registry.execute("list_collections", {})
    assert result == {"collections": []}

    # 2. create_collection via registry.execute
    coll_result = await registry.execute("create_collection", {"name": "Smoke_Test"})
    assert coll_result["name"] == "Smoke_Test"
    coll_id = coll_result["collection_id"]

    # 3. add_file_to_collection
    await _insert_vault_file(db, "f" * 32)
    add_result = await registry.execute(
        "add_file_to_collection",
        {"file_id": "f" * 32, "collection_id": coll_id},
    )
    assert add_result["added"] is True

    # 4. remove_file_from_collection (with allow_last since this is the only link)
    rm_result = await registry.execute(
        "remove_file_from_collection",
        {"file_id": "f" * 32, "collection_id": coll_id, "allow_last": True},
    )
    assert rm_result["removed"] is True

    # 5. move_file_to_collection (re-add then move to a new collection)
    target_result = await registry.execute("create_collection", {"name": "Smoke_Target"})
    target_id = target_result["collection_id"]
    await registry.execute(
        "add_file_to_collection",
        {"file_id": "f" * 32, "collection_id": coll_id},
    )
    move_result = await registry.execute(
        "move_file_to_collection",
        {
            "file_id": "f" * 32,
            "from_collection_id": coll_id,
            "to_collection_id": target_id,
        },
    )
    assert move_result["moved"] is True
    assert move_result["added"] is True
    assert move_result["removed"] is True


# Static schema validation moved to tests/unit/test_collections_tools_schemas.py
# (sync tests, no asyncio marker, avoids pytest warning).
