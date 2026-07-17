"""Tests documenting bugs found during self-review of Sprint 19 Slice 1+2.

These tests fail RED on the current code. They will go GREEN after
the BLOCKING bugs are fixed.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from hermes.llm.router import LLMRouter
from hermes.memory.collections import VaultCollectionsRepo
from hermes.receivers.http_api import create_app
from hermes.tools.registry import ToolRegistry

# --- Fixtures (duplicated from test_collections_api.py) ---


@pytest.fixture
def router_mock(settings: Any, respx_mock: Any) -> LLMRouter:
    """LLMRouter with respx-mocked OpenAI/Anthropic endpoints."""
    openai_url = f"{settings.opencode_go_base_url}/chat/completions"
    anthropic_url = f"{settings.opencode_go_base_url}/messages"
    respx_mock.post(openai_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ),
    )
    respx_mock.post(anthropic_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
    )
    return LLMRouter(settings)


@pytest.fixture
def collections_repo(db: Any) -> VaultCollectionsRepo:
    return VaultCollectionsRepo(db)


@pytest.fixture
def app_with_collections(
    settings: Any,
    db: Any,
    router_mock: Any,
    collections_repo: VaultCollectionsRepo,
) -> Any:
    app = create_app(settings, db, router_mock, ToolRegistry())
    app.state.collections_repo = collections_repo
    return app


@pytest.fixture
def client_with_collections(app_with_collections: Any) -> Any:
    from fastapi.testclient import TestClient

    with TestClient(app_with_collections) as c:
        yield c


# --- Bug 1 (BLOCKING): PATCH cannot UNPARENT (set parent_collection_id=null) ---


def test_patch_collection_unparent_sets_null_to_root(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
) -> None:
    """PATCH with parent_collection_id=null should unparent the collection
    (move it to root). Current code skips the update when parent is None,
    making unparent impossible.
    """
    import asyncio

    parent = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("parent"),
    )
    child = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection(
            "child",
            parent_collection_id=parent.collection_id,
        ),
    )
    # Confirm initial state: child has parent
    assert child.parent_collection_id == parent.collection_id

    # Send PATCH with explicit null for parent
    response = client_with_collections.patch(
        f"/v1/collections/{child.collection_id}",
        json={"parent_collection_id": None},
    )
    # After fix: should return 200 with parent_collection_id=None
    assert response.status_code == 200
    assert response.json()["parent_collection_id"] is None


# --- Bug 2 (BLOCKING): DELETE parent with sub-collections fails with FK 500 ---


def test_delete_collection_with_subcollections_cascades(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
) -> None:
    """DELETE parent collection should recursively delete child collections too.

    Current code: only deletes bridge rows + the parent collection itself.
    Children with parent_collection_id pointing to the parent would violate
    FK ON DELETE RESTRICT and the DB would raise an IntegrityError -> 500.
    """
    import asyncio

    parent = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("parent"),
    )
    child = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection(
            "child",
            parent_collection_id=parent.collection_id,
        ),
    )

    # DELETE parent with confirm=true
    response = client_with_collections.delete(
        f"/v1/collections/{parent.collection_id}",
        params={"confirm": "true"},
    )
    # After fix: should return 204 (cascade delete)
    assert response.status_code == 204
    # Both should be gone
    assert (
        asyncio.get_event_loop().run_until_complete(
            collections_repo.get_collection(parent.collection_id),
        )
        is None
    )
    assert (
        asyncio.get_event_loop().run_until_complete(
            collections_repo.get_collection(child.collection_id),
        )
        is None
    )


def test_delete_collection_with_many_children_no_orphans(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
    db: Any,
) -> None:
    """Verifier B2 finding (2026-07-09): DELETE with >1000 descendants
    must not leave orphan rows.

    Pre-fix: the DELETE endpoint used a flat `LIMIT 1000` in the
    recursive CTE. With >1000 children, the limit cut the result
    mid-subtree, leaving orphans with parent_collection_id pointing to
    a now-deleted row.

    Post-fix: the recursive CTE is depth-bounded (d.depth < 19,
    same as archive_collection), not row-bounded. Realistic PARA
    trees are <5 levels deep; 20 levels handles any reasonable tree.

    This test creates 1500 direct children (1 level deep), so the
    flat LIMIT would have triggered the bug. Verifies that DELETE
    leaves zero orphans.
    """
    import asyncio

    parent = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("parent_1500"),
    )

    # Create 1500 children
    for i in range(1500):
        asyncio.get_event_loop().run_until_complete(
            collections_repo.create_collection(
                f"child_{i:04d}",
                parent_collection_id=parent.collection_id,
            ),
        )

    # DELETE parent with confirm=true
    response = client_with_collections.delete(
        f"/v1/collections/{parent.collection_id}",
        params={"confirm": "true"},
    )
    assert response.status_code == 204

    # Verify parent gone
    assert (
        asyncio.get_event_loop().run_until_complete(
            collections_repo.get_collection(parent.collection_id),
        )
        is None
    )

    # Verify zero orphans (no rows with parent_collection_id set)
    async def _count_orphans() -> int:
        async with db_conn.execute(
            "SELECT COUNT(*) FROM vault_collections " "WHERE parent_collection_id IS NOT NULL",
        ) as cur:
            row = await cur.fetchone()
        return int(row[0])

    db_conn = db.conn
    orphan_count = asyncio.get_event_loop().run_until_complete(_count_orphans())
    assert orphan_count == 0, f"Expected 0 orphans after DELETE, got {orphan_count}"


def test_delete_collection_with_deep_tree_no_orphans(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
    db: Any,
) -> None:
    """Verifier B2 follow-up (2026-07-09 00:37, project owner): multi-level depth.

    Pre-fix regression test (test_delete_collection_with_many_children_no_orphans)
    covered flat trees (depth=1, 1500 direct children). project owner pointed out
    that real PARA setups have folders within folders — depth > 1.

    This test creates a 4-level tree with 4 children per level (1 + 4 + 16
    + 64 = 85 collections total) and verifies DELETE propagates correctly
    through every depth.

    Pre-fix: the flat LIMIT 1000 in the recursive CTE would also break
    here because the total CTE result (85 rows) would fit, but the
    depth-bound fix (d.depth < 19) is what we want to verify works for
    deep trees too. If the depth bound was mis-coded (e.g., d.depth < 1
    instead of < 19), this test would catch it.
    """
    import asyncio

    root = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("deep_root"),
    )

    def build_subtree(parent_id: str, parent_name: str, depth_remaining: int) -> int:
        """Recursively create a balanced subtree. Returns total nodes.
        Names are unique via path-based naming (parent_name_i)."""
        if depth_remaining == 0:
            return 0
        total = 0
        for i in range(4):
            child_name = f"{parent_name}_{i}"
            child = asyncio.get_event_loop().run_until_complete(
                collections_repo.create_collection(
                    child_name,
                    parent_collection_id=parent_id,
                ),
            )
            total += 1
            total += build_subtree(
                child.collection_id,
                child_name,
                depth_remaining - 1,
            )
        return total

    build_subtree(root.collection_id, "deep_root", 3)

    async def _count_subtree() -> int:
        async with db.conn.execute(
            """
            WITH RECURSIVE subtree(id) AS (
                SELECT collection_id FROM vault_collections
                WHERE collection_id = ?
                UNION ALL
                SELECT vc.collection_id FROM vault_collections vc
                JOIN subtree s ON vc.parent_collection_id = s.id
            )
            SELECT COUNT(*) FROM subtree
            """,
            (root.collection_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row[0])

    setup_count = asyncio.get_event_loop().run_until_complete(_count_subtree())
    assert setup_count == 85, f"Setup expected 85 collections, got {setup_count}"

    response = client_with_collections.delete(
        f"/v1/collections/{root.collection_id}",
        params={"confirm": "true"},
    )
    assert response.status_code == 204

    remaining = asyncio.get_event_loop().run_until_complete(_count_subtree())
    assert remaining == 0, f"DELETE should remove all 85 collections; {remaining} remain"

    async def _count_orphans() -> int:
        async with db.conn.execute(
            "SELECT COUNT(*) FROM vault_collections " "WHERE parent_collection_id IS NOT NULL",
        ) as cur:
            row = await cur.fetchone()
        return int(row[0])

    orphan_count = asyncio.get_event_loop().run_until_complete(_count_orphans())
    assert orphan_count == 0, f"Expected 0 orphans after DELETE deep tree, got {orphan_count}"
