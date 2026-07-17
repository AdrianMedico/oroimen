"""Tests for collections API endpoints (Sprint 19 Slice 2).

TDD-first: 8 endpoints under /v1/collections/*.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from hermes.llm.router import LLMRouter
from hermes.memory.collections import VaultCollectionsRepo
from hermes.receivers.http_api import create_app
from hermes.tools.registry import ToolRegistry

# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def router_mock(settings: Any, respx_mock: Any) -> LLMRouter:
    """LLMRouter with respx-mocked OpenAI/Anthropic endpoints.

    Returns simple JSON responses so /v1/chat/completions tests (if any) don't
    fail. We don't exercise chat in these tests, but the router is required
    by create_app().
    """
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
    """VaultCollectionsRepo wired to the test db."""
    return VaultCollectionsRepo(db)


@pytest.fixture
def app_with_collections(
    settings: Any,
    db: Any,
    router_mock: Any,
    collections_repo: VaultCollectionsRepo,
) -> Any:
    """FastAPI app with VaultCollectionsRepo wired into state."""
    app = create_app(settings, db, router_mock, ToolRegistry())
    app.state.collections_repo = collections_repo
    return app


@pytest.fixture
def client_with_collections(app_with_collections: Any) -> Any:
    """TestClient for the app with collections."""
    from fastapi.testclient import TestClient

    with TestClient(app_with_collections) as c:
        yield c


# --- POST /v1/collections (Create) -----------------------------------------


def test_create_collection_returns_201(client_with_collections: Any) -> None:
    """POST /v1/collections with valid body returns 201 + collection JSON."""
    response = client_with_collections.post(
        "/v1/collections",
        json={"name": "01_Proyectos_Activos"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "01_Proyectos_Activos"
    assert len(data["collection_id"]) == 32
    assert data["archived"] == 0
    assert data["created_at"].endswith("Z")


def test_create_collection_with_parent(client_with_collections: Any) -> None:
    """POST with parent_collection_id sets the parent link."""
    parent_resp = client_with_collections.post(
        "/v1/collections",
        json={"name": "parent"},
    )
    parent_id = parent_resp.json()["collection_id"]

    child_resp = client_with_collections.post(
        "/v1/collections",
        json={"name": "child", "parent_collection_id": parent_id},
    )
    assert child_resp.status_code == 201
    assert child_resp.json()["parent_collection_id"] == parent_id


def test_create_collection_duplicate_name_returns_409(
    client_with_collections: Any,
) -> None:
    """POST with name that already exists returns 409."""
    client_with_collections.post("/v1/collections", json={"name": "foo"})
    response = client_with_collections.post("/v1/collections", json={"name": "foo"})
    assert response.status_code == 409


def test_create_collection_missing_parent_returns_404(
    client_with_collections: Any,
) -> None:
    """POST with non-existent parent_collection_id returns 404."""
    response = client_with_collections.post(
        "/v1/collections",
        json={"name": "orphan", "parent_collection_id": "0" * 32},
    )
    assert response.status_code == 404


def test_create_collection_empty_name_returns_422(
    client_with_collections: Any,
) -> None:
    """POST with empty name returns 422 (Pydantic validation)."""
    response = client_with_collections.post("/v1/collections", json={"name": ""})
    assert response.status_code == 422


# --- GET /v1/collections (List tree) ---------------------------------------


def test_list_collections_returns_empty_array(
    client_with_collections: Any,
) -> None:
    """GET with no collections returns {'collections': []}."""
    response = client_with_collections.get("/v1/collections")
    assert response.status_code == 200
    assert response.json() == {"collections": []}


def test_list_collections_returns_active_only_by_default(
    client_with_collections: Any,
) -> None:
    """Default excludes archived collections."""
    a_resp = client_with_collections.post("/v1/collections", json={"name": "a"})
    a_id = a_resp.json()["collection_id"]
    client_with_collections.post("/v1/collections", json={"name": "b"})

    # Archive a via the repo (no DELETE endpoint exercised in this test)
    repo = client_with_collections.app.state.collections_repo
    import asyncio

    asyncio.get_event_loop().run_until_complete(repo.archive_collection(a_id))

    response = client_with_collections.get("/v1/collections")
    assert response.status_code == 200
    names = [c["name"] for c in response.json()["collections"]]
    assert names == ["b"]


def test_list_collections_includes_archived_when_requested(
    client_with_collections: Any,
) -> None:
    """?include_archived=true shows both."""
    client_with_collections.post("/v1/collections", json={"name": "a"})
    client_with_collections.post("/v1/collections", json={"name": "b"})
    response = client_with_collections.get(
        "/v1/collections",
        params={"include_archived": "true"},
    )
    names = sorted(c["name"] for c in response.json()["collections"])
    assert names == ["a", "b"]


# --- POST /v1/collections/{id}/restore ------------------------------------


def test_restore_collection_unarchives(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
) -> None:
    """POST /restore on archived collection sets archived=0."""
    import asyncio

    c = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("foo"),
    )
    asyncio.get_event_loop().run_until_complete(
        collections_repo.archive_collection(c.collection_id),
    )

    response = client_with_collections.post(
        f"/v1/collections/{c.collection_id}/restore",
    )
    assert response.status_code == 200
    assert response.json()["archived"] == 0


def test_restore_collection_not_found_returns_404(
    client_with_collections: Any,
) -> None:
    response = client_with_collections.post(f"/v1/collections/{'0' * 32}/restore")
    assert response.status_code == 404


# --- DELETE /v1/collections/{id} (Hard delete) ----------------------------


def test_delete_collection_without_confirm_returns_400(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
) -> None:
    """DELETE without ?confirm=true returns 400 (safety check)."""
    import asyncio

    c = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("foo"),
    )

    response = client_with_collections.delete(
        f"/v1/collections/{c.collection_id}",
    )
    assert response.status_code == 400


def test_delete_collection_with_confirm_returns_204(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
) -> None:
    """DELETE with ?confirm=true removes the collection."""
    import asyncio

    c = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("foo"),
    )

    response = client_with_collections.delete(
        f"/v1/collections/{c.collection_id}",
        params={"confirm": "true"},
    )
    assert response.status_code == 204


def test_delete_collection_not_found_returns_404(
    client_with_collections: Any,
) -> None:
    response = client_with_collections.delete(
        f"/v1/collections/{'0' * 32}",
        params={"confirm": "true"},
    )
    assert response.status_code == 404


# --- PATCH /v1/collections/{id} --------------------------------------------


def test_patch_collection_renames(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
) -> None:
    """PATCH with new name updates the collection."""
    import asyncio

    c = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("foo"),
    )

    response = client_with_collections.patch(
        f"/v1/collections/{c.collection_id}",
        json={"name": "bar"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "bar"


def test_patch_collection_duplicate_name_returns_409(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
) -> None:
    """PATCH to a name that already exists returns 409."""
    import asyncio

    asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("foo"),
    )
    c2 = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("bar"),
    )

    response = client_with_collections.patch(
        f"/v1/collections/{c2.collection_id}",
        json={"name": "foo"},
    )
    assert response.status_code == 409


def test_patch_collection_reparent_blocks_cycle(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
) -> None:
    """PATCH parent_collection_id that would create a cycle returns 409."""
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

    # Try to reparent parent under child (would create cycle)
    response = client_with_collections.patch(
        f"/v1/collections/{parent.collection_id}",
        json={"parent_collection_id": child.collection_id},
    )
    assert response.status_code == 409


# --- POST /v1/collections/{id}/files --------------------------------------


def test_add_file_to_collection_returns_201(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
    db: Any,
) -> None:
    """POST adds the bridge row, returns 201."""
    import asyncio

    c = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("foo"),
    )
    file_id = "deadbeef" * 4
    asyncio.get_event_loop().run_until_complete(
        db.conn.execute(
            "INSERT INTO vault_files "
            "(file_id, source_path, content_sha256, mtime, size_bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_id, "/tmp/test.pdf", "a" * 64, 100.0, 100),
        ),
    )
    asyncio.get_event_loop().run_until_complete(db.conn.commit())

    response = client_with_collections.post(
        f"/v1/collections/{c.collection_id}/files",
        json={"file_id": file_id},
    )
    assert response.status_code == 201


def test_add_file_duplicate_returns_409(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
    db: Any,
) -> None:
    """POST same file twice returns 409 (UNIQUE constraint)."""
    import asyncio

    c = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("foo"),
    )
    file_id = "feedface" * 4
    asyncio.get_event_loop().run_until_complete(
        db.conn.execute(
            "INSERT INTO vault_files "
            "(file_id, source_path, content_sha256, mtime, size_bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_id, "/tmp/test.pdf", "b" * 64, 100.0, 100),
        ),
    )
    asyncio.get_event_loop().run_until_complete(db.conn.commit())

    client_with_collections.post(
        f"/v1/collections/{c.collection_id}/files",
        json={"file_id": file_id},
    )
    response = client_with_collections.post(
        f"/v1/collections/{c.collection_id}/files",
        json={"file_id": file_id},
    )
    assert response.status_code == 409


# --- DELETE /v1/collections/{id}/files/{file_id} --------------------------


def test_remove_file_returns_204(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
    db: Any,
) -> None:
    """DELETE removes the bridge row."""
    import asyncio

    c = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("foo"),
    )
    file_id = "cafebabe" * 4
    asyncio.get_event_loop().run_until_complete(
        db.conn.execute(
            "INSERT INTO vault_files "
            "(file_id, source_path, content_sha256, mtime, size_bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_id, "/tmp/test.pdf", "c" * 64, 100.0, 100),
        ),
    )
    asyncio.get_event_loop().run_until_complete(db.conn.commit())
    asyncio.get_event_loop().run_until_complete(
        collections_repo.add_file_to_collection(file_id, c.collection_id),
    )

    response = client_with_collections.delete(
        f"/v1/collections/{c.collection_id}/files/{file_id}",
    )
    assert response.status_code == 204


# --- GET /v1/collections/{id}/files ---------------------------------------


def test_list_files_in_collection_returns_files(
    client_with_collections: Any,
    collections_repo: VaultCollectionsRepo,
    db: Any,
) -> None:
    """GET returns files in collection with pagination cursor."""
    import asyncio

    c = asyncio.get_event_loop().run_until_complete(
        collections_repo.create_collection("foo"),
    )
    for i in range(2):
        file_id = f"{i:08x}" * 4
        asyncio.get_event_loop().run_until_complete(
            db.conn.execute(
                "INSERT INTO vault_files "
                "(file_id, source_path, content_sha256, mtime, size_bytes) "
                "VALUES (?, ?, ?, ?, ?)",
                (file_id, f"/tmp/test{i}.pdf", f"{i:02x}" * 32, 100.0, 100),
            ),
        )
    asyncio.get_event_loop().run_until_complete(db.conn.commit())

    for i in range(2):
        asyncio.get_event_loop().run_until_complete(
            collections_repo.add_file_to_collection(f"{i:08x}" * 4, c.collection_id),
        )

    response = client_with_collections.get(f"/v1/collections/{c.collection_id}/files")
    assert response.status_code == 200
    data = response.json()
    assert len(data["files"]) == 2
    for f in data["files"]:
        assert "\\" not in f["path"]
