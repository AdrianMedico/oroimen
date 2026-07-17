"""Tests for `hermes.memory.drop_watcher.DropWatcher` (Sprint 19 Slice 4 §4).

TDD-first: 6 tests covering the contract (Gemini 3.1 Pro reviewed 2026-07-10):
- extension whitelist enforced
- SHA-256 file_id (not random UUID)
- collection auto-create
- vault_file_collections link
- manifest written
- idempotent on restart
- skips drop-root files
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from hermes.memory.collections import VaultCollectionsRepo
from hermes.memory.db import Database
from hermes.memory.drop_watcher import (
    ALLOWED_EXTENSIONS,
    DropWatcher,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def db(tmp_path) -> AsyncGenerator[Database, None]:
    d = Database(tmp_path / "test_drop.db")
    await d.initialize()
    try:
        yield d
    finally:
        await d.close()


@pytest.fixture
def collections_repo(db: Database) -> VaultCollectionsRepo:
    return VaultCollectionsRepo(db)


@pytest.fixture
def drop_root(tmp_path: Path) -> Path:
    p = tmp_path / "drop"
    p.mkdir()
    return p


@pytest.fixture
def watcher(
    db: Database,
    collections_repo: VaultCollectionsRepo,
    drop_root: Path,
) -> DropWatcher:
    return DropWatcher(
        db=db,
        collections_repo=collections_repo,
        drop_root=drop_root,
    )


# --- Extension whitelist --------------------------------------------------


async def test_drop_watcher_skips_unknown_extension_ds_store(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """`.DS_Store` (macOS) is NOT in the whitelist → silent skip with log."""
    ds = drop_root / "01_Proyectos" / ".DS_Store"
    ds.parent.mkdir()
    ds.write_bytes(b"\x00\x01\x02")
    result = await watcher.process_path(ds)
    assert result.action == "skipped_unknown_ext"
    # File should NOT be in vault_files
    async with watcher._db.conn.execute("SELECT COUNT(*) as cnt FROM vault_files") as cur:
        row = await cur.fetchone()
    assert row["cnt"] == 0


async def test_drop_watcher_accepts_all_allowed_extensions(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """All 8 extensions in ALLOWED_EXTENSIONS are accepted by the watcher."""
    for ext in ALLOWED_EXTENSIONS:
        sub = drop_root / f"sub_{ext.strip('.')}" / f"file{ext}"
        sub.parent.mkdir(exist_ok=True)
        sub.write_bytes(b"some content " + ext.encode())
    # Walk all files
    accepted: list[str] = []
    for f in drop_root.rglob("*"):
        if f.is_file():
            result = await watcher.process_path(f)
            if result.action in {"inserted", "linked_existing"}:
                accepted.append(f.suffix)
    assert sorted(accepted) == sorted(ALLOWED_EXTENSIONS)


async def test_extension_whitelist_is_case_insensitive(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """Extension whitelist is case-insensitive (report.PDF passes)."""
    f = drop_root / "01_Proyectos" / "REPORT.PDF"
    f.parent.mkdir()
    f.write_bytes(b"%PDF-1.4 fake")
    result = await watcher.process_path(f)
    assert result.action == "inserted"


# --- SHA-256 file_id contract (Gemini feedback) --------------------------


async def test_drop_watcher_uses_sha256_file_id_not_random_uuid(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """`file_id` in vault_files is the SHA-256 truncation, NOT a UUID.

    The contract: same content = same file_id, always. Random UUIDs
    would break idempotency on container restart and duplicate identical
    embedding blocks.
    """
    f = drop_root / "01_Proyectos" / "doc.pdf"
    f.parent.mkdir()
    f.write_bytes(b"contract content for sha256 test")
    result = await watcher.process_path(f)
    assert result.action == "inserted"
    # file_id must be 32 lowercase hex chars
    assert len(result.file_id) == 32
    assert all(c in "0123456789abcdef" for c in result.file_id)
    # The file_id must equal the SHA-256 truncation
    from hermes.memory.file_id import file_id_from_path

    assert result.file_id == file_id_from_path(f)


async def test_drop_watcher_idempotent_on_restart(watcher: DropWatcher, drop_root: Path) -> None:
    """Calling process_path twice for the same file does NOT create a
    duplicate row in vault_files. Action is 'linked_existing' on 2nd call."""
    f = drop_root / "01_Proyectos" / "doc.pdf"
    f.parent.mkdir()
    f.write_bytes(b"contract content")

    result1 = await watcher.process_path(f)
    assert result1.action == "inserted"

    result2 = await watcher.process_path(f)
    assert result2.action == "linked_existing"
    assert result2.file_id == result1.file_id  # same ID, not a new one

    # DB has exactly 1 row
    async with watcher._db.conn.execute("SELECT COUNT(*) as cnt FROM vault_files") as cur:
        row = await cur.fetchone()
    assert row["cnt"] == 1


# --- Collection auto-create + link ---------------------------------------


async def test_drop_watcher_creates_collection_for_new_subdir(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """A file in `drop/<new_subdir>/` auto-creates a vault_collection."""
    f = drop_root / "fresh_subdir" / "doc.pdf"
    f.parent.mkdir()
    f.write_bytes(b"hello")
    result = await watcher.process_path(f)
    assert result.action == "inserted"
    assert result.collection_name == "fresh_subdir"
    # Collection exists in DB
    coll = await watcher._collections_repo.get_collection_by_name("fresh_subdir")
    assert coll is not None
    assert coll.archived is False
    assert coll.parent_collection_id is None


async def test_drop_watcher_links_file_to_collection(watcher: DropWatcher, drop_root: Path) -> None:
    """File is linked to its collection in vault_file_collections."""
    f = drop_root / "01_Proyectos_Activos" / "factura.pdf"
    f.parent.mkdir()
    f.write_bytes(b"factura content")
    result = await watcher.process_path(f)
    assert result.action == "inserted"
    # bridge row exists
    async with watcher._db.conn.execute(
        "SELECT COUNT(*) as cnt FROM vault_file_collections WHERE file_id = ? AND collection_id = ?",
        (result.file_id, result.collection_id),
    ) as cur:
        row = await cur.fetchone()
    assert row["cnt"] == 1


# --- Manifest write -------------------------------------------------------


async def test_drop_watcher_writes_manifest(watcher: DropWatcher, drop_root: Path) -> None:
    """Watcher writes `<file>.md.json` next to the file with file_id + path."""
    f = drop_root / "01_Proyectos" / "doc.pdf"
    f.parent.mkdir()
    f.write_bytes(b"hello")
    result = await watcher.process_path(f)
    manifest_file = f.with_suffix(f.suffix + ".json")
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    # Contract (Sprint 19 R1 integration check): process_inbox reads
    # `vault_file_id`, NOT `id`. If we write only `id`, process_inbox
    # silently drops the file. We write both for backwards-compat
    # (id is the alias, vault_file_id is canonical).
    assert manifest["vault_file_id"] == result.file_id
    assert manifest["id"] == result.file_id  # alias
    assert manifest["path"] == f.as_posix()
    assert "created_at" in manifest


# --- Drop root files ------------------------------------------------------


async def test_drop_watcher_skips_root_level_file(watcher: DropWatcher, drop_root: Path) -> None:
    """Files at `drop/foo.pdf` (no subdir) are rejected with skipped_root."""
    f = drop_root / "loose.pdf"
    f.write_bytes(b"stray file")
    result = await watcher.process_path(f)
    assert result.action == "skipped_root"


async def test_drop_watcher_routes_root_level_to_default_collection(
    tmp_path, db: Database, collections_repo: VaultCollectionsRepo, drop_root: Path
) -> None:
    """Sprint 19 followup: when default_collection is set, root-level
    files in drop_root get routed to that collection instead of
    being skipped.

    True opt-in: setting default_collection enables the behavior.
    Without it (test_drop_watcher_skips_root_level_file), the
    legacy skip behavior is preserved.
    """
    from hermes.memory.drop_watcher import DropWatcher

    dw = DropWatcher(
        db=db,
        collections_repo=collections_repo,
        drop_root=drop_root,
        default_collection="_inbox",
    )

    f = drop_root / "loose.pdf"
    f.write_bytes(b"loose file at drop root")
    result = await dw.process_path(f)

    # File inserted, not skipped
    assert (
        result.action == "inserted"
    ), f"Expected inserted (default collection route), got {result.action}"
    # Linked to the default collection
    assert result.collection_name == "_inbox"

    # Verify the row in the DB
    coll = await collections_repo.get_active_collection_by_name("_inbox")
    assert coll is not None
    async with db.conn.execute(
        "SELECT collection_id FROM vault_file_collections "
        "WHERE file_id = ? AND superseded_at IS NULL",
        (result.file_id,),
    ) as cur:
        bridges = await cur.fetchall()
    assert len(bridges) == 1
    assert bridges[0]["collection_id"] == coll.collection_id


async def test_drop_watcher_default_collection_existing_is_reused(
    tmp_path, db: Database, collections_repo: VaultCollectionsRepo, drop_root: Path
) -> None:
    """Sprint 19 followup: if the default_collection already exists
    in the DB, route to it (idempotent, no duplicate)."""
    from hermes.memory.drop_watcher import DropWatcher

    # Pre-create the default collection
    await collections_repo.create_collection(name="_inbox", description="Pre-existing inbox")

    dw = DropWatcher(
        db=db,
        collections_repo=collections_repo,
        drop_root=drop_root,
        default_collection="_inbox",
    )

    f = drop_root / "another_loose.pdf"
    f.write_bytes(b"another loose file")
    result = await dw.process_path(f)
    assert result.action == "inserted"
    assert result.collection_name == "_inbox"

    # Only 1 _inbox collection (not duplicated)
    inbox = await collections_repo.get_active_collection_by_name("_inbox")
    assert inbox is not None
    assert inbox.description == "Pre-existing inbox"  # original preserved


async def test_drop_watcher_skips_manifest_file(watcher: DropWatcher, drop_root: Path) -> None:
    """`*.md.json` files are skipped (they're watcher-written manifests)."""
    f = drop_root / "01_Proyectos" / "doc.pdf.md.json"
    f.parent.mkdir()
    f.write_text('{"id": "fake"}')
    result = await watcher.process_path(f)
    assert result.action == "skipped_manifest"


# --- scan_existing on startup ---------------------------------------------


async def test_scan_existing_processes_existing_files(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """scan_existing() picks up files already in drop/ at startup."""
    f = drop_root / "01_Proyectos" / "stale.pdf"
    f.parent.mkdir()
    f.write_bytes(b"stale file from previous run")
    results = await watcher.scan_existing()
    assert len(results) == 1
    assert results[0].action == "inserted"
    assert results[0].file_path == f.as_posix()


# --- Fix 1: Path traversal rejection (LLM feedback) ---------------------


async def test_drop_watcher_rejects_file_outside_drop_root(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """Symlink pointing OUTSIDE drop_root is rejected.

    Defense-in-depth: watchfiles should not emit events for paths outside
    the watch root, but a symlink could be a sneaky way to escape. The
    watcher resolves the file and checks `is_relative_to(drop_root)`.
    """
    outside = drop_root.parent / "evil.pdf"
    outside.write_bytes(b"data outside drop")
    try:
        link = drop_root / "01_Proyectos" / "evil_link.pdf"
        link.parent.mkdir()
        link.symlink_to(outside)
        result = await watcher.process_path(link)
        assert result.action == "skipped_outside_drop"
        # No vault_files row created
        async with watcher._db.conn.execute("SELECT COUNT(*) as cnt FROM vault_files") as cur:
            row = await cur.fetchone()
        assert row["cnt"] == 0
    finally:
        outside.unlink(missing_ok=True)


async def test_drop_watcher_accepts_dotdot_path_that_resolves_under_drop(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """Path with '..' that resolves to a location INSIDE drop_root is OK.

    The '..' just normalizes; it doesn't escape. The real protection is
    `resolve().is_relative_to(drop_root)`. If a file at
    `drop_root/sub/../file.pdf` exists, it IS under drop_root (resolves
    to `drop_root/file.pdf`) and should be accepted.
    """
    # Create the file at the normalized location first
    f = drop_root / "01_Proyectos" / "real.pdf"
    f.parent.mkdir()
    f.write_bytes(b"legit file")
    # Process with the '..' in the path
    traversal = drop_root / "01_Proyectos" / ".." / "01_Proyectos" / "real.pdf"
    result = await watcher.process_path(traversal)
    # Should be accepted — resolved path is under drop_root
    assert result.action == "inserted"
    assert result.file_id != ""


# --- Fix 2: File moved/removed during processing ------------------------


async def test_drop_watcher_skips_disappeared_file(watcher: DropWatcher, drop_root: Path) -> None:
    """File is detected then moved/removed before hash → skipped gracefully.

    Race condition: watchfiles fires 'added', but by the time process_path
    runs, the file is gone. We must not crash; just skip with a log.
    """
    f = drop_root / "01_Proyectos" / "ghost.pdf"
    f.parent.mkdir()
    f.write_bytes(b"transient content")
    # Simulate the race: delete the file between event and hash
    f.unlink()
    result = await watcher.process_path(f)
    assert result.action == "skipped_disappeared"
    # No row in DB
    async with watcher._db.conn.execute("SELECT COUNT(*) as cnt FROM vault_files") as cur:
        row = await cur.fetchone()
    assert row["cnt"] == 0


# --- Fix 3 + 4: SHA-256 single-pass + empty file handling ---------------


async def test_drop_watcher_handles_empty_file(watcher: DropWatcher, drop_root: Path) -> None:
    """Empty file (0 bytes) is processed with deterministic file_id."""
    f = drop_root / "01_Proyectos" / "empty.txt"
    f.parent.mkdir()
    f.write_bytes(b"")
    result = await watcher.process_path(f)
    assert result.action == "inserted"
    # SHA-256 of empty bytes truncated to 32 chars
    assert result.file_id == "e3b0c44298fc1c149afbf4c8996fb924"


# --- Fix 5: Collection archived → skip ---------------------------------


async def test_drop_watcher_skips_archived_collection(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """File in a subdir whose collection is archived is skipped.

    Reason: archived collections are JOIN-filtered out in queries, so
    linking new files to them would make the files invisible. Better to
    skip with warning so the user can un-archive or rename the subdir.
    """
    # Create + archive a collection
    coll = await watcher._collections_repo.create_collection(name="01_Proyectos")
    await watcher._collections_repo.archive_collection(coll.collection_id, cascade=True)

    f = drop_root / "01_Proyectos" / "factura.pdf"
    f.parent.mkdir()
    f.write_bytes(b"new file in archived collection")
    result = await watcher.process_path(f)
    assert result.action == "skipped_archived_collection"
    assert result.collection_id == coll.collection_id
    # No file_id is associated (the file is skipped entirely)
    # No row in vault_files
    async with watcher._db.conn.execute("SELECT COUNT(*) as cnt FROM vault_files") as cur:
        row = await cur.fetchone()
    assert row["cnt"] == 0


# --- Fix 6: File replaced (same path, different content) ---------------


async def test_drop_watcher_replaces_file_with_different_content(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """User replaces file at same path with different content → replaced.

    The old vault_files row is deleted (cascading to vault_file_collections),
    a new row is inserted with the new file_id. action='replaced'.
    """
    f = drop_root / "01_Proyectos" / "doc.pdf"
    f.parent.mkdir()
    f.write_bytes(b"version 1 content")
    result1 = await watcher.process_path(f)
    assert result1.action == "inserted"
    old_file_id = result1.file_id

    # Replace with new content
    f.write_bytes(b"version 2 content, completely different")
    result2 = await watcher.process_path(f)
    assert result2.action == "replaced"
    new_file_id = result2.file_id
    assert new_file_id != old_file_id

    # Exactly 1 row in vault_files (the new one)
    async with watcher._db.conn.execute("SELECT COUNT(*) as cnt FROM vault_files") as cur:
        row = await cur.fetchone()
    assert row["cnt"] == 1
    # The row has the new file_id
    async with watcher._db.conn.execute(
        "SELECT file_id FROM vault_files WHERE source_path = ?", (f.as_posix(),)
    ) as cur:
        row = await cur.fetchone()
    assert row["file_id"] == new_file_id

    # Bridge: still links to the new file_id
    async with watcher._db.conn.execute("SELECT file_id FROM vault_file_collections") as cur:
        rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["file_id"] == new_file_id


async def test_drop_watcher_linked_existing_when_content_unchanged(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """Same content re-processed (e.g., file re-touched) → linked_existing.

    The file_id is the same, the row is updated with new mtime/size, and
    action='linked_existing'. This is the happy idempotent path.
    """
    f = drop_root / "01_Proyectos" / "doc.pdf"
    f.parent.mkdir()
    f.write_bytes(b"stable content")
    result1 = await watcher.process_path(f)
    assert result1.action == "inserted"
    # Re-touch (write same content) — simulates `touch` or git pull
    import time

    time.sleep(0.01)
    f.write_bytes(b"stable content")
    result2 = await watcher.process_path(f)
    assert result2.action == "linked_existing"
    assert result2.file_id == result1.file_id


# --- New active-collection repo method --------------------------------


async def test_get_active_collection_by_name_filters_archived(
    db: Database,
    collections_repo: VaultCollectionsRepo,
) -> None:
    """`get_active_collection_by_name` returns None for archived collections.

    Used by the DropWatcher to skip files in archived subdirs.
    """
    coll = await collections_repo.create_collection(name="01_Proyectos")
    # Active: returns the collection
    active = await collections_repo.get_active_collection_by_name("01_Proyectos")
    assert active is not None
    assert active.collection_id == coll.collection_id
    # Archive it
    await collections_repo.archive_collection(coll.collection_id, cascade=True)
    # Now active returns None
    assert await collections_repo.get_active_collection_by_name("01_Proyectos") is None
    # But get_collection_by_name (no filter) still returns the archived one
    archived = await collections_repo.get_collection_by_name("01_Proyectos")
    assert archived is not None
    assert archived.archived is True


# --- Sprint 19 Slice 4d v2: M1 fix - inline move detection (R1 v0.6 M1) ---


async def test_process_path_detects_move_across_collections(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """A file moved to a new subdir is detected as 'moved' (not
    'linked_existing') and its vault_file_collections bridge is rebuilt
    via _move_bridge (old → new).

    R1 v0.6 M1: previously _move_bridge was defined but never called
    from production. This test proves the watcher now detects moves
    end-to-end.
    """
    # 1. Create file in 01_Proyectos/foo.pdf
    old_dir = drop_root / "01_Proyectos"
    old_dir.mkdir()
    foo_old = old_dir / "foo.pdf"
    foo_old.write_bytes(b"the same content")
    result1 = await watcher.process_path(foo_old)
    assert result1.action == "inserted"
    file_id = result1.file_id
    assert file_id

    # 2. Move file to 02_Areas/foo.pdf (same content, different path)
    new_dir = drop_root / "02_Areas"
    new_dir.mkdir()
    foo_new = new_dir / "foo.pdf"
    foo_new.write_bytes(b"the same content")
    # The old_dir is now empty, but the manifest.json from the first
    # process_path may still be there. Clean it up.
    for child in old_dir.iterdir():
        if child.is_file():
            child.unlink()
    old_dir.rmdir()

    result2 = await watcher.process_path(foo_new)
    # 3. Result must be 'moved' (not 'linked_existing' or 'replaced')
    assert result2.action == "moved", f"expected 'moved', got {result2.action}"
    assert result2.file_id == file_id

    # 4. vault_files row was updated (source_path = new path, file_id unchanged)
    async with watcher._db.conn.execute(
        "SELECT source_path FROM vault_files WHERE file_id = ?", (file_id,)
    ) as cur:
        row = await cur.fetchone()
    assert row["source_path"] == foo_new.as_posix()

    # 5. Active bridge points to the NEW collection only
    async with watcher._db.conn.execute(
        "SELECT collection_id FROM vault_file_collections "
        "WHERE file_id = ? AND superseded_at IS NULL",
        (file_id,),
    ) as cur:
        active_bridges = await cur.fetchall()
    assert len(active_bridges) == 1
    new_coll = await watcher._collections_repo.get_active_collection_by_name("02_Areas")
    assert active_bridges[0]["collection_id"] == new_coll.collection_id

    # 6. History: old bridge exists with superseded_at set
    async with watcher._db.conn.execute(
        "SELECT collection_id, superseded_at FROM vault_file_collections "
        "WHERE file_id = ? AND superseded_at IS NOT NULL",
        (file_id,),
    ) as cur:
        superseded = await cur.fetchall()
    assert len(superseded) == 1
    old_coll = await watcher._collections_repo.get_active_collection_by_name("01_Proyectos")
    assert superseded[0]["collection_id"] == old_coll.collection_id
    assert superseded[0]["superseded_at"] is not None


async def test_process_path_ignores_move_to_same_collection(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """A file re-touched in the same collection is 'linked_existing' (not moved)."""
    d = drop_root / "01_Proyectos"
    d.mkdir()
    f = d / "foo.pdf"
    f.write_bytes(b"content")
    r1 = await watcher.process_path(f)
    assert r1.action == "inserted"

    # Re-touch (mtime changes, content same, same path)
    import time

    time.sleep(0.01)
    f.write_bytes(b"content")
    r2 = await watcher.process_path(f)
    assert r2.action == "linked_existing"
    assert r2.file_id == r1.file_id


async def test_scan_existing_includes_monitor_roots(
    tmp_path,
) -> None:
    """scan_existing() iterates monitor_roots too (R1 v0.6 M2 fix).

    Previously, scan_existing only iterated _drop_root, so files added
    to monitor_roots during downtime were missed. Restart recovery
    must cover monitor_roots explicitly.
    """
    from hermes.memory.collections import VaultCollectionsRepo
    from hermes.memory.db import Database

    db_path = tmp_path / "test.db"
    drop_root = tmp_path / "drop"
    monitor_root = tmp_path / "monitor"
    drop_root.mkdir()
    monitor_root.mkdir()

    db = Database(db_path)
    await db.initialize()
    collections_repo = VaultCollectionsRepo(db)
    dw = DropWatcher(
        db=db,
        collections_repo=collections_repo,
        drop_root=drop_root,
        monitor_roots=[monitor_root],
    )

    # Drop a file in the monitor_root (not in drop_root)
    f = monitor_root / "01_Proyectos" / "stale.pdf"
    f.parent.mkdir()
    f.write_bytes(b"from monitor root")

    results = await dw.scan_existing()
    # The file in monitor_root should be picked up
    assert len(results) == 1
    assert results[0].action == "inserted"
    assert results[0].file_path == f.as_posix()

    await db.close()


async def test_record_dropped_event_writes_correct_columns(
    watcher: DropWatcher, drop_root: Path
) -> None:
    """_record_dropped_event writes source_path + detected_at (R1 v0.6 B1 fix).

    Previously, the INSERT used (source_path, reason, created_at) but the
    v23 schema has (event_id, source_path, detected_at, processed_at).
    The error was silently caught. Now it writes the correct columns.
    """
    await watcher._record_dropped_event(Path("/tmp/test.pdf"), "queue_full")
    async with watcher._db.conn.execute(
        "SELECT source_path, detected_at FROM dropped_events"
    ) as cur:
        rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["source_path"] == "/tmp/test.pdf"
    assert rows[0]["detected_at"] is not None
async def test_process_path_serializes_concurrent_writes(
    watcher: DropWatcher,
    collections_repo: VaultCollectionsRepo,
    drop_root: Path,
) -> None:
    """Parallel workers share Database._write_lock around BEGIN IMMEDIATE."""
    await collections_repo.create_collection(name="concurrent", description="pre-created")
    folder = drop_root / "concurrent"
    folder.mkdir()
    files: list[Path] = []
    for index in range(8):
        path = folder / f"document-{index}.txt"
        path.write_text(f"unique concurrent content {index}", encoding="utf-8")
        files.append(path)

    results = await asyncio.gather(*(watcher.process_path(path) for path in files))

    assert {result.action for result in results} == {"inserted"}
    async with watcher._db.conn.execute("SELECT COUNT(*) AS cnt FROM vault_files") as cur:
        files_row = await cur.fetchone()
    async with watcher._db.conn.execute(
        "SELECT COUNT(*) AS cnt FROM vault_file_collections WHERE superseded_at IS NULL"
    ) as cur:
        bridges_row = await cur.fetchone()
    assert files_row["cnt"] == 8
    assert bridges_row["cnt"] == 8
    assert not watcher._db._write_lock.locked()
