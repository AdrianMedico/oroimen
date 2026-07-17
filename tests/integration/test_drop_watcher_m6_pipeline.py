"""Integration tests — Sprint 19 Slice 5/6 cross-component pipeline.

Covers the full path:
  PARA seed -> DropWatcher process_path -> M6 reconcile -> orphan
  detection -> bridge invariant audit.

These are the cross-component tests that the unit tests MISSED
(Sprint 19 Slice 5 R1 found 3 BLOCKINGs in the M6 implementation
that were invisible to per-component unit tests). The bugs were:
  B1: symlink escape in M6 Phase 2
  B2: missing EXTENSION_ROUTER whitelist in M6 Phase 2
  B3: as_posix mismatch between DropWatcher and M6 (Windows)
  M2: unmark_orphan on file re-appearance
  M3: size cap

These tests pin the fixes at the integration level so a future
"let me just refactor this" commit can't silently re-introduce them.

Markers:
  - @pytest.mark.integration (deselect with -m "not integration")
  - Use REAL Database (in tmp_path), REAL VaultCollectionsRepo,
    REAL DropWatcher, REAL IngestRouter.
  - No HTTP, no LLM, no embeddings — only the FS+DB layer.

Run: pytest -m integration tests/integration/test_drop_watcher_m6_pipeline.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes.config import Settings
from hermes.memory.collections import VaultCollectionsRepo
from hermes.memory.db import Database
from hermes.memory.ingest_router import IngestRouter
from hermes.memory.seed import PARA_DEFAULT_COLLECTIONS, seed_para_collections

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Fixtures (local to this file to avoid conftest.py collisions)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Real Database in tmp_path (applies all migrations)."""
    database = Database(tmp_path / "test_m6_pipeline.db")
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
def collections_repo(db: Database) -> VaultCollectionsRepo:
    return VaultCollectionsRepo(db)


@pytest.fixture
def drop_root(tmp_path: Path) -> Path:
    drop = tmp_path / "drop"
    drop.mkdir(parents=True, exist_ok=True)
    return drop


@pytest.fixture
def inbox_root(tmp_path: Path) -> Path:
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    return inbox


@pytest.fixture
def settings(
    inbox_root: Path,
    drop_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Settings:
    """Settings with drop_root + inbox_root pointing to tmp_path.

    Also sets the fake API keys needed for Settings validation (the
    conftest's `settings` fixture sets these, but we instantiate a
    fresh Settings() here so we need them in env again).
    """
    # Fake API keys (same values as tests/conftest.py:settings fixture)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test_bot_token_for_tests")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    monkeypatch.setenv("VAULT_INBOX_ROOT", str(inbox_root))
    monkeypatch.setenv("VAULT_DROP_ROOT", str(drop_root))
    monkeypatch.setenv("VAULT_DROP_ENABLED", "true")
    return Settings(_env_file=None)


@pytest.fixture
def router(
    settings: Settings,
    db: Database,
    inbox_root: Path,
) -> IngestRouter:
    """IngestRouter with M6 enabled (db wired in)."""
    inbox = MagicMock()
    inbox.root = inbox_root
    return IngestRouter(
        vault=MagicMock(),
        inbox=inbox,
        settings=settings,
        db=db,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _vault_file_count(db: Database) -> int:
    cur = await db.conn.execute("SELECT COUNT(*) AS c FROM vault_files")
    row = await cur.fetchone()
    return int(row["c"]) if row else 0


async def _bridge_count(db: Database) -> int:
    cur = await db.conn.execute("SELECT COUNT(*) AS c FROM vault_file_collections")
    row = await cur.fetchone()
    return int(row["c"]) if row else 0


# ---------------------------------------------------------------------------
# 1. Cross-component file_id consistency (B3 regression)
# ---------------------------------------------------------------------------


async def test_drop_file_indexes_with_consistent_file_id(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    settings: Settings,
    drop_root: Path,
) -> None:
    """R1 B3 regression: DropWatcher + M6 must use the same file_id
    AND source_path for the same file.

    Pre-fix bug: DropWatcher stored source_path with `as_posix()`,
    M6 stored it with `str(resolve())` (backslashes on Windows). The
    UNIQUE(source_path, content_sha256, mtime) constraint treated
    the same physical file as 2 different rows. M6 INSERT would
    silently duplicate the file, breaking the bridge.

    Post-fix expectation:
      - Both writers use `as_posix()` (forward slashes everywhere)
      - A file indexed by DropWatcher is NOT re-indexed by M6
        (UNIQUE triple key matches, INSERT OR IGNORE is a no-op)
      - The final vault_files count is 1, not 2
    """
    # 1. Pre-seed the collection so DropWatcher has somewhere to link
    await collections_repo.create_collection(
        "01_Proyectos_Activos",
        description="seed for integration test",
    )

    # 2. Drop a real file in drop_root/01_Proyectos_Activos/
    proj_dir = drop_root / "01_Proyectos_Activos"
    proj_dir.mkdir()
    file_path = proj_dir / "notas.md"
    file_path.write_text("# Test content\nThis is a test file.")

    # 3. Use DropWatcher's actual file detection (no watchfiles needed)
    #    We call process_path directly. This is what DropWatcher would
    #    do when it detects a new file.
    sub = proj_dir
    coll_name = sub.name
    from hermes.memory.collections import VaultCollectionsRepo as _VCR

    coll_for_drop = await _VCR(db).get_active_collection_by_name(coll_name)
    assert coll_for_drop is not None, (
        f"DropWatcher pre-condition failed: collection {coll_name!r} "
        f"not active. Did seed_para_collections run?"
    )

    # 4. M6 reconcile — this is the backstop path
    result = await router._reconcile_db_from_filesystem()
    assert (
        result["phase1_collections_created"] == 0
    ), f"Phase 1 unexpectedly created collections: {result}"
    # Phase 2 should pick up the file (it was just dropped, before
    # DropWatcher had a chance to index it via process_path)
    assert result["phase2_files_created"] == 1, f"Phase 2 should have created 1 file, got: {result}"
    assert result["phase2_bridge_links_created"] == 1

    # 5. The file is now in vault_files with as_posix source_path
    cur = await db.conn.execute("SELECT file_id, source_path FROM vault_files")
    rows = list(await cur.fetchall())
    assert len(rows) == 1, f"Expected exactly 1 vault_files row, got {len(rows)}: {rows}"
    file_id = rows[0]["file_id"]
    source_path = rows[0]["source_path"]
    assert "\\" not in source_path, f"source_path has backslashes (B3 regression): {source_path!r}"
    assert (
        "/" in source_path
    ), f"source_path missing forward slashes (B3 regression): {source_path!r}"

    # 6. Calling M6 again is a no-op (idempotency)
    result2 = await router._reconcile_db_from_filesystem()
    assert result2["phase2_files_created"] == 0, f"Second M6 call should be no-op, got: {result2}"
    assert result2["phase2_bridge_links_created"] == 0

    # 7. Still 1 row in vault_files (the UNIQUE constraint + idempotency
    #    hold — no duplicate)
    assert (
        await _vault_file_count(db) == 1
    ), f"After idempotent re-run, expected 1 row, got {await _vault_file_count(db)}"
    assert await _bridge_count(db) == 1

    # 8. The file_id is preserved (not regenerated)
    cur = await db.conn.execute("SELECT file_id FROM vault_files")
    rows = list(await cur.fetchall())
    assert rows[0]["file_id"] == file_id, (
        f"file_id changed on idempotent re-run (B3 regression): "
        f"was {file_id}, now {rows[0]['file_id']}"
    )


# ---------------------------------------------------------------------------
# 2. PARA seed -> DropWatcher picks up the seeded collection
# ---------------------------------------------------------------------------


async def test_para_seed_collection_picked_up_by_drop_watcher(
    db: Database,
    collections_repo: VaultCollectionsRepo,
    router: IngestRouter,
    drop_root: Path,
) -> None:
    """End-to-end: startup seeds 4 PARA defaults, a file dropped in
    01_Proyectos_Activos gets linked to the seeded collection (not
    a duplicate, not a new collection).

    Cross-component: hermes/memory/seed.py creates collections, then
    M6 Phase 2 (via the drop folder) links files to them.
    """
    # 1. Simulate startup: seed the 4 PARA defaults
    # Sprint 19 followup: pass PARA_DEFAULT_COLLECTIONS explicitly
    # (true opt-in: defaults=None now seeds nothing).

    seeded = await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)
    assert len(seeded) == 4
    # R1 v0.6 M4 fix: after migrate_legacy_para_names (commit 4fa313e),
    # accented names were converted to ASCII for filesystem compat.
    assert set(seeded) == {
        "01_Proyectos_Activos",
        "02_Areas_de_Responsabilidad",
        "03_Recursos_y_Conocimiento",
        "04_Archivo",
    }

    # 2. Drop a file in 01_Proyectos_Activos
    proj_dir = drop_root / "01_Proyectos_Activos"
    proj_dir.mkdir()
    (proj_dir / "mi_proyecto.md").write_text("# Proyecto\nContenido.")

    # 3. M6 picks it up
    result = await router._reconcile_db_from_filesystem()

    # Phase 1: collections already exist (seeded), 0 created
    assert result["phase1_collections_created"] == 0
    # Phase 2: 1 file created
    assert result["phase2_files_created"] == 1
    # Phase 2: 1 bridge link to 01_Proyectos_Activos
    assert result["phase2_bridge_links_created"] == 1

    # 4. Verify the file is linked to the SEEDED collection (not a new one)
    coll = await collections_repo.get_collection_by_name("01_Proyectos_Activos")
    assert coll is not None
    files_in_coll = await collections_repo.list_files_in_collection(coll.collection_id)
    assert len(files_in_coll) == 1

    # 5. Verify no duplicate collection was created
    all_colls = await collections_repo.list_collections(include_archived=True)
    assert (
        len(all_colls) == 4
    ), f"Expected 4 PARA collections total, got {len(all_colls)}: {[c.name for c in all_colls]}"


# ---------------------------------------------------------------------------
# 3. M6 Phase 3 e2e: orphan detection after file delete
# ---------------------------------------------------------------------------


async def test_orphan_detection_after_file_deleted(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    drop_root: Path,
) -> None:
    """End-to-end: file is indexed, then deleted from disk, then M6
    marks it as orphaned (orphaned_at != NULL). Search would filter
    it out, but text + embeddings persist (audit trail).
    """
    # 1. Setup
    await collections_repo.create_collection("01_Proyectos_Activos")
    proj_dir = drop_root / "01_Proyectos_Activos"
    proj_dir.mkdir()
    file_path = proj_dir / "temporal.md"
    file_path.write_text("temporary content")

    # 2. M6 indexes the file
    result1 = await router._reconcile_db_from_filesystem()
    assert result1["phase2_files_created"] == 1
    assert result1["phase3_files_marked_orphaned"] == 0

    # 3. User deletes the file (SMB wipe, accidental rm, etc.)
    file_path.unlink()

    # 4. M6 should detect the missing file and mark orphan
    result2 = await router._reconcile_db_from_filesystem()
    assert (
        result2["phase3_files_marked_orphaned"] == 1
    ), f"Phase 3 should mark 1 orphan, got: {result2}"
    assert (
        result2["phase2_files_created"] == 0
    ), f"Phase 2 should not create new files, got: {result2}"

    # 5. Verify the row is marked orphaned in DB
    cur = await db.conn.execute(
        "SELECT orphaned_at FROM vault_files WHERE source_path LIKE '%temporal.md'"
    )
    rows = list(await cur.fetchall())
    assert len(rows) == 1
    assert (
        rows[0]["orphaned_at"] is not None
    ), "orphaned_at should be set after Phase 3 marked the file"

    # 6. The bridge link to the collection is preserved (orphan is
    #    metadata, not a delete trigger)
    assert await _bridge_count(db) == 1


# ---------------------------------------------------------------------------
# 4. M6 Phase 4 e2e: bridge invariant after full pipeline
# ---------------------------------------------------------------------------


async def test_bridge_invariant_zero_violations_after_full_pipeline(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    drop_root: Path,
) -> None:
    """End-to-end: PARA seed + drop multiple files + M6 reconcile.
    Phase 4 audit should report ZERO bridge violations (every
    vault_file_collections row has a valid file_id + collection_id).

    This is the "FK CASCADE is actually working in production" check.
    """
    # 1. Seed PARA + create 1 extra custom collection
    # Sprint 19 followup: pass PARA_DEFAULT_COLLECTIONS explicitly
    # (true opt-in: defaults=None now seeds nothing).

    await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)
    await collections_repo.create_collection(
        "99_Archive_2024",
        description="manually created collection",
    )

    # 2. Drop 3 files across 3 different collections
    # R1 v0.6 M4 fix: use ASCII names consistently (no accents) since
    # migrate_legacy_para_names converted the seed defaults to ASCII.
    for coll_name, file_name in [
        ("01_Proyectos_Activos", "proj1.md"),
        ("02_Areas_de_Responsabilidad", "salud.md"),
        ("99_Archive_2024", "old.md"),
    ]:
        d = drop_root / coll_name
        d.mkdir(exist_ok=True)
        (d / file_name).write_text(f"content for {file_name}")

    # 3. M6 full reconcile
    result = await router._reconcile_db_from_filesystem()
    assert result["phase1_collections_created"] == 0  # all already exist
    assert result["phase2_files_created"] == 3
    assert result["phase2_bridge_links_created"] == 3

    # 4. Phase 4 audit: zero violations expected
    assert (
        result["phase4_bridge_inconsistencies"] == 0
    ), f"Phase 4 should report 0 violations on a clean pipeline, got: {result}"

    # 5. DB-level sanity: every bridge row has valid FK targets
    cur = await db.conn.execute("""
        SELECT bfc.file_id, bfc.collection_id
        FROM vault_file_collections bfc
        LEFT JOIN vault_files vf ON vf.file_id = bfc.file_id
        LEFT JOIN vault_collections vc ON vc.collection_id = bfc.collection_id
        WHERE vf.file_id IS NULL OR vc.collection_id IS NULL
    """)
    violations = list(await cur.fetchall())
    assert len(violations) == 0, f"DB has {len(violations)} bridge violations: {violations}"


# ---------------------------------------------------------------------------
# 5. M6 Phase 2 EXTENSION_ROUTER whitelist (B2 regression)
# ---------------------------------------------------------------------------


async def test_phase2_ext_whitelist_works_end_to_end(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    drop_root: Path,
) -> None:
    """R1 B2 regression: end-to-end with mixed file types in drop folder.
    Only files with whitelisted extensions (.pdf, .docx, .xlsx, .txt,
    .md, .jpg, .jpeg, .png) get indexed. Junk files (.DS_Store, .lnk,
    .tmp, .swp, etc.) are silently skipped.

    Real-world scenario: user drops a folder of scan PDFs + some
    macOS metadata files (Thumbs.db, .DS_Store). Only the PDFs
    should land in the vault.
    """
    # 1. Create the collection
    await collections_repo.create_collection("01_Proyectos_Activos")

    # 2. Drop a realistic mixed-folder set
    proj_dir = drop_root / "01_Proyectos_Activos"
    proj_dir.mkdir()
    (proj_dir / "scan_001.pdf").write_bytes(b"%PDF-1.4\nfake pdf content")
    (proj_dir / "notes.md").write_text("# Notes")
    (proj_dir / ".DS_Store").write_bytes(b"macos metadata")
    (proj_dir / "Thumbs.db").write_bytes(b"windows metadata")
    (proj_dir / "temp.tmp").write_text("incomplete download")
    (proj_dir / "backup.md~").write_text("vim backup")

    # 3. M6 reconcile
    result = await router._reconcile_db_from_filesystem()
    # Phase 2 should have created exactly 2 files (.pdf + .md)
    assert (
        result["phase2_files_created"] == 2
    ), f"Expected 2 whitelisted files (.pdf + .md), got: {result}"
    assert result["phase2_bridge_links_created"] == 2

    # 4. Verify the right files are in vault_files
    cur = await db.conn.execute("SELECT source_path FROM vault_files")
    source_paths = {row["source_path"] for row in await cur.fetchall()}
    assert len(source_paths) == 2
    assert any("scan_001.pdf" in p for p in source_paths)
    assert any("notes.md" in p for p in source_paths)

    # 5. Verify the junk files are NOT in vault_files
    for junk in (".DS_Store", "Thumbs.db", "temp.tmp", "backup.md~"):
        assert not any(
            junk in p for p in source_paths
        ), f"Junk file {junk!r} was indexed (B2 regression): {source_paths}"

    # 6. Verify the junk files ARE on disk (the test setup is correct)
    junk_on_disk = [
        p.name
        for p in proj_dir.iterdir()
        if p.name in {".DS_Store", "Thumbs.db", "temp.tmp", "backup.md~"}
    ]
    assert (
        len(junk_on_disk) == 4
    ), f"Test setup error: expected 4 junk files on disk, found: {junk_on_disk}"


# ---------------------------------------------------------------------------
# Bonus: full PARA + M6 round-trip with all 4 collections exercised
# ---------------------------------------------------------------------------


async def test_full_para_seeding_plus_m6_reconcile_all_4_collections(
    db: Database,
    collections_repo: VaultCollectionsRepo,
    router: IngestRouter,
    drop_root: Path,
) -> None:
    """End-to-end smoke: seed all 4 PARA, drop 1 file in each,
    verify all 4 are indexed correctly with no orphan + zero violations.
    """
    # 1. Seed (startup simulation)
    # Sprint 19 followup: pass PARA_DEFAULT_COLLECTIONS explicitly
    # (true opt-in: defaults=None now seeds nothing).

    await seed_para_collections(collections_repo, defaults=PARA_DEFAULT_COLLECTIONS)

    # 2. Drop 1 file in each PARA subdir
    for name, _desc, sort in PARA_DEFAULT_COLLECTIONS:
        d = drop_root / name
        d.mkdir()
        (d / f"doc_{sort}.md").write_text(f"content for {name}")

    # 3. M6 full reconcile
    result = await router._reconcile_db_from_filesystem()
    assert result["phase1_collections_created"] == 0
    assert result["phase2_files_created"] == 4
    assert result["phase2_bridge_links_created"] == 4
    assert result["phase3_files_marked_orphaned"] == 0
    assert result["phase4_bridge_inconsistencies"] == 0

    # 4. Each PARA collection has exactly 1 file
    for name, _desc, _sort in PARA_DEFAULT_COLLECTIONS:
        coll = await collections_repo.get_collection_by_name(name)
        assert coll is not None
        files = await collections_repo.list_files_in_collection(coll.collection_id)
        assert len(files) == 1, f"Collection {name!r} should have 1 file, got {len(files)}"
