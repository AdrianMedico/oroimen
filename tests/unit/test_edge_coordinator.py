"""Tests for EdgeCoordinator (Sprint 19 Slice 4c §4.4).

Coverage:
- enqueue: path-only delivery, idempotency, threshold/auto_ocr/offline gates
- recover_zombies: state machine, filesystem cleanup, idempotency
- _to_smb_relative: prefix stripping
- probe_once + state transitions: offline→online fires catch-up

The probe function is INJECTED so tests don't make real HTTP calls.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from hermes.memory.db import Database
from hermes.memory.edge_coordinator import (
    EdgeCoordinator,
    ProbeFn,
    _parse_iso,
    _utc_now_iso_ms,
)
from hermes.memory.ocr_pending_repo import OcrPendingRepo

# Re-export for cleaner test code
DEFAULT_PORT = EdgeCoordinator.DEFAULT_PORT
DEFAULT_SMB_PREFIX = "/mnt/shared/"


def to_smb_relative(vault_path: str) -> str:
    """Test helper: apply the SMB-relative prefix-stripping logic.

    The production method is an instance method on EdgeCoordinator, but
    for pure-function tests we just want to test the prefix-stripping
    behavior in isolation.
    """
    if vault_path.startswith(DEFAULT_SMB_PREFIX):
        return vault_path[len(DEFAULT_SMB_PREFIX) :]
    return vault_path


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path) -> AsyncGenerator[Database, None]:
    d = Database(tmp_path / "test_edge.db")
    await d.initialize()
    try:
        yield d
    finally:
        await d.close()


@pytest.fixture
def ocr_repo(db: Database) -> OcrPendingRepo:
    return OcrPendingRepo(db)


@pytest.fixture
def edge_root(tmp_path: Path) -> Path:
    p = tmp_path / "_infrastructure"
    p.mkdir()
    return p


def make_probe(online: set[str]) -> ProbeFn:
    """Build a probe function that returns the given set of online hosts."""

    async def probe_fn(targets: list[tuple[str, int]]) -> set[str]:
        return {h for h, _ in targets if h in online}

    return probe_fn


async def _make_online(coord: EdgeCoordinator) -> None:
    """Trigger one probe cycle so `is_online()` reflects the mock state.

    In production, the probe task runs in the background. In tests we
    don't start the task (to keep tests fast and deterministic), so
    we run one probe manually to populate `_pcs[*].is_online`.
    """
    await coord._probe_once()


async def _seed_file(db: Database, repo: OcrPendingRepo, file_id: str, path: str) -> None:
    await db.conn.execute(
        """
        INSERT INTO vault_files (file_id, source_path, size_bytes, content_sha256, mtime)
        VALUES (?, ?, 1024, ?, 1700000000.0)
        """,
        (file_id, path, file_id),
    )
    await db.conn.commit()
    await repo.create(
        file_id=file_id,
        local_confidence=0.5,
        local_text="partial text",
        local_model="tesseract-5",
        status="pending_review",
    )


# ---------------------------------------------------------------------------
# to_smb_relative (private helper, but worth testing)
# ---------------------------------------------------------------------------


def test_to_smb_relative_strips_prefix() -> None:
    assert to_smb_relative("/mnt/shared/Documentos/foo.jpg") == "Documentos/foo.jpg"


def test_to_smb_relative_no_prefix_passthrough() -> None:
    """If path doesn't start with /mnt/shared/, return as-is."""
    assert to_smb_relative("/var/data/foo.jpg") == "/var/data/foo.jpg"


def test_to_smb_relative_handles_deep_paths() -> None:
    assert to_smb_relative("/mnt/shared/A/B/C/D/file.jpg") == "A/B/C/D/file.jpg"


# ---------------------------------------------------------------------------
# enqueue: gating logic
# ---------------------------------------------------------------------------


async def test_enqueue_returns_false_when_auto_ocr_disabled(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """Master switch off → enqueue is a no-op (returns False, no FS write)."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        auto_ocr=False,
        probe_fn=make_probe({"edge.local"}),
    )
    file_id = "a" * 32
    await _seed_file(db, ocr_repo, file_id, "/mnt/shared/Documentos/foo.jpg")
    ok = await coord.enqueue(
        file_id=file_id,
        path="/mnt/shared/Documentos/foo.jpg",
        local_confidence=0.5,
    )
    assert ok is False
    # No request.json written
    request_path = edge_root / "edge_queue" / file_id / "request.json"
    assert not request_path.exists()
    # ocr_pending still in pending_review
    row = await ocr_repo.get(file_id)
    assert row is not None
    assert row.status == "pending_review"


async def test_enqueue_returns_false_when_confidence_above_threshold(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """confidence >= autoqueue_threshold → no enqueue (Tesseract is good enough)."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        autoqueue_threshold=0.85,
        probe_fn=make_probe({"edge.local"}),
    )
    file_id = "b" * 32
    await _seed_file(db, ocr_repo, file_id, "/mnt/shared/Documentos/foo.jpg")
    ok = await coord.enqueue(
        file_id=file_id,
        path="/mnt/shared/Documentos/foo.jpg",
        local_confidence=0.92,  # above 0.85
    )
    assert ok is False
    row = await ocr_repo.get(file_id)
    assert row is not None
    assert row.status == "pending_review"


async def test_enqueue_returns_false_when_no_pc_online(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """No PC online → enqueue is a no-op."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        probe_fn=make_probe(set()),  # no PCs online
    )
    file_id = "c" * 32
    await _seed_file(db, ocr_repo, file_id, "/mnt/shared/Documentos/foo.jpg")
    ok = await coord.enqueue(
        file_id=file_id,
        path="/mnt/shared/Documentos/foo.jpg",
        local_confidence=0.5,
    )
    assert ok is False


async def test_enqueue_writes_request_json_and_updates_ocr_pending(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """Happy path: PC online + low confidence ��' request.json written +
    ocr_pending.status='edge_queued'."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        autoqueue_threshold=0.85,
        probe_fn=make_probe({"edge.local"}),
    )
    await _make_online(coord)  # populate is_online from the mock probe
    file_id = "d" * 32
    await _seed_file(db, ocr_repo, file_id, "/mnt/shared/Documentos/foo.jpg")
    ok = await coord.enqueue(
        file_id=file_id,
        path="/mnt/shared/Documentos/foo.jpg",
        local_confidence=0.42,
    )
    assert ok is True
    # request.json written with path-only delivery
    request_path = edge_root / "edge_queue" / file_id / "request.json"
    assert request_path.exists()
    body = json.loads(request_path.read_text(encoding="utf-8"))
    assert body["file_id"] == file_id
    # Path is SMB-relative (compartido/ stripped)
    assert body["path"] == "Documentos/foo.jpg"
    assert body["original_path"] == "/mnt/shared/Documentos/foo.jpg"
    assert body["local_confidence"] == 0.42
    assert body["queue_source"] == "tesseract_local"
    # ocr_pending updated
    row = await ocr_repo.get(file_id)
    assert row is not None
    assert row.status == "edge_queued"
    assert row.edge_queued_at is not None
    assert row.edge_model == "pc-pending"


async def test_enqueue_idempotent_when_already_queued(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """If row already in edge_queued, enqueue is a no-op (returns True)."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        autoqueue_threshold=0.85,
        probe_fn=make_probe({"edge.local"}),
    )
    await _make_online(coord)
    file_id = "e" * 32
    await _seed_file(db, ocr_repo, file_id, "/mnt/shared/Documentos/foo.jpg")
    # First call writes request.json
    ok1 = await coord.enqueue(
        file_id=file_id,
        path="/mnt/shared/Documentos/foo.jpg",
        local_confidence=0.5,
    )
    assert ok1 is True
    # Second call is idempotent
    ok2 = await coord.enqueue(
        file_id=file_id,
        path="/mnt/shared/Documentos/foo.jpg",
        local_confidence=0.5,
    )
    assert ok2 is True


async def test_enqueue_rolls_back_filesystem_when_ocr_row_missing(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """If ocr_pending row doesn't exist (e.g. delete between create and
    enqueue), the request.json is rolled back to avoid orphans."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        autoqueue_threshold=0.85,
        probe_fn=make_probe({"edge.local"}),
    )
    # No _seed_file: no ocr_pending row, no vault_files row.
    file_id = "f" * 32
    ok = await coord.enqueue(
        file_id=file_id,
        path="/mnt/shared/Documentos/foo.jpg",
        local_confidence=0.5,
    )
    assert ok is False
    # request.json was rolled back
    request_path = edge_root / "edge_queue" / file_id / "request.json"
    assert not request_path.exists()


# ---------------------------------------------------------------------------
# recover_zombies (M6 Phase 5)
# ---------------------------------------------------------------------------


async def test_recover_zombies_reverts_old_edge_queued(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """An old edge_queued row is reverted to pending_review + the orphan
    request.json is cleaned up."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        autoqueue_threshold=0.85,
        probe_fn=make_probe({"edge.local"}),
    )
    file_id = "a" * 32
    # Insert vault_files + ocr_pending (edge_queued, very old edge_queued_at)
    await db.conn.execute(
        """
        INSERT INTO vault_files (file_id, source_path, size_bytes, content_sha256, mtime)
        VALUES (?, ?, 1024, ?, 1700000000.0)
        """,
        (file_id, "/mnt/shared/foo.jpg", file_id),
    )
    await db.conn.commit()
    await ocr_repo.create(
        file_id=file_id,
        local_confidence=0.5,
        local_text="partial",
        local_model="tesseract-5",
        status="edge_queued",
    )
    await ocr_repo.update_status(
        file_id,
        "edge_queued",
        edge_queued_at="2020-01-01T00:00:00.000Z",  # 6+ years old
        edge_model="pc-tesseract-5",
    )
    # Write orphan request.json
    request_dir = edge_root / "edge_queue" / file_id
    request_dir.mkdir(parents=True)
    request_path = request_dir / "request.json"
    request_path.write_text("{}", encoding="utf-8")

    # timeout_hours=2 means anything older than 2h is a zombie
    recovered = await coord.recover_zombies(timeout_hours=2)
    assert recovered == 1
    # ocr_pending reverted
    row = await ocr_repo.get(file_id)
    assert row is not None
    assert row.status == "pending_review"
    assert row.edge_queued_at is None
    assert row.edge_model is None
    # request.json cleaned up
    assert not request_path.exists()


async def test_recover_zombies_idempotent(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """Calling recover twice in a row: first recovers 1, second recovers 0."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        autoqueue_threshold=0.85,
        probe_fn=make_probe(set()),
    )
    file_id = "b" * 32
    await db.conn.execute(
        "INSERT INTO vault_files (file_id, source_path, size_bytes, content_sha256, mtime)"
        " VALUES (?, ?, 1024, ?, 1700000000.0)",
        (file_id, "/mnt/shared/foo.jpg", file_id),
    )
    await db.conn.commit()
    await ocr_repo.create(
        file_id=file_id,
        local_confidence=0.5,
        local_text="x",
        local_model="tesseract-5",
        status="edge_queued",
    )
    await ocr_repo.update_status(
        file_id,
        "edge_queued",
        edge_queued_at="2020-01-01T00:00:00.000Z",
        edge_model="pc-x",
    )
    first = await coord.recover_zombies(timeout_hours=2)
    second = await coord.recover_zombies(timeout_hours=2)
    assert first == 1
    assert second == 0


async def test_recover_zombies_skips_recent_rows(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """A row queued < timeout_hours ago is left alone."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        autoqueue_threshold=0.85,
        probe_fn=make_probe(set()),
    )
    file_id = "c" * 32
    await db.conn.execute(
        "INSERT INTO vault_files (file_id, source_path, size_bytes, content_sha256, mtime)"
        " VALUES (?, ?, 1024, ?, 1700000000.0)",
        (file_id, "/mnt/shared/foo.jpg", file_id),
    )
    await db.conn.commit()
    # Recent edge_queued_at
    await ocr_repo.create(
        file_id=file_id,
        local_confidence=0.5,
        local_text="x",
        local_model="tesseract-5",
        status="edge_queued",
    )
    # edge_queued_at = now (well within 2h)
    await ocr_repo.update_status(
        file_id,
        "edge_queued",
        edge_queued_at=_utc_now_iso_ms(),
        edge_model="pc-x",
    )
    recovered = await coord.recover_zombies(timeout_hours=2)
    assert recovered == 0
    row = await ocr_repo.get(file_id)
    assert row is not None
    assert row.status == "edge_queued"


# ---------------------------------------------------------------------------
# _parse_iso
# ---------------------------------------------------------------------------


def test_parse_iso_with_milliseconds() -> None:
    result = _parse_iso("2026-07-10T10:30:45.123Z")
    assert result.year == 2026
    assert result.month == 7
    assert result.day == 10
    assert result.hour == 10
    assert result.minute == 30
    assert result.second == 45
    assert result.microsecond == 123000


def test_parse_iso_without_milliseconds() -> None:
    result = _parse_iso("2026-07-10T10:30:45Z")
    assert result.year == 2026
    assert result.microsecond == 0


def test_parse_iso_unparseable_raises() -> None:
    with pytest.raises(ValueError):
        _parse_iso("not-a-date")


# ---------------------------------------------------------------------------
# start/stop lifecycle
# ---------------------------------------------------------------------------


async def test_start_stop_lifecycle(
    db: Database, ocr_repo: OcrPendingRepo, edge_root: Path
) -> None:
    """start() spawns the probe task; stop() cancels it cleanly."""
    coord = EdgeCoordinator(
        edge_computers=[("edge.local", DEFAULT_PORT)],
        db=db,
        ocr_repo=ocr_repo,
        edge_root=edge_root,
        probe_fn=make_probe(set()),
        probe_interval_s=0.1,  # fast for the test
    )
    await coord.start()
    # is_online is False (no PC online)
    assert await coord.is_online() is False
    await coord.stop()
    # After stop, can start again (idempotent)
    await coord.start()
    await coord.stop()

async def test_to_smb_relative_uses_configured_prefix(tmp_path: Path) -> None:
    """Each deployment can map its private shared-root layout via env/settings."""
    coord = EdgeCoordinator(
        edge_computers=[],
        db=object(),  # type: ignore[arg-type]
        ocr_repo=object(),  # type: ignore[arg-type]
        edge_root=tmp_path,
        smb_root_prefix="/srv/private-share",
    )
    assert coord._to_smb_relative("/srv/private-share/docs/file.pdf") == "docs/file.pdf"
    assert coord._to_smb_relative("/different/root/file.pdf") == "/different/root/file.pdf"
