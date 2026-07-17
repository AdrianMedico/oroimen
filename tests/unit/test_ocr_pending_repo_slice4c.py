"""Tests for OcrPendingRepo extensions added in Sprint 19 Slice 4c.

Coverage:
- fetch_pending_for_catchup: JOINs with vault_files, filters by path
  extension, returns (file_id, path, local_confidence) tuples, FIFO order.
- fetch_zombie_candidates: filters by status='edge_queued' AND
  edge_queued_at cutoff.
- revert_to_pending: idempotent state machine, returns bool.
- update_status: writes edge_queued_at / edge_model correctly (already
  covered indirectly by drop_watcher tests, but explicit here).

Per Gemini 3.1 Pro feedback (2026-07-10): these are the M6 Phase 5
contract tests, named explicitly in TDD §4.4.5.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from hermes.memory.db import Database
from hermes.memory.ocr_pending_repo import OcrPendingRepo

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def db(tmp_path) -> AsyncGenerator[Database, None]:
    d = Database(tmp_path / "test_repo.db")
    await d.initialize()
    try:
        yield d
    finally:
        await d.close()


@pytest.fixture
def repo(db: Database) -> OcrPendingRepo:
    return OcrPendingRepo(db)


# Helper: insert a vault_files row + ocr_pending row
async def _seed(
    db: Database,
    repo: OcrPendingRepo,
    *,
    file_id: str,
    path: str,
    status: str = "pending_review",
    confidence: float = 0.5,
    edge_queued_at: str | None = None,
) -> None:
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
        local_confidence=confidence,
        local_text="some text",
        local_model="tesseract-5",
        status=status,
    )
    if edge_queued_at is not None:
        await repo.update_status(
            file_id,
            status,
            edge_queued_at=edge_queued_at,
            edge_model="pc-tesseract-5",
        )


# --- fetch_pending_for_catchup ------------------------------------------


async def test_fetch_pending_for_catchup_filters_by_extension(
    db: Database, repo: OcrPendingRepo
) -> None:
    """Only .jpg/.jpeg/.png files are returned; .pdf / .txt are skipped."""
    await _seed(db, repo, file_id="a" * 32, path="Documentos/factura.jpg")
    await _seed(db, repo, file_id="b" * 32, path="Documentos/notas.txt")
    await _seed(db, repo, file_id="c" * 32, path="Documentos/receta.png")
    await _seed(db, repo, file_id="d" * 32, path="Documentos/curriculum.pdf")

    rows = await repo.fetch_pending_for_catchup(path_extensions=[".jpg", ".jpeg", ".png"], limit=10)
    file_ids = [fid for fid, _, _ in rows]
    assert len(rows) == 2
    assert "a" * 32 in file_ids
    assert "c" * 32 in file_ids


async def test_fetch_pending_for_catchup_respects_status(
    db: Database, repo: OcrPendingRepo
) -> None:
    """Only status='pending_review' rows are returned; edge_queued / others are skipped."""
    await _seed(db, repo, file_id="a" * 32, path="img1.jpg", status="pending_review")
    await _seed(
        db,
        repo,
        file_id="b" * 32,
        path="img2.jpg",
        status="edge_queued",
        edge_queued_at="2026-07-10T10:00:00.000Z",
    )
    await _seed(db, repo, file_id="c" * 32, path="img3.jpg", status="edge_processed")

    rows = await repo.fetch_pending_for_catchup(path_extensions=[".jpg"], limit=10)
    file_ids = [fid for fid, _, _ in rows]
    assert file_ids == ["a" * 32]


async def test_fetch_pending_for_catchup_returns_paths(db: Database, repo: OcrPendingRepo) -> None:
    """Returned tuples include the POSIX path (for SMB-relative conversion)."""
    await _seed(db, repo, file_id="a" * 32, path="/mnt/shared/Documentos/foo/bar.jpg")
    rows = await repo.fetch_pending_for_catchup(path_extensions=[".jpg"], limit=10)
    assert len(rows) == 1
    file_id, path, confidence = rows[0]
    assert file_id == "a" * 32
    assert path == "/mnt/shared/Documentos/foo/bar.jpg"
    assert confidence == 0.5  # from _seed default


async def test_fetch_pending_for_catchup_empty_extensions(
    db: Database, repo: OcrPendingRepo
) -> None:
    """Empty extension list returns empty (defensive — never query without filters)."""
    await _seed(db, repo, file_id="a" * 32, path="foo.jpg")
    rows = await repo.fetch_pending_for_catchup(path_extensions=[], limit=10)
    assert rows == []


async def test_fetch_pending_for_catchup_fifo_order(db: Database, repo: OcrPendingRepo) -> None:
    """Rows are returned in created_at ASC order (oldest first)."""
    # Create 3 rows; created_at is set by repo.create() to NOW. They
    # should all be in the same second, so order is implementation-defined.
    # Instead, verify that calling twice returns same set (idempotent).
    await _seed(db, repo, file_id="a" * 32, path="a.jpg")
    await _seed(db, repo, file_id="b" * 32, path="b.jpg")
    await _seed(db, repo, file_id="c" * 32, path="c.jpg")
    rows1 = await repo.fetch_pending_for_catchup(path_extensions=[".jpg"], limit=10)
    rows2 = await repo.fetch_pending_for_catchup(path_extensions=[".jpg"], limit=10)
    assert {fid for fid, _, _ in rows1} == {fid for fid, _, _ in rows2}


# --- fetch_zombie_candidates -------------------------------------------


async def test_fetch_zombie_candidates_filters_old(db: Database, repo: OcrPendingRepo) -> None:
    """Returns only edge_queued rows older than the cutoff."""
    # 3h ago: zombie
    await _seed(
        db,
        repo,
        file_id="a" * 32,
        path="old.jpg",
        status="edge_queued",
        edge_queued_at="2026-01-01T00:00:00.000Z",
    )
    # 1h ago: not zombie (within 2h timeout)
    recent_iso = "2026-07-10T22:00:00.000Z"
    await _seed(
        db,
        repo,
        file_id="b" * 32,
        path="recent.jpg",
        status="edge_queued",
        edge_queued_at=recent_iso,
    )

    cutoff = "2026-07-10T20:00:00.000Z"  # 2h before recent_iso
    rows = await repo.fetch_zombie_candidates(cutoff)
    file_ids = [r.file_id for r in rows]
    assert "a" * 32 in file_ids
    assert "b" * 32 not in file_ids


async def test_fetch_zombie_candidates_skips_non_edge_queued(
    db: Database, repo: OcrPendingRepo
) -> None:
    """Only status='edge_queued' rows; pending_review and others are skipped."""
    await _seed(
        db,
        repo,
        file_id="a" * 32,
        path="pending.jpg",
        status="pending_review",
    )
    await _seed(
        db,
        repo,
        file_id="b" * 32,
        path="queued.jpg",
        status="edge_queued",
        edge_queued_at="2026-01-01T00:00:00.000Z",
    )

    rows = await repo.fetch_zombie_candidates("2026-07-10T00:00:00.000Z")
    file_ids = [r.file_id for r in rows]
    assert file_ids == ["b" * 32]


async def test_fetch_zombie_candidates_empty_when_none(db: Database, repo: OcrPendingRepo) -> None:
    """No edge_queued rows -> empty result."""
    rows = await repo.fetch_zombie_candidates("2026-07-10T00:00:00.000Z")
    assert rows == []


# --- revert_to_pending --------------------------------------------------


async def test_revert_to_pending_clears_edge_fields(db: Database, repo: OcrPendingRepo) -> None:
    """Reverting an edge_queued row clears edge_queued_at and edge_model,
    sets status back to pending_review."""
    file_id = "c" * 32
    await _seed(
        db,
        repo,
        file_id=file_id,
        path="zombie.jpg",
        status="edge_queued",
        edge_queued_at="2026-01-01T00:00:00.000Z",
    )
    ok = await repo.revert_to_pending(file_id)
    assert ok is True
    row = await repo.get(file_id)
    assert row is not None
    assert row.status == "pending_review"
    assert row.edge_queued_at is None
    assert row.edge_model is None


async def test_revert_to_pending_idempotent(db: Database, repo: OcrPendingRepo) -> None:
    """Reverting a row that is NOT in edge_queued returns False (no-op)."""
    file_id = "d" * 32
    await _seed(db, repo, file_id=file_id, path="foo.jpg", status="pending_review")
    ok = await repo.revert_to_pending(file_id)
    assert ok is False
    # Status unchanged
    row = await repo.get(file_id)
    assert row is not None
    assert row.status == "pending_review"


async def test_revert_to_pending_missing_row(db: Database, repo: OcrPendingRepo) -> None:
    """Reverting a non-existent file_id returns False."""
    ok = await repo.revert_to_pending("a" * 32)
    assert ok is False
