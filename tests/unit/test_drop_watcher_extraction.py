"""Tests for DropWatcher's integration with extractors + ocr_pending.

Sprint 19 Slice 4b: the watcher calls an extractor after the
vault_files + bridge transaction commits. The extractor's result
determines whether text goes to vault_files.text (good confidence) or
the file is queued to ocr_pending (low confidence / extraction failure).

These tests verify the integration WITHOUT mocking the extractors —
we use real extractor functions (plain for txt/md, the actual
Tesseract import path for tesseract_not_installed fallback).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from hermes.memory.collections import VaultCollectionsRepo
from hermes.memory.db import Database
from hermes.memory.drop_watcher import DropWatcher
from hermes.memory.ocr_pending_repo import OcrPendingRepo

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def db(tmp_path) -> AsyncGenerator[Database, None]:
    d = Database(tmp_path / "test_extract.db")
    await d.initialize()
    try:
        yield d
    finally:
        await d.close()


@pytest.fixture
def collections_repo(db: Database) -> VaultCollectionsRepo:
    return VaultCollectionsRepo(db)


@pytest.fixture
def ocr_repo(db: Database) -> OcrPendingRepo:
    return OcrPendingRepo(db)


@pytest.fixture
def drop_root(tmp_path: Path) -> Path:
    p = tmp_path / "drop"
    p.mkdir()
    return p


# --- High-confidence path: text written to vault_files ----------------


async def test_txt_file_writes_text_to_vault_files(
    db: Database,
    collections_repo: VaultCollectionsRepo,
    ocr_repo: OcrPendingRepo,
    drop_root: Path,
) -> None:
    """Plain text file: text is extracted, written to vault_files.text,
    text_source='plain-read', NO ocr_pending row."""
    watcher = DropWatcher(
        db=db,
        collections_repo=collections_repo,
        ocr_pending_repo=ocr_repo,
        drop_root=drop_root,
    )
    f = drop_root / "01_Proyectos" / "notes.txt"
    f.parent.mkdir()
    f.write_text("hello second brain", encoding="utf-8")
    result = await watcher.process_path(f)
    assert result.action == "inserted"
    # text written to vault_files
    async with db.conn.execute(
        "SELECT text, text_source FROM vault_files WHERE file_id = ?",
        (result.file_id,),
    ) as cur:
        row = await cur.fetchone()
    assert row["text"] == "hello second brain"
    assert row["text_source"] == "plain-read"
    # No ocr_pending row (high confidence: 1.0 from plain)
    ocr_row = await ocr_repo.get(result.file_id)
    assert ocr_row is None


async def test_md_file_writes_text_to_vault_files(
    db: Database,
    collections_repo: VaultCollectionsRepo,
    ocr_repo: OcrPendingRepo,
    drop_root: Path,
) -> None:
    """Markdown file: same as txt, goes through plain extractor."""
    watcher = DropWatcher(
        db=db,
        collections_repo=collections_repo,
        ocr_pending_repo=ocr_repo,
        drop_root=drop_root,
    )
    f = drop_root / "01_Proyectos" / "readme.md"
    f.parent.mkdir()
    f.write_text("# Title\n\nBody", encoding="utf-8")
    result = await watcher.process_path(f)
    assert result.action == "inserted"
    async with db.conn.execute(
        "SELECT text, text_source FROM vault_files WHERE file_id = ?",
        (result.file_id,),
    ) as cur:
        row = await cur.fetchone()
    assert row["text"] == "# Title\n\nBody"
    assert row["text_source"] == "plain-read"
    assert await ocr_repo.get(result.file_id) is None


# --- Low-confidence / failure path: ocr_pending row created -------------


async def test_jpg_file_creates_ocr_pending_when_tesseract_not_installed(
    db: Database,
    collections_repo: VaultCollectionsRepo,
    ocr_repo: OcrPendingRepo,
    drop_root: Path,
) -> None:
    """JPG with no Tesseract available → ocr_pending row with error info.

    In our CI env, Tesseract binary isn't installed (pytesseract may or
    may not be). Either way, the watcher should gracefully queue the
    file to ocr_pending with status='pending_review' and a descriptive
    error. vault_files row is still created (the file is registered).
    """
    watcher = DropWatcher(
        db=db,
        collections_repo=collections_repo,
        ocr_pending_repo=ocr_repo,
        drop_root=drop_root,
    )
    f = drop_root / "01_Proyectos" / "scan.jpg"
    f.parent.mkdir()
    f.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header
    result = await watcher.process_path(f)
    assert result.action == "inserted"
    # vault_files row exists
    async with db.conn.execute(
        "SELECT file_id, source_path FROM vault_files WHERE file_id = ?",
        (result.file_id,),
    ) as cur:
        row = await cur.fetchone()
    assert row["file_id"] == result.file_id
    # ocr_pending row created with the appropriate status
    ocr_row = await ocr_repo.get(result.file_id)
    assert ocr_row is not None
    assert ocr_row.status == "pending_review"
    # local_model is 'tesseract-*' (unknown if pytesseract missing)
    assert "tesseract" in ocr_row.local_model


async def test_pdf_file_with_pymupdf_writes_text(
    db: Database,
    collections_repo: VaultCollectionsRepo,
    ocr_repo: OcrPendingRepo,
    drop_root: Path,
) -> None:
    """PDF (digital-born, has extractable text): text in vault_files.

    This test requires a real PDF. We use pymupdf to generate one in
    tmp_path. If pymupdf is missing, this test is skipped.
    """
    pytest.importorskip("pymupdf")
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 in points
    page.insert_text((72, 72), "test contract content", fontsize=12)
    pdf_path = drop_root / "01_Proyectos" / "contract.pdf"
    pdf_path.parent.mkdir(exist_ok=True)
    doc.save(pdf_path.as_posix())
    doc.close()

    watcher = DropWatcher(
        db=db,
        collections_repo=collections_repo,
        ocr_pending_repo=ocr_repo,
        drop_root=drop_root,
    )
    result = await watcher.process_path(watcher._drop_root / "01_Proyectos" / "contract.pdf")
    assert result.action == "inserted"
    async with db.conn.execute(
        "SELECT text, text_source FROM vault_files WHERE file_id = ?",
        (result.file_id,),
    ) as cur:
        row = await cur.fetchone()
    assert "test contract content" in (row["text"] or "")
    assert row["text_source"] == "pymupdf"


# --- Edge case: ocr_pending_repo is None (no extraction) --------------


async def test_no_ocr_pending_repo_means_no_extraction(
    db: Database,
    collections_repo: VaultCollectionsRepo,
    drop_root: Path,
) -> None:
    """If ocr_pending_repo is None, the watcher skips extraction entirely.

    The vault_files row is still created, but text is NULL and there's
    no ocr_pending entry. This is the legacy behavior pre-Slice 4b.
    """
    watcher = DropWatcher(
        db=db,
        collections_repo=collections_repo,
        ocr_pending_repo=None,
        drop_root=drop_root,
    )
    f = drop_root / "01_Proyectos" / "notes.txt"
    f.parent.mkdir()
    f.write_text("hello", encoding="utf-8")
    result = await watcher.process_path(f)
    assert result.action == "inserted"
    async with db.conn.execute(
        "SELECT text, text_source FROM vault_files WHERE file_id = ?",
        (result.file_id,),
    ) as cur:
        row = await cur.fetchone()
    assert row["text"] is None
    assert row["text_source"] is None
