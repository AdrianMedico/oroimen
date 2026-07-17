"""Tests for hermes.memory.extractors (Sprint 19 Slice 4b).

TDD-first. Each extractor is a pure function (input: file path; output:
ExtractionResult). Tests cover:
- Happy path (text extracted, no error, correct confidence)
- File not found
- Decode error (binary file with .txt extension)
- Empty file
- Unsupported extension (dispatch error)

Tesseract tests are limited: we don't have a real Tesseract binary in
the test env, so we test the error paths (tesseract_not_installed,
file_not_found, empty_extraction) but not the actual OCR. The OCR
correctness is verified in integration / e2e tests when Tesseract is
available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.memory.extractors import (
    EXTENSION_ROUTER,
    ExtractionResult,
    extract_for_extension,
)
from hermes.memory.extractors import plain as plain_extractor
from hermes.memory.extractors import tesseract as tesseract_extractor

pytestmark = pytest.mark.asyncio


# --- plain extractor -----------------------------------------------------


def test_plain_extracts_txt(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hello world", encoding="utf-8")
    result = plain_extractor.extract(f)
    assert result.error is None
    assert result.text == "hello world"
    assert result.confidence == 1.0
    assert result.model == "plain-read"


def test_plain_extracts_md(tmp_path: Path) -> None:
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nBody", encoding="utf-8")
    result = plain_extractor.extract(f)
    assert result.error is None
    assert result.text == "# Title\n\nBody"
    assert result.confidence == 1.0


def test_plain_handles_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    result = plain_extractor.extract(f)
    assert result.error is None
    assert result.text == ""
    assert result.confidence == 1.0


def test_plain_handles_missing_file(tmp_path: Path) -> None:
    f = tmp_path / "ghost.txt"  # never created
    result = plain_extractor.extract(f)
    assert result.error == "file_not_found"
    assert result.text == ""


def test_plain_handles_decode_error(tmp_path: Path) -> None:
    f = tmp_path / "binary.txt"
    f.write_bytes(b"\x00\x01\x02\xff\xfe")  # not valid UTF-8
    result = plain_extractor.extract(f)
    # decode_error uses errors='replace', so text has replacement chars;
    # error is None (we don't fail the extraction)
    assert result.error is None
    assert len(result.text) > 0  # has something, even if garbled


# --- tesseract extractor -------------------------------------------------


def test_tesseract_returns_not_installed_when_binary_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Tesseract binary is not on PATH, error='tesseract_not_installed'.

    We force this by making pytesseract.get_tesseract_version() raise
    TesseractNotFoundError. If pytesseract itself isn't installed in
    the env, we get 'tesseract_deps_missing' which is also acceptable
    (graceful error path).
    """
    try:
        import pytesseract
    except ImportError:
        # pytesseract not installed in env; verify the graceful path
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = tesseract_extractor.extract(f)
        assert (result.error or "").startswith("tesseract_deps_missing")
        return

    def _raise_not_found() -> str:
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "get_tesseract_version", _raise_not_found)
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
    result = tesseract_extractor.extract(f)
    assert result.error == "tesseract_not_installed"
    assert result.text == ""
    assert result.confidence == 0.0


def test_tesseract_handles_missing_file(tmp_path: Path) -> None:
    """File not found → error='file_not_found' (or deps missing / not installed)."""
    f = tmp_path / "ghost.png"
    # Tesseract will be called; if not installed or pytesseract missing,
    # we get one of those errors instead of file_not_found. Accept all.
    result = tesseract_extractor.extract(f)
    assert result.error in {
        "file_not_found",
        "tesseract_not_installed",
    } or (result.error or "").startswith("tesseract_deps_missing")


# --- dispatch (extract_for_extension) -----------------------------------


def test_dispatch_routes_pdf_to_pymupdf() -> None:
    assert EXTENSION_ROUTER[".pdf"] == "pymupdf_extractor"


def test_dispatch_routes_jpg_to_tesseract() -> None:
    assert EXTENSION_ROUTER[".jpg"] == "tesseract"
    assert EXTENSION_ROUTER[".jpeg"] == "tesseract"
    assert EXTENSION_ROUTER[".png"] == "tesseract"


def test_dispatch_routes_docx_to_python_docx() -> None:
    assert EXTENSION_ROUTER[".docx"] == "python_docx"


def test_dispatch_routes_xlsx_to_openpyxl() -> None:
    assert EXTENSION_ROUTER[".xlsx"] == "openpyxl_extractor"


def test_dispatch_routes_txt_md_to_plain() -> None:
    assert EXTENSION_ROUTER[".txt"] == "plain"
    assert EXTENSION_ROUTER[".md"] == "plain"


def test_dispatch_returns_unsupported_for_unknown_ext(tmp_path: Path) -> None:
    f = tmp_path / "weird.xyz"
    f.write_bytes(b"data")
    result = extract_for_extension(f)
    assert result.error is not None
    assert "unsupported_extension" in result.error


def test_dispatch_returns_no_extension_for_no_suffix(tmp_path: Path) -> None:
    f = tmp_path / "noext"
    f.write_bytes(b"data")
    result = extract_for_extension(f)
    assert result.error == "no_extension"


def test_dispatch_infers_ext_from_path(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hello", encoding="utf-8")
    result = extract_for_extension(f)
    # Plain extractor returns content
    assert result.error is None
    assert result.text == "hello"


def test_dispatch_with_tesseract_image_uninstalled(tmp_path: Path) -> None:
    """JPG with Tesseract not installed → graceful error."""
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header
    result = extract_for_extension(f, ext=".jpg")
    # Tesseract not installed (CI env) → error='tesseract_not_installed'
    # If Tesseract IS installed, this would proceed to actual OCR.
    # If pytesseract package missing → 'tesseract_deps_missing: ...'
    assert result.error in {"tesseract_not_installed", "empty_extraction", None} or (
        result.error or ""
    ).startswith("tesseract_deps_missing")


# --- ExtractionResult dataclass ------------------------------------------


def test_extraction_result_defaults() -> None:
    r = ExtractionResult(text="", confidence=0.0, model="test", error=None)
    assert r.text == ""
    assert r.confidence == 0.0
    assert r.model == "test"
    assert r.error is None
    # frozen dataclass
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        r.text = "modified"  # type: ignore[misc]


# --- Sprint 19.5 (PR-A): PDF OCR fallback -------------------------------


def test_pymupdf_digital_born_no_ocr_fallback(tmp_path: Path) -> None:
    """Sprint 19.5 (PR-A): PDF with extractable text does NOT trigger OCR fallback.

    Without this guard, every digital-born PDF would also pay the OCR
    latency cost (~2-5s per page) even though pymupdf text is sufficient.
    """
    pytest.importorskip("pymupdf")
    import pymupdf

    from hermes.memory.extractors.pymupdf_extractor import extract

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "test contract content", fontsize=12)
    pdf_path = tmp_path / "digital_born.pdf"
    doc.save(pdf_path.as_posix())
    doc.close()

    result = extract(pdf_path)
    assert result.error is None
    assert (
        result.model == "pymupdf"
    ), f"Digital-born PDF should NOT trigger OCR fallback, got model={result.model!r}"
    assert "test contract content" in result.text


def test_pymupdf_scanned_triggers_ocr_fallback(tmp_path: Path) -> None:
    """Sprint 19.5 (PR-A): PDF with no text layer triggers OCR fallback.

    Creates a PDF with an empty page (no text). pymupdf.get_text() returns
    empty, so the extractor should fall back to OCR. Without tesseract
    in the test env, the fallback fails gracefully (error='ocr_fallback_failed'),
    but the model should be 'pymupdf+ocr_fallback' to indicate the
    fallback was attempted.
    """
    pytest.importorskip("pymupdf")
    import pymupdf

    from hermes.memory.extractors.pymupdf_extractor import extract

    # Create a PDF with empty page (no text, no image). pymupdf.get_text()
    # will return empty string, triggering the OCR fallback path.
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)  # A4 in points, no text inserted
    pdf_path = tmp_path / "scanned.pdf"
    doc.save(pdf_path.as_posix())
    doc.close()

    result = extract(pdf_path)
    # The model should be 'pymupdf+ocr_fallback' indicating the fallback
    # was attempted (regardless of tesseract availability).
    # If tesseract is installed in test env, the text would be filled.
    # If not (typical for CI), the error is 'ocr_fallback_failed'.
    if result.error is None:
        # tesseract available + succeeded
        assert result.model == "pymupdf+ocr_fallback"
        assert result.text  # non-empty
    else:
        # tesseract not available — fallback attempted but failed
        assert result.model == "pymupdf+ocr_fallback"
        assert result.error == "ocr_fallback_failed"


def test_pymupdf_dpi_hard_cap(tmp_path: Path) -> None:
    """Sprint 19.5 (PR-A, R1 M1 fix 2026-07-13): DPI is hard-capped at 600 (anti-OOM).

    Without the cap, a misconfigured env var (e.g., VAULT_OCR_FALLBACK_DPI=9999)
    would rasterize pages to huge images (9999 DPI page = several GB
    per page), causing OOM. The extractor should clamp DPI to [100, 600].

    R1 M1 fix: previous version called `_ocr_fallback()` directly, bypassing
    the clamp in `extract()`. This test now calls the public `extract()` entry
    point so it actually exercises the clamp. Also added a separate unit test
    (`test_pymupdf_dpi_clamp_formula`) that pins the clamp formula explicitly
    via mocking.
    """
    pytest.importorskip("pymupdf")
    import pymupdf

    from hermes.memory.extractors.pymupdf_extractor import extract

    # Create a minimal PDF (one page, no text, rasterizable)
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    pdf_path = tmp_path / "test.pdf"
    doc.save(pdf_path.as_posix())
    doc.close()

    # Call the public entry point with extreme DPI. The clamp is inside extract()
    # (line 114: ocr_dpi=min(max(ocr_dpi, 100), 600)), so the internal
    # _ocr_fallback should receive DPI=600, not 9999. If tesseract is missing,
    # the function returns an ExtractionResult with error='tesseract_not_installed'.
    # Either way it must NOT crash or OOM with extreme DPI.
    result = extract(pdf_path, ocr_dpi=9999, ocr_grayscale=True, ocr_lang="eng")
    assert result is not None
    # If OCR ran, model will indicate pymupdf+ocr_fallback; if tesseract missing,
    # we'll get pymupdf with error='tesseract_not_installed'. Either is acceptable.
    assert result.model in ("pymupdf", "pymupdf+ocr_fallback")


def test_pymupdf_dpi_clamp_formula(tmp_path: Path) -> None:
    """R1 M1 fix (companion to test_pymupdf_dpi_hard_cap): explicitly verifies
    the clamp formula `min(max(ocr_dpi, 100), 600)` by mocking `_ocr_fallback`
    and inspecting the DPI value it received.

    This catches regressions where the clamp in `extract()` is removed, since
    the previous test called `_ocr_fallback()` directly and would silently
    bypass any clamp changes.
    """
    pytest.importorskip("pymupdf")
    import pymupdf

    from hermes.memory.extractors import pymupdf_extractor as pme

    # Create a minimal PDF
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    pdf_path = tmp_path / "test.pdf"
    doc.save(pdf_path.as_posix())
    doc.close()

    # Capture the DPI that extract() passes to _ocr_fallback
    captured: dict[str, int] = {}

    def fake_ocr_fallback(path, *, ocr_dpi, ocr_grayscale, ocr_lang):
        captured["ocr_dpi"] = ocr_dpi
        return "fake ocr text"

    # Patch _ocr_fallback
    original = pme._ocr_fallback
    pme._ocr_fallback = fake_ocr_fallback
    try:
        # Extreme high: should clamp to 600
        pme.extract(pdf_path, ocr_dpi=9999, ocr_grayscale=True, ocr_lang="eng")
        assert captured["ocr_dpi"] == 600, f"Expected 600, got {captured['ocr_dpi']}"

        # Extreme low: should clamp to 100
        pme.extract(pdf_path, ocr_dpi=50, ocr_grayscale=True, ocr_lang="eng")
        assert captured["ocr_dpi"] == 100, f"Expected 100, got {captured['ocr_dpi']}"

        # Normal: should pass through
        pme.extract(pdf_path, ocr_dpi=300, ocr_grayscale=True, ocr_lang="eng")
        assert captured["ocr_dpi"] == 300, f"Expected 300, got {captured['ocr_dpi']}"
    finally:
        pme._ocr_fallback = original


def test_dispatch_skips_ocr_kwargs_for_legacy_extractors(tmp_path: Path) -> None:
    """Sprint 19.5 (PR-A, LLM cascade fix 2026-07-13): dispatch checks for
    SPECIFIC OCR kwarg names, not just "any keyword-only param".

    Before fix: dispatch called `extract_fn(path, ocr_dpi=..., ...)` for
    ANY extractor with a keyword-only param. A future extractor with
    `def extract(path, *, verbose=True)` would receive the wrong kwargs
    and crash with TypeError.

    After fix: dispatch only passes OCR kwargs if the extractor's signature
    contains 'ocr_dpi' (our specific kwarg). Legacy extractors get the
    positional call.

    Test: verify that the .txt extractor (plain, signature `def extract(path)`)
    is called with positional args only, not with the OCR kwargs. If the
    dispatch passed OCR kwargs to a non-OCR extractor, it would crash
    with TypeError (the extractor would reject unknown kwargs).
    """
    from hermes.memory.extractors import extract_for_extension

    # Create a .txt file
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("hello world", encoding="utf-8")

    # The plain extractor's signature is `def extract(path: Path)` —
    # no kwargs. If the dispatch incorrectly passes OCR kwargs, this
    # would raise TypeError. After the fix, the dispatch falls back
    # to positional call.
    result = extract_for_extension(
        txt_path,
        ext=".txt",
        ocr_dpi=300,
        ocr_grayscale=True,
        ocr_lang="eng",
    )
    assert result.text == "hello world"
    assert result.error is None
    # Model name comes from plain extractor; current value is "plain-read"
    # (read mode). We just check the dispatch didn't error out.
    assert "plain" in result.model
