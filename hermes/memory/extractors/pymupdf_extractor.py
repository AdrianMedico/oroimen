"""PyMuPDF extractor — for .pdf files.

Sprint 17 baseline extractor (re-uses the logic). PyMuPDF gives clean
text for most PDFs (digital-born, not scanned). For scanned PDFs with
no embedded text, the extractor falls back to OCR: rasterize each page
to a temp PNG and run tesseract on it.

Sprint 19.5 (PR-A, 2026-07-13): added OCR fallback for scanned PDFs.
Before this, scanned PDFs (image wrapped in PDF, no text layer) ended
up tracked in vault_files but with 0 chunks — invisible to semantic
search. 33 of 131 vault files (25%) were affected in production.

Anti-OOM design (Gemini review 2026-07-13):
- Default 200 DPI grayscale (~13MB/page A4, sufficient for tesseract
  on printed text; 300 DPI is overkill and triples memory)
- Streaming pattern: write PNG to temp file, tesseract, delete, next
  page. Memory bounded to 1 page at a time.
- Hard cap 600 DPI (anti-OOM)

Why 200 DPI grayscale vs 300 DPI color:
- 300 DPI color = 25-30MB/page (Gemini)
- 200 DPI grayscale = 13MB/page (our choice)
- Tesseract reads perfectly text in B&W at 200 DPI for printed text
  (90% of cases). 300+ DPI only for handwriting/microtext.
- Grayscale reduces ~3x vs RGB; tesseract doesn't need color.

Configurable via Settings (env vars):
- VAULT_OCR_FALLBACK_DPI (default 200, hard cap 600)
- VAULT_OCR_FALLBACK_GRAYSCALE (default True)
- VAULT_OCR_FALLBACK_LANG (default deu+eng+spa)
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf

from hermes.memory.extractors import ExtractionResult, logger


def extract(
    path: Path, *, ocr_dpi: int = 200, ocr_grayscale: bool = True, ocr_lang: str = "deu+eng+spa"
) -> ExtractionResult:
    """Extract text from a PDF using PyMuPDF, with OCR fallback for scanned PDFs.

    Algorithm:
    1. Try pymupdf text extraction (fast path for digital-born PDFs).
    2. If text is empty (no embedded text, scanned PDF), rasterize each
       page to a temp PNG and run tesseract (OCR fallback).
    3. Concatenate OCR text from all pages.

    Returns ExtractionResult with text=<all pages joined by '\\n\\n'>,
    confidence=1.0 (deterministic, no model confidence for fallback),
    model='pymupdf' or 'pymupdf+ocr_fallback', error=None on success.

    Error cases:
    - pymupdf not installed: error='pymupdf_not_installed'
    - File not found: error='file_not_found'
    - Corrupted PDF: error='parse_error: <reason>'
    - Tesseract not installed (OCR fallback): error='tesseract_not_installed'
    """
    try:
        import pymupdf
    except ImportError:
        return ExtractionResult(
            text="",
            confidence=1.0,
            model="pymupdf",
            error="pymupdf_not_installed",
        )

    try:
        doc = pymupdf.open(path)
    except FileNotFoundError:
        return ExtractionResult(text="", confidence=1.0, model="pymupdf", error="file_not_found")
    except Exception as exc:
        # pymupdf raises various exceptions for corrupted PDFs (fitz.FileDataError,
        # RuntimeError, etc.). Catch broadly; surface a clean error string.
        logger.warning(
            "pymupdf_open_error",
            extra={"path": path.as_posix(), "error": str(exc)},
        )
        return ExtractionResult(
            text="",
            confidence=1.0,
            model="pymupdf",
            error=f"parse_error: {exc}",
        )

    try:
        pages_text: list[str] = []
        for page in doc:  # type: ignore[attr-defined]
            page_text = page.get_text("text")
            if page_text:
                pages_text.append(page_text)
        text = "\n\n".join(pages_text)
    finally:
        doc.close()

    # Sprint 19.5 (PR-A): if pymupdf returned empty text, this is a scanned
    # PDF without text layer. Fall back to OCR: rasterize each page and run
    # tesseract. Memory-bounded via streaming (write PNG, tesseract, delete).
    if not text.strip():
        logger.info(
            "pymupdf_empty_text_trying_ocr_fallback",
            extra={"path": path.as_posix(), "dpi": ocr_dpi, "grayscale": ocr_grayscale},
        )
        ocr_text = _ocr_fallback(
            path,
            ocr_dpi=min(max(ocr_dpi, 100), 600),  # hard cap 600 (anti-OOM)
            ocr_grayscale=ocr_grayscale,
            ocr_lang=ocr_lang,
        )
        if ocr_text is not None:
            return ExtractionResult(
                text=ocr_text,
                confidence=1.0,
                model="pymupdf+ocr_fallback",
                error=None,
            )
        # OCR failed (e.g., tesseract not installed). Return empty text with
        # the OCR-specific error. The original pymupdf error was None (we
        # just had no text), so use a specific error code.
        return ExtractionResult(
            text="",
            confidence=1.0,
            model="pymupdf+ocr_fallback",
            error="ocr_fallback_failed",
        )

    return ExtractionResult(text=text, confidence=1.0, model="pymupdf", error=None)


def _ocr_fallback(path: Path, *, ocr_dpi: int, ocr_grayscale: bool, ocr_lang: str) -> str | None:
    """OCR fallback for PDFs with no text layer.

    Rasterizes each page to a temp PNG (memory-bounded streaming pattern)
    and runs tesseract on each. Concatenates results.

    Returns the OCR text on success, None on failure (e.g., tesseract not
    installed, all pages fail).
    """
    import pymupdf  # already imported by caller, but be safe

    # Check tesseract is available (cross-platform: shutil.which works on
    # Linux/macOS/Windows; `which` is Unix-only).
    if shutil.which("tesseract") is None:
        logger.warning(
            "tesseract_not_installed",
            extra={"path": path.as_posix()},
        )
        return None

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        logger.warning(
            "ocr_fallback_pymupdf_open_error",
            extra={"path": path.as_posix(), "error": str(exc)},
        )
        return None

    # Anti-OOM: process page-by-page, write to temp, tesseract, delete.
    # Memory bounded to 1 page at a time (~13MB at 200 DPI grayscale).
    page_texts: list[str] = []
    try:
        # pymupdf.Document iteration type stubs are incomplete (they return Never).
        # Use indexed access via page_count + load_page to side-step the stubs entirely.
        total_pages = doc.page_count
        for page_num in range(1, total_pages + 1):
            page: pymupdf.Page = doc.load_page(page_num - 1)
            try:
                # Rasterize to pixmap
                # colorspace csGRAY = grayscale (anti-OOM: ~3x smaller than RGB)
                # Sprint 19.5 (PR-A): pymupdf API uses fitz.csGRAY for grayscale
                pix = page.get_pixmap(
                    dpi=ocr_dpi,
                    colorspace=pymupdf.csGRAY if ocr_grayscale else pymupdf.csRGB,
                )
                # Write to temp file (pymupdf Pixmap.save requires filename)
                with tempfile.NamedTemporaryFile(
                    suffix=".png", prefix=f"pymupdf_ocr_page{page_num}_", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                try:
                    pix.save(tmp_path)
                    # Release pixmap BEFORE running tesseract (anti-OOM)
                    del pix
                    # Run tesseract
                    tesseract_cmd = ["tesseract", tmp_path, "-", "-l", ocr_lang, "--psm", "1"]
                    result = subprocess.run(
                        tesseract_cmd, capture_output=True, text=True, timeout=60
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        page_texts.append(result.stdout.strip())
                    else:
                        logger.debug(
                            "ocr_fallback_page_failed",
                            extra={
                                "path": path.as_posix(),
                                "page": page_num,
                                "returncode": result.returncode,
                                "stderr": result.stderr[:200] if result.stderr else "",
                            },
                        )
                finally:
                    # Always delete the temp PNG (don't leak disk)
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_path)
            except Exception as exc:
                logger.warning(
                    "ocr_fallback_page_exception",
                    extra={
                        "path": path.as_posix(),
                        "page": page_num,
                        "error": str(exc),
                    },
                )
                # Continue with next page
                continue
    finally:
        doc.close()

    return "\n\n".join(page_texts) if page_texts else None
