"""Extractors package — Sprint 19 Slice 4b.

Each module here implements a single text extractor for one file type
(extension family). The package is invoked via `extract_for_extension()`
which dispatches by file extension. Each extractor returns a
`ExtractionResult` with:
- text: extracted text (or empty string if extraction failed)
- confidence: 0.0-1.0 reported by the extractor (Tesseract gives per-word
  confidence aggregated to a mean; other extractors return 1.0 because
  their extraction is "deterministic" — the text is what it is, no
  OCR involved)
- model: identifier of the model/extractor used (e.g. 'tesseract-5',
  'pymupdf', 'python-docx')
- error: None on success, error string on failure (e.g.
  'tesseract_not_installed' for OCR without binary)

The `EXTENSION_ROUTER` dict maps file extension (lowercase, with dot) to
the (extractor_module_name, privacy_level) tuple. The drop_watcher
uses this to dispatch.

Why a separate package: keep the extractors testable in isolation
(no DB, no filesystem events, no async). Each is a pure function
(input: file path; output: ExtractionResult).

North star (Sprint 19 §4.1): zero data leaves the NAS. All extractors
in this package are LOCAL (no network calls, no hosted APIs). Hosted
VLM (MiniMax-M3) is opt-in via /externalOCR command (separate code path
in Slice 4d, NOT in this package).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Outcome of an extractor call. Used by tests, the drop_watcher, and
    the future `/pendingOCR` user command for display.

    Fields:
    - text: the extracted text. May be empty (e.g., empty file, image
      with no detectable text, or extraction failure). The drop_watcher
      checks confidence + error to decide whether to write text to
      vault_files or queue to ocr_pending.
    - confidence: 0.0-1.0.
        - 1.0 for deterministic extractors (plain, pymupdf, python-docx,
          openpyxl) — the text is what it is, no OCR involved.
        - Tesseract: mean of per-word confidences (filtering -1 and 0).
        - 0.0 if extraction failed or no text was detected.
    - model: identifier of the model/extractor (e.g. 'tesseract-5.3.0',
      'pymupdf', 'python-docx', 'openpyxl', 'plain-read'). This is the
      provenance for vault_files.text_source.
    - error: None on success. On failure, a short error code:
        - 'tesseract_not_installed': Tesseract binary missing (apt not run)
        - 'tesseract_deps_missing': pytesseract or Pillow not installed
        - 'tesseract_error': Tesseract runtime error
        - 'empty_extraction': No text detected (Tesseract found nothing)
        - 'pymupdf_not_installed': pymupdf not installed
        - 'parse_error: <reason>': Corrupted file
        - 'file_not_found': File missing (race with drop_watcher detection)
        - 'read_error: <reason>': OS error reading
    """

    text: str
    confidence: float
    model: str
    error: str | None


#: Extension router (Sprint 19 §4.1). Single source of truth for the
#: drop_watcher AND M6 Phase 2 reconciliation. Map: lowercase extension
#: (with dot) -> extractor module name. Import is lazy (inside
#: `extract_for_extension`) to avoid forcing every importer to have
#: all extract deps installed (e.g., for unit tests that only use plain).
EXTENSION_ROUTER: dict[str, str] = {
    ".pdf": "pymupdf_extractor",
    ".docx": "python_docx",
    ".xlsx": "openpyxl_extractor",
    ".txt": "plain",
    ".md": "plain",
    ".jpg": "tesseract",
    ".jpeg": "tesseract",
    ".png": "tesseract",
}


def extract_for_extension(
    path: Path,
    ext: str | None = None,
    *,
    ocr_dpi: int = 200,
    ocr_grayscale: bool = True,
    ocr_lang: str = "deu+eng+spa",
) -> ExtractionResult:
    """Dispatch to the right extractor based on file extension.

    Args:
        path: file to extract from. Must exist (caller is responsible).
        ext: file extension (lowercase, with dot). If None, inferred
             from path.suffix.
        ocr_dpi: Sprint 19.5 (PR-A) — DPI for PDF OCR fallback. Default 200.
        ocr_grayscale: convert pages to grayscale before OCR. Default True.
        ocr_lang: tesseract language codes. Default deu+eng+spa.

    Returns:
        ExtractionResult from the dispatched extractor. If ext is not
        in EXTENSION_ROUTER, returns ExtractionResult with error=
        'unsupported_extension' and an empty text.

    The dispatch is lazy: each extractor's module is imported only when
    needed. This means tests that only use plain (no pytesseract / pymupdf
    / etc.) don't need the heavier deps installed.
    """
    if ext is None:
        ext = path.suffix.lower()
    if not ext:
        return ExtractionResult(
            text="",
            confidence=0.0,
            model="unknown",
            error="no_extension",
        )
    module_name = EXTENSION_ROUTER.get(ext)
    if module_name is None:
        return ExtractionResult(
            text="",
            confidence=0.0,
            model="unknown",
            error=f"unsupported_extension: {ext}",
        )

    # Lazy import — see module docstring.
    import importlib

    try:
        module = importlib.import_module(f"hermes.memory.extractors.{module_name}")
    except ImportError as exc:
        # Module-level ImportError (e.g., pymupdf not installed). The
        # extract() function inside the module will also catch this and
        # return error='xxx_not_installed', but we catch here too as a
        # belt-and-suspenders for the dispatch path.
        logger.warning(
            "extractor_module_import_failed",
            extra={"module": module_name, "error": str(exc)},
        )
        return ExtractionResult(
            text="",
            confidence=0.0,
            model=module_name,
            error=f"extractor_import_failed: {exc}",
        )

    # Sprint 19.5 (PR-A): pass OCR settings through to extractors that
    # support them (pymupdf_extractor). Other extractors ignore them
    # (their extract() signature doesn't accept kwargs).
    #
    # LLM cascade review 2026-07-13: check for SPECIFIC kwarg names, not
    # just "any keyword-only param". A future extractor with a different
    # keyword-only param (e.g. verbose=True) would incorrectly receive
    # our OCR kwargs. Only pass kwargs if the extractor's signature
    # actually contains ocr_dpi (or one of the other OCR kwargs).
    import inspect

    extract_fn = module.extract
    try:
        sig = inspect.signature(extract_fn)
        param_names = set(sig.parameters.keys())
    except (TypeError, ValueError):
        # Built-in or C-implemented function — can't inspect, fall back
        # to positional call (safe for extractors without OCR support)
        return extract_fn(path)

    if "ocr_dpi" in param_names:
        # Extractor explicitly supports OCR kwargs (Sprint 19.5+)
        return extract_fn(path, ocr_dpi=ocr_dpi, ocr_grayscale=ocr_grayscale, ocr_lang=ocr_lang)
    # Extractor doesn't support OCR kwargs (legacy extractors).
    # Fall back to positional call.
    return extract_fn(path)
