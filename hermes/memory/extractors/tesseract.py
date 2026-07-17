"""Tesseract OCR extractor — for .jpg, .jpeg, .png files.

Uses `pytesseract.image_to_data()` to get per-word confidence, then
aggregates to a mean (TDD §4.1 contract: "mean of non-zero word
confs, excluding punctuation"). Returns:
- text: extracted text (may be empty for low-quality images)
- confidence: 0.0-1.0 (mean of non-zero word confs; 0.0 if no words
  detected at all)
- model: 'tesseract-5' (or 'tesseract-<version>' from pytesseract.get_tesseract_version)
- error: 'tesseract_not_installed' if the binary is missing; 'empty_extraction'
  if the image has no detectable text; None on success

Privacy level: local (the binary is on the NAS, no network). The
Sprint 19 north star says NO data leaves the NAS by default. This
extractor honors that.

Confidence aggregation (Sprint 19 §4.1):
- For each word detected by Tesseract, the 'conf' field is 0-100
  (-1 for "no detection" or punctuation-only)
- We compute the mean of non-negative, non-zero confs
- Confidences are integer 0-100 in Tesseract's output; we normalize
  to 0.0-1.0 by dividing by 100
- Result: 0.85+ for clean printed text, 0.60-0.85 for medium quality,
  <0.60 for handwriting / poor scans / low DPI

Threshold decision (drop_watcher reads this):
- >= 0.85: text is good, indexing as-is
- 0.60-0.85: text extracted but flagged for low-confidence search ranking
- < 0.60: text NOT extracted into vault_files.text, ocr_pending row
  created for manual review or hosted VLM escalation (Sprint 19 §4.4.1)
"""

from __future__ import annotations

import logging
from pathlib import Path

from hermes.memory.extractors import ExtractionResult

logger = logging.getLogger(__name__)


# Minimum confidence threshold for accepting the extraction.
# Below this, the watcher routes the file to ocr_pending (instead of
# writing text to vault_files). Per Sprint 19 §4.1: "0.60 <= confidence
# < 0.85 → text_confidence flag; < 0.60 → text NULL, ocr_pending row".
LOW_CONFIDENCE_THRESHOLD = 0.60


def _tesseract_version() -> str:
    """Return the tesseract binary version, e.g. '5.3.0' or 'unknown'.

    Defensive: if pytesseract isn't installed or the binary is missing,
    returns 'unknown' (the caller doesn't care about the exact version
    for routing decisions).

    Cached (Sprint 19 LLM review 2026-07-10): the version is a per-process
    constant. Calling get_tesseract_version() spawns a subprocess
    (`tesseract --version`) each time, which is wasteful when the
    watcher is processing a directory of 100+ images. We memoize the
    first successful call.
    """
    cached = getattr(_tesseract_version, "_cached", None)
    if cached is not None:
        return cached
    try:
        import pytesseract  # local import: optional dep

        v = pytesseract.get_tesseract_version()
        result = f"tesseract-{v}"
    except Exception:
        result = "tesseract-unknown"
    _tesseract_version._cached = result  # type: ignore[attr-defined]
    return result


def extract(path: Path) -> ExtractionResult:
    """Run Tesseract OCR on a JPG/PNG file.

    Returns ExtractionResult with:
    - text: extracted text (may be empty if Tesseract found nothing)
    - confidence: 0.0-1.0 mean of per-word Tesseract confidences
    - model: 'tesseract-<version>' (e.g. 'tesseract-5.3.0')
    - error: 'tesseract_not_installed' if binary missing;
      'empty_extraction' if no text detected; None on success

    Race conditions: the file may be moved/deleted between drop_watcher
    detection and OCR. We catch FileNotFoundError and return error.
    """
    try:
        import pytesseract
    except ImportError as exc:
        return ExtractionResult(
            text="",
            confidence=0.0,
            model="tesseract",
            error=f"tesseract_deps_missing: {exc}",
        )

    # Check if the tesseract binary is installed. pytesseract raises
    # TesseractNotFoundError if the binary is missing. We catch it
    # explicitly to give a clear error.
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        logger.warning(
            "tesseract_not_installed",
            extra={"path": path.as_posix()},
        )
        return ExtractionResult(
            text="",
            confidence=0.0,
            model="tesseract-unknown",
            error="tesseract_not_installed",
        )

    try:
        # image_to_data returns a dict with 'text' (per-word) and 'conf'
        # (per-word confidence 0-100, -1 for no detection).
        # Output level 3 = full info per word (default for image_to_data
        # is 3 already, but explicit).
        from PIL import Image

        with Image.open(path) as img:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except FileNotFoundError:
        return ExtractionResult(
            text="",
            confidence=0.0,
            model=_tesseract_version(),
            error="file_not_found",
        )
    except OSError as exc:
        logger.warning(
            "tesseract_image_open_error",
            extra={"path": path.as_posix(), "error": str(exc)},
        )
        return ExtractionResult(
            text="",
            confidence=0.0,
            model=_tesseract_version(),
            error=f"image_open_error: {exc}",
        )
    except pytesseract.TesseractError as exc:
        # Tesseract binary exists but failed on this image
        logger.warning(
            "tesseract_runtime_error",
            extra={"path": path.as_posix(), "error": str(exc)},
        )
        return ExtractionResult(
            text="",
            confidence=0.0,
            model=_tesseract_version(),
            error=f"tesseract_error: {exc}",
        )

    # Aggregate: per-word confidences, filter -1 (no detection) and
    # 0 (Tesseract reports these for "I'm not sure"), then mean.
    words = data.get("text", [])
    confs = data.get("conf", [])
    text_parts: list[str] = []
    valid_confs: list[float] = []
    for word, conf in zip(words, confs, strict=False):
        if not word or not word.strip():
            continue  # skip empty / whitespace-only
        if conf is None or conf < 0:
            continue  # -1 = "Tesseract didn't even try this"
        text_parts.append(word)
        # Tesseract's per-word conf is integer 0-100. Normalize to 0.0-1.0.
        valid_confs.append(float(conf) / 100.0)

    text = " ".join(text_parts)
    if not valid_confs:
        # No text detected at all (image is blank, all noise, etc.)
        return ExtractionResult(
            text="",
            confidence=0.0,
            model=_tesseract_version(),
            error="empty_extraction",
        )

    confidence = sum(valid_confs) / len(valid_confs)
    return ExtractionResult(
        text=text,
        confidence=confidence,
        model=_tesseract_version(),
        error=None,
    )
