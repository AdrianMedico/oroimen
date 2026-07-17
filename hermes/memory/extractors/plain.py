"""Plain text extractor — for .txt and .md files.

Trivial: just read the file as UTF-8 and return the text. Markdown
files are passed through as-is (we don't strip the syntax; search
indexes the raw markdown and the LLM can render it).

Privacy level: local. No network, no model. The text is what it is.
"""

from __future__ import annotations

from pathlib import Path

from hermes.memory.extractors import ExtractionResult, logger


def extract(path: Path) -> ExtractionResult:
    """Read a .txt or .md file as UTF-8 text.

    Returns ExtractionResult with text=content, confidence=1.0,
    model='plain-read', error=None on success.

    Error cases:
    - File not found: error='file_not_found', text=''
    - Decode error (binary file with .txt extension): error='decode_error',
      text=<partial content with replacement chars>
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        logger.warning(
            "plain_extractor_file_not_found",
            extra={"path": path.as_posix()},
        )
        return ExtractionResult(text="", confidence=1.0, model="plain-read", error="file_not_found")
    except OSError as exc:
        logger.warning(
            "plain_extractor_read_error",
            extra={"path": path.as_posix(), "error": str(exc)},
        )
        return ExtractionResult(
            text="", confidence=1.0, model="plain-read", error=f"read_error: {exc}"
        )
    return ExtractionResult(text=text, confidence=1.0, model="plain-read", error=None)
