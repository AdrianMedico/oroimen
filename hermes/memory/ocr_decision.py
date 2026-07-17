"""OCR decision logic for user commands (Sprint 19 Slice 4d).

Single source of truth for all 6 OCR-related user commands:
- accept_null
- edit_text
- external_ocr (request + confirm, 2-step)
- edge_ocr
- skip
- pending_list / pending_detail (read-only, no state change)

Called by:
- hermes/handlers/ocr_commands.py (Telegram)
- hermes/receivers/ocr_api.py (WebUI)
- hermes/agent/tools/* (NOT -- see NORTH STAR enforcement)

Contract: TDD §4.3.1.

Process-local state:
- `_confirmations`: external_ocr 2-step pending dict (lost on restart, OK)
- `_rate_limits`: per-user daily external_ocr counter (lost on restart, OK)

NORTH STAR:
- The LLM agent never calls this module directly. external_ocr is a
  USER-ONLY command. See test_ocr_decision.py::test_external_ocr_not_in_llm_tool_registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes.memory.db import Database
    from hermes.memory.edge_coordinator import EdgeCoordinator
    from hermes.memory.ocr_pending_repo import OcrPendingRepo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (override-able via env, but defaults here for clarity)
# ---------------------------------------------------------------------------

#: Max chars for /editText via Telegram (hard Telegram message limit).
EDIT_TEXT_TELEGRAM_MAX_CHARS = 4096
#: Max chars for /editText via WebUI (relaxed, no Telegram limit).
EDIT_TEXT_WEBUI_MAX_CHARS = 100_000
#: TTL for /externalOCR 2-step confirmation entries.
EXTERNAL_OCR_CONFIRMATION_TTL_S = 60
#: Per-user daily limit for /externalOCR request step.
EXTERNAL_OCR_DAILY_LIMIT_DEFAULT = 10
#: Source states from which /acceptNull is valid.
ACCEPT_NULL_SOURCES = ("pending_review", "edge_failed", "edge_queued")
#: Statuses for the 2-step external_ocr flow.
#: Step 1 (request) creates a confirmation with TTL; step 2 (confirm) applies it.


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DecisionError(Exception):
    """Base for all ocr_decision errors."""


class FileNotFoundError_(DecisionError):
    """file_id does not exist in vault_files."""


class InvalidStatusError(DecisionError):
    """Source state doesn't allow the requested action."""

    def __init__(self, file_id: str, current: str, valid_sources: tuple[str, ...]) -> None:
        self.file_id = file_id
        self.current = current
        self.valid_sources = valid_sources
        super().__init__(
            f"file_id={file_id} status={current!r} not in valid sources {valid_sources}"
        )


class RateLimitedError(DecisionError):
    """User exceeded EXTERNAL_OCR_DAILY_LIMIT."""


class TextEmptyError(DecisionError):
    """/editText with empty text."""


class TextTooLongError(DecisionError):
    """/editText text exceeds the limit."""

    def __init__(self, text_len: int, limit: int) -> None:
        self.text_len = text_len
        self.limit = limit
        super().__init__(f"text length {text_len} > limit {limit}")


class EdgeUnavailableError(DecisionError):
    """/edgeOCR but enqueue() returned False (PC offline)."""


class EdgeDisabledError(DecisionError):
    """/edgeOCR but edge_computers env var is empty (no coordinator)."""


class ConfirmationNotFoundError(DecisionError):
    """/externalOCR confirm with unknown/expired confirmation_id."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class DecisionResult:
    file_id: str
    status: str
    ts: datetime
    confirmation_id: str | None = None
    expires_at: datetime | None = None
    text_len: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Process-local state (single Hermes process)
# ---------------------------------------------------------------------------

#: Pending /externalOCR confirmations. Key = composite
#: `f"{user_id}:{file_id}"` (per TDD §4.3.1 line 720). Predictable
#: (not random) — trade-off accepted: cid is process-local + 60s TTL,
#: so predictability is OK for single-user product. TDD also says
#: "per-user, per-file. Prevents User B confirming User A's prompt" —
#: the composite key encodes that constraint at the dict level.
#: Value = (expires_at). user_id and file_id are recoverable from key.
_confirmations: dict[str, datetime] = {}

#: Per-user daily /externalOCR counters. Key = user_id.
#: Value = sorted list of request timestamps (UTC).
_rate_limits: dict[int, list[datetime]] = {}


def _now() -> datetime:
    return datetime.now(UTC)


def _gc_confirmations() -> None:
    """Evict expired confirmation entries. Called before each lookup."""
    now = _now()
    expired = [cid for cid, exp in _confirmations.items() if exp < now]
    for cid in expired:
        del _confirmations[cid]


def _gc_rate_limits() -> None:
    """Evict /externalOCR requests older than 24h. Called before each check."""
    now = _now()
    cutoff = now.timestamp() - 86400
    for user_id, timestamps in list(_rate_limits.items()):
        # Keep only requests within the last 24h
        _rate_limits[user_id] = [ts for ts in timestamps if ts.timestamp() > cutoff]
        if not _rate_limits[user_id]:
            del _rate_limits[user_id]


def _check_rate_limit(user_id: int, limit: int) -> None:
    """Raise RateLimitedError if user has hit the daily limit.

    `limit` is REQUIRED (caller must pass settings.external_ocr_daily_limit).
    R1 fix (2026-07-11): removed os.environ.get fallback — Settings is the
    single source of truth, validated at startup (ge=1, le=1000).
    """
    _gc_rate_limits()
    history = _rate_limits.setdefault(user_id, [])
    if len(history) >= limit:
        raise RateLimitedError(f"user {user_id} hit EXTERNAL_OCR_DAILY_LIMIT={limit}")
    history.append(_now())


def _log_audit(
    action: str,
    file_id: str,
    prev_status: str,
    new_status: str,
    user_id: int,
    **extra: Any,
) -> None:
    """Emit structured audit log per TDD §4.3.1 schema."""
    logger.info(
        "ocr_user_decision",
        extra={
            "action": action,
            "file_id": file_id,
            "prev_status": prev_status,
            "new_status": new_status,
            "user_id": user_id,
            "ts": _now().isoformat(),
            **extra,
        },
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def decide(
    db: Database,
    ocr_repo: OcrPendingRepo,
    edge_coord: EdgeCoordinator | None,
    file_id: str,
    action: str,
    user_id: int,
    text: str | None = None,
    confirmation_id: str | None = None,
    model: str | None = None,
    confidence: float | None = None,
    edit_text_limit: int = EDIT_TEXT_TELEGRAM_MAX_CHARS,
    external_ocr_daily_limit: int = 10,
) -> DecisionResult:
    """Decide the next state for an OCR-related user command.

    Single source of truth for both Telegram handlers and WebUI endpoints.
    Returns DecisionResult or raises a typed exception.

    Args:
        db: Database connection (real sqlite in production, tmp in tests).
        ocr_repo: OcrPendingRepo for CRUD on the ocr_pending table.
        edge_coord: EdgeCoordinator (or None if edge disabled).
        file_id: Target vault_file. Must exist in vault_files (FK enforced).
        action: One of accept_null, edit_text, external_ocr_request,
            external_ocr_confirm, edge_ocr, skip.
        user_id: Telegram/WebUI user id (for audit + rate limit).
        text: For edit_text, the new text content.
        confirmation_id: For external_ocr_confirm, the id from step 1.
        model: For external_ocr_confirm, the hosted model that produced
            the OCR result (e.g., "MiniMax-M3").
        confidence: For external_ocr_confirm, the OCR confidence (0.0-1.0).
        edit_text_limit: Max chars for edit_text. Pass WEBUI limit for
            WebUI calls (100K), TELEGRAM limit for Telegram (4.096).
        external_ocr_daily_limit: Per-user daily cap for external_ocr_request.
            Required: caller MUST pass settings.external_ocr_daily_limit
            (R1 fix: removed env fallback, Settings is source of truth).

    Returns:
        DecisionResult with file_id, new status, ts, and (for
        external_ocr_request) confirmation_id + expires_at.

    Raises:
        FileNotFoundError_: file_id not in vault_files.
        InvalidStatusError: source state not in valid set.
        TextEmptyError: edit_text with empty text.
        TextTooLongError: edit_text > limit.
        RateLimitedError: external_ocr_request hit daily limit.
        EdgeUnavailableError: edge_ocr but PC offline.
        EdgeDisabledError: edge_ocr but no edge_computers configured.
        ConfirmationNotFoundError: external_ocr_confirm with unknown id.
    """
    # Pre-flight 1: file must exist in vault_files (master table).
    # M1 fix (2026-07-11): the check was against ocr_pending, but
    # vault_files is the master table. Files with high-confidence text
    # (>= 0.85) have no ocr_pending row, so /editText on those files
    # raised FileNotFoundError_ even though the file exists. Now we
    # check vault_files first.
    async with db.conn.execute("SELECT 1 FROM vault_files WHERE file_id=?", (file_id,)) as cur:
        vf_row = await cur.fetchone()
    if vf_row is None:
        raise FileNotFoundError_(f"file_id {file_id!r} not found in vault_files")

    # Pre-flight 2: ocr_pending row. M1 fix: /editText is the only
    # action that works on files WITHOUT an ocr_pending row (because
    # edit_text updates vault_files, not ocr_pending). Other actions
    # (accept_null, external_ocr_*, edge_ocr, skip) require an
    # ocr_pending row — they're operations on the review queue.
    row = await ocr_repo.get(file_id)
    if row is None and action != "edit_text":
        raise FileNotFoundError_(
            f"file_id {file_id!r} has no ocr_pending row (action {action!r} requires review queue)"
        )

    if action == "accept_null":
        return await _decide_accept_null(ocr_repo, row, user_id)
    elif action == "edit_text":
        return await _decide_edit_text(db, ocr_repo, file_id, row, user_id, text, edit_text_limit)
    elif action == "external_ocr_request":
        return await _decide_external_ocr_request(
            ocr_repo,
            row,
            user_id,
            external_ocr_daily_limit,
        )
    elif action == "external_ocr_confirm":
        return await _decide_external_ocr_confirm(
            db,
            ocr_repo,
            row,
            user_id,
            confirmation_id,
            model,
            confidence,
            text=text,
        )
    elif action == "edge_ocr":
        return await _decide_edge_ocr(db, ocr_repo, row, user_id, edge_coord)
    elif action == "skip":
        return await _decide_skip(ocr_repo, row, user_id)
    else:
        raise DecisionError(f"unknown action: {action!r}")


# ---------------------------------------------------------------------------
# Per-action implementations
# ---------------------------------------------------------------------------


async def _decide_accept_null(
    ocr_repo: OcrPendingRepo,
    row: Any,
    user_id: int,
) -> DecisionResult:
    if row.status not in ACCEPT_NULL_SOURCES:
        raise InvalidStatusError(row.file_id, row.status, ACCEPT_NULL_SOURCES)
    prev = row.status
    await ocr_repo.update_status(row.file_id, "accepted_null")
    ts = _now()
    _log_audit("accept_null", row.file_id, prev, "accepted_null", user_id)
    return DecisionResult(file_id=row.file_id, status="accepted_null", ts=ts)


async def _decide_edit_text(
    db: Database,
    ocr_repo: OcrPendingRepo,
    file_id: str,
    row: Any | None,
    user_id: int,
    text: str | None,
    limit: int,
) -> DecisionResult:
    """M1 fix (2026-07-11): /editText works on vault_files even if no
    ocr_pending row exists.

    Args:
        row: ocr_pending row, or None if the file is high-confidence
            (no OCR review needed). The text update goes to vault_files
            regardless. The ocr_pending status update is conditional.
    """
    if text is None or not text.strip():
        raise TextEmptyError("edit_text text is empty")
    if len(text) > limit:
        raise TextTooLongError(text_len=len(text), limit=limit)
    prev = row.status if row is not None else "n/a"
    # Update vault_files: text + source + version. text_version is TEXT
    # (per migration 17.5), so we set it to a new string tier rather than
    # numeric addition. The format is "{prev}_manual_{ts_short}" so it's
    # monotonically increasing per manual edit (rough ordering, not strict).
    import time

    version_tag = f"v{file_id[:4]}_manual_{int(time.time())}"
    async with db.conn.execute(
        "UPDATE vault_files SET text=?, text_source='manual', text_version=? WHERE file_id=?",
        (text, version_tag, file_id),
    ) as cur:
        await cur.fetchall()
    await db.conn.commit()
    # Update ocr_pending status if a row exists. If not, the edit
    # succeeds on vault_files (the master table) and the audit log
    # captures prev="n/a" (no prior review state).
    if row is not None:
        await ocr_repo.update_status(file_id, "manually_edited")
    ts = _now()
    _log_audit(
        "edit_text",
        file_id,
        prev,
        "manually_edited",
        user_id,
        text_len=len(text),
    )
    return DecisionResult(
        file_id=file_id,
        status="manually_edited",
        ts=ts,
        text_len=len(text),
    )


async def _decide_external_ocr_request(
    ocr_repo: OcrPendingRepo,
    row: Any,
    user_id: int,
    daily_limit: int,
) -> DecisionResult:
    # Rate limit check (caller must pass Settings.external_ocr_daily_limit)
    _check_rate_limit(user_id, daily_limit)
    # Create confirmation entry. Key = composite per TDD §4.3.1 line 720:
    # f"{user_id}:{file_id}". Mobile-typing friendly: user only types
    # `yes <file_id>`, the handler reconstructs the composite from
    # message.from_user.id + parsed file_id.
    _gc_confirmations()
    cid = f"{user_id}:{row.file_id}"
    expires_at = _now() + timedelta(seconds=EXTERNAL_OCR_CONFIRMATION_TTL_S)
    _confirmations[cid] = expires_at
    # Don't change status yet -- step 2 (confirm) does that.
    _log_audit(
        "external_ocr_request",
        row.file_id,
        row.status,
        row.status,
        user_id,
        model="MiniMax-M3",
    )
    return DecisionResult(
        file_id=row.file_id,
        status=row.status,
        ts=_now(),
        confirmation_id=cid,
        expires_at=expires_at,
    )


async def _decide_external_ocr_confirm(
    db: Database,
    ocr_repo: OcrPendingRepo,
    row: Any,
    user_id: int,
    confirmation_id: str | None,
    model: str | None,
    confidence: float | None,
    text: str | None = None,
) -> DecisionResult:
    if confirmation_id is None:
        raise ConfirmationNotFoundError("confirmation_id required")
    _gc_confirmations()
    # Composite key check (B2 fix): the cid is f"{user_id}:{file_id}".
    # - If caller passes a cid for a different user_id -> cid doesn't
    #   match any dict key -> ConfirmationNotFoundError. (User A cannot
    #   forge User B's cid because user_id is server-side, not
    #   user-controlled.)
    # - If caller passes a cid for a different file_id -> same: cid
    #   doesn't match -> ConfirmationNotFoundError.
    # - If composite key is malformed (no colon, wrong arity) -> not in
    #   dict -> ConfirmationNotFoundError.
    entry_exp = _confirmations.get(confirmation_id)
    if entry_exp is None:
        raise ConfirmationNotFoundError(f"confirmation_id {confirmation_id!r} not found or expired")
    if entry_exp < _now():
        del _confirmations[confirmation_id]
        raise ConfirmationNotFoundError(f"confirmation_id {confirmation_id!r} expired")
    # Defense in depth: the caller's user_id must match the one encoded
    # in the composite key. Same outcome as a dict miss (because the
    # composite key includes user_id), but explicit for clarity.
    expected_cid = f"{user_id}:{row.file_id}"
    if confirmation_id != expected_cid:
        raise ConfirmationNotFoundError(
            f"confirmation_id {confirmation_id!r} does not match "
            f"user {user_id} + file {row.file_id!r}"
        )
    # Valid. Apply: update ocr_pending with external_* fields + vault_files.text.
    if model is None:
        model = "MiniMax-M3"
    if confidence is None:
        confidence = 0.0
    # Use update_status to set status + fields (relies on whitelist)
    await ocr_repo.update_status(
        row.file_id,
        "external_processed",
        external_model=model,
        external_confidence=confidence,
    )
    # Also persist the OCR text in vault_files (per TDD §4.3.1 /externalOCR
    # postcondition). The text is passed by the handler (after the model
    # produced it). If not passed, we don't touch vault_files.text (caller
    # can re-edit or we accept the gap -- /externalOCR result is implicit
    # in ocr_pending.external_model + external_confidence).
    if text is not None and text.strip():
        import time

        version_tag = f"v_external_{int(time.time())}"
        async with db.conn.execute(
            "UPDATE vault_files SET text=?, text_source='vlm_hosted', "
            "text_version=? WHERE file_id=?",
            (text, version_tag, row.file_id),
        ) as cur:
            await cur.fetchall()
        await db.conn.commit()
    # Pop confirmation (one-shot)
    _confirmations.pop(confirmation_id, None)
    prev = row.status
    _log_audit(
        "external_ocr_confirm",
        row.file_id,
        prev,
        "external_processed",
        user_id,
        model=model,
        confidence=confidence,
    )
    return DecisionResult(file_id=row.file_id, status="external_processed", ts=_now())


async def _decide_edge_ocr(
    db: Database,
    ocr_repo: OcrPendingRepo,
    row: Any,
    user_id: int,
    edge_coord: EdgeCoordinator | None,
) -> DecisionResult:
    if edge_coord is None:
        raise EdgeDisabledError("edge_computers env var empty; edge disabled")
    # Fetch the file path from vault_files (OcrPendingRow doesn't carry it).
    async with db.conn.execute(
        "SELECT source_path FROM vault_files WHERE file_id=?",
        (row.file_id,),
    ) as cur:
        path_row = await cur.fetchone()
    if path_row is None or not path_row[0]:
        raise DecisionError(f"file_id {row.file_id!r} has no source_path in vault_files")
    source_path = path_row[0]
    # Call edge coordinator. enqueue() is async.
    ok = await edge_coord.enqueue(
        file_id=row.file_id,
        path=source_path,
        local_confidence=row.local_confidence,
    )
    if not ok:
        raise EdgeUnavailableError("edge coordinator enqueue returned False (PC offline?)")
    prev = row.status
    await ocr_repo.update_status(row.file_id, "edge_queued")
    _log_audit("edge_ocr", row.file_id, prev, "edge_queued", user_id)
    return DecisionResult(file_id=row.file_id, status="edge_queued", ts=_now())


async def _decide_skip(
    ocr_repo: OcrPendingRepo,
    row: Any,
    user_id: int,
) -> DecisionResult:
    # Idempotent: skip from any state (including terminal) is valid.
    # If already user_skipped, no DB write needed.
    prev = row.status
    if prev != "user_skipped":
        await ocr_repo.update_status(row.file_id, "user_skipped")
    _log_audit("skip", row.file_id, prev, "user_skipped", user_id)
    return DecisionResult(file_id=row.file_id, status="user_skipped", ts=_now())


__all__ = [
    "ACCEPT_NULL_SOURCES",
    "EDIT_TEXT_TELEGRAM_MAX_CHARS",
    "EDIT_TEXT_WEBUI_MAX_CHARS",
    "EXTERNAL_OCR_CONFIRMATION_TTL_S",
    "EXTERNAL_OCR_DAILY_LIMIT_DEFAULT",
    "ConfirmationNotFoundError",
    "DecisionError",
    "DecisionResult",
    "EdgeDisabledError",
    "EdgeUnavailableError",
    "FileNotFoundError_",
    "InvalidStatusError",
    "RateLimitedError",
    "TextEmptyError",
    "TextTooLongError",
    "decide",
    "logger",
]
