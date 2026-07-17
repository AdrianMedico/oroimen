"""HTTP API endpoints for OCR user decisions (Sprint 19 Slice 4d).

TDD §4.3.1 WebUI surface:
- GET /v1/ocr/pending?limit=20&offset=0&status=pending_review|edge_failed
- GET /v1/ocr/pending/{file_id}
- POST /v1/ocr/decision  (body: {file_id, action, ...action_specific})
- POST /v1/ocr/decision/confirm  (body: {confirmation_id})

All endpoints delegate to `hermes.memory.ocr_decision.decide()`. The
endpoints are responsible for:
- Parsing + validating request bodies
- Calling decide() with the right kwargs
- Translating DecisionError into appropriate HTTP status codes
- Formatting responses as JSON

Response shape (per TDD §4.3.1):
- Success: {"file_id": ..., "status": ..., "ts": ...}
- Error:   {"error": "code", "message": "human readable", "file_id": ...}

NORTH STAR: this module is reachable only via the WebUI (with auth).
The LLM agent does not call these endpoints.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query

from hermes.memory.ocr_decision import (
    EDIT_TEXT_WEBUI_MAX_CHARS,
    ConfirmationNotFoundError,
    DecisionError,
    EdgeDisabledError,
    EdgeUnavailableError,
    FileNotFoundError_,
    InvalidStatusError,
    RateLimitedError,
    TextEmptyError,
    TextTooLongError,
    decide,
)

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.edge_coordinator import EdgeCoordinator
    from hermes.memory.ocr_pending_repo import OcrPendingRepo

logger = logging.getLogger(__name__)


def build_ocr_api_router(
    db: Database,
    ocr_repo: OcrPendingRepo,
    edge_coord: EdgeCoordinator | None,
    settings: Settings,
) -> APIRouter:
    """Build the FastAPI router for OCR decisions.

    Returns an APIRouter that should be included in the main FastAPI app
    via app.include_router(ocr_router, prefix="/v1/ocr").
    """
    router = APIRouter(prefix="/v1/ocr", tags=["ocr"])

    # ----------------------------------------------------------------
    # GET /v1/ocr/pending?limit=20&offset=0&status=...
    # ----------------------------------------------------------------

    @router.get("/pending")
    async def list_pending(
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        status: str = Query("pending_review", regex="^(pending_review|edge_failed)$"),
    ) -> dict[str, Any]:
        """List files in a given ocr_pending status (read-only)."""
        rows = await ocr_repo.list_by_status(status, limit=limit)
        # offset is applied in Python (no SQL OFFSET in the repo to keep
        # the repo's API minimal). For small `limit` (default 20) this
        # is fine; Sprint 22+ could add SQL OFFSET.
        rows_page = rows[offset : offset + limit]
        now = datetime.now().timestamp()
        return {
            "files": [
                {
                    "file_id": r.file_id,
                    "local_confidence": r.local_confidence,
                    "local_model": r.local_model,
                    "status": r.status,
                    "created_at": r.created_at,
                    "age_hours": round((now - _parse_iso(r.created_at)) / 3600, 1)
                    if r.created_at
                    else None,
                }
                for r in rows_page
            ],
            "total": len(rows),
        }

    # ----------------------------------------------------------------
    # GET /v1/ocr/pending/{file_id}
    # ----------------------------------------------------------------

    @router.get("/pending/{file_id}")
    async def get_pending_detail(file_id: str) -> dict[str, Any]:
        """Detail of a single ocr_pending row + the linked vault_files row."""
        row = await ocr_repo.get(file_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"file_id {file_id!r} not found")
        # Fetch source_path from vault_files
        async with db.conn.execute(
            "SELECT source_path FROM vault_files WHERE file_id=?", (file_id,)
        ) as cur:
            vf = await cur.fetchone()
        source_path = vf[0] if vf else None
        return {
            "file_id": row.file_id,
            "source_path": source_path,
            "status": row.status,
            "local_confidence": row.local_confidence,
            "local_text": row.local_text,
            "local_model": row.local_model,
            "external_model": row.external_model,
            "external_confidence": row.external_confidence,
            "edge_model": row.edge_model,
            "edge_queued_at": row.edge_queued_at,
            "edge_processed_at": row.edge_processed_at,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    # ----------------------------------------------------------------
    # POST /v1/ocr/decision
    # Body: {file_id, action, text?, confirmation_id?, model?, confidence?, user_id?}
    # ----------------------------------------------------------------

    @router.post("/decision")
    async def post_decision(body: dict[str, Any]) -> dict[str, Any]:
        """Apply a user decision. Returns DecisionResult or raises 4xx/5xx."""
        file_id = body.get("file_id")
        action = body.get("action")
        if not file_id or not action:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_params",
                    "message": "file_id and action required",
                },
            )
        user_id = int(body.get("user_id", 0))
        try:
            result = await decide(
                db=db,
                ocr_repo=ocr_repo,
                edge_coord=edge_coord,
                file_id=file_id,
                action=action,
                user_id=user_id,
                text=body.get("text"),
                confirmation_id=body.get("confirmation_id"),
                model=body.get("model"),
                confidence=body.get("confidence"),
                edit_text_limit=EDIT_TEXT_WEBUI_MAX_CHARS,
                external_ocr_daily_limit=settings.external_ocr_daily_limit,
            )
            return {
                "file_id": result.file_id,
                "status": result.status,
                "ts": result.ts.isoformat(),
                "confirmation_id": result.confirmation_id,
                "expires_at": result.expires_at.isoformat() if result.expires_at else None,
                "text_len": result.text_len,
            }
        except FileNotFoundError_ as e:
            raise HTTPException(  # noqa: B904
                status_code=404,
                detail={
                    "error": "file_not_found",
                    "message": str(e),
                    "file_id": file_id,
                },
            )
        except InvalidStatusError as e:
            raise HTTPException(  # noqa: B904
                status_code=409,
                detail={
                    "error": "invalid_status",
                    "message": f"status {e.current!r} not in valid {e.valid_sources}",
                    "file_id": file_id,
                    "current": e.current,
                    "valid_sources": list(e.valid_sources),
                },
            )
        except TextEmptyError as e:
            raise HTTPException(  # noqa: B904
                status_code=400,
                detail={
                    "error": "text_empty",
                    "message": str(e),
                    "file_id": file_id,
                },
            )
        except TextTooLongError as e:
            raise HTTPException(  # noqa: B904
                status_code=413,
                detail={
                    "error": "text_too_long",
                    "message": str(e),
                    "file_id": file_id,
                    "text_len": e.text_len,
                    "limit": e.limit,
                },
            )
        except RateLimitedError as e:
            raise HTTPException(  # noqa: B904
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": str(e),
                    "file_id": file_id,
                },
            )
        except EdgeDisabledError as e:
            raise HTTPException(  # noqa: B904
                status_code=503,
                detail={
                    "error": "edge_disabled",
                    "message": str(e),
                    "file_id": file_id,
                },
            )
        except EdgeUnavailableError as e:
            raise HTTPException(  # noqa: B904
                status_code=503,
                detail={
                    "error": "edge_unavailable",
                    "message": str(e),
                    "file_id": file_id,
                },
            )
        except DecisionError as e:
            raise HTTPException(  # noqa: B904
                status_code=500,
                detail={
                    "error": "decision_error",
                    "message": str(e),
                    "file_id": file_id,
                },
            )

    # ----------------------------------------------------------------
    # POST /v1/ocr/decision/confirm  (2-step external_ocr confirm)
    # Body: {confirmation_id, model, text, confidence, user_id}
    # ----------------------------------------------------------------

    @router.post("/decision/confirm")
    async def post_decision_confirm(body: dict[str, Any]) -> dict[str, Any]:
        """Step 2 of /externalOCR: confirm a pending request with the OCR result."""
        confirmation_id = body.get("confirmation_id")
        file_id = body.get("file_id")  # required to find the OcrPendingRow
        if not confirmation_id or not file_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_params",
                    "message": "confirmation_id and file_id required",
                },
            )
        user_id = int(body.get("user_id", 0))
        try:
            result = await decide(
                db=db,
                ocr_repo=ocr_repo,
                edge_coord=edge_coord,
                file_id=file_id,
                action="external_ocr_confirm",
                user_id=user_id,
                confirmation_id=confirmation_id,
                model=body.get("model", "MiniMax-M3"),
                confidence=body.get("confidence", 0.0),
                text=body.get("text"),
            )
            return {
                "file_id": result.file_id,
                "status": result.status,
                "ts": result.ts.isoformat(),
            }
        except FileNotFoundError_ as e:
            raise HTTPException(  # noqa: B904
                status_code=404,
                detail={
                    "error": "file_not_found",
                    "message": str(e),
                    "file_id": file_id,
                },
            )
        except ConfirmationNotFoundError as e:
            raise HTTPException(  # noqa: B904
                status_code=400,
                detail={"error": "confirmation_not_found", "message": str(e)},
            )
        except DecisionError as e:
            raise HTTPException(  # noqa: B904
                status_code=500,
                detail={
                    "error": "decision_error",
                    "message": str(e),
                    "file_id": file_id,
                },
            )

    return router


def _parse_iso(s: str) -> float:
    """Parse ISO timestamp to unix epoch. Lenient: ignores trailing Z."""
    if not s:
        return 0.0
    try:
        # SQLite uses 'YYYY-MM-DD HH:MM:SS' or with 'T'
        s2 = s.replace("T", " ").replace("Z", "").strip()
        return datetime.fromisoformat(s2).timestamp()
    except (ValueError, TypeError):
        return 0.0
