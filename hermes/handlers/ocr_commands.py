"""Telegram handlers for OCR user commands (Sprint 19 Slice 4d).

Implements the 6 OCR-related user commands defined in TDD §4.3.1:
- /pendingOCR [file_id] -- list or detail
- /acceptNull <file_id> -- mark as accepted_null
- /editText <file_id> <text> -- manually edit OCR text
- /externalOCR <file_id> -- 2-step confirmation for hosted VLM
  (with inline buttons + free-text fallback)
- /edgeOCR <file_id> -- re-queue to edge PC
- /skipOCR <file_id> -- mark as user_skipped
- yes <file_id> -- free-text confirmation for /externalOCR step 2

All handlers delegate to `hermes.memory.ocr_decision.decide()` for the
core state transition logic. The handlers are responsible for:
- Parsing command args from message.text
- Calling decide() with the right kwargs
- Formatting the response (Markdown)
- Catching DecisionError and reporting user-friendly errors

NORTH STAR: this module is reachable only by the Telegram user. The
LLM agent never imports it directly. (See TDD §4.3.1.)

B1 fix (2026-07-11): /externalOCR now renders inline buttons (Gemini-
safe byte budget: 64-byte Telegram limit) and a free-text `yes <file_id>`
fallback. Without this, the 2-step flow was broken end-to-end: the
user's confirmation reply fell to the LLM chat handler and the
confirmation was lost after 60s GC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from hermes.llm.ocr import OcrError, OcrProvider
from hermes.memory.ocr_decision import (
    EDIT_TEXT_TELEGRAM_MAX_CHARS,
    ConfirmationNotFoundError,
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


# Telegram callback_data byte budget (Gemini 2026-07-11): 64 bytes max.
# Our format: "ext_ok:<user_id>:<file_id>" or "ext_no:<user_id>:<file_id>".
# Breakdown with 10-digit user_id + 32-char file_id:
#   ext_ok:  7+1+10+1+32 = 51 bytes (margin: 13)
#   ext_no:  7+1+10+1+32 = 51 bytes (margin: 13)
_EXT_OK_PREFIX = "ext_ok:"
_EXT_NO_PREFIX = "ext_no:"


def build_ocr_command_router(
    bot: Bot,
    db: Database,
    ocr_repo: OcrPendingRepo,
    edge_coord: EdgeCoordinator | None,
    settings: Settings,
    ocr_provider: OcrProvider,
) -> Router:
    router = Router(name="ocr_commands")

    # ----------------------------------------------------------------
    # /pendingOCR [file_id]
    # ----------------------------------------------------------------

    @router.message(Command("pendingOCR"))
    async def cmd_pending_ocr(message: Message) -> None:
        text = (message.text or "").strip()
        parts = text.split(maxsplit=1)
        file_id = parts[1] if len(parts) > 1 else None

        if file_id:
            # Detail
            row = await ocr_repo.get(file_id)
            if row is None:
                await message.answer(f"No existe `{file_id}` en vault_files.")
                return
            local_text = (row.local_text or "")[:500]
            await message.answer(
                f"*OCR pending detail*\n"
                f"`file_id`: `{row.file_id}`\n"
                f"`status`: `{row.status}`\n"
                f"`local_confidence`: {row.local_confidence}\n"
                f"`local_model`: `{row.local_model}`\n"
                f"`local_text` (truncated 500):\n```\n{local_text}\n```"
            )
        else:
            # List (read-only, no state change)
            pending = await ocr_repo.list_by_status("pending_review", limit=20)
            edge_failed = await ocr_repo.list_by_status("edge_failed", limit=20)
            total = len(pending) + len(edge_failed)
            if total == 0:
                await message.answer("No hay archivos pendientes de OCR.")
                return
            lines = [f"*Archivos pendientes de OCR ({total}):*"]
            for row in pending[:10]:
                lines.append(f"- `{row.file_id[:12]}...` conf={row.local_confidence:.2f} pending")
            for row in edge_failed[:10]:
                lines.append(
                    f"- `{row.file_id[:12]}...` conf={row.local_confidence:.2f} edge_failed"
                )
            if total > 20:
                lines.append(f"\n_(mostrando 20 de {total})_")
            await message.answer("\n".join(lines))

    # ----------------------------------------------------------------
    # /acceptNull <file_id>
    # ----------------------------------------------------------------

    @router.message(Command("acceptNull"))
    async def cmd_accept_null(message: Message) -> None:
        file_id = _extract_file_id(message)
        if file_id is None:
            await message.answer("Uso: `/acceptNull <file_id>`")
            return
        user_id = _user_id(message)
        try:
            result = await decide(
                db=db,
                ocr_repo=ocr_repo,
                edge_coord=edge_coord,
                file_id=file_id,
                action="accept_null",
                user_id=user_id,
            )
            await message.answer(
                f"OK: `{result.file_id}` marcado como `accepted_null`.\n"
                f"Texto queda NULL (no OCR result)."
            )
        except FileNotFoundError_:
            await message.answer(f"No existe `{file_id}` en vault_files.")
        except InvalidStatusError as e:
            await message.answer(
                f"Estado invalido para `/acceptNull`.\n"
                f"Actual: `{e.current}`\n"
                f"Valido: `{'`, `'.join(e.valid_sources)}`"
            )

    # ----------------------------------------------------------------
    # /editText <file_id> <text>
    # M1 fix: works for files in vault_files even without ocr_pending row
    # ----------------------------------------------------------------

    @router.message(Command("editText"))
    async def cmd_edit_text(message: Message) -> None:
        # Parse: skip command name, first remaining token = file_id, rest = text
        text = (message.text or "").strip()
        # Remove the command name (with optional @botusername suffix)
        for prefix in ("/editText@", "/editText"):
            if text.startswith(prefix):
                text = text[len(prefix) :].lstrip()
                break
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await message.answer(
                f"Uso: `/editText <file_id> <texto corregido>`\n"
                f"Max {EDIT_TEXT_TELEGRAM_MAX_CHARS} chars (limite Telegram)."
            )
            return
        file_id = parts[0].strip()
        new_text = parts[1]
        user_id = _user_id(message)
        try:
            result = await decide(
                db=db,
                ocr_repo=ocr_repo,
                edge_coord=edge_coord,
                file_id=file_id,
                action="edit_text",
                user_id=user_id,
                text=new_text,
            )
            await message.answer(
                f"OK: `{result.file_id}` texto actualizado "
                f"({result.text_len} chars, version bumped)."
            )
        except FileNotFoundError_:
            await message.answer(f"No existe `{file_id}` en vault_files.")
        except TextEmptyError:
            await message.answer("Texto vacio. Envia algo despues del file_id.")
        except TextTooLongError as e:
            await message.answer(
                f"Texto demasiado largo ({e.text_len} > {e.limit} chars). "
                f"Acorta o edita el archivo original y re-dropea."
            )

    # ----------------------------------------------------------------
    # /externalOCR <file_id>  (2-step confirmation with inline buttons)
    # ----------------------------------------------------------------

    async def _do_external_ocr_confirm(
        message: Message,
        user_id: int,
        file_id: str,
    ) -> None:
        """Apply /externalOCR step 2: invoke OCR provider, persist result.

        Closure: captures db, ocr_repo, edge_coord, ocr_provider from
        build_ocr_command_router scope. Called from both the inline
        button callback (on_ext_ok) and the free-text 'yes <file_id>'
        handler (cmd_yes). Same logic for both.
        """
        # 1. Get the file path from vault_files (master table)
        async with db.conn.execute(
            "SELECT source_path FROM vault_files WHERE file_id=?", (file_id,)
        ) as cur:
            vf_row = await cur.fetchone()
        if vf_row is None or not vf_row[0]:
            await message.answer(f"No existe `{file_id}` en vault_files.")
            return
        file_path = vf_row[0]

        # 2. Invoke the OcrProvider (provider-agnostic, configured via Settings)
        try:
            ocr_result = await ocr_provider.ocr(file_path, file_id)
        except OcrError as e:
            await message.answer(
                f"OCR fallo ({e.provider}): {e}\n" f"Re-intenta o contacta al admin si persiste."
            )
            return

        # 3. Apply via decide() with the composite key (B2 fix)
        cid = f"{user_id}:{file_id}"
        try:
            await decide(
                db=db,
                ocr_repo=ocr_repo,
                edge_coord=edge_coord,
                file_id=file_id,
                action="external_ocr_confirm",
                user_id=user_id,
                confirmation_id=cid,
                model=ocr_result.model,
                confidence=ocr_result.confidence,
                text=ocr_result.text,
            )
        except ConfirmationNotFoundError:
            await message.answer("Confirmation expired (60s). Re-invoke `/externalOCR <file_id>`.")
            return
        except FileNotFoundError_:
            await message.answer(f"No existe `{file_id}` en vault_files.")
            return

        # 4. Reply with the OCR result (truncated for Telegram 4096 limit)
        truncated = ocr_result.text[:500] if ocr_result.text else "(empty)"
        await message.answer(
            f"OK: `{file_id}` procesado.\n"
            f"Modelo: `{ocr_result.model}` ({ocr_result.latency_ms}ms)\n"
            f"Texto ({len(ocr_result.text)} chars):\n```\n{truncated}\n```"
        )

    @router.message(Command("externalOCR"))
    async def cmd_external_ocr(message: Message) -> None:
        # Step 1: request confirmation
        file_id = _extract_file_id(message)
        if file_id is None:
            await message.answer(
                "Uso: `/externalOCR <file_id>`\n"
                "Esto envia tu archivo a un VLM hosted (NORTH STAR opt-in).\n"
                "Recibiras un prompt de confirmacion antes de enviar."
            )
            return
        user_id = _user_id(message)
        try:
            result = await decide(
                db=db,
                ocr_repo=ocr_repo,
                edge_coord=edge_coord,
                file_id=file_id,
                action="external_ocr_request",
                user_id=user_id,
                external_ocr_daily_limit=settings.external_ocr_daily_limit,
            )
            # message.date may be None in tests; use now() as fallback.
            import datetime as _dt

            msg_ts = message.date.timestamp() if message.date else _dt.datetime.now().timestamp()
            expires_in = int((result.expires_at.timestamp() if result.expires_at else 0) - msg_ts)
            # B1 fix: render inline buttons (Telegram byte budget OK).
            # callback_data: "ext_ok:<user_id>:<file_id>" = 51 bytes max
            # (7+1+10+1+32). Margin: 13 bytes under 64-byte limit.
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text=f"Enviar a {ocr_provider.name}",
                            callback_data=f"{_EXT_OK_PREFIX}{user_id}:{file_id}",
                        ),
                        InlineKeyboardButton(
                            text="Cancelar",
                            callback_data=f"{_EXT_NO_PREFIX}{user_id}:{file_id}",
                        ),
                    ],
                ],
            )
            await message.answer(
                f"Enviar `{file_id}` a `{ocr_provider.name}`?\n"
                f"Responde `yes {file_id}` en {expires_in}s para confirmar "
                f"o pulsa un boton.\n"
                f"Responde `no` para cancelar.",
                reply_markup=keyboard,
            )
        except FileNotFoundError_:
            await message.answer(f"No existe `{file_id}` en vault_files.")
        except RateLimitedError:
            await message.answer(
                f"Has alcanzado el limite diario "
                f"({settings.external_ocr_daily_limit} requests/dia). "
                f"Resetea a las 00:00 UTC."
            )

    # ----------------------------------------------------------------
    # Callback: [Enviar a hosted_llm] button
    # ----------------------------------------------------------------

    @router.callback_query(lambda c: bool(c.data and c.data.startswith(_EXT_OK_PREFIX)))
    async def on_ext_ok(callback_query: CallbackQuery) -> None:
        # Parse "ext_ok:<user_id>:<file_id>"
        parts = (callback_query.data or "").split(":", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2]:
            await callback_query.answer("Invalid button payload", show_alert=True)
            return
        cb_user_id = int(parts[1])
        file_id = parts[2]
        # callback_query.message is Message | InaccessibleMessage | None.
        msg = callback_query.message
        if msg is None:
            await callback_query.answer("No message context", show_alert=True)
            return
        # Security: callback is for the user who clicked the button.
        # message.from_user is the clicker; cb_user_id is the original
        # requester (from the message that rendered the button).
        msg_user_id = _user_id(msg)  # type: ignore[arg-type]
        if msg_user_id != cb_user_id:
            await callback_query.answer("Not for you", show_alert=True)
            return
        await _do_external_ocr_confirm(msg, cb_user_id, file_id)  # type: ignore[arg-type]
        await callback_query.answer()

    # ----------------------------------------------------------------
    # Callback: [Cancelar] button
    # ----------------------------------------------------------------

    @router.callback_query(lambda c: bool(c.data and c.data.startswith(_EXT_NO_PREFIX)))
    async def on_ext_no(callback_query: CallbackQuery) -> None:
        # CallbackQuery.message is Message | InaccessibleMessage | None.
        # In practice, for fresh /externalOCR messages it's a real Message
        # that supports edit_text. Fall back to answer() if not.
        msg = callback_query.message
        if msg is None:
            await callback_query.answer("Cancelled.")
            return
        try:
            await msg.edit_text("Cancelled.")  # type: ignore[union-attr]
        except Exception:
            # edit_text can fail if message is too old or was deleted
            await msg.answer("Cancelled.")
        await callback_query.answer()

    # ----------------------------------------------------------------
    # Free-text fallback: "yes <file_id>"
    # Used when inline buttons are unavailable (e.g., old Telegram
    # clients). Same end-state as the button click.
    # ----------------------------------------------------------------

    @router.message(lambda m: bool(m.text and m.text.lower().startswith("yes ")))
    async def cmd_yes(message: Message) -> None:
        text = (message.text or "").strip()
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            await message.answer("Uso: `yes <file_id>`")
            return
        file_id = parts[1].strip()
        user_id = _user_id(message)
        await _do_external_ocr_confirm(message, user_id, file_id)

    # ----------------------------------------------------------------
    # /edgeOCR <file_id>
    # ----------------------------------------------------------------

    @router.message(Command("edgeOCR"))
    async def cmd_edge_ocr(message: Message) -> None:
        file_id = _extract_file_id(message)
        if file_id is None:
            await message.answer("Uso: `/edgeOCR <file_id>`")
            return
        user_id = _user_id(message)
        try:
            result = await decide(
                db=db,
                ocr_repo=ocr_repo,
                edge_coord=edge_coord,
                file_id=file_id,
                action="edge_ocr",
                user_id=user_id,
            )
            await message.answer(
                f"OK: `{result.file_id}` encolado al edge. "
                f"Se procesara en el siguiente probe cycle del PC."
            )
        except FileNotFoundError_:
            await message.answer(f"No existe `{file_id}` en vault_files.")
        except EdgeDisabledError:
            await message.answer("Edge deshabilitado. Configura `EDGE_COMPUTERS` en .env.")
        except EdgeUnavailableError:
            await message.answer(
                "Edge no disponible (PC offline o encolamiento fallo). "
                "Re-intenta cuando el PC este online."
            )

    # ----------------------------------------------------------------
    # /skipOCR <file_id>
    # ----------------------------------------------------------------

    @router.message(Command("skipOCR"))
    async def cmd_skip_ocr(message: Message) -> None:
        file_id = _extract_file_id(message)
        if file_id is None:
            await message.answer("Uso: `/skipOCR <file_id>`")
            return
        user_id = _user_id(message)
        try:
            result = await decide(
                db=db,
                ocr_repo=ocr_repo,
                edge_coord=edge_coord,
                file_id=file_id,
                action="skip",
                user_id=user_id,
            )
            await message.answer(f"OK: `{result.file_id}` marcado como `user_skipped`.")
        except FileNotFoundError_:
            await message.answer(f"No existe `{file_id}` en vault_files.")

    return router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_id(message: Message) -> int:
    """Extract user_id from message, or 0 if missing (system msg)."""
    if message.from_user is None:
        return 0
    return message.from_user.id


def _extract_file_id(message: Message) -> str | None:
    """Extract file_id from command message. Returns None if no arg."""
    text = (message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        return None
    # Strip the bot username if present (e.g., /acceptNull@botusername)
    fid = parts[1].strip()
    return fid
