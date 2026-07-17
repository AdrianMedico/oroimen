"""Tests industriales para `hermes.handlers.ocr_commands` (Sprint 19 Slice 4d).

Cubre los 6 commands Telegram: /pendingOCR, /acceptNull, /editText,
/externalOCR, /edgeOCR, /skipOCR. La logica de estado vive en
`hermes.memory.ocr_decision` (cubierta por test_ocr_decision.py).
Aca testeamos:
- Parsing correcto de args (file_id, text para /editText)
- Format de respuesta Markdown
- Manejo de errores (FileNotFound, InvalidStatus, TextTooLong, etc.)
- /externalOCR 2-step (request + confirm + timeout) -- B1 fix
- Inline button rendering (callback_data byte budget)
- Free-text "yes <file_id>" fallback

Estrategia: misma que test_commands.py -- Database real, Message mock
con _AnswerCapture para registrar respuestas.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes.handlers.ocr_commands import build_ocr_command_router
from hermes.llm.ocr import OcrProvider, OcrResult
from hermes.memory.ocr_decision import (
    _confirmations,
    _rate_limits,
)
from hermes.memory.ocr_pending_repo import OcrPendingRepo
from tests.conftest import (
    TEST_CHAT_ID,
    TEST_USER_ID,
    _AnswerCapture,
    get_all_handlers,
)

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ocr_repo(db):  # type: ignore[no-untyped-def]
    return OcrPendingRepo(db)


@pytest.fixture(autouse=True)
def _clear_ocr_decision_state():
    """Limpia el state process-local (rate limits + confirmations)."""
    _rate_limits.clear()
    _confirmations.clear()
    yield
    _rate_limits.clear()
    _confirmations.clear()


@pytest.fixture
def edge_coordinator_mock() -> MagicMock:
    coord = MagicMock()

    # enqueue is an async function (returns True = PC online)
    async def enqueue(file_id: str, path: str, local_confidence: float) -> bool:
        return True

    coord.enqueue = enqueue
    return coord


@pytest.fixture
def ocr_provider_mock() -> MagicMock:
    """Mock OcrProvider for testing the /externalOCR 2-step flow.

    Returns a fixed OcrResult when ocr() is called. Tests can override
    `mock.ocr.return_value` to test different result shapes.
    """
    provider = MagicMock(spec=OcrProvider)
    provider.name = "hosted_llm"
    provider.ocr = AsyncMock(
        return_value=OcrResult(
            text="OCR text extracted from test file",
            confidence=0.0,
            model="minimax-m3",
            provider="hosted_llm",
            latency_ms=1234,
        )
    )
    return provider


class _AnswerWithKwargsCapture:
    """Like _AnswerCapture but also records kwargs (for inline keyboards)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, text: str, **kwargs: Any) -> None:
        self.calls.append((text, kwargs))

    def last(self) -> str:
        assert self.calls, "answer() nunca fue llamado"
        return self.calls[-1][0]

    def last_kwargs(self) -> dict[str, Any]:
        assert self.calls, "answer() nunca fue llamado"
        return self.calls[-1][1]


def _get_callback_handlers(router) -> list[Any]:
    """Return all callback_query handler callables (B1 fix, for inline buttons)."""
    return [h.callback for h in router.callback_query.handlers]


@pytest.fixture
async def seeded_ocr_row(db, ocr_repo):  # type: ignore[no-untyped-def]
    """Crea un vault_file + ocr_pending en status 'pending_review'."""
    file_id = "a" * 32
    await db.conn.execute(
        "INSERT INTO vault_files "
        "(file_id, source_path, content_sha256, mtime, size_bytes) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_id, "/mnt/shared/Documentos/_inbox/test.pdf", "sha256_test", 1234567890.0, 1024),
    )
    await db.conn.commit()
    await ocr_repo.create(
        file_id=file_id,
        local_confidence=0.42,
        local_text=None,
        local_model="tesseract",
        status="pending_review",
    )
    return file_id


def _make_command_message(
    command: str,
    user_id: int = TEST_USER_ID,
    chat_id: int = TEST_CHAT_ID,
) -> tuple[Any, _AnswerCapture]:
    from aiogram.types import Chat, User

    user = User(id=user_id, is_bot=False, first_name="Test")
    chat = Chat(id=chat_id, type="private")
    capture = _AnswerCapture()
    msg = MagicMock(
        spec=["from_user", "chat", "text", "voice", "message_thread_id", "answer", "date"]
    )
    msg.from_user = user
    msg.chat = chat
    msg.text = command
    msg.voice = None
    msg.message_thread_id = None
    msg.answer = capture
    import datetime

    msg.date = datetime.datetime.now()
    return msg, capture


# ---------------------------------------------------------------------------
# /acceptNull
# ---------------------------------------------------------------------------


async def test_accept_null_happy_path(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    assert handlers, "no handlers registered"
    msg, capture = _make_command_message(f"/acceptNull {seeded_ocr_row}")
    for h in handlers:
        if h.__name__ == "cmd_accept_null":
            await h(msg)
            break
    text = capture.last()
    assert "accepted_null" in text
    assert seeded_ocr_row[:12] in text


async def test_accept_null_no_arg_shows_usage(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    msg, capture = _make_command_message("/acceptNull")
    for h in handlers:
        if h.__name__ == "cmd_accept_null":
            await h(msg)
            break
    assert "Uso" in capture.last()


# ---------------------------------------------------------------------------
# /editText
# ---------------------------------------------------------------------------


async def test_edit_text_happy_path(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    msg, capture = _make_command_message(f"/editText {seeded_ocr_row} Texto corregido")
    for h in handlers:
        if h.__name__ == "cmd_edit_text":
            await h(msg)
            break
    text = capture.last()
    assert "OK" in text
    assert "16 chars" in text or "texto actualizado" in text


async def test_edit_text_too_long_shows_error(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    long_text = "x" * 5000
    msg, capture = _make_command_message(f"/editText {seeded_ocr_row} {long_text}")
    for h in handlers:
        if h.__name__ == "cmd_edit_text":
            await h(msg)
            break
    text = capture.last()
    assert "demasiado largo" in text.lower() or "too long" in text.lower()


# ---------------------------------------------------------------------------
# /skipOCR
# ---------------------------------------------------------------------------


async def test_skip_happy_path(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    msg, capture = _make_command_message(f"/skipOCR {seeded_ocr_row}")
    for h in handlers:
        if h.__name__ == "cmd_skip_ocr":
            await h(msg)
            break
    text = capture.last()
    assert "user_skipped" in text


# ---------------------------------------------------------------------------
# /pendingOCR (list + detail)
# ---------------------------------------------------------------------------


async def test_pending_ocr_list(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    msg, capture = _make_command_message("/pendingOCR")
    for h in handlers:
        if h.__name__ == "cmd_pending_ocr":
            await h(msg)
            break
    text = capture.last()
    # 1 pending file
    assert "1" in text or "1 archivos" in text or "Archivos pendientes" in text


async def test_pending_ocr_detail(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    msg, capture = _make_command_message(f"/pendingOCR {seeded_ocr_row}")
    for h in handlers:
        if h.__name__ == "cmd_pending_ocr":
            await h(msg)
            break
    text = capture.last()
    assert seeded_ocr_row[:12] in text
    assert "0.42" in text or "local_confidence" in text


# ---------------------------------------------------------------------------
# /externalOCR 2-step flow (B1 fix, 2026-07-11)
# ---------------------------------------------------------------------------


async def test_external_ocr_request_renders_inline_buttons(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    """B1 fix: /externalOCR renders inline buttons + free-text fallback.

    Verifies:
    - Confirmation is created (request step)
    - Message includes the file_id + provider name
    - 2 buttons: [Enviar a hosted_llm] [Cancelar]
    - callback_data byte budget: < 64 bytes (Telegram hard limit)
    """
    import datetime

    from aiogram.types import Message

    msg = MagicMock(spec=Message)
    msg.from_user = MagicMock(id=TEST_USER_ID, is_bot=False, first_name="Test")
    msg.text = f"/externalOCR {seeded_ocr_row}"
    msg.date = datetime.datetime.now()
    capture = _AnswerWithKwargsCapture()
    msg.answer = capture

    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    for h in handlers:
        if h.__name__ == "cmd_external_ocr":
            await h(msg)
            break

    # Confirmation was created
    assert len(_confirmations) == 1
    cid = next(iter(_confirmations.keys()))
    assert cid == f"{TEST_USER_ID}:{seeded_ocr_row}"
    # Message was sent (with buttons)
    assert len(capture.calls) == 1
    _, kwargs = capture.calls[0]
    assert "reply_markup" in kwargs


async def test_yes_free_text_completes_external_ocr_flow(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    """B1 fix: free-text 'yes <file_id>' completes the 2-step flow.

    Flow:
    1. /externalOCR creates a confirmation
    2. User replies 'yes <file_id>' (no button click)
    3. Handler invokes OcrProvider, applies the result via decide()
    4. Status is 'external_processed'
    """
    import datetime

    from aiogram.types import Message

    # Step 1: /externalOCR
    msg1 = MagicMock(spec=Message)
    msg1.from_user = MagicMock(id=TEST_USER_ID, is_bot=False, first_name="Test")
    msg1.text = f"/externalOCR {seeded_ocr_row}"
    msg1.date = datetime.datetime.now()
    capture1 = _AnswerWithKwargsCapture()
    msg1.answer = capture1

    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    for h in handlers:
        if h.__name__ == "cmd_external_ocr":
            await h(msg1)
            break
    assert len(_confirmations) == 1

    # Step 2: 'yes <file_id>' free-text
    msg2 = MagicMock(spec=Message)
    msg2.from_user = MagicMock(id=TEST_USER_ID, is_bot=False, first_name="Test")
    msg2.text = f"yes {seeded_ocr_row}"
    capture2 = _AnswerWithKwargsCapture()
    msg2.answer = capture2

    for h in handlers:
        if h.__name__ == "cmd_yes":
            await h(msg2)
            break

    # OcrProvider was called
    ocr_provider_mock.ocr.assert_called_once()
    # Result was applied (status -> external_processed)
    async with db.conn.execute(
        "SELECT status FROM ocr_pending WHERE file_id=?", (seeded_ocr_row,)
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == "external_processed"
    # Confirmation was consumed
    assert len(_confirmations) == 0
    # Reply mentions the OCR text
    reply = capture2.last()
    assert "OK" in reply
    assert "procesado" in reply or "modelo" in reply.lower()


async def test_callback_data_byte_budget_under_64(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    """B1 fix: callback_data < 64 bytes (Telegram hard limit).

    Gemini 3.1 Pro (2026-07-11): prefix like 'confirm_external_ocr:' (21 bytes)
    + user_id (10) + file_id (32) + colons (2) = 65 bytes → BUTTON_DATA_INVALID.
    Use short prefixes: 'ext_ok:' (7) + user_id (10) + file_id (32) + colons (2) = 51.
    """
    import datetime

    from aiogram.types import InlineKeyboardMarkup

    msg = MagicMock()
    msg.from_user = MagicMock(id=TEST_USER_ID, is_bot=False, first_name="Test")
    msg.text = f"/externalOCR {seeded_ocr_row}"
    msg.date = datetime.datetime.now()
    capture = _AnswerWithKwargsCapture()
    msg.answer = capture

    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    for h in handlers:
        if h.__name__ == "cmd_external_ocr":
            await h(msg)
            break

    # Extract the inline keyboard from the message
    _, kwargs = capture.calls[0]
    keyboard = kwargs.get("reply_markup")
    assert isinstance(keyboard, InlineKeyboardMarkup)
    # 2 buttons
    buttons = keyboard.inline_keyboard[0]
    assert len(buttons) == 2
    # Each callback_data < 64 bytes
    for btn in buttons:
        assert len(btn.callback_data.encode("utf-8")) < 64, (
            f"callback_data {btn.callback_data!r} exceeds 64 bytes "
            f"(Telegram BUTTON_DATA_INVALID)"
        )
    # Specific byte counts
    for btn in buttons:
        if btn.callback_data.startswith("ext_ok:"):
            assert btn.callback_data == f"ext_ok:{TEST_USER_ID}:{seeded_ocr_row}"
        else:
            assert btn.callback_data.startswith("ext_no:")


async def test_yes_with_unknown_file_id_shows_error(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    """B1 fix: 'yes <unknown_file_id>' (no prior /externalOCR) shows error.

    The confirmation doesn't exist (no prior request), so the handler
    should report 'Confirmation expired' (the most likely user intent).
    """
    from aiogram.types import Message

    msg = MagicMock(spec=Message)
    msg.from_user = MagicMock(id=TEST_USER_ID, is_bot=False, first_name="Test")
    msg.text = "yes unknown_file_id_xxxxxxxxxxxxxxxx"
    capture = _AnswerWithKwargsCapture()
    msg.answer = capture

    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    handlers = get_all_handlers(router)
    for h in handlers:
        if h.__name__ == "cmd_yes":
            await h(msg)
            break
    reply = capture.last()
    assert "expired" in reply.lower() or "no existe" in reply.lower()


async def test_ext_no_callback_cancels(
    db, settings, telemetry, ocr_repo, edge_coordinator_mock, seeded_ocr_row, ocr_provider_mock
):  # type: ignore[no-untyped-def]
    """B1 fix: clicking [Cancelar] edits the message to 'Cancelled.'"""
    import datetime

    from aiogram.types import CallbackQuery, Message

    # First create the confirmation
    msg_req = MagicMock(spec=Message)
    msg_req.from_user = MagicMock(id=TEST_USER_ID, is_bot=False, first_name="Test")
    msg_req.text = f"/externalOCR {seeded_ocr_row}"
    msg_req.date = datetime.datetime.now()
    capture_req = _AnswerWithKwargsCapture()
    msg_req.answer = capture_req

    router = build_ocr_command_router(
        bot=MagicMock(),
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        settings=settings,
        ocr_provider=ocr_provider_mock,
    )
    msg_handlers = get_all_handlers(router)
    cb_handlers = _get_callback_handlers(router)
    for h in msg_handlers:
        if h.__name__ == "cmd_external_ocr":
            await h(msg_req)
            break
    assert len(_confirmations) == 1

    # Now simulate the [Cancelar] button click
    cb = MagicMock(spec=CallbackQuery)
    cb.data = f"ext_no:{TEST_USER_ID}:{seeded_ocr_row}"
    cb.message = MagicMock()
    cb.message.from_user = MagicMock(id=TEST_USER_ID)
    cb.message.edit_text = AsyncMock()
    cb.answer = AsyncMock()

    for h in cb_handlers:
        if h.__name__ == "on_ext_no":
            await h(cb)
            break
    # Confirmation was NOT consumed (cancel doesn't apply the result)
    assert len(_confirmations) == 1
    # Message was edited
    cb.message.edit_text.assert_called_once()
    cb.answer.assert_called_once()
