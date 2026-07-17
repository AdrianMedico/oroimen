"""Tests industriales para `hermes.handlers.commands`.

Cubre los 5 comandos:
- `/start`: bienvenida + log
- `/help`: lista de comandos
- `/clear`: archiva conversación actual + crea nueva + confirma
- `/status`: estado del sistema
- `/chatid`: muestra el chat_id de Telegram (utilidad para setup de
  push notifications S10.4; NO toca BD)

Estrategia de mocking (justificación):
- `Database` real (sqlite en tmp_path) — más fidelidad que un mock.
- `aiogram.Message` se mockea con MagicMock (pydantic frozen impide
  reasignar `answer`).
- `aiogram.Bot` real (aiogram valida el formato del token).
- Cada handler se extrae del Router con `get_handler_at(router, idx)` y se
  invoca directamente con un Message mock. Esto ejecuta la MISMA función
  que corre en producción (con la closure capturada: `db`).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from hermes.handlers.commands import build_command_router

# Las fixtures (settings, db, telemetry, bot) vienen automáticamente de
# conftest.py — pytest las descubre sin necesidad de importarlas.
# Solo importamos los helpers y constantes que pytest NO auto-descubre.
from tests.conftest import (
    TEST_CHAT_ID,
    TEST_USER_ID,
    _AnswerCapture,
    get_all_handlers,
    get_handler_at,
)


def _make_command_message(
    command: str,
    *,
    user_id: int = TEST_USER_ID,
    chat_id: int = TEST_CHAT_ID,
    thread_id: int | None = None,
) -> tuple[Any, _AnswerCapture]:
    """Mensaje con texto de comando (e.g. "/start"). `text` empieza por `/`."""
    from aiogram.types import Chat, User

    user = User(id=user_id, is_bot=False, first_name="Test")
    chat = Chat(id=chat_id, type="private")
    capture = _AnswerCapture()
    msg = MagicMock(spec=["from_user", "chat", "text", "voice", "message_thread_id", "answer"])
    msg.from_user = user
    msg.chat = chat
    msg.text = command
    msg.voice = None
    msg.message_thread_id = thread_id
    msg.answer = capture
    return msg, capture


def _make_edge_message(
    *,
    with_user: bool = True,
    with_chat: bool = True,
    text: str = "/clear",
) -> Any:
    """Mensaje con campos None opcionales para edge cases.

    A diferencia de `aiogram.Message` (pydantic frozen), aquí permitimos
    `chat=None` y `from_user=None` para ejercitar las guardas defensivas.
    """
    from aiogram.types import Chat, User

    user = User(id=TEST_USER_ID, is_bot=False, first_name="Test") if with_user else None
    chat = Chat(id=TEST_CHAT_ID, type="private") if with_chat else None
    msg = MagicMock(spec=["from_user", "chat", "text", "voice", "message_thread_id", "answer"])
    msg.from_user = user
    msg.chat = chat
    msg.text = text
    msg.voice = None
    msg.message_thread_id = None
    msg.answer = _AnswerCapture()
    return msg


@pytest.fixture
def cmd_router(bot: Any, db: Any, settings: Any):
    """Router de comandos con todos los componentes reales."""
    return build_command_router(bot, db, settings)


# ---------------------------------------------------------------------------
# Tests de estructura del router
# ---------------------------------------------------------------------------


class TestCommandRouterStructure:
    """El router debe tener el nombre correcto y 5 handlers (uno por comando).

    Sprint 12 (ADR-007): 5 comandos en total. /chatid fue añadido al final
    del bloque de registro para preservar compatibilidad con tests que
    asumen [start, help, clear, status] en indices 0/1/2/3 via
    `get_handler_at(cmd_router, N)`. Ver
    `hermes/handlers/commands.py:build_command_router` para el orden
    actual de registro y la justificacion.
    """

    def test_router_name_is_commands(self, cmd_router: Any) -> None:
        assert cmd_router.name == "commands"

    def test_router_registers_five_handlers(self, cmd_router: Any) -> None:
        """Los 5 handlers (start, help, clear, status, chatid) están registrados."""
        handlers = get_all_handlers(cmd_router)
        assert len(handlers) == 5

    def test_handlers_are_distinct_callables(self, cmd_router: Any) -> None:
        """Cada handler es una función distinta (no se reutilizan)."""
        handlers = get_all_handlers(cmd_router)
        # Los callables son únicos (cada uno cierra sobre `db`, `bot`, `settings`)
        ids = {id(h) for h in handlers}
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Tests de /start
# ---------------------------------------------------------------------------


class TestStartCommand:
    """`/start` envía bienvenida y registra log."""

    @pytest.mark.asyncio
    async def test_start_sends_welcome(self, cmd_router: Any) -> None:
        handler = get_handler_at(cmd_router, 0)  # /start es el primero
        msg, capture = _make_command_message("/start")

        await handler(msg)

        assert capture.count() == 1
        text = capture.last()
        # Contiene el nombre "Hermes" y los comandos disponibles
        assert "Oroimen" in text
        assert "/start" in text
        assert "/help" in text
        assert "/clear" in text
        assert "/status" in text


# ---------------------------------------------------------------------------
# Tests de /help
# ---------------------------------------------------------------------------


class TestHelpCommand:
    """`/help` lista los comandos disponibles."""

    @pytest.mark.asyncio
    async def test_help_lists_commands(self, cmd_router: Any) -> None:
        handler = get_handler_at(cmd_router, 1)  # /help es el segundo
        msg, capture = _make_command_message("/help")

        await handler(msg)

        assert capture.count() == 1
        text = capture.last()
        # Lista los 4 comandos
        assert "/start" in text
        assert "/help" in text
        assert "/clear" in text
        assert "/status" in text

    @pytest.mark.asyncio
    async def test_help_message_uses_markdown(self, cmd_router: Any) -> None:
        """El mensaje de /help usa asteriscos (parse_mode=markdown)."""
        handler = get_handler_at(cmd_router, 1)
        msg, capture = _make_command_message("/help")

        await handler(msg)

        # El texto contiene un encabezado en negrita
        assert "*Comandos disponibles:*" in capture.last()


# ---------------------------------------------------------------------------
# Tests de /clear
# ---------------------------------------------------------------------------


class TestClearCommand:
    """`/clear` archiva la conversación actual y crea una nueva."""

    @pytest.mark.asyncio
    async def test_clear_responds_to_user(self, cmd_router: Any) -> None:
        handler = get_handler_at(cmd_router, 2)  # /clear
        msg, capture = _make_command_message("/clear")

        await handler(msg)

        assert capture.count() == 1
        # Mensaje de confirmación
        assert "reiniciada" in capture.last().lower() or "✓" in capture.last()

    @pytest.mark.asyncio
    async def test_clear_archives_current_conversation(self, cmd_router: Any, db: Any) -> None:
        """`/clear` marca la conversación actual como archivada (is_archived=1)."""
        handler = get_handler_at(cmd_router, 2)

        # 1. Crear conversación y añadir un mensaje
        conv_id = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        await db.add_message(conv_id, "user", "hola")

        # 2. Llamar a /clear
        msg, _ = _make_command_message("/clear")
        await handler(msg)

        # 3. Verificar que la conversación quedó archivada
        async with db.conn.execute(
            "SELECT is_archived FROM conversations WHERE id=?", (conv_id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["is_archived"] == 1

    @pytest.mark.asyncio
    async def test_clear_creates_new_conversation(self, cmd_router: Any, db: Any) -> None:
        """`/clear` crea una conversación nueva (no archivada) para el mismo user/chat."""
        handler = get_handler_at(cmd_router, 2)

        # 1. Pre-existente: crear y archivar una conversación
        old_id = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        await db.archive_conversation(old_id)

        # 2. Llamar a /clear
        msg, _ = _make_command_message("/clear")
        await handler(msg)

        # 3. Verificar que existe una nueva conversación no archivada
        async with db.conn.execute(
            "SELECT id, is_archived FROM conversations "
            "WHERE chat_id=? AND user_id=? AND is_archived=0",
            (TEST_CHAT_ID, TEST_USER_ID),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["id"] != old_id
        assert row["is_archived"] == 0

    @pytest.mark.asyncio
    async def test_clear_with_no_history_works(self, cmd_router: Any, db: Any) -> None:
        """`/clear` funciona aunque no haya conversación previa."""
        handler = get_handler_at(cmd_router, 2)

        # No hay conversación previa
        msg, capture = _make_command_message("/clear")
        await handler(msg)

        # Debe responder con el mensaje de confirmación igualmente
        assert capture.count() == 1
        # El flujo de /clear es:
        #   1) get_or_create_conversation -> crea nueva (count=1)
        #   2) archive_conversation(id)   -> la archiva (count sigue 1, is_archived=1)
        #   3) new_conversation()          -> crea OTRA nueva (count=2)
        # Por lo tanto al final hay 2 conversaciones: 1 archivada + 1 activa
        async with db.conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE chat_id=? AND user_id=?",
            (TEST_CHAT_ID, TEST_USER_ID),
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 2
        # Una está archivada, la otra no
        async with db.conn.execute(
            "SELECT is_archived FROM conversations WHERE chat_id=? AND user_id=?",
            (TEST_CHAT_ID, TEST_USER_ID),
        ) as cur:
            rows = await cur.fetchall()
        archived_flags = sorted([r["is_archived"] for r in rows])
        assert archived_flags == [0, 1]

    @pytest.mark.asyncio
    async def test_clear_skips_when_no_user(self, cmd_router: Any, db: Any) -> None:
        """`/clear` retorna silenciosamente si `from_user is None`."""
        handler = get_handler_at(cmd_router, 2)
        msg = _make_edge_message(with_user=False)

        await handler(msg)

        # No se llama a answer
        assert msg.answer.count() == 0
        # No se crea ninguna conversación
        async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
            row = await cur.fetchone()
        assert row[0] == 0

    @pytest.mark.asyncio
    async def test_clear_skips_when_no_chat(self, cmd_router: Any, db: Any) -> None:
        """`/clear` retorna silenciosamente si `chat is None`."""
        handler = get_handler_at(cmd_router, 2)
        msg = _make_edge_message(with_chat=False)

        await handler(msg)

        # No se llama a answer
        assert msg.answer.count() == 0
        # No se crea ninguna conversación
        async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
            row = await cur.fetchone()
        assert row[0] == 0

    @pytest.mark.asyncio
    async def test_clear_preserves_thread_id(self, cmd_router: Any, db: Any) -> None:
        """`/clear` mantiene el `thread_id` en la conversación nueva."""
        handler = get_handler_at(cmd_router, 2)
        msg, _ = _make_command_message("/clear", thread_id=42)

        await handler(msg)

        # La nueva conversación debe tener thread_id=42
        async with db.conn.execute(
            "SELECT thread_id FROM conversations WHERE chat_id=?",
            (TEST_CHAT_ID,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["thread_id"] == 42


# ---------------------------------------------------------------------------
# Tests de /status
# ---------------------------------------------------------------------------


class TestStatusCommand:
    """`/status` muestra el estado del sistema."""

    @pytest.mark.asyncio
    async def test_status_responds_ok(self, cmd_router: Any) -> None:
        handler = get_handler_at(cmd_router, 3)  # /status
        msg, capture = _make_command_message("/status")

        await handler(msg)

        assert capture.count() == 1
        # Mensaje de status
        assert "Oroimen" in capture.last() or "operativo" in capture.last().lower()

    @pytest.mark.asyncio
    async def test_status_does_not_touch_db(self, cmd_router: Any, db: Any) -> None:
        """`/status` no debe modificar la DB (es solo lectura)."""
        handler = get_handler_at(cmd_router, 3)
        msg, _ = _make_command_message("/status")

        # Pre-condición: no hay conversaciones
        async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
            before = (await cur.fetchone())[0]
        assert before == 0

        await handler(msg)

        # Post-condición: sigue sin haber conversaciones
        async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
            after = (await cur.fetchone())[0]
        assert after == 0


# ---------------------------------------------------------------------------
# Tests de integración: comando no conocido
# ---------------------------------------------------------------------------


class TestUnknownCommand:
    """Comportamiento del router ante mensajes que no son comandos."""

    def test_each_handler_has_a_command_filter(self, cmd_router: Any) -> None:
        """Cada uno de los 4 handlers tiene un filtro Command("xxx") asociado.

        Esto garantiza que el handler SOLO se dispara para su comando
        específico, no para cualquier mensaje. Los filtros viven en
        `handler_obj.filters` (no en `.check`).
        """
        expected_cmds = ["start", "help", "clear", "status"]
        for i, expected_cmd in enumerate(expected_cmds):
            handler_obj = cmd_router.message.handlers[i]
            # Verificamos que tiene al menos un filter
            assert len(handler_obj.filters) >= 1
            # El primer filter debe ser una instancia de Command
            filter_obj = handler_obj.filters[0]
            command_filter = filter_obj.callback
            assert hasattr(command_filter, "commands")
            assert expected_cmd in command_filter.commands

    def test_filters_match_expected_commands(self, cmd_router: Any) -> None:
        """Los filtros Command están en el orden correcto (start, help, clear, status)."""
        expected_cmds = ["start", "help", "clear", "status"]
        for i, expected_cmd in enumerate(expected_cmds):
            handler_obj = cmd_router.message.handlers[i]
            filter_obj = handler_obj.filters[0]
            command_filter = filter_obj.callback
            assert command_filter.commands == (expected_cmd,)


# ---------------------------------------------------------------------------
# Tests de integración con flujo de mensajes
# ---------------------------------------------------------------------------


class TestCommandFlow:
    """Tests end-to-end del flujo de comandos en secuencia."""

    @pytest.mark.asyncio
    async def test_clear_then_add_message_creates_new_history(
        self, cmd_router: Any, db: Any
    ) -> None:
        """Después de /clear, los mensajes van a una conversación nueva."""
        clear_handler = get_handler_at(cmd_router, 2)
        msg, _ = _make_command_message("/clear")

        # 1. /clear: archiva + crea nueva
        await clear_handler(msg)

        # 2. Añadir un mensaje a la conversación actual
        new_conv = await db.get_or_create_conversation(chat_id=TEST_CHAT_ID, user_id=TEST_USER_ID)
        await db.add_message(new_conv, "user", "mensaje post-clear")

        # 3. Verificar que la conversación post-clear tiene el mensaje
        history = await db.get_history(new_conv)
        assert len(history) == 1
        assert history[0]["content"] == "mensaje post-clear"

    @pytest.mark.asyncio
    async def test_help_does_not_create_conversation(self, cmd_router: Any, db: Any) -> None:
        """`/help` no debe crear conversaciones (es solo informativo)."""
        help_handler = get_handler_at(cmd_router, 1)
        msg, _ = _make_command_message("/help")

        await help_handler(msg)

        # No debe haber conversaciones creadas
        async with db.conn.execute("SELECT COUNT(*) FROM conversations") as cur:
            row = await cur.fetchone()
        assert row[0] == 0
