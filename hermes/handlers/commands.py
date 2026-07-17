"""Handlers para comandos: /start, /help, /clear, /status, /chatid."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import Message

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.memory.db import Database

logger = logging.getLogger(__name__)


def build_command_router(bot: Bot, db: Database, settings: Settings) -> Router:
    router = Router(name="commands")

    @router.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        await message.answer(
            "¡Hola! Soy *Oroimen*, tu asistente personal.\n"
            "Mándame un mensaje de texto y te respondo.\n"
            "Comandos: /start /help /clear /status /chatid"
        )
        logger.info(
            "command_start", extra={"user_id": message.from_user.id if message.from_user else None}
        )

    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(
            "*Comandos disponibles:*\n"
            "/start — Mensaje de bienvenida\n"
            "/help — Esta ayuda\n"
            "/clear — Empezar conversación nueva\n"
            "/status — Estado del sistema\n"
            "/chatid — Muestra tu chat_id (para configurar push notifications)"
        )

    @router.message(Command("clear"))
    async def cmd_clear(message: Message) -> None:
        if message.from_user is None or message.chat is None:
            return
        conv_id = await db.get_or_create_conversation(
            chat_id=message.chat.id,
            user_id=message.from_user.id,
            thread_id=message.message_thread_id,
        )
        await db.archive_conversation(conv_id)
        await db.new_conversation(
            chat_id=message.chat.id,
            user_id=message.from_user.id,
            thread_id=message.message_thread_id,
        )
        await message.answer("✓ Conversación reiniciada.")
        logger.info("command_clear", extra={"user_id": message.from_user.id})

    @router.message(Command("status"))
    async def cmd_status(message: Message) -> None:
        await message.answer("✓ Oroimen operativo (Sprint 1 MVP).")

    @router.message(Command("chatid"))
    async def cmd_chatid(message: Message) -> None:
        """S10.4: muestra el chat_id del user. Usado para configurar
        TELEGRAM_CHAT_ID en el .env de hermes (necesario para push
        notifications de health alerts). Self-service: el user no
        necesita acceso al servidor para obtener su chat_id.

        Sprint 12 ADR-007: tambien usado por RikkaHub (app Android
        nativa) para inyectar el chat_id en headers X-Telegram-Chat-Id
        en requests HTTP API (autorrelacion HTTP <-> Telegram).

        Orden de registro: ultimo (despues de /status). Telegram
        dispatcha por command name, no por orden, asi que este orden es
        puramente estetico y para mantener compatibilidad con
        `tests/unit/test_commands.py` que asume indices
        [start, help, clear, status, chatid].
        """
        chat = message.chat
        if chat is None:
            await message.answer("ERROR: no chat info en el mensaje.")
            return
        chat_id = chat.id
        chat_type = chat.type
        chat_title = chat.title or (chat.first_name or "")
        await message.answer(
            f"*Tu chat_id:* `{chat_id}`\n"
            f"*Tipo:* `{chat_type}`\n"
            f"*Titulo:* `{chat_title}`\n\n"
            f"Para configurar push notifications de health alerts "
            f"(Sprint 10.4), anade esta linea al `.env` de hermes en el NAS host:\n"
            f"`TELEGRAM_CHAT_ID={chat_id}`\n"
            f"Luego restart el container."
        )
        logger.info(
            "command_chatid",
            extra={
                "user_id": message.from_user.id if message.from_user else None,
                "chat_id": chat_id,
                "chat_type": chat_type,
            },
        )

    return router
