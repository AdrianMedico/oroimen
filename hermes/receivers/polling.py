"""Long polling de Telegram usando aiogram (liviano)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from hermes.handlers.commands import build_command_router
from hermes.handlers.messages import build_message_router
from hermes.receivers.base import UpdateReceiver

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.llm.ocr import OcrProvider
    from hermes.memory.db import Database
    from hermes.memory.edge_coordinator import EdgeCoordinator
    from hermes.memory.ocr_pending_repo import OcrPendingRepo
    from hermes.services.embeddings import EmbeddingsService
    from hermes.telemetry import Telemetry
    from hermes.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class PollingReceiver(UpdateReceiver):
    def __init__(
        self,
        bot_token: str,
        allowed_user_ids: list[int],
        db: Database,
        settings: Settings,
        telemetry: Telemetry,
        tool_registry: ToolRegistry | None = None,
        embeddings_service: EmbeddingsService | None = None,
        ocr_repo: OcrPendingRepo | None = None,
        edge_coordinator: EdgeCoordinator | None = None,
        ocr_provider: OcrProvider | None = None,
    ) -> None:
        self.bot_token = bot_token
        self.allowed_user_ids = set(allowed_user_ids)
        self.db = db
        self.settings = settings
        self.telemetry = telemetry
        self.tool_registry = tool_registry
        # Sprint 16 (US-3.2): pipea el EmbeddingsService para que
        # build_message_router lo pase a AgentLoop. Si es None, la
        # feature de memory facts injection se desactiva silenciosamente.
        self.embeddings_service = embeddings_service
        # Sprint 19 Slice 4d: OCR user commands. Optional -- if None,
        # the OCR router is not registered (commands return usage help).
        self.ocr_repo = ocr_repo
        self.edge_coordinator = edge_coordinator
        # Sprint 19 Slice 4d (B1 fix): provider-agnostic OcrProvider.
        # Required for /externalOCR 2-step flow. None means no hosted
        # OCR available (the router is still registered, but /externalOCR
        # returns a config error).
        self.ocr_provider = ocr_provider
        self.bot = Bot(
            token=bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        )
        self.dp = Dispatcher()
        self._register_handlers()

    def _register_handlers(self) -> None:
        router = build_command_router(self.bot, self.db, self.settings)
        self.dp.include_router(router)
        msg_router = build_message_router(
            self.bot,
            self.db,
            self.settings,
            self.telemetry,
            tool_registry=self.tool_registry,
            embeddings_service=self.embeddings_service,
        )
        self.dp.include_router(msg_router)
        # Sprint 19 Slice 4d: OCR user commands (6 commands from TDD §4.3.1).
        if self.ocr_repo is not None and self.ocr_provider is not None:
            from hermes.handlers.ocr_commands import build_ocr_command_router

            ocr_router = build_ocr_command_router(
                self.bot,
                self.db,
                self.ocr_repo,
                self.edge_coordinator,
                self.settings,
                self.ocr_provider,
            )
            self.dp.include_router(ocr_router)

    async def _authorize(self, event_from_user_id: int) -> bool:
        if not self.allowed_user_ids:
            return True
        return event_from_user_id in self.allowed_user_ids

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        logger.info(
            "polling_started",
            extra={
                "allowed": sorted(self.allowed_user_ids),
                "tools_enabled": self.settings.tools_enabled,
                "tools_count": len(self.tool_registry.list_tools()) if self.tool_registry else 0,
            },
        )
        try:
            await self.dp.start_polling(self.bot, allowed_updates=["message"])
        finally:
            await self.bot.session.close()
            logger.info("polling_stopped")
