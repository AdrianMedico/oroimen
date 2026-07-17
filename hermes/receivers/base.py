"""Interfaz abstracta para recibir updates de Telegram."""

from __future__ import annotations

import abc
import asyncio


class UpdateReceiver(abc.ABC):
    """Contrato para cualquier mecanismo de recepción de updates."""

    @abc.abstractmethod
    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """Bloquea hasta que stop_event se active."""
        raise NotImplementedError
