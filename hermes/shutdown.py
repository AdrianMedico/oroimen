"""Graceful shutdown: traduce SIGTERM/SIGINT a un asyncio.Event."""

from __future__ import annotations

import asyncio
import signal


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop_event.set())
