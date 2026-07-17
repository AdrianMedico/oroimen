"""Logging estructurado en JSON."""

import logging
import sys

from pythonjsonlogger import jsonlogger

JsonFormatter = jsonlogger.JsonFormatter


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    fmt = JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("httpx", "httpcore", "aiogram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
