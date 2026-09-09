"""Development logging setup.

Writes to the console and to a rotating file under ./logs so latency and
errors can be diagnosed after the fact. No secrets are ever logged here.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from assistant.config import LOG_DIR


def get_logger(name: str = "assistant") -> logging.Logger:
    """Return a configured logger (idempotent across calls)."""
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console (INFO+).
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Rotating file (DEBUG+), for latency/error diagnosis.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_DIR / "assistant.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger._configured = True  # type: ignore[attr-defined]
    return logger
