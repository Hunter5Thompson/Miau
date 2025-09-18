"""Logging utilities for the MCP server."""

from __future__ import annotations

import logging
import os
from typing import Iterable


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(level: str = "INFO", *, extra_loggers: Iterable[str] | None = None) -> None:
    """Configure structured-ish logging for the service."""

    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
    for logger_name in extra_loggers or ():
        logging.getLogger(logger_name).setLevel(level)


def configure_from_env() -> None:
    """Read log level from environment variable if present."""

    level = os.getenv("C4MCP_LOG_LEVEL")
    if level:
        configure_logging(level)


__all__ = ["configure_logging", "configure_from_env"]
