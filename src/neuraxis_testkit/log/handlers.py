#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.log.handlers - Log Handler Management

Responsible for creating and configuring various log handlers.
"""

import sys, logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from neuraxis_testkit.log.config import (
    LOGS_DIR,
    LOG_FILE_NAME,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOG_ENCODING,
    LOG_ROTATION,
    LOG_WHEN,
    LOG_INTERVAL,
)
from neuraxis_testkit.log.formatters import create_console_formatter, create_file_formatter


def create_console_handler(level: int, use_color: bool = None) -> logging.Handler:
    """
    Creates a console log handler.

    Args:
        level: Log level
        use_color: Whether to use color

    Returns:
        An instance of logging.Handler
    """
    stream = sys.stdout

    try:
        cur_encoding = (getattr(stream, "encoding", "") or "").lower()
        if cur_encoding not in ("utf-8", "utf8"):
            stream.reconfigure(
                encoding=LOG_ENCODING,
                errors="replace",
                line_buffering=True,
            )
    except (AttributeError, OSError, ValueError):
        pass

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(create_console_formatter(use_color))
    return handler


def create_file_handler(level: int) -> logging.Handler:
    """
    Creates a file log handler.

    Args:
        level: Log level.

    Returns:
        An instance of logging.Handler.
    """
    log_file_path = LOGS_DIR / LOG_FILE_NAME

    if LOG_ROTATION == "time":
        handler = TimedRotatingFileHandler(
            filename=str(log_file_path),
            when=LOG_WHEN,
            interval=LOG_INTERVAL,
            backupCount=LOG_BACKUP_COUNT,
            encoding=LOG_ENCODING,
        )
    else:
        # Default: Rotate by size
        handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding=LOG_ENCODING,
        )

    handler.setLevel(level)
    handler.setFormatter(create_file_formatter())
    return handler


def create_handlers(level: int, console_output: bool, file_output: bool, use_color: bool = None) -> list[logging.Handler]:
    """
    Creates log handlers in batch.

    Args:
        level: Log level.
        console_output: Whether to create a console handler.
        file_output: Whether to create a file handler.
        use_color: Whether to use color for console output.

    Returns:
        A list of logging.Handler instances.
    """
    handlers = []

    if console_output:
        handlers.append(create_console_handler(level, use_color))

    if file_output:
        handlers.append(create_file_handler(level))

    return handlers


__all__ = [
    'create_console_handler',
    'create_file_handler',
    'create_handlers',
]
