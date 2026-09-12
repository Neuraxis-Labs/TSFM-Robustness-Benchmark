#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.log.formatters - Logging Formatters

Contains formatters for colored console output, etc.
"""

import logging

from neuraxis_testkit.log.config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_USE_COLOR


class ColoredFormatter(logging.Formatter):
    """Log formatter with colored output."""

    COLORS = {
        'TRACE': '\033[36m',      # Cyan
        'DEBUG': '\033[34m',      # Blue
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Purple/Magenta
    }
    RESET = '\033[0m'

    def __init__(self, fmt: str = None, datefmt: str = None, use_color: bool = True):
        super().__init__(fmt or LOG_FORMAT, datefmt or LOG_DATE_FORMAT)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record."""
        if self.use_color and record.levelname in self.COLORS:
            # Save original levelname
            original_levelname = record.levelname
            # Apply color
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            formatted = super().format(record)
            # Restore original levelname
            record.levelname = original_levelname
            return formatted
        return super().format(record)


def create_console_formatter(use_color: bool = None) -> logging.Formatter:
    """
    Creates a console log formatter.

    Args:
        use_color: Whether to use color. If None, uses the global configuration.

    Returns:
        An instance of logging.Formatter.
    """
    if use_color is None:
        use_color = LOG_USE_COLOR

    if use_color:
        return ColoredFormatter(LOG_FORMAT, LOG_DATE_FORMAT)
    else:
        return logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)


def create_file_formatter() -> logging.Formatter:
    """Creates a file log formatter."""
    return logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)


__all__ = [
    'ColoredFormatter',
    'create_console_formatter',
    'create_file_formatter',
]
