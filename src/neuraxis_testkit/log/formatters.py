#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.log.formatters - Logging Formatters

ColoredFormatter is a custom factory for dictConfig.
See logging.yaml: formatters.colored."(): neuraxis_testkit.log.formatters.ColoredFormatter".
dictConfig calls it with kwargs: ColoredFormatter(fmt=..., datefmt=..., use_color=...),
so __init__ signature must be compatible with parameters declared in YAML.
"""

import os, sys, logging

from neuraxis_testkit.log.config import LOG_FORMAT, LOG_DATE_FORMAT

class ColoredFormatter(logging.Formatter):
    """
    ANSI colored formatter for terminal output (console only).
    """
    COLORS = {
        'TRACE':    '\033[90m',   # bright gray
        'DEBUG':    '\033[36m',   # cyan
        'INFO':     '\033[32m',   # green
        'WARNING':  '\033[33m',   # yellow
        'ERROR':    '\033[31m',   # red
        'CRITICAL': '\033[35m',   # magenta
    }
    RESET = '\033[0m'

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, use_color: bool | str = True):
        # use_color passed from dictConfig may be a string "true"/"false".
        super().__init__(fmt=fmt or LOG_FORMAT, datefmt=datefmt or LOG_DATE_FORMAT)
        self.use_color = use_color if isinstance(use_color, bool) else str(use_color).lower() in ("true", "1", "yes")
        if self.use_color and sys.platform == "win32":
            os.system("")   # Enable ANSI VT processing for legacy conhost (Win10+ Terminal no need, idempotent harmless)

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record."""
        message = super().format(record)
        if not self.use_color:
            return message
        color = self.COLORS.get(record.levelname)
        return message if color is None else f"{color}{message}{self.RESET}"


__all__ = ["ColoredFormatter"]
