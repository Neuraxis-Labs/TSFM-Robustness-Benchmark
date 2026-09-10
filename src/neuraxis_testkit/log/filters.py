#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit/log/filters.py -- Logging Filters

Contains filters for module-level log level overrides and ignoring specific loggers.
"""

from __future__ import annotations

import logging
from neuraxis_testkit.log.config import MODULE_LEVEL_OVERRIDES, IGNORED_LOGGERS, LEVEL_MAP


class ModuleLevelFilter(logging.Filter):
    """
    Dynamically adjusts log levels based on module names.
    Supports the MODULE_LEVEL_OVERRIDES configuration.

    Example:
        MODULE_LEVEL_OVERRIDES = {"thirdparty": "WARNING"}
        # INFO/DEBUG for thirdparty.* is silenced at the handler level.
    """

    def __init__(self, overrides: dict[str, str] | None = None):
        super().__init__()
        self.overrides = overrides if overrides is not None else MODULE_LEVEL_OVERRIDES
        self._level_cache: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        # Check if there is a level override for this specific module
        module_name = record.name
        if module_name in self._level_cache:
            min_level = self._level_cache[module_name]
        else:
            min_level = None
            for mod, level_str in self.overrides.items():
                if module_name.startswith(mod):
                    min_level = LEVEL_MAP.get(level_str.upper(), logging.INFO)
                    self._level_cache[module_name] = min_level
                    break
        if min_level is not None:
            return record.levelno >= min_level
        return True


class IgnoredLoggerFilter(logging.Filter):
    """Silence loggers with specified prefixes (configuration source: IGNORED_LOGGERS)."""

    def __init__(self, ignored_loggers: list[str] | None = None):
        super().__init__()
        self.ignored_loggers = set(ignored_loggers if ignored_loggers is not None else IGNORED_LOGGERS)

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(record.name.startswith(i) for i in self.ignored_loggers)


__all__ = [
    'ModuleLevelFilter',
    'IgnoredLoggerFilter',
]
