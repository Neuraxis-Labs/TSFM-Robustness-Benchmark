#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.log.filters - Logging Filters

Contains filters for module-level log level overrides and ignoring specific loggers.
"""

import logging
from neuraxis_testkit.log.config import MODULE_LEVEL_OVERRIDES, IGNORED_LOGGERS, LEVEL_MAP


class ModuleLevelFilter(logging.Filter):
    """
    Dynamically adjusts log levels based on module names.
    Supports the MODULE_LEVEL_OVERRIDES configuration.
    """
    def __init__(self, overrides: dict[str, str] = None):
        self.overrides = overrides or MODULE_LEVEL_OVERRIDES
        self._level_cache = {}

    def filter(self, record: logging.LogRecord) -> bool:
        # Check if there is a level override for this specific module
        module_name = record.name
        if module_name in self._level_cache:
            min_level = self._level_cache[module_name]
        else:
            for mod, level_str in self.overrides.items():
                if module_name.startswith(mod):
                    min_level = LEVEL_MAP.get(level_str.upper(), logging.INFO)
                    self._level_cache[module_name] = min_level
                    break
            else:
                min_level = None

        if min_level is not None:
            return record.levelno >= min_level
        return True


class IgnoredLoggerFilter(logging.Filter):
    """Filters out ignored loggers based on logger name prefixes."""
    def __init__(self, ignored_loggers: list[str] = None):
        self.ignored_loggers = set(ignored_loggers or IGNORED_LOGGERS)

    def filter(self, record: logging.LogRecord) -> bool:
        for ignored in self.ignored_loggers:
            if record.name.startswith(ignored):
                return False
        return True


__all__ = [
    'ModuleLevelFilter',
    'IgnoredLoggerFilter',
]
