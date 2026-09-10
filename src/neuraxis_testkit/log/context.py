#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit/log/context.py -- Logging Context Manager

Provides a context manager for temporarily modifying log levels.
"""
from __future__ import annotations

import logging
from neuraxis_testkit.log.config import VALID_LEVELS, LEVEL_MAP

class LogLevelContext:
    """
    Context manager to temporarily modify log levels.

    Example:
        >>> with LogLevelContext(logger, 'DEBUG'):
        ...     logger.debug("This line will be displayed")

        >>> logger.debug("After reverting to the original level, this line will not be displayed")
    """

    def __init__(self, logger: logging.Logger, new_level: str):
        level_upper = new_level.upper()
        if level_upper not in VALID_LEVELS:
            raise ValueError(
                f"Invalid log level: '{new_level}'. "
                f"Valid options: {', '.join(sorted(VALID_LEVELS))}"
            )
        self.logger = logger
        self.new_level_int = LEVEL_MAP[level_upper]
        self._old_level: int | None = None

    def __enter__(self) -> logging.Logger:
        # Store the explicitly set level (not getEffectiveLevel()):
        # if not set explicitly, it is NOTSET(0); on exit, setLevel(NOTSET) restores parent inheritance.
        self._old_level = self.logger.level
        self.logger.setLevel(self.new_level_int)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Logic Check: Using 'is not None' is safer than truthy check 
        # in case the level is 0 (NOTSET).
        if self._old_level is not None:
            self.logger.setLevel(self._old_level)   # NOTSET -> restores parent inheritance
        return False


__all__ = ['LogLevelContext']
