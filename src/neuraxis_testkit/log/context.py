#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.log.context - Logging Context Manager

Provides a context manager for temporarily modifying log levels.
"""

from neuraxis_testkit.log.core import Logger
from neuraxis_testkit.log.config import VALID_LEVELS


class LogLevelContext:
    """
    Context manager to temporarily modify log levels.

    Example:
        >>> with LogLevelContext(logger, 'DEBUG'):
        ...     logger.debug("This line will be displayed")

        >>> logger.debug("After reverting to the original level, this line will not be displayed")
    """

    def __init__(self, logger: Logger, new_level: str):
        level_upper = new_level.upper()
        if level_upper not in VALID_LEVELS:
            raise ValueError(f"Invalid log level: '{new_level}'. Valid options: {', '.join(sorted(VALID_LEVELS))}")
        self.logger = logger
        self.new_level = level_upper
        self.old_level = None

    def __enter__(self) -> Logger:
        self.old_level = self.logger.get_level()
        self.logger.set_level(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Logic Check: Using 'is not None' is safer than truthy check 
        # in case the level is 0 (NOTSET).
        if self.old_level is not None:
            self.logger.set_level(self.old_level)
        return False


__all__ = ['LogLevelContext']
