#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit/log/decorators.py -- Logging Decorators

Provides decorators for function execution logging and execution time tracking.
"""
from __future__ import annotations

import time, logging
from functools import wraps
from typing import Callable
from neuraxis_testkit.log.config import VALID_LEVELS
from neuraxis_testkit.log.core import get_default_logger

def log_execution(
    logger: logging.Logger | None = None,
    level: str = 'INFO',
    log_args: bool = False,
    log_result: bool = False,
    log_exception: bool = True,
):
    """
    Decorator for logging function execution.

    Args:
        logger: Logger instance. If None, uses the default logger.
        level: Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_args: Whether to log function arguments.
        log_result: Whether to log the return value.
        log_exception: Whether to log exception details.

    Example:
        >>> @log_execution(logger, level='DEBUG', log_args=True)
        ... def my_function(x, y):
        ...     return x + y
    """

    level_upper = level.upper()
    if level_upper not in VALID_LEVELS:
        raise ValueError(f"Invalid log level: '{level}'. Valid options: {', '.join(sorted(VALID_LEVELS))}")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            lg = logger or get_default_logger()
            # TRACE already registered by config.py on standard Logger, getattr hits directly.
            log_method = getattr(lg, level_upper.lower(), lg.info)
            extra = f" args={args!r}, kwargs={kwargs!r}" if log_args else ""
            log_method(f"Starting execution: {func.__name__}{extra}")
            try:
                result = func(*args, **kwargs)
                tail = f" result={result!r}" if log_result else ""
                log_method(f"Execution completed: {func.__name__}{tail}")
                return result
            except Exception as exp:
                if log_exception:
                    lg.exception(f"Execution exception: {func.__name__} - {exp}")
                raise

        return wrapper
    return decorator


def log_time(logger: logging.Logger | None = None, level: str = 'INFO'):
    """
    Decorator for tracking and logging function execution time.

    Args:
        logger: Logger instance.
        level: Log level.

    Example:
        >>> @log_time()
        ... def slow_function():
        ...     time.sleep(1)
    """
    level_upper = level.upper()
    if level_upper not in VALID_LEVELS:
        raise ValueError(f"Invalid log level: '{level}'. Valid options: {', '.join(sorted(VALID_LEVELS))}")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            lg = logger or get_default_logger()
            log_method = getattr(lg, level_upper.lower(), lg.info)

            start_time = time.time()
            log_method(f"Starting execution: {func.__name__}")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log_method(f"Execution completed: {func.__name__} (elapsed: {elapsed:.3f}s)")
                return result
            except Exception as exp:
                elapsed = time.time() - start_time
                lg.exception(f"Execution exception: {func.__name__} (elapsed: {elapsed:.3f}s) - {exp}")
                raise

        return wrapper
    return decorator


__all__ = [
    'log_execution',
    'log_time',
]
