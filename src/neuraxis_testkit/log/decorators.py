#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.log.decorators - Logging Decorators

Provides decorators for function execution logging and execution time tracking.
"""

import time
from functools import wraps
from typing import Callable
from neuraxis_testkit.log.config import VALID_LEVELS
from neuraxis_testkit.log.core import Logger

def log_execution(
    logger: Logger | None = None,
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
    from neuraxis_testkit.log import get_default_logger
    
    level_upper = level.upper()
    if level_upper not in VALID_LEVELS:
        raise ValueError(f"Invalid log level: '{level}'. Valid options: {', '.join(sorted(VALID_LEVELS))}")
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            lg = logger or get_default_logger()
            log_method = getattr(lg, level_upper.lower(), lg.info)

            msg = f"Starting execution: {func.__name__}"
            if log_args:
                args_str = ', '.join(repr(a) for a in args)
                kwargs_str = ', '.join(f"{k}={v!r}" for k, v in kwargs.items())
                params = ', '.join(filter(None, [args_str, kwargs_str]))
                if params:
                    msg += f"({params})"
            log_method(msg)

            try:
                result = func(*args, **kwargs)
                msg = f"Execution completed: {func.__name__}"
                if log_result:
                    msg += f" -> {result!r}"
                log_method(msg)
                return result

            except Exception as exp:
                if log_exception:
                    lg.exception(f"Execution exception: {func.__name__} - {exp}")
                raise

        return wrapper
    return decorator


def log_time(logger: Logger | None = None, level: str = 'INFO'):
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
    from neuraxis_testkit.log import get_default_logger

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
