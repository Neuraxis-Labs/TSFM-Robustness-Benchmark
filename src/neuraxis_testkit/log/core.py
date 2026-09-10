#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit/log/core.py -- Core Logger (thin wrapper over standard library)
"""

from __future__ import annotations

import logging
from .config import setup_logging

_DEFAULT_LOGGER = "neuraxis_testkit"

def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger; automatically loads logging.yaml on first call.

    Usage:
        logger = get_logger(__name__)

    Returns a standard logging.Logger (with dynamically registered trace method).
    """
    setup_logging()
    return logging.getLogger(name or _DEFAULT_LOGGER)

def get_default_logger() -> logging.Logger:
    """
    Get a default global logger for quick scripts.

    Note: logging.getLogger(name) is cached by name and singleton per process, no locking required.
    """
    return get_logger('default')


__all__ = ["get_logger", "get_default_logger"]
