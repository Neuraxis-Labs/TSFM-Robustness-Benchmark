"""
neuraxis_testkit/log - Neuraxis TestKit Logging Module

A thin wrapper on top of standard logging + logging.yaml (dictConfig),
hiding configuration loading details and exposing only a unified logging interface.

Usage Examples:
    from neuraxis_testkit.log import get_logger

    logger = get_logger(__name__)
    logger.info("Hello World")
    logger.trace("TRACE Level Log")   # Need logger.setLevel(LEVEL_MAP['TRACE'])
"""
from .config import setup_logging, get_log_file_path
from .context import LogLevelContext
from .core import get_logger, get_default_logger
from .decorators import log_execution, log_time


__all__ = [
    # Core
    # Functions
    'get_logger',
    'get_default_logger',
    'setup_logging',
    'get_log_file_path',
    # Decorators
    'log_execution',
    'log_time',
    # Context
    'LogLevelContext',
]