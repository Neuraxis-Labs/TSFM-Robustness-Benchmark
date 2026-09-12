"""
neuraxis_testkit.log - Neuraxis TestKit Logging Module

Provides a concise logging interface and hides internal implementation details.

Usage Examples:
    from neuraxis_testkit.log import get_logger, setup_logging

    logger = get_logger(__name__)
    logger.info("Hello World")
"""
import threading
from .core import Logger
from .decorators import log_execution, log_time
from .context import LogLevelContext

# Convenience Functions
_default_logger_lock = threading.Lock()
_default_logger: Logger | None = None


def get_logger(name: str = "neuraxis", **kwargs) -> Logger:
    """
    Get a logger instance. Recommended to use module name for 'name'.

    Args:
        name: Name of the logger. It is recommended to use the module name.
        **kwargs: Optional configuration parameters to override defaults.

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger('testcases.futureCovs.dirtyData.test_dirty')
    """
    return Logger.get_logger(name, **kwargs)

def get_default_logger() -> Logger:
    """Get a default global logger for quick scripts."""
    global _default_logger
    with _default_logger_lock:
        if _default_logger is None:
            _default_logger = get_logger('default')
    return _default_logger

def flush_all_logs():
    """Forces all log handlers to flush immediately."""
    Logger.flush()

__all__ = [
    # Core
    'Logger',
    # Functions
    'get_logger',
    'get_default_logger',
    'flush_all_logs',
    # Decorators
    'log_execution',
    'log_time',
    # Context
    'LogLevelContext',
]