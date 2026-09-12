#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.log.config - Logging Configuration Definitions

Supports overriding configurations via environment variables.
"""

import os, sys, logging
from pathlib import Path
from datetime import datetime
from config.settings import LOGS_DIR

def _running_under_pytest() -> bool:
    return "pytest" in sys.modules

def _xdist_worker() -> bool:
    return os.environ.get("PYTEST_XDIST_WORKER") is not None

_raw_console = os.getenv("LOG_CONSOLE_OUTPUT", "").lower()
if _raw_console in ("true", "false", "1", "0", "yes", "no"):
    LOG_CONSOLE_OUTPUT: bool = _raw_console in ("true", "1", "yes")
else:
    LOG_CONSOLE_OUTPUT: bool = not (_running_under_pytest() or _xdist_worker())

# ============================================================
# Basic Logging Configuration (Supports Environment Variable Overrides)
# ============================================================

# Log Level: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")

# Log file base name
LOG_FILE_BASENAME: str = os.getenv("LOG_FILE_BASENAME", "neuraxis_testkit")  # standalone fallback only

# Whether to include date in the filename
LOG_FILE_WITH_DATE: bool = os.getenv("LOG_FILE_WITH_DATE", "true").lower() == "true"

# Whether to output to file
LOG_FILE_OUTPUT: bool = os.getenv("LOG_FILE_OUTPUT", "true").lower() == "true"

# Log file size limit (bytes), default 50MB
LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", str(50 * 1024 * 1024)))

# Number of backup log files
LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "10"))

# Log file encoding
LOG_ENCODING: str = os.getenv("LOG_ENCODING", "utf-8")

# Log format (includes Process ID, for multi-concurrency scenarios)
LOG_FORMAT: str = os.getenv(
    "LOG_FORMAT",
    '[%(asctime)s.%(msecs)03d][%(levelname)8s][%(process)d][%(threadName)s][%(name)s][%(filename)s:%(lineno)d] - %(message)s'
)

# Simplified log format
LOG_SIMPLE_FORMAT: str = os.getenv(
    "LOG_SIMPLE_FORMAT",
    '[%(asctime)s.%(msecs)03d][%(levelname)s] - %(message)s'
)

# Date format
LOG_DATE_FORMAT: str = os.getenv(
    "LOG_DATE_FORMAT",
    '%Y-%m-%d %H:%M:%S'
)

# Whether to enable colored output (console only)
LOG_USE_COLOR: bool = os.getenv("LOG_USE_COLOR", "true").lower() == "true"

# Log rotation type: 'size' for size-based, 'time' for time-based
LOG_ROTATION: str = os.getenv("LOG_ROTATION", "size")

# Time rotation specifier (Effective when LOG_ROTATION == 'time')
LOG_WHEN: str = os.getenv("LOG_WHEN", "midnight")

# Time rotation interval count
LOG_INTERVAL: int = int(os.getenv("LOG_INTERVAL", "1"))

# Log queue size (0 disables async mode, uses sync mode)
LOG_QUEUE_SIZE: int = int(os.getenv("LOG_QUEUE_SIZE", "1000"))

# Module-level log overrides (for debugging specific modules)
# Format: {"module.name": "DEBUG"}
MODULE_LEVEL_OVERRIDES: dict[str, str] = {}

# Names of loggers to ignore (do not write to log)
IGNORED_LOGGERS: list[str] = [
    "urllib3",
    "requests",
    "botocore",
    "boto3",
]


# ============================================================
# Dynamic Log Filename Generation
# ============================================================

def _generate_log_file_name() -> str:
    """
    Generates the log filename with optional date suffix.

    Returns:
        Log filename, e.g., neuraxis_testkit_20260820.log
    """
    if LOG_FILE_WITH_DATE:
        return f"{LOG_FILE_BASENAME}_{datetime.now().strftime('%Y%m%d')}.log"
    return f"{LOG_FILE_BASENAME}.log"

# Dynamically generated log filename
LOG_FILE_NAME: str = _generate_log_file_name()   # standalone fallback only


# ============================================================
# Log Level Mapping
# ============================================================

if not hasattr(logging, 'TRACE'):
    logging.TRACE = logging.DEBUG - 5
    logging.addLevelName(logging.TRACE, "TRACE")

if not hasattr(logging, 'trace'):
    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.TRACE):
           self._log(logging.TRACE, msg, args, **kwargs)
    setattr(logging.Logger, 'trace', trace)

LEVEL_MAP = {
    'TRACE': logging.TRACE,
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}
VALID_LEVELS = set(LEVEL_MAP.keys())

# ============================================================
# Convenience Functions
# ============================================================

def force_console_encoding(encoding: str = LOG_ENCODING) -> None:
    """
    Force reconfigure stdout/stderr encoding.
    """
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding=encoding, errors="backslashreplace")
        except Exception:    # noqa: BLE001 - best-effort
            try:
                print(f"[WARN] force_console_encoding: reconfigure {name} failed",
                      file=sys.__stderr__, flush=True)
            except Exception:
                pass

def get_log_level() -> int:
    """Gets the integer representation of the log level."""
    return LEVEL_MAP.get(LOG_LEVEL.upper(), logging.INFO)

def is_async_enabled() -> bool:
    """Checks if asynchronous logging is enabled."""
    return LOG_QUEUE_SIZE > 0

def get_log_file_path() -> Path:
    """Gets the full path to the log file (standalone fallback only)."""
    return LOGS_DIR / LOG_FILE_NAME


__all__ = [
    # Configuration Items
    'LOG_LEVEL',
    'LOG_FILE_BASENAME',
    'LOG_FILE_WITH_DATE',
    'LOG_FILE_NAME',
    'LOG_CONSOLE_OUTPUT',
    'LOG_FILE_OUTPUT',
    'LOG_MAX_BYTES',
    'LOG_BACKUP_COUNT',
    'LOG_ENCODING',
    'LOG_FORMAT',
    'LOG_SIMPLE_FORMAT',
    'LOG_DATE_FORMAT',
    'LOG_USE_COLOR',
    'LOG_ROTATION',
    'LOG_WHEN',
    'LOG_INTERVAL',
    'LOG_QUEUE_SIZE',
    'MODULE_LEVEL_OVERRIDES',
    'IGNORED_LOGGERS',
    'LOGS_DIR',
    # Mappings
    'LEVEL_MAP',
    'VALID_LEVELS',
    # Convenience Functions
    'get_log_level',
    'is_async_enabled',
]
