#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.log.core - Core Logger Class

Provides a singleton logger with support for asynchronous logging and concurrency safety.
"""

import logging, threading, atexit, queue
from logging.handlers import QueueHandler, QueueListener
from neuraxis_testkit.log.config import (
    LOG_LEVEL,
    LOG_CONSOLE_OUTPUT,
    LOG_FILE_OUTPUT,
    LOG_USE_COLOR,
    LOG_QUEUE_SIZE,
    LEVEL_MAP,
    VALID_LEVELS,
)
from neuraxis_testkit.log.handlers import create_handlers
from neuraxis_testkit.log.filters import ModuleLevelFilter, IgnoredLoggerFilter


class Logger:
    """
    Logger Management Class - Singleton Pattern

    Core Features:
        - Unified Logging: Consolidates all logs into a single log file (e.g., outputs/logs/neuraxis_testkit.log).
        - Thread-Safe: Ensures concurrency safety (uses locking).
        - Async Support: Supports asynchronous logging via QueueHandler and QueueListener.
        - Module Overrides: Supports module-level log level overrides.
        - Auto-directory Creation: Automatically creates the log directory.
    """

    _instances: dict[str, 'Logger'] = {}
    _lock = threading.Lock()

    # Global shared resources (initialized only once)
    _initialized_global = False
    _queue_listener: QueueListener | None = None
    _log_queue: queue.Queue | None = None
    _shared_handlers: list[logging.Handler] = []  # Handlers shared by all loggers

    def __new__(cls, name: str = 'root', *args, **kwargs):
        """Singleton pattern: Returns the same instance for identical names."""
        with cls._lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            return cls._instances[name]

    def __init__(
        self,
        name: str = 'root',
        level: str | None = None,
        console_output: bool | None = None,
        file_output: bool | None = None,
        use_color: bool | None = None,
        force_reconfigure: bool = False,
        use_async: bool = True,
    ):
        """
        Initializes the logger instance.

        Args:
            name: Logger name.
            level: Log level (None uses global configuration).
            console_output: Whether to output to console.
            file_output: Whether to output to file.
            use_color: Whether to use color in console.
            force_reconfigure: Whether to force reconfiguration.
            use_async: Whether to use asynchronous logging (improves concurrency performance).
        """
        # Prevent duplicate initialization (Singleton pattern)
        if hasattr(self, '_initialized') and self._initialized:
            if not force_reconfigure:
                return
            self._clear_handlers()

        # Set attributes
        self.name = name
        self._level = (level or LOG_LEVEL).upper()
        self.console_output = console_output if console_output is not None else LOG_CONSOLE_OUTPUT
        self.file_output = file_output if file_output is not None else LOG_FILE_OUTPUT
        self.use_color = use_color if use_color is not None else LOG_USE_COLOR
        self.use_async = use_async

        # Create underlying logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LEVEL_MAP.get(self._level, logging.INFO))
        self.logger.propagate = False  # Prevent propagation to root logger

        # Initialize global shared resources (executed only once)
        self._init_global_resources()

        # Add shared handlers (executed only once)
        self._setup_shared_handlers()

        # Add handlers to the current logger instance
        self._setup_handlers()

        self._initialized = True

    def _clear_handlers(self):
        """Clears existing handlers from the current logger."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    @classmethod
    def _init_global_resources(cls):
        """Initializes global shared resources (executed only once)."""
        if cls._initialized_global:
            return

        with cls._lock:
            if cls._initialized_global:
                return

            # Create log queue (for asynchronous logging)
            if LOG_QUEUE_SIZE > 0:
                cls._log_queue = queue.Queue(maxsize=LOG_QUEUE_SIZE)

            cls._initialized_global = True

    @classmethod
    def _setup_shared_handlers(cls):
        """
        Creates global shared handlers (executed only once).
        These handlers are shared by all logger instances.
        """
        if cls._shared_handlers:
            return

        with cls._lock:
            if cls._shared_handlers:
                return

            # Get log level
            level = LEVEL_MAP.get(LOG_LEVEL.upper(), logging.INFO)

            # Create handlers
            handlers = create_handlers(
                level, 
                LOG_CONSOLE_OUTPUT, 
                LOG_FILE_OUTPUT, 
                LOG_USE_COLOR
            )

            if handlers:
                if LOG_QUEUE_SIZE > 0 and cls._log_queue is not None:
                    # Async mode: Use QueueHandler
                    cls._setup_async_handlers(handlers)
                else:
                    # Sync mode: Store handlers directly
                    cls._shared_handlers = handlers

    @classmethod
    def _setup_async_handlers(cls, handlers: list[logging.Handler]):
        """Async mode: All handlers are managed via QueueListener."""
        if cls._queue_listener is None:
            cls._queue_listener = QueueListener(
                cls._log_queue,
                *handlers,
                respect_handler_level=True
            )
            cls._queue_listener.start()
            atexit.register(cls._stop_queue_listener)

        # Store handler references (for adding to loggers later)
        cls._shared_handlers = handlers

    @classmethod
    def _stop_queue_listener(cls):
        """Stops the queue listener."""
        if cls._queue_listener is not None:
            cls._queue_listener.stop()
            cls._queue_listener = None

    def _get_shared_handlers(self) -> list[logging.Handler]:
        """Retrieves shared handlers; creates QueueHandler if async mode is enabled."""
        if LOG_QUEUE_SIZE > 0 and Logger._log_queue is not None:
            # Async mode: Each logger uses a QueueHandler to enqueue logs
            queue_handler = QueueHandler(Logger._log_queue)
            queue_handler.setLevel(self.logger.level)
            return [queue_handler]
        else:
            # Sync mode: Use shared handlers directly
            return Logger._shared_handlers

    def _setup_handlers(self):
        """Adds handlers to the current logger (all loggers share the same handlers)."""
        # Clear existing handlers
        self._clear_handlers()

        # Apply filters
        self.logger.addFilter(ModuleLevelFilter())
        self.logger.addFilter(IgnoredLoggerFilter())

        # Retrieve shared handlers
        handlers = self._get_shared_handlers()

        # Add to current logger
        for handler in handlers:
            self.logger.addHandler(handler)

    # ========== Logging Methods ==========

    def trace(self, message: str, *args, **kwargs):
        self.logger.log(logging.TRACE, message, *args, **kwargs, stacklevel=2)

    def debug(self, message: str, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs, stacklevel=2)

    def info(self, message: str, *args, **kwargs):
        self.logger.info(message, *args, **kwargs, stacklevel=2)

    def warning(self, message: str, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs, stacklevel=2)

    def error(self, message: str, *args, **kwargs):
        self.logger.error(message, *args, **kwargs, stacklevel=2)

    def critical(self, message: str, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs, stacklevel=2)

    def exception(self, message: str, *args, **kwargs):
        kwargs.setdefault('exc_info', True)
        self.logger.error(message, *args, **kwargs, stacklevel=2)

    def log(self, level: str | int, message: str, *args, **kwargs):
        if isinstance(level, str):
            level = LEVEL_MAP.get(level.upper(), logging.INFO)
        self.logger.log(level, message, *args, **kwargs, stacklevel=2)

    # ========== Configuration Methods ==========

    def set_level(self, level: str):
        level_upper = level.upper()
        if level_upper not in VALID_LEVELS:
            raise ValueError(f"Invalid log level: '{level}'. Valid options: {', '.join(sorted(VALID_LEVELS))}")
        self._level = level_upper
        log_level = LEVEL_MAP.get(level_upper, logging.INFO)
        self.logger.setLevel(log_level)
        for handler in self.logger.handlers:
            handler.setLevel(log_level)

    def get_level(self) -> str:
        return self._level

    @property
    def level(self) -> str:
        return self._level

    @level.setter
    def level(self, value: str):
        self.set_level(value)

    # ========== Class Methods ==========

    @classmethod
    def get_logger(cls, name: str = 'root', **kwargs) -> 'Logger':
        return cls(name, **kwargs)

    @classmethod
    def get_all_loggers(cls) -> dict[str, 'Logger']:
        return cls._instances.copy()

    @classmethod
    def flush(cls):
        """Forces a flush of all logs (ensures all logs are written to disk)."""
        if cls._queue_listener is not None and cls._log_queue is not None:
            # Wait for all tasks in the queue to be processed
            cls._log_queue.join()
        # Flush all handlers synchronously
        for handler in cls._shared_handlers:
            try:
                handler.flush()
            except Exception:
                pass


__all__ = ['Logger']
