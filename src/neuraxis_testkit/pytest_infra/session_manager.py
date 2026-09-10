#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit/pytest_infra/session_manager.py - Test session management (shared resources, cross-process locks)

Design notes:
  1. portalocker lock file is placed in the same directory and with the same stem as the target file (results_20260828.csv.lock).
  2. Under xdist multiprocessing: each worker holds the lock for a short time (only wrapping the few lines of file writing).
  3. On lock acquisition failure, degrade to lock-free mode and issue a warning — risk of recording result loss is acceptable over interrupting tests.
  4. Follows context manager protocol to ensure release on any exception path across platforms.

"""
from __future__ import annotations

import portalocker
from pathlib import Path
from types import TracebackType
from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_POLL_INTERVAL = 0.1


class SessionFileLock:
    """
    Cross-platform file lock (portalocker wrapper).
    Windows -> LockFileEx / POSIX -> fcntl.flock, automatically adapted by portalocker.
    """

    def __init__(
        self,
        target_file: Path | str,
        timeout: float = DEFAULT_TIMEOUT,
        fail_when_locked: bool = False,
    ):
        target = Path(target_file)
        self.lock_path = target.parent / f"{target.stem}.lock"
        self.timeout = timeout
        self.fail_when_locked = fail_when_locked
        self._lock: portalocker.Lock | None = None
        self._acquired = False

    def acquire(self) -> bool:
        self._lock = portalocker.Lock(
            str(self.lock_path),
            fail_when_locked=self.fail_when_locked,
            timeout=self.timeout,
            poll_interval=DEFAULT_POLL_INTERVAL,
        )
        try:
            self._lock.acquire()
            self._acquired = True
            return True
        except portalocker.exceptions.AlreadyLocked:
            logger.warning(f"File lock held (timeout {self.timeout}s): {self.lock_path}")
            return False
        except Exception as exp:
            logger.warning(f"Failed to acquire file lock, degrading to lock‑free mode: {exp}")
            return False

    def release(self) -> None:
        if self._acquired and self._lock is not None:
            try:
                self._lock.release()
            except Exception:
                pass
            self._acquired = False

    # Context manager protocol
    def __enter__(self) -> "SessionFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb: TracebackType | None) -> None:
        self.release()


class SessionManager:
    """
    Session-level shared resource container: held by the session fixture in conftest,
    ensuring that the entire test session initializes only once (client, directories, CSV header, etc.).
    """

    def __init__(self, results_dir: Path, csv_path: Path):
        self.results_dir = results_dir
        self.csv_path = csv_path
        self._lock = SessionFileLock(csv_path)
        self._initialized = False

    def ensure_initialized(self) -> None:
        """Idempotent initialization: create directory + CSV header if absent."""
        if self._initialized:
            return
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            import csv
            with self._lock, open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["test_id", "module_path", "func_name", "status", "message", "duration", "timestamp"]
                )
        self._initialized = True

    def locked_append(self, write_fn) -> None:
        """Execute write operation under lock protection. write_fn: Callable[[None], None]"""
        with self._lock:
            write_fn()

    def close(self) -> None:
        self._lock.release()
