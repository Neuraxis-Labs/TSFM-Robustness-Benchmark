#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.utils.concurrent - Concurrent-Safety Common Module
====================================
Provides cross-process safe caches, locks, and counters supporting concurrent read/write operations.
Designed for pytest-xdist multi-process testing environments.

Platform Support:
    Utilizes `portalocker` for unified cross-platform file locking:
    - Unix/Linux/macOS: Underlying `fcntl.flock`, supports shared/exclusive locks.
    - Windows: Underlying `LockFileEx` (Win32 API), supports shared/exclusive locks.
    - Both platforms support true LOCK_SH / LOCK_EX.

Design Notes:
    `portalocker` relies on OS-level advisory locks (`fcntl.flock` / `LockFileEx`).
    Lock information is stored in the kernel; the OS automatically reclaims file descriptors (fd) 
    and releases locks upon process crashes. 
    Therefore, stale lock cleanup is theoretically unnecessary.

    However, as a defensive measure, `_clean_stale_lock()` employs atomic rename + PID checking 
    to handle residual lock files in extreme scenarios (e.g., fd not closed properly due to SIGKILL).
    This method deletes the `.stale` file regardless of expiration status and does not restore 
    the original path, avoiding overwriting new lock files potentially created by other processes.

Create Date: 2026/08/25.
"""

import os, time, json, atexit, tempfile, portalocker
from pathlib import Path
from typing import Any
from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)

# ============================================================
# 1. Distributed Lock (Supports Shared/Exclusive)
# ============================================================

class FileLock:
    """
    File-based distributed lock supporting Shared (Read) and Exclusive (Write) locks.

    Unified cross-platform implementation via `portalocker`:
    - Unix/Linux/macOS: Underlying `fcntl.flock`, kernel-managed, auto-release on crash.
    - Windows: Underlying `LockFileEx` (Win32 API), kernel-managed, auto-release on crash.
    - Both platforms support true shared locks (LOCK_SH) and exclusive locks (LOCK_EX).

    Stale Cleanup:
        Kernel-level locks from `portalocker` are usually released automatically on process crashes.
        `_clean_stale_lock()` serves as a defensive measure using atomic rename to prevent TOCTOU issues:
        1. Atomically rename the lock file to .stale (atomic operation on the same filesystem).
        2. Exclusively check on .stale: is the PID alive + is stale_timeout exceeded.
        3. Regardless of expiration, delete .stale and do not restore the original path.
        This completely avoids the race condition of "overwriting a new lock file created by another process during recovery."

        `_clean_stale_lock()` is called before `acquire()` (outside the lock).
        Placing it inside the lock would create a deadlock: "need lock to clean, need clean to acquire lock."
        Atomic rename ensures only one process succeeds in renaming; others return safely without race conditions.

    Thread Safety: This class is NOT thread-safe; each thread should create an independent FileLock instance.
    Process Safety: This class is process-safe (guaranteed by portalocker file locks).
    """

    def __init__(self, lock_name: str, lock_dir: Path | None = None,
                 timeout: float = 30.0, stale_timeout: float = 300.0):
        """
        Initialize the file lock instance.

        Args:
            lock_name: Lock name (used to generate a unique lock filename).
            lock_dir: Directory for lock files. Defaults to the system temp directory.
            timeout: Timeout (in seconds) for acquiring the lock. Raises LockException on timeout.
            stale_timeout: Lock expiration time (in seconds) for the stale cleanup fallback mechanism.
                          Should be set sufficiently long (e.g., 300s) to prevent accidental deletion of valid locks.
                          Default is 300s.
        """
        self.lock_name = lock_name   # Name of the lock
        self.lock_dir = Path(lock_dir or tempfile.gettempdir())  # Lock file storage directory
        self.lock_dir.mkdir(parents=True, exist_ok=True)  # Ensure lock directory exists
        self._lock_file = self.lock_dir / f"tsfm_lock_{lock_name}.lock"  # Lock file path
        self._timeout = timeout      # Lock acquisition timeout
        self._stale_timeout = stale_timeout  # Lock expiration time
        self._fh: Any = None         # File handle (managed by portalocker)
        self._acquired = False       # Whether the lock is currently held
        self._is_shared = False      # Whether it is a shared lock
        self._my_pid = os.getpid()   # Current process ID (written to lock file metadata)
        # Use atexit to ensure lock is released on process exit (more reliable than __del__)
        atexit.register(self._atexit_release)

    def _atexit_release(self):
        """atexit callback: Release lock on process exit. More reliable than __del__."""
        try:
            self.release()
        except Exception:
            pass  # Silent ignore to avoid interfering with normal exit flow

    # ---------- Context Manager Entry Points ----------
    def shared(self):
        """Context manager for acquiring a Shared (Read) lock. Cross-platform support."""
        return _LockContext(self, shared=True)

    def exclusive(self):
        """Context manager for acquiring an Exclusive (Write) lock."""
        return _LockContext(self, shared=False)

    # Compatible with legacy usage (defaults to exclusive)
    def __enter__(self):
        self.acquire(shared=False)     # Acquire exclusive lock upon entering context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()       # Release lock upon exiting context

    def __del__(self):
        """
        Ensure lock is released upon destruction.
        Module-level variables may already be destroyed during interpreter shutdown, 
        so this acts only as a fallback.
        Primary reliance is on the atexit callback for lock release.
        """
        try:
            self.release()
        except Exception:
            pass       # Errors possible during interpreter shutdown, silently ignore

    # ---------- Core Acquire/Release ----------
    def acquire(self, shared: bool = False) -> None:
        """
        Acquire the lock. Raises portalocker.LockException on timeout.

        Uses `portalocker` for cross-platform file locking:
        - Shared Lock (LOCK_SH): Can be held by multiple processes simultaneously, for concurrent reads.
        - Exclusive Lock (LOCK_EX): Exclusive access, for write operations.

        File Mode:
        - Uses "a+b" mode (append + binary); open() creates the file automatically.
        - Shared Lock: Does not modify file content (holds lock only).
        - Exclusive Lock: Writes pid:timestamp metadata (for stale detection).

        Args:
            shared: True=Shared Lock (Read Lock), False=Exclusive Lock (Write Lock).
                    Cross-platform support; both Windows and Unix support shared locks.

        Raises:
            portalocker.LockException: Timeout or failure to acquire lock.
            RuntimeError: Lock already acquired by this instance.
        """
        if self._acquired:
            raise RuntimeError("Lock already acquired by this instance")

        self._clean_stale_lock()

        # Acquire lock using portalocker
        lock_type = portalocker.LOCK_SH if shared else portalocker.LOCK_EX
        # Non-blocking flag: used to implement timeout
        lock_type_nb = lock_type | portalocker.LOCK_NB

        start_time = time.time()
        last_exception = None

        while True:
            # Timeout check
            if time.time() - start_time > self._timeout:
                if last_exception:
                    raise last_exception
                raise portalocker.LockException(
                    f"Timeout acquiring lock: {self.lock_name} (>{self._timeout}s)"
                )

            try:
                self._fh = open(self._lock_file, "a+b")   # Open file
                portalocker.lock(self._fh, lock_type_nb)  # Non-blocking acquisition
                break  # Acquired successfully

            except portalocker.LockException as exp:
                # Lock is busy, close handle and retry
                last_exception = exp
                if self._fh is not None:
                    try:
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
                time.sleep(0.1)

            except OSError as osErr:
                # Other filesystem errors
                last_exception = osErr
                if self._fh is not None:
                    try:
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
                time.sleep(0.1)

        # Acquired successfully, write metadata (exclusive lock only)
        if not shared:
            try:
                self._fh.seek(0)
                self._fh.truncate(0)
                meta = f"{self._my_pid}:{time.time()}\n".encode()
                self._fh.write(meta)
                self._fh.flush()
                try:
                    os.fsync(self._fh.fileno())
                except OSError:
                    pass
            except Exception:
                # Metadata write failed, release lock and re-raise
                try:
                    portalocker.unlock(self._fh)
                except Exception:
                    pass
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None
                raise

        self._acquired = True
        self._is_shared = shared

    def release(self) -> None:
        """
        Release the lock.

        Removes portalocker lock and closes file handle:
        - Unix: portalocker.unlock -> fcntl.flock LOCK_UN
        - Windows: portalocker.unlock -> UnlockFileEx
        """
        if not self._acquired:
            return

        try:
            if self._fh is not None:
                # 1. Explicit unlock
                try:
                    portalocker.unlock(self._fh)
                except Exception:
                    pass  # Unlock failure does not affect subsequent close

                # 2. Close file handle (close implicitly releases lock on Linux)
                try:
                    self._fh.close()
                except Exception:
                    pass  # Ignore close failures

        finally:
            # 3. Always reset state
            self._fh = None
            self._acquired = False
            self._is_shared = False


    # ---------- Helper Methods ----------
    def _clean_stale_lock(self) -> None:
        """
        Atomically clean expired lock files (rename only, no deletion).

        Design Decision: 设计决策
        - Only rename .lock to .stale; do not delete .stale files.
        - .stale files will be overwritten by the next call to replace().
        - Avoids issues with unlink() blocking on Windows.
        - FileLock.cleanup_locks() can be used for batch cleanup at the end of tests.
        """
        if not self._lock_file.exists():
            return

        temp_stale = self._lock_file.with_suffix('.stale')
        try:
            # Path.replace = os.rename, which is atomic on the same filesystem
            # If .stale exists, replace() will overwrite it
            self._lock_file.replace(temp_stale)
            logger.trace(f"Renamed {self._lock_file.name} to {temp_stale.name}")
        except OSError:
            # File processed or deleted by another process
            pass

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """
        Check if the process with the specified PID is still alive.

        Uses os.kill(pid, 0) to send signal 0; does not actually send a signal.
        Only checks if the PID exists, not if it is the same process (PID may be reused).

        Known Risk: PID reuse may cause false positives; mitigated by stale_timeout at the upper layer.
        """
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    @classmethod
    def cleanup_locks(cls, lock_dir: Path | None = None):
        """
        Clean up all lock files and remaining .stale files.
        Recommended to call at the end of the test suite.
        """
        lock_dir = Path(lock_dir or tempfile.gettempdir())
        for pattern in ("tsfm_lock_*.lock", "tsfm_lock_*.stale"):
            for f in lock_dir.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass


class _LockContext:
    """Context manager for `with` statements."""
    def __init__(self, lock: FileLock, shared: bool):
        self._lock = lock
        self._shared = shared

    def __enter__(self):
        self._lock.acquire(shared=self._shared)
        return self._lock

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


# ============================================================
# 2. Process-Safe Cache (Using Read-Write Lock)
# ============================================================

class ProcessSafeCache:
    """
    Process-safe cache — supports multi-process concurrent reads, exclusive writes.

    Uses `portalocker` for cross-platform read-write locks:
    - Read operations: Shared Lock (LOCK_SH), concurrent reads by multiple processes.
    - Write operations: Exclusive Lock (LOCK_EX), exclusive write.
    - Both Windows and Unix support true shared locks.
    """

    def __init__(self, cache_name: str, cache_dir: Path | None = None):
        """
        Args:
            cache_name: Cache name (used as filename prefix).
            cache_dir: Cache directory. Defaults to the system temp directory.
        """
        self.cache_name = cache_name
        self.cache_dir = Path(cache_dir or tempfile.gettempdir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / f"tsfm_cache_{cache_name}.json"
        # Use file lock to protect cross-process access
        self._file_lock = FileLock(
            f"cache_{cache_name}", lock_dir=self.cache_dir, timeout=10.0)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cache value (concurrent read, uses shared lock).
        """
        with self._file_lock.shared():
            data = self._load()
            return data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set cache value (exclusive write)"""
        with self._file_lock.exclusive():
            data = self._load()
            data[key] = value
            self._save(data)

    def delete(self, key: str) -> None:
        """Delete cache value (exclusive write)"""
        with self._file_lock.exclusive():
            data = self._load()
            if key in data:
                del data[key]
                self._save(data)

    def clear(self) -> None:
        """Clear cache (exclusive write)"""
        with self._file_lock.exclusive():
            if self._cache_file.exists():
                try:
                    self._cache_file.unlink()
                except OSError as osErr:
                    logger.error(f"Failed to clear cache file: {osErr}")

    def _load(self) -> dict:
        """
        Load cache from file.
        Contract: Caller must hold the lock (shared or exclusive).
        """
        if not self._cache_file.exists():
            return {}
        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as jsonErr:
            logger.error(f"Cache file corrupted ({self._cache_file}): {jsonErr}")
            return {}
        except OSError as osErr:
            logger.error(f"Failed to read cache file: {osErr}")
            return {}

    def _save(self, data: dict) -> None:
        """
        Atomically save cache (caller must hold exclusive lock).
        Uses temp file + rename to ensure atomicity.
        Logs errors on failure without raising exceptions (test framework should not crash due to cache issues).
        """
        tmp_file = self._cache_file.with_suffix('.tmp')
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_file.replace(self._cache_file)
        except OSError as osErr:
            logger.error(f"Failed to save cache: {osErr}")
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except OSError:
                pass

    @classmethod
    def cleanup_all(cls, cache_dir: Path | None = None):
        """Clean up all cache files (including temp files)."""
        cache_dir = Path(cache_dir or tempfile.gettempdir())
        for pattern in ("tsfm_cache_*.json", "tsfm_cache_*.json.tmp"):
            for f in cache_dir.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass


# ============================================================
# 3. Process-Safe Counter (Exclusive Lock, Reads Shared)
# ============================================================

class ProcessSafeCounter:
    """
    Process-safe counter — reads can be shared, writes are exclusive.

    Uses `portalocker` for cross-platform read-write locks:
    - Read operations: Shared Lock (LOCK_SH), concurrent reads by multiple processes.
    - Write operations: Exclusive Lock (LOCK_EX), exclusive write.
    - Both Windows and Unix support true shared locks.
    """

    def __init__(self, counter_name: str, counter_dir: Path | None = None):
        self.counter_name = counter_name
        self.counter_dir = Path(counter_dir or tempfile.gettempdir())
        self.counter_dir.mkdir(parents=True, exist_ok=True)
        self._counter_file = (
            self.counter_dir / f"tsfm_counter_{counter_name}.txt"
        )
        self._file_lock = FileLock(
            f"counter_{counter_name}",
            lock_dir=self.counter_dir,
            timeout=10.0,
        )

    def increment(self, delta: int = 1) -> int:
        """Increment counter (exclusive write)."""
        with self._file_lock.exclusive():
            current = self._load()
            new_value = current + delta
            self._save(new_value)
            return new_value

    def get(self) -> int:
        """Get current value (shared read)."""
        with self._file_lock.shared():
            return self._load()

    def reset(self) -> None:
        """
        Reset counter (exclusive write).
        Uniformly uses _save(0) instead of deleting the file to maintain consistent file state.
        """
        with self._file_lock.exclusive():
            self._save(0)

    def _load(self) -> int:
        """Load counter value. Contract: Caller must hold the lock."""
        if not self._counter_file.exists():
            return 0
        try:
            with open(self._counter_file, 'r', encoding='utf-8') as f:
                return int(f.read().strip())
        except (ValueError, OSError) as err:
            logger.error(f"Failed to load counter: {err}")
            return 0

    def _save(self, value: int) -> None:
        """
        Atomically save counter value (caller must hold exclusive lock).
        Uses temp file + rename to ensure atomicity.
        Logs errors on failure without raising exceptions (test framework should not crash due to counter issues).
        """
        tmp_file = self._counter_file.with_suffix('.tmp')
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                f.write(str(value))
            tmp_file.replace(self._counter_file)
        except OSError as osErr:
            logger.error(f"Failed to save counter: {osErr}")
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except OSError:
                pass

    @classmethod
    def cleanup_all(cls, counter_dir: Path | None = None):
        """Clean up all counter files (including temp files)."""
        counter_dir = Path(counter_dir or tempfile.gettempdir())
        for pattern in ("tsfm_counter_*.txt", "tsfm_counter_*.txt.tmp"):
            for f in counter_dir.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass


# ============================================================
# 4. xdist Environment Detection
# ============================================================

def is_xdist_worker() -> bool:
    """Check if the current process is a pytest-xdist worker."""
    return os.environ.get("PYTEST_XDIST_WORKER") is not None

def get_worker_id() -> str:
    """Get the current worker ID (returns 'master' for the main process)."""
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


# ============================================================
# 5. Result Merging Utility
# ============================================================

def merge_results_from_workers(
    result_dir: Path,
    pattern: str = "tsfm_results_worker_*.json",
    timeout: float = 5.0,  # Timeout for waiting for file completion
) -> list[dict]:
    """
    Merge result files generated by all workers.

    Safety Guarantees:
    1. Only reads final files (non-.tmp), skipping temporary files being written.
    2. Workers should use the "write temp file + atomic rename" pattern:
       tmp = result_dir / f"tsfm_results_worker_{wid}.json.tmp"
       with open(tmp, 'w') as f:
           json.dump(results, f)
       tmp.replace(result_dir / f"tsfm_results_worker_{wid}.json")
    3. Reads with timeout retry to wait for file write completion.
    4. Retry on JSONDecodeError (file may be writing); skip on OSError (file issue).
    5. Failed reads are logged via logger.error() and skipped.

    Args:
        result_dir: Directory containing results.
        pattern: Glob pattern for result files.
        timeout: Timeout (in seconds) for waiting for a single file read to complete.

    Returns:
        Merged list of results (each element is a dictionary)
    """
    merged = []
    for result_file in sorted(result_dir.glob(pattern)):
        # Skip temp files (may be being written)
        if result_file.suffix == '.tmp':
            continue

        # Wait for file integrity (with timeout retry)
        data = None
        start = time.time()
        while time.time() - start < timeout:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                break  # Read successfully, exit retry loop
            except json.JSONDecodeError:
                # File may be being written, wait and retry
                time.sleep(0.1)
            except OSError as osErr:
                # File deleted or permission issue, skip file and exit retry loop
                logger.error(f"Cannot read {result_file}: {osErr}")
                break

        if data is None:
            logger.warning(f"File {result_file} incomplete after {timeout}s, skipping")
            continue

        # Process data
        if isinstance(data, list):
            merged.extend(data)
        elif isinstance(data, dict):
            merged.append(data)
        # Other types ignored

    return merged
