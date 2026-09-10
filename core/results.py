#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/results.py - Test Result Manager
====================================
Module Purpose:
  Provides centralized management for test results, including result loading,
  caching, batch writing, and querying. Encapsulates business logic while
  leveraging the utils layer for file operations.

Core Features:
  - Result persistence with batch buffering (reduces I/O overhead)
  - Historical result loading with error classification
  - Multi-dimensional result querying (by model, scene, pass)
  - Independent buffer per file (safe for concurrent tasks)
  - Supports pytest-xdist multi-process environments

Module Position in Architecture:
  - Calls: neuraxis_testkit.utils.concurrent (Concurrent Safety), neuraxis_testkit.utils.files (file operations), neuraxis_testkit.logger (logging)
  - Called by: testcases.* (business logic), resume.py (breakpoint logic)

Author: Janesong
Create Date: 2026/07/19, Updated on 2026/08/25.
"""

import json, tempfile
from pathlib import Path
from typing import Any
from core.resume import is_rate_limited
from neuraxis_testkit.utils.concurrent import FileLock, ProcessSafeCache
from neuraxis_testkit.utils.files import append_to_csv, csv_exists_and_not_empty, read_csv_to_list
from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)

# ============================================================
# Process-Safe Buffer Manager (using ProcessSafeCache)
# ============================================================

class _ConcurrentResultBufferManager:
    """
    Concurrent-safe buffer manager supporting xdist multi-process environments.

    Uses ProcessSafeCache (underlying portalocker file lock) to ensure cross-process safety:
    - Buffer Storage: Each target file corresponds to one ProcessSafeCache entry.
    - Lock Reuse: Each file path corresponds to a cached FileLock instance to avoid recreation.
    - Atomic Operation: append_and_maybe_flush combines append + check + flush into one atomic operation.
    - Atomic Write: Buffer files are saved via ProcessSafeCache's temp file + rename pattern.

    Buffer Data Format (JSON):
        {
            "results": [dict, dict, ...],   # List of results
            "target_file": "/path/to.csv",  # Target CSV file path
            "batch_size": 20                # Batch size
        }
    """

    def __init__(self, default_batch_size: int = 20):
        """
        Initialize buffer manager.

        Args:
            default_batch_size: Default batch size for auto-flush (default 20)
        """
        self._default_batch_size = default_batch_size
        self._temp_dir = Path(tempfile.gettempdir()) / "tsfm_buffers"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        # Use ProcessSafeCache to manage buffer data (underlying portalocker file lock)
        self._cache = ProcessSafeCache(
            cache_name="result_buffers", cache_dir=self._temp_dir)
        # FileLock instance cache: Avoids recreating instances which causes atexit list bloat
        self._lock_cache: dict[str, FileLock] = {}

    def _get_lock(self, file_path: str) -> FileLock:
        """Get the FileLock instance corresponding to the file path (cached for reuse)."""
        safe_name = Path(file_path).stem.replace("\\", "_").replace("/", "_")
        cache_key = f"buffer_{safe_name}"
        if cache_key not in self._lock_cache:
            self._lock_cache[cache_key] = FileLock(
                cache_key, lock_dir=self._temp_dir, timeout=10.0)
        return self._lock_cache[cache_key]

    def _get_cache_key(self, file_path: str) -> str:
        """Get the key name of the buffer in ProcessSafeCache."""
        safe_name = Path(file_path).stem.replace("\\", "_").replace("/", "_")
        return f"buffer_{safe_name}"

    def append_and_maybe_flush(
        self,
        file_path: str,
        result: dict[str, Any],
        batch_size: int | None = None,
        force_flush: bool = False,
    ) -> int:
        """
        Append result to buffer and atomically flush when conditions are met.
        Combines append + check flush + flush into a single atomic operation to avoid race conditions.

        Operation Flow (within exclusive lock):
        1. Read current buffer.
        2. Append new result.
        3. Determine if batch threshold is reached or force flush is needed.
        4. If flush needed: Clear buffer -> Write CSV (restore buffer if failed).
        5. If no flush needed: Save buffer.

        Args:
            file_path: Target CSV file path.
            result: Single result dictionary.
            batch_size: Batch size (effective after first set, None uses default).
            force_flush: Whether to force flush immediately.

        Returns:
            Number of flushed records (0 indicates no flush).
        """
        cache_key = self._get_cache_key(file_path)
        lock = self._get_lock(file_path)

        with lock.exclusive():  # Exclusive lock, ensuring atomicity of the entire operation group
            # Read current buffer
            buffer_data = self._cache.get(cache_key, default=None)
            if buffer_data is None:
                # Initialize buffer for the first time
                buffer_data = {
                    "results": [],
                    "target_file": file_path,
                    "batch_size": batch_size or self._default_batch_size,
                }

            results = buffer_data.get("results", [])
            # Update target_file (in case of path changes)
            buffer_data["target_file"] = file_path
            # Update batch_size (if new value is provided)
            if batch_size is not None:
                buffer_data["batch_size"] = batch_size

            # Append result
            results.append(result)
            buffer_data["results"] = results

            # Check if flush is needed
            effective_batch_size = buffer_data.get(
                "batch_size", self._default_batch_size)
            need_flush = (
                force_flush or len(results) >= effective_batch_size
            )

            if need_flush and results:
                # Atomic flush: Clear buffer first, then write CSV
                buffer_data["results"] = []
                self._cache.set(cache_key, buffer_data)

                try:
                    append_to_csv(file_path, results)
                    count = len(results)
                    logger.debug(
                        f"Batch written {count} records "
                        f"to {Path(file_path).name}")
                    return count
                except Exception as exp:
                    # CSV write failed: Restore buffer data to avoid loss
                    logger.error(f"Failed to flush buffer: {exp}")
                    buffer_data["results"] = results
                    self._cache.set(cache_key, buffer_data)
                    return 0
            else:
                # No flush: Just save buffer
                self._cache.set(cache_key, buffer_data)
                return 0

    def flush(self, file_path: str) -> int:
        """
        Flush buffer to CSV file.

        Args:
            file_path: Target CSV file path.

        Returns:
            Number of records flushed.
        """
        cache_key = self._get_cache_key(file_path)
        lock = self._get_lock(file_path)

        with lock.exclusive():
            buffer_data = self._cache.get(cache_key, default=None)
            if buffer_data is None:
                return 0

            results = buffer_data.get("results", [])
            if not results:
                return 0

            # Clear buffer
            buffer_data["results"] = []
            self._cache.set(cache_key, buffer_data)

            try:
                append_to_csv(file_path, results)
                count = len(results)
                logger.debug(f"Flushed {count} records to {Path(file_path).name}")
                return count
            except Exception as exp:
                # Write failed: Restore buffer
                logger.error(f"Failed to flush buffer: {exp}")
                buffer_data["results"] = results
                self._cache.set(cache_key, buffer_data)
                return 0

    def flush_all(self) -> dict[str, int]:
        """
        Flush all buffers (call this before program exit).

        Design Note (Eventual Consistency, No Global Lock):
        Does not use global_lock, but utilizes file-level locks for each flush to ensure atomicity.
        Reason:
        1. ProcessSafeCache uses atomic rename writes; reads won't see corrupted data.
        2. Each flush(file_path) has its own file-level exclusive lock, which is atomic.
        3. If buffer is cleared by another process during flush, flush returns 0 (correct).
        4. If other processes append data after flush, the next call will handle it.
        5. Avoids lock nesting (global lock -> file lock) and blocking caused by long-held global locks.

        Flow:
        1. Read cache file to get target_files (no lock, relying on atomic rename safety).
        2. Iteratively flush(file_path), each with its own independent file lock.

        Returns:
            Dict of {file_path: record_count}
        """
        flush_results = {}

        cache_file = self._temp_dir / "tsfm_cache_result_buffers.json"
        if not cache_file.exists():
            return flush_results

        # Read cache file (no lock, ProcessSafeCache uses atomic rename to ensure consistency)
        # Might read slightly stale data, but correctness is unaffected:
        # - If a buffer is cleared by another process after reading, flush returns 0.
        # - If a buffer is appended to by another process after reading, next flush_all handles it.
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                all_buffers = json.load(f)
        except (json.JSONDecodeError, OSError) as exp:
            logger.error(f"Failed to read buffer cache: {exp}")
            return flush_results

        # Collect all target file paths
        target_files = set()
        for cache_key, buffer_data in all_buffers.items():
            if isinstance(buffer_data, dict):
                target = buffer_data.get("target_file")
                if target:
                    target_files.add(target)

        # Flush individually (each flush has its own file-level exclusive lock, no global lock needed)
        for file_path in target_files:
            try:
                count = self.flush(file_path)
                if count > 0:
                    flush_results[file_path] = count
            except Exception as exp:
                logger.error(f"Failed to flush {file_path}: {exp}")

        return flush_results


    def get_buffer_size(self, file_path: str) -> int:
        """
        Get buffer size for specific file (read-only, for debugging).

        Args:
            file_path: Target CSV file path.

        Returns:
            Number of records in the current buffer.
        """
        cache_key = self._get_cache_key(file_path)
        lock = self._get_lock(file_path)

        with lock.shared():  # Shared lock, allows concurrent reads
            buffer_data = self._cache.get(cache_key, default=None)
            if buffer_data is None:
                return 0
            return len(buffer_data.get("results", []))


# ============================================================
# Global Buffer Manager
# ============================================================

_buffer_manager = _ConcurrentResultBufferManager(default_batch_size=20)


# ============================================================
# Public API
# ============================================================

def load_results_from_csv(result_csv_path_file: str) -> tuple[list[dict[str, Any]], int]:
    """
    Load historical results from CSV with error classification.

    This method reads existing results and classifies them into:
        - Successful records
        - Permanent failures (non-rate-limit errors)
        - Rate-limit errors (429)

    Uses neuraxis_testkit.utils.files for file operations (follows layer architecture).

    Args:
        result_csv_path_file: Result CSV file path.

    Returns:
        (all_records, non_rate_limit_error)
        - all_records: List of all records (each row as dict)
        - non_rate_limit_error: Count of non-rate-limit errors
    """
    # Use methods provided by neuraxis_testkit.utils.files (follows layered architecture)
    if not csv_exists_and_not_empty(result_csv_path_file):
        logger.info(f"{Path(result_csv_path_file).name} not found, starting fresh")
        return [], 0

    try:
        # Call neuraxis_testkit.utils.files.read_csv_to_list()
        all_records = read_csv_to_list(result_csv_path_file)

        # Classify error types
        non_rate_limit_error = 0
        retry_count = 0
        for record in all_records:
            success_val = record.get("success", "")
            if str(success_val).strip().lower() == "true":
                continue

            # Classify failure reason
            if is_rate_limited(str(record.get("error", ""))):
                retry_count += 1
            else:
                non_rate_limit_error += 1

        msg = (
            f"Loaded {Path(result_csv_path_file).name}: "
            f"{len(all_records)} records"
        )
        success_count = (
            len(all_records) - non_rate_limit_error - retry_count
        )
        msg += f" (Success: {success_count}"

        if non_rate_limit_error > 0:
            msg += f", Failed: {non_rate_limit_error}"
        if retry_count > 0:
            msg += f", Pending Retry: {retry_count}"
        msg += ")"
        logger.info(msg)
        return all_records, non_rate_limit_error

    except Exception as exp:
        logger.error(f"Failed to load {Path(result_csv_path_file).name}: {exp}")
        return [], 0

def append_result_to_csv(
    result_csv_path_file: str,
    result: dict[str, Any],
    batch_size: int = 10,
    force_flush: bool = False,
    validate: bool = True,
) -> None:
    """
    Append test result to CSV with batch buffering.

    Core Features:
        - Batch buffering to reduce I/O overhead
        - Auto-flush when buffer reaches batch_size
        - Independent buffer per file (safe for concurrent tasks)
        - Atomic append+flush (no race condition)

    Example:
        # Normal append (buffered)
        append_result_to_csv("./results.csv", result)

        # Force flush immediately
        append_result_to_csv("./results.csv", result, force_flush=True)

        # Custom batch size
        append_result_to_csv("./results.csv", result, batch_size=50)

    Args:
        result_csv_path_file: Result CSV file path
        result: Single result dictionary
        batch_size: Batch size for auto-flush (default 20)
        force_flush: Force flush immediately (default False)
        validate: Validate result format (default True)
    """
    # Business logic 1: Data validation
    # if validate:
    #    _validate_result_format(result)

    # Business logic 2: Field normalization
    result = _normalize_result(result)

    # Business logic 3: Atomic append + maybe flush (merged operation to avoid race condition)
    flushed = _buffer_manager.append_and_maybe_flush(
        result_csv_path_file,
        result,
        batch_size=batch_size,
        force_flush=force_flush,
    )

    if flushed > 0:
        logger.debug(
            f"Flushed {flushed} records for "
            f"{Path(result_csv_path_file).name}")


def flush_all_results() -> dict[str, int]:
    """
    Flush all result buffers (call before program exit).

    Returns:
        Dict of {file_path: flushed_record_count}

    Example:
        >>> from core.results import flush_all_results
        >>> stats = flush_all_results()
        >>> print(f"Flushed: {stats}")
    """
    return _buffer_manager.flush_all()


def get_results(results_data: list[dict[str, Any]], 
               model_id: str,
               scene_prefix: str,
               pass_name: str = "Preprocessed") -> dict[str, Any] | None:
    """
    Retrieve the first matching test result for a specific model, scene, and pass.

    Args:
        results_data: List of result dictionaries (typically from CSV).
        model_id: Model identifier (e.g., "Timer-3.0", "Timer-3.5")
        scene_prefix: Scene prefix to match (e.g., "S0", "S1", "S4")
                     Uses prefix matching, so "S0" matches "S0-Clean[Preprocessed]".
        pass_name: Pass name ("Preprocessed" or "Raw"). Defaults to "Preprocessed".

    Returns:
        Matching result dictionary, or None if not found.

    Example:
        >>> results = [
        ...     {"model_id": "Timer-3.5", "scene": "S0-Clean[Preprocessed]", "pass_name": "Preprocessed", "mae": 0.5},
        ...     {"model_id": "Timer-3.0", "scene": "S1-Missing5%[Raw]", "pass_name": "Raw", "mae": None}
        ... ]
        >>> get_results(results, "Timer-3.5", "S0", "Preprocessed")
        {'model_id': 'Timer-3.5', 'scene': 'S0-Clean[Preprocessed]', 'pass_name': 'Preprocessed', 'mae': 0.5}
    """
    for record in results_data:
        if (record.get("model_id") == model_id 
            and record.get("scene", "").startswith(scene_prefix)
            and record.get("pass_name") == pass_name):
            return record
    return None

def get_results_by_model(results_data: list[dict[str, Any]], 
                         model_id: str) -> list[dict[str, Any]]:
    """
    Retrieve all results for a specific model.

    Args:
        results_data: List of result dictionaries
        model_id: Model identifier (e.g., "Timer-3.0", "Timer-3.5")

    Returns:
        List of matching result dictionaries. Empty list if none found.

    Example:
        >>> results = [
        ...     {"model_id": "Timer-3.0", "scene": "S0-Clean[Preprocessed]", "pass_name": "Preprocessed", "mae": 0.5},
        ...     {"model_id": "Timer-3.5", "scene": "S1-Missing5%[Raw]", "pass_name": "Raw", "mae": None}
        ... ]
        >>> get_results_by_model(results, "Timer-3.5")
        [{'model_id': 'Timer-3.5', 'scene': 'S1-Missing5%[Raw]', 'pass_name': 'Raw', 'mae': None}]
    """
    return [record for record in results_data if record.get("model_id") == model_id]

def get_results_by_scene(results_data: list[dict[str, Any]],
                         scene_prefix: str) -> list[dict[str, Any]]:
    """
    Retrieve all results for a specific scene prefix.

    Args:
        results_data: List of result dictionaries
        scene_prefix: Scene prefix (e.g., "S0", "S1", "S4")

    Returns:
        List of matching result dictionaries. Empty list if none found.

    Example:
        >>> results = [
        ...     {"model_id": "Timer-3.5", "scene": "S0-Clean[Preprocessed]", "pass_name": "Preprocessed", "mae": 0.5},
        ...     {"model_id": "Timer-3.5", "scene": "S1-Missing5%[Raw]", "pass_name": "Raw", "mae": None}
        ... ]
        >>> get_results_by_scene(results, "S0")
        [{'model_id': 'Timer-3.5', 'scene': 'S0-Clean[Preprocessed]', 'pass_name': 'Preprocessed', 'mae': 0.5}]
    """
    return [record for record in results_data if record.get("scene", "").startswith(scene_prefix)]

def get_results_by_passname(results_data: list[dict[str, Any]], 
                        pass_name: str) -> list[dict[str, Any]]:
    """
    Retrieve all results for a specific pass.

    Args:
        results_data: List of result dictionaries.
        pass_name: Pass name ("Preprocessed" or "Raw").

    Returns:
        List of matching result dictionaries. Empty list if none found.

    Example:
        >>> results = [
        ...     {"model_id": "Timer-3.0", "scene": "S0-Clean[Preprocessed]", "pass_name": "Preprocessed", "mae": 0.5},
        ...     {"model_id": "Timer-3.5", "scene": "S1-Missing5%[Raw]", "pass_name": "Raw", "mae": None}
        ... ]
        >>> get_results_by_passname(results, "Preprocessed")
        [{'model_id': 'Timer-3.0', 'scene': 'S0-Clean[Preprocessed]', 'pass_name': 'Preprocessed', 'mae': 0.5}]
    """
    return [record for record in results_data if record.get("pass_name") == pass_name]


def _validate_result_format(result: dict) -> None:
    """Validate that the result contains required fields."""
    required_fields = ["test_name", "timestamp"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Result missing required field: {field}")

def _normalize_result(result: dict) -> dict:
    """Normalize result format (business logic)."""
    from neuraxis_testkit.utils.data_sanitizer import clean_nan_values
    return clean_nan_values(result)

