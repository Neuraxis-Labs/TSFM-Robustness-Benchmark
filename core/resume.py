#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.resume - Resume from Checkpoint Strategy Controller

Module Purpose:
  Provides checkpoint resumption strategy and rate limit detection.
  Encapsulates decision logic for skipping tests or retrying failed ones.

Core Features:
  - Rate limit detection (429 error identification)
  - Checkpoint resumption strategy
  - Retry decision logic

Usage Examples:
    from core.resume import is_rate_limited, should_skip_test

    # Check if error is rate limit
    if is_rate_limited(error_msg):
        # Handle rate limit (retry later)
        pass

    # Check if should skip test
    if should_skip_test(completed_keys, test_key):
        continue

Author: Janesong
Create Date: 2026/07/10, Updated on 2026/08/17.
"""

from typing import Any, Literal
from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)

# Rate limit keywords for detection
RATE_LIMIT_KEYWORDS = ["429", "limit", "quota", "exceed", "rate", "too many"]
def is_rate_limited(error_msg: str) -> bool:
    """
    Check if error is rate-limit (429 Too Many Requests) error.

    This method detects rate limit errors based on common keywords in error messages returned by API providers.

    Args:
        error_msg: Error message string

    Returns:
        True if rate-limit error, False otherwise

    Example:
        >>> is_rate_limited("Error 429: Too Many Requests")
        True
        >>> is_rate_limited("Rate limit exceeded")
        True
        >>> is_rate_limited("Connection timeout")
        False
    """
    if not error_msg:
        return False

    error_lower = error_msg.lower()
    return any(k in error_lower for k in RATE_LIMIT_KEYWORDS)

RecordStatus = Literal["success", "rate_limited", "failed", "unknown"]
def classify_record(record: dict) -> RecordStatus:
    """
    Classify a result record into one of four categories.

    Categories:
        - "success":      ``success`` field is "true"
        - "rate_limited": ``success`` is "false" and the error is rate-limit related
        - "failed":       ``success`` is "false" and the error is NOT rate-limit related
        - "unknown":      ``success`` field is missing or not a valid boolean string
                          (treated as incomplete; caller should allow rerun)

    Args:
        record: A single result dictionary loaded from CSV.

    Returns:
        One of "success", "rate_limited", "failed", "unknown".
    """
    success_val = str(record.get("success", "")).strip().lower()
    if success_val == "true":
        return "success"
    if success_val == "false":
        error_msg = str(record.get("error", ""))
        return "rate_limited" if is_rate_limited(error_msg) else "failed"
    return "unknown"

def should_skip_test(
    completed_keys: set[tuple[Any, ...]],
    test_key: tuple[Any, ...],
    failed_keys: set[tuple[Any, ...]] | None = None
) -> bool:
    """
    Determine if a test should be skipped (already completed or permanently failed).

    Args:
        completed_keys: Set of already completed test keys
        test_key: Current test key to check
        failed_keys: Set of permanently failed test keys (optional)

    Returns:
        True if test should be skipped, False otherwise

    Example:
        >>> completed = {("model_a", "scene_1"), ("model_b", "scene_2")}
        >>> should_skip_test(completed, ("model_a", "scene_1"))
        True
        >>> should_skip_test(completed, ("model_c", "scene_1"))
        False
    """
    if test_key in completed_keys:
        return True

    if failed_keys and test_key in failed_keys:
        return True

    return False

def build_completed_keys(
    records: list[dict],
    key_columns: list[str]
) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    """
    Build completed and failed test key sets from records.

    This method extracts test keys from historical results for checkpoint resumption.

    Args:
        records: List of result dictionaries (from CSV)
        key_columns: List of column names to build key (e.g., ["model_id", "scene"])

    Returns:
        (completed_keys, failed_keys)
        - completed_keys: Set of successfully completed test keys
        - failed_keys: Set of permanently failed test keys (non-rate-limit)

    Example:
        >>> records = [
        ...     {"model_id": "Timer-3.0", "scene": "S0", "success": "true"},
        ...     {"model_id": "Timer-3.0", "scene": "S1", "success": "false", "error": "timeout"}
        ... ]
        >>> completed, failed = build_completed_keys(records, ["model_id", "scene"])
        >>> completed
        {("Timer-3.0", "S0")}
        >>> failed
        {("Timer-3.0", "S1")}
    """
    completed_keys = set()
    failed_keys = set()

    for record in records:
        # Build key from specified columns
        key = tuple(record.get(col) for col in key_columns)
        status = classify_record(record)

        if status == "success":
            completed_keys.add(key)   # Successfully completed
        elif status == "failed":
            failed_keys.add(key)
        elif status == "rate_limited":
            pass  # Retry later; not added to failed_keys
        else:  # "unknown"
            logger.warning(f"Record missing valid 'success' field, key={key}, will rerun")

    return completed_keys, failed_keys

