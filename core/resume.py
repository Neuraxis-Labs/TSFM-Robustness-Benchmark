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
import re
from typing import Any

# Rate limit keywords for detection
RATE_LIMIT_PATTERNS = [
    r"\b429\b",                                       # HTTP status code
    r"\brate[\s_-]?limit",                            # "rate limit" / "rate-limit" / "rate_limit"
    r"\bquota\b",                                     # generic quota errors
    r"\binsufficient[\s_]?quota\b",                   # OpenAI: insufficient_quota (NOT covered by \bquota\b: '_'
                                                      # is a word char, so no word boundary before 'quota')
    r"\btoo many requests\b",
    r"\bthrottl",                                     # intentional prefix match: throttled/throttling
    r"\bcapacity\s+(exceeded|reached|limit|full)",
    r"\bresource[\s_]?exhausted\b",                   # Google gRPC: RESOURCE_EXHAUSTED
    r"\bplease\s+(slow down|try again later)\b",      # OpenAI / Anthropic friendly-wording errors
]
_RATE_LIMIT_RE = re.compile("|".join(f"(?:{p})" for p in RATE_LIMIT_PATTERNS), re.IGNORECASE,)
def is_rate_limited(error_msg: str) -> bool:
    """
    Check if error is a rate-limit (429 Too Many Requests) error.

    Detects rate limit errors based on common error messages returned by
    major LLM API providers (OpenAI, Anthropic, Google, etc.).

    Args:
        error_msg: Error message string

    Returns:
        True if rate-limit error, False otherwise

    Example:
        >>> is_rate_limited("Error 429: Too Many Requests")
        True
        >>> is_rate_limited("Rate limit exceeded")
        True
        >>> is_rate_limited("insufficient_quota: billing quota exceeded")
        True
        >>> is_rate_limited("RESOURCE_EXHAUSTED: gRPC call failed")
        True
        >>> is_rate_limited("Model output exceeds max context length")
        False
        >>> is_rate_limited("Connection timeout")
        False
    """
    if not error_msg:
        return False

    return bool(_RATE_LIMIT_RE.search(error_msg))


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

        success_val = str(record.get("success", "")).strip().lower()

        if success_val == "true":
            # Successfully completed
            completed_keys.add(key)
        else:
            # Check if it's a rate limit error
            error_msg = str(record.get("error", ""))
            if not is_rate_limited(error_msg):
                # Permanent failure, should skip
                failed_keys.add(key)
            # Rate limit errors should be retried, not added to failed_keys

    return completed_keys, failed_keys

