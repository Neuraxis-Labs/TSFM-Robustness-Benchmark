#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.utils.data_sanitizer - Data Sanitization & Type Safety Utilities

Module Purpose:
    Provides atomic, stateless data sanitization functions.
    Ensures data integrity and format compatibility (especially for JSON/CSV serialization).

Core Features:
    - Special Value Handling: Clean/replace NaN, Inf values to ensure JSON compatibility.
    - Safe Conversion: Robust type casting (safe_float, safe_int) with default values.
    - JSON Parsing: Load JSON strings containing non-standard literals (NaN, Infinity).

Module Position in Architecture:
    - Position: Base utility layer (utils)
    - Dependencies: json, math (standard libraries only)
    - Called by: core.* (business logic layer)

Usage Examples:
--------------
>>> from neuraxis_testkit.utils.data_sanitizer import clean_nan_values, safe_float
>>> 
>>> # Clean data before saving to JSON
>>> data = {"mae": 0.5, "rmse": float('nan')}
>>> clean_data = clean_nan_values(data)
>>> # Result: {"mae": 0.5, "rmse": None}  (JSON serializable)
>>> 
>>> # Safe conversion
>>> val = safe_float("N/A", default=0.0)  # Returns 0.0 instead of raising error

Author: Janesong
Create Date: 2026/08/17
"""

import json, re
import numpy as np
from typing import Any

def clean_nan_values(obj: Any, 
                      _depth: int = 0, 
                      _max_depth: int = 100) -> Any:
    """
    Recursively clean NaN/Inf values, converting them to None (JSON-compatible).

    Note: Circualr reference detection is removed since JSON/CSV data sources are tree-structured and cannot contain cycles.

    Args:
        obj: Object to be processed (dict, list, or other types)
        Safety Measures:
            _depth: Current recursion depth (internal use)
            _max_depth: Tracks the IDs of visited objects to prevent infinite recursion caused by circular references (internal use)

    Returns:
        Cleaned object with NaN/Inf replaced by None.

    Example:
        >>> import numpy as np
        >>> data = {"mae": np.nan, "rmse": 0.5, "tags": [np.inf, 1.0]}
        >>> clean_nan_values(data)
        {'mae': None, 'rmse': 0.5, 'tags': [None, 1.0]}
    """
    # Depth protection
    if _depth > _max_depth:
        # print(f"clean_nan_values:
        print(f"clean_nan_values: Recursion depth exceeded {_max_depth}, skipping remaining cleaning.")
        return obj

    if isinstance(obj, dict):
        return {k: clean_nan_values(v, _depth + 1, _max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_values(item, _depth + 1, _max_depth) for item in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        # Handle numpy numeric types
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj

def load_json_with_nan(json_str: str) -> Any:
    """
    Safely parses JSON strings containing NaN / Infinity / -Infinity.

    Replaces non-standard JSON values with 'null' before parsing.

    Args:
        json_str: JSON string potentially containing NaN/Infinity values.

    Returns:
        Parsed Python object (dict, list, or primitive type).

    Raises:
        TypeError: If input is not a string.
        json.JSONDecodeError: If the sanitized string is not valid JSON.

    Note:
        The regex uses negative lookbehind/lookahead to ensure only
        standalone tokens are matched. For example, 'NaN' in "myNaN" will NOT be replaced.

    Example:
        >>> load_json_with_nan('{"value": NaN, "neg_inf": -Infinity, "text": "myNaN"}')
        {'value': None, 'neg_inf': None, 'text': 'myNaN'}
    """
    if not isinstance(json_str, str):
        raise TypeError(f"Expected str type, received {type(json_str).__name__}")

    # Replace non-standard JSON values using regex with negative lookbehind/lookahead
    # Handles: NaN, Infinity, -Infinity
    # Does NOT replace tokens inside strings like "myNaN"
    sanitized = re.sub(
        r'(?<![.\w])(NaN|-?Infinity)(?![.\w])',
        'null',
        json_str
    )

    return json.loads(sanitized)

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely converts a value to float, handling NaN/None values.

    Args:
        value: The value to convert.
        default: The default value to return if conversion fails.

    Returns:
        The converted floating-point number.

    Example:
        >>> safe_float(np.nan)
        0.0
        >>> safe_float(None, default=-1.0)
        -1.0
        >>> safe_float(3.14)
        3.14
    """
    if value is None:
        return default
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely converts a value to int, handling NaN/None values.

    Args:
        value: The value to convert.
        default: The default value to return if conversion fails.

    Returns:
        The converted integer.

    Example:
        >>> safe_int(42)
        42
        >>> safe_int(None, default=-1)
        -1
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

