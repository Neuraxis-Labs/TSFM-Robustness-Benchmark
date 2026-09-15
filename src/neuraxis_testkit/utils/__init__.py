"""
neuraxis_testkit.utils - Neuraxis TestKit Common Utility Layer

This package provides stateless pure functions and generic entity wrappers, serving as the
underlying support for the entire project. It currently includes:

Modules:
-------
assertions.py - Generic Assertions Library (Pure Logic)
  This module is pure logic: it does not depend on pytest and should not call pytest.fail.
  On assertion failure, key parameter values are logged via logger.error for easier debugging.

concurrent.py - (Internal) Concurrent Utilities
  Provides thread-safe primitives (FileLock, ProcessSafeCache) used internally.

data_sanitizer.py - Data Sanitization & Type Safety
  Handles NaN/Inf values for JSON compatibility and provides robust type conversion.
  Ensures data integrity before persistence or transmission.

files.py - File Operation Utilities
  Provides functionality for reading, writing, appending, and status checking for files (CSV/JSON).
  Unified error handling and path management.

runner.py - Execution Primitives
  Process-level timeout and retry primitives for callables invoked inside test cases.
  pytest owns discovery/execution/reporting/result-tracking; this module is not an entry point.

Usage Conventions:
--------------------------
  Files and data_sanitizer can be used directly by core layers, but testcases should prefer core interfaces.

Usage Examples:
-----------------------------
>>> # Utils layer usage (for core layer developers)
>>> from neuraxis_testkit.assertions import assert_almost_equal, assert_no_nan_inf, assert_in_range, assert_length
>>> from neuraxis_testkit.utils.data_sanitizer import clean_nan_values, safe_float
>>> from neuraxis_testkit.utils.files import save_to_csv, append_to_csv
>>> from neuraxis_testkit.utils.runner import run_with_timeout, run_test_with_retry
"""

__all__ = [
    "assertions",
    "concurrent",
    "data_sanitizer",
    "files",
    "runner",
]
