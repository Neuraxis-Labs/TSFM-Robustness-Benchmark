#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.utils.assertions - Generic Assertions Library (Pure Logic)

Usage Examples:
    from neuraxis_testkit.utils.assertions import (
        assert_almost_equal,
        assert_no_nan_inf,
        assert_in_range,
        assert_length,
    )

    assert_almost_equal(actual=0.3, expected=0.1 + 0.2, tolerance=1e-9, label="score")
    assert_no_nan_inf([1.0, 2.0, 3.0], label="predictions")
    assert_in_range(0.12, low=0.0, high=1.0, label="MAE")
    assert_length([1, 2, 3], min_length=1, max_length=5, label="samples")
"""

import math
from typing import Sequence
from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)


def assert_almost_equal(
    actual: float,
    expected: float,
    tolerance: float = 1e-6,
    label: str = "",
) -> None:
    """
    Assert floating-point approximate equality.
    """
    label = f"[{label}] " if label else ""
    diff = abs(actual - expected)

    if diff > tolerance:
        logger.error(
            "%sassert_almost_equal failed: actual=%s, expected=%s, diff=%s, tolerance=%s",
            label,
            actual,
            expected,
            diff,
            tolerance,
        )

    assert diff <= tolerance, (
        f"{label}Floating-point approximate assertion failed: "
        f"actual={actual}, expected={expected}, diff={diff}, tolerance={tolerance}"
    )


def assert_no_nan_inf(values: Sequence[float], label: str = "") -> None:
    """
    Assert that a numeric sequence contains no NaN / Inf values.
    """
    label = f"[{label}] " if label else ""
    for i, v in enumerate(values):
        if math.isnan(v):
            logger.error("%sassert_no_nan_inf failed: index=%d, value=NaN", label, i)
        if math.isinf(v):
            logger.error("%sassert_no_nan_inf failed: index=%d, value=%s", label, i, v)

        assert not math.isnan(v), f"{label}Value at index {i} is NaN"
        assert not math.isinf(v), f"{label}Value at index {i} is Inf"


def assert_in_range(
    value: float,
    low: float | None = None,
    high: float | None = None,
    label: str = "",
) -> None:
    """
    Generic numeric range assertion.
    """
    label = f"[{label}] " if label else ""

    if low is not None and value < low:
        logger.error(
            "%sassert_in_range failed: value=%s < low=%s",
            label,
            value,
            low,
        )
    if high is not None and value > high:
        logger.error(
            "%sassert_in_range failed: value=%s > high=%s",
            label,
            value,
            high,
        )

    if low is not None:
        assert value >= low, f"{label}Value {value} is below lower bound {low}"
    if high is not None:
        assert value <= high, f"{label}Value {value} exceeds upper bound {high}"


def assert_length(
    obj: Sequence,
    min_length: int = 0,
    max_length: int | None = None,
    label: str = "",
) -> None:
    """
    Generic length assertion.
    """
    label = f"[{label}] " if label else ""
    n = len(obj)

    if n < min_length:
        logger.error(
            "%sassert_length failed: length=%s < min_length=%s",
            label,
            n,
            min_length,
        )
    if max_length is not None and n > max_length:
        logger.error(
            "%sassert_length failed: length=%s > max_length=%s",
            label,
            n,
            max_length,
        )

    assert n >= min_length, f"{label}Length should be >= {min_length}, actual: {n}"
    if max_length is not None:
        assert n <= max_length, f"{label}Length should be <= {max_length}, actual: {n}"

