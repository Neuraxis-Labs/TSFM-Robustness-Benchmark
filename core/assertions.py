#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.assertions - Time Series Foundation Model Business Assertions
                  (Prediction Validity / Metric Thresholds / API Graceful Failure)

Dependency direction: core -> neuraxis_testkit.utils.assertions
(only depends downward on generic capabilities)

Provides:
  - assert_prediction_valid(): prediction result assertion
  - assert_graceful_failure(): graceful failure assertion (API errors are acceptable)
"""

from neuraxis_testkit.utils import assertions as generic
from neuraxis_testkit.log import get_logger

logger = get_logger("core.assertions")

# Default acceptable exception types for graceful failure —
GRACEFUL_EXCEPTION_TYPES = (
    ValueError, TypeError, KeyError, IndexError,
    ConnectionError, TimeoutError, RuntimeError,
    OverflowError, ZeroDivisionError,
)


def assert_prediction_valid(prediction: list | None,
                            min_length: int = 1,
                            max_length: int | None = None) -> None:
    """
    Assert that the prediction result is valid
    (wraps the generic length assertion with business semantics).
    """
    assert prediction is not None, "Prediction result must not be None"
    assert isinstance(prediction, list), \
        (f"Prediction result must be a list, got: {type(prediction).__name__}")
    generic.assert_length(prediction, min_length, max_length, label="prediction")


def assert_metrics_within_range(metrics: dict,
                                max_mae: float = float('inf'),
                                max_rmse: float = float('inf'),
                                max_mape: float = float('inf')) -> None:
    """
    Assert that evaluation metrics are within acceptable ranges.
    """
    generic.assert_in_range(metrics.get("mae", 0), high=max_mae, label="MAE ")
    generic.assert_in_range(metrics.get("rmse", 0), high=max_rmse, label="RMSE ")
    generic.assert_in_range(metrics.get("mape", 0), high=max_mape, label="MAPE ")


def assert_graceful_failure(exception: Exception,
                            allowed_types: tuple | None = None) -> None:
    """
    Assert that the API fails gracefully on invalid input rather than crashing.
    """
    allowed = allowed_types or GRACEFUL_EXCEPTION_TYPES
    assert isinstance(exception, allowed), (
        f"API raised an unexpected exception type: "
        f"{type(exception).__name__}: {exception}. "
        f"Expected one of: {[t.__name__ for t in allowed]}")

    logger.info(
        f"Graceful failure (acceptable): "
        f"{type(exception).__name__}: {exception}")

