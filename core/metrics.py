#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.metrics — Evaluation Metrics Calculator

Provides standard evaluation metrics for time series forecasting models.

Functions:
  - calc_metrics(): Compute MAE, RMSE, MAPE
  - calc_diff(): Compute mean absolute difference between predictions
  - evaluate_prediction(): Evaluate predictions with full and segmented metrics for multi-step forecasting analysis

Author: Janesong
Create Date: 2026/08/17.
"""

import numpy as np

def calc_metrics(
    predictions: np.ndarray | None,
    ground_truth: np.ndarray,
    threshold: float = 1e-7   # Threshold for valid MAPE calculation
) -> dict[str, float | None]:
    """
    Calculate MAE / RMSE / MAPE.

    Args:
        predictions: Prediction array, None indicates prediction failure
        ground_truth: Ground truth array
        threshold: absolute value for valid MAPE denominator

    Returns:
        dict: {"MAE": ..., "RMSE": ..., "MAPE": ...}
        MAE: Mean Absolute Error
        RMSE: Root Mean Squared Error
        MAPE: Mean Absolute Percentage Error (%)
        Returns None if all ground truth values are near zero
    """
    if predictions is None:
        return {"MAE": None, "RMSE": None, "MAPE": None}

    # Convert to numpy arrays (zero-copy if already ndarray with matching dtype)
    predictions = np.asarray(predictions, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Prediction and ground truth length mismatch: {len(predictions)} vs {len(ground_truth)}"
        )
    if len(predictions) == 0:
        raise ValueError("Cannot evaluate empty arrays")

    # 1. Calculate MAE and RMSE on the full dataset
    mae = float(np.mean(np.abs(predictions - ground_truth)))
    rmse = float(np.sqrt(np.mean((predictions - ground_truth) ** 2)))

    # 2. MAPE: Calculate only on data points with valid denominators (mask out near-zero/zero values)
    mask = np.abs(ground_truth) > threshold
    if np.any(mask):
        # Calculate MAPE only for valid points
        mape = float(np.mean(
            np.abs((predictions[mask] - ground_truth[mask]) / ground_truth[mask])
        ) * 100)
    else:
        # All ground truth values are close to 0; MAPE is undefined
        mape = None

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def calc_diff(pred1: np.ndarray | None, pred2: np.ndarray | None) -> float | None:
    """
    Calculate the mean absolute difference between two prediction arrays.

    Args:
        pred1: First prediction array
        pred2: Second prediction array

    Returns:
        Mean absolute difference; returns None if either input is None.

    Raises:
        ValueError: If arrays have different lengths or are empty.
    """
    if pred1 is None or pred2 is None:
        return None

    pred1 = np.asarray(pred1, dtype=np.float64)
    pred2 = np.asarray(pred2, dtype=np.float64)

    if len(pred1) != len(pred2):
        raise ValueError(
            f"Pred1 and pred2 length mismatch: {len(pred1)} vs {len(pred2)}"
        )
    if len(pred1) == 0:
        raise ValueError("Cannot compute difference of empty arrays")

    return float(np.mean(np.abs(pred1 - pred2)))

def evaluate_prediction(
    predictions: np.ndarray | None,
    ground_truth: np.ndarray,
    steps: list[int] | None = None,
    threshold: float = 1e-7,
    ddof: int = 0
) -> dict[str, float | None]:
    """
    Calculate comprehensive prediction metrics with segmented evaluation.

    Evaluates model performance at multiple forecast horizons (e.g., 16, 32, 64 steps)
    to analyze prediction stability over different time scales.

    Args:
        predictions: Prediction array, None indicates prediction failure
        ground_truth: Ground truth array
        steps: Evaluation horizon steps, default [16, 32, 64]
        threshold: Threshold for MAPE calculation
        ddof: Delta degrees of freedom for std calculation, default 0 (population std).
              Use ddof=1 for sample standard deviation.

    Returns:
        dict: Metrics dictionary containing:
            - MAE: Mean Absolute Error on full data
            - RMSE: Root Mean Squared Error
            - MAPE: Mean Absolute Percentage Error (%)
            - MAE_STD: Standard deviation of absolute errors
            - MAX_ERROR: Maximum absolute error
            - MAE_{step}: MAE at each evaluation step (e.g., MAE_16, MAE_32, MAE_64)

    Example:
        >>> predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        >>> ground_truth = np.array([1.1, 2.1, 2.9, 3.2, 3.6, 3.3, 4.0, 4.7, 4.8])
        >>> metrics = evaluate_prediction(predictions, ground_truth, steps=[3, 6, 9])
        >>> print(metrics['MAE_3'], metrics['MAE_6'], metrics['MAE_9'])
    """
    # Default steps (use None default to avoid mutable default parameter)
    if steps is None:
        steps = [16, 32, 64]

    # Handle None case: return default metrics dict
    if predictions is None:
        default_metrics = {"MAE": None, "RMSE": None, "MAPE": None, "MAE_STD": None, "MAX_ERROR": None}
        # Add None values for default steps
        for step in steps:
            default_metrics[f"MAE_{step}"] = None
        return default_metrics

    # Convert to numpy arrays
    predictions = np.asarray(predictions, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    base_metrics = calc_metrics(predictions, ground_truth, threshold)
    abs_errors = np.abs(predictions - ground_truth)    # Calculate errors
    # Build metrics dictionary
    metrics = {
        "MAE": base_metrics["MAE"],
        "RMSE": base_metrics["RMSE"],
        "MAPE": base_metrics["MAPE"],
        "MAE_STD": float(np.std(abs_errors, ddof=ddof)),
        "MAX_ERROR": float(np.max(abs_errors))
    }

    valid_steps = [step for step in steps if step <= len(predictions)]   # Filter valid steps 过滤有效步数
    # Segmented MAE
    for step in valid_steps:
        metrics[f"MAE_{step}"] = float(np.mean(abs_errors[:step]))

    return metrics
