#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
irregular_sampling_test.py -- Irregular Sampling Robustness Test
Scenario A: Variable Sampling Rate and Irregular Timestamp Test
====================================
Industrial Context:
  On-site industrial sensors do not sample at equal intervals. When aggregated 
  by gateways and written to time-series databases, the following occurs:
    - Timestamp jitter (clock drift)
    - Out-of-order data arrival (network retransmission)
    - Uneven sampling intervals (multi-source asynchronous aggregation)
    - Periodic packet loss and delayed retransmission

Test Objective:
  Verify that the SDK correctly handles the timestamp semantics of 'time_col', 
  rather than simply processing data based on row index order.
    Core Hypothesis:
    - If the SDK ignores timestamps and processes solely by row order, prediction 
      results across different timestamp scenarios should be identical.
    - If the SDK correctly interprets timestamps, variations in timestamps should 
      lead to differences in prediction results.

Test Methodology:
    1. Keep the target value sequence completely consistent.
    2. Modify only the timestamp column to construct 4 irregular scenarios.
    3. Compare the variation in prediction accuracy across scenarios.
    4. Determine if the SDK utilizes timestamp semantics based on MAE differences.

Author: Janesong
Create Date: 2026/07/12, Updated on 2026/08/14.
"""

import time
import numpy as np
import pandas as pd
from typing import Any

from config.settings import RESULTS_DIR
from config.constants import FORECAST_POINT_LEN_64, FORECAST_POINT_LEN_256
from core.timecho import forecast
from core.metrics import calc_metrics
from neuraxis_testkit.utils.files import save_to_csv, ensure_dir

# ============================================================
# 0. Configuration Constants
# ============================================================
OUTPUT_SUBDIR = RESULTS_DIR / "futureCovs" / "irregularSampling"
ensure_dir(OUTPUT_SUBDIR)
RESULT_CSV_PATH = OUTPUT_SUBDIR / "irregular_sampling_result.csv"    # Prediction results file

HISTORY_LEN = FORECAST_POINT_LEN_256     # 256 historical points for input
FORECAST_LEN = FORECAST_POINT_LEN_64     # 64 future points to predict
TOTAL_LEN = HISTORY_LEN + FORECAST_LEN   # Total sequence length

MODELS = ["Timer-3.5", "Chronos-2"]
SCENARIOS = [
    "A1-Baseline(Equidistant)",
    "A2-MinorJitter(5%)",
    "A3-ModerateDrift(20%)",
    "A4-SevereDisorder",
]


# ============================================================
# 1. Signal Generation Functions
# ============================================================
def _generate_base_signal(seed: int = 42) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """
    Generate base signal with trend, seasonality, and noise.

    All test scenarios use the SAME base signal values to ensure
    any prediction difference comes ONLY from timestamp variation.

    Args:
        seed: Random seed for reproducibility

    Returns:
        target_values: Array of target values (shape: TOTAL_LEN)
        ideal_dates: Ideal equispaced timestamps (1h interval)
        ground_truth: Last FORECAST_LEN values as ground truth
    """
    rng = np.random.RandomState(seed)

    # Ideal equispaced timestamps (1 hour interval)
    ideal_dates = pd.date_range("2026-08-01", periods=TOTAL_LEN, freq="1h")

    # Base signal components: trend + seasonality + noise
    trend = np.linspace(50, 80, TOTAL_LEN)  # Linear trend
    seasonal = 15 * np.sin(2 * np.pi * np.arange(TOTAL_LEN) / 24)  # 24h period
    noise = rng.randn(TOTAL_LEN) * 2  # Gaussian noise

    target_values = (trend + seasonal + noise).round(4)
    ground_truth = target_values[-FORECAST_LEN:]

    return target_values, ideal_dates, ground_truth


# ============================================================
# 2. Timestamp Generation Functions
# ============================================================
def _make_timestamps(
    mode: str,
    ideal_dates: pd.DatetimeIndex,
    total_len: int,
    seed: int = 42
) -> pd.DatetimeIndex:
    """
    Construct timestamps with different irregularity levels.

    Crucial: Only timestamps change, target values remain identical.

    Args:
        mode: Scenario name indicating irregularity level
        ideal_dates: Ideal equispaced timestamps
        total_len: Total number of timestamps
        seed: Random seed for perturbation

    Returns:
        DatetimeIndex with irregular timestamps according to mode
    """
    rng = np.random.RandomState(seed)

    if mode == "A1-Baseline(Equidistant)":
        # Scenario A1: Strictly equispaced 1h interval (baseline)
        # Corresponds to: Ideal lab acquisition environment
        return ideal_dates.copy()

    elif mode == "A2-MinorJitter(5%)":
        # Scenario A2: Slight jitter with ±5% interval variation
        # Interval: 1h ± 180s (±5%)
        # Corresponds to: Gateway clock drift
        deltas_seconds = 3600 + rng.randn(total_len - 1) * 180
        deltas_seconds = np.maximum(deltas_seconds, 60)   # Minimum 1 minute
        ts = [ideal_dates[0]]
        for d in deltas_seconds:
            ts.append(ts[-1] + pd.Timedelta(seconds=int(d)))
        return pd.DatetimeIndex(ts)

    elif mode == "A3-ModerateDrift(20%)":
        # Scenario A3: Moderate drift with ±20% variation + occasional gaps
        # Interval: 1h ± 720s (±20%), 10% points have 2-3x gap
        # Corresponds to: Multi-source async aggregation, packet loss + retransmission
        deltas_seconds = 3600 + rng.randn(total_len - 1) * 720
        deltas_seconds = np.maximum(deltas_seconds, 60)

        # Randomly enlarge 10% of intervals by 2-3x
        mask = rng.rand(len(deltas_seconds)) < 0.1
        deltas_seconds[mask] *= rng.choice([2, 3], size=mask.sum())

        ts = [ideal_dates[0]]
        for d in deltas_seconds:
            ts.append(ts[-1] + pd.Timedelta(seconds=int(d)))
        return pd.DatetimeIndex(ts)

    elif mode == "A4-SevereDisorder":
        # Scenario A4: Severe disorder with non-monotonic timestamps
        # Based on A3, then randomly shuffle 10% of timestamps
        # Corresponds to: Network retransmission, out-of-order arrival
        ts_list = _make_timestamps("A3-ModerateDrift(20%)", ideal_dates, total_len, seed).tolist()
        n_shuffle = len(ts_list) // 10
        indices = rng.choice(len(ts_list), size=n_shuffle, replace=False)

        shuffled_vals = [ts_list[i] for i in indices]
        rng.shuffle(shuffled_vals)
        for i, idx in enumerate(indices):
            ts_list[idx] = shuffled_vals[i]
        return pd.DatetimeIndex(ts_list)

    else:
        raise ValueError(f"Unknown scenario: {mode}")

def _make_test_result(
    scenario: str,
    model_id: str,
    pred_values: np.ndarray | None,
    ground_truth: np.ndarray,
    elapsed_ms: float,
    success: bool,
    error: str | None
) -> dict[str, Any]:
    """
    Create a standardized test result dictionary.

    Args:
        scenario: Test scenario name
        model_id: Model identifier
        pred_values: Prediction values (None if failed)
        ground_truth: Ground truth values
        elapsed_ms: Execution time in milliseconds
        success: Whether prediction succeeded
        error: Error message (None if succeeded)

    Returns:
        Standardized result dictionary
    """
    # Calculate metrics (automatically handles None)
    metrics = calc_metrics(pred_values, ground_truth)

    return {
        "scenario": scenario,
        "model_id": model_id,
        "mae": metrics["MAE"],
        "rmse": metrics["RMSE"],
        "mape": metrics["MAPE"],
        "latency_ms": elapsed_ms,
        "success": success,
        "error": error,
    }


# ============================================================
# 3. Timestamp Usage Analysis
# ============================================================
def _analyze_timestamp_usage(results: list[dict]) -> dict[str, Any]:
    """
    Analyze whether SDK utilized timestamp semantics based on metric variance.

    Core logic: If all scenarios have identical MAE (variance < threshold),
    SDK likely ignores timestamps and processes by row order only.

    Args:
        results: List of test results

    Returns:
        analysis: Dictionary containing analysis results and conclusions
    """
    analysis = {
        "model_analysis": {},
        "overall_conclusion": ""
    }

    for model_id in MODELS:
        model_results = [r for r in results if r["model_id"] == model_id and r["success"]]

        if len(model_results) < 2:
            analysis["model_analysis"][model_id] = {
                "status": "insufficient_data",
                "message": "Insufficient successful scenarios for analysis"
            }
            continue

        # Get baseline MAE (A1 scenario or first successful)
        baseline_mae = None
        for r in model_results:
            if "A1" in r["scenario"]:
                baseline_mae = r["mae"]
                break
        if baseline_mae is None:
            baseline_mae = model_results[0]["mae"]

        # Calculate MAE variance across scenarios
        mae_values = [r["mae"] for r in model_results]
        mae_variance = np.var(mae_values)

        # A relative change of 1% is considered significant timestamp usage
        RELATIVE_THRESHOLD = 0.01

        # Prevent division by zero if baseline_mae is 0
        if baseline_mae == 0:
            max_relative_diff = 0.0
        else:
            max_relative_diff = max(abs(mae - baseline_mae) / baseline_mae for mae in mae_values)

        timestamps_utilized = max_relative_diff > RELATIVE_THRESHOLD

        analysis["model_analysis"][model_id] = {
            "baseline_mae": baseline_mae,
            "mae_values": {r["scenario"]: r["mae"] for r in model_results},
            "mae_variance": mae_variance,
            "max_relative_diff": max_relative_diff,
            "threshold_used": f"{RELATIVE_THRESHOLD*100}%",
            "timestamps_utilized": timestamps_utilized,
            "conclusion": "Timestamp semantics utilized" if timestamps_utilized else "SDK might ignore timestamps"
        }

    # Overall conclusion
    all_utilized = all(
        analysis["model_analysis"].get(m, {}).get("timestamps_utilized", False)
        for m in MODELS
    )

    if all_utilized:
        analysis["overall_conclusion"] = "PASS: SDK correctly understands and utilizes timestamp semantics"
    else:
        analysis["overall_conclusion"] = "WARNING: SDK might ignore time_col and process by row order only"

    return analysis


# ============================================================
# 4. Main Test Function
# ============================================================
def run_irregular_sampling_test(
    models: list[str] | None = None,
    scenarios: list[str] | None = None,
    verbose: bool = True
) -> tuple[list[dict], dict[str, Any]]:
    """
    Execute irregular sampling robustness test.

    Args:
        models: List of model IDs to test (default: MODELS)
        scenarios: List of scenario names (default: SCENARIOS)
        verbose: Whether to print progress info

    Returns:
        results: List of test result dictionaries
            Each dict contains: scenario, model_id, mae, rmse, latency_ms, success, error
        details: Dictionary containing analysis and metadata
            Contains: analysis, config, summary
    """
    # Use default parameters if not provided
    models = MODELS if models is None else models
    scenarios = SCENARIOS if scenarios is None else scenarios

    if verbose:
        print("=" * 80)
        print("Scenario A: Variable Sampling Rate & Irregular Timestamp Test")
        print(f"   {len(models)} Models x {len(scenarios)} Scenarios = {len(models) * len(scenarios)} Calls")
        print("=" * 80)

    # Step 1: Generate base signal (identical for all scenarios)
    target_values, ideal_dates, ground_truth = _generate_base_signal(seed=42)

    # Step 2: Execute tests across all scenarios and models
    results = []

    for scenario in scenarios:
        if verbose:
            print(f"\n[Scenario] {scenario}")

        # Generate timestamps for current scenario
        timestamps = _make_timestamps(scenario, ideal_dates, TOTAL_LEN, seed=42)

        # Construct DataFrame (target values unchanged, only timestamps vary)
        df = pd.DataFrame({"time": timestamps, "target": target_values})
        history = df.iloc[:HISTORY_LEN][["time", "target"]].copy()
        assert len(history) == HISTORY_LEN, f"History data length error: {len(history)} != {HISTORY_LEN}"

        # Print timestamp statistics
        if scenario == "A4-SevereDisorder":
            # Log disordered state before sorting for analysis
            is_original_monotonic = history['time'].is_monotonic_increasing
            if verbose:
                print(f"   Original timestamp monotonicity: {is_original_monotonic}")

            # Force ascending sort to meet SDK input requirements
            # After sorting, if SDK ignores timestamps, results will differ from baseline (due to order change)
            history = history.sort_values('time').reset_index(drop=True)
            if verbose:
                print(f"   [Preprocessing] Sorted by timestamp ascending to comply with time-series model input specs")

        # Print timestamp interval statistics (based on sorted history)
        if verbose:
            times = history['time'].values.astype('datetime64[s]').astype(np.int64)
            deltas = np.diff(times)
            if len(deltas) > 0:
                print(f"   Intervals: min={deltas.min()/60:.1f}min  max={deltas.max()/3600:.1f}h  mean={deltas.mean()/3600:.1f}h")
            else:
                print("   Intervals: N/A (Insufficient data points)")

        # Model prediction loop
        for model_id in models:
            t0 = time.perf_counter()

            try:
                pred_values, elapsed_ms, error = forecast(
                    targets=history,
                    model_id=model_id,
                    output_length=FORECAST_LEN,
                    time_col="time",
                    auto_adapt=True,
                )

                # Validate prediction length (only if no error and prediction values exist)
                if error is None and pred_values is not None:
                    if len(pred_values) != FORECAST_LEN:
                        error = f"Prediction length mismatch: Expected {FORECAST_LEN}, Actual {len(pred_values)}"
                        pred_values = None

                if error:
                    result = _make_test_result(
                        scenario, model_id, None, ground_truth,
                        elapsed_ms, False, str(error)
                    )
                    if verbose:
                        print(f"   [{model_id}] Failed: {str(error)[:80]}")
                else:
                    result = _make_test_result(
                        scenario, model_id, pred_values, ground_truth,
                        elapsed_ms, True, None
                    )
                    if verbose:
                        print(f"   [{model_id}] Success MAE={result['mae']:.4f}  RMSE={result['rmse']:.4f}  MAPE={result['mape']:.2f}%  Latency={elapsed_ms:.0f}ms")

                results.append(result)
            except Exception as exp:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                result = _make_test_result(
                    scenario, model_id, None, ground_truth,
                    elapsed_ms, False, str(exp)
                )
                results.append(result)

                if verbose:
                    print(f"   [{model_id}] Exception: {str(exp)[:80]}")

            time.sleep(1)

    # Step 3: Analyze timestamp usage
    analysis = _analyze_timestamp_usage(results)

    # Step 4: Prepare details
    details = {
        "analysis": analysis,
        "config": {
            "history_len": HISTORY_LEN,
            "forecast_len": FORECAST_LEN,
            "total_len": TOTAL_LEN,
            "models": models,
            "scenarios": scenarios,
        },
        "summary": {
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r["success"]),
            "failed_tests": sum(1 for r in results if not r["success"]),
        }
    }

    return results, details


# ============================================================
# 5. Results Reporting
# ============================================================
def _print_results_summary(results: list[dict], analysis: dict[str, Any]) -> None:
    """
    Print formatted summary of test results and analysis.

    Args:
        results: List of test results
        analysis: Analysis results from _analyze_timestamp_usage
    """
    print("\n" + "=" * 100)
    print("Test Results Summary")
    print("=" * 100)

    print(f"\n{'Scenario':>22s} | {'Model':>12s} | {'MAE':>10s} | {'RMSE':>10s} | {'MAPE':>10s} | {'Latency(ms)':>12s} | Status")
    print("-" * 100)

    for r in results:
        if r["success"]:
            # Success: Display all metrics
            print(f"{r['scenario']:>22s} | {r['model_id']:>12s} | "
                  f"{r['mae']:>10.4f} | {r['rmse']:>10.4f} | {r['mape']:>9.2f}% | "
                  f"{r['latency_ms']:>12.0f} | Success")
        else:
            # Failed: Display N/A
            error_msg = r['error'][:20] if r.get('error') else 'Unknown'
            print(f"{r['scenario']:>22s} | {r['model_id']:>12s} | "
                  f"{'N/A':>10s} | {'N/A':>10s} | {'N/A':>9s} | "
                  f"{'N/A':>12s} | Failed {error_msg}")

    # Print timestamp usage analysis
    print("\n" + "=" * 80)
    print("Core Analysis: Does the SDK Understand Timestamp Semantics?")
    print("=" * 80)

    for model_id, model_analysis in analysis["model_analysis"].items():
        print(f"\n  [{model_id}]")

        if model_analysis.get("status") == "insufficient_data":
            print(f"     Insufficient successful scenarios, unable to analyze")
            continue

        baseline_mae = model_analysis["baseline_mae"]
        mae_values = model_analysis["mae_values"]

        for scenario, mae in mae_values.items():
            if baseline_mae > 0:
                ratio = mae / baseline_mae
                print(f"     {scenario:>22s}: MAE={mae:.4f} ({ratio:.2f}x of Baseline)")
            else:
                print(f"     {scenario:>22s}: MAE={mae:.4f} (Baseline MAE is 0)")

        conclusion = model_analysis["conclusion"]
        if "ignore" in conclusion:
            print(f"     [Warning] {conclusion} -> SDK might ignore time_col and process by row index only")
        else:
            print(f"     [Pass] {conclusion}")

    print(f"\nOverall Conclusion: {analysis['overall_conclusion']}")


# ============================================================
# 6. Main Entry Point
# ============================================================
def main():
    """
    Main entry point for irregular sampling robustness test.

    This function provides unified interface for running the test,
    saving results, and printing analysis.
    """
    print("Starting Irregular Sampling Robustness Test...")
    print("=" * 80)

    # Execute test
    results, details = run_irregular_sampling_test(
        models=MODELS,
        scenarios=SCENARIOS,
        verbose=True
    )

    # Print summary
    _print_results_summary(results, details["analysis"])

    # Save results to CSV
    print("=" * 80)
    csv_path = save_to_csv(RESULT_CSV_PATH, results)
    print(f"\nDetailed results saved to CSV: {csv_path}")

    # Print summary statistics
    summary = details["summary"]
    print(f"\nTest Statistics: Total {summary['total_tests']} runs, Success {summary['successful_tests']}, Failed {summary['failed_tests']}")
    print("=" * 80)
    print(" Test completed!")
    print("=" * 80)

    return results, details

if __name__ == "__main__":
    main()

