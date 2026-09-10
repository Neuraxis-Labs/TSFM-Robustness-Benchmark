#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
input_length_test.py -- input_length Ablation Test
====================================
Test Purpose: Evaluate the responsiveness of the TimechoAI prediction API under varying historical input lengths.

Design Goal: Verify the impact of different historical lengths (96/192/256/384/512) on model prediction accuracy.
              Provide historical data of a specified length and return the expected number of prediction points (output_length).

Test Principle: Vary the input_length (96/192/256/384/512) and compare changes in MAE/RMSE.

Author: Janesong
Create Date: 2026/06/29, Updated on 2026/08/14.
"""

import time
import numpy as np
import pandas as pd
from config.settings import RESULTS_DIR
from config.constants import FORECAST_POINT_LEN_64
from core.timecho import forecast
from core.metrics import calc_metrics
from neuraxis_testkit.utils.files import save_to_csv, ensure_dir

# ============================================================
# 0. Data related configuration
# ============================================================
OUTPUT_SUBDIR = RESULTS_DIR / "futureCovs" / "inputLength"
ensure_dir(OUTPUT_SUBDIR)
RESULT_CSV_PATH = OUTPUT_SUBDIR / "input_length_result.csv"    # Prediction results file

# List of historical input lengths under test
INPUT_LENGTHS = [96, 192, 256, 384, 512]
MODELS = ["Timer-3.5", "Chronos-2"]

np.random.seed(42)


# ============================================================
# 1. Generate synthetic data (same source as the dirty-data test to ensure comparability)
# ============================================================
def generate_synthetic_data(total_points: int = 576) -> pd.DataFrame:
    """
    Generate synthetic time-series data (trend + seasonality + noise).

    Args:
        total_points: Total number of data points, default 576 (512 max input + 64 forecast).

    Returns:
        DataFrame with columns: time, target
    """
    dates = pd.date_range("2026-08-14", periods=total_points, freq="1h")
    trend = np.linspace(50, 80, total_points)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(total_points) / 24)
    noise = np.random.randn(total_points) * 2
    target = trend + seasonal + noise

    return pd.DataFrame({"time": dates, "target": target.round(4)})


# ============================================================
# 2. Main Test Flow
# ============================================================
def run_input_length_test() -> list:
    """
    Execute the main logic of the input_length ablation test.

    Returns:
        A list containing all test results.
    """
    # 1. Generate synthetic data
    print(" Generating synthetic data...")
    total_points = max(INPUT_LENGTHS) + FORECAST_POINT_LEN_64  # 512 + 64 = 576
    df = generate_synthetic_data(total_points)

    # Ground truth (last 64 points)
    forecast_len = FORECAST_POINT_LEN_64
    ground_truth = df.iloc[-forecast_len:]["target"].values

    # 2. Iterate over models and input lengths
    total_calls = len(MODELS) * len(INPUT_LENGTHS)
    print(f" Ablation test: {len(MODELS)} models * {len(INPUT_LENGTHS)} lengths = {total_calls} calls")
    print("=" * 80)

    all_results = []

    for model_id in MODELS:
        print(f"\n Model: {model_id}")

        for in_len in INPUT_LENGTHS:
            # Slice historical data: take the first in_len rows of the last (in_len + forecast_len) rows
            history = df.iloc[-(in_len + forecast_len):-forecast_len][["time", "target"]].copy()

            t0 = time.perf_counter()
            try:
                # Call API through core/timecho.py wrapper
                pred_values, elapsed_ms, error = forecast(
                    targets=history,
                    model_id=model_id,
                    output_length=forecast_len,
                    time_col="time",
                    auto_adapt=True,
                )

                if error:
                    print(f"  [{model_id}] input={in_len:>3d} | Failed: {str(error)[:80]}")
                    all_results.append({
                        "model_id": model_id, "input_length": in_len,
                        "mae": None, "rmse": None, "mape": None, "latency_ms": elapsed_ms,
                        "success": False, "error": str(error)
                    })
                else:
                    metrics = calc_metrics(pred_values, ground_truth)

                    print(f"  [{model_id}] | input={in_len:>3d} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | MAPE={metrics['MAPE']:.4f} | Latency={elapsed_ms:.0f}ms")
                    all_results.append({
                        "model_id": model_id, "input_length": in_len,
                        "mae": metrics["MAE"], "rmse": metrics["RMSE"], "mape": metrics["MAPE"], "latency_ms": elapsed_ms,
                        "success": True, "error": None
                    })
            except Exception as exp:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(f"  [{model_id}] input={in_len:>3d} | Exception: {str(exp)[:80]}")
                all_results.append({
                    "model_id": model_id, "input_length": in_len,
                    "mae": None, "rmse": None, "mape": None, "latency_ms": elapsed_ms,
                    "success": False, "error": str(exp)
                })

            time.sleep(1)

    return all_results


def print_summary(all_results: list) -> None:
    """Print the summary of results."""
    print("\n" + "=" * 80)
    print(" Ablation Test Summary")
    print("=" * 80)
    print(f"  {'Model':>12s} | {'input_len':>9s} | {'MAE':>10s} | {'RMSE':>10s} | {'MAPE':>10s} | {'Latency(ms)':>12s}")
    print(f"  {'─'*12}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*12}")

    for r in all_results:
        if r["success"]:
            print(f"  {r['model_id']:>12s} | {r['input_length']:>9d} | {r['mae']:>10.4f} | {r['rmse']:>10.4f} | {r['mape']:>10.4f} | {r['latency_ms']:>12.0f}")
        else:
            print(f"  {r['model_id']:>12s} | {r['input_length']:>9d} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>12s}")


def main():
    """Main entry function."""
    # Run the test
    all_results = run_input_length_test()

    # Print summary
    print_summary(all_results)

    # Save Results
    print("=" * 80)
    csv_path = save_to_csv(RESULT_CSV_PATH, all_results)
    print(f"\nDetailed results saved to CSV: {csv_path}")
    print(" Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
