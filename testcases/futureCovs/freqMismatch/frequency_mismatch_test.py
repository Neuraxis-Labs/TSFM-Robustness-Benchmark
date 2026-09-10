#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
frequency_mismatch_test.py -- Frequency Mismatch Test
====================================
Industrial Scenarios:
  - Signal period changes caused by equipment speed or production changes.
  - Quantifying prediction degradation when the model is not updated timely.
Test Objective:
  Training period is fixed at 24h, while prediction period shifts to 24h/12h/8h/48h
  to quantify performance degradation caused by training-prediction frequency inconsistency.

Test Principle:
  1. Training Data: Fixed 24h period, model learns 24h seasonal pattern.
  2. Prediction Phase:
     - Model predicts based on training results, continuing the 24h period signal.
     - Ground Truth period changes (12h/8h/48h).
  3. Performance Difference: Reflects "prediction degradation due to frequency mismatch".


Input Data Characteristics:
  - Training Segment: 512 points, 1h sampling rate, 24h period sine wave + linear trend + noise
  - Prediction Segment: 64 points, sine waves with different periods (24h/12h/8h/48h)

Output Results:
  - CSV File: 8 records (2 models * 4 modes)
  - Metrics: MAE, MAE_16, MAE_32, MAE_STD, MAX_ERROR

Total calls: 8 times (2 models * 4 modes)

Author: Janesong
Create Date: 2026/07/05, Updated on 2026/08/13.
"""

import numpy as np
import pandas as pd

from config.settings import RESULTS_DIR
from config import constants as CONSTANTS
from core.timecho import forecast
from core.metrics import evaluate_prediction
from neuraxis_testkit.utils.files import save_to_csv, ensure_dir

# ============================================================
# 0.Data related configuration
# ============================================================
OUTPUT_SUBDIR = RESULTS_DIR / "futureCovs" / "freqMismatch"
ensure_dir(OUTPUT_SUBDIR)
RESULT_CSV_PATH = OUTPUT_SUBDIR / "frequency_mismatch_result.csv"    # Prediction results file

FORECAST_LEN = CONSTANTS.FORECAST_POINT_LEN_64  # 64
N_TRAIN = CONSTANTS.TRAIN_SEQ_LEN_512
TRAIN_PERIOD = CONSTANTS.TRAIN_PERIOD_HOUR_24   # Training period (hours)
SEED = 42

np.random.seed(SEED)

# ============================================================
# 1. Data Generation Function
# ============================================================
def generate_frequency_mismatch_data(
    n_train: int,
    forecast_len: int,
    train_period: int,
    eval_periods: list,
    base_value: float = CONSTANTS.SIGNAL_BASE_VALUE_50,
    trend_amp: float = CONSTANTS.SIGNAL_TREND_AMP_15,
    seasonal_amp: float = CONSTANTS.SIGNAL_SEASONAL_AMP_15,
    noise_std: float = CONSTANTS.SIGNAL_NOISE_STD_15,
    seed: int = 42
):
    """
    Generate frequency mismatch test data

    Args:
        n_train: Training data length
        forecast_len: Prediction length
        train_period: Training period (hours)
        eval_periods: List of prediction periods (hours)
        base_value: Base value
        trend_amp: Trend amplitude (total change)
        seasonal_amp: Seasonal amplitude
        noise_std: Noise standard deviation
        seed: Random seed

    Returns:
        df_history: Training data DataFrame
        futures: {mode_name: ground_truth_array}
        mode_configs: {mode_name: configuration info}
        time_info: Time related info dictionary
    """
    np.random.seed(seed)

    # Time series
    time_full = pd.date_range("2026-08-13", periods=n_train + forecast_len, freq="1h")
    time_history = time_full[:n_train]
    time_future = time_full[n_train:]

    # Training segment: Fixed period
    trend_train = np.linspace(base_value, base_value + trend_amp, n_train)
    seasonal_train = seasonal_amp * np.sin(2 * np.pi * np.arange(n_train) / train_period)
    noise_train = np.random.randn(n_train) * noise_std
    history = (trend_train + seasonal_train + noise_train).round(4)

    # Prediction segment: Different periods
    trend_fc = np.linspace(base_value + trend_amp, base_value + 2 * trend_amp, forecast_len)
    noise_fc = np.random.randn(forecast_len) * noise_std

    futures = {}
    mode_configs = {}

    for period in eval_periods:
        # Period description
        if period == train_period:
            mode_name = f"1-Normal({train_period}h->{period}h)"
            desc = "Baseline: Period unchanged"
        elif period < train_period:
            factor = train_period / period
            mode_name = f"2-Speedup{factor:.0f}x({train_period}h->{period}h)"
            desc = f"Speed change: Period becomes 1/{factor:.0f}"
        else:
            factor = period / train_period
            mode_name = f"4-Slowdown{factor:.0f}x({train_period}h->{period}h)"
            desc = f"Slowdown: Period increases {factor:.0f}x"

        # Seasonal component (Key: Continue from training end phase)
        # Note: Includes phase discontinuity to test model robustness
        indices = np.arange(n_train, n_train + forecast_len)
        seasonal_fc = seasonal_amp * np.sin(2 * np.pi * indices / period)

        ground_truth = (trend_fc + seasonal_fc + noise_fc).round(4)
        futures[mode_name] = ground_truth
        mode_configs[mode_name] = {
            "period": period,
            "desc": desc,
            "full_periods": forecast_len / period
        }

    df_history = pd.DataFrame({
        "time": time_history,
        "target": history
    })

    time_info = {
        "time_full": time_full,
        "time_history": time_history,
        "time_future": time_future,
        "train_period": train_period
    }

    return df_history, futures, mode_configs, time_info


# ============================================================
# 2. Main Test Flow 主测试流程
# ============================================================
def run_frequency_mismatch_test():
    """Execute frequency mismatch test"""

    print("=" * 90)
    print("C5 Frequency Mismatch Test")
    print("=" * 90)

    # Test configuration
    eval_periods = [24, 12, 8, 48]  # Prediction periods
    models = ["Chronos-2", "Timer-3.5"]

    # --------------------------------------------------------
    # 2.1 Data Generation
    # --------------------------------------------------------
    print(f"\n[Data Generation]")
    print(f"  Training Length: {N_TRAIN}, Prediction Length: {FORECAST_LEN}")
    print(f"  Training Period: {TRAIN_PERIOD}h (Fixed)")
    print(f"  Prediction Periods: {eval_periods}h")

    df_history, futures, mode_configs, time_info = generate_frequency_mismatch_data(
        n_train=N_TRAIN,
        forecast_len=FORECAST_LEN,
        train_period=TRAIN_PERIOD,
        eval_periods=eval_periods,
        base_value=CONSTANTS.SIGNAL_BASE_VALUE_50,
        trend_amp=CONSTANTS.SIGNAL_TREND_AMP_15,
        seasonal_amp=CONSTANTS.SIGNAL_SEASONAL_AMP_15,
        noise_std=CONSTANTS.SIGNAL_NOISE_STD_15,
        seed=SEED
    )

    # Data statistics
    print(f"\n  Training Data Statistics:")
    print(f"    - Mean: {df_history['target'].mean():.2f}")
    print(f"    - Std:  {df_history['target'].std():.2f}")
    print(f"    - Range: [{df_history['target'].min():.2f}, {df_history['target'].max():.2f}]")

    # --------------------------------------------------------
    # 2.2 Execute Prediction
    # --------------------------------------------------------
    print(f"\n[Model Prediction]")

    all_results = []
    result_details = []     # Detailed results for analysis

    for model_id in models:
        print(f"\n  Model: {model_id}")
        print("-" * 70)

        for mode_name in futures.keys():
            gt = futures[mode_name]
            cfg = mode_configs[mode_name]

            try:
                # Call API through core/timecho.py wrapper
                pred, _, _ = forecast(
                    targets=df_history,
                    model_id=model_id,
                    output_length=FORECAST_LEN,
                    time_col="time"
                )

                # Calculate error metrics
                metrics = evaluate_prediction(pred, gt, steps=[16, 32, 64])
                print(f"    {mode_name:<25s}  MAE={metrics['MAE']:.4f}  "
                      f"(First 16 steps: {metrics['MAE_16']:.4f}, First 32 steps: {metrics['MAE_32']:.4f})")

                # Record results
                result_row = [
                    model_id,
                    mode_name,
                    cfg["period"],
                    cfg["desc"],
                    metrics["MAE"],
                    metrics.get("MAE_16"),
                    metrics.get("MAE_32"),
                    metrics["MAE_STD"],
                    metrics["MAX_ERROR"],
                    metrics["RMSE"]
                ]
                all_results.append(result_row)

                # Detailed record (for analysis)
                result_details.append({
                    "model": model_id,
                    "mode": mode_name,
                    "period": cfg["period"],
                    "pred": pred,
                    "gt": gt,
                    "metrics": metrics
                })

            except Exception as exp:
                print(f"    {mode_name:<25s}  Failed: {type(exp).__name__}: {str(exp)}")
                all_results.append([model_id, mode_name, cfg["period"], cfg["desc"]] + [None] * 6)


    # --------------------------------------------------------
    # 2.3 Result Summary
    # --------------------------------------------------------
    print(f"\n{'='*90}")
    print("C5 Frequency Mismatch - Summary Results")
    print(f"{'='*90}")

    # Table header
    header = f"{'Model':<12s} | {'Mode':<25s} | {'Period':>6s} | {'MAE':>8s} | {'MAE_16':>8s} | {'MAE_32':>8s} | {'RMSE':>8s}"
    print(header)
    print("-" * 90)

    # Iterate results
    for row in all_results:
        model_id, mode_name, period, desc, MAE, MAE_16, MAE_32, MAE_STD, MAX_ERROR, RMSE = row

        mae_s = f"{MAE:.4f}" if MAE is not None else "N/A"
        m16_s = f"{MAE_16:.4f}" if MAE_16 is not None else "N/A"
        m32_s = f"{MAE_32:.4f}" if MAE_32 is not None else "N/A"
        m64_s = f"{RMSE:.4f}" if RMSE is not None else "N/A"

        print(f"{model_id:<12s} | {mode_name:<25s} | {period:>4d}h | {mae_s:>8s} | {m16_s:>8s} | {m32_s:>8s} | {m64_s:>8s}")

    # --------------------------------------------------------
    # 2.4 Comparative Analysis
    # --------------------------------------------------------
    print(f"\n{'='*90}")
    print("Key Comparative Analysis")
    print(f"{'='*90}")

    for model_id in models:
        model_results = [record for record in all_results if record[0] == model_id and record[4] is not None]

        if len(model_results) < 2:
            continue

        print(f"\n[{model_id}]")

        # Baseline(24->24)
        base_result = model_results[0]
        mae_base = base_result[4]

        print(f"  Baseline MAE (Period Unchanged): {mae_base:.4f}")
        print(f"  Performance Degradation caused by Frequency Mismatch:")

        for row in model_results[1:]:
            mode_name, period, mae = row[1], row[2], row[4]
            ratio = mae / mae_base if mae_base > 0 else 0
            degradation = (ratio - 1) * 100

            print(f"    {mode_name:<25s}  MAE: {mae:.4f}  (Degradation {degradation:+.1f}%)")            


    # --------------------------------------------------------
    # 2.5 Save Results
    # --------------------------------------------------------
    columns = [
        "Model", "Mode", "Period", "Description", "MAE", "MAE_16", "MAE_32", "MAE_STD", "MAX_ERROR", "RMSE"
    ]

    df_results = pd.DataFrame(all_results, columns=columns)
    all_results_csv_path = save_to_csv(RESULT_CSV_PATH, df_results, index=False)
    print(f"\nAll results saved to CSV: {all_results_csv_path}")

    return all_results, result_details


# ============================================================
# 4. Entry Point
# ============================================================
def main():
    results, details = run_frequency_mismatch_test()

if __name__ == "__main__":
    main()
