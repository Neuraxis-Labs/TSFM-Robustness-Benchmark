#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
concept_drift_test_v1.py - Concept Drift Test (Simplified Edition)
====================================
Industrial Context:
  Equipment start-stop cycles, load steps, and seasonal operating condition switches cause inconsistencies between 
  training data and prediction target distributions. It is necessary to evaluate the model's resistance to distribution drift.

Test Principle:
  Compare prediction accuracy differences across different drift scenarios.
  1. Drift Modes: Trend drift, Cyclic drift, Noise drift.
  2. Drift Timing: Before prediction segment, At start of prediction segment, At end of prediction segment.
  3. Drift Intensity: Noise std dev 2.0, 3.0, 4.0.

Test Method:
  1. Generate base stationary signal (Training segment).
  2. Generate drifted signal (Forecast segment).
  3. Call prediction interface, calculate evaluation metrics.
  4. Save prediction results.

Test Objective:
  Construct data with a stationary training segment and a prediction segment exhibiting distribution drift to test 
  the model's resistance to three typical drift modes. Verify whether a long context window becomes a burden under drift conditions.

Author: Janesong
Create Date: 2026/07/06, Update on 2026/07/21.
"""

import time
import numpy as np
import pandas as pd

from config.settings import RESULTS_DIR
from config.constants import MODEL_LIST, FORECAST_POINT_LEN_64, TRAIN_SEQ_LEN_512
from core.timecho import forecast
from core.metrics import calc_metrics
from neuraxis_testkit.utils.files import save_to_csv, ensure_dir

# ============================================================
# 1. Data related configuration
# ============================================================
OUTPUT_SUBDIR = RESULTS_DIR / "futureCovs" / "conceptDrift"
ensure_dir(OUTPUT_SUBDIR)
RESULT_CSV_PATH = OUTPUT_SUBDIR / "concept_drift_result_v1.csv"    # Prediction results file

N_TRAIN = TRAIN_SEQ_LEN_512          # Total length of training segment (including history window and preceding data)
N_FORECAST = FORECAST_POINT_LEN_64   # Forecast length (64 points)
TOTAL = N_TRAIN + N_FORECAST         # Total sequence length

# Drift lead time: Introduce drift starting DRIFT_LEAD points before the end of the training segment
# This ensures the history window contains drift information when input_length > DRIFT_LEAD
DRIFT_LEAD = 20

# Noise Parameters
BASE_NOISE_STD = 2.0                           # Base noise standard deviation
VAR_3X_NOISE_STD = BASE_NOISE_STD * np.sqrt(3) # Standard deviation (≈3.4641, var=12)

# ============================================================
# 2. Generate Base Stationary Signal (Training Segment)
# ============================================================
dates = pd.date_range("2026-07-01", periods=TOTAL, freq="1h")

# Training segment (first N_TRAIN points): Trend + 24h cycle + Small noise
trend_base = np.linspace(50, 65, N_TRAIN)
seasonal_base = 15 * np.sin(2 * np.pi * np.arange(N_TRAIN) / 24)
noise_base = np.random.randn(N_TRAIN) * BASE_NOISE_STD
train_steady = trend_base + seasonal_base + noise_base

# ============================================================
# 3. Generate Full Sequences for Different Drift Modes (Drift starts at N_TRAIN - DRIFT_LEAD)
# ============================================================
def generate_full_sequence(mode):
    """
    Generate a full sequence (length TOTAL) where drift starts DRIFT_LEAD points before the end of the training segment.

    Args:
        mode (str): Drift mode string (e.g., 'B1-Baseline(No Drift)')

    Returns:
        tuple[np.ndarray, int]:
            - full_sequence: The generated full signal array
            - drift_start_index: Index where drift starts (N_TRAIN - DRIFT_LEAD)
    """
    t_full = np.arange(TOTAL)
    # Base trend (entire sequence: training segment trend + forecast segment trend continuation)
    trend_full = np.concatenate([
        np.linspace(50, 65, N_TRAIN),
        np.linspace(65, 80, N_FORECAST)
    ])
    # Base seasonality (continuous)
    seasonal_full = 15 * np.sin(2 * np.pi * t_full / 24)
    # Base noise (generated entirely to ensure continuity)
    noise_full = np.random.randn(TOTAL) * BASE_NOISE_STD

    # Copy to overlay drift
    signal = trend_full + seasonal_full + noise_full

    # Drift start index
    drift_start = N_TRAIN - DRIFT_LEAD

    if mode == "B1-Baseline(No Drift)":
        # No drift applied
        pass

    elif mode == "B2-Mean Shift(+15)":
        # Mean shift +15, starting from drift_start
        signal[drift_start:] += 15

    elif mode == "B3-Variance Expansion(3x)":
        # Variance expanded by 3x, starting from drift_start. Regenerate noise (keeping trend and seasonality unchanged).
        # To maintain continuity, only replace noise after drift_start
        noise_new = np.random.randn(TOTAL - drift_start) * VAR_3X_NOISE_STD
        signal[drift_start:] = (trend_full + seasonal_full)[drift_start:] + noise_new

    elif mode == "B4-Phase Shift(90deg)":
        # Seasonal phase shift by 90°, starting from drift_start
        seasonal_shifted = 15 * np.sin(2 * np.pi * t_full / 24 + np.pi / 2)
        signal[drift_start:] = trend_full[drift_start:] + seasonal_shifted[drift_start:] + noise_full[drift_start:]

    elif mode == "B5-Composite Drift(Mean+Var+Phase)":
        # Combination of three: Mean shift(+15) + Variance 3x + Phase shift 90°
        noise_new = np.random.randn(TOTAL - drift_start) * VAR_3X_NOISE_STD
        seasonal_shifted = 15 * np.sin(2 * np.pi * t_full / 24 + np.pi / 2)
        signal[drift_start:] = (
            (trend_full + seasonal_shifted)[drift_start:]
            + noise_new
            + 15
        )

    else:
        raise ValueError(f"Unknown scenario: {mode}")

    return signal.round(4), drift_start

# ============================================================
# 4. Execute Tests (for each scenario, input length, and model)
# ============================================================
SCENARIOS = [
    "B1-Baseline(No Drift)",
    "B2-Mean Shift(+15)",
    "B3-Variance Expansion(3x)",
    "B4-Phase Shift(90deg)",
    "B5-Composite Drift(Mean+Var+Phase)",
]
INPUT_LENGTHS = [96, 256, 512]

total_calls = len(MODEL_LIST) * len(SCENARIOS) * len(INPUT_LENGTHS)
print("=" * 80)
print("Scenario: Concept Drift Test (Drift appears early at the end of history window)")
print(f"   {len(MODEL_LIST)} Models x {len(SCENARIOS)} Scenarios x {len(INPUT_LENGTHS)} Input Lengths = {total_calls} Calls")
print(f"   Drift Lead: {DRIFT_LEAD} points (History window must be > {DRIFT_LEAD} to be visible)")
print(f"   Base Noise: std={BASE_NOISE_STD}, var={BASE_NOISE_STD**2:.0f}")
print(f"   B3/B5 Expanded Noise: std={VAR_3X_NOISE_STD:.4f}, var={VAR_3X_NOISE_STD**2:.0f} (3x variance)")
print("=" * 80)

all_results = []

for mode in SCENARIOS:
    print(f"\n[Scenario] {mode}")
    # Fix seed for reproducibility, but use different seeds per scenario for independent noise (optional)
    np.random.seed(42 + SCENARIOS.index(mode))
    full_seq, drift_start = generate_full_sequence(mode)

    # Construct full DataFrame
    df = pd.DataFrame({"time": dates, "target": full_seq})

    # Ground truth forecast segment (for error calculation)
    target_forecast = full_seq[N_TRAIN:]

    for in_len in INPUT_LENGTHS:
        # Slice history window: from drift_start - (in_len - DRIFT_LEAD) to N_TRAIN
        # But to ensure window length is in_len, start point is N_TRAIN - in_len
        start_idx = N_TRAIN - in_len
        history = df.iloc[start_idx:N_TRAIN][["time", "target"]].copy()

        # Check if history window contains the drift start point
        contains_drift = start_idx < drift_start
        print(f"   [Input Length {in_len}] History window contains drift start: {contains_drift}")

        for model_id in MODEL_LIST:
            t0 = time.perf_counter()
            try:
                pred_values, elapsed_ms, error = forecast(
                    targets=history,
                    model_id=model_id,
                    output_length=N_FORECAST,
                    time_col="time",
                    auto_adapt=True,
                )

                if error:
                    print(f"      [{model_id}] Failed: {str(error)[:60]}")
                    all_results.append({
                        "scenario": mode, "model_id": model_id, "input_length": in_len,
                        "mae": None, "rmse": None, "mape": None, "latency_ms": elapsed_ms,
                        "success": False, "error": str(error),
                        "contains_drift": contains_drift,
                    })
                else:
                    metrics = calc_metrics(pred_values, target_forecast)
                    print(f"      [{model_id}] MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  MAPE={metrics['MAPE']:.4f}  Latency={elapsed_ms:.0f}ms")
                    all_results.append({
                        "scenario": mode, "model_id": model_id, "input_length": in_len,
                        "mae": metrics["MAE"], "rmse": metrics["RMSE"], "mape": metrics["MAPE"], "latency_ms": elapsed_ms,
                        "success": True, "error": None,
                        "contains_drift": contains_drift,
                    })
            except Exception as exp:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(f"      [{model_id}] Exception: {str(exp)[:60]}")
                all_results.append({
                    "scenario": mode, "model_id": model_id, "input_length": in_len,
                    "mae": None, "rmse": None, "mape": None, "latency_ms": elapsed_ms,
                    "success": False, "error": str(exp),
                    "contains_drift": contains_drift,
                })

            time.sleep(1)


# ============================================================
# 5. Summary Print
# ============================================================
print("\n" + "=" * 80)
print("Test Results Summary")
print("=" * 80)

print(f"\n{'Scenario':>28s} | {'InLen':>6s} | {'Model':>12s} | {'Drift?':>8s} | {'MAE':>10s} | {'RMSE':>10s} | Status")
print("-" * 110)

for r in all_results:
    if r["success"]:
        contains = "Yes" if r["contains_drift"] else "No"
        print(f"{r['scenario']:>28s} | {r['input_length']:>6d} | {r['model_id']:>12s} | {contains:>8s} | {r['mae']:>10.4f} | {r['rmse']:>10.4f} | Success")
    else:
        print(f"{r['scenario']:>28s} | {r['input_length']:>6d} | {r['model_id']:>12s} | {'-':>8s} | {'N/A':>10s} | {'N/A':>10s} | Failed")


# ============================================================
# 6. Core Analysis
# ============================================================
print("\n" + "=" * 80)
print("Core Analysis")
print("=" * 80)

# Analysis 1: Degradation ratio of drift scenarios vs baseline (only for input_length=256 and history containing drift)
print("\n[Analysis 1] Prediction Accuracy Degradation: Drift Scenarios vs Baseline (input_length=256, history contains drift)")
print("-" * 70)

for model_id in MODEL_LIST:
    print(f"\n  [{model_id}]")
    # Get baseline MAE (B1 scenario)
    baseline_mae = None
    for r in all_results:
        if (r["model_id"] == model_id and "B1" in r["scenario"]
                and r["input_length"] == 256 and r["success"]):
            baseline_mae = r["mae"]
            break

    if baseline_mae is None:
        print(f"     Baseline data missing, skipping")
        continue

    print(f"     Baseline(B1) MAE = {baseline_mae:.4f}")
    for r in all_results:
        if (r["model_id"] == model_id and r["input_length"] == 256
                and r["success"] and "B1" not in r["scenario"]
                and r["contains_drift"]):   # Only analyze cases where history contains drift
            ratio = r["mae"] / baseline_mae
            if ratio < 1.2:
                verdict = "[Normal] No impact"
            elif ratio < 2.0:
                verdict = "[Slight] Slight degradation"
            elif ratio < 5.0:
                verdict = "[Warning] Significant degradation"
            else:
                verdict = "[Critical] Severe degradation"
            print(f"     {r['scenario']:>28s}: MAE={r['mae']:.4f} ({ratio:.1f}x) -> {verdict}")

# Analysis 2: Compare performance between 'History with Drift' vs 'History without Drift' (using B2 as example)
print("\n[Analysis 2] Impact of Historical Information on Drift Adaptation (Example: B2-Mean Shift, input_length=256)")
print("-" * 70)
for model_id in MODEL_LIST:
    print(f"\n  [{model_id}]")
    # Find results for B2 scenario, with and without drift
    mae_with = None
    mae_without = None
    for r in all_results:
        if r["model_id"] == model_id and "B2" in r["scenario"] and r["input_length"] == 256 and r["success"]:
            if r["contains_drift"]:
                mae_with = r["mae"]
            else:
                mae_without = r["mae"]
    if mae_with is not None and mae_without is not None:
        print(f"     History w/ Drift MAE = {mae_with:.4f}")
        print(f"     History w/o Drift MAE = {mae_without:.4f}")
        print(f"     Improvement with drift info = {(mae_without - mae_with) / mae_without * 100:.1f}%")
    else:
        print(f"     Insufficient data, skipping")

# Analysis 3: Effectiveness of Long Context under Drift (B5 Composite Drift, only history containing drift)
print("\n[Analysis 3] Benefits of Long Context under Drift (B5-Composite Drift, history contains drift)")
print("-" * 70)

for model_id in MODEL_LIST:
    print(f"\n  [{model_id}]")
    b5_results = {}
    for r in all_results:
        if (r["model_id"] == model_id and "B5" in r["scenario"]
                and r["success"] and r["contains_drift"]):
            b5_results[r["input_length"]] = r["mae"]

    if len(b5_results) < 2:
        print(f"     B5 data insufficient, skipping")
        continue

    for in_len in sorted(b5_results.keys()):
        mae = b5_results[in_len]
        print(f"     input={in_len:>3d}: MAE={mae:.4f}")

    # Determine trend
    mae_list = [b5_results[k] for k in sorted(b5_results.keys())]
    if mae_list[-1] < mae_list[0] * 0.9:
        print(f"     [Pass] Long window(512) has lower MAE than Short window(96) -> Long context provides benefit under drift")
    elif mae_list[-1] > mae_list[0] * 1.1:
        print(f"     [Warning] Long window(512) has higher MAE than Short window(96) -> Long context may introduce redundant information")
    else:
        print(f"     -> MAE similar across lengths; input_length has minimal impact on drift scenarios")


# ============================================================
# 7. Save Results
# ============================================================
result_path = save_to_csv(RESULT_CSV_PATH, all_results)
print(f"   Results saved to CSV: {result_path}")
print("=" * 80)
print(" Test completed!")
