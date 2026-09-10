"""
forecast_horizon_ablation.py -- Forecast Horizon Ablation Experiment
====================================
Test Principle: Fix the input length and vary the forecast horizon to observe changes in accuracy

Test Objective:
  Fix input length to 512 and gradually increase output_length (16->32->64->128->256).
  Observe the decay pattern of prediction accuracy over the forecast horizon under 
  both normal and drift scenarios, providing quantitative evidence for the 
  "Safe Forecast Window" in industrial deployment.

Total calls: 20 (2 models * 5 lengths * 2 scenarios)

Author: Janesong
Create Date: 2026/07/05
"""

import numpy as np
import pandas as pd
from core.timecho import forecast
from core.metrics import calc_metrics
from config.constants import TRAIN_SEQ_LEN_512, FORECAST_POINT_LEN_256

# ============================================================
# Configuration
# ============================================================
N_TRAIN = TRAIN_SEQ_LEN_512
MAX_FORECAST = FORECAST_POINT_LEN_256
DRIFT_OFFSET = 15.0         # Magnitude of drift offset
DRIFT_TREND_START = 65.0    # Start value of drift trend
DRIFT_TREND_END = 90.0      # End value of drift trend
NOISE_STD_NORMAL = 2.0
NOISE_STD_DRIFT = 4.0

FORECAST_LENGTHS = [16, 32, 64, 128, 256]
MODELS = ["Chronos-2", "Timer-3.5"]
SCENARIOS = ["normal", "drift"]

def generate_test_data():
    """Generate test data"""
    np.random.seed(42)

    # Construct test data (generate 512+256=768 points in total)
    time_full = pd.date_range("2026-07-01", periods=N_TRAIN + MAX_FORECAST, freq="1h")
    time_history = time_full[:N_TRAIN]

    # --- Normal data ---
    trend_normal = np.linspace(50, 65, N_TRAIN + MAX_FORECAST)
    seasonal_normal = 15 * np.sin(2 * np.pi * np.arange(N_TRAIN + MAX_FORECAST) / 24)
    noise_normal = np.random.randn(N_TRAIN + MAX_FORECAST) * NOISE_STD_NORMAL
    target_normal = (trend_normal + seasonal_normal + noise_normal).round(4)

    # --- B5 Compound Drift data (history segment same as normal data) ---
    target_drift = target_normal.copy()
    trend_drift_fc = np.linspace(DRIFT_TREND_START, DRIFT_TREND_END, MAX_FORECAST)
    seasonal_drift_fc = DRIFT_OFFSET * np.sin(2 * np.pi * np.arange(N_TRAIN, N_TRAIN + MAX_FORECAST) / 12)
    noise_drift_fc = np.random.randn(MAX_FORECAST) * NOISE_STD_DRIFT
    target_drift[N_TRAIN:] = (trend_drift_fc + seasonal_drift_fc + noise_drift_fc + DRIFT_OFFSET).round(4)

    # History segment (shared by both scenarios)
    df_history = pd.DataFrame({"time": time_history, "target": target_normal[:N_TRAIN]})

    # Ground truth dictionary (sliced by length)
    # Ground truth 字典(按长度切片)
    gt_normal = {L: target_normal[N_TRAIN:N_TRAIN + L] for L in FORECAST_LENGTHS}
    gt_drift = {L: target_drift[N_TRAIN:N_TRAIN + L] for L in FORECAST_LENGTHS}

    print(f"Training length: {N_TRAIN}, Max forecast length: {MAX_FORECAST}")
    print(f"Models: {MODELS}")
    print(f"Forecast lengths: {FORECAST_LENGTHS}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Estimated calls: {len(MODELS) * len(FORECAST_LENGTHS) * len(SCENARIOS)}")
    print()

    return df_history, gt_normal, gt_drift


# Main Execution Block 执行测试
def run_forecast_experiments(df_history, gt_normal, gt_drift):
    """Run forecast experiments"""

    # results[scenario][model][length] = {"mae": float, "step_mae": np.ndarray, "pred_len": int}
    results = {}

    for scenario in SCENARIOS:
        results[scenario] = {}
        scenario_name = "Normal Data" if scenario == "normal" else "B5 Compound Drift"
        print(f"{'='*70}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*70}")

        for model_id in MODELS:
            results[scenario][model_id] = {}
            print(f"\n  Model: {model_id}")

            for L in FORECAST_LENGTHS:
                gt = gt_normal[L] if scenario == "normal" else gt_drift[L]
                try:
                    pred, _, _ = forecast(
                        targets=df_history,
                        model_id=model_id,
                        output_length=L,
                        time_col="time"
                    )
                    metrics = calc_metrics(pred, gt)
                    mae = metrics["MAE"]
                    step_mae = np.abs(pred - gt)
                    results[scenario][model_id][L] = {
                        "mae": mae,
                        "step_mae": step_mae,
                        "pred_len": len(pred)
                    }
                    print(f"    L={L:>3d}  MAE={mae:.4f}  (pred_len={len(pred)})")
                except Exception as exp:
                    print(f"    L={L:>3d}  Failed: {type(exp).__name__}: {exp}")
                    results[scenario][model_id][L] = {"mae": None, "step_mae": None, "pred_len": 0}

        print()

    return results


def print_summary_table(results):
    """Print summary table"""
    print(f"{'='*80}")
    print("C2 Forecast Horizon Ablation - Summary")
    print(f"{'='*80}")
    print(f"{'Scenario':<12s} | {'Model':<15s} | {'L=16':>8s} | {'L=32':>8s} | {'L=64':>8s} | {'L=128':>8s} | {'L=256':>8s}")
    print("-" * 80)

    for scenario in SCENARIOS:
        scenario_name = "Normal Data" if scenario == "normal" else "B5 Drift"
        for model_id in MODELS:
            vals = []
            for L in FORECAST_LENGTHS:
                mae = results[scenario][model_id][L]["mae"]
                vals.append(f"{mae:.4f}" if mae is not None else "N/A")
            print(f"{scenario_name:<12s} | {model_id:<15s} | {vals[0]:>8s} | {vals[1]:>8s} | {vals[2]:>8s} | {vals[3]:>8s} | {vals[4]:>8s}")
        print("-" * 80)


def print_step_decay_analysis(results):
    """Print step-by-step decay analysis"""
    print(f"\n{'='*80}")
    print("Step-by-step MAE Decay (First 16 steps, cross-horizon comparison)")
    print(f"{'='*80}")

    for scenario in SCENARIOS:
        scenario_name = "Normal Data" if scenario == "normal" else "B5 Drift"
        for model_id in MODELS:
            print(f"\n  [{scenario_name} - {model_id}]")
            print(f"  {'Step':<6s} | ", end="")
            for L in FORECAST_LENGTHS:
                print(f"L={L:<5d}", end=" | ")
            print()
            print("  " + "-" * (8 + 10 * len(FORECAST_LENGTHS)))

            for step in range(16):
                print(f"  t+{step+1:<4d} | ", end="")
                for L in FORECAST_LENGTHS:
                    sm = results[scenario][model_id][L]["step_mae"]
                    if sm is not None and step < len(sm):
                        print(f"{sm[step]:.3f} ", end=" | ")
                    else:
                        print(f"  N/A ", end=" | ")
                print()


def print_ratio_analysis(results):
    """Print key ratio analysis"""
    print(f"\n{'='*80}")
    print("Accuracy Decay Ratio (MAE at L=256 / MAE at L=16)")
    print(f"{'='*80}")

    for scenario in SCENARIOS:
        scenario_name = "Normal Data" if scenario == "normal" else "B5 Drift"
        for model_id in MODELS:
            mae_16 = results[scenario][model_id][16]["mae"]
            mae_256 = results[scenario][model_id][256]["mae"]
            if mae_16 and mae_256 and mae_16 > 0:
                ratio = mae_256 / mae_16
                print(f"  {scenario_name:<12s} | {model_id:<15s} | L=16: {mae_16:.4f} -> L=256: {mae_256:.4f} | Decay Ratio: {ratio:.2f}x")


def main():
    """Main function"""
    print(f"Training length: {N_TRAIN}, Max forecast length: {MAX_FORECAST}")
    print(f"Models: {MODELS}")
    print(f"Forecast lengths: {FORECAST_LENGTHS}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Estimated calls: {len(MODELS) * len(FORECAST_LENGTHS) * len(SCENARIOS)}")

    print()

    # 1. Generate test data
    df_history, gt_normal, gt_drift = generate_test_data()

    # 2. Run forecast experiments
    results = run_forecast_experiments(df_history, gt_normal, gt_drift)

    # 3. Print summary table
    print_summary_table(results)

    # 4. Print step-by-step decay analysis
    print_step_decay_analysis(results)

    # 5. Print ratio analysis
    print_ratio_analysis(results)


# ============================================================
# Program Entry
# ============================================================
if __name__ == "__main__":
    main()
