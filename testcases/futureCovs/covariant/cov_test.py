#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cov_test.py - Covariate Validity Test

Test Purpose: Verify if TimechoAI uses "future covariates"

Test Principle: Pass different future_covs and check if prediction results differ

Author: Janesong
Create Date: 2026/06/29, Update on 2026/07/12.
"""

import time
import numpy as np

from config.settings import TESTCASES_DIR, RESULTS_DIR
from config.constants import HISTORY_POINT_LEN_256, FORECAST_POINT_LEN_64
from core.timecho import forecast
from core.metrics import calc_metrics, calc_diff
from neuraxis_testkit.utils.files import read_csv_to_dataframe, save_with_json_backup, ensure_dir

# ============================================================
# Data related configuration
# ============================================================
CSV_PATH = TESTCASES_DIR / "futureCovs" / "convariant" / "data" / "cov_test_data.csv"    # Test data file
OUTPUT_SUBDIR = RESULTS_DIR / "futureCovs" / "convariant"
ensure_dir(OUTPUT_SUBDIR)
RESULT_PATH = OUTPUT_SUBDIR / "cov_test_results.csv"    # Prediction results file

MODEL_ID = "Auto"  # Model list: Auto / Timer-3.5 / Timer-3.0 / Chronos-2 / AutoARIMA / Holt-Winters

# Read data
print("  Reading data...")
raw_df = read_csv_to_dataframe(CSV_PATH)
print(f"   Total rows: {len(raw_df)}")
print(f"   Columns: {list(raw_df.columns)}")
print()

# Split data
history = raw_df.iloc[:HISTORY_POINT_LEN_256].copy()
future_real = raw_df.iloc[HISTORY_POINT_LEN_256:].copy()
ground_truth = future_real["target"].values

print(f"   History data: {len(history)} rows (time range: {history['time'].iloc[0]} ~ {history['time'].iloc[-1]})")
print(f"   Future real values: {len(future_real)} rows (time range: {future_real['time'].iloc[0]} ~ {future_real['time'].iloc[-1]})")
print()

# ============================================================
# Construct 4 test scenarios
# ============================================================
#
# Scenario description:
# A. Pass "real" future covariates -> If model uses cov, prediction should be most accurate
# B. Pass "random noise" covariates -> If model uses cov, prediction should degrade
# C. Pass "completely opposite" covariates -> If model uses cov, prediction should deviate significantly
# D. Pass "no" covariates at all -> For comparison: univariate prediction performance
#
# If model [really uses] covariates: A is best, B/C degrade, D is in between
# If model [does not use] covariates: A/B/C/D are almost the same (passing anything doesn’t matter)
# ============================================================

print("  Constructing 4 test scenarios...")

# Scenario A: Real future covariates
future_cov_real = future_real[["time", "cov"]].copy()
print(f"   A. Real covariates cov mean: {future_cov_real['cov'].mean():.2f}")

# Scenario B: Pure random noise (mean 0, std 100)
future_cov_noise = future_real[["time"]].copy()
future_cov_noise["cov"] = np.random.randn(FORECAST_POINT_LEN_64) * 100
print(f"   B. Noise covariates  cov mean: {future_cov_noise['cov'].mean():.2f}(pure random)")

# Scenario C: Completely opposite (negate real values)
future_cov_anti = future_real[["time"]].copy()
future_cov_anti["cov"] = -future_real["cov"].values
print(f"   C. Opposite covariates  cov mean: {future_cov_anti['cov'].mean():.2f}(negated)")

# Scenario D: No covariates passed
print(f"   D. No covariates passed  (future_covs=None)")
print()

# ============================================================
# Common parameters
# ============================================================

history_targets = history[["time", "target"]]
history_covs = history[["time", "cov"]]

SCENARIOS = [
    ("A-Real covariates", future_cov_real),
    ("B-Noise covariates", future_cov_noise),
    ("C-Opposite covariates", future_cov_anti),
    ("D-No covariates", None),
]


# ============================================================
# Call prediction for each scenario
# ============================================================

print("  Starting prediction (4 scenarios total, consuming 4 quota)...")
print("-" * 60)

results = []
for scene_name, fc in SCENARIOS:
    print(f"  Scenario {scene_name} predicting...", end="", flush=True)
    pred_values, elapsed_ms, error = forecast(
        targets=history_targets,
        history_covs=history_covs,
        future_covs=fc,
        model_id=MODEL_ID,
        output_length=FORECAST_POINT_LEN_64,
        time_col="time",
        auto_adapt=True,
    )

    if error:
        print(f" Failed ({elapsed_ms:.0f}ms): {error}")
    else:
        print(f" Completed ({elapsed_ms:.0f}ms)")

    results.append({
        "scene": scene_name,
        "pred": pred_values,
        "latency_ms": elapsed_ms,
        "error": error,
    })
    time.sleep(1)

print()

# ============================================================
# Calculate errors and differences across scenarios
# ============================================================

metrics = {}
for r in results:
    metrics[r["scene"]] = calc_metrics(r["pred"], ground_truth)
    r["metrics"] = metrics[r["scene"]]

# Prediction differences across scenarios
diffs = {
    "AB": calc_diff(results[0]["pred"], results[1]["pred"]),  # Real vs Noise
    "AC": calc_diff(results[0]["pred"], results[2]["pred"]),  # Real vs Opposite
    "AD": calc_diff(results[0]["pred"], results[3]["pred"]),  # Real vs No covariates
}

# ============================================================
# Print result summary
# ============================================================

print("\n" + "=" * 80)
print("  Test Result Summary")
print("=" * 80)

print("\n【1】Prediction accuracy for each scenario (vs ground truth)")
print("-" * 70)
print(f"{'Scenario':>20s} | {'MAE':>10s} | {'RMSE':>10s} | {'MAPE(%)':>10s} | {'Latency(ms)':>10s}")
print("-" * 70)

for r in results:
    m = r["metrics"]
    mae_str = f"{m['MAE']:.4f}" if m['MAE'] is not None else "N/A"
    rmse_str = f"{m['RMSE']:.4f}" if m['RMSE'] is not None else "N/A"
    mape_str = f"{m['MAPE']:.2f}" if m['MAPE'] is not None else "N/A"
    print(f"{r['scene']:>20s} | {mae_str:>10s} | {rmse_str:>10s} | {mape_str:>10s} | {r['latency_ms']:>10.0f}")

print("\n【2】Prediction differences across scenarios (core metrics)")
print("-" * 70)
print(f"{'Comparison':>30s} | {'Mean Absolute Diff':>12s} | {'Judgment':>20s}")
print("-" * 70)

diff_labels = [
    ("A(Real) vs B(Noise) prediction diff", diffs["AB"]),
    ("A(Real) vs C(Opposite) prediction diff", diffs["AC"]),
    ("A(Real) vs D(No covariates) prediction diff", diffs["AD"]),
]
for name, diff in diff_labels:
    if diff is None:
        print(f"{name:>30s} | {'N/A':>12s} | {'N/A':>20s}")
    elif diff < 0.5:
        print(f"{name:>30s} | {diff:>12.4f} | {'[Warning] Almost no difference':>20s}")
    elif diff < 2.0:
        print(f"{name:>30s} | {diff:>12.4f} | {'  Weak influence':>20s}")
    else:
        print(f"{name:>30s} | {diff:>12.4f} | {'[Pass]Significant influence':>20s}")

print("\n【3】Automatic conclusion")
print("-" * 70)

if all(v is not None for v in diffs.values()):
    max_diff = max(diffs["AB"], diffs["AC"], diffs["AD"])

    if max_diff < 0.5:
        print("  Serious issue: Covariates are not taking effect at all!")
        print("   Passing real covariates, random noise, or opposite values yields almost identical prediction results. ")
        print(f"   -> {MODEL_ID} may not be loading future_covs weights during inference, or has degenerated to univariate mode. ")
    elif max_diff < 2.0:
        print(" Covariates have weak influence:")
        print("   Passing different covariates produces slight changes in predictions, but the influence is far less than expected. ")
        print("   -> Covariate weights may be compressed, or canceled by auto_adapt normalization. ")
    else:
        ma = metrics["A-Real covariates"]
        mb = metrics["B-Noise covariates"]
        mc = metrics["C-Opposite covariates"]
        if ma["MAE"] is not None and mb["MAE"] is not None:
            if ma["MAE"] < mb["MAE"] and ma["MAE"] < mc["MAE"]:
                print("[Pass] Covariates are effective and direction is correct:")
                print("   Passing real covariates yields the best accuracy, while noise/opposite degrades accuracy. ")
                print(f"   -> {MODEL_ID} is indeed utilizing future_covs information. ")
            else:
                print(" Covariates have influence but direction is abnormal:")
                print("   Passing different covariates changes predictions, but real covariates do not yield optimal accuracy. ")
                print("   -> Possible covariate normalization or alignment issue, needs further investigation. ")

# ============================================================
# Save results
# ============================================================

print("\n【4】Save results")

results_data = []
for r in results:
    scene_key = r["scene"][:1] + "_" + r["scene"][2:]  # "A-Real covariates" -> "A_Real covariates"
    if r["pred"] is not None:
        for i, v in enumerate(r["pred"]):
            results_data.append({
                "scene": scene_key,
                "step": i + 1,
                "pred_value": float(v),
                "ground_truth": float(ground_truth[i]),
                **{k: v for k, v in r["metrics"].items()},
                "latency_ms": r["latency_ms"],
            })

csv_path, json_path = save_with_json_backup(RESULT_PATH, results_data)
print(f"   Detailed results saved to CSV: {csv_path}")
print(f"   Prediction values saved to JSON: {json_path}")


print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
