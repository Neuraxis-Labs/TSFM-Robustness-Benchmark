#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cov_test_models.py — Covariate Support Test (Iterate All Models)

Test Purpose: Find out which models truly support covariates and which don't
Test Principle: Pass "real covariates" to each model and see who reports errors and who can use them

Author: Janesong
Create Date: 2026/06/30, Update on 2026/07/19.
"""

import time

from config.settings import TESTCASES_DIR, RESULTS_DIR
from config.constants import MODEL_LIST, HISTORY_POINT_LEN_256, FORECAST_POINT_LEN_64
from core.timecho import forecast
from core.metrics import calc_metrics
from neuraxis_testkit.utils.files import read_csv_to_dataframe, save_with_json_backup, ensure_dir

# ============================================================
# Data related configuration
# ============================================================
CSV_PATH = TESTCASES_DIR / "futureCovs" / "convariant" / "data" / "cov_test_data.csv"    # Test data file
OUTPUT_SUBDIR = RESULTS_DIR / "futureCovs" / "convariant"
ensure_dir(OUTPUT_SUBDIR)
RESULT_PATH = OUTPUT_SUBDIR / "cov_test_models_result.csv"    # Prediction results file

# Read data
print("  Reading data...")
raw_df = read_csv_to_dataframe(CSV_PATH)
print(f"   Total rows: {len(raw_df)}")
print(f"   Columns: {list(raw_df.columns)}")
print()

# Split data into history (input) and future (prediction target) parts
# History: first HISTORY_POINT_LEN_256 rows for model input
# Future: remaining rows for prediction and evaluation
history = raw_df.iloc[:HISTORY_POINT_LEN_256].copy()
future_real = raw_df.iloc[HISTORY_POINT_LEN_256:].copy()
ground_truth = future_real["target"].values  # Ground truth for metric calculation

# Prepare covariate data
# future_cov_real: "real" future covariates (known in advance, e.g., time features, holidays)
# If a model supports covariates, passing real values should yield better predictions
future_cov_real = future_real[["time", "cov"]].copy()

# Prepare history data
# history_targets: historical target values (required for all models)
# history_covs: historical covariate values (optional, depends on model)
history_targets = history[["time", "target"]]
history_covs = history[["time", "cov"]]

print(f"   Data: {len(raw_df)} rows")
print(f"   Correlation coefficient between target and cov: {raw_df['target'].corr(raw_df['cov']):.4f}")
print()

# ============================================================
# Iterate each model to test covariate support
# 
# Test methodology:
# 1. Pass real future covariates to each model
# 2. Models that support covariates will use them and return predictions
# 3. Models that don't support covariates will return 422 error
# 4. Calculate metrics (MAE, RMSE, MAPE) for models that succeeded
# ============================================================
print(f" Starting test for {len(MODEL_LIST)} models (consuming {len(MODEL_LIST)} quota)...")
print("=" * 80)

results = []

for model_id in MODEL_LIST:
    print(f"\n  Testing model: {model_id}")

   # Call API through core.timecho wrapper
    pred_values, elapsed_ms, error = forecast(
        targets=history_targets,
        history_covs=history_covs,
        future_covs=future_cov_real,
        model_id=model_id,
        output_length=FORECAST_POINT_LEN_64,
        time_col="time",
        auto_adapt=True,
    )

    # Check if error occurred
    if error:
        # Check if error is specifically about covariate support
        # Models return different error messages, need to check both Chinese and English versions
        is_cov_not_supported = (
            "不支持协变量" in error or
            "does not support covariates" in error
        )
        if is_cov_not_supported:
            print(f"     Covariates not supported (422 error)")
            print(f"     Error message: {error[:100]}...")
        else:
            print(f"     [Warning] Other error: {error[:120]}")

        results.append({
            "model_id": model_id,         # Model identifier
            "supports_cov": False,        # Whether model supports covariates
            "mae": None,                  # Mean Absolute Error (N/A for failed models)
            "rmse": None,                 # Root Mean Square Error (N/A for failed models)
            "mape": None,                 # Mean Absolute Percentage Error (N/A)
            "latency_ms": elapsed_ms,     # Prediction latency in milliseconds
            "error": error,               # Error message (if any)
            "pred_values": None,          # Predicted values (N/A for failed models)
        })
    else:
        # Calculate evaluation metrics for successful prediction
        m = calc_metrics(pred_values, ground_truth)
        print(f"     Supported! MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, MAPE={m['MAPE']:.2f}%, Latency={elapsed_ms:.0f}ms")

        results.append({
            "model_id": model_id,
            "supports_cov": True,
            "mae": m["MAE"],
            "rmse": m["RMSE"],
            "mape": m["MAPE"],
            "latency_ms": elapsed_ms,
            "error": None,
            "pred_values": pred_values.tolist(),
        })

    time.sleep(1)

# ============================================================
# Summary comparison
# ============================================================
print("\n" + "=" * 80)
print(" Summary: Covariate Support Status for Each Model")
print("=" * 80)
print()
print(f"{'Model':>15s} | {'Cov Support':>12s} | {'MAE':>10s} | {'RMSE':>10s} | {'MAPE(%)':>10s} | {'Latency(ms)':>12s}")
print("-" * 80)

for r in results:
    if r["supports_cov"]:
        print(f"{r['model_id']:>15s} | {' Yes':>12s} | {r['mae']:>10.4f} | {r['rmse']:>10.4f} | {r['mape']:>10.2f} | {r['latency_ms']:>12.0f}")
    else:
        print(f"{r['model_id']:>15s} | {' No':>12s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>10s} | {r['latency_ms']:>12.0f}")

print("\n【3】Automatic conclusion")
print()
print("=" * 80)
print(" Conclusion")
print("=" * 80)

# Separate models into supported and not_supported groups
supported = [r for r in results if r["supports_cov"]]
not_supported = [r for r in results if not r["supports_cov"]]

print(f"\n  Models supporting covariates ({len(supported)} models):")
for r in supported:
    print(f"    {r['model_id']:>12s}  MAE={r['mae']:.4f}  Latency={r['latency_ms']:.0f}ms")

print(f"\n  Models not supporting covariates ({len(not_supported)} models):")
for r in not_supported:
    print(f"    {r['model_id']:>12s}")

timer_models = [r for r in results if "Timer" in r["model_id"]]
timer_not_supported = [r for r in timer_models if not r["supports_cov"]]
if timer_not_supported:
    print(f"\n  Key Finding: ")
    print(f"     Timer series (proprietary flagship models) do not support covariates: ")
    for r in timer_not_supported:
        print(f"       - {r['model_id']}")
    print(f"     While third-party models (Chronos-2/AutoARIMA/Holt-Winters) may support them. ")
    print(f"     This indicates a capability gap in Tianmou's core technology roadmap for multivariate forecasting. ")

if supported:
    best = min(supported, key=lambda x: x["mae"])
    print(f"\n   Model with highest covariate prediction accuracy: {best['model_id']} (MAE={best['mae']:.4f})")

# ============================================================
# Save results
# ============================================================
summary_data = [
    {
        "model_id": r["model_id"],
        "supports_cov": r["supports_cov"],
        "mae": r["mae"],
        "rmse": r["rmse"],
        "mape": r["mape"],
        "latency_ms": r["latency_ms"],
        "error": r["error"],
    }
    for r in results
]
csv_path, json_path = save_with_json_backup(RESULT_PATH, summary_data)
print(f"   Detailed results saved to CSV: {csv_path}")
print(f"   Prediction values saved to JSON: {json_path}")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
