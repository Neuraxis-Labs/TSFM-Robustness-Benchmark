#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dirty_test.py — Dirty Data Robustness Test (No NaN Support)
====================================
Test Purpose: Validate model's robustness against missing values and anomalous spikes

Test Principle: Predict using 7 types of dirty data (including baseline) and compare accuracy degradation

API Limitation: TimechoAI API does not support NaN values input (including target and cov columns)

Author: Janesong
Create Date: 2026/06/29, Update on 2026/08/07.
"""

import time
import numpy as np
import pandas as pd
from config.settings import TESTCASES_DIR, RESULTS_DIR
from config.constants import MODEL_LIST, HISTORY_POINT_LEN_256, FORECAST_POINT_LEN_64
from core.timecho import forecast
from core.results import get_results, load_results_from_csv, append_result_to_csv
from core.resume import is_rate_limited
from core.metrics import calc_metrics
from neuraxis_testkit.utils.files import read_csv_to_dataframe, ensure_dir
from neuraxis_testkit.utils.data_sanitizer import clean_nan_values

# ============================================================
# Data related configuration
# ============================================================
DATA_SUBDIR = TESTCASES_DIR / "futureCovs" / "dirtyData" / "data"    # Test data file path
DATA_CSV_PATH = DATA_SUBDIR / "dirty_clean.csv"
OUTPUT_SUBDIR = RESULTS_DIR / "futureCovs" / "dirtyData"
ensure_dir(OUTPUT_SUBDIR)
RESULT_CSV_PATH = OUTPUT_SUBDIR / "dirty_test_result.csv"    # Prediction results file

# ============================================================
# Calculate test quantity
# ============================================================
completed_records, perm_fail_count = load_results_from_csv(str(RESULT_CSV_PATH))

completed_keys = set()  # Build key set for completed tests (model_id, scene, pass_name)
retry_keys = set()      # Rate limit errors pending retry
for record in completed_records:
    key = (record.get("model_id"), record.get("scene"), record.get("pass_name"))
    if record.get("success") == True:
        completed_keys.add(key)
    elif is_rate_limited(str(record.get("error", ""))):
        retry_keys.add(key)  # Rate limit error, add to retry set
    # Other failures not counted in completed_keys, will be retested

# 7 test scenarios
SCENES = [
    ("S0-Clean",       "dirty_s0_clean.csv"),
    ("S1-Missing5%",     "dirty_s1_miss5.csv"),
    ("S2-Missing15%",    "dirty_s2_miss15.csv"),
    ("S3-ContinuousMissing",   "dirty_s3_miss_block.csv"),
    ("S4-SingleSpike",   "dirty_s4_spike_single.csv"),
    ("S5-MultiSpike",   "dirty_s5_spike_multi.csv"),
    ("S6-MixedDirty",     "dirty_s6_mixed.csv"),
]

total_tests = len(MODEL_LIST) * len(SCENES) * 2   # Raw and Preprocessed passes
skipped_tests = len(completed_keys)
remaining_tests = total_tests - skipped_tests - perm_fail_count

print(f"Total tasks: {total_tests} | Remaining: {remaining_tests} | Completed: {skipped_tests} | Permanent failures(Skip): {perm_fail_count}")
print()

# Read Data: ground truth (use last 64 rows of clean data as true values)
print("  Reading data for ground truth...")
clean_df = read_csv_to_dataframe(DATA_CSV_PATH)
clean_df["time"] = pd.to_datetime(clean_df["time"])
ground_truth = clean_df.iloc[HISTORY_POINT_LEN_256:]["target"].values
future_cov = clean_df.iloc[HISTORY_POINT_LEN_256:][["time", "cov"]].copy()
print(f"   ground_truth: {len(ground_truth)} points")
print(f"   ground_truth range: {ground_truth.min():.2f} ~ {ground_truth.max():.2f}")
print()

def _make_base_record(model_id, csv_file, nan_count, ground_truth):
    """Create base_record, ensure fixed column order"""
    return {
        "model_id": model_id,
        "scene": None,
        "csv_file": csv_file,
        "pass_name": None,
        "success": None,
        "mae": None,
        "rmse": None,
        "mape": None,
        "latency_ms": None,
        "pred_min": None,
        "pred_max": None,
        "truth_min": float(np.min(ground_truth)),
        "truth_max": float(np.max(ground_truth)),
        "is_explosion": None,
        "nan_count": nan_count,
        "error": None,
    }

# ============================================================
# Test by scene * model
# ============================================================

api_call_count = 0     # API call counter
success_count = 0
fail_count = 0

print(f" Starting test...")
print("=" * 90)

for model_id in MODEL_LIST:
    # Determine if it's a univariate model
    is_univariate = model_id.startswith("Timer")
    print(f"\n{'─' * 90}")
    print(f" Model: {model_id} (Univariate mode: {is_univariate})")
    print(f"{'─' * 90}")

    for scene_name, csv_file in SCENES:
        print(f"\n   Scenario: {scene_name} ({csv_file})")

        # Read dirty data
        df = read_csv_to_dataframe(DATA_SUBDIR / csv_file)
        df["time"] = pd.to_datetime(df["time"])

        history = df.iloc[:HISTORY_POINT_LEN_256].copy()
        # Check dirty data overview
        nan_count = history["target"].isna().sum()
        valid_vals = history["target"].dropna()
        data_range = f"{valid_vals.min():.1f}~{valid_vals.max():.1f}" if len(valid_vals) > 0 else "All NaN"
        print(f"     History target: NaN={nan_count}, Range={data_range}")

        scene_base = _make_base_record(model_id, csv_file, nan_count, ground_truth)

        # ===== Two passes: Raw + Preprocessed =====
        for pass_name, pass_df in [("Raw", history.copy()), ("Preprocessed", history.copy())]:
            label = f"{scene_name}[{pass_name}]"
            test_key = (model_id, label, pass_name)

            # Resume from checkpoint: Check if completed or pending retry
            if test_key in completed_keys and test_key not in retry_keys:
                # Already successfully completed, skip
                print(f"     [{pass_name}] Completed, skipped")
                continue

            # If rate-limited retry needed, notify user
            if test_key in retry_keys:
                print(f"     [{pass_name}] Retrying (previous 429 rate limit)")

            # Fill covariate NaNs in both passes to prevent API errors
            # (Covariate NaNs are not the focus of this test and would cause API rejection)
            if not is_univariate and "cov" in pass_df.columns:
                cov_nan_before = pass_df["cov"].isna().sum()
                if cov_nan_before > 0:
                    pass_df["cov"] = pass_df["cov"].ffill().bfill()
                    cov_nan_after = pass_df["cov"].isna().sum()
                    print(f"     [{pass_name}] Covariate preprocessing: Filled {cov_nan_before} NaN values")

            if pass_name == "Preprocessed":
                # Preprocessed pass: Fill target NaNs (test prediction capability on preprocessed data)
                target_nan_before = pass_df["target"].isna().sum()
                if target_nan_before > 0:
                    pass_df["target"] = pass_df["target"].ffill().bfill()
                    target_nan_after = pass_df["target"].isna().sum()
                    print(f"     [{pass_name}] Target preprocessing: Filled {target_nan_before} NaN values")
            else:
                # Raw pass: Keep target NaNs (test reaction to missing values)
                # If API doesn't support NaN, it will error out, which is a valid test result
                target_nan_count = pass_df["target"].isna().sum()
                if target_nan_count > 0:
                    print(f"     [{pass_name}] Target kept raw: {target_nan_count} NaN values (testing missing value robustness)")

            history_targets = pass_df[["time", "target"]]
            history_covs = pass_df[["time", "cov"]]

            try:
                # Dynamically build parameters: If Timer series, don't pass covariates
                forecast_kwargs = {
                    "targets": history_targets,
                    "model_id": model_id,
                    "output_length": FORECAST_POINT_LEN_64,
                    "time_col": "time",
                    "auto_adapt": True,
                }

                # Multivariate models need covariates
                if not is_univariate:
                    forecast_kwargs["history_covs"] = history_covs
                    forecast_kwargs["future_covs"] = future_cov

                # Call API through core.timecho wrapper
                api_call_count += 1
                print(f"     [{pass_name}] API call #{api_call_count}...")

                pred_values, elapsed_ms, error = forecast(**forecast_kwargs)

                if error:
                    # Check if rate limit error
                    if is_rate_limited(error):
                        print(f"     [{pass_name}] Rate limited (429), recorded, will retry next time")
                    else:
                        print(f"     [{pass_name}] Failed: {error[:80]}")

                    fail_count += 1

                    result_record = {
                        **scene_base,
                        "scene": label,
                        "pass_name": pass_name,
                        "success": False,
                        "latency_ms": elapsed_ms,
                        "error": error,
                    }
                    result_record = clean_nan_values(result_record)
                    append_result_to_csv(str(RESULT_CSV_PATH), result_record)

                else:
                    metrics = calc_metrics(pred_values, ground_truth)

                    # Detect if prediction "exploded" or "collapsed"
                    pred_max = float(np.max(pred_values))
                    pred_min = float(np.min(pred_values))
                    truth_max = float(np.max(ground_truth))
                    truth_min = float(np.min(ground_truth))
                    is_explosion = pred_max > truth_max * 1.5 or pred_min < truth_min * 0.5

                    status = " Explosion/Collapse!" if is_explosion else " Normal"
                    print(f"     [{pass_name}] {status} MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f},MAPE={metrics['MAPE']:.4f}, Range={pred_min:.2f}~{pred_max:.2f}")

                    success_count += 1

                    result_record = {
                        **scene_base,
                        "scene": label,
                        "pass_name": pass_name,
                        "success": True,
                        "mae": metrics["MAE"],
                        "rmse": metrics["RMSE"],
                        "mape": metrics["MAPE"],
                        "latency_ms": elapsed_ms,
                        "pred_min": pred_min,
                        "pred_max": pred_max,
                        "is_explosion": is_explosion,
                    }
                    result_record = clean_nan_values(result_record)
                    append_result_to_csv(str(RESULT_CSV_PATH), result_record)

            except Exception as exp:
                error_msg = str(exp)

                # Check if rate limit error
                if is_rate_limited(error_msg):
                    print(f"     [{pass_name}] Rate limited (429), recorded, will retry next time")
                else:
                    print(f"     [{pass_name}] Failed: {error_msg[:80]}")

                fail_count += 1

                result_record = {
                    **scene_base,
                    "scene": label,
                    "pass_name": pass_name,
                    "success": False,
                    "latency_ms": 0,
                    "error": error_msg,
                }
                result_record = clean_nan_values(result_record)
                append_result_to_csv(str(RESULT_CSV_PATH), result_record)

            time.sleep(1)

# ============================================================
# Test statistics
# ============================================================
print()
print("=" * 90)
print("Test Statistics")
print(f"   API calls: {api_call_count} | Success: {success_count} | Failed: {fail_count} | Skipped(completed): {skipped_tests}")
print("=" * 90)

# ============================================================
# Read complete results and generate summary report
# ============================================================
print("=" * 90)
print("Reading complete results, generating summary report")
print("=" * 90)

# Read all results (including previously completed)
results_data, _ = load_results_from_csv(str(RESULT_CSV_PATH))

print("\n" + "=" * 100)
print(" Robustness Analysis Conclusion")
print("=" * 100)

for model_id in MODEL_LIST:
    model_results = [r for r in results_data if r["model_id"] == model_id]

    if len(model_results) == 0:
        print(f"  [{model_id}] No result data")
        continue

    print(f"\n  【{model_id}】")
    s0_pre = get_results(results_data, model_id, "S0", "Preprocessed")

    if not s0_pre or not s0_pre["success"]:
        print(f"    [Warning] Baseline scenario failed, unable to evaluate robustness.")
        continue

    baseline_mae = s0_pre["mae"]
    print(f"    Baseline(S0-Clean[Preprocessed]) MAE = {baseline_mae:.4f}")

    # Analyze missing value scenarios
    print(f"\n     Missing Value Resistance (based on [Preprocessed] results):")
    for scene_prefix in ["S1", "S2", "S3"]:
        r_pre = get_results(results_data, model_id, scene_prefix, "Preprocessed")
        r_raw = get_results(results_data, model_id, scene_prefix, "Raw")

        # Raw pass results
        if r_raw and not r_raw["success"]:
            if "API doesn't support NaN input" in str(r_raw.get("error", "")):
                print(f"      {r_raw['scene'].replace('[Raw]',''):>14s} (Raw): API doesn't support NaN input")
            else:
                print(f"      {r_raw['scene'].replace('[Raw]',''):>14s} (Raw): Error ({r_raw.get('error', '')[:30]})")

        # Preprocessed pass results
        if r_pre and r_pre["success"]:
            ratio = r_pre["mae"] / baseline_mae if baseline_mae > 0 else float("inf")
            verdict = " No impact" if ratio < 1.5 else (" Slight degradation" if ratio < 3 else " Significant degradation")
            print(f"      {r_pre['scene'].replace('[Preprocessed]',''):>14s} (Preprocessed): MAE={r_pre['mae']:.4f} ({ratio:.1f}x of baseline) -> {verdict}")

    # Anomalous Spike Resistance
    print(f"\n     Anomalous Spike Resistance (based on [Raw] results):")
    for scene_prefix in ["S4", "S5"]:
        r_raw = get_results(results_data, model_id, scene_prefix, "Raw")
        if r_raw and r_raw["success"]:
            ratio = r_raw["mae"] / baseline_mae if baseline_mae > 0 else float("inf")
            verdict = " Explosion/Collapse!" if r_raw["is_explosion"] else (" Resisted" if ratio < 1.5 else "[Warning] Accuracy degradation")
            print(f"      {r_raw['scene'].replace('[Raw]',''):>14s}: MAE={r_raw['mae']:.4f} ({ratio:.1f}x of baseline) -> {verdict}")

    # Mixed Dirty Data
    print(f"\n     Mixed Dirty Data (based on [Preprocessed] results):")
    r_pre = get_results(results_data, model_id, "S6", "Preprocessed")
    r_raw = get_results(results_data, model_id, "S6", "Raw")

    if r_raw and not r_raw["success"]:
        if "API doesn't support NaN input" in str(r_raw.get("error", "")):
            print(f"      S6-MixedDirty (Raw): API doesn't support NaN input")

    if r_pre and r_pre["success"]:
        ratio = r_pre["mae"] / baseline_mae if baseline_mae > 0 else float("inf")
        verdict = " Explosion/Collapse!" if r_pre["is_explosion"] else (" Production-ready" if ratio < 1.5 else " Not usable")
        print(f"      S6-MixedDirty (Preprocessed): MAE={r_pre['mae']:.4f} ({ratio:.1f}x of baseline) -> {verdict}")

    print()

# ============================================================
# Results File
# ============================================================
print("=" * 90)
print(" Results File")
print("=" * 90)
print(f"   CSV results path: {RESULT_CSV_PATH}")
print(" Test completed!")
print("=" * 90)
