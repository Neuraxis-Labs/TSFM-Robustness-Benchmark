#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
concept_drift_test_v2.py - Concept Drift Test (XYZ Scenario,Revised Version) 概念漂移测试(XYZ场景,修正版)
====================================
Industrial Context:
  Equipment start-stop cycles, load steps, and seasonal operating condition switches cause inconsistencies between 
  training data and prediction target distributions. It is necessary to evaluate the model's resistance to distribution drift.
  设备启停、负载阶跃、季节性工况切换会导致训练数据与预测目标分布不一致. 需要评估模型对分布漂移的抵抗力.

Test Principle:
  Compare prediction accuracy across different drift visibility conditions.
  1. Drift Modes: Mean shift, Variance expansion, Phase shift, Covariate signal.
  2. Drift Visibility: Invisible (X), Partially visible (Y), Covariate-driven (Z).
  3. Context Lengths: 96, 256, 512.
  对比不同漂移可见性条件下的预测精度差异
  1. 漂移模式: 均值漂移、方差扩张、相位偏移、协变量传递
  2. 漂移可见性: 不可见(X)、部分可见(Y)、协变量驱动(Z)
  3. 上下文长度: 96, 256, 512

Test Method:
  1. Generate base stationary signal (Training segment).
  2. Generate drifted signal (Forecast segment) with X/Y/Z visibility.
  3. Call prediction interface, calculate evaluation metrics.
  4. Save prediction results.
  1. 生成基础平稳信号(训练段)
  2. 生成漂移信号(预测段), 区分X/Y/Z可见性
  3. 调用预测接口, 计算评估指标
  4. 保存预测结果

Test Objective:
  Construct data with a stationary training segment and a prediction segment exhibiting distribution drift to test 
  the model's resistance to three typical drift modes. Verify whether a long context window becomes a burden under drift conditions.
  构造训练段平稳、预测段发生分布漂移的数据, 检验模型对三种典型漂移
  模式的抵抗力, 并验证长上下文窗口在漂移下是否反而是负担.

Key Features:
  1. Automatically checks if concept_drift_result_v2.csv exists in the same directory.
  2. If CSV exists: Reads records, skips completed tasks, and resumes from checkpoint.
  3. If no CSV: Runs the full test suite from scratch.
  4. Immediately appends and saves results after each call is completed.
  5. 429 Rate Limit: Stops current run, does not decrement total count, retry allowed next time.
  6. 422 Permanent Failure: Decrements total count, no retry, does not affect final analysis.
  7. Timer-3.5/Timer-3.0 running Z-type scenarios: Skipped directly as they do not support covariates [422].
  1. 自动判断同级目录是否有 concept_drift_result_v2.csv
  2. 有 CSV: 读取记录, 跳过已完成, 走断点续跑
  3. 无 CSV: 从零开始跑全量
  4. 每完成一次调用立即追加保存
  5. 429 限流: 停止本次运行, 不扣总数, 下次可重试
  6. 422 永久失败: 扣总数, 不再重试, 不影响最终分析
  7. Timer-3.5/Timer-3.0 跑 Z 类场景: 直接跳过, 因为不支持协变量【422】

Revision Notes (Fixed concept_drift_test_v2_error.py):
  1. Future Trend Calculation: Maintains consistency with historical trend slope to eliminate error interference from sudden trend changes.
  2. Variance Multiplier Definition: Uses sqrt(multiplier) to calculate the standard deviation multiplier, ensuring variance expansion meets expectations.
  3. Y-Type History Construction: Explicitly separates trend/seasonality/noise to avoid pseudo-drift caused by mean downshift.
  1. 未来趋势计算: 历史趋势斜率保持一致, 排除趋势突变对误差的干扰.
  2. 方差倍率定义: 使用 sqrt(multiplier) 计算标准差倍率, 确保方差扩张符合预期.
  3. Y类历史构造: 显式分离趋势/季节/噪声, 避免均值下移的伪漂移.

Total Calls:
  Main Test (XYZ): 11 scenarios * 6 models * 3 lengths = 198 calls
  Ablation Test (Y4 auto_adapt): 6 models * 2 switches * 3 lengths = 36 calls
  Skipped Items:
    - Z Scenarios * Models without covariate support (NO_COV): 2 scenarios * 2 models * 3 lengths = 12 calls
    - Ablation Deduplication (Y4/adapt=True duplicate): 6 models * 1 switch * 3 lengths = 18 calls
  Original Total Tasks: 234 calls, Actual Required: 204 calls.
  主测试(XYZ): 11 场景 * 6 模型 * 3 长度 = 198 次
  消融测试(Y4 auto_adapt): 6 模型 * 2 开关 * 3 长度 = 36 次
  跳过项:
    - Z场景 * 不支持协变量模型(NO_COV): 2场景 * 2模型 * 3长度 = 12次
    - 消融去重(Y4/adapt=True重复): 6模型 * 1开关 * 3长度 = 18次
  原始任务总数 234 次, 实际需完成 204 次.

Author: Janesong
Create Date: 2026/07/21.
"""

import time
import numpy as np
import pandas as pd

from config.settings import RESULTS_DIR
from config.constants import MODEL_LIST, FORECAST_POINT_LEN_64, CONTEXT_LENGTH_512
from core.results import load_results_from_csv, append_result_to_csv
from core.resume import is_rate_limited
from core.timecho import forecast
from core.metrics import calc_metrics
from neuraxis_testkit.utils.files import ensure_dir

# ============================================================
# 1. Data related configuration
# ============================================================
OUTPUT_SUBDIR = RESULTS_DIR / "futureCovs" / "conceptDrift"
ensure_dir(OUTPUT_SUBDIR)
RESULT_CSV_PATH = OUTPUT_SUBDIR / "concept_drift_result_v2.csv"

N_CONTEXT = CONTEXT_LENGTH_512      # 上下文窗口总长度(历史段)
N_FORECAST = FORECAST_POINT_LEN_64  # Forecast length (64 points) 预测长度(64)
N_TOTAL = N_CONTEXT + N_FORECAST    # Total sequence length 序列总长度 576

NO_COV_MODELS = {"Timer-3.5", "Timer-3.0"}     # 不支持协变量的模型列表(Z类场景直接跳过)

DATES = pd.date_range("2026-07-06", periods=N_TOTAL, freq="1h")

# 基础信号参数
BASE_TREND_START = 50
BASE_TREND_END = 65
BASE_SEASONAL_AMP = 15
BASE_SEASONAL_PERIOD = 24
BASE_NOISE_STD = 2

# 漂移参数
DRIFT_MEAN_SHIFT = 15           # 均值平移幅度
DRIFT_VARIANCE_MULTIPLIER = 3   # 方差扩张倍数
# 计算对应的标准差扩张倍数 (方差是标准差的平方)
DRIFT_NOISE_STD_MULTIPLIER = np.sqrt(DRIFT_VARIANCE_MULTIPLIER)

DRIFT_PHASE_SHIFT = np.pi/2     # 相位偏移(90度)

# 漂移过渡区长度(在历史窗口末端逐步引入漂移, 模拟真实工况切换)
DRIFT_RAMP_LEN = 64             # 最后 64 个历史点逐步过渡

INPUT_LENGTHS = [96, 256, 512]

AUTO_ADAPT_ABLATION_SCENARIO = "Y4"
AUTO_ADAPT_VALUES = [True, False]

# 原始任务总数 = 主测试(11场景*6模型*3长度) + 消融(6模型*2开关*3长度)
TOTAL_RAW = 11 * 6 * 3 + 6 * 2 * 3  # = 234
# Z场景 * 不支持协变量的2个模型 * 3长度 = 12(代码层跳过, 不调API)
NO_COV_SKIP_COUNT = 2 * len(NO_COV_MODELS) * len(INPUT_LENGTHS)  # = 12
# 消融测试中 Y4/adapt=True 与主测试完全重复 = 6模型*1*3长度 = 18
DEDUP_SKIP_COUNT = 6 * 1 * len(INPUT_LENGTHS)  # = 18


# ============================================================
# 2. 信号生成与场景构造
# ============================================================
def generate_base_signal(n, seed=42):
    np.random.seed(seed)
    t = np.arange(n)
    trend = np.linspace(BASE_TREND_START, BASE_TREND_END, n)
    seasonal = BASE_SEASONAL_AMP * np.sin(2 * np.pi * t / BASE_SEASONAL_PERIOD)
    noise = np.random.randn(n) * BASE_NOISE_STD
    return (trend + seasonal + noise).round(4), t

def build_scenarios():
    scenarios = []
    full_base, t = generate_base_signal(N_TOTAL, seed=42)
    base_history = full_base[:N_CONTEXT].copy()
    base_future = full_base[N_CONTEXT:].copy()

    # --- [修正1] 未来趋势生成: 保持历史斜率, 不加速 ---
    # 历史斜率
    history_slope = (BASE_TREND_END - BASE_TREND_START) / N_CONTEXT
    # 未来趋势: 从历史终点继续延伸, 保持相同斜率
    future_trend = np.linspace(BASE_TREND_END, BASE_TREND_END + history_slope * N_FORECAST, N_FORECAST)

    t_future = np.arange(N_CONTEXT, N_TOTAL)
    future_seasonal = BASE_SEASONAL_AMP * np.sin(2 * np.pi * t_future / BASE_SEASONAL_PERIOD)

    np.random.seed(42)
    future_noise_base = np.random.randn(N_FORECAST) * BASE_NOISE_STD

    # --- [修正2] 噪声倍率: 使用标准差倍率 ---
    future_noise_expanded = future_noise_base * DRIFT_NOISE_STD_MULTIPLIER

    # 构造未来目标序列
    future_mean_shift = (future_trend + future_seasonal + future_noise_base + DRIFT_MEAN_SHIFT).round(4)
    future_variance_expansion = (future_trend + future_seasonal + future_noise_expanded).round(4)
    future_phase_shift = (future_trend + BASE_SEASONAL_AMP * np.sin(2 * np.pi * t_future / BASE_SEASONAL_PERIOD + DRIFT_PHASE_SHIFT) + future_noise_base).round(4)
    future_compound = (future_trend + future_seasonal + future_noise_expanded + DRIFT_MEAN_SHIFT + BASE_SEASONAL_AMP * np.sin(2 * np.pi * t_future / BASE_SEASONAL_PERIOD + DRIFT_PHASE_SHIFT)).round(4)

    # ================================================================
    # X 类: 漂移不可见(历史段完全平稳, 未来段突变)
    # ================================================================
    for sid, label, fut, desc in [
        ("X1", "X1-base(invisible)", base_future, "base"),
        ("X2", "X2-mean+15(invisible)", future_mean_shift, "mean"),
        ("X3", "X3-var3x(invisible)", future_variance_expansion, "variance"),
        ("X4", "X4-phase90(invisible)", future_phase_shift, "phase"),
        ("X5", "X5-compound(invisible)", future_compound, "compound"),
    ]:
        scenarios.append({
            "scenario_id": sid, "category": "X", "label": label,
            "history": pd.DataFrame({"time": DATES[:N_CONTEXT], "target": base_history}),
            "future_target": fut, "future_covs": None, "description": desc
        })

    # ================================================================
    # Y 类: 漂移部分可见(历史段末端逐步过渡, 未来段延续漂移, 修正构造逻辑)
    # ================================================================
    def build_y_history(drift_type):
        # 1. 获取历史段的确定性分量 (趋势和季节)
        t_ctx = np.arange(N_CONTEXT)
        hist_trend = np.linspace(BASE_TREND_START, BASE_TREND_END, N_CONTEXT)
        hist_seasonal = BASE_SEASONAL_AMP * np.sin(2 * np.pi * t_ctx / BASE_SEASONAL_PERIOD)

        # 2. 计算纯噪声: 原始数据 - 趋势 - 季节
        pure_noise = base_history - hist_trend - hist_seasonal

        stable_len = N_CONTEXT - DRIFT_RAMP_LEN
        ramp_t = np.arange(stable_len, N_CONTEXT)
        ramp_idx = np.arange(DRIFT_RAMP_LEN)

        # S型权重
        weight = 1 / (1 + np.exp(-(ramp_idx - DRIFT_RAMP_LEN / 2) / 8))

        # 3. 仅对目标分量进行漂移操作
        if drift_type == 'mean_shift':
            # 仅趋势上移
            hist_trend[stable_len:] += DRIFT_MEAN_SHIFT * weight
        elif drift_type == 'variance':
            # 仅噪声放大 (避免趋势突变)
            amplification = 1 + (DRIFT_NOISE_STD_MULTIPLIER - 1) * weight
            pure_noise[stable_len:] *= amplification
        elif drift_type == 'phase':
            # 仅相位偏移
            hist_seasonal[stable_len:] = BASE_SEASONAL_AMP * np.sin(2 * np.pi * ramp_t / BASE_SEASONAL_PERIOD + DRIFT_PHASE_SHIFT * weight)
        elif drift_type == 'compound':
            # 所有分量叠加
            hist_trend[stable_len:] += DRIFT_MEAN_SHIFT * weight
            hist_seasonal[stable_len:] = BASE_SEASONAL_AMP * np.sin(2 * np.pi * ramp_t / BASE_SEASONAL_PERIOD + DRIFT_PHASE_SHIFT * weight)
            amplification = 1 + (DRIFT_NOISE_STD_MULTIPLIER - 1) * weight
            pure_noise[stable_len:] *= amplification

        # 4. 重构信号 = 趋势 + 季节 + 噪声
        return (hist_trend + hist_seasonal + pure_noise).round(4)

    for sid, label, dt, fut, desc in [
        ("Y1", "Y1-mean+15(visible)", 'mean_shift', future_mean_shift, "mean"),
        ("Y2", "Y2-var3x(visible)", 'variance', future_variance_expansion, "variance"),
        ("Y3", "Y3-phase90(visible)", 'phase', future_phase_shift, "phase"),
        ("Y4", "Y4-compound(visible)", 'compound', future_compound, "compound"),
    ]:
        scenarios.append({
            "scenario_id": sid, "category": "Y", "label": label,
            "history": pd.DataFrame({"time": DATES[:N_CONTEXT], "target": build_y_history(dt)}),
            "future_target": fut, "future_covs": None, "description": desc
        })

    # ================================================================
    # Z 类: 协变量传递(历史段平稳, 通过未来协变量传递漂移信号)
    # ================================================================
    # Z1: 历史段无协变量(全0), 未来协变量为全1
    history_cov_z1 = pd.DataFrame({
        "time": DATES[:N_CONTEXT],
        "target": base_history,
        "load_level": np.zeros(N_CONTEXT)
    })
    future_cov_z1 = pd.DataFrame({
        "time": DATES[N_CONTEXT:N_TOTAL],
        "load_level": np.ones(N_FORECAST)
    })
    scenarios.append({
        "scenario_id": "Z1", "category": "Z", "label": "Z1-mean(cov)",
        "history": history_cov_z1,
        "future_target": future_mean_shift,
        "future_covs": future_cov_z1,
        "description": "cov signal"
    })

    # Z2: 历史段末尾逐渐升高的协变量, 未来继续为全1
    def _safe_logspace(start, end, n):
        if start <= 0:
            k = max(1, n // 3)
            p1 = np.linspace(start, 0.01, k)
            p2 = np.logspace(np.log10(0.01), np.log10(end), n - k + 1)[1:]
            return np.concatenate([p1, p2])
        return np.logspace(np.log10(start), np.log10(end), n)

    y_history_mean = build_y_history('mean_shift')  # 复用 Y1 的历史目标值
    history_cov_z2 = pd.DataFrame({
        "time": DATES[:N_CONTEXT],
        "target": y_history_mean,
        "load_level": np.concatenate([
            np.zeros(N_CONTEXT - DRIFT_RAMP_LEN),
            _safe_logspace(0.01, 1.0, DRIFT_RAMP_LEN)
        ])
    })
    future_cov_z2 = pd.DataFrame({
        "time": DATES[N_CONTEXT:N_TOTAL],
        "load_level": np.ones(N_FORECAST)
    })
    scenarios.append({
        "scenario_id": "Z2", "category": "Z", "label": "Z2-mean(cov+vis)",
        "history": history_cov_z2,
        "future_target": future_mean_shift,
        "future_covs": future_cov_z2,
        "description": "dual signal"
    })

    return scenarios


# ============================================================
# 3. 执行预测与评估
# ============================================================
def run_forecast(scenario, model_id, in_len, auto_adapt=True):
    # 提取 targets (必须包含 time 和 target)
    targets_df = scenario["history"][["time", "target"]].iloc[-in_len:].copy()

    # 提取历史协变量(如果存在除 time, target 之外的列)
    history_covs_df = None
    if len(scenario["history"].columns) > 2:
        # 必须包含 time 列, SDK 要求历史协变量与目标序列在时间上对齐
        cov_cols = [c for c in scenario["history"].columns if c not in ["time", "target"]]
        if cov_cols:
            history_covs_df = scenario["history"][["time"] + cov_cols].iloc[-in_len:].copy()

    # 未来协变量
    future_covs_df = scenario.get("future_covs")
    target = scenario["future_target"]

    t0 = time.perf_counter()
    try:
        kwargs = dict(
            targets=targets_df,
            model_id=model_id,
            output_length=N_FORECAST,
            time_col="time",
            auto_adapt=auto_adapt
        )
        if history_covs_df is not None:
            kwargs["history_covs"] = history_covs_df
        if future_covs_df is not None:
            kwargs["future_covs"] = future_covs_df

        pred_values, elapsed_ms, error = forecast(**kwargs)

        if error:
            return {"scenario_id": scenario["scenario_id"], "model_id": model_id, "input_length": in_len, "auto_adapt": auto_adapt, "success": False, "error": str(error), "mae": None, "rmse": None, "mape": None, "latency_ms": elapsed_ms}

        m = calc_metrics(pred_values, target)
        return {"scenario_id": scenario["scenario_id"], "model_id": model_id, "input_length": in_len, "auto_adapt": auto_adapt, "success": True, "error": None, "mae": m["MAE"], "rmse": m["RMSE"], "mape": m["MAPE"], "latency_ms": elapsed_ms}
    except Exception as exp:
        return {"scenario_id": scenario["scenario_id"], "model_id": model_id, "input_length": in_len, "auto_adapt": auto_adapt, "success": False, "error": str(exp)[:120], "mae": None, "rmse": None, "mape": None, "latency_ms": (time.perf_counter()-t0)*1000}


# ============================================================
# 4. 主流程
# ============================================================
def main():
    scenarios = build_scenarios()
    completed_records, perm_fail_count = load_results_from_csv(str(RESULT_CSV_PATH))

    completed_keys = set()
    for record in completed_records:
        aa_val = str(record.get("auto_adapt", True)).strip() == "True"
        key = (str(record["scenario_id"]), str(record["model_id"]), int(record["input_length"]), aa_val)
        if record.get("success"):
            completed_keys.add(key)
        elif not is_rate_limited(str(record.get("error", ""))):
            completed_keys.add(key)

    total_needed = TOTAL_RAW - NO_COV_SKIP_COUNT - DEDUP_SKIP_COUNT - perm_fail_count
    success_so_far = len([record for record in completed_records if record.get("success")])

    print("=" * 90)
    print(f"总任务: {TOTAL_RAW} | 需完成: {total_needed} | 已完成: {success_so_far} | 永久失败(Skip): {perm_fail_count}")
    print("=" * 90)

    stop_run = False
    runned = 0

    # 主测试
    for sc in scenarios:
        if stop_run: break
        for il in INPUT_LENGTHS:
            if stop_run: break
            for mid in MODEL_LIST:
                if sc["category"] == "Z" and mid in NO_COV_MODELS: continue
                key = (sc["scenario_id"], mid, il, True)
                if key in completed_keys: continue

                res = run_forecast(sc, mid, il)
                append_result_to_csv(str(RESULT_CSV_PATH), res)
                runned += 1

                if res["success"]:
                    print(f"[{sc['scenario_id']}] {mid:14s} in={il} MAE={res['mae']:.4f}")
                else:
                    print(f"[{sc['scenario_id']}] {mid:14s} in={il} Error: {res['error'][:50]}")
                    if is_rate_limited(res["error"]):
                        stop_run = True
                        print("Rate limited. Stop.")
                        break
                    else:
                        # 422等永久失败: 加入completed, 扣total, 继续运行
                        completed_keys.add(key)
                time.sleep(1)

    # 消融测试
    if not stop_run:
        ablation_sc = next((s for s in scenarios if s["scenario_id"] == AUTO_ADAPT_ABLATION_SCENARIO), None)
        if ablation_sc:
            for il in INPUT_LENGTHS:
                if stop_run: break
                for mid in MODEL_LIST:
                    if stop_run: break
                    for aa in AUTO_ADAPT_VALUES:
                        key = (AUTO_ADAPT_ABLATION_SCENARIO, mid, il, aa)
                        if key in completed_keys: continue
                        res = run_forecast(ablation_sc, mid, il, auto_adapt=aa)
                        append_result_to_csv(str(RESULT_CSV_PATH), res)
                        runned += 1
                        if res["success"]:
                            print(f"[Y4] {mid:14s} in={il} adapt={aa} MAE={res['mae']:.4f}")
                        else:
                            if is_rate_limited(res["error"]): stop_run = True; break
                        time.sleep(1)

    print(f"\nDone. New runs: {runned}")

if __name__ == "__main__":
    main()
