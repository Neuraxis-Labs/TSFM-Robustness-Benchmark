"""
core.timecho - TimechoAI CRUD Wrapper

Provides a general calling layer for Timecho prediction interface, including:
  - forecast(): Wraps API call, timing, and exception handling
  - extract_pred_values(): Extracts prediction values from API-returned result DataFrame

Author: Janesong
Create Date: 2026/07/10.
"""

# ============================================================
import aiohttp

_original_aiohttp_request = aiohttp.ClientSession._request

async def _hooked_aiohttp_request(self, method, url, **kwargs):
    """Intercept aiohttp requests to capture full response headers for 429"""
    resp = await _original_aiohttp_request(self, method, url, **kwargs)
    if resp.status == 429:
        print("\n" + "=" * 60)
        print("[429 Interceptor] Caught Too Many Requests (aiohttp)")
        print(f"  URL: {url}")
        print(f"  Status Code: {resp.status}")
        print(f"  Retry-After: {resp.headers.get('Retry-After', 'Not returned')}")
        print(f"  X-RateLimit-Remaining: {resp.headers.get('X-RateLimit-Remaining', 'Not returned')}")
        print(f"  X-RateLimit-Reset: {resp.headers.get('X-RateLimit-Reset', 'Not returned')}")
        print(f"  All Response Headers:")
        for k, v in resp.headers.items():
            print(f"    {k}: {v}")
        print("=" * 60 + "\n")
    return resp

aiohttp.ClientSession._request = _hooked_aiohttp_request
# ============================================================

import time

import numpy as np
import pandas as pd

from core.client import get_timecho_client

# ============================================================
# Prediction Value Extraction
# ============================================================

def extract_pred_values(pred_df: pd.DataFrame) -> np.ndarray:
    """
    Extract numeric columns from prediction result DataFrame (excluding time column).

    Args:
        pred_df: Prediction result DataFrame returned by API

    Returns:
        numpy array of float type
    """
    if "target" in pred_df.columns:
        return pred_df["target"].values.astype(float)
    non_time_cols = [c for c in pred_df.columns if c != "time"]
    return pred_df[non_time_cols[0]].values.astype(float)


# ============================================================
# Prediction Call (Core Wrapper)
# ============================================================

def forecast(
    *,
    targets: pd.DataFrame,
    history_covs: pd.DataFrame | None = None,
    future_covs: pd.DataFrame | None = None,
    model_id: str = "Holt-Winters",
    output_length: int = 64,
    time_col: str = "time",
    auto_adapt: bool = True,
    api_key: str | None = None,
) -> tuple[np.ndarray | None, float, str | None]:
    """
    Call TimechoAI prediction interface and return prediction values, elapsed time, and error message.

    Args:
        targets: Historical target values DataFrame (must contain time and target columns)
        history_covs: Historical covariates DataFrame (optional)
        future_covs: Future covariates DataFrame (optional, pass None to indicate no covariates)
        model_id: Model ID
        output_length: Prediction length
        time_col: Time column name
        auto_adapt: Whether to auto-adapt
        api_key: API key (optional, uses global configuration by default)

    Returns:
        (pred_values, elapsed_ms, error_msg)
        - pred_values: Prediction value array (None on failure)
        - elapsed_ms: Elapsed time (milliseconds)
        - error_msg: Error message (None on success)
    """
    client = get_timecho_client(api_key)
    t0 = time.perf_counter()

    try:
        # Only pass to API when covariates are not None, avoiding errors from certain models due to None values
        api_kwargs: dict = {
            "targets": targets,
            "model_id": model_id,
            "output_length": output_length,
            "time_col": time_col,
            "auto_adapt": auto_adapt,
        }
        if history_covs is not None:
            api_kwargs["history_covs"] = history_covs
        if future_covs is not None:
            api_kwargs["future_covs"] = future_covs

        result = client.forecast(**api_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        pred_values = extract_pred_values(result[0])
        return pred_values, elapsed_ms, None
    except Exception as exp:
        if hasattr(exp, 'response'):
            resp = exp.response
            print(f"Status Code: {resp.status_code}")
            print(f"Retry-After: {resp.headers.get('Retry-After', 'Not returned')}")
        else:
            print(f"Exception Type: {type(exp)}")
            # print(f"Exception Attributes: {dir(e)}")
            print(f"Exception Message: {exp}")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return None, elapsed_ms, str(exp)

