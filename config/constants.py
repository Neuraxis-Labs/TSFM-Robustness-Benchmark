"""
config.constants - Global Constants Definition

Configure global constants, including model and time-series large model parameters.

Author: Janesong
Create Date: 2026/07/19
"""

# ============================================================
# Default Model Parameters
# ============================================================
DEFAULT_MODEL_ID = "Timer-3.5"
MODEL_LIST = [
    "Auto",          # Automatic selection
    "Timer-3.5",     # Timecho flagship model
    "Timer-3.0",     # Previous generation
    "Chronos-2",     # Developed by Amazon
    "AutoARIMA",     # Statistical method, AutoRegressive Integrated Moving Average
    "Holt-Winters",  # Statistical method, Holt-Winters model / Triple exponential smoothing
]

# ============================================================
# Time Series Large Model Parameters
# ============================================================
# Forecasting Parameters
HISTORY_POINT_LEN_128 = 128     # 128 historical points
HISTORY_POINT_LEN_256 = 256
HISTORY_POINT_LEN_512 = 512

FORECAST_POINT_LEN_64 = 64      # Predict 64 future points
FORECAST_POINT_LEN_128 = 128
FORECAST_POINT_LEN_256 = 256
FORECAST_POINT_LEN_512 = 512

# Training Parameters
TRAIN_SEQ_LEN_128 = 128      # Training segment length
TRAIN_SEQ_LEN_256 = 256
TRAIN_SEQ_LEN_512 = 512

TRAIN_PERIOD_HOUR_24 = 24    # Training period (hours)

CONTEXT_LENGTH_256 = 256     # Context window length
CONTEXT_LENGTH_512 = 512

# Drift Parameters
DRIFT_MEAN_SHIFT_15 = 15         # Magnitude of mean shift
DRIFT_NOISE_MULTIPLIER_3 = 3     # Noise variance multiplier

DRIFT_RAMP_LEN_64 = 64            # Length of drift transition zone

# Signal Parameters
SIGNAL_TREND_AMP_15 = 15          # Trend amplitude
SIGNAL_SEASONAL_AMP_15 = 15       # Seasonal amplitude
SIGNAL_NOISE_STD_15 = 2           # Standard deviation of noise
SIGNAL_BASE_VALUE_50 = 50         # Signal baseline value
