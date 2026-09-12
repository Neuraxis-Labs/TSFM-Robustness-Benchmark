"""
core — Business Core Layer

Provides centralized management for business logic, state, and external interactions.
Serves as the bridge between the ``testcases`` layer and ``utils`` layer.

Modules:
-------
client.py — TimechoAI Client Connection
  Provides factory functions get_timecho_client() / get_timecho_async_client(),
  unifying the creation and lifecycle management of TimechoAIClient / TimechoAIAsyncClient instances.
  NOT exposed in __all__. Used by timecho.py for client operation safety.

metrics.py — Evaluation Metrics Calculator
  Provides standard evaluation metrics (MAE, RMSE, MAPE) for time series forecasting models.
  Pure mathematical calculation functions without side effects.

models.py — Shared Data Models
  Defines core data structures (TestStatus, TestResult, BatchReport) used across
  the entire framework. Supports i18n via TEST_LANG environment variable.

results.py — Test Result Manager
  Manages result persistence (batch buffering), historical loading, and querying.
  Internally uses neuraxis_testkit.utils.concurrent for thread-safe file operations.

resume.py — Strategy Controller
  Provides checkpoint resumption logic and rate limit detection strategy.

timecho.py — TimechoAI Interaction Layer
  Encapsulates API requests and response handling, offering a unified high-level API.

Usage Examples:
--------------
1. Business modules (e.g., testcases/) should access TimechoAI services indirectly through core.timecho.
2. The core layer is the only module that directly uses utils.client.

>>> # Evaluation metrics calculator
>>> from core.metrics import calc_metrics, calc_diff, evaluate_prediction

>>> # Data models
>>> from core.models import ForecastResult, BatchForecastReport

>>> # Result management (with auto buffering)
>>> from core.results import load_results_from_csv, append_result_to_csv, flush_all_results
>>> records, fails = load_results_from_csv("./results/test.csv")
>>> append_result_to_csv("./results/test.csv", {"mae": 0.5})  # Auto buffered
>>> flush_all_results()  # Must call before exit

>>> # Strategy control
>>> from core.resume import is_rate_limited, should_skip_test, build_completed_keys, get_last_failed_from_cache
>>> if is_rate_limited("Error 429"):
...     print("Rate limit detected")

>>> # Recommended: Access via core layer
>>> from core.timecho import forecast
>>> forecast(data)
"""

__all__ = [
    "metrics",
    "models",
    "results",
    "resume",
    "timecho",
]

