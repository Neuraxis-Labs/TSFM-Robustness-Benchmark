
"""
neuraxis_testkit/pytest_infra - Neuraxis Pytest Infrastructure Layer
====================================

Modules:
--------
manifest_loader.py 


models.py —— Shared Data Models
  Defines core data structures (TestStatus, TestResult, BatchReport) used across
  the entire framework. Supports i18n via TEST_LANG environment variable.

Import Path Examples:
-----------------------------
>>> # Data models
>>> from neuraxis_testkit.pytest_infra import TestStatus, TestResult, BatchReport
>>> result = TestResult(module_path="test_xxx.py")
>>> result.mark_start()
>>> result.mark_end(TestStatus.PASSED)
"""

from .models import BatchReport, TestResult, TestStatus

__all__ = [
    "collection",
    "conftest",
    "manifest_loader",
    "models",
    "paths",
    "resume",
    "session_manager",
    "test_helpers",
    "test_recorder",
    # Core
    "TestStatus",
    "TestResult",
    "BatchReport",
]

_submodules = {
    "collection", "conftest", "manifest_loader", 
    "models", "resume", "session_manager"
}

def __getattr__(name):
    if name in _submodules:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(__all__)