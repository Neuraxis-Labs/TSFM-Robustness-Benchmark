
"""
neuraxis_testkit.pytest_infra - Pytest Infrastructure Layer for Neuraxis
========================================================================
This package is the plugin-level infrastructure layer that sits on top of pytest
and turns a plain pytest suite into a resumable, dynamically collected,
session-aware test framework.

Four core capabilities:

1. Dynamic collection
   YAML manifests are loaded at collection time and expanded into parametrized
   test cases, so a test suite can be declared as data instead of code.

2. Resume-from-breakpoint
   Historical results are read back so a re-run can skip already-passed cases
   and continue from the last failure — essential for long-running suites.

3. Session management
   Shared resources, cross-process file locks (SessionFileLock), and session
   lifecycle (SessionManager) keep parallel and sequential runs consistent.

4. Shared fixtures & hooks
   A curated set of @pytest.fixture definitions and hookimpl implementations
   (including pytest_addoption / pytest_configure) wire the above capabilities
   into pytest with zero boilerplate.

Design goals:
    - Test authors write tests; this layer handles collection, orchestration
      and state.
    - Submodules are lazily imported, so importing the package is cheap even
      when only one capability is needed.

Modules:
    - collection       : Dynamic collection (calls manifest_loader)
    - fixtures         : All @pytest.fixture definitions
    - hooks            : All hookimpl (incl. pytest_addoption / pytest_configure)
    - manifest_loader  : Loads YAML manifests and generates parametrization
    - models           : Test-specific data models (e.g. test case parameters)
    - paths            : NeuraxisPaths + get_paths (pure data + accessors)
    - resume           : Resume-from-breakpoint logic (based on historical results)
    - session_manager  : Test session management (shared resources, locks);
                         SessionFileLock + SessionManager
    - test_helpers     : pytest test-process helpers (glue layer)
    - test_recorder    : Test result recorder

Import Path Examples:
    >>> from neuraxis_testkit.pytest_infra import TestStatus, TestResult, BatchReport
    >>> result = TestResult(module_path="test_xxx.py")
    >>> result.mark_start()
    >>> result.mark_end(TestStatus.PASSED)
"""

from .models import BatchReport, TestResult, TestStatus

# Public API
_SUBMODULES = (
    "collection",
    "fixtures",
    "hooks",
    "manifest_loader",
    "models",
    "paths",
    "resume",
    "session_manager",
    "test_helpers",
    "test_recorder",
)

__all__ = [
    # Submodules
    *_SUBMODULES,
    # Core
    "TestStatus",
    "TestResult",
    "BatchReport",
]


def __getattr__(name: str):
    if name in _SUBMODULES:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Support dir() completion
def __dir__():
    return sorted(__all__)