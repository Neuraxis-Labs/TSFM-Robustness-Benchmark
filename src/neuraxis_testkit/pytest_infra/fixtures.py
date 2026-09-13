#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.pytest_infra.fixtures - Framework-level fixtures (no hooks)

Key Fixtures:
  session:
    - test_logger:                 Unified logger
    - output_dir / results_dir:    Output directory
    - report_csv_path:             CSV path for this run (read-only)
  module:
    - entry_module:                Reference to current test module
  function:
    - test_runner:                 TestRunner instance (timeout/retry)
    - result_recorder:             Result recorder (teardown auto-writes run-level CSV)
"""
from __future__ import annotations

import pytest
from pathlib import Path
from neuraxis_testkit.pytest_infra.test_recorder import TestResultRecorder
from neuraxis_testkit.pytest_infra.paths import get_paths
from neuraxis_testkit.utils.runner import TestRunner
from neuraxis_testkit.log import get_logger

# ============================================================
# Provide Session-level Fixtures
# ============================================================
@pytest.fixture(scope="session")
def test_logger():
    """
    Session-level logger.
    """
    return get_logger("test_session")


@pytest.fixture(scope="session")
def output_dir(request: pytest.FixtureRequest) -> Path:
    """
    Get the output directory resolved from CLI options.
    """
    return get_paths(request.config).output

@pytest.fixture(scope="session")
def results_dir(request: pytest.FixtureRequest) -> Path:
    """
    Get the results sub-directory under the output directory.
    """
    return get_paths(request.config).results

@pytest.fixture(scope="session")
def report_csv_path(request) -> Path:
    """
    CSV path for the execution results of this run (read-only, same file as written by hook).
    """
    return get_paths(request.config).report_csv


# ============================================================
# Provide Module-level Fixtures
# ============================================================
@pytest.fixture(scope="module")
def entry_module(request):
    """Reference to current test module."""
    return request.module


# ============================================================
# Provide Function-level Fixtures
# ============================================================
@pytest.fixture(scope="function")
def test_runner(request):
    """
    TestRunner instance (function-level).

    Provides execution-enhancement primitives for callables invoked inside a test case:
      - run_with_timeout(): process-level timeout (hard kill across Win/Linux/macOS)
      - run_with_retry():   retry with optional per-attempt timeout

    Out of scope (handled by pytest / framework, do NOT go through this fixture):
      - test discovery / execution scheduling / result determination / reporting
      - per-case timeout and per-case retry (use pytest-timeout / pytest-rerunfailures)

    Result tracking is handled automatically by test_recorder; no need to manipulate it through this fixture.
    """
    timeout_override = request.config.getoption("--test-timeout")
    return TestRunner(
        default_timeout=timeout_override if timeout_override and timeout_override > 0 else 0,
        default_retries=0,
    )


@pytest.fixture(scope="function")
def result_recorder(request, test_logger):
    """
    Result recorder — Automatically writes to CSV at test end.

    As a yield fixture, automatically records results during teardown.
    """
    recorder = TestResultRecorder(
        csv_path=get_paths(request.config).report_csv,
        lock=None,
        logger=test_logger,
    )
    yield recorder
    recorder.finalize(request)
