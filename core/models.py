#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.models — Shared Data Models

Extracted from utils.runner to serve as the single source of truth for
test-status / test-result / batch-report data structures.

Consumers:
  - utils.runner  (legacy AST runner)
  - conftest      (pytest bridge layer)
  - core.resume   (checkpoint / resume status controller)
  - core.results  (result persistence manager)

Design Principles:
  - Pure data structures — no I/O, no side effects, no business logic
  - Both run.py and pytest share the SAME model types
  - Serialisable (to_dict) for disk persistence via core/results.py
  - Language can be switched via environment variable TEST_LANG (zh_CN/en_US)
"""

import os, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# Internationalisation (i18n) Configuration
# ============================================================
_I18N_TRANSLATIONS = {
    "zh_CN": {
        "PENDING": "等待中",
        "RUNNING": "运行中",
        "PASSED": "通过",
        "FAILED": "失败",
        "SKIPPED": "跳过",
        "ERROR": "错误",
        "TIMEOUT": "超时",
    },
    "en_US": {
        "PENDING": "Pending",
        "RUNNING": "Running",
        "PASSED": "Passed",
        "FAILED": "Failed",
        "SKIPPED": "Skipped",
        "ERROR": "Error",
        "TIMEOUT": "Timeout",
    },
}

_DEFAULT_LANG = "en_US"

def _get_i18n_text(status_value: str) -> str:
    """
    Get localized text for a given TestStatus value.
    Falls back to English if translation is missing.
    """
    lang = os.getenv("TEST_LANG", "").strip() or _DEFAULT_LANG
    return _I18N_TRANSLATIONS.get(lang, _I18N_TRANSLATIONS[_DEFAULT_LANG]).get(
        status_value, status_value
    )

# ============================================================
# TestStatus
# ============================================================

class TestStatus(Enum):
    """Canonical test status enumeration used across the entire framework."""

    PENDING  = "PENDING"
    RUNNING  = "RUNNING"
    PASSED   = "PASSED"
    FAILED   = "FAILED"
    SKIPPED  = "SKIPPED"
    ERROR    = "ERROR"
    TIMEOUT  = "TIMEOUT"

    # -- convenience predicates --------------------------------

    @property
    def is_success(self) -> bool:
        return self == TestStatus.PASSED

    @property
    def is_failure(self) -> bool:
        return self in (TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT)

    @property
    def is_skip(self) -> bool:
        return self == TestStatus.SKIPPED

    @classmethod
    def from_pytest_outcome(cls, passed: bool, skipped: bool, failed: bool) -> "TestStatus":
        """Map pytest outcome flags to TestStatus."""
        if skipped:
            return cls.SKIPPED
        if failed:
            return cls.FAILED
        if passed:
            return cls.PASSED
        return cls.ERROR


    @classmethod
    def get_display(cls, status_value: str) -> str:
        """Get localized display text for a status value."""
        return _get_i18n_text(status_value)


# ============================================================
# TestResult  —  single test-case result (in-memory)
# ============================================================

@dataclass
class TestResult:
    """
    Execution result of a single test case (in-memory).

    Used by:
      - run.py        -> printing console summary
      - conftest.py   -> pytest_runtest_makereport hook
      - core/results  -> to_dict() for disk persistence
    """

    module_path: str
    status: TestStatus = TestStatus.PENDING
    duration: float = 0.0
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    retries: int = 0

    def mark_start(self) -> None:
        self.start_time = time.time()
        self.status = TestStatus.RUNNING

    def mark_end(self, status: TestStatus, error: str | None = None, duration: float | None = None) -> None:
        self.end_time = time.time()
        if duration is not None:
            self.duration = duration
        else:
            self.duration = self.end_time - (self.start_time or self.end_time)
        self.status = status
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Dictionary provided for core.results to persist to disk."""
        return {
            "module_path": self.module_path,
            "status": self.status.value,
            "duration": round(self.duration, 3),
            "error": self.error,
            "start_time": self.status.value if self.start_time else None,
            "end_time": self.end_time,
            "retries": self.retries,
        }

    def __str__(self) -> str:
        """ASCII-only display for cross-platform compatibility."""
        status_value = self.status.value
        prefix = {
            "PASSED":   "[PASS]",
            "FAILED":   "[FAIL]",
            "SKIPPED":  "[SKIP]",
            "ERROR":    "[ERROR]",
            "TIMEOUT":  "[TIME]",
            "PENDING":  "[PEND]",
            "RUNNING":  "[RUN ]",
        }.get(status_value, "[????]")

        retry_str = f" (retries={self.retries})" if self.retries else ""
        return f"{prefix} {self.module_path} ({self.duration:.2f}s){retry_str}"

# ============================================================
# BatchReport  —  batch execution summary (in-memory)
# ============================================================

@dataclass
class BatchReport:
    """Batch execution summary report (in-memory)."""

    results: list[TestResult] = field(default_factory=list)
    total_duration: float = 0.0

    def add(self, result: TestResult) -> None:
        self.results.append(result)
        self.total_duration += result.duration

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status.is_success)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status.is_failure)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status.is_skip)

    def summary(self) -> str:
        return (
            f"  Total: {self.total}  |  "
            f"Passed: {self.passed}  |  "
            f"Failed: {self.failed}  |  "
            f"Skipped: {self.skipped}  |  "
            f"Duration: {self.total_duration:.2f}s"
        )

    def __str__(self) -> str:
        lines = [str(result) for result in self.results]
        lines.append("-" * 60)
        lines.append(self.summary())
        return "\n".join(lines)
