#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit/pytest_infra/models.py - Shared Data Models
"""
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field


class TestStatus(Enum):
    """Test status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class TestResult:
    """Single test result."""
    test_id: str          # pytest nodeid or module path
    module_path: str      # test file path
    func_name: str        # test function name
    status: TestStatus
    message: str = ""     # failure message / skip reason
    duration: float = 0.0 # execution time (seconds)
    timestamp: str = ""   # ISO format timestamp

    @property
    def is_passed(self) -> bool:
        return self.status == TestStatus.PASSED

    @property
    def is_failed(self) -> bool:
        return self.status in (TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT)

    @property
    def is_skipped(self) -> bool:
        return self.status == TestStatus.SKIPPED

    def __post_init__(self):
        """Validation/fallback hook automatically called after dataclass initialization."""
        if self.duration < 0:
            raise ValueError(f"duration cannot be negative: {self.duration}")
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class BatchReport:
    """Batch test report."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    results: list[TestResult] = field(default_factory=list)
    timestamp: str = ""

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    def add_result(self, result: TestResult):
        self.results.append(result)
        self.total += 1
        if result.status == TestStatus.PASSED:
            self.passed += 1
        elif result.status == TestStatus.ERROR:
            self.errors += 1
        elif result.status in (TestStatus.FAILED, TestStatus.TIMEOUT):
            self.failed += 1
        elif result.status == TestStatus.SKIPPED:
            self.skipped += 1
        else:
            self.errors += 1

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration": f"{self.duration:.3f}",
            "pass_rate": f"{self.pass_rate:.1f}%",
            "timestamp": self.timestamp,
        }
