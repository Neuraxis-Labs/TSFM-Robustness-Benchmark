#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.pytest_infra.test_recorder - Test result recorder
"""
import time
from datetime import datetime
from typing import Any
from dataclasses import asdict
from neuraxis_testkit.pytest_infra.models import TestStatus, TestResult
from neuraxis_testkit.utils.files import append_to_csv

class TestResultRecorder:
    """
    Test result recorder: Writes pytest test results to CSV.
    """

    def __init__(self, csv_path, lock, logger, no_csv=False):
        """
        Initialize result recorder.

        Args:
            csv_path: CSV file path
            lock: File lock object
            logger: Logger
            no_csv: Whether to disable CSV writing
        """
        self.csv_path = csv_path
        self.lock = lock
        self.logger = logger
        self.no_csv = no_csv
        self._result = None

    def set_result(self, status: TestStatus, message: str = "", extra: dict[str, Any] | None = None) -> None:
        """
        Set current test result.

        Args:
            status: Test status
            message: Result message
            extra: Extra data
        """
        self._result = {
            "status": status,
            "message": message,
            "extra": extra or {},
        }

    def finalize(self, request):
        """
        Called at test end, writes to CSV.

        Args:
            request: pytest request object
        """
        if self.no_csv or self._result is None:
            return

        nodeid = request.node.nodeid
        module_path = nodeid.split("::")[0] if "::" in nodeid else nodeid
        func_name = nodeid.split("::")[-1] if "::" in nodeid else ""

        result = TestResult(
            test_id=nodeid,
            module_path=module_path,
            func_name=func_name,
            status=self._result["status"],
            message=self._result.get("message", ""),
            duration=time.time(),
            timestamp=datetime.now().isoformat(),
        )

        try:
            append_to_csv(self.csv_path, asdict(result))
        except Exception as exp:
            self.logger.error(f"Failed to write CSV: {exp}")

