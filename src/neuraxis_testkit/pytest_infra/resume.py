#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.pytest_infra.resume - pytest checkpoint resumption based on historical test result CSV

Responsibility boundary:
  - This file: Resumption at the pytest test flow level (skip tests that already PASSED in the previous run).

Design notes:
  1. Completed statuses include PASSED, FAILED, and ERROR (i.e., terminal statuses). Tests with these statuses are considered "completed" and will be skipped on resume.
     All other statuses (TIMEOUT, SKIPPED, etc.) will be rerun.
  2. Uses pytest nodeid as the primary key for exact matching; parameterized tests are naturally distinguished.
  3. Supports resume scope control via --resume-scope (all / failed_only) – future extension.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)

# Statuses that need rerun (any status not in completed set will be rerun; documented here for semantics)
# RERUN_STATUSES = {"failed", "error", "timeout", "skipped"}
TERMINAL_STATUSES = frozenset({"PASSED", "FAILED", "ERROR"})

def build_completed_keys(
    rows: list[dict[str, Any]],
    terminal_statuses: frozenset[str] = TERMINAL_STATUSES,
) -> tuple[set[str], set[str]]:
    """
    Build a set of completed nodeids from historical result rows (status in terminal_statuses).

    When the same test_id appears multiple times, the last row (most recent) status is used for determination —
    supports scenarios where a fixed --csv-output is appended across runs.

    Returns: (completed_set, empty_set) — second element reserved for future use.
    """

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = str(row.get("test_id", "")).strip()
        if tid:
            # later rows override earlier ones -> keep only the latest record per nodeid
            latest[tid] = row

    completed = {
        tid for tid, row in latest.items()
        if str(row.get("status", "")).strip().upper() in terminal_statuses
    }
    # keep binary tuple signature for backward compatibility
    return completed, set()


def should_skip_test(nodeid: str, completed: set[str]) -> bool:
    """Determine whether the test case should be skipped due to resumption."""
    return nodeid in completed


def filter_items_for_resume(items, completed: set[str]) -> int:
    """
    Mark tests for skipping directly during collection_modifyitems (more efficient than skipping at setup stage,
    because under xdist, distribution occurs at collection time, avoiding occupation of worker execution slots).

    Returns: Number of tests marked for skip
    """
    count = 0
    for item in items:
        if should_skip_test(item.nodeid, completed):
            # depends on --strict-markers registration; see conftest
            item.add_marker("skip")
            count += 1
    return count


def resolve_resume_file(cli_value: str | None, cache) -> Path | None:
    """Prefer command-line --resume-file, otherwise use the last path recorded in pytest cache."""
    if cli_value:
        p = Path(cli_value)
        return p if p.exists() else None
    if cache is not None:
        last = cache.get("tsfm/last_csv_path", None)
        if last:
            p = Path(last)
            return p if p.exists() else None
    return None


def remember_csv_path(cache, csv_path: Path) -> None:
    """Record the CSV path to cache after this run, for use by the next --resume."""
    if cache is not None:
        cache.set("tsfm/last_csv_path", str(csv_path))
