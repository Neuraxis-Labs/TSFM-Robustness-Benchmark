#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSFM-Robustness-Benchmark - Unified Entry Point

Usage Examples:
  # Single test case
  python run.py testcases.futureCovs.dirtyData.test_dirty

  # File path is also acceptable
  python run.py ./testcases/futureCovs/dirtyData/test_dirty.py

  # To run all cases in batch, please use pytest

Environment Variable Options:
  LOG_LEVEL=DEBUG              Set log level
  LOG_CONSOLE_OUTPUT=false     Disable console output
  LOG_FILE_OUTPUT=false        Disable file output
  LOG_MAX_BYTES=52428800       Set log file size limit (default 50MB)
"""

import os, sys, logging, argparse
import traceback
from pathlib import Path

# Bootstrap: Allow Python to find packages under the project root directory
_BOOTSTRAP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from config.settings import PROJECT_ROOT
from neuraxis_testkit.log import get_logger, flush_all_logs
from neuraxis_testkit.log.config import get_log_file_path, get_log_level, force_console_encoding
from neuraxis_testkit.utils.runner import TestRunner, parse_module_path

force_console_encoding()

def main():
    parser = argparse.ArgumentParser(
        description="TSFM-Robustness-Benchmark Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
For batch execution, please use pytest:
  pytest testcases/                  # Run all tests
  pytest testcases/ -k test_dirty    # Filter by name
  pytest testcases/ -v               # Verbose output
""",
    )

    parser.add_argument(
        "module",
        help="Test module path (e.g., testcases.futureCovs.dirtyData.test_dirty)"
    )
    args = parser.parse_args()

    # ── Initialize logging ──
    logger = get_logger("run")
    log_file_path = get_log_file_path()

    # Log level: int -> readable name
    level_int = get_log_level()
    level_name = logging.getLevelName(level_int)

    logger.info("=" * 70)
    logger.info("Project started")
    logger.info("=" * 70)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Log level: {level_name} ({level_int})")

    # ── Path resolution ──
    try:
        module_path = parse_module_path(args.module)
        logger.info(f"Started executing module: {module_path}")
    except FileNotFoundError as exp:
        logger.error(str(exp))
        sys.exit(1)

    # ── Execution ──
    runner = TestRunner(logger=logger)

    try:
        runner.run_single(module_path)
    except Exception as exp:
        error_detail = traceback.format_exc()
        logger.error(f"Execution failed:\n{error_detail}")
        sys.exit(1)

    finally:
        flush_all_logs()
        logger.info("=" * 70)
        logger.info("Project finished")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
