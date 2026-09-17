
"""
Usage Examples:
  pytest tests/

  pytest tests/ -k dirty
  pytest tests/ -k dirty -v

  pytest tests/ -m dirty
  pytest tests/ -m "concept_drift and not slow"

  pytest tests.futureCovs.dirtyData.test_dirty

  pytest tests.futureCovs.dirtyData.test_dirty::test_dirty_basic

  pytest tests/ -n auto

  pytest tests/ --reruns 3 --reruns-delay 5

  pytest tests/ --junitxml=outputs/reports/report.xml

  pytest tests/ --lf
  pytest tests/xxx --resume --resume-file "outputs/reports/report-<run-ts>.csv"

  pytest tests/ -s --pdb

  python -m pytest --collect-only -q --output-dir=/tmp/x
"""

import pytest
from config.settings import PROJECT_ROOT, OUTPUT_DIR, LOGS_DIR, RESULTS_DIR


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--project-root", action="store", dest="project_root", default=str(PROJECT_ROOT), help="项目根目录")
    parser.addoption("--output-dir", action="store", dest="output_dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.addoption("--results-dir", action="store", dest="results_dir", default=str(RESULTS_DIR), help="结果目录")
    parser.addoption("--logs-dir", action="store", dest="logs_dir", default=str(LOGS_DIR), help="日志目录")

def pytest_report_header(config):
    """
    Custom report header.
    """

    return [
        "TSFM Time Series Forecasting Benchmark Framework (pytest native mode)",
    ]
