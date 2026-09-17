#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.pytest_infra.hooks - pytest hooks (no fixtures)

Registered via pytest11 entry_points in pyproject.toml, automatically effective after business-side pip install.

Hooks
  pytest_addoption              Register command-line options (--csv-output / --resume / --resume-file)
  pytest_configure              Path resolution (single source of truth) / create directories / log file / report files / markers / resume loading
  pytest_runtest_makereport     Capture results -> write outputs/reports/report_<ts>.csv
  pytest_collection_modifyitems Resume: skip completed test cases
  pytest_sessionstart/finish    Session logs / cache record last_csv / summary

Output Contract:
  outputs/logs/      Framework logs
  outputs/reports/   Run-level artifacts (single source of truth resolved in pytest_configure):
    report_<ts>.csv  Default; --csv-output fully overrides (path is used as-is, /dev/null disables writing).
                     If --html is passed and --csv-output is not, CSV follows HTML (same dir & stem, .csv suffix).
                     Note: if --html points to a transient directory, a later --resume may fail to find the CSV and fall back
                     to a full run. Use --resume-file to pin an explicit source, or keep reports under outputs/reports/.
    report_<ts>.html Default when pytest-html is installed and --html is NOT passed; carries run timestamp.
                     If --csv-output=path/my.csv is passed and --html is not, HTML follows CSV (same dir & stem, .html suffix).
                     If --csv-output=/dev/null is passed and --html is not, HTML falls back to framework default.
  outputs/results/   Business-side interface results (managed by business, framework only creates directory but does not write)
"""
from __future__ import annotations

import os, sys, pytest, threading
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from neuraxis_testkit.pytest_infra.models import TestStatus, TestResult
from neuraxis_testkit.pytest_infra.resume import build_completed_keys
from neuraxis_testkit.pytest_infra.paths import NeuraxisPaths, init_paths, get_paths
from neuraxis_testkit.utils.concurrent import FileLock, is_xdist_worker, get_worker_id
from neuraxis_testkit.utils.files import read_csv_to_list, append_to_csv, ensure_dir
from neuraxis_testkit.log.config import force_console_encoding, set_log_file
from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)
_LAST_CSV_KEY = "neuraxis/last_execution_csv"

_csv_locks: dict[str, FileLock] = {}
_locks_guard = threading.Lock()   # Thread-safe within process

def _get_csv_lock(csv_path: Path) -> FileLock:
    """
    Get the corresponding FileLock singleton by CSV path.
    """
    key = str(csv_path)
    with _locks_guard:
        if key not in _csv_locks:
            _csv_locks[key] = FileLock(
                lock_name=f"csv_{csv_path.stem}",   # lock file in system temp dir, competition issue disappears with business CSV same directory deletion
                timeout=30.0
            )
        return _csv_locks[key]


# ============================================================
# Command-line option registration
# ============================================================
def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register framework options for dependency injection.

    Business projects register these same options with concrete defaults in their root conftest.py to inject paths.
    """
    parser.addoption("--csv-output", action="store", default=None, dest="csv_output", help="Override execution-result CSV path"
	                 " (default: outputs/reports/results_<run_ts>.csv; pass /dev/null or nul to disable CSV writing).")
    parser.addoption("--resume", action="store_true", dest="resume", default=False, help="Enable checkpoint resumption: skip completed tests.")
    parser.addoption("--resume-file", action="store", dest="resume_file", default=None, help="Explicit CSV file path for resume.")

    parser.addini(
        "neuraxis_session_label",
        default="Neuraxis TestKit",
        help="Session label for log output.",
    )
    parser.addini(
        "log_file_name_prefix",
        default="neuraxis_testkit",
        help="Base name for log file, can be overridden by business side.",
    )


# ================================================================
# 2. Pytest Configuration / Pytest 配置阶段
# ================================================================
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """
    Register markers and initialize resume logic.
    """
    force_console_encoding()

    root, output, results, logs, reports, run_ts = _resolve_paths(config)
    report_csv, log_file_path = _resolve_report_paths(config, reports, logs, run_ts)


    paths = NeuraxisPaths(
        root=root, output=output, logs=logs, log_file=log_file_path,
        results=results, reports=reports, report_csv=report_csv,
    )
    init_paths(config, paths)

    _load_resume_state(config)
    _register_markers(config)


def _resolve_paths(config: pytest.Config) -> tuple[Path, Path, Path, Path, Path, str]:
    """
    Resolve the single source of truth for all framework paths, create directories, and resolve the run timestamp.
    """
    root    = Path(config.getoption("project_root", default=None) or config.rootpath or Path.cwd())
    output  = Path(config.getoption("output_dir", default=None)   or root / "outputs")
    results = Path(config.getoption("results_dir", default=None)  or output / "results")
    logs    = Path(config.getoption("logs_dir", default=None)     or output / "logs")
    reports = output / "reports"

    # Ensure key directories exist (idempotent operation): create directories immediately upon module load
    for _dir in (output, results, logs, reports):
        ensure_dir(_dir)

    run_ts = _resolve_run_ts()
    return root, output, results, logs, reports, run_ts

def _resolve_report_paths(
    config: pytest.Config,
    reports: Path,
    logs: Path,
    run_ts: str,
) -> tuple[Path, Path]:
    """
    Resolve the CSV report path and the framework log file path, and inject the
    HTML report path into pytest-html's option (when the framework decides it).

    Returns:
        (report_csv, log_file_path) -- HTML path is set on config.option.htmlpath

    Naming contract:
      Neither passed -> reports/report_<ts>.html + reports/report_<ts>.csv (both timestamped)
      --html given   -> <html_path> and <html_path with .csv suffix> (CSV follows HTML, same dir & stem)
      --csv-output   -> <csv_path> and <csv_path with .html suffix> (HTML follows CSV, same dir & stem)
                        Exception: --csv-output=/dev/null disables CSV writing;
                        HTML falls back to framework default reports/report_<ts>.html.
      Both given     -> user HTML path + user CSV path
    """
    csv_override = config.getoption("csv_output", default=None)
    html_override = getattr(config.option, "htmlpath", None)

    if csv_override and html_override:
        # Both given: user takes full control.
        report_csv = Path(csv_override)
        report_html = Path(html_override)
        if not _is_null_path(report_csv):
            ensure_dir(report_csv.parent)
        ensure_dir(report_html.parent)
    elif html_override:
        # Only --html given: CSV follows HTML, same dir & stem.
        report_html = Path(html_override)
        report_csv = report_html.with_suffix(".csv")
        ensure_dir(report_csv.parent)
    elif csv_override:
        # Only --csv-output given: HTML follows CSV, same dir & stem.
        report_csv = Path(csv_override)
        if _is_null_path(report_csv):
            # CSV writing disabled; HTML falls back to framework default.
            report_html = reports / f"report_{run_ts}.html"
        else:
            ensure_dir(report_csv.parent)
            report_html = report_csv.with_suffix(".html")
        _inject_html_path(config, report_html)
    else:
        # Neither given: both timestamped in reports/.
        report_csv = reports / f"report_{run_ts}.csv"
        report_html = reports / f"report_{run_ts}.html"
        ensure_dir(report_csv.parent)
        _inject_html_path(config, report_html)

    log_basename = config.getini("log_file_name_prefix") or "neuraxis_testkit"
    log_file_path = logs / f"{log_basename}_{run_ts[:8]}.log"
    set_log_file(log_file_path)

    return report_csv, log_file_path

def _is_null_path(path: Path) -> bool:
    """Return True for /dev/null, nul, NUL, etc."""
    name = path.name.lower().rstrip(".")
    if name in ("nul", "devnull"):
        return True
    normalized = str(path).replace("\\", "/").lower()
    return normalized.endswith("/dev/null")

def _load_resume_state(config: pytest.Config) -> None:
    """
    Load resume state from --resume-file or the last execution CSV cached by a previous run.
    """
    config._completed_keys = set()
    config._completed_status_map = {}
    if not config.getoption("resume"):
        return

    resume_file = (config.getoption("resume_file") or config.cache.get(_last_csv_key(config), None))
    if not resume_file:
        logger.warning("Resume enabled but no previous CSV found (cache empty). Running ALL tests.")
        return
    if not Path(resume_file).exists():
        logger.warning(f"Resume file not found: {resume_file}. Running ALL tests.")
        return

    try:
        rows = read_csv_to_list(resume_file)
        # Rerun SKIPPED/TIMEOUT/UNKNOWN; skip PASSED/FAILED/ERROR (see resume.build_completed_keys)
        config._completed_keys, config._completed_status_map = build_completed_keys(rows)
        logger.info(f"Resume enabled: skip {len(config._completed_keys)} terminal tests, "
                    f"rerun SKIPPED/TIMEOUT/UNKNOWN. Source: {resume_file}")
    except Exception as exp:
        logger.warning(f"Resume load failed: {exp}, running all tests")

def _register_markers(config: pytest.Config) -> None:
    """
    Register framework markers.
    """
    markers = {
        "slow": "Long-running test",
        "smoke": "Smoke test (core validation)",
        "flaky": "Flaky test requiring retries",
        "freeze": "Skip test",
    }
    for name, desc in markers.items():
        config.addinivalue_line("markers", f"{name}: {desc}")

def _resolve_run_ts() -> str:
    """Run timestamp: controller generates once, workers inherit via env."""
    run_ts = os.environ.get("NEURAXIS_RUN_TS")
    if not run_ts:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["NEURAXIS_RUN_TS"] = run_ts
    return run_ts

def _inject_html_path(config: pytest.Config, html_path: Path) -> None:
    """
    Inject a concrete HTML path into pytest-html's option, creating its parent dir.

    Note: self-contained output is NOT forced; users may pass --self-contained-html to inline CSS and avoid the assets/ directory.
    """
    if not config.pluginmanager.hasplugin("html"):
        return    # pytest-html is not installed; injection is meaningless, skip

    ensure_dir(html_path.parent)
    config.option.htmlpath = str(html_path)


# ============================================================
# 3. Hooks: Result capture / Resume / Session
# ============================================================

# Global session result cache (for sessionfinish use)
_session_results: list = []   # only statistics from call phase (xdist workers each shard)

# Hook: Test result capture
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """
    Capture results from three phases:
      - call:     record all
      - setup:    record only on failure (fixture init failure must not be missed)
      - teardown: record only on failure (exception inside teardown)
    """
    outcome = yield
    report = outcome.get_result()

    # ---- Phase filtering: call all, setup/teardown only failures ----
    if report.when == "call":
        _session_results.append(report)
    elif report.when in ("setup", "teardown") and not report.passed:
        _session_results.append(report)
    else:
        return

    # ---- Status mapping (overrides setup/teardown semantics) ----
    if report.passed:
        status, message = TestStatus.PASSED, ""
    elif report.failed:
        status = TestStatus.FAILED
        message = str(report.longrepr) if report.longrepr else "failed"
        if report.when == "setup":
            status = TestStatus.ERROR          # fixture init failure classified as ERROR
            message = f"[setup] {message}"
        elif report.when == "teardown":
            message = f"[teardown] {message}"
    elif report.skipped:
        status, message = TestStatus.SKIPPED, str(report.longrepr or "skipped")
    else:
        status, message = TestStatus.UNKNOWN, "unknown"

    # Write execution result CSV for this run (suppressed by --csv-output /dev/null)
    nodeid = item.nodeid
    result = TestResult(
        test_id=nodeid,
        module_path=nodeid.split("::")[0],
        func_name=nodeid.split("::")[-1],
        status=status,
        message=message,
        duration=round(report.duration, 3),
        timestamp=datetime.now().isoformat(),
    )
    report_csv_path = get_paths(item.config).report_csv
    _append_result_safe(report_csv_path, result)

def _append_result_safe(csv_path: Path, result: TestResult) -> None:
    """
    Append a single result under lock protection. strict_suffix=False allows paths like /dev/null with no suffix.
    """
    lock = _get_csv_lock(csv_path)
    try:
        with lock.exclusive():
            append_to_csv(csv_path, asdict(result), strict_suffix=False)
    except Exception as exp:
        logger.error(f"Failed to write CSV [{get_worker_id()}]: {exp}")


# Hook: Post-collection modification
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item]
) -> None:
    """
    Skip test items that were completed in the previous run (Resume feature).
    """
    completed: set[str] = getattr(config, "_completed_keys", set())
    if not completed:
        return

    skipped_count = 0
    for item in items:
        if item.nodeid in completed:
            # Add the "skip" marker to prevent execution
            item.add_marker(pytest.mark.skip(
                reason="Already passed in previous run"
            ))
            skipped_count += 1

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} tests due to resume")

@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    """
    Session start: Initialization.
    Session 开始: 初始化.
    """
    config_path = get_paths(session.config)
    label = session.config.getini("neuraxis_session_label") or "Neuraxis TestKit"

    logger.info("=" * 60)
    logger.info(f"{label} Session Started")
    logger.info(f"Python: {sys.version}")
    #logger.info(f"pytest: {pytest.__version__}")
    logger.info(f"Project Root: {config_path.root}")
    logger.info(f"Project Report: {config_path.report_csv}")
    logger.info(f"Project LogFile: {config_path.log_file}")
    logger.info("=" * 60)

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """
    Session end: Summary report.
    """
    if not is_xdist_worker():
        FileLock.cleanup_locks()
        report_csv = get_paths(session.config).report_csv
        if report_csv.suffix == ".csv":     # do not cache when overridden to /dev/null/nul
            session.config.cache.set(_LAST_CSV_KEY, str(report_csv))

    total = len(_session_results)
    passed = sum(1 for result in _session_results if result.passed)
    failed = sum(1 for result in _session_results if result.failed)
    skipped = sum(1 for result in _session_results if result.skipped)
    label = session.config.getini("neuraxis_session_label") or "Neuraxis TestKit"

    logger.info("=" * 60)
    logger.info(f"{label} Session Finished")
    logger.info(f"Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    logger.info(f"Exit Status: {exitstatus}")
    logger.info("=" * 60)
