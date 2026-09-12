#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.utils.runner — Test Runner Core
Core runner: test discovery + single-case execution + result tracking (in-memory)

Design Principles:
  - This module is the [Execution Capability Layer], no CLI / argparse / print
  - Only responsible for discovering test cases and executing their main() functions
  - Both run.py and conftest.py reuse the capabilities of this module
  - Batch scheduling is delegated to pytest, but the primitives for discovery / execution / result are defined here
"""

import os, sys, ast, time, importlib, traceback
import multiprocessing as mp
from pathlib import Path
from queue import Empty

# Bootstrap (defensive)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import PROJECT_ROOT
from neuraxis_testkit.log import get_logger
from core.models import TestStatus, TestResult  # noqa: F401


# ============================================================
# 1. Test Discoverer — AST static analysis, zero side effects
# ============================================================

class TestDiscoverer:
    """
    Discovers test modules via AST static analysis.
    Never imports modules — zero side effects.

    Usage Examples:
      - run.py --list: List all tests
      - conftest.py: Can replace pytest_collect_file for pre-filtering
      - core.resume: Determine the full scope for checkpoint recovery
    """

    ENTRY_POINTS = ("main", "run", "start")
    TEST_PATTERNS = ("test_*.py", "*_test.py")

    def __init__(self, logger=None):
        self.logger = logger or get_logger("discoverer")
        self.testcases_root = PROJECT_ROOT / "testcases"

    def discover(
        self,
        directory: Path | None = None,
        tags: list[str] | None = None,
        skip: list[str] | None = None,
    ) -> list[str]:
        """
        Discover all test modules and return a list of module paths.
        """
        search_root = Path(directory) if directory else self.testcases_root

        if not search_root.exists():
            self.logger.warning(f"Search directory does not exist: {search_root}")
            return []

        skip = skip or []
        discovered = []

        for py_file in sorted(search_root.rglob("*.py")):
            if not self._is_test_file(py_file):
                continue

            module_path = self._file_to_module_path(py_file)
            if module_path is None:
                continue
            if module_path in skip:
                self.logger.info(f"Skipped (--skip): {module_path}")
                continue
            if tags and not self._match_tags(module_path, tags):
                continue
            if not self._has_entry_point_ast(py_file):
                self.logger.debug(f"Skipped (no entry function): {module_path}")
                continue

            discovered.append(module_path)
            self.logger.debug(f"Discovered: {module_path}")

        self.logger.info(f"Total {len(discovered)} test module(s) discovered")
        return discovered

    def _is_test_file(self, py_file: Path) -> bool:
        import fnmatch

        name = py_file.name
        if name.startswith("_"):
            return False

        return any(fnmatch.fnmatchcase(name, p) for p in self.TEST_PATTERNS)

    def _file_to_module_path(self, py_file: Path) -> str | None:
        try:
            rel = py_file.resolve().relative_to(PROJECT_ROOT)
            if rel.suffix == ".py":
                rel = rel.with_suffix("")
            return str(rel).replace("/", ".").replace(os.sep, ".")
        except ValueError:
            return None

    def _match_tags(self, module_path: str, tags: list[str]) -> bool:
        parts = module_path.replace(".", "/").split("/")
        return any(tag in parts or tag in module_path for tag in tags)

    def _has_entry_point_ast(self, py_file: Path) -> bool:
        """
        AST static analysis: check whether the file defines an entry function.
        Never imports the module to avoid triggering top-level side effects.
        """
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            top_funcs = {
                node.name for node in ast.iter_child_nodes(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            return any(ep in top_funcs for ep in self.ENTRY_POINTS)
        except SyntaxError as synExp:
            self.logger.error(f"Syntax error in {py_file}: {synExp}")
            return False
        except Exception as exp:
            self.logger.error(f"AST analysis failed for {py_file}: {exp}")
            return False


# ============================================================
# 2. Test Runner — single-case execution with timeout + retry
# ============================================================

class TimeoutError_(Exception):
    """Test execution timed out"""
    pass


class TestRunner:
    """
    Test runner: imports and invokes the entry function of a single test module.
    Entry function priority: main() > run() > start()
    """

    ENTRY_POINTS = ("main", "run", "start")

    def __init__(self, logger=None, default_timeout: int = 0, default_retries: int = 0):
        self.logger = logger or get_logger("runner")
        self.default_timeout = default_timeout
        self.default_retries = default_retries

    def run_single(self, module_path, timeout=None, retries=None):
        timeout = timeout if timeout is not None else self.default_timeout
        retries = retries if retries is not None else self.default_retries
        result = TestResult(module_path=module_path)

        # First, get the entry function name via AST (zero side effects)
        entry_name = self._find_entry_name_ast(module_path)
        if entry_name is None:
            result.mark_end(TestStatus.SKIPPED, "No entry function found")
            self.logger.info(str(result))
            return result

        attempt = 0
        max_attempts = retries + 1

        while attempt < max_attempts:
            attempt += 1
            result.retries = attempt - 1
            result.mark_start()

            try:
                if timeout > 0:
                    # Timeout mode: main process does not import, delegates directly to subprocess
                    actual_duration = self._run_with_timeout(module_path, entry_name, timeout)
                    result.end_time = time.time()
                    result.duration = actual_duration
                    result.status = TestStatus.PASSED
                else:
                    # Normal mode: main process imports and executes
                    module = importlib.import_module(module_path)
                    func = getattr(module, entry_name)
                    func()

                result.mark_end(TestStatus.PASSED)
                self.logger.info(str(result))
                return result

            except TimeoutError_ as timeExp:
                result.mark_end(TestStatus.TIMEOUT, str(timeExp))
                if attempt < max_attempts:
                    self.logger.error(str(result))
                    continue
                self.logger.error(str(result))
                return result

            except Exception as exp:
                error_detail = traceback.format_exc()
                if attempt < max_attempts:
                    self.logger.error(str(result))
                    self.logger.debug(error_detail)
                    continue
                result.mark_end(TestStatus.FAILED, str(exp))
                self.logger.error(str(result))
                self.logger.debug(error_detail)
                return result
        return result


    def _find_entry_name_ast(self, module_path: str) -> str | None:
        """
        AST static analysis: find the entry function name.
        Does not import the module to avoid triggering top-level side effects.

        Returns: Entry function name (e.g., "main") or None (if not found)
        """
        # module_path -> file path
        parts = module_path.replace(".", "/")
        py_file = PROJECT_ROOT / f"{parts}.py"

        if not py_file.exists():
            self.logger.warning(f"File not found: {py_file}")
            return None

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            top_funcs = {
                node.name for node in ast.iter_child_nodes(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            # Return the first match based on priority
            for ep in self.ENTRY_POINTS:
                if ep in top_funcs:
                    return ep
            return None
        except SyntaxError as synExp:
            self.logger.warning(f"Syntax error in {py_file}: {synExp}")
            return None
        except Exception as exp:
            self.logger.warning(f"AST analysis failed for {py_file}: {exp}")
            return None


    def _run_with_timeout(self, module_path: str, entry_name: str, timeout: int):
        """
        Implements timeout control using multiprocessing.Process (cross-platform).
        The process is forcibly terminated on timeout, simulating the hard interrupt effect of the original SIGALRM.
        """
        result_queue = mp.Queue()
        process = mp.Process(target=self._wrap_func, args=(module_path, entry_name, result_queue))
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError_(f"Execution timed out ({timeout}s)")

        if process.exitcode != 0:
            raise RuntimeError(
                f"Subprocess crashed (exitcode={process.exitcode})"
            )

        try:
            result = result_queue.get_nowait()
            if isinstance(result, dict):
                if result.get("status") == "success" and "duration" in result:
                    return result["duration"]
                elif "type" in result:
                    raise RuntimeError(
                        f"{result['type']}: {result['message']}\n"
                        f"{result['traceback']}"
                    )
        except Empty:
            pass
        finally:
            result_queue.close()
            result_queue.join_thread()
        return None

    @staticmethod
    def _wrap_func(module_path: str, entry_name: str, queue):
        try:
            if str(_PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(_PROJECT_ROOT))

            start_time = time.time()
            module = importlib.import_module(module_path)
            func = getattr(module, entry_name)
            func()
            duration = time.time() - start_time

            queue.put({"duration": duration, "status": "success"})
        except Exception as exp:
            queue.put({
                "type": type(exp).__name__,
                "message": str(exp),
                "traceback": traceback.format_exc()
            })  # Pass exception details to the main process


# ============================================================
# 3. Path Resolution Utility
# ============================================================

def parse_module_path(raw_path: str) -> str:
    """
    Resolve a user-provided path into a standard module path.
    Supports:
      - testcases.futureCovs.dirtyData.test_dirty
      - ./testcases/futureCovs/dirtyData/test_dirty.py
      - testcases/futureCovs/dirtyData/test_dirty.py
    """
    if raw_path.startswith("./") or raw_path.endswith(".py") or "/" in raw_path:
        file_path = Path(raw_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        rel_path = file_path.relative_to(PROJECT_ROOT)
        if rel_path.suffix == ".py":
            rel_path = rel_path.with_suffix("")
        module_path = str(rel_path).replace("/", ".").replace(os.sep, ".")
    else:
        module_path = raw_path
    return module_path
