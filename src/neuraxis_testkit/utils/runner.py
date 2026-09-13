#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit.utils.runner - Test Runner Core (Lite Edition)

Test functions can invoke via the `test_runner` fixture:
  - run_with_timeout(): process-level timeout execution
  - run_with_retry():   retry execution (composable with timeout)


Responsibility boundary:

This module does NOT provide test discovery / execution scheduling /
result determination / reporting — those are handled by pytest itself
(collection / runtest protocol / test_recorder). This module only
provides "optional execution-enhancement primitives usable inside a
test case", avoiding redundancy or semantic conflicts with pytest's
main flow.

Cross-platform design notes (Win / Linux / macOS):

1. Subprocesses uniformly use the "spawn" start method to avoid forking
   and inheriting pytest runtime state.
2. Timeout determination is based on proc.join(timeout) + is_alive(),
   not on the queue; subprocess crashes (segfault / early exit) will not
   be misreported as timeouts.
3. Timeout / interrupt cleanup kills the entire process tree per platform:
     - Windows: taskkill /F /T (falls back to proc.kill() on failure)
     - POSIX:   os.setsid() inside the child establishes a new session /
                process group; the parent killpg(SIGKILL)s it.
                Note: if the child has not yet completed setsid (the
                window right after start), its PGID equals the parent's,
                so killpg would kill the entire pytest session — hence
                the parent MUST verify PGID before calling killpg.
                See _kill_process_tree.
4. func/args/kwargs/return value must be picklable (hard spawn
   constraint); violating this yields a readable error instead of the
   obscure stack at the subprocess bootstrap.
5. Subprocess exceptions (including BaseException: sys.exit /
   KeyboardInterrupt / GeneratorExit) preferentially reconstruct the
   original exception object (with a remote-traceback note), and degrade
   to type + message + traceback strings when unpicklable. BaseException
   is uniformly downgraded to RuntimeError to prevent escape that would
   interrupt the pytest session.


Cleanup on Ctrl+C / abnormal parent exit:

The finally branch actively kills still-alive subprocesses — otherwise,
when Python exits, the multiprocessing atexit handler would call an
unbounded join() on non-daemon subprocesses; if the tested code is still
running long tasks (LLM calls, etc.), the pytest main process would hang
forever.

Orphan-process window (known trade-off):

Subprocesses use daemon=False to allow nested process spawning from
within a test case (nested-timeout scenarios). The cost: when the
parent is SIGKILLed or pytest itself crashes, the child, having left
the parent's process group via os.setsid(), will remain as an orphan.
This is a trade-off between "usability vs. crash safety", and usability
has been chosen. See README "Orphan processes" for details.
"""
from __future__ import annotations
import os, sys, time, traceback, subprocess, signal, pickle
import multiprocessing as mp
from queue import Empty

from neuraxis_testkit.log import get_logger

logger = get_logger(__name__)

class TimeoutError_(Exception):
    """
    Test execution timed out.

    NOTE: This exception inherits from Exception rather than the built-in
    TimeoutError. Callers must catch it with `except TimeoutError_`;
    `except TimeoutError` (built-in) will NOT catch it.
    """
    pass


class TestRunner:
    """
    Generic test-execution primitives (no CLI, no discovery, no batch,no reporting).

    Responsibilities:
      1. Timeout execution: run_with_timeout() — cross-platform
         process-level timeout.
      2. Retry execution:   run_with_retry()   — retry logic, composable
         with timeout.

    pytest has taken over test discovery / execution / reporting; this
    class only offers optional execution-enhancement capabilities.
    """

    def __init__(self, logger=None, default_timeout: int = 0,
                 default_retries: int = 0):
        # Initialize the test runner and set default parameters.
        self.logger = logger or get_logger("runner")  # Acquire logger
        self.default_timeout = default_timeout        # Default timeout in seconds
        self.default_retries = default_retries        # Default retry count

    # ============================================================
    # Timeout execution (cross-platform, based on multiprocessing.Process + spawn)
    # ============================================================

    def run_with_timeout(self, func, args=(), kwargs=None, timeout=None):
        """
        Cross-platform timeout execution of an arbitrary callable.

        timeout tri-state semantics (same as run_with_retry):
          None -> use self.default_timeout
          0    -> no limit for this call (overrides a nonzero default)
          >0   -> use the given seconds (timing includes the subprocess spawn startup cost)

        Args:
            func: callable (must be a module-level function / picklable object, hard spawn constraint)
            args:    positional arguments
            kwargs:  keyword arguments
            timeout: timeout in seconds

        Returns:
            The return value of `func`.

        Raises:
            TimeoutError_: timed out (process tree has been terminated)
            RuntimeError:  subprocess exited abnormally without returning
                           a result (crash / killed / unpicklable result);
                           or the subprocess raised a BaseException
                           (e.g. SystemExit)
            TypeError:     func/args/kwargs are not picklable
            Exception:     ordinary exceptions raised inside `func`
                           (restored to the original type, with a remote
                           traceback note attached)
        """
        kwargs = kwargs or {}
        # Tri-state: None=use default, 0=unlimited, >0=specified.
        if timeout is None:
            timeout = self.default_timeout
        if timeout <= 0:
            timeout = 0    # Explicitly unlimited
        if not timeout:
            return func(*args, **kwargs)

        # Pre-check the spawn constraint: yield a readable error instead
        # of the obscure stack at the subprocess bootstrap.
        self._check_picklable(func, args, kwargs)

        # Daemon processes cannot spawn children — raise a readable error early.
        if mp.current_process().daemon:
            raise RuntimeError(
                "Current process is a daemon process; cannot create subprocesses. "
                "Please call run_with_timeout in a non-daemon context."
            )

        ctx = mp.get_context("spawn")  # Cross-platform consistent, avoids fork inheriting pytest state / 跨平台一致, 避免 fork 继承 pytest 状态
        queue = ctx.Queue()
        proc = ctx.Process(
            target=self._wrap_func_callable,
            args=(func, args, kwargs, queue),
            daemon=False,   # Non-daemon: allows test cases to spawn their own children (nested-timeout scenarios) / 非 daemon: 允许用例内部再派生进程 (嵌套超时场景)
            # NOTE: Do not pass start_new_session here — that is a subprocess.
            # Popen argument; multiprocessing.Process does not accept it.
            # The POSIX independent session is established by os.setsid() inside the child.
        )

        try:
            # Timing starts before proc.start() to cover the spawn startup
            # cost (Windows cold start can reach seconds).
            deadline = time.monotonic() + timeout
            proc.start()

            remaining = max(0.0, deadline - time.monotonic())
            proc.join(timeout=remaining)

            if proc.is_alive():
                self._kill_process_tree(proc)
                raise TimeoutError_(
                    f"Execution timed out: {getattr(func, '__name__', str(func))} "
                    f"exceeded {timeout}s (including process startup cost); "
                    f"the entire process tree has been terminated."
                )

            # The process has exited; the queue content is either already available or will never arrive.
            try:
                result = queue.get(timeout=1)
            except (Empty, EOFError, OSError):
                # Subprocess crashed (segfault / early exit) or result was
                # not serializable — raise an explicit error, do NOT misreport it as a timeout.
                raise RuntimeError(
                    f"Subprocess exited abnormally (exitcode={proc.exitcode}) "
                    f"without returning any result. Common causes: segfault / "
                    f"os._exit() / unpicklable return value. See subprocess logs."
                ) from None

            if not isinstance(result, dict):
                # Defensive: unexpected structure on the queue.
                raise RuntimeError(
                    f"Subprocess returned an unexpected result structure: {result!r}"
                )

            # Prefer restoring the original exception object (preserves precise `except` matching).
            if "_exc" in result:
                exc = result["_exc"]
                if not isinstance(exc, Exception):
                    # Fallback: BaseException must not escape as-is
                    # (SystemExit would directly interrupt the pytest session).
                    exc = RuntimeError(
                        f"BaseException raised in subprocess "
                        f"[{type(exc).__name__}]: {exc}"
                    )

                tb = result.get("_tb")
                if tb:
                    self.logger.error(f"Traceback of exception inside subprocess:\n{tb}")
                    # Python 3.11+: attach the remote traceback to the exception so it prints after `except`.
                    if hasattr(exc, "add_note"):
                        try:
                            exc.add_note(f"--- remote traceback ---\n{tb}")
                        except Exception:
                            pass
                raise exc

            # Unpicklable exception / BaseException: returned as string message.
            if result.get("_error_type"):
                tb = result.get("_tb")
                if tb:
                    self.logger.error(f"Traceback of exception inside subprocess:\n{tb}")
                raise RuntimeError(
                    f"Exception inside subprocess [{result['_error_type']}]: "
                    f"{result.get('_error_msg', '')}"
                )

            return result.get("_value")
        finally:
            # ── Cleanup on Ctrl+C / abnormal parent exit ──
            # KeyboardInterrupt interrupts proc.join(); on entering finally,
            # the subprocess may still be alive. Without an active kill,
            # Python's exit path triggers the multiprocessing atexit handler
            # (_exit_function), which performs an unbounded p.join() on
            # non-daemon children — if the tested code is still running a
            # long task (LLM call, etc.), the pytest main process hangs forever.
            try:
                if proc.is_alive():
                    self._kill_process_tree(proc)
            except Exception:
                pass

            # Detach the queue first (skip waiting for the feeder thread to
            # flush, avoiding a secondary block on the interrupt path), then
            # close to release the fd; finally close proc to release the sentinel.
            try:
                queue.cancel_join_thread()
            except Exception:
                pass
            try:
                queue.close()
                queue.join_thread()
            except Exception:
                pass
            try:
                proc.close()
            except Exception:
                # proc.close() raises ValueError while still alive; the kill branch about should
                # already have handled that, but no extreme case should disturb the main flow.
                pass

    @staticmethod
    def _wrap_func_callable(func, args, kwargs, queue):
        """
        Subprocess wrapper: executes `func` and returns the result via the queue.

        Return protocol (dict): 回传协议 (dict)
          {"_value": value}           -> success
          {"_exc": exc, "_tb": tb}    -> picklable Exception, restored as-is
          {"_error_type": str, "_error_msg": str, "_tb": str} -> unpicklable exception / BaseException
        """
        # POSIX: establish an independent session / process group so the
        # parent can killpg the whole process tree in one call. Must be
        # done inside the child — multiprocessing.Process does not accept
        # start_new_session (that is subprocess.Popen's). Note: before
        # setsid completes, this process's PGID still equals the parent's;
        # the parent must verify PGID before killing (see _kill_process_tree).
        if os.name == "posix":
            try:
                os.setsid()
            except OSError as exp:
                # Rare: e.g., some container/sandbox environments disallow
                # setsid. Consequence: subsequent killpg degrades to
                # single-process kill, and grandchildren may remain. Print
                # a line to stderr so the silent degradation is visible.
                try:
                    print(
                        f"[runner] os.setsid() failed in child: {exp!r}; "
                        f"process-tree cleanup will degrade to single-process kill",
                        file=sys.stderr,
                    )
                except Exception:
                    pass

        try:
            value = func(*args, **kwargs)
        except BaseException as exp:
            # Covers sys.exit / KeyboardInterrupt / GeneratorExit and other escaped exceptions.
            payload = TestRunner._pack_exception(exp)
        else:
            # Perform a round-trip picklability check on the return value:
            # pure pickle.dumps misses "dumpable but not loadable" cases (e.g., custom __reduce__).
            try:
                pickle.loads(pickle.dumps(value))
            except Exception as exp:
                payload = {
                    "_error_type": "UnpicklableResult",
                    "_error_msg": f"Return value not picklable (round-trip check failed): {exp!r}",
                    "_tb": "",
                }
            else:
                payload = {"_value": value}

        try:
            queue.put(payload)
        except Exception:
            # Fallback: payload itself failed to put (extremely rare); deliver a readable error.
            try:
                queue.put({
                    "_error_type": "QueuePutError",
                    "_error_msg": "Failed to return subprocess result",
                    "_tb": "",
                })
            except Exception:
                pass  # Queue is completely unusable; rely on the parent's exitcode branch / 队列彻底不可用, 只能靠父进程的 exitcode 分支兜底

    @staticmethod
    def _pack_exception(exp: BaseException) -> dict:
        """
        Pack an exception into a returnable structure.

        Rules:
          - Only Exception subclasses are attempted for as-is return
            (the parent can then `except` the original type).
          - BaseException (SystemExit / KeyboardInterrupt / GeneratorExit)
            is uniformly downgraded to a string, so it cannot be raised
            as-is in the parent and interrupt the pytest session.
          - A failed round-trip check also downgrades to a string.
        """
        tb = traceback.format_exc()
        if isinstance(exp, Exception):
            try:
                pickle.loads(pickle.dumps(exp))
            except Exception:
                pass
            else:
                return {"_exc": exp, "_tb": tb}
        return {"_error_type": type(exp).__name__, "_error_msg": str(exp), "_tb": tb}

    @staticmethod
    def _check_picklable(func, args, kwargs) -> None:
        """
        Under spawn mode, func/args/kwargs must be picklable; check in
        advance to yield a readable error.

        NOTE: This really serializes args/kwargs once. If arguments
        contain large objects, this adds overhead; the spawn phase will
        serialize them again. Keeping it simple for now — could be
        narrowed to only checking `func` if performance becomes an issue.
        """
        checks = (("func", func), ("args", args), ("kwargs", kwargs))
        for name, obj in checks:
            try:
                pickle.loads(pickle.dumps(obj))
            except Exception as exp:
                raise TypeError(
                    f"run_with_timeout: {name} is not picklable "
                    f"(hard spawn constraint): {exp}. "
                    f"Use a module-level function; avoid lambda / closure / "
                    f"locally-defined function / objects holding connections or locks."
                ) from exp

    @staticmethod
    def _kill_process_tree(proc) -> None:
        """
        Terminate the entire process tree per platform, to prevent
        grandchildren from becoming orphans (especially severe on Windows).
        """
        if sys.platform == "win32":
            # TerminateProcess kills only a single process; use
            # taskkill /T to kill the whole tree. taskkill may be absent
            # from PATH / lack permissions / return non-zero — all need a fallback.
            killed = False
            try:
                r = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, timeout=10,
                )
                killed = (r.returncode == 0)
                if not killed:
                    stderr = (r.stderr or b"").decode(errors="replace").strip()
                    logger.warning(
                        "taskkill returned non-zero (rc=%s, pid=%s): %s",
                        r.returncode, proc.pid, stderr,
                    )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exp:
                logger.warning(
                    "taskkill unavailable or timed out (pid=%s): %s, "
                    "falling back to proc.kill()",
                    proc.pid, exp,
                )
            except Exception as exp:
                logger.warning(
                    "taskkill raised an exception (pid=%s): %s, "
                    "falling back to proc.kill()",
                    proc.pid, exp,
                )
            if not killed:
                try:
                    proc.kill()
                except Exception:
                    pass
        else:
            # POSIX: the child has already called os.setsid() in _wrap_func_callable,
            # establishing an independent process group;
            # killpg takes out the whole group (including grandchildren) in one shot.
            #
            # DEFENSE: during the brief window right after the child is started
            # (before the spawn interpreter boots and unpickles func/args/kwargs),
            # setsid has not yet run, so os.getpgid(proc.pid) returns the PARENT's own PGID — killpg
            # would SIGKILL the entire pytest session, appearing as a silent self-kill.
            # Therefore we MUST verify PGID: if it equals the parent's PGID, degrade to single-process kill.
            try:
                child_pgid = os.getpgid(proc.pid)
                parent_pgid = os.getpgrp()
            except (ProcessLookupError, PermissionError):
                # Child already gone, or insufficient permission: kill only itself.
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            else:
                if child_pgid == parent_pgid:
                    # Window period: the child has not yet setsid; killpg
                    # would kill the parent -> degrade to single-process
                    # kill. Consequence: grandchildren spawned by the child
                    # may remain (acceptable — safety first).
                    logger.warning(
                        "Subprocess %s has not yet established an independent "
                        "process group (PGID=%s equals parent's); degrading to "
                        "single-process kill; grandchildren may remain.",
                        proc.pid, child_pgid,
                    )
                    try:
                        proc.kill()
                    except Exception:
                        pass
                else:
                    try:
                        os.killpg(child_pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass

        proc.join(timeout=5)
        if proc.is_alive():
            logger.warning(
                "Process %s still alive after kill; manual inspection may be required.",
                proc.pid,
            )

    # ============================================================
    # Retry execution (composable with timeout: each attempt is independently wrapped in a process-level timeout)
    # ============================================================

    def run_with_retry(self, func, args=(), kwargs=None,
                       retries=None, delay=0, timeout=None):
        """
        Execution with retry.

        timeout tri-state semantics (same as run_with_timeout):
          None -> use self.default_timeout
          0    -> unlimited (each attempt runs in the current process,
                  no spawn overhead)
          >0   -> each attempt is an independent subprocess; the function
                  must NOT rely on in-process state across attempts

        NOTE: TimeoutError_ inherits from Exception, so a timeout will
        also trigger a retry — for slow, intermittently failing LLM
        cases this is the intended behavior; if "no retry on timeout" is
        desired, catch and distinguish at the caller's level.

        Args:
            func:    callable
            args:    positional arguments
            kwargs:  keyword arguments
            retries: retry count (None -> default_retries; 0 -> no retry)
            delay:   seconds between retries
            timeout: per-attempt timeout in seconds (see tri-state above)

        Returns:
            The return value of `func`.

        Raises:
            The exception from the last attempt.
        """
        kwargs = kwargs or {}
        retries = retries if retries is not None else self.default_retries
        # Guard against negative / non-integer: range(0) doesn't loop -> raise None -> TypeError.
        try:
            retries = max(0, int(retries))
        except (TypeError, ValueError):
            retries = 0

        # Tri-state: None=use default, 0=unlimited, >0=specified.
        if timeout is None:
            timeout = self.default_timeout
        if timeout <= 0:
            timeout = 0    # Explicitly unlimited

        last_exc: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                if timeout:
                    return self.run_with_timeout(func, args=args, kwargs=kwargs, timeout=timeout)
                return func(*args, **kwargs)
            except Exception as exp:
                last_exc = exp
                if attempt < retries:
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{retries + 1} failed: {exp}, "
                        f"retrying in {delay}s..."
                    )
                    if delay > 0:
                        time.sleep(delay)
                else:
                    raise

        # Pure defensive: theoretically unreachable (loop either raises or returns).
        raise last_exc if last_exc is not None else RuntimeError(
            "run_with_retry exited abnormally"
        )

