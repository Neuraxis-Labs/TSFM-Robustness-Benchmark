#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuraxis_testkit/log/config.py - Log Configuration Center (logging.yaml + environment variable overrides)

Responsibilities:
    1. Environment variables -> module constants (LOG_LEVEL / LOG_FORMAT / ...)
    2. TRACE level registration (not present in standard library; dynamically registered on logging.Logger)
    3. Log directory management:
       - Default `./outputs/logs` (fallback to system temp directory if not writable)
       - Framework pytest_configure passes full file path via `set_log_file()`, which automatically extracts parent directory as log root
       - Use `get_logs_dir()` to obtain current log root (read-only)
    4. logging.yaml placeholder resolution (${VAR} / ${VAR:default}), fail-fast on undefined variables
    5. dictConfig assembly + filter wiring + optional asynchronous queue (bounded, drops on full)
    6. Two-stage file naming: `set_log_file()` for pytest_configure hot‑replacement of log file
    7. Cross‑platform path handling: forces POSIX style (/) to avoid YAML escaping issues
"""
from __future__ import annotations

import atexit, logging, logging.config, logging.handlers, os, sys, queue, re, yaml, threading, tempfile
from datetime import datetime
from pathlib import Path

# ============================================================
# Configuration file location (can be overridden by LOG_CONFIG_FILE to point to external project config)
# ============================================================
_DEFAULT_CONFIG_FILE = Path(__file__).with_name("logging.yaml")
LOG_CONFIG_FILE: Path = Path(os.getenv("LOG_CONFIG_FILE", str(_DEFAULT_CONFIG_FILE)))

# ============================================================
# Runtime environment detection
# ============================================================
def _xdist_worker() -> bool:
    return os.environ.get("PYTEST_XDIST_WORKER") is not None

# ============================================================
# Log root directory management (supports dynamic injection)
# ============================================================
_LOGS_DIR: Path | None = None   # internal variable, set by set_logs_dir or set_log_file

def get_logs_dir() -> Path:
    """
    Get the current log root directory (read-only).
    If not set (i.e., set_log_file has not been called), returns the default directory
    (current_directory/outputs/logs) and creates it automatically; if the default is not writable,
    falls back to the system temporary directory.

    """
    global _LOGS_DIR
    if _LOGS_DIR is not None:
        return _LOGS_DIR
    # Default: outputs/logs under current working directory; fallback to system temp if fails
    try:
        default = Path.cwd() / "outputs" / "logs"
        default.mkdir(parents=True, exist_ok=True)
        _LOGS_DIR = default
        return default
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "neuraxis_testkit_logs"
        fallback.mkdir(parents=True, exist_ok=True)
        _LOGS_DIR = fallback
        return fallback

# ============================================================
# Environment variables -> constants (context for YAML placeholder resolution)
# ============================================================
def _env_bool(key: str, default: str) -> bool:
    return os.getenv(key, default).strip().lower() in ("true", "1", "yes")

# Log Level: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Log file base name
LOG_FILE_BASENAME: str = os.getenv("LOG_FILE_BASENAME", "neuraxis_testkit")

# Whether to include date in the filename
LOG_FILE_WITH_DATE: bool = _env_bool("LOG_FILE_WITH_DATE", "true")

# Whether to output to file
LOG_FILE_OUTPUT = _env_bool("LOG_FILE_OUTPUT", "true")

# Log file size limit (bytes), default 50MB
LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", str(50 * 1024 * 1024)))

# Number of backup log files
LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "10"))

# Log file encoding
LOG_ENCODING: str = os.getenv("LOG_ENCODING", "utf-8")

# Log format (includes Process ID, for multi-concurrency scenarios)
LOG_FORMAT: str = os.getenv(
    "LOG_FORMAT",
    '[%(asctime)s.%(msecs)03d][%(levelname)8s][%(process)d][%(threadName)s][%(name)s][%(filename)s:%(lineno)d] - %(message)s'
)

# Simplified log format
LOG_SIMPLE_FORMAT: str = os.getenv(
    "LOG_SIMPLE_FORMAT",
    '[%(asctime)s.%(msecs)03d][%(levelname)s] - %(message)s'
)

# Date format
LOG_DATE_FORMAT: str = os.getenv(
    "LOG_DATE_FORMAT",
    '%Y-%m-%d %H:%M:%S'
)

# Whether to enable colored output (console only)
LOG_USE_COLOR: bool = _env_bool("LOG_USE_COLOR", "true")

# Log rotation type: 'size' for size-based, 'time' for time-based
LOG_ROTATION: str = os.getenv("LOG_ROTATION", "size").lower()      # "size" | "time"

# Time rotation specifier (Effective when LOG_ROTATION == 'time')
LOG_WHEN: str = os.getenv("LOG_WHEN", "midnight")

# Time rotation interval count
LOG_INTERVAL: int = int(os.getenv("LOG_INTERVAL", "1"))

# Log queue size (0 disables async mode, uses sync mode)
LOG_QUEUE_SIZE: int = int(os.getenv("LOG_QUEUE_SIZE", "10000"))

_UNDER_PYTEST = "pytest" in sys.modules
_raw_console = os.getenv("LOG_CONSOLE_OUTPUT", "").strip().lower()
if _raw_console:    # explicit: user's decision takes precedence
    LOG_CONSOLE_OUTPUT = _raw_console in ("true", "1", "yes")
else:               # auto mode: off under pytest, on when running standalone
    LOG_CONSOLE_OUTPUT = not _UNDER_PYTEST

# Module-level log overrides (for debugging specific modules)
# Format: {"module.name": "DEBUG"}
MODULE_LEVEL_OVERRIDES: dict[str, str] = {}

# Names of loggers to ignore (do not write to log)
IGNORED_LOGGERS: list[str] = ["urllib3", "requests", "botocore", "boto3"]

# ============================================================
# TRACE level registration (standard library extension, once per process)
# ============================================================
if not hasattr(logging, 'TRACE'):
    logging.TRACE = logging.DEBUG - 5
    logging.addLevelName(logging.TRACE, "TRACE")
if not hasattr(logging.Logger, 'trace'):
    def _trace(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.TRACE):
            self._log(logging.TRACE, message, args, **kwargs)
    logging.Logger.trace = _trace

LEVEL_MAP: dict[str, int] = {
    'TRACE': logging.TRACE,
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}
VALID_LEVELS = frozenset(LEVEL_MAP.keys())

def get_log_level() -> int:
    return LEVEL_MAP.get(LOG_LEVEL, logging.INFO)

def get_level_name(level: int) -> str:
    """int -> level name. Standard getLevelName accepts int input."""
    name = logging.getLevelName(level)
    return name if isinstance(name, str) else f"Level {level}"


# ============================================================
# Log file naming (two-stage: fallback name -> hot-replaced by set_log_file())
# ============================================================
def _generate_log_file_name() -> str:
    """
    Generates the log filename with optional date suffix.

    Returns:
        Log filename, e.g., neuraxis_testkit_20260820.log
    """
    name = LOG_FILE_BASENAME
    if LOG_FILE_WITH_DATE:
        name = f"{name}_{datetime.now():%Y%m%d}"
    if _xdist_worker():
        name += f"_{os.environ['PYTEST_XDIST_WORKER']}"
    return f"{name}.log"

LOG_FILE_NAME: str = _generate_log_file_name()    # fallback name: under pytest, set_log_file() takes precedence

# ============================================================
# Two-stage log file name (for pytest_configure to call)
#   Stage 1: get_logger -> setup_logging uses fallback name on module import
#   Stage 2: pytest_configure obtains business-side basename/run_ts, then
#            set_log_file() forces reload and hot‑replaces file handlers
# ============================================================
_LOG_FILE_OVERRIDE: Path | None = None
_listener: logging.handlers.QueueListener | None = None

def get_log_file_path() -> Path:
    """Current effective log file path (single authoritative entry; do not reference LOG_FILE_NAME directly)."""
    if _LOG_FILE_OVERRIDE is not None:
        return _LOG_FILE_OVERRIDE
    return get_logs_dir() / LOG_FILE_NAME

def set_log_file(file_path: str | Path) -> None:
    """
     Two-stage naming, stage 2: called by pytest_configure after obtaining business-side file name.

    - Automatically sets the file's directory as the log root (calls set_logs_dir internally)
    - Automatically appends _gwN suffix for xdist workers to avoid multiple processes writing to the same file
    - Internally forces reload (force=True), hot‑replacing file handlers
    """
    global _LOG_FILE_OVERRIDE, LOG_FILE_NAME, _LOGS_DIR

    p = Path(file_path)
    logs_dir = p.parent.expanduser().resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    _LOGS_DIR = logs_dir
    if _xdist_worker():
        p = p.with_name(f"{p.stem}_{os.environ['PYTEST_XDIST_WORKER']}{p.suffix}")
    _LOG_FILE_OVERRIDE = p
    LOG_FILE_NAME = p.name      # sync fallback name for compatibility with external references
    setup_logging(force=True)

def force_console_encoding(encoding: str | None = None) -> None:
    """
    Force reconfigure stdout/stderr encoding.
    Cross-platform console encoding enforcement:
      - Windows: default GBK/cp936, fixes garbled Chinese (primary beneficiary)
      - Linux/CI: locale C/POSIX only supports ASCII, fixes UnicodeEncodeError
      - macOS: already UTF‑8, idempotent no‑op
    Best-effort: silently falls back if stream cannot be reconfigured (e.g., pytest capture wrapper, non‑tty stream).

    """
    enc = encoding or LOG_ENCODING
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding=enc, errors="backslashreplace")
        except Exception:     # noqa: BLE001 - best-effort, fallback on failure
            try:
                print(f"[WARN] force_console_encoding: reconfigure {name} failed",
                      file=sys.__stderr__, flush=True)
            except Exception:
                pass

# ============================================================
# logging.yaml placeholder resolution: ${VAR} / ${VAR:default}
# ============================================================
def _load_yaml_text() -> str:
    env_file = os.getenv("LOG_CONFIG_FILE")
    if env_file:    # project-level external config takes precedence
        return Path(env_file).read_text(encoding="utf-8")
    try:        # installed scenario (including zip)
        from importlib import resources
        return resources.files(__package__).joinpath("logging.yaml").read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError, ImportError):
        return Path(__file__).with_name("logging.yaml").read_text(encoding="utf-8")

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")
_INT_VARS = {"LOG_MAX_BYTES", "LOG_BACKUP_COUNT", "LOG_INTERVAL", "LOG_QUEUE_SIZE"}
_BOOL_VARS = {"LOG_FILE_WITH_DATE", "LOG_FILE_OUTPUT", "LOG_USE_COLOR", "LOG_CONSOLE_OUTPUT"}

def _coerce(name: str, raw):
    if name in _INT_VARS:
        return int(float(raw))
    if name in _BOOL_VARS:
        return str(raw).strip().lower() in ("true", "1", "yes")
    return raw

def _build_context() -> dict:
    # Key: use as_posix() to convert path to '/', avoiding Windows backslash escaping in YAML
    ctx: dict = {"LOG_FILE_PATH": get_log_file_path().as_posix()}
    for name, value in globals().items():
        if name.startswith("LOG_") and not name.startswith("LOG_CONFIG"):
            ctx[name] = value
    return ctx

def _resolve_placeholders(text: str, ctx: dict) -> str:
    def _repl(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in ctx and ctx[name] is not None:
            return str(_coerce(name, ctx[name]))
        if default is not None:
            return str(_coerce(name, default))
        # fail-fast: leaving ${VAR} in dictConfig only causes obscure errors
        raise ValueError(
        )

    return "\n".join(
        line if line.lstrip().startswith("#") else _PLACEHOLDER_RE.sub(_repl, line)
        for line in text.splitlines())

# ============================================================
# Post configuration pruning (console / file switches, rotation mutual exclusivity)
# ============================================================
def _prune_handlers(cfg: dict) -> dict:
    """
    Prune handlers according to LOG_CONSOLE_OUTPUT / LOG_FILE_OUTPUT / LOG_ROTATION.

    Critical guarantee: at most one of 'file' and 'file_timed' is kept —— dual handlers writing to the same file
    would interleave rotations and cause PermissionError on Windows due to file locking.
    """
    root = cfg.setdefault("root", {})
    root.setdefault("handlers", [])

    def _strip(names: set[str]) -> None:
        root["handlers"] = [h for h in root["handlers"] if h not in names]
        for lg in cfg.get("loggers", {}).values():
            if isinstance(lg, dict) and lg.get("handlers"):
                lg["handlers"] = [h for h in lg["handlers"] if h not in names]

    if not LOG_CONSOLE_OUTPUT:
        _strip({"console"})
    if not LOG_FILE_OUTPUT:
        _strip({"file", "file_timed"})
    else:
        if LOG_ROTATION == "time":
            _strip({"file"})
            if "file_timed" not in root["handlers"] and "file_timed" in cfg.get("handlers", {}):
                root["handlers"].append("file_timed")
        else:
            _strip({"file_timed"})
            if "file" not in root["handlers"] and "file" in cfg.get("handlers", {}):
                root["handlers"].append("file")
    return cfg

_HANDLER_INT_KEYS = {"maxBytes", "backupCount", "interval"}
_HANDLER_BOOL_KEYS = {"delay"}

def _coerce_handler_types(cfg: dict) -> dict:
    """Fallback for YAML quoted values: restore type when numbers/booleans are parsed as strings."""
    for h in cfg.get("handlers", {}).values():
        if not isinstance(h, dict):
            continue
        for k in _HANDLER_INT_KEYS:
            if isinstance(h.get(k), str):
                h[k] = int(float(h[k]))
        for k in _HANDLER_BOOL_KEYS:
            if isinstance(h.get(k), str):
                h[k] = h[k].strip().lower() in ("true", "1", "yes")
    for f in cfg.get("formatters", {}).values():
        if isinstance(f, dict) and isinstance(f.get("use_color"), str):
            f["use_color"] = f["use_color"].strip().lower() in ("true", "1", "yes")
    return cfg

# ============================================================
# setup_logging (thread-safe; bounded async queue, drops on full)
# ============================================================
_config_lock = threading.Lock()
_configured = False

class _DropOnFullQueueHandler(logging.handlers.QueueHandler):
    """Drop log records when queue is full instead of blocking business threads (drop优于阻塞 on log floods)."""
    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass

def setup_logging(force: bool = False) -> None:
    """Load logging.yaml and complete logging assembly; idempotent after first call (force=True for reload)."""
    global _configured, _listener
    with _config_lock:
        if _configured and not force:
            return
        if _configured and force:
            # Hot reload: unregister atexit -> stop listener -> close old file handles
            if _listener is not None:
                atexit.unregister(_listener.stop)
                _listener.stop()
                _listener = None
            for handler in logging.getLogger().handlers[:]:
                if isinstance(handler, (logging.handlers.RotatingFileHandler, logging.handlers.TimedRotatingFileHandler)):
                    handler.close()

        from .filters import IgnoredLoggerFilter, ModuleLevelFilter   # lazy import to avoid circular

        resolved = _resolve_placeholders(_load_yaml_text(), _build_context())
        cfg = yaml.safe_load(resolved)
        _coerce_handler_types(cfg)
        _prune_handlers(cfg)
        logging.config.dictConfig(cfg)

        root = logging.getLogger()
        file_handlers = []
        for h in list(root.handlers):
            if isinstance(h, (logging.handlers.RotatingFileHandler,
                              logging.handlers.TimedRotatingFileHandler)):
                h.addFilter(IgnoredLoggerFilter(IGNORED_LOGGERS))
                h.addFilter(ModuleLevelFilter(MODULE_LEVEL_OVERRIDES))
                file_handlers.append(h)

        if LOG_QUEUE_SIZE > 0 and file_handlers:
            q: queue.Queue = queue.Queue(maxsize=LOG_QUEUE_SIZE)   # bounded
            root.addHandler(_DropOnFullQueueHandler(q))
            _listener = logging.handlers.QueueListener(q, *file_handlers,
                                                       respect_handler_level=True)
            for fh in file_handlers:
                root.removeHandler(fh)
            _listener.start()
            atexit.register(_listener.stop)     # only "alive" listener will be left for atexit

        _configured = True


__all__ = [
    # Configuration Items
    'LOG_LEVEL',
    'LOG_FILE_NAME',
    'LOG_CONSOLE_OUTPUT',
    'LOG_FILE_OUTPUT',
    'LOG_MAX_BYTES',
    'LOG_BACKUP_COUNT',
    'LOG_ENCODING',
    'LOG_FORMAT',
    'LOG_SIMPLE_FORMAT',
    'LOG_DATE_FORMAT',
    'LOG_USE_COLOR',
    'LOG_QUEUE_SIZE',
    'MODULE_LEVEL_OVERRIDES',
    'IGNORED_LOGGERS',
    # Mappings
    'LEVEL_MAP',
    'VALID_LEVELS',
    # Convenience Functions
    'setup_logging',
    'get_log_level',
    'get_level_name',
    'get_logs_dir',
    'get_log_file_path',
    'force_console_encoding',
]
