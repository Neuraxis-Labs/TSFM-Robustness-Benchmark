"""
config/settings.py -- Global Configuration

Centrally manages all environment-configurable items.
This module is the single source of truth for paths and secrets.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# Load .env environment files
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_DIR: Path = PROJECT_ROOT / "env"
# Default to load .production (if exists), then load .env.{APP_ENV} according to APP_ENV.
# Possible values for APP_ENV: test, production, development, etc. Default is 'test'.
APP_ENV = os.getenv("APP_ENV", "example").lower()

# Explicitly list files to load (in priority order, later overrides earlier for same variables).
env_files = [
    ENV_DIR / ".env.example",       # Base config (shared)
    ENV_DIR / f".env.{APP_ENV}",    # Environment-specific
]

# Load each existing file; later files override earlier ones for duplicate keys.
for env_file in env_files:
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)

TESTCASES_DIR: Path = PROJECT_ROOT / "testcases"
OUTPUT_DIR:  Path = PROJECT_ROOT / "outputs"
RESULTS_DIR: Path = OUTPUT_DIR / "results"
LOGS_DIR:    Path = OUTPUT_DIR / "logs"

# ============================================================
# API Configuration
# ============================================================
# Priority: Environment variables > Default values here
# If modification is needed, it is recommended to set TIMECHO_API_KEY in .env or via environment variables.

API_KEY: str = os.getenv(
    "TIMECHO_API_KEY",
    "ts-Update-Your-TIMECHO_API_KEY",
)

# Automatically ensure key directories exist (idempotent operation)
from neuraxis_testkit.utils.files import ensure_dir

for _dir in (OUTPUT_DIR, LOGS_DIR, RESULTS_DIR):
    ensure_dir(_dir)

