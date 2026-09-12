# TSFM Robustness Benchmark

[English](./README.md) | [中文](./README.zh-CN.md)

TSFM Robustness Benchmark is a systematic testing tool designed to evaluate the engineering robustness of time series foundation models under edge scenarios, such as frequency mismatch, data contamination, and covariate interference.  
This release includes a systematic evaluation of TimechoAI as the first target model. More models will be integrated in subsequent iterations.

## 1. Core Architecture - Layered Architecture

- This project is developed based on **Python 3.12+**, with core dependencies on `pytest`, `timecho-ai`, and `pandas`.
- The system adopts a clear layered architecture, decoupling business logic, infrastructure utilities, and test execution.
- The middle infrastructure layer natively supports **cross-platform execution** (Windows / macOS / Linux) and **concurrent execution** (distributed test scheduling via `pytest-xdist`, plus process-level concurrency control).
- The test toolkit (`neuraxis_testkit`) is packaged as a standalone SDK under the `src/` directory, fully decoupled from business code and ready for reuse across other product lines.

## 2. Directory & File Specifications

The project follows a standard layered architecture, with the directory structure as follows:

```text
project/
├── config/
│   ├── constants.py             # Global business constants
│   └── settings.py              # Global environment variable configuration (global path configuration, etc.)
│
├── core/                        # Business core common components layer (encapsulates business logic and state management)
│   ├── client.py                  # Low-level client connection (get_timecho_client, etc.)
│   ├── metrics.py                 # Evaluation metrics calculation
│   ├── models.py                  # Business data models (request/response)
│   ├── results.py                 # Business result persistence (CSV/JSON), depends on utils.files
│   ├── resume.py                  # Policy controller (rate limiting / resume from breakpoint)
│   └── timecho.py                 # Timecho API client
│
├── env/                         # Environment variable configuration directory, loaded by `config/settings.py` via load_dotenv()
│   └── .env.example               # Environment variable example file
│
├── src/                         # SDK source directory
│   └── neuraxis_testkit/          # Test toolkit
│       ├── log/                     # Logging management
│       │   ├── __init__.py          # Unified interface exposed externally
│       │   ├── config.py            # Variable configuration
│       │   ├── context.py           # Context manager (LogLevelContext)
│       │   ├── core.py              # Core Logger class
│       │   ├── decorators.py        # Decorators (log_execution, log_time)
│       │   ├── filters.py           # Filters (ModuleLevelFilter, IgnoredLoggerFilter)
│       │   ├── formatters.py        # Formatters (ColoredFormatter)
│       │   └── logging.yaml         # Logging configuration
│       ├── pytest_infra/          # Pytest infrastructure layer
│       │   ├── __init__.py          # Unified interface exposed externally
│       │   ├── collection.py        # Dynamic collection (calls manifest_loader)
│       │   ├── fixtures.py          # All @pytest.fixture
│       │   ├── hooks.py             # All hookimpl (including pytest_addoption / pytest_configure)
│       │   ├── manifest_loader.py   # Loads YAML manifests and generates parametrization
│       │   ├── models.py            # Test-specific data models (e.g., test case parameters)
│       │   ├── paths.py             # NeuraxisPaths + get_paths (pure data + accessors)
│       │   ├── resume.py            # Resume-from-breakpoint logic (based on historical results)
│       │   ├── session_manager.py   # Test session management (shared resources, locks) SessionFileLock + SessionManager
│       │   └── test_recorder.py     # Test result recorder
│       └── utils/                 # Common utilities layer
│           ├── __init__.py          # Unified interface exposed externally
│           ├── assertions.py        # Generic Assertions Library (Pure Logic)
│           ├── concurrent.py        # Concurrency-safe utilities (portalocker wrapper)
│           ├── data_sanitizer.py    # Data cleaning and type safety utilities
│           ├── files.py             # File operation utilities
│           └── runner.py            # Core test execution primitives (AST static discovery + single test execution + in-memory result tracking)
│
├── testcases/                   # LLM business test cases
│   └── futureCovs/
│       └── dirtyData/
│           ├── test_dirty.py
│           └── data/              # Test data files (inputs required by test cases)
│               └── test_dirty_s0.csv
│
├── outputs/                     # Generated at runtime: logs, results, HTML reports
│   ├── results/                   # Business results (CSV/JSON)
│   ├── reports/                   # pytest reports (HTML/XML)
│   ├── analytics/                 # Model analysis results
│   └── logs/                      # Log files
│       └── tsfm_benchmark_20260824.log  # Filename dynamically includes execution date
│
├── conftest.py                  # Repository-level pytest adaptation entry point
├── pyproject.toml               # Project configuration management
├── README.md                    # Project documentation (English), providing project overview, usage, notes, etc.
└── .python-version
```

Key file descriptions:

- `conftest.py`: Repository-level pytest adaptation entry point. It declares project-level CLI options such as `--project-root`, `--output-dir`, `--results-dir`, `--logs-dir`, and sets a custom report header. Common pytest hooks/fixtures are automatically discovered by `neuraxis_testkit.pytest_infra` via the `pytest11` entry point, and do not need to be manually bridged in `conftest.py`.
- `pyproject.toml`: Project configuration, dependency declarations, pytest configuration, and `pytest11` plugin entry points.
- `src/neuraxis_testkit/`: SDK source directory, which will be split into an independent project later.
- `testcases/`: Business test cases in the current repository, which will be split into an independent business test repository later.

## 3. Test Workflow

1. **Configuration Initialization**: `config/settings.py` loads environment variable files (e.g., `.env`) from the `env/` directory via `load_dotenv()`, and exports global paths and runtime configurations.
2. **Model Initialization**: Initialize the TimechoAI model using the provided API key.
3. **Test Execution**: Execute the specified test workflow according to the provided command-line arguments.
4. **Result Output**: Output test results to the console or a specified file.

## 4. Commands and Installation

- Python **3.12 or higher**
- Virtual environment recommended

### 4.1 Create Virtual Environment

```bash
python -m venv .venv
```

### 4.2 Activate Virtual Environment

Choose the corresponding command based on your operating system:

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

> **Note for Windows PowerShell users**: If you encounter a "running scripts is disabled" error, run PowerShell as Administrator and execute:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4.3 Deactivate the Virtual Environment

```bash
deactivate
```

### 4.4 Install Dependencies

It is recommended to install the project in editable mode. `-e` stands for editable, meaning source code changes take effect without reinstallation.

**Development/Complete installation (recommended):**

```bash
python -m pip install -e ".[test]"
```

This command installs:

- Runtime dependencies: `timecho_ai`, `pandas`, `requests`, `pytest`, `portalocker`, `python-dotenv`, `pyyaml`
- Development dependencies: `pytest-xdist`, `pytest-html`, `pytest-cov`, `pytest-timeout`, `pytest-mock`, `pytest-randomly`, `pre-commit`

**Install runtime dependencies only:**

```bash
python -m pip install -e .
```

**Windows Platform Note:**

The `pyproject.toml` declares `portalocker>=4.3.0`, which **does not automatically install** the `portalocker[win32]` extension.  
If cross-process file locking issues occur on Windows, install it additionally:

```bash
python -m pip install "portalocker[win32]"
```

### 4.5 Refresh Entry Points and Verify Plugin Registration

When `[project.entry-points.pytest11]` in `pyproject.toml` changes, or when you need to refresh the package metadata without touching other dependencies, run:

```bash
python -m pip install -e ".[test]" --force-reinstall --no-deps
```

After installation, verify that the pytest plugin is registered successfully:

```bash
python -c "import importlib.metadata as m; [print(ep) for ep in m.entry_points(group='pytest11') if 'neuraxis' in ep.name]"
```

Expected output similar to:

```text
EntryPoint(name='neuraxis_testkit_hooks', value='neuraxis_testkit.pytest_infra.hooks', group='pytest11')
EntryPoint(name='neuraxis_testkit_fixtures', value='neuraxis_testkit.pytest_infra.fixtures', group='pytest11')
```

## 5. Quick Start

The project has fully switched to **pytest native mode** and no longer uses `run.py`. It is recommended to use `python -m pytest` consistently to ensure the Python interpreter from the current virtual environment is used.

**Run all tests:**

```bash
python -m pytest
```

**Run by module name or keyword:**

```bash
python -m pytest -k <module_name>
```

**Run by file path:**

```bash
python -m pytest testcases/path/to/test_file.py
```

**Run by marker:**

```bash
python -m pytest -m smoke
python -m pytest -m "not slow"
```

**Run concurrently:**

```bash
python -m pytest -n auto
```

**Override path configurations:**

```bash
python -m pytest \
  --project-root . \
  --output-dir ./outputs \
  --results-dir ./outputs/results \
  --logs-dir ./outputs/logs
```

The default HTML report path is controlled by `addopts` in `pyproject.toml`:

```text
outputs/reports/report.html
```

## 6. pytest Configuration

The `[tool.pytest.ini_options]` section in `pyproject.toml` is configured as follows:

- `testpaths = ["testcases"]`
- Test file pattern: `test_*.py`
- Test class pattern: `Test*`
- Test function pattern: `test_*`
- Detailed output, HTML report, strict markers, and short traceback enabled by default
- Default failure limit: `--maxfail=5`
- CLI logging enabled
- Custom markers registered
- Custom session label and log basename

`pytest11` entry points:

```toml
[project.entry-points.pytest11]
neuraxis_testkit_hooks = "neuraxis_testkit.pytest_infra.hooks"
neuraxis_testkit_fixtures = "neuraxis_testkit.pytest_infra.fixtures"
```

Therefore, as long as the project is installed via `pip install -e .` or a regular installation, pytest will automatically load the hooks and fixtures from `neuraxis_testkit.pytest_infra`.

## 7. Testing Objectives

- Edge scenario detection: Systematically verify the engineering robustness of models against boundary conditions such as complex queries, multi-replica inconsistency, and out-of-order time series writes.
- Defensive architecture validation: Stress-test with strict engineering standards to examine model degradation behavior and recovery capability under non-ideal inputs.

## 8. Scope Statement

The test results of this framework are limited by specific model versions, data preprocessing strategies, and runtime environments. This tool is intended to provide an objective reference perspective for the engineering defensive architecture design of time series models, rather than an absolute assertion of the final performance of any commercial product.

## 9. License

MIT License