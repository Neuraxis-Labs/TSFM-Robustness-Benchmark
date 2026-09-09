# TSFM Robustness Benchmark

[English](./README.md) | [中文](./README.zh-CN.md)

The TSFM Robustness Benchmark is a systematic testing tool designed to evaluate the engineering robustness of Time Series Foundation Models (TSFMs) in edge cases (e.g., frequency mismatch, data contamination, covariate interference). This release includes a systematic evaluation of TimechoAI as the first targeted model. More models will be integrated in subsequent iterations.

## 1. Core Architecture - Layered Architecture
- This project is built on **Python 3.12+**, with core dependencies including `pytest`, `timecho-ai`, and `pandas`.
- The system adopts a clear layered architecture, decoupling business logic, infrastructure utilities, and test execution.
- The middle infrastructure layer natively supports **cross-platform execution** (Windows / macOS / Linux) and **concurrent execution** (distributed test scheduling via `pytest-xdist`, plus process-level concurrency control).
- The test toolkit (`neuraxis_testkit`) is packaged as a standalone SDK under the `src/` directory, fully decoupled from business code and ready for reuse across other product lines.

## 2. Directory & File Structure

The project follows a standard layered architecture:

- `config/`: Global Configuration Management Module
  - `constants.py`: Global constant definitions.
  - `settings.py`: Global environment variables (e.g., `TIMECHO_API_KEY`).
- `core/`: Core Business Components Layer (Encapsulates logic and state management)
  - `client.py`: Low-level client connection (internal bridging module; business code should access it indirectly via timecho.py).
  - `metrics.py`: Evaluation and computation metrics.
  - `models.py`: Shared data models (TestStatus, TestResult, BatchReport).
  - `results.py`: Test result manager (batch buffering/persistence).
  - `resume.py`: Strategy controller (rate-limiting/checkpoint resume).
  - `timecho.py`: API interaction wrapper.
- `env/`: Directory for environment variable configs, loaded by settings.py via load_dotenv()
   - `.env.example`: Example environment variables file.
- `src/`: **SDK source directory**
  - `neuraxis_testkit/`: Test toolkit
    - `log/`: Logging management module.
      - `config.py`: Logging variable configuration.
      - `context.py`: Context managers (`LogLevelContext`).
      - `core.py`: Core `Logger` class.
      - `decorators.py`: Decorators (`log_execution`, `log_time`).
      - `filters.py`: Log filters (`ModuleLevelFilter`, `IgnoredLoggerFilter`).
      - `formatters.py`: Log formatters (`ColoredFormatter`).
      - `handlers.py`: Handler management.
    - `pytest_infra/`: pytest infrastructure layer 【WIP】— Provides encapsulated fixtures, manifest-driven parameterization, and resumable-execution infrastructure. Currently under active development and debugging; interfaces are not yet stable and direct external dependency is discouraged. The internal module layout will be documented once the interfaces are finalized.
    - `utils/`: Basic utilities layer
      - `concurrent.py`: Concurrency control and process coordination module (internal bridge module).
      - `data_sanitizer.py`: Data sanitization and type-safety utilities.
      - `files.py`: File operation utilities.
      - `runner.py`: Core test-running primitives (AST-based static discovery + single-case execution + in-memory result tracking).
- `testcases/`: Business Scenario Test Cases
- `README.md`: Project documentation, providing an overview, usage instructions, and notes.
- `conftest.py`: pytest entry-point configuration; internally bridges to [`neuraxis_testkit.pytest_infra.conftest`](https://github.com/Neuraxis-Labs/TSFM-Robustness-Benchmark/blob/main/src/neuraxis_testkit/pytest_infra/conftest.py) to reuse root-level fixtures and hooks.
- `run.py`: **Unified project entry point**, responsible for bootstrapping `sys.path` and launching the specified test script by module name or file path. 【Planned for removal】— Once the pytest-driven workflow is stable, this entry point will be replaced by pytest directly and removed.

## 3. Testing Workflow

1. **Configuration Initialization**: Read environment variables from `config/settings.py`.
2. **Model Initialization**: Initialize the TimechoAI model using the provided API key.
3. **Test Execution**: Execute specific testing workflows based on command-line arguments.
4. **Result Output**: Output test results to the console or specified files.

## 4. Setup & Installation

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

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

>  **Windows PowerShell users**: If you see an error about script execution being disabled, open PowerShell as Administrator and run:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 4.3 Quit Virtual Environment

**Deactivate the virtual environment (universal)** :
```bash
  deactivate
```

### 4.4 Install Dependencies

> **Note:** This project requires **Python 3.12 or later**. Please verify your Python version before installing.

After activating the virtual environment, execute the following command to install the core dependencies:

```bash
python -m pip install timecho-ai pandas pytest pytest-xdist portalocker
```

> ** Platform Note:** 
> If you are running the code on Windows, it is recommended to install `portalocker` with the Windows extension to ensure proper cross-process file locking:
> ```bash
> python -m pip install "portalocker[win32]"
> ```

## 5. Quick Start

Launch tests via the unified project entry point `run.py`:

```bash
# Run tests by module name
python run.py <module_name>

# Or run tests by file path
python run.py <path/to/test_file.py>
```

## 6. Testing Objectives
- Edge case exploration: Systematically verify the engineering robustness of the model against boundary conditions such as complex queries, replica inconsistencies, and out-of-order time-series writes.  
- Defensive architecture verification: Apply strict engineering standards to test the model's degradation behavior and recovery capabilities under non-ideal inputs.

## 7. Scope of Testing Disclaimer
The test results of this framework are limited by the specific model version, data preprocessing strategy, and runtime environment. This tool aims to provide an objective reference perspective for the engineering defensive architecture design of time-series models, rather than an absolute assertion of the final performance of any commercial product.

