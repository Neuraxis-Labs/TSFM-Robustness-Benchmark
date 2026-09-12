# TSFM Robustness Benchmark

[English](./README.md) | [中文](./README.zh-CN.md)

TSFM 鲁棒性基准测试是一种系统化的测试工具, 旨在检验时间序列基础模型在边缘场景(如频率不匹配、数据污染、协变量干扰等)下的工程鲁棒性.
本次版本包含对 TimechoAI 作为首个靶向模型的系统性评估, 更多模型将在后续迭代中逐步整合.

## 1. 核心架构 - 分层架构

- 本项目基于 **Python 3.12+** 开发, 核心依赖 `pytest`、`timecho-ai` 和 `pandas`.
- 系统采用清晰的分层架构, 确保业务逻辑、基础工具与测试执行解耦.
- 中间基础层原生支持 **跨平台运行** (Windows / macOS / Linux) 与 **并发执行** (基于 `pytest-xdist` 的分布式用例调度, 以及进程级并发控制).
- 测试工具箱 (`neuraxis_testkit`) 以 SDK 形式独立打包于 `src/` 目录下, 与业务代码解耦, 可直接复用至其他产品线.

## 2. 目录与文件规范

项目遵循标准分层架构, 目录结构如下:

```text
project/
├── config/
│   ├── constants.py             # 全局业务常量定义
│   └── settings.py              # 全局环境变量配置(全局路径配置等)
│
├── core/                        # 业务核心通用组件层 (封装业务逻辑与状态管理)
│   ├── client.py                  # 底层客户端连接(get_timecho_client等)
│   ├── metrics.py                 # 评估指标计算
│   ├── models.py                  # 业务数据模型(请求/响应)
│   ├── results.py                 # 业务结果持久化(CSV/JSON), 依赖utils.files
│   ├── resume.py                  # 策略控制器 (限流判断/断点续跑)
│   └── timecho.py                 # Timecho API 客户端
│
├── env/                         # 环境变量配置目录，供 `config/settings.py` 通过 load_dotenv() 加载
│   └── .env.example               # 环境变量示例文件
│
├── src/                         # SDK 源码目录
│   └── neuraxis_testkit/          # 测试工具箱
│       ├── log/                     # 日志管理
│       │   ├── __init__.py            # 对外暴露的统一接口
│       │   ├── config.py              # 变量配置
│       │   ├── context.py             # 上下文管理器 (LogLevelContext)
│       │   ├── core.py                # 核心 Logger 类
│       │   ├── decorators.py          # 装饰器 (log_execution, log_time)
│       │   ├── filters.py             # 过滤器 (ModuleLevelFilter, IgnoredLoggerFilter)
│       │   ├── formatters.py          # 格式化器 (ColoredFormatter)
│       │   └── logging.yaml           # 日志配置
│       ├── pytest_infra/            # Pytest 基础设施层
│       │   ├── __init__.py            # 对外暴露的统一接口
│       │   ├── collection.py          # 动态收集(调用manifest_loader)
│       │   ├── fixtures.py            # 全部 @pytest.fixture
│       │   ├── hooks.py               # 全部 hookimpl(含 pytest_addoption / pytest_configure)
│       │   ├── manifest_loader.py     # 加载YAML清单并生成参数化
│       │   ├── models.py              # 测试专用数据模型(如用例参数)
│       │   ├── paths.py               # NeuraxisPaths + get_paths (纯数据 + 访问器)
│       │   ├── resume.py              # 断点续跑逻辑(基于历史结果)
│       │   ├── session_manager.py     # 测试会话管理(共享资源、锁) SessionFileLock + SessionManager
│       │   └── test_recorder.py       # 测试结果记录器
│       └── utils/                   # 通用工具层
│          ├── __init__.py             # 对外暴露的统一接口
│          ├── assertions.py           # 通用断言(纯逻辑)
│          ├── concurrent.py           # 并发安全工具 (portalocker 封装)
│          ├── data_sanitizer.py       # 数据清洗与类型安全工具
│          ├── files.py                # 文件操作工具
│          └── runner.py               # 测试运行核心原语 (AST 静态发现 + 单用例执行 + 内存态结果追踪)
│
├── testcases/                  # 大模型业务测试用例
│   └── futureCovs/
│       └── dirtyData/
│           ├── test_dirty.py
│           └── data/             # 测试数据文件(用例依赖的输入)
│               └── test_dirty_s0.csv
│
├── outputs/                    # 运行时生成：日志、结果、HTML 报告
│   ├── results/                  # 业务结果(CSV/JSON)
│   ├── reports/                  # pytest报告(HTML/XML)
│   ├── analytics/                # 模型分析结果
│   └── logs/                     # 日志文件
│       └── tsfm_benchmark_20260824.log  # 文件名动态加上执行日期
│
├── conftest.py                 # 仓库级 pytest 适配入口
├── pyproject.toml              # 项目配置管理
├── README.zh-CN.md             # 项目说明文档, 提供项目概述、使用方法、注意事项等
└── .python-version
```

关键文件说明：

- `conftest.py`：仓库级 pytest 适配入口。用于声明项目级 CLI 选项，例如 `--project-root`、`--output-dir`、`--results-dir`、`--logs-dir`，并设置自定义报告头。通用 pytest hooks/fixtures 由 `neuraxis_testkit.pytest_infra` 通过 `pytest11` entry point 自动发现，不需要在 `conftest.py` 中手动桥接。
- `pyproject.toml`：项目配置、依赖声明、pytest 配置、`pytest11` 插件入口。
- `src/neuraxis_testkit/`：SDK 源码目录，后续会拆分为独立项目。
- `testcases/`：当前仓库中的业务测试用例，后续会拆分为独立业务测试仓库。

## 3. 测试流程

1. **配置初始化**: `config/settings.py` 通过 `load_dotenv()` 加载 `env/` 目录下的环境变量文件（如 `.env`），并导出全局路径与运行配置
2. **模型初始化**: 使用提供的 API 密钥对 TimechoAI 模型进行初始化.
3. **测试执行**: 根据提供的命令行参数执行指定的测试流程.
4. **结果输出**: 将测试结果输出至控制台或指定文件.

## 4. 命令与安装

- Python **3.12 或更高版本**
- 推荐使用虚拟环境

### 4.1 创建虚拟环境

```bash
python -m venv .venv
```

### 4.2 激活虚拟环境

请根据操作系统选择对应的命令:

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (CMD 命令提示符):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

> **Windows PowerShell 用户注意**: 若遇到"禁止运行脚本"的报错, 请以管理员身份运行 PowerShell 并执行:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4.3 退出虚拟环境

```bash
deactivate
```

### 4.4 安装项目依赖

推荐使用 editable 模式安装本项目。`-e` 表示 editable，即“可编辑安装”，源码变更后无需重新安装即可生效。

**开发/完整安装，推荐：**

```bash
python -m pip install -e ".[test]"
```

该命令会安装：

- 运行依赖：`timecho_ai`、`pandas`、`requests`、`pytest`、`portalocker`、`python-dotenv`、`pyyaml`
- 开发依赖：`pytest-xdist`、`pytest-html`、`pytest-cov`、`pytest-timeout`、`pytest-mock`、`pytest-randomly`、`pre-commit`

**仅安装运行依赖：**

```bash
python -m pip install -e .
```

**Windows 平台提示：**

`pyproject.toml` 中声明的是 `portalocker>=4.3.0`，它 **不会自动安装** `portalocker[win32]` 扩展。  
如果 Windows 下跨进程文件锁异常，请额外安装：

```bash
python -m pip install "portalocker[win32]"
```

### 4.5 刷新 entry points 与验证插件注册

当 `pyproject.toml` 中的 `[project.entry-points.pytest11]` 发生变化，或需要刷新本包元数据但不改动其他依赖时，可执行：

```bash
python -m pip install -e ".[test]" --force-reinstall --no-deps
```

安装后建议验证 pytest 插件是否注册成功：

```bash
python -c "import importlib.metadata as m; [print(ep) for ep in m.entry_points(group='pytest11') if 'neuraxis' in ep.name]"
```

预期能看到类似输出：

```text
EntryPoint(name='neuraxis_testkit_hooks', value='neuraxis_testkit.pytest_infra.hooks', group='pytest11')
EntryPoint(name='neuraxis_testkit_fixtures', value='neuraxis_testkit.pytest_infra.fixtures', group='pytest11')
```

## 5. 快速运行

项目已切换为 **pytest 原生模式**，不再使用 `run.py`。推荐统一使用 `python -m pytest`，以确保使用当前虚拟环境中的 Python 解释器。

**运行全部测试：**

```bash
python -m pytest
```

**按模块名或关键字运行：**

```bash
python -m pytest -k <module_name>
```

**按文件路径运行：**

```bash
python -m pytest testcases/path/to/test_file.py
```

**按标记运行：**

```bash
python -m pytest -m smoke
python -m pytest -m "not slow"
```

**并发运行：**

```bash
python -m pytest -n auto
```

**覆盖路径配置：**

```bash
python -m pytest \
  --project-root . \
  --output-dir ./outputs \
  --results-dir ./outputs/results \
  --logs-dir ./outputs/logs
```

默认 HTML 报告路径由 `pyproject.toml` 中的 `addopts` 控制：

```text
outputs/reports/report.html
```

## 6. pytest 配置说明

`pyproject.toml` 中的 `[tool.pytest.ini_options]` 已配置：

- `testpaths = ["testcases"]`
- 测试文件匹配：`test_*.py`
- 测试类匹配：`Test*`
- 测试函数匹配：`test_*`
- 默认启用详细输出、HTML 报告、严格标记、短 traceback
- 默认失败上限：`--maxfail=5`
- 日志 CLI 输出开启
- 自定义标记注册
- 自定义 session label 与日志 basename

`pytest11` entry points：

```toml
[project.entry-points.pytest11]
neuraxis_testkit_hooks = "neuraxis_testkit.pytest_infra.hooks"
neuraxis_testkit_fixtures = "neuraxis_testkit.pytest_infra.fixtures"
```

因此，只要通过 `pip install -e .` 或正式安装方式安装了本项目，pytest 就会自动加载 `neuraxis_testkit.pytest_infra` 中的 hooks 与 fixtures。

## 7. 测试目标

- 边缘场景探测: 针对复杂查询、多副本不一致、时间序列乱序写入等边界条件, 系统性验证模型的工程鲁棒性.
- 防御性架构验证: 以严格的工程标准施压, 检验模型在非理想输入下的退化行为与恢复能力.

## 8. 测试范围声明

本框架的测试结果受限于模型特定版本、数据预处理策略及运行环境. 本工具旨在为时序模型的工程防御性架构设计提供客观参考视角, 而非对任何商业产品最终性能的绝对断言.

## 9. 许可证

MIT License
