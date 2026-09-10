# TSFM Robustness Benchmark

[English](./README.md) | [中文](./README.zh-CN.md)

TSFM 鲁棒性基准测试是一种系统化的测试工具, 旨在检验时间序列基础模型在边缘场景(如频率不匹配、数据污染、协变量干扰等)下的工程鲁棒性.
本次版本包含对TimechoAI作为首个靶向模型的系统性评估, 更多模型将在后续迭代中逐步整合.

## 1. 核心架构 - 分层架构
- 本项目基于 **Python 3.12+** 开发, 核心依赖 `pytest`、`timecho-ai` 和 `pandas`.
- 系统采用清晰的分层架构, 确保业务逻辑、基础工具与测试执行解耦.
- 中间基础层原生支持 **跨平台运行** (Windows / macOS / Linux) 与 **并发执行** (基于 `pytest-xdist` 的分布式用例调度, 以及进程级并发控制).
- 测试工具箱 (`neuraxis_testkit`) 以 SDK 形式独立打包于 `src/` 目录下, 与业务代码解耦, 可直接复用至其他产品线.

## 2. 目录与文件规范

项目遵循标准分层架构, 目录结构如下:

- `config/`: 全局配置管理模块
   - `constants.py`: 全局常量定义.
   - `settings.py`: 全局环境变量配置(如 `TIMECHO_API_KEY`等).
- `core/`: 业务核心通用组件层 (封装业务逻辑与状态管理)
   - `client.py`: 底层客户端连接 (内部桥接模块, 业务侧请通过 `timecho.py` 间接调用).
   - `metrics.py`: 评估计算指标.
   - `models.py`: 共享数据模型 (TestStatus、TestResult、BatchReport).
   - `results.py`: 测试结果管理器 (批量缓冲/持久化).
   - `resume.py`: 策略控制器 (限流判断/断点续跑).
   - `timecho.py`: API 交互封装.
- `env/`: 环境变量配置目录，供 `config/settings.py` 通过 load_dotenv() 加载其中的环境文件
   - `.env.example`: 环境变量示例文件.
- `src/`: **SDK 源码目录**
  - `neuraxis_testkit/`: 测试工具箱
    - `log/`: 日志管理模块
      - `config.py`: 日志变量配置.
      - `context.py`: 上下文管理器 (`LogLevelContext`).
      - `core.py`: 核心 `Logger` 类.
      - `decorators.py`: 装饰器 (`log_execution`, `log_time`).
      - `filters.py`: 日志过滤器 (`ModuleLevelFilter`, `IgnoredLoggerFilter`).
      - `formatters.py`: 日志格式化器 (`ColoredFormatter`).
      - `logging.yaml`: 日志配置.
    - `pytest_infra/`: pytest 基础设施层 【开发中 / WIP】— 提供测试夹具、清单驱动参数化与断点续跑等基础设施封装. 当前处于调试完善阶段, 接口暂不稳定, 暂不建议外部直接依赖; 内部模块结构将在接口定稿后于文档中披露.
    - `utils/`: 基础工具层
      - `concurrent.py`: 并发控制与进程协同模块(内部桥接模块).
      - `data_sanitizer.py`: 数据清洗与类型安全工具.
      - `files.py`: 文件操作工具.
      - `runner.py`: 测试运行核心原语 (AST 静态发现 + 单用例执行 + 内存态结果追踪)
- `testcases/`: 业务场景测试用例
- `README.zh-CN.md`: 项目说明文档, 提供项目概述、使用方法、注意事项等.
- `conftest.py`: pytest 入口配置, 内部桥接至 [`neuraxis_testkit.pytest_infra.conftest`](https://github.com/Neuraxis-Labs/TSFM-Robustness-Benchmark/blob/main/src/neuraxis_testkit/pytest_infra/conftest.py), 以复用根级 fixtures 与 hooks.
- `run.py`: **项目统一入口**, 负责引导 `sys.path` 并按模块名或文件路径启动指定测试脚本. 【计划废弃】待 pytest 驱动链路调试稳定后, 该入口将由 pytest 直接替代并移除.

## 3. 测试流程

1. **配置初始化**: 从 `config/settings.py` 中读取环境变量.
2. **模型初始化**: 使用提供的 API 密钥对 TimechoAI 模型进行初始化.
3. **测试执行**: 根据提供的命令行参数执行指定的测试流程.
4. **结果输出**: 将测试结果输出至控制台或指定文件.

## 4. 命令与安装

### 4.1 创建虚拟环境

```bash
python -m venv .venv
```

### 4.2 激活虚拟环境

请根据您的操作系统选择对应的命令:

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

> **Windows PowerShell 用户注意**: 若遇到“禁止运行脚本”的报错, 请以管理员身份运行 PowerShell 并执行:   
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 4.3 退出虚拟环境

**退出虚拟环境(通用)** :   
```bash
  deactivate
```

### 4.4 安装项目依赖

> **注意：** 本项目需使用 **Python 3.12 或更高版本**.请在安装前确认您的 Python 版本.

在激活虚拟环境后首次执行以下命令安装核心依赖:

```bash
python -m pip install timecho-ai pandas pytest pytest-xdist portalocker pytest-html python-dotenv pyyaml
```

> ** 平台提示:** 
> 如果您的代码在 Windows 上运行, 为了确保跨进程文件锁的正常工作, 建议安装带有 Windows 扩展的 `portalocker`:
> ```bash
> python -m pip install "portalocker[win32]"
> ```

## 5. 快速运行

通过项目统一入口 `run.py` 启动测试:
```bash
# 按模块名启动测试
python run.py <module_name>

# 或按文件路径启动测试
python run.py <path/to/test_file.py>
```

## 6. 测试目标
- 边缘场景探测: 针对复杂查询、多副本不一致、时间序列乱序写入等边界条件, 系统性验证模型的工程鲁棒性.
- 防御性架构验证: 以严格的工程标准施压, 检验模型在非理想输入下的退化行为与恢复能力.

## 7. 测试范围声明
本框架的测试结果受限于模型特定版本、数据预处理策略及运行环境. 本工具旨在为时序模型的工程防御性架构设计提供客观参考视角, 而非对任何商业产品最终性能的绝对断言.

