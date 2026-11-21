# 需求细化：运行态临时文件与日志目录治理（昇腾 NPU）

> 版本：v0.1（需求细化稿）　日期：2026-11-20
> 范围决定：**仅昇腾 NPU 栈**（sglang + torch/torch_npu(PTA) + CANN + HCCL），NVIDIA 侧不在本期范围。
> 交付形态：需求文档 + 技术方案骨架。文中所有 `file:line` 均指本仓库（sglang fork）当前代码位置，作为方案落点与证据。

---

## 1. 需求背景与问题陈述

### 1.1 原声需求摘要

> 部署服务过程中，不清楚组件生成的临时文件（cache、算子编译元数据等）和日志（推理引擎、CANN、PTA 等）有哪些、默认路径在哪、是否可配置，影响问题定位和缓存复用。需要梳理运行态全量组件的临时文件和日志目录，做成环境变量可配置，并尽量收敛到同一个大目录（作业目录）下。

### 1.2 两个痛点场景展开

| 场景 | 描述 | 根因 |
|---|---|---|
| **S1 容器挂死/重启后日志难收集** | 作业调度平台通过 K8S 下发容器，容器挂死（被重启/回收）后，容器可写层与 `/tmp`、`/dev/shm` 中的内容随之丢失，难以收集全量日志定位问题 | 各组件日志/临时文件分散在 `stdout`、`~/ascend/log`、`/tmp`、`~/.cache` 等多处，且部分只打 stdout、从不落盘；没有统一目录可以"挂一个卷全收集" |
| **S2 目录无写权限导致拉起失败** | 容器内临时文件目录（如 `$HOME`、`/tmp`）映射到主机目录后无写权限，服务启动报错甚至无法拉起，且报错信息看不出是哪个组件、哪个路径出了问题 | 各组件对目录的创建/写入分散初始化，缺少启动前的统一"建目录 + 写探测"预检；报错发生在深层（CANN/文件系统层面），信息不可读 |

### 1.3 现状结论（一句话）

- **缓存类**：本 fork 已有统一缓存根 `SGLANG_CACHE_DIR`（默认 `~/.cache/sglang`），并在 import 时重定向 triton/inductor 等第三方缓存（`python/sglang/srt/environ.py:2702-2733`），但 **昇腾工具链侧的缓存（`~/.cache/ascend`、`~/.cache/torchair`）未被纳入**；
- **日志类**：引擎主日志只打 stdout/stderr，**默认不落盘**；CANN/HCCL 的 plog 落在 `~/ascend/log`，官方可配（`ASCEND_PROCESS_LOG_PATH` 等）但**引擎从未设置**；
- **临时文件类**：profiler 默认写 `/tmp`；ZMQ IPC socket 文件、下载锁文件等也散落在 `$TMPDIR`，多数**不可配**；
- 全量清单目前只存在于老员工脑子里，没有文档，也就谈不上"通过环境变量配置"。

---

## 2. 术语与缩写

| 术语 | 含义 |
|---|---|
| 作业目录 | 调度平台为一次作业分配/挂载的持久根目录（通常挂 PVC），容器重启后内容保留。本需求的目标收敛点 |
| plog | CANN 应用进程日志（slog 体系），按进程 pid 分文件，默认 `~/ascend/log/plog` |
| PTA | PyTorch Adapter，即 torch_npu（PyTorch 昇腾后端） |
| torchair | torch_npu 的图模式编译前端，NPU 上 `torch.compile` 的实际后端（`python/sglang/srt/utils/device.py:1273-1289`），其编译缓存是"算子编译元数据"的主体之一 |
| 算子编译缓存 | CANN 侧对 tbe/AscendC 算子编译结果的缓存，默认 `~/.cache/ascend` |
| OPP | 算子仓库（安装态，`/usr/local/Ascend/.../opp`），只读，不属于运行态临时文件，仅作边界说明 |

---

## 3. 范围

### 3.1 在范围内

1. **推理引擎**：sglang（本 fork）全部运行态进程（http_server / tokenizer / scheduler / detokenizer / DP controller / rust server 子进程等）；
2. **推理框架层**：torch + torch_npu（PTA）、torchair、`torch.utils.cpp_extension` 即时编译；
3. **工具链层**：CANN（ACL Runtime / GE 图引擎 / 算子编译 / msprofiler / msprobe）、HCCL（日志随 slog 体系）；
4. **模型与数据仓库缓存**：HuggingFace hub、ModelScope 下载缓存（影响拉起与磁盘占用，属"运行态生成的临时数据"）；
5. **OS 层**：`$TMPDIR`(`/tmp`)、`/dev/shm`、容器内 coredump 路径的治理建议。

### 3.2 不在范围内（边界）

- NVIDIA/CUDA 栈（triton CUDA、CUDA core dump 等）——本期不做，附录仅一句话带过；
- 平台侧组件（K8S sidecar、监控 agent、DCGM/Ascend dcgm 等主机侧采集）；
- NPU 驱动与主机侧日志（`/var/log/npu*`、`/var/log/ascend`，需改主机 slog 配置，属平台责任，文档中说明收集接口即可）；
- 镜像构建期产物、OPP 安装目录；
- MindSpore 后端（仓库有 `docs/docs/hardware-platforms/ascend-npus/mindspore_backend.mdx`，如启用需补充 `MS_COMPILER_CACHE_PATH` 等，列为开放问题）。

---

## 4. 现状盘点：运行态文件全量清单（昇腾栈）

> 结论标记：✅=已可配且纳入统一根；⚠️=官方/代码可配但引擎未接管；❌=当前不可配。
> "引擎落盘证据"列为本仓库代码位置；CANN 侧为官方文档结论（见 §10 参考资料）。

### 4.1 日志类

| # | 组件 | 内容 | 默认路径 | 配置手段 | 状态 |
|---|---|---|---|---|---|
| L1 | sglang 引擎（全部进程） | 主日志（调度/加载/请求处理） | **stdout/stderr，不落盘** | `--log-level`；`SGLANG_LOGGING_CONFIG_PATH`（可选 dictConfig JSON，可装 FileHandler） | ⚠️ 无默认落盘文件（`python/sglang/srt/utils/log_utils.py:232-252`） |
| L2 | sglang 请求日志 | 每请求一行，`{hostname}_{rank}.log`，按小时轮转 | 关闭；开启且指定目录后落盘 | `--log-requests-target stdout <dir>...`、`--log-requests-level/-format`（`python/sglang/srt/observability/request_logger.py:129-152`；`log_utils.py:56-78`） | ⚠️ 默认关 |
| L3 | sglang 指标导出 | `sglang-request-metrics-<hour>.log` | 关闭 | `--export-metrics-to-file` + `--export-metrics-to-file-dir`（`observability/request_metrics_exporter.py:81-127`） | ⚠️ 默认关 |
| L4 | sglang 调度器状态 dump | 周期性调度状态快照 | 关闭 | `SGLANG_LOG_SCHEDULER_STATUS_TARGET`（可含目录）、`SGLANG_LOG_SCHEDULER_STATUS_INTERVAL`，需 `--enable-metrics`（`utils/scheduler_status_logger.py:14-33`） | ⚠️ 默认关 |
| L5 | sglang kernel-API 调试日志/张量 dump | `inputs.pt/outputs.pt/metadata.json` | stdout；dump 目录为 CWD 相对路径 `sglang_kernel_api_dumps` | `SGLANG_KERNEL_API_LOGDEST/LOGLEVEL/DUMP_DIR`（`python/sglang/kernels/kernel_api_logging.py:55-61,297-329`） | ⚠️ 默认相对 CWD |
| L6 | CANN/PTA/HCCL plog | ACL Runtime、GE、HCCL、torchair 等所有 CANN 侧进程日志 | `~/ascend/log/plog`（另有 `~/ascend/log/{atc,debug,driver}` 等） | 路径优先级 `ASCEND_PROCESS_LOG_PATH` > `ASCEND_WORK_PATH` > 默认；级别 `ASCEND_GLOBAL_LOG_LEVEL`（0 debug / 1 info / 2 warn / 3 error，默认值随 CANN 版本）；打屏 `ASCEND_LOG_PRINT_TO_STDOUT`（旧名 `ASCEND_SLOG_PRINT_TO_STDOUT` 已废弃） | ⚠️ 官方可配，**引擎与镜像均未设置**（全仓 grep 无此变量） |
| L7 | msprobe（昇腾精度 dump） | 算子输入输出 dump | 由 `--msprobe-dump-config` 所指 JSON 决定 | 已接入：`arg_groups/fields/observability.py:244-247`、`model_executor/model_runner_components/misc_utils.py:54-68` | ✅ |

### 4.2 缓存 / 编译产物类

| # | 组件 | 内容 | 默认路径 | 配置手段 | 状态 |
|---|---|---|---|---|---|
| C1 | sglang 统一缓存根 | 下述 C2~C8 的父目录 | `~/.cache/sglang` | `SGLANG_CACHE_DIR`（`environ.py:1533`） | ✅ 已有统一根 |
| C2 | triton-ascend / inductor / cuda / tilelang / humming / flashinfer 缓存重定向 | 第三方编译缓存 | 由 C1 派生（`<root>/triton` 等） | import 时 setdefault，底层变量预设则优先（`environ.py:2702-2733`、`python/sglang/__init__.py:12-15`） | ✅ 机制现成，**可复用于昇腾侧注入** |
| C3 | sglang piecewise torch.compile 缓存 | 图代码 + 元数据 | `<root>/torch_compile_cache/<hash>/...` | 随 C1（`srt/compilation/backend.py:408-475`） | ✅ |
| C4 | sgl-kernel JIT / cute_aot / rust_extensions / deep_gemm / p2p 探测缓存等 | 引擎自有编译与探测缓存 | `<root>/jit`、`<root>/cute_aot`、`<root>/rust_extensions`、`<root>/deep_gemm` 等 | 各自 `SGLANG_*_CACHE_DIR`，未设时落 C1（`environ.py:1521-1557`） | ✅ |
| C5 | **CANN 算子编译缓存**（含 tbe/AscendC 编译元数据） | 算子二进制与编译缓存 | `~/.cache/ascend` | `ASCEND_CACHE_PATH`；容量上限 `ASCEND_MAX_OP_CACHE_SIZE`；跨 CANN 版本不保证兼容 | ⚠️ **引擎未接管**（全仓无引用） |
| C6 | CANN 进程独享文件 | 运行期独享中间文件 | （随 `ASCEND_WORK_PATH`） | `ASCEND_WORK_PATH`（亦影响 plog 路径，优先级低于 `ASCEND_PROCESS_LOG_PATH`） | ⚠️ 引擎未接管 |
| C7 | **torchair 图编译缓存**（NPU 上 torch.compile 的主体缓存） | 图编译产物 | `~/.cache/torchair` | `TORCHAIR_CACHE_DIR` | ⚠️ **引擎未接管** |
| C8 | torch cpp_extension 即时编译 | `load_inline` 构建产物 + 锁文件 | torch 默认 `~/.cache/torch_extensions` | `TORCH_EXTENSIONS_DIR`（torch 所有；CI 曾临时设 `/tmp/torch_extensions`，见 `.github/workflows/_npu-pr-test-stage.yml:82-88`） | ⚠️ 未随 C1 |
| C9 | HuggingFace / ModelScope 模型仓库 | 权重/tokenizer 下载缓存 | `~/.cache/huggingface/hub`、`~/.cache/modelscope/hub` | `HF_HOME`/`HF_HUB_CACHE`/`MODELSCOPE_CACHE`（torch/hf 所有，引擎按原样读，如 `srt/utils/common.py:533-555`） | ⚠️ 平台侧通常自挂 |
| C10 | （附录）MindSpore 编译缓存 | 图/算子编译缓存 | `~/.cache/mindspore` | `MS_COMPILER_CACHE_PATH` | ⏸ 视是否启用该后端 |

### 4.3 临时文件 / IPC 类

| # | 组件 | 内容 | 默认路径 | 配置手段 | 状态 |
|---|---|---|---|---|---|
| T1 | torch_npu profiler（msprof 文本/tensorboard 输出） | `..._ascend_pt/`、`ASCEND_PROFILER_OUTPUT/`（trace_view.json 等） | `/tmp` | `SGLANG_TORCH_PROFILER_DIR`（`managers/scheduler_components/profiler_manager.py:130-135,242-264`；文档 `ascend-npus/optimization/profiling.mdx`） | ⚠️ 可配但默认 `/tmp`（S1 场景下即丢） |
| T2 | graph capture profiler（NPU） | 图捕获 trace | `/tmp/graph_capture_profile` | 同上 + `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE`（`hardware_backend/npu/graph_runner/npu_cudagraph_backend.py:361-386`） | ⚠️ |
| T3 | ZMQ `ipc://` socket 文件（tokenizer/scheduler/detokenizer/metrics 等全链路） | socket 文件，进程存活期存在 | `$TMPDIR/tmp*`（NamedTemporaryFile） | **无**（`server_args.py:892-931`、`entrypoints/http_server.py:267-270`） | ❌（随 `TMPDIR` 间接可控） |
| T4 | `/dev/shm` 共享内存段 | load snapshot、多模态传输、host pool、multi-tokenizer args、offloader 段 | `/dev/shm/sglang_*` | **无**（`managers/load_snapshot.py:300-313`、`managers/mm_utils.py:1425-1442` 等；K8S 模板已给 emptyDir 16Gi，`test/ascend/e2e/k8s_single.yaml.jinja2:85-119`） | ❌（按容器 tmpfs 治理即可，不建议改） |
| T5 | 模型下载锁 / 连接器暂存 / 临时 JSON | `.lock`、`mkdtemp` 目录、tmp json | `$TMPDIR` | **无**（`model_loader/weight_utils.py:167-185`、`connector/base_connector.py:23-32`） | ❌（短生命周期，间接随 `TMPDIR`） |
| T6 | Prometheus 多进程指标 | `*.db` | `$TMPDIR/tmp*` | `PROMETHEUS_MULTIPROC_DIR`（`entrypoints/server_utils.py:38-58`） | ⚠️ |
| T7 | NCCL 对称显存分配器 inline 编译（NPU 上不启用，列为完备性） | 构建目录 | `$TMPDIR/symm_allocator` | **无**（`distributed/device_communicators/pynccl_allocator.py:273-295`） | ❌（昇腾路径不触发） |
| T8 | HiCache/NPU memcache 等后端存储 | KV 落盘目录 | 须显式设置 | `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` 等（`environ.py:906-994`） | ✅（本就强制配置） |

### 4.4 转储 / 调试产物类

| # | 组件 | 内容 | 默认路径 | 配置手段 | 状态 |
|---|---|---|---|---|---|
| D1 | 请求崩溃 dump | `crash_dump_<ts>.pkl` + 标记文件 | 关闭 | `--crash-dump-folder`（`observability/request_dump.py:98-149`） | ⚠️ 默认关 |
| D2 | 专家分布记录 | `torch.save` 产物 | `/tmp` | `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`（`environ.py:1487`） | ⚠️ 默认 `/tmp` |
| D3 | 张量级 debug dump | `<folder>/...tags....pt` | 关闭 | `--debug-tensor-dump-output-folder`（`debug_utils/dumper.py:590-640`） | ⚠️ |
| D4 | CANN 异常算子 dump | 算子 dump 数据 | CANN 默认 | `ASCEND_DUMP_PATH` | ⚠️ 引擎未设置 |
| D5 | 容器 coredump | core 文件 | 容器 `core_pattern`（平台侧） | `ulimit -c` + `core_pattern`（部署文档给出建议值即可） | ➖ 平台协作 |

### 4.5 现状小结（需求 GAP）

1. **G1 缺统一"作业根目录"入口**：缓存有 `SGLANG_CACHE_DIR`，但日志、profiler、dump、`/tmp` 类各自为政；
2. **G2 昇腾工具链缓存脱管**：`ASCEND_CACHE_PATH`、`ASCEND_WORK_PATH`、`TORCHAIR_CACHE_DIR` 未被引擎注入，散在 `~/.cache` 下（S2 场景的典型受害者：`~` 挂只读卷时 CANN 首次建缓存失败、报错深、EngineCore 拉不起来）；
3. **G3 日志不落盘**：引擎主日志仅 stdout，plog 在 `~/ascend/log`，两者都不在作业目录（S1 场景收集不全的直接原因）；
4. **G4 无启动预检与路径自描述**：没有任何"启动时列出全部将写路径并做写探测"的机制（S2 场景报错不可读的原因）。

---

## 5. 目标与非目标

### 5.1 目标（重构后的需求陈述）

为昇腾栈提供**一个环境变量入口**（作业根目录）+ **分层覆盖**的运行态文件路径体系：设置一个根变量后，引擎及昇腾工具链的**全量缓存、日志、profiler、dump** 均落在该根目录下约定的子目录中；各组件仍可用自身既有变量精确覆盖；启动时输出**路径生效清单**并对全部将写目录做**写权限预检**，预检失败即 fail-fast 并给出可读报错；不设置任何新变量时行为与现状完全一致。

### 5.2 非目标

- 不改变各组件日志的**内容与格式**（只管"写到哪"）；
- 不做日志采集/上报系统本身（只保证"可被一次挂载/一次拷贝收集"）；
- 不治理 `/dev/shm`（容器 tmpfs 语义，由平台挂载策略管理）；
- 不覆盖 NVIDIA 栈与主机侧驱动日志。

---

## 6. 功能需求（FR）

> 优先级：P0 = 首个可用版本必须；P1 = 紧随其后；P2 = 增强。

### FR1（P0）全量清单文档化
将 §4 盘点表落为仓库正式文档（`docs/docs/hardware-platforms/ascend-npus/reference/` 下新增"运行态文件与目录参考"页），并纳入版本维护：新增写盘点必须同步登记（可在 CI 或 code review checklist 中约束）。**直接对应原验收标准"全量临时文件/日志目录梳理"。**

### FR2（P0）统一作业根目录
新增 sglang 侧根变量（候选名 `SGLANG_WORKSPACE_DIR`，命名遵循 `.agents/skills/env-var-conventions`：注册进 `environ.py` 的 `Envs`，`EnvStr(None)`，未设置=完全维持现状）。约定二级布局：

```
$SGLANG_WORKSPACE_DIR/
├── cache/        # 缓存与编译产物（可指向共享缓存卷）
│   ├── sglang/   #   = SGLANG_CACHE_DIR 派生默认
│   ├── ascend/   #   = ASCEND_CACHE_PATH 派生默认（~/.cache/ascend 的替代）
│   ├── torchair/ #   = TORCHAIR_CACHE_DIR 派生默认
│   └── ...
├── log/          # 日志（作业私有，必须持久）
│   ├── engine/   #   引擎主日志 + 请求/指标日志
│   └── cann/     #   = ASCEND_PROCESS_LOG_PATH 派生默认（plog/atc 等）
├── tmp/          # 临时文件（默认注入 TMPDIR）
├── profile/      # profiler 输出（SGLANG_TORCH_PROFILER_DIR 派生默认）
└── dump/         # crash dump / expert distribution / CANN ASCEND_DUMP_PATH
```

覆盖优先级（必须写进文档与实现）：**组件既有变量 > 根目录派生默认 > 组件原默认**。即用户显式设置了 `TORCHAIR_CACHE_DIR` 等变量时以用户为准；实现采用 `setdefault` 语义，与既有 C2 机制一致。

### FR3（P0）缓存类统一注入（收编 G2）
在 `environ.py:2702-2733` 既有"第三方缓存重定向"函数中追加昇腾侧变量：`ASCEND_CACHE_PATH`、`ASCEND_WORK_PATH`、`TORCHAIR_CACHE_DIR`（以及 `TORCH_EXTENSIONS_DIR` 视评审结论）。约束：
- 仅在 `SGLANG_WORKSPACE_DIR` 已设置（或对应 `SGLANG_CACHE_DIR` 已设置）时注入，未设置时不动 `os.environ`（零行为变化）；
- 注入必须发生在 **torch_npu/CANN 初始化之前**（当前 C2 机制由 `python/sglang/__init__.py:12-15` 在 import 期触发，时机天然满足，方案沿用）；
- `ASCEND_*`/`TORCHAIR_*` 为 vendor 前缀，按 env-var-conventions 决策表作为"上游别名"注册进 `Envs` 以便测试 `.override()`，不改为 `SGLANG_` 前缀。

### FR4（P0）日志类统一（收编 G3）
- 新增 `SGLANG_LOG_DIR`（默认 `None`=现状 stdout-only）。设置后：引擎主日志双写（stdout 保留 + `<dir>/engine/{hostname}_{role}_{rank}.log`，复用 `log_utils.py:56-78` 的 TimedRotatingFileHandler 机制）；`--log-requests-target`、`--export-metrics-to-file-dir` 未显式给目录时的派生默认指向 `<dir>/engine/`；
- `SGLANG_WORKSPACE_DIR` 已设置而 `SGLANG_LOG_DIR` 未设置时，`SGLANG_LOG_DIR` 取 `<root>/log/engine`；
- 注入 `ASCEND_PROCESS_LOG_PATH=<root>/log/cann`（优先级高于 `ASCEND_WORK_PATH`，见 §4.1 L6），并文档化 `ASCEND_GLOBAL_LOG_LEVEL`（排障期建议 debug）与 `ASCEND_LOG_PRINT_TO_STDOUT` 的用法；
- 每文件带 `RUN_LABEL`/时间戳前缀（多副本、多次重启不互相覆盖）。

### FR5（P0）启动预检 + 路径生效清单（收编 G4，直接回应 S2）
新增统一 preflight（建议 `srt/entrypoints` 最前置调用单函数，如 `workspace_preflight()`）：
1. 汇总本进程将写的全部目录（按 §4 清单动态收集：含引擎侧、已注入的 ASCEND/TORCHAIR 变量、profiler/dump 目标）；
2. 逐目录 `mkdir -p` + 写探针文件 + 删除，`os.access` 检查；
3. 任一失败 → 立即 fail-fast，报错格式固定：`[preflight] 组件=<name> 路径=<path> 环境变量=<var> 错误=<errno/原因> 修复建议=<...>`；
4. 全部通过 → 以**路径生效清单**逐行打印（同格式），作为启动日志的第一屏，问题定位时"看首屏即知全貌"。

### FR6（P1）规范与文档
- 新变量全部注册 `Envs`（`environ.py` 对应 section）；昇腾 vendor 变量按别名注册；
- 更新 `docs/docs/references/environment_variables.mdx` 与 `docs/docs/hardware-platforms/ascend-npus/reference/environment_variables.mdx`，新增 FR1 的目录参考页与"K8S 部署挂载建议"（作业目录=一个 PVC 挂载点即可全收集；缓存子目录可单独绑定共享缓存卷）。

### FR7（P1）收集友好（回应 S1）
- 提供 `scripts/collect_logs.sh`（或引擎内置 `/dump_runtime_info` 管理接口，二选一评审）：打包 `<root>/log`、最近 N 个 profile、crash dump、环境快照（env 生效清单 + `pip list` + CANN/torch_npu 版本 + NPU 型号）为单 tar；
- 日志 flush 策略：文件 handler 默认无缓冲或短间隔 flush，保证容器被强杀时日志尽量完整（hang 场景）。

### FR8（P1）缓存复用指引
- 文档明确各缓存的 key 构成与失效条件（torchair/CANN 缓存跨 CANN 版本不保证兼容，升版需清理重建——参照 torchair cache_compile 说明）；
- 推荐拓扑：`cache/` 指向节点级共享卷（只放缓存），`log/`、`tmp/`、`dump/` 指向作业私有 PVC；并发写安全依赖各组件既有锁机制（C4 已有 filelock/flock）；
- 提供容量治理建议：`ASCEND_MAX_OP_CACHE_SIZE`、`SGLANG_JIT_CACHE_KEEP`（`environ.py:1553`）。

### FR9（P2）hang 场景增强（可选）
- faulthandler 输出重定向到 `<root>/log/engine/faulthandler.log`（当前仅 stderr，`managers/scheduler.py:7052`）；
- 看门狗超时时自动触发一次 §4.4 的 crash dump 与日志 flush。

---

## 7. 技术方案骨架（改动落点）

| # | 改动 | 文件（本仓库） | 说明 | 优先级 |
|---|---|---|---|---|
| 1 | 新增 `SGLANG_WORKSPACE_DIR`、`SGLANG_LOG_DIR` 等字段 | `python/sglang/srt/environ.py`（Envs + section 注释） | 按 env-var-conventions Rule 1/2；`EnvStr(None)` 区分"未设置" | P0 |
| 2 | 扩展缓存重定向：注入 `ASCEND_CACHE_PATH/ASCEND_WORK_PATH/TORCHAIR_CACHE_DIR(/TORCH_EXTENSIONS_DIR)` | `environ.py:2702-2733` 的重定向函数（import 期由 `python/sglang/__init__.py:12-15` 触发） | setdefault 语义；vendor 变量同时注册 `Envs` 别名 | P0 |
| 3 | `ASCEND_PROCESS_LOG_PATH` 注入（须在 torch_npu import 前生效） | 同上（与 #2 同一时机） | plog 路径只认进程启动时环境，晚于 CANN 初始化注入无效——**时机是本方案最大风险点**，需在 #2 的 import 钩子内完成 | P0 |
| 4 | 引擎主日志双写 + 轮转 | `python/sglang/srt/utils/log_utils.py`（`configure_logger`，`log_utils.py:232-252`；复用 `:56-78` 文件 handler） | stdout 保留；按 role/rank 命名 | P0 |
| 5 | preflight + 路径清单 | 新文件 `python/sglang/srt/utils/workspace_preflight.py`；在 `srt/entrypoints/{http_server,engine}.py` 最前置调用 | 目录收集器复用 §4 清单的注册表（建议维护一个集中 `RUNTIME_WRITE_PATHS` 声明，供 preflight 与 FR1 文档同源生成） | P0 |
| 6 | profiler 默认目录派生 | `profiler_manager.py:130-135`、`observability/profile_utils.py:38-71`、`npu_cudagraph_backend.py:361-386` | 未设 `SGLANG_TORCH_PROFILER_DIR` 且设了根目录 → `<root>/profile`；原默认 `/tmp` 仅在两者皆未设时保留 | P1 |
| 7 | dump 类默认派生 | `environ.py:1487`（expert distribution 默认 `/tmp`）、`--crash-dump-folder` 帮助文案 | 同上规则 | P1 |
| 8 | 收集脚本 | `scripts/collect_logs.sh` 或管理接口 | 打包作业目录内产物 + 环境快照 | P1 |
| 9 | 文档 | `docs/docs/hardware-platforms/ascend-npus/reference/`（新增目录参考页 + 环境变量页更新）、K8S 挂载建议、FAQ 链接 | FR1/FR6 | P0/P1 |
| 10 | 镜像/模板 | `docker/npu.Dockerfile`（无 entrypoint，无需改）；`test/ascend/e2e/k8s_*.yaml.jinja2` 增加 workspace 挂载示例 | CI 先行验证整套机制 | P1 |

**风险与对策**
- R1 注入时机晚于 CANN 初始化 → plog 路径不生效。对策：全部注入集中在 `sglang/__init__` import 钩子；验收用例 AC2 专门覆盖"plog 实际落在新目录"。
- R2 多副本/多进程写同一目录 → 日志按 `{hostname}_{role}_{rank}` 命名（既有惯例）；CANN 缓存本身支持多进程并发（官方语义），必要时加 `ASCEND_MAX_OP_CACHE_SIZE`。
- R3 权限模型：容器以非 root 运行 + 共享缓存卷首建者为 root → 复现 vLLM-Ascend 已知问题。对策：preflight 明确报错 + 文档要求缓存卷初始化（fsGroup/初始化 Job）。
- R4 缓存跨版本失效导致"共享卷膨胀"→ FR8 的容量治理与清理责任写清。
- R5 兼容性：所有注入均为 setdefault，任何变量被显式预设时用户优先；不设根变量时零改动（AC4 回归保障）。

**分期建议**：P0（FR1–FR5，改动 #1–#5、#9 文档）→ P1（FR6–FR8，#6–#8、#10）→ P2（FR9）。

---

## 8. 验收标准（可执行用例）

> 与原验收标准的映射：原"全量临时文件目录梳理，可通过环境变量配置" → AC1/AC2/AC4；原"全量日志目录梳理，可通过环境变量配置" → AC1/AC3/AC4。

| # | 用例 | 通过判据 |
|---|---|---|
| AC1 | 文档评审：目录参考页覆盖 §4 全量表（组件/内容/默认路径/环境变量/优先级） | 评审通过；文档与代码内 `RUNTIME_WRITE_PATHS` 注册表同源 |
| AC2 | 设置 `SGLANG_WORKSPACE_DIR=/mnt/job`（单变量）拉起服务，检查首屏"路径生效清单"，并逐一 `find` 验证 | engine 日志文件、`plog`（`ASCEND_PROCESS_LOG_PATH` 生效）、`ASCEND_CACHE_PATH`、`TORCHAIR_CACHE_DIR`、profiler 输出均实际位于 `/mnt/job` 对应子目录 |
| AC3 | 同 AC2 环境，运行一段时间后 `kill -9` 容器（模拟挂死重启） | 作业目录内可完整取得：引擎日志（至 kill 前）、请求日志（若开）、plog、指标导出（若开） |
| AC4 | **不设置任何新变量**拉起服务（回归） | 全部路径与现状一致（`~/.cache/sglang`、`~/ascend/log`、stdout-only 等），无行为差异 |
| AC5 | 负向：将 `SGLANG_WORKSPACE_DIR` 指向只读挂载目录拉起服务 | 启动 fail-fast；报错包含组件名、路径、环境变量名、原因与修复建议（格式见 FR5） |
| AC6 | 覆盖优先级：同 AC2 但显式预设 `TORCHAIR_CACHE_DIR=/x` | torchair 缓存落 `/x`，其余仍落作业目录 |
| AC7 | 缓存复用：共享缓存卷预热后二次拉起 | 二次启动显著快于冷启动（torchair/CANN 编译缓存命中，对比启动耗时） |
| AC8 | CI：K8S 模板按建议挂载跑通一条 NPU e2e 用例 | 现有 e2e 框架（`test/ascend/e2e/`）通过，且用例产物/日志按新布局落盘 |

---

## 9. 开放问题（需评审/与平台方确认）

1. **根变量命名与注入约定**：`SGLANG_WORKSPACE_DIR` 是否符合平台已有约定？平台是否愿意直接透传作业目录到该变量（否则需要平台侧加一层映射）？
2. **日志默认行为**：设置了根变量后，引擎主日志是否**默认**双写文件（方案倾向是），还是仍需显式 `SGLANG_LOG_DIR`/开关？轮转大小/保留天数默认值？
3. **CANN 日志级别默认值**：生产默认维持现状（随 CANN 版本，通常较低）还是引擎统一设 INFO（量增大）？排障期一键 debug 的开关归平台还是引擎？
4. **共享缓存卷的责任边界**：谁负责初始化权限与周期清理；跨 CANN 版本升级时的清缓存流程责任人。
5. **MindSpore 后端**是否纳入本期（`MS_COMPILER_CACHE_PATH`、MindSpore 自身日志路径）。
6. **`/tmp` 类不可配项**（T3/T5）是否需要为"作业目录 tmp"做迁移（改 `TMPDIR` 即可大部分覆盖），还是接受随容器生命周期丢弃。
7. coredump：平台是否统一配置容器 `core_pattern` 指向作业目录（D5），引擎侧只在文档配合。

---

## 10. 参考资料

代码证据均在本仓库，代表性锚点：
- 统一缓存根与第三方重定向：`python/sglang/srt/environ.py:1533`、`environ.py:2702-2733`、`python/sglang/__init__.py:12-15`
- NPU torch.compile=torchair：`python/sglang/srt/utils/device.py:1273-1289`
- NPU profiler：`python/sglang/srt/managers/scheduler_components/profiler_manager.py:242-264`、`docs/docs/hardware-platforms/ascend-npus/optimization/profiling.mdx`
- CI 对 plog 的既有收集（佐证默认路径 `~/ascend/log/debug/plog`）：`python/sglang/test/ascend/e2e/run_npu_testcase.sh:155-165`
- 环境变量规范：`.agents/skills/env-var-conventions/SKILL.md`

CANN/torchair 官方文档：
- [ASCEND_PROCESS_LOG_PATH 环境变量参考（昇腾社区）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/envvar/envref_07_0120.html)
- [ASCEND_LOG_PRINT_TO_STDOUT（昇腾社区）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/maintenref/envvar/docs/zh/env_vars/ASCEND_LOG_PRINT_TO_STDOUT.md)、[ASCEND_SLOG_PRINT_TO_STDOUT（deprecated）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/maintenref/envvar/docs/zh/env_vars/ASCEND_SLOG_PRINT_TO_STDOUT_deprecated.md)
- [如何查看并读懂昇腾平台的应用日志（昇腾社区）](https://www.hiascend.com/developer/techArticles/20230321-3)、[【CANN 文档速递】了解与使用日志](https://zhuanlan.zhihu.com/p/566451778)
- [CANN 图开发指南·环境准备（ASCEND_CACHE_PATH / ASCEND_WORK_PATH）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/programug/graphdevg/docs/zh/user_guides/graph_dev/overview/environment_setup.md)
- [CANN 模型迁移·算子编译与图编译（ASCEND_MAX_OP_CACHE_SIZE 等）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/programug/migratdevg/docs/zh/user_guides/porting/op_support/op_run)
- [torchair 编译缓存官方文档（Ascend/torchair）](https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/compile_cache.md)
- [vLLM-Ascend：持久化 csrc 构建缓存实践（缓存权限问题案例）](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/developer_guide/Design_Documents/persistent_csrc_build_cache.html)
- [MindSpore 环境变量列表（MS_COMPILER_CACHE_PATH）](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/env/env_var_list.html)
