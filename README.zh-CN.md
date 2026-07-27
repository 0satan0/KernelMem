## 项目概述：KernelMem

KernelMem 是一个**基于 PyTorch 模型代码的自动 CUDA kernel 生成与优化系统，并增强了“长短期记忆”机制**。
其核心思想是：从 PyTorch 的前向传播代码开始，系统利用大语言模型（LLM）迭代生成候选 CUDA kernel，并结合历史优化经验、性能/正确性反馈以及 kernel 优化的专家知识构建一个“记忆环路”，从而持续演进出更高效的 kernel。长期记忆组件涵盖了 kernel 优化的通用知识和最佳实践，使系统能够跨任务利用已验证的优化策略。

该项目的主要入口是 `main_memory_latest.py` 中的 `main()` 函数（在直接运行脚本时触发）。

---

## 核心特性

- **PyTorch 算子/模型到 CUDA kernel 的自动迁移**
  - 自动从 PyTorch 任务脚本中读取算子/网络定义。
  - 根据任务构建 LLM 提示词（Prompts），并要求模型生成相应的 CUDA kernel。

- **带有“记忆”的多轮自我演进**
  - 对于每一轮生成的 kernel，系统会记录：
    - 正确性结果（是否能运行，是否通过数值检查）
    - 性能指标（加速比，NVIDIA Nsight Compute / Nsight Systems 指标等）
    - 采用的优化策略、失败原因、修复历史
  - 这些信息被写入 `code/`、`evaluation/`、`profile/` 等目录，随后作为短期记忆反馈，指导后续的 kernel 生成与修复。

- **自动基准测试与错误修复**
  - 使用 `utils/compile_and_run.py` 对生成的 kernel 进行编译和基准测试：
    - 将数值误差与 PyTorch 参考实现进行对比（`tol`）。
    - 测量平均前向延迟，并计算 **speedup = ref_latency / test_latency**。
  - 针对编译错误 / 运行时错误 / 精度失败：
    - 通过 `prompts/judger_repair_memory.py` 和 `prompts/error_memory.py` 构建“感知记忆”的错误分析与修复提示词。
    - 要求 LLM 基于历史错误日志和修复记录生成更可靠的 kernel 版本。

- **NCU & NSYS Profiling 驱动的优化**
  - 通过 `run_ncu_memory.py` 调用 NVIDIA Nsight Compute (`ncu`) 以获取细粒度的性能指标：
    - 内存效率、SM 利用率、启动/占用率、瓶颈阶段等。
  - 通过 `run_nsys.py` 调用 Nsight Systems (`nsys`) 以测量 kernel 启动次数和运行时行为。
  - 这些分析结果由 `prompts/judger_optimization_memory_latest.py` / `prompts/optimization_memory_latest.py` 转换为优化建议，进而驱动新一轮的 kernel 生成。

---

## 代码结构

- **`main_memory_latest.py`**: 项目主入口
  - 解析命令行参数（任务选择、GPU、LLM 设置、迭代轮数等）。
  - 调用 LLM 进行 kernel 的生成 / 修复 / 优化。
  - 编排基准测试、NCU/NSYS 分析、可视化和汇总。

- **`KernelBench/`**: PyTorch 参考任务
  - `level1`, `level2`: 各种基础算子和小规模子网络。
  - `level3`: 代表性的深度学习模型（ResNet, VGG, LSTM, Transformer 等）。

- **`prompts/`**: 提示词设计与“记忆机制”
  - `generate_custom_cuda_memory.py`: 首轮 kernel 生成的种子提示词。
  - `optimization_memory_latest.py`: 将历史 kernel 与分析结果融合的优化提示词。
  - `judger_*_memory*.py`: 针对优化策略、编译超时、运行时错误等进行评判和分析的模块，并产生修复/优化建议。
  - `few_shot/`: 提供给 LLM 的少样本示例。

- **`memorybank/`**:
  - 存储关于硬件瓶颈和 kernel 结构的先验知识。
  - 这些知识充当“长期记忆”，被注入到提示词中以引导更好的优化选择。

- **`utils/`**:
  - `compile_and_run.py`: 负责编译、运行、比对精度并测量性能。
  - `kernel_io.py`: 从 LLM 回复中提取代码块，保存为 Python/CUDA 文件，并读写指标。

- **`agents/query_server.py`**:
  - 与实际 LLM 后端（OpenAI, 本地 vLLM/sglang 等）通信的统一接口。

---

## 环境要求

建议在 **Linux + NVIDIA GPU** 环境下运行（在 Windows 上您需要自行准备 CUDA 工具链和 Nsight 工具）。
典型依赖（仅供参考；请根据您的环境调整版本）：

- Python 3.9+
- PyTorch (带 GPU 支持)
- CUDA Toolkit 及匹配的驱动程序
- NVIDIA Nsight Compute (`ncu`) 和 Nsight Systems (`nsys`)
- Python 包：
  - `matplotlib`
  - `pandas`, `numpy` (用于处理 profiling CSV 文件)
  - 对应 LLM 服务的 SDK (例如 `openai` 或自定义 HTTP 客户端)

强烈建议使用虚拟环境或 Conda 环境。

---

## 快速上手

### 1. 安装依赖

在项目根目录下创建虚拟环境并安装所需包，例如：

```bash
conda create -n kernelmem python=3.10 -y
conda activate kernelmem

# 根据需要安装依赖（示例）
pip install torch matplotlib pandas numpy
# 如果使用 OpenAI 模型，同时安装：openai
```

确保 `ncu` 和 `nsys` 命令在您的 shell 中可用。

### 2. 运行单个任务

最基本的用法是指定一个 PyTorch 任务脚本作为 `arch_py`：

```bash
python main_memory_latest.py KernelBench/level1/001_xxx.py \
  --gpu A100-80GB \
  --server_type openai \
  --server_address localhost \
  --server_port 8000 \
  --model_name gpt-5.1-chat \
  --round 10 \
  --work_dir run \
  --device 0
```

关键参数说明：

- **`arch_py`**: PyTorch 任务脚本的路径，或包含多个任务的目录。
- **`--gpu`**: 提示词中使用的 GPU 名称（不改变实际设备，仅告知 LLM 硬件规格）。
- **`--server_type` / `--server_address` / `--server_port` / `--model_name`**: LLM 后端配置。
- **`--round`**: 每个任务的总轮数（包括种子生成、修复和优化）。
- **`--device`**: CUDA 设备 ID。
- **`--warmup` / `--repeat` / `--tol`**: 预热迭代次数、基准测试重复次数和误差容限。

### 3. 批量任务与过滤

- 从目录中随机采样任务：

```bash
python main_memory_latest.py KernelBench/level3 \
  --num_tasks 5 --shuffle_seed 42
```

- 使用之前运行的 `summary.json` 来仅重新运行那些最佳 kernel 仍无法运行的任务：

```bash
python main_memory_latest.py KernelBench/level3 \
  --filter_from_summary path/to/previous/summary.json
```

---

## 输出与可视化

单个任务的典型结构：

- `code/`: 为该任务生成的所有 kernel（Python/CUDA），可能包含优化/修复历史 JSON。
- `evaluation/`:
  - `llm_io/`: 每一轮的所有提示词和 LLM 原始回复。
  - 每轮指标 JSON：是否可运行、错误类型、加速比等。
- `figures/`:
  - `taskname_score.png`: 跨轮次的加速比曲线，区分可运行/不可运行点。
- `profile/`:
  - `*_ncu*.csv`: Nsight Compute 指标。
  - `*_nsys*.nsys-rep` / `*_nsys*.csv`: Nsight Systems 追踪文件和统计数据。
- `optimization_tree.json`:
  - 该任务所有 kernel 的“族谱”，包含父子关系、加速比、NCU 状态以及是否匹配了优化方法。
- `usage.csv`:
  - 所有 LLM 调用的 Token 使用量，末尾附有总计行。

对于每个批量处理目录，您还将获得：

- `summary.json` / `summary.csv`: 跨任务汇总，包括平均加速比、准确率和总 Token 消耗。

---

## 长短期记忆机制（概念）

- **短期记忆 (局部上下文)**：
  - 当前运行中最近生成的 kernel 片段、最新的错误日志和 profiling 结果。
  - 通过 `_build_history_block` 等辅助函数构建为 Markdown 代码块，直接嵌入到优化提示词中。
  - 历史产物如 `optimization_tree.json` 和每轮的 `opt_round_*.json` / `repair_round_*.json`。

- **长期记忆 (跨轮次/跨任务经验)**：
  - 存储在 `memorybank/` 下的先验知识（硬件瓶颈、常见 kernel 结构、可行的优化策略）。

在生成、修复或优化 kernel 时，LLM 将这些记忆作为额外上下文，以便能够：

- 避免重复相同的编译/运行时错误。
- 复用过去行之有效的优化策略。
- 针对特定的硬件和算子模式做出更具针对性的设计选择。

---

## 注意事项

- 本项目会频繁编译并运行 GPU kernel。请确保您的机器有足够的 GPU 显存，并设置适当的超时/监控机制，以防止因 buggy kernel 导致的死锁或挂起。
- NCU / NSYS Profiling 可能非常耗时，尤其是 `KernelBench/level3` 中的大模型任务。建议先在较小规模的任务上用较少的轮数调试流程。

如果您想在自己的环境中部署或扩展此项目（例如连接自定义 LLM 后端、添加新 kernel 模板/任务集），请从阅读和修改 `main_memory_latest.py` 以及 `prompts/` 目录下的文件开始。
