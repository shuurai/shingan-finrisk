# ADR-0002: Windows / Blackwell 上的训练栈选择

- 状态：Accepted
- 日期：2026-09-21
- 决策者：Shane（`shuurai`）

## Context

目标机器是 Windows 11 + RTX 5090（32 GB，Blackwell，compute capability `sm_120`）。平台约束与常规 Linux 环境差异很大，且这些差异大多在运行时才暴露：

1. `sm_120` 的 kernel 从 PyTorch 2.7.0 稳定版起才随官方 wheel 发布，且需要 CUDA 12.8（cu128）构建。装更早的 cu124/cu126 wheel 会正常安装、正常 import、`torch.cuda.is_available()` 返回 `True`，然后在第一次真正计算时抛 `CUDA error: no kernel image is available for execution on the device`。
2. pip 的 `--extra-index-url` 只是追加候选源，PyPI 上版本号更高的 CPU-only wheel 会被优先选中，导致"安装成功但 torch 是 `+cpu`"。必须用 `--index-url` 整体替换源。
3. `bitsandbytes` 现已发布官方原生 Windows x86-64 wheel，按 CUDA 线拆分（11.8–12.6 与 12.8–12.9）。RTX 50 系必须使用 12.8–12.9 线的构建。第三方重编译 wheel（`jllllll/bitsandbytes-windows-webui`）只适用于旧 CUDA 环境。
4. `flash-attn` 在 Windows 上需要自行编译，工具链与 CUDA 版本强耦合，编译成功也可能在 `sm_120` 上运行时失败。
5. 原生 Windows 无 Triton，因此任何依赖自定义 Triton kernel 的加速路径不可用。
6. 原生 Windows 无 `fcntl`、无 `os.fork`；`DataLoader` 必须 `num_workers=0`；文本 I/O 必须显式 `encoding="utf-8"`；路径受 `MAX_PATH 260` 限制；HF 缓存在 Linux 上依赖的 symlink 在 Windows 上可能失败。

## Decision

1. **PyTorch**：固定使用 **cu128** 构建，版本 **>= 2.7.0**，安装命令必须使用 `--index-url https://download.pytorch.org/whl/cu128`，禁止使用 `--extra-index-url`。
2. **训练栈**：**TRL + PEFT + bitsandbytes** 为**必须可用**的路径。核心训练能力不得依赖任何其他框架。
3. **注意力实现**：使用 `attn_implementation="sdpa"`（PyTorch 内置）。**不安装、不依赖 `flash-attn`**。
4. **bitsandbytes**：使用官方原生 Windows wheel 的 **12.8–12.9** 构建线。第三方重编译 wheel 仅作为旧 CUDA 环境的兜底，不在 RTX 50 系环境使用。
5. **Triton**：不依赖。任何需要自定义 Triton kernel 的库都视为**可选**。
6. **Unsloth**：**可选加速器**。原生 Windows 安装可行，但存在 Triton 相关 caveat，且要求 torch 必须预先安装。缺少它不得影响任何必要功能。
7. **WSL2 + Ubuntu 24.04**：作为**一等替代方案**完整记录。在依赖兼容性上 WSL2 全面优于原生 Windows（有 Triton、有 `fcntl`/`os.fork`、`DataLoader` workers 可用、无路径长度限制），代价是多一层虚拟化与跨文件系统 I/O 的性能损失。
8. **环境验证**：`shingan doctor` 必须显式检测并报告 `sm_120` 与 CUDA 构建线的匹配情况、torch 是否为 `+cpu`、bitsandbytes 构建线、`PYTHONUTF8`、`HF_HOME` 路径长度风险。

## Consequences

正面：

- 训练栈在任何情况下都有一条保证可用的路径，`flash-attn`/Triton/Unsloth 的缺失不会阻塞核心功能。
- `shingan doctor` 把"安装成功但运行时失败"这类最难排查的问题提前到环境检查阶段，并直接给出修复命令。
- 明确的回退方案（WSL2）意味着平台问题不会成为项目阻塞项。

代价与约束：

- 放弃 `flash-attn` 会损失部分长序列训练吞吐；在 4096 序列长度与 14B 模型下，sdpa 是可接受的。
- 放弃 Unsloth 的 fast path 会增加训练时间。POC 阶段的样本量不大，可接受。
- 显式记录 WSL2 路径意味着维护两套安装说明，文档需要同步更新。
- 必须持续跟踪 PyTorch 与 bitsandbytes 的 Windows/Blackwell 支持状态：当上游 wheel 覆盖变化时，本文档的第 1、4 条可能需要更新。

后续动作：

- 在 [06 Windows 环境](../06-windows-setup.md) 中维护完整的命令与排错表。
- 在 [04 训练](../04-training.md) 中维护 QLoRA 配置与显存预算。
- 若某天 `flash-attn` 或 Triton 在原生 Windows 上变得可靠，"不使用"这一决策需重新评估并发新的 ADR，而不是直接改实现。
