# 06 Windows 11 + RTX 5090 环境搭建

本文是**执行清单**：每一步都可以直接复制到 PowerShell 里跑。目标机器为 Windows 11 + RTX 5090 (32 GB, Blackwell, compute capability `sm_120`)。

**核心风险**：RTX 5090 是 Blackwell 架构（`sm_120`）。为更早架构编译的 PyTorch wheel 可以正常安装、正常 import、`torch.cuda.is_available()` 返回 `True`，然后在第一次真正算 kernel 时抛出 `CUDA error: no kernel image is available for execution on the device`。这不是环境坏掉，是轮子里没有 `sm_120` 的 kernel。本文第 3 节的安装命令就是为绕开这个陷阱写的。

## 1. 前置检查

| 项 | 要求 | 检查方式 |
| --- | --- | --- |
| GPU | NVIDIA RTX 5090, 32 GB, `sm_120` | `nvidia-smi` |
| 驱动 | **>= 570** | `nvidia-smi` 右上角 `Driver Version` |
| OS | Windows 11 x64 | `winver` |
| Python | 3.12（推荐），允许 3.11–3.13 | `python --version` |
| 磁盘 | 至少 80 GB 可用（torch cu128 约 3 GB + CUDA 运行时 + 模型权重 14B≈28 GB fp16 或约 9 GB 4-bit） | `Get-PSDrive C` |
| 环境变量 | `HF_HOME` 指向短路径（见第 2 节） | `$env:HF_HOME` |

```powershell
nvidia-smi
```

`nvidia-smi` 输出中需要确认两点：`Driver Version` >= 570，以及 `CUDA Version` 显示 12.8 或更高（这一行是驱动**支持**的最高 runtime 版本，不是你装了什么）。

如果驱动过旧：

```powershell
winget upgrade --id Nvidia.GeForceExperience
# 或从 NVIDIA 官网下载 Game Ready / Studio 驱动手动安装，选择 >= 570 的版本
```

装完必须重启，然后重新跑 `nvidia-smi` 确认。

### 1.1 关于 CUDA Toolkit

**你不需要安装完整的 CUDA Toolkit 来跑 PyTorch。** PyTorch 的 cu128 wheel 自带所需的 CUDA runtime 库。

若你因为其他工具需要装 toolkit，选择版本时注意：

| CUDA Toolkit | 是否包含 `sm_120` | 说明 |
| --- | --- | --- |
| 12.8 | 是 | 首选 |
| 12.6 | 是（12.6 起加入 sm_120 支持） | 可用 |
| **12.4** | **否** | 用 12.4 编译的任何东西都不支持 RTX 50 系 |
| >= 12.9 | 是 | 可用，与 cu128 的 PyTorch 混用通常无问题 |

一条容易踩的坑：本机装了 CUDA 12.4 的 `nvcc`，然后用它去编译某个需要编译的扩展（例如有人试图编译 flash-attn）。编译会成功，运行时才会因为缺少 `sm_120` 而失败。因此本项目的策略是**不编译任何 CUDA 扩展**，见第 6 节。

## 2. 目录与环境准备

把缓存与虚拟环境放在短路径下，避免 Windows 的 `MAX_PATH = 260` 限制。HF 快照路径会嵌套很多层，非常容易超限。

```powershell
# 长路径支持（需要管理员权限的 PowerShell）
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force

# 缓存放到驱动器根下的短路径
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf", "User")
[Environment]::SetEnvironmentVariable("HF_HUB_DISABLE_SYMLINKS_WARNING", "1", "User")
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
```

设置后**重开一个 PowerShell 窗口**，确认生效：

```powershell
$env:HF_HOME
$env:PYTHONUTF8
```

`PYTHONUTF8=1` 是必要的：Windows 的默认编码是 `gbk`/`cp936`，读取 EDGAR 文本、JSONL、含中文的 YAML 时会直接抛 `UnicodeDecodeError`。代码里也要求所有文本读写显式写 `encoding="utf-8"`，两者互为保险。

## 3. 创建环境并安装 PyTorch cu128

### 3.1 创建虚拟环境

**方式 A：venv（推荐，最少的活动部件）**

```powershell
cd $HOME\Workspace\shingan

# 用 py 启动器选择 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 若报"无法加载文件 ... Activate.ps1，因为在此系统上禁止运行脚本"
# 在当前用户范围放开（不需要管理员）：
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

python -m pip install --upgrade pip setuptools wheel
```

**方式 B：uv（更快，依赖解析更稳）**

```powershell
cd $HOME\Workspace\shingan

# 安装 uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --upgrade pip setuptools wheel
```

### 3.2 安装 PyTorch（关键步骤）

**必须使用 `--index-url`，不要使用 `--extra-index-url`。** 区别很重要：`--extra-index-url` 只是追加一个候选源，pip 仍会优先选择 PyPI 上版本号更高的 CPU-only wheel，结果是安装"成功"但 `torch.__version__` 带 `+cpu`。`--index-url` 会把下载源整体替换为 PyTorch 官方源，从根上避免这个问题。

版本要求：**PyTorch >= 2.7.0**。`sm_120` 的 kernel 从 2.7.0 稳定版才开始随 wheel 发布。

```powershell
# venv 方式
pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7.0" torchvision torchaudio
```

```powershell
# uv 方式
uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7.0" torchvision torchaudio
```

安装后**立刻**验证，不要继续往下走：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

期望形如 `2.7.0+cu128 12.8 True`。出现 `+cpu` 就回到本节开头重装。

### 3.3 安装 bitsandbytes

bitsandbytes 现在发布**官方原生 Windows x86-64 wheel**，按 CUDA 线拆分：

| 构建线 | 适用 |
| --- | --- |
| 11.8 – 12.6 | 旧 CUDA 环境 |
| **12.8 – 12.9** | **RTX 50 系（Blackwell / `sm_120`）— 本机选这个** |

```powershell
# venv 方式
pip install bitsandbytes
```

```powershell
# uv 方式
uv pip install bitsandbytes
```

若 pip 解析到了错误的构建线（症状：4-bit 加载时崩溃并报 CUDA 版本不匹配），显式钉住 12.8–12.9 线的版本，或从官方 release 页下载对应 wheel 后本地安装：

```powershell
pip install "bitsandbytes @ file:///D:/wheels/<下载的 12.8-12.9 线 wheel 文件名>"
```

历史包袱说明：早期在 Windows 上跑 bitsandbytes 需要第三方重编译 wheel（`jllllll/bitsandbytes-windows-webui`）。这个来源**只作为旧 CUDA 环境的兜底**，不应在 RTX 50 系的新环境里使用——它通常不包含 12.8/12.9 线，且维护状态与官方无关。

### 3.4 安装项目本体

```powershell
# 核心依赖（不含训练栈）
pip install -e .
```

```powershell
# 含训练栈：torch, transformers, peft, trl, datasets, accelerate,
#            bitsandbytes, sentencepiece, protobuf, safetensors, einops
pip install -e ".[train]"
```

```powershell
# uv 方式
uv pip install -e ".[train]"
```

注意：`.[train]` 会尝试从 PyPI 解析 torch。如果它覆盖了第 3.2 节装好的 cu128 版本，重新跑一次 3.2 节的命令即可（cu128 wheel 的版本号通常更高，一般不会被降级，但值得确认）。

## 4. 验证

### 4.1 环境自检脚本

```powershell
python -c @"
import torch, platform, sys
print("python:", sys.version.split()[0], platform.machine())
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("compute capability:", f"sm_{cap[0]}{cap[1]}")
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    print("total vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
"@
```

正确安装的期望输出：

```
python: 3.12.7 AMD64
torch: 2.7.0+cu128
cuda build: 12.8
cuda available: True
device: NVIDIA GeForce RTX 5090
compute capability: sm_120
bf16 supported: True
total vram GB: 31.8
```

**四个必须核对的值**：

1. `torch.version.cuda` 是 `12.8`（不是 `None`，不是 `12.4`）。
2. `torch.__version__` 带 `+cu128`（不是 `+cpu`）。
3. `compute capability: sm_120`（这是判断"轮子是否支持你的卡"的直接证据；若这里是 `sm_120` 但下一步算 kernel 却报 no kernel image，说明装的是支持检测但不支持执行的混合环境）。
4. `bf16 supported: True`。

### 4.2 真正算一次 kernel

`torch.cuda.is_available()` 通过不代表 kernel 可用。这一条才是 `sm_120` 陷阱的真正判据：

```powershell
python -c @"
import torch
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
y = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
z = (x @ y).float().mean()
print('matmul ok:', z.item())
"@
```

报 `CUDA error: no kernel image is available for execution on the device` 就是缺 `sm_120` kernel，回到第 3.2 节用 `--index-url` 重装。

### 4.3 `shingan doctor`

```powershell
shingan doctor
```

期望输出（格式示例，值为本机正确安装时的形态）：

```
shingan doctor
────────────────────────────────────────────────────────────
platform            Windows 11 (10.0.26100)  x86_64
python              3.12.7
package             shingan 0.1.0
torch               2.7.0+cu128                 [ok]
torch.cuda build    12.8                        [ok]
cuda available      True                        [ok]
device              NVIDIA GeForce RTX 5090     [ok]
compute capability  sm_120                      [ok]
bf16                supported                   [ok]
vram total          31.8 GB                     [ok]
vram free           30.9 GB                     [ok]
bitsandbytes        0.4x (cuda 12.8-12.9 line)  [ok]
flash-attn          not installed (expected)    [ok]
triton              not available (expected)    [ok]
dataloader workers  0 (Windows default)         [ok]
encoding            PYTHONUTF8=1                [ok]
HF_HOME             D:\hf                       [ok]
path length risk    low                         [ok]
────────────────────────────────────────────────────────────
BLOCKING ISSUES     none
WARNINGS            none
```

`doctor` 会显式检查的失败项：

| 检测项 | 失败信息 | 含义 |
| --- | --- | --- |
| sm_120 vs cu 线 | `GPU is sm_120 but torch was built for CUDA 12.x (< 12.8); reinstall from .../whl/cu128` | 命中了本文开头描述的陷阱 |
| CPU wheel | `torch reports +cpu build; reinstall with --index-url (not --extra-index-url)` | pip 静默保留了 CPU wheel |
| bitsandbytes 线 | `bitsandbytes built for CUDA 11.x/12.0-12.6; RTX 50-series needs the 12.8-12.9 line` | 构建线不匹配 |
| Python 版本 | `python 3.14 is outside the supported range >=3.11,<3.14` | 版本越界 |
| 编码 | `PYTHONUTF8 not set` | 会在读数据时抛解码错误 |
| 路径长度 | `HF_HOME path length N > 200; long paths may fail` | 建议换短路径 |
| flash-attn / triton | 仅作为 info 行，不报错 | 缺失是预期行为（见第 6 节） |

## 5. 冒烟测试

不需要 GPU、不需要网络、不需要真实数据：

```powershell
shingan data synth
shingan demo
```

`shingan demo` 是 CPU-only 的端到端运行，产出 `runs/<timestamp>-demo/report.md`。它验证的是**流水线连通性**（数据生成 → as-of 检查 → 特征 → 两条轨 → 融合 → 切分 → 指标 → 报告），不是预测能力。报告页首会标记数据来源为 synthetic。

有 GPU 之后，跑最小的文本轨：

```powershell
shingan train lora --base Qwen/Qwen3-8B --config configs/train/qlora_smoke.yaml
```

先跑 8B 而不是 14B：第一次跑 QLoRA 会依次暴露 cu128、bitsandbytes、序列长度、梯度检查点、DataLoader 这几类与模型规模无关的问题。用 8B 把这些环节过一遍，再换 14B，能把环境问题和配置问题分开定位。[训练](04-training.md) 第 2.1 节有完整理由。

## 6. 不要装的东西

| 包 | 原因 | 替代 |
| --- | --- | --- |
| `flash-attn` | Windows 上需要自行编译，且编译工具链与 CUDA 版本高度耦合。编译成功也可能在运行时因 `sm_120` 失败 | `attn_implementation="sdpa"`（PyTorch 内置，已足够） |
| `triton` | 原生 Windows 无官方支持 | 无。任何依赖自定义 Triton kernel 的加速路径都当作**可选** |
| `unsloth` | 原生 Windows 安装可行，但 fast path 依赖 Triton kernel，且有 torch 必须预装的顺序约束 | 视为可选加速器。不可用时不影响主流程。TRL + PEFT + bitsandbytes 是保证可行的路径 |
| `xformers` | Windows 轮子覆盖不全，且与 sdpa 功能重叠 | 不装 |

这三条对应 [ADR-0002](adr/0002-training-stack-windows.md)。核心原则：**POC 需要的训练栈是 TRL + PEFT + bitsandbytes，这条路径在任何情况下都必须可用。** Unsloth 是锦上添花，不能用它作为任何必要功能的前提。

## 7. WSL2 回退路径

若原生 Windows 路径在任何一步卡住且无法在本机解决，回退到 WSL2 + Ubuntu。这是**一等替代方案**，不是应急手段。

```powershell
# 在 Windows PowerShell（管理员）中
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
```

重启后进入 Ubuntu：

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv build-essential

cd ~/shingan
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# cu128，同样必须用 --index-url
pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7.0" torchvision torchaudio

pip install bitsandbytes
pip install -e ".[train]"
```

WSL2 下的差异：

| 项 | 原生 Windows | WSL2 + Ubuntu |
| --- | --- | --- |
| CUDA 驱动 | Windows 驱动（>= 570） | **不需要在 WSL 内装驱动**，由 Windows 驱动经 WSL 直通；不要在 WSL 里装 NVIDIA 驱动 |
| `DataLoader` workers | 必须 0 | 可以用 > 1，速度更好 |
| `fcntl` / `os.fork` | 不可用 | 可用 |
| Triton | 不可用 | 可用，因此 Unsloth 的 fast path 可用 |
| 文件系统性能 | 原生 | **代码放在 WSL 的 ext4 内（如 `~/shingan`），不要放在 `/mnt/c/...`**，跨文件系统 I/O 会显著拖慢数据加载 |
| 显存 | 全部 32 GB | 全部 32 GB（WSL2 的 CUDA 直通不额外占用显存） |

WSL2 在依赖兼容性上全面优于原生 Windows。选择原生 Windows 作为默认路径的原因是运维简单（单文件系统、无虚拟化层、IDE 与调试器直接可用），不是性能。

## 8. 排错表

| 症状 | 根因 | 修复 |
| --- | --- | --- |
| `CUDA error: no kernel image is available for execution on the device` | PyTorch wheel 缺少 `sm_120` kernel（装了 cu124/cu126 线） | `pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7.0"`，卸载重装，不要用 `--extra-index-url` |
| `torch.__version__` 结尾是 `+cpu`；`torch.cuda.is_available()` 为 `False` | pip 从 PyPI 解析到了 CPU wheel | 同上。若重装后仍为 `+cpu`，先 `pip uninstall -y torch torchvision torchaudio` 再装 |
| `torch.version.cuda` 是 `None` | 同上（CPU build 无 CUDA 版本） | 同上 |
| `shingan doctor` 报 `sm_120 but torch built for CUDA 12.6` | 环境里同时存在两个 torch（例如 base 环境与 venv 混用） | 确认 `.venv\Scripts\Activate.ps1` 已激活；`python -c "import torch; print(torch.__file__)"` 看路径是否在 `.venv` 内 |
| bitsandbytes 导入时报 CUDA 版本不匹配，或 4-bit 加载崩溃 | 装了 11.8–12.6 线的 wheel | 装 12.8–12.9 线；`python -c "import bitsandbytes as b; print(b.__version__)"` 确认 |
| `ImportError: cannot import name 'flash_attn'` | 配置里写了 flash_attention_2 | 改成 `attn_implementation="sdpa"`；不要把 flash-attn 加进依赖 |
| Unsloth 报 Triton 相关错误（`ModuleNotFoundError: triton`、`cannot find tl` 等） | 原生 Windows 无 Triton | 去掉 Unsloth，走 TRL+PEFT+bitsandbytes；或用 WSL2 |
| `AttributeError: module 'os' has no attribute 'fork'` / `ModuleNotFoundError: fcntl` | 依赖了 POSIX-only 模块 | 该库在原生 Windows 不可用。找替代，或用 WSL2 |
| `RuntimeError: DataLoader worker (pid ...) is killed by signal` / spawn pickle 错误 | `num_workers > 0` | 设 `dataloader_num_workers=0`。这是 Windows 默认值，不要改 |
| 训练脚本启动后递归启动多个进程 | 入口缺少 `__main__` 守卫，spawn 模式重入 | 所有入口加 `if __name__ == "__main__":` |
| `UnicodeDecodeError: 'gbk' codec can't decode byte ...` | 系统默认编码 | 设 `PYTHONUTF8=1`（第 2 节），并确认代码里显式 `encoding="utf-8"` |
| `OSError: [Errno 206] Filename too long` / `[Errno 2] No such file or directory` 但路径确实存在 | MAX_PATH 260 | 启用 `LongPathsEnabled`；把 `HF_HOME` 移到短路径（如 `D:\hf`） |
| Windows 无法创建符号链接（HF 快照时报权限错误） | 开发者模式未开启或策略限制 | 使用避免 symlink 的下载方式（`local_dir` 语义），不要依赖 Linux 式缓存软链 |
| 报告里 `evidence[].quote` 的逐字校验全部失败 | CRLF/LF 混用导致子串不匹配 | `.gitattributes` 统一 LF；读取后规范化行尾再校验 |
| `torch.cuda.OutOfMemoryError` | 显存不足 | 见下方"OOM 减少顺序" |
| 训练 loss 立即为 `nan` | bf16/fp16 组合不当或 lr 过高 | 确认 `bf16=true`、`fp16=false`；lr 从 1e-4 降到 5e-5 |
| `pip install -e ".[train]"` 把 torch 降级成 CPU 版 | PyPI 解析覆盖 | 重新跑第 3.2 节的 cu128 安装命令，再验证第 4.1 节 |

### OOM 时按此顺序减少

从代价最小的一步开始，每做一步重跑一次：

1. `max_seq_length`: `4096` → `3072` → `2048`。
2. `gradient_accumulation_steps`: `16` → `8`（同时按比例下调 `learning_rate`，否则有效学习率翻倍）。
3. 打开 `optim="paged_adamw_8bit"`。
4. 确认 `gradient_checkpointing=true`。
5. `per_device_train_batch_size` 已经是 1，不能再降。
6. 换更小的底座：14B → 8B。

不要用 `packing=true` 来"省显存"。packing 会把多条样本拼进同一序列，破坏每条样本独立的 `horizon_days` 与标签边界。

期间用 `nvidia-smi -l 1` 或 `nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 2` 观察实际占用，判断是权重、激活还是优化器状态吃掉了显存。

## 9. 一分钟版

已经有驱动 >= 570 与 Python 3.12 的前提下：

```powershell
cd $HOME\Workspace\shingan
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7.0" torchvision torchaudio
pip install bitsandbytes
pip install -e ".[train]"

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
shingan doctor
shingan demo
```

期望倒数第二行形如：`2.7.0+cu128 12.8 True NVIDIA GeForce RTX 5090`。

## 10. 相关文档

- 训练配置与显存预算：[04 训练](04-training.md)
- 平台选择的决策理由：[ADR-0002](adr/0002-training-stack-windows.md)
- 命令面完整说明：[01 架构](01-architecture.md)
