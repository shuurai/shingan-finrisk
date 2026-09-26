# 04 训练

本文覆盖两条轨的训练配置：Track A（结构化 GBDT + 校准）与 Track B（QLoRA 指令微调）。所有配置值都对应 `configs/train/*.yaml`，并可由 CLI 参数覆盖。

**前置条件**：`train` extra 未安装时，`shingan train lora` 会给出明确报错并指向本文第 6 节。环境搭建步骤见 [06 Windows 环境](06-windows-setup.md)。

## 1. Track A：结构化轨

### 1.1 特征集

输入列来自[数据字典](02-data.md)的三组结构化特征：

| 组 | 模块 | 列数（POC 目标） | 说明 |
| --- | --- | --- | --- |
| 财务比率 | `features/ratios.py` | 约 13 | 杠杆、覆盖、盈利、应计、增长 |
| 技术/微观结构 | `features/technical.py` | 约 20 | 收益、波动、尾部、流动性、regime |
| 文本计数量 | `features/text.py` | 约 10 | 新闻条数、情感聚合、词典命中 |

三组共约 43 列，相对 POC 的样本规模（Stage 2 约 5–10 家公司 × 10 年日频，去重后数万个 `(ticker, as_of)` 行）属于宽表，因此需要较强的正则。

**列名黑名单**：`label_`、`fwd_`、`event_` 前缀的列一律不得进入 X。`leakage.py` 在 `fit` 之前校验，命中即抛错。

**缺失值**：不做插补。GBDT 直接处理 NaN，插补反而会引入人造的确定性。唯一的例外是 `ratios_missing_frac` 这类显式的缺失标记列。

### 1.2 模型选择

默认 `lightgbm`，可选 `xgboost`。理由：

| 考虑 | 结论 |
| --- | --- |
| 样本量与宽度 | 数万行 × 43 列，树集成的舒适区 |
| 缺失值 | 原生支持，无需插补 |
| 单调性先验 | 可以给 `debt_to_equity`、`vol_60d` 等列设置单调约束，提升可解释性与稳定性（可选，默认关闭） |
| 校准 | 树输出的原始分数不平滑，必须外接校准器；这本身是有意设计，见 1.3 |
| 训练成本 | CPU 可完成，不需要 GPU，与 Track B 的 GPU 需求互不争抢 |

超参数不做大规模搜索。POC 阶段固定一组保守值（较浅的树、较强的行/列采样、较早的早停），**把调参留给真实数据到位之后**。过早调参会把 valid 的有效信息消耗掉。

### 1.3 校准：只在 validation fold 上拟合

这是本项目的一处明确设计决策（见[评测](05-evaluation.md)的陷阱清单第 6 条）。

做法：

1. 在 train fold 上拟合 GBDT，得到原始分数。
2. 在 **valid fold** 上拟合校准器：
   - 若 valid 的正样本数 >= `min_pos_for_isotonic`（默认 50），使用 isotonic regression（`sklearn.isotonic.IsotonicRegression`，`out_of_bounds="clip"`）。
   - 否则退化为 Platt scaling（对原始分数做 logistic 回归）。
3. 冻结校准器，在 test fold 上只做 transform。

**不使用** `CalibratedClassifierCV(cv="prefit")` 一类 API：该类接口在新版 scikit-learn 中已废弃，且语义上容易把校准数据与评估数据混在一起。本项目把"在哪个区间拟合校准器"作为报告的必填字段。

三个标签各自有独立的校准器。`tail_risk` 的正样本相对充裕，通常能走 isotonic；`fraud_risk` 在 POC 数据规模下大概率走 Platt。

### 1.4 为什么不用 target encoding

按公司或行业做 target encoding（用该组的标签均值替换类别）在面板数据上有三个问题：

1. **时间泄漏**：组的均值包含了 `as_of` 之后的样本标签。即使用 out-of-fold 也不能解决，因为折是按行随机的，未来仍会进同一折。
2. **与 as-of 语义冲突**：一个特征列的取值必须能由 `as_of` 时刻的信息唯一确定。target encoding 的取值依赖于整份数据集。
3. **稀有事件下方差极大**：`fraud_risk` 的组内正样本可能只有 1–2 个，编码值几乎等于噪声。

替代方案：类别变量（`sector`）只做 one-hot，或交给 LightGBM 的原生 categorical 处理；`ticker` 不作为特征。若确实需要捕捉公司固定效应，用量价类特征（beta、波动）间接表达。

### 1.5 种子策略

- 单一来源：`src/shingan/seed.py` 暴露 `set_seed(seed)`，同时固定 Python `random`、numpy、以及框架侧随机源（torch 的 CPU/CUDA 手动种子）。
- 默认种子 `42`，由 `configs/default.yaml` 的 `seed` 键提供，CLI `--seed` 覆盖。
- **不做多种子取平均作为 headline 结果**。多seed 只用于估计方差：报告中的指标若来自单次 seed，必须在报告中标注 `n_seeds=1`。
- `PYTHONHASHSEED` 在 `scripts/run_poc.*` 中固定，避免字典迭代顺序带来不可复现。
- 值得注意的诚实边界：即使固定了所有这些种子，GPU 上的非确定性 kernel（尤其涉及 atomic 操作与 cudnn autotune）仍可能带来微小差异。因此**不承诺** bit-level 复现，只承诺统计意义上的可复现，报告必须记录环境指纹（torch 版本、CUDA 版本、GPU 型号）。

## 2. Track B：文本轨（QLoRA）

### 2.1 底座模型选择

四个候选，都在 Qwen3 家族内（中文语料友好、长上下文、Apache-2.0）：

| 底座 | 参数量 | QLoRA 4-bit 显存（32 GB 卡的估位） | 定位 |
| --- | --- | --- | --- |
| `Qwen/Qwen3-4B` | 4B | 约 3–4 GB | 仅用于最快打通流水线；能力不足，不用于任何结果 |
| `Qwen/Qwen3-8B` | 8B | 约 6 GB | **起点**。流水线 bring-up 用 |
| `Qwen/Qwen3-14B` | 14B | 约 9–12 GB | **正式配置**。HF 发布名 `shuurai2000/shingan-qwen3-14b-finrisk` |
| `Qwen/Qwen3-30B-A3B` | 30B 总 / 3B 激活（MoE） | 与 27B 相当，约 22 GB 量级 | 可行但更紧；MoE 训练在 Windows 上风险更高 |

选择理由：

- **14B 是默认**。32 GB 的预算下，14B 的 QLoRA 占用 9–12 GB，为激活、优化器状态、长序列注意力和 gradient checkpointing 留出充足余量。任务性质是小样本判别而不是知识注入，14B 的指令遵循能力已经足够；更大的模型不会自动提升一个小样本稀有事件任务的 PR-AUC。
- **8B 是起点，理由是工程而非能力**。第一次打通 QLoRA 会踩 Blackwell/cu128、bitsandbytes 版本、序列长度、DataLoader 等一连串与模型规模无关的问题。用 8B 走通这些环节，再切到 14B，能把"环境问题"与"配置问题"分开定位。**不要在 8B 上做任何结论性评估**，它只是流水线验证。
- **30B-A3B 不排除，但不作为 POC 目标**。MoE 架构在 4-bit 量化下的行为需要额外验证，且其激活参数量与稠密模型不同，显存估算不能直接按总参数量外推。Stage 4 之后可以试。
- **金融领域已微调的底座（如社区发布的 Qwen3 finance LoRA）不在默认路径中**。它引入了来源与许可不透明的第二层权重，会让"文本轨带来了什么"这一核心问题更难归因。可作为可选冷启动实验，但必须与从 `Qwen/Qwen3-14B` 起训的版本分别报告。

### 2.2 QLoRA 配置

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `load_in_4bit` | `true` | QLoRA |
| `bnb_4bit_quant_type` | `nf4` | 4-bit NormalFloat |
| `bnb_4bit_use_double_quant` | `true` | 二次量化量化常数，省约 0.4 bit/参数 |
| `bnb_4bit_compute_dtype` | `bfloat16` | Blackwell 支持 bf16，优于 fp16 |
| `lora_r` | `32` | 常用区间 16–32；任务容量需求中等，取上界 |
| `lora_alpha` | `64` | alpha/r = 2，常见稳定选择 |
| `lora_dropout` | `0.05` | 小样本下轻微正则 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 注意力投影全覆盖；不含 MLP 层，控制可训练参数量 |
| `bias` | `none` | — |
| `learning_rate` | `1e-4` | 常用区间 1e-4 ~ 3e-5，取上界；若 loss 震荡则下调至 5e-5 |
| `lr_scheduler_type` | `cosine` | — |
| `warmup_ratio` | `0.03` | 3% warmup。`transformers` v5 已移除该键，运行时按本轮真实步数折算成 `warmup_steps`，见 2.6 |
| `num_train_epochs` | `3` | 可按早停调整；小样本上超过 3 epoch 极易过拟合 |
| `per_device_train_batch_size` | `1` | 受长序列限制 |
| `gradient_accumulation_steps` | `16` | 有效 batch 16 |
| `optim` | `paged_adamw_8bit` | 分页 8-bit AdamW，降低优化器状态显存 |
| `gradient_checkpointing` | `true` | 用计算换显存；必须开，否则 14B@4096 会 OOM |
| `attn_implementation` | `sdpa` | **不使用 flash-attn**，见 [ADR-0002](adr/0002-training-stack-windows.md) |
| `max_seq_length` | `4096` | 财报摘录 + 新闻 + 结构化信号摘要的拼接长度 |
| `packing` | `false` | 每条样本对应一个独立的标签与 horizon，packing 会跨样本拼接，破坏 prompt 边界 |
| `dataloader_num_workers` | `0` | **Windows 必填**，见 2.5 |
| `fp16` / `bf16` | `bf16=true`, `fp16=false` | — |
| `eval_strategy` | `epoch` | 用 valid fold |
| `save_total_limit` | `3` | — |
| `report_to` | `none` | 不默认上传；需要时显式开启 |

序列长度与截断顺序（`max_seq_length` 触顶时按此顺序丢弃）：

1. 先截掉新闻条目（从最早的一条开始丢，保留最近的在语义上更相关）。
2. 再截掉 10-K/10-Q 中优先级最低的段落（保留 Item 1A Risk Factors 与 Item 7 MD&A）。
3. 最后截断 `STRUCTURED_SIGNALS` 的自然语言摘要。

丢掉的任何内容都必须在 `meta.truncated = true` 中留痕，并计入后续的消融分析（"被截断的样本表现是否更差"本身就是一个值得报告的结果）。

### 2.3 指令格式与 prompt 模板

模板定义在 `src/shingan/prompts.py`。输出被约束为单个 JSON 对象，字段与[架构文档](01-architecture.md)的输出契约一致。

```
[system]
You are a financial risk analyst producing evidence-grounded risk assessments.
Output exactly one JSON object with these keys:
  label        : one of default_risk, fraud_risk, tail_risk
  severity     : one of low, medium, high, critical
  score        : float in [0, 1], your probability estimate
  horizon_days : integer, must equal the horizon stated in the user request
  reasons      : array of short strings
  evidence     : array of {source_type, source_ref, quote}
Rules:
  - Every quote must appear verbatim in the provided documents.
  - Do not use knowledge dated after <AS_OF>.
  - If the documents do not support an assessment, say so in reasons and use severity "low".

[user]
<AS_OF>2020-03-02</AS_OF>
<TASK>label=default_risk horizon_days=365</TASK>

<STRUCTURED_SIGNALS>
interest_coverage: 4.1 (down 38% yoy)
debt_short_term_ratio: 0.44 (up from 0.31)
vol_60d: 0.41
n_news_30d: 12, sent_mean_30d: -0.22
</STRUCTURED_SIGNALS>

<FILING_EXCERPTS>
[doc=10-Q, filed=2019-11-01, section=Item 1A]
adverse changes in credit markets could constrain our liquidity ...
</FILING_EXCERPTS>

<NEWS>
[published=2020-02-24, source=...]
the bank flagged higher provisions for credit losses ...
</NEWS>

[assistant]
{"label": "default_risk", "severity": "high", "score": 0.37, "horizon_days": 365, ...
```

设计要点：

- `<AS_OF>` 同时出现在 system 与 user 中，让模型在两侧都能看到时间边界；训练时这就是一条指令，推理时它也是防穿越的显式提示。
- `STRUCTURED_SIGNALS` 用**自然语言摘要**而不是原始 JSON 矩阵：数字以文本形式给出更接近预训练分布，且更容易让模型在 `reasons` 中引用。
- `horizon_days` 是强制字段且必须与 user 请求一致。这是把三个不同 horizon 标签区分开的唯一手段；若模型输出不一致，该样本在评估中记为格式错误。
- `evidence[].quote` 的"逐字"约束在训练数据里天然满足（引文直接从源文本截取），模型学到的是"复制"而不是"生成"。

### 2.4 32 GB 显存预算

下表是 **QLoRA 4-bit 的权重占用**，不含激活、优化器状态、gradient checkpointing 的临时开销和 KV cache。

| 底座 | 4-bit 权重量级 | 加上激活/优化器/grad-ckpt 后的实际占用（seq 4096, bs 1） | 32 GB 是否可行 |
| --- | --- | --- | --- |
| 8B | 约 6 GB | 约 10–14 GB | 充裕 |
| 14B | 约 9–12 GB | 约 16–22 GB | **舒适，默认配置** |
| 27B（稠密） | 约 22 GB | 约 28–34 GB | 边界，需 seq 降到 2048 |
| 32B（稠密） | 约 26 GB | 超出 32 GB | 需要更小的序列或 offload |
| 30B-A3B（MoE） | 与 27B 稠密相当 | 约 28–34 GB | 可行但紧 |

"8B≈6 GB / 14B≈9–12 GB / 27B≈22 GB / 32B≈26 GB"是**权重**量级。选择 14B 作为默认，正是因为它在权重之外还留下了约 20 GB 的余量，足以吸收长序列注意力与优化器状态的波动。任何把权重占用当总占用、然后按剩余显存配序列长度的做法都会 OOM。

调参顺序（OOM 时按此顺序减少，从代价最小的开始）：

1. `max_seq_length` 4096 → 3072 → 2048。
2. `gradient_accumulation_steps` 16 → 8（同时按比例降 `learning_rate`，或接受有效 batch 减小）。
3. 打开 `optim="paged_adamw_8bit"`（若尚未开启）。
4. 确认 `gradient_checkpointing=true`。
5. `per_device_train_batch_size` 已经是 1，不能再降。
6. 换更小的底座（14B → 8B）。

**不要**把降 `max_seq_length` 作为第一步的替代方案去"顺手"打开 `packing`，packing 会破坏 prompt 边界。

### 2.5 Windows 特有问题

| 问题 | 处理 |
| --- | --- |
| `DataLoader` worker pickle 失败 | `dataloader_num_workers=0`。这是默认值，不要改 |
| `if __name__ == "__main__"` 守卫 | 所有可能被 spawn 的入口（含 `scripts/run_poc.ps1` 调起的 Python 入口）必须有 `__main__` 守卫，否则 spawn 模式会递归启动子进程 |
| 文件编码 | 所有文本读写显式 `encoding="utf-8"`。Windows 本地默认编码（`gbk`/`cp936`）会让 EDGAR 文本与 JSONL 立刻抛 `UnicodeDecodeError` |
| 行尾 | `.gitattributes` 统一为 LF。混合 CRLF 会让基于字节偏移的 `span` 引用失准 |
| symlink | HF 缓存在 Linux 上用 symlink，Windows 上可能因权限失败。下载用 `local_dir` 语义的等价参数，避免 symlink |
| 路径长度 | 保持路径浅（MAX_PATH 260）。HF 快照路径很容易超限，建议把 `HF_HOME` 设在驱动器根下的短路径，并启用长路径支持 |
| 无 `fcntl` / 无 `os.fork` | 任何依赖这两个模块的库（部分文件锁、部分 dataloader 库）在原生 Windows 上不可用。选依赖时先排除 |

### 2.6 配置与训练器 API 的边界

本项目的 YAML 描述**意图**，`transformers` / `trl` 的训练器参数描述**某一版的接口**。两者会分叉，而且分叉的方向代价最大：一个已经不存在的键会在构造 `SFTConfig` 时抛 `TypeError`，而那一步发生在底座模型**下载与量化之后**。

这个坑踩过一次，代价是 2h05m 的下载换来一行报错：`transformers` v5 移除了 `warmup_ratio`，配置里还是旧键。因此现在的做法是：

- **映射是纯函数。** `build_sft_config_kwargs()` 把 `lora` 配置块翻译成训练器参数，不导入 torch，所以"映射是否还在本机 API 之内"这件事可以在没装训练栈的机器上被测试（`tests/test_lora_arguments.py`）。
- **校验在花钱之前。** `assert_trainer_arguments_supported()` 把映射结果与本机版本的签名逐键比对，不匹配则立即失败，并报出**所有**不认识的键与本机版本号。这一步和映射都在 `from_pretrained` 之前。
- **`warmup_ratio` 留在 YAML，运行时折算成 `warmup_steps`。** 依据是本轮真实步数：566 条训练样本 / 有效 batch 16 → 36 步/epoch × 3 epoch = 108 步，`0.03 × 108 ≈ 3`。v5 之后预热只能用绝对步数表达，而把步数写死会在改 batch size 或 epoch 数时静默失真。
- **`dtype` 按版本选名。** v5 把 `torch_dtype` 改名为 `dtype`，旧名保留为弃用别名：两个名字在 v5 都能加载，只有旧名在 v4 可用，所以解析不出版本时回退到旧名。
- **`run.json` 记录真正传下去的参数与库版本。** 配置快照本身说明不了实际跑了什么——键会被翻译（`warmup_ratio` → `warmup_steps`），也可能被旧版本忽略。

端到端验证（约一分钟，不需要 GPU，只下载几 MB 的 tiny 模型）：

```powershell
.venv\Scripts\python.exe scripts\smoke_train.py --keep
```

它以同族 tiny 模型跑完整条链路——配置合并 → 映射 → 训练器构造 → 一步优化 → 保存 adapter → `run.json`——并断言 `run.json` 里没有 `warmup_ratio`、有折算后的 `warmup_steps`、且有 adapter 权重。签名检查能挡住"键被移除"，只有这一步能挡住"键还在但语义变了"。

## 3. 训练流程与命令

```powershell
# 1) 生成合成数据（离线，确定性）
shingan data synth --config configs/data/synth.yaml

# 2) 组装 processed 数据集（as-of join + 特征 + 标签）
shingan data build --config configs/data/default.yaml

# 3) 导出 SFT 指令 JSONL
shingan data sft --out data/processed/sft --labels default_risk,fraud_risk,tail_risk

# 4) 结构化轨（CPU 可跑）。注意它用 --config（基础配置），没有单独的 train 覆盖文件
shingan train structured --label default_risk

# 5) 文本轨（需要 train extra 与 GPU）
shingan train lora --base Qwen/Qwen3-14B --config configs/train/qlora_14b.yaml

# 6) 评测：指标 + 基线对比
shingan eval run --config configs/eval/default.yaml
shingan eval compare --runs structured text fused
```

配置叠加规则见[架构文档](01-architecture.md)第 5 节：`configs/default.yaml` + `configs/train/*.yaml` 覆盖 + CLI 参数优先。所有 YAML 由 pydantic v2 模型（`config.py`）校验，未知键报错。

训练产物写入 `runs/<timestamp>-<label>-<track>/`，包含：配置快照、环境指纹（torch/CUDA/GPU）、校准器拟合区间、checkpoint 引用、指标 JSON。报告（`eval/report.py`）缺任一字段即视为无效结果。

### 3.1 训练数据的溯源记录（`run.json` 的 `data` 段）

每次 `train lora` 写出的 `run.json` 携带一个 `data` 段（`shingan/data/provenance.py`）：

- `train_file` / `eval_file`：路径、SHA-256、字节数——"这个适配器吃的是什么"变成可校验的哈希，而不是模型卡里的一句话；
- `sft_manifest`：内嵌训练文件同级目录的 `manifest.json`（该 manifest 自带 `data` 段：来源集合、`is_synthetic`、面板形状与**原始表哈希**），形成 `run.json → SFT manifest → raw 哈希` 的完整链；
- manifest 缺失或不可读时记录原因（`sft_manifest_note` / `sft_manifest_error`），不留无法解释的 `null`。

注意 SFT 目录约定：`data/processed/sft/` 是合成 contract-v2 语料（与 `artifacts/lora-contract-v2` 配对）；真实语料归档在 `data/processed/sft_stage2_real/`。两个目录只靠名字区分——重新生成前先哈希、后核对（`docs/09` §14.2 记录了一次踩坑与恢复）。

### 3.2 安慰剂对照（ADR-0003 要求的实验）

把训练语料的 **user turn 在每个 split 内固定种子置换**（system 轮与 assistant 目标不动），重训一次。文本→标签的连接被切断，而格式、长度分布、标签分布全部保留：安慰剂适配器若仍输出常量分数，"微调只买到格式"就从推断变成实验结论。

```powershell
# 1) 生成安慰剂语料（确定性，seed 7；约 1 分钟）
python scripts/placebo_corpus.py --src data/processed/sft --dst data/processed/sft_placebo --seed 7

# 2) 重训（约 48 分钟 GPU，不覆盖现有产物）
python -m shingan train lora --train-file data/processed/sft_placebo/train.jsonl --eval-file data/processed/sft_placebo/valid.jsonl --output-dir artifacts/lora-placebo

# 3) 评分（合成面板，与 lora-contract-v2 评测同一配置）
python -m shingan eval lora --mode adapter --adapter artifacts/lora-placebo/adapter --batch-size 4 --no-verify-prompts
```

第 3 步必须 `--no-verify-prompts`：重建 prompt 与安慰剂语料**本来就该**不一致，这正是置换的目的。链路完整性改由安慰剂 run.json 的 `data` 段承担（它记录了安慰剂语料的哈希）。对照量：安慰剂适配器 vs `artifacts/lora-contract-v2`（合成评测 `artifacts/lora-eval/20260923T094317Z`，64 行、9 正）的两臂指标与逐行分数。

第 1 步同时会在输出目录写确定性的 `manifest.json`（seed、源/输出文件 SHA-256、逐 split 计数），`train lora` 经 `training_data_block` 把它内嵌进 run.json——安慰剂臂的溯源链与真实臂同构。重跑同 seed 逐字节一致。


## 4. 失败模式与修复

| 症状 | 根因 | 修复 |
| --- | --- | --- |
| `CUDA error: no kernel image is available for execution on the device` | 装了为旧架构（cu124/cu126）编译的 PyTorch，缺少 sm_120 kernel | 按[06 Windows 环境](06-windows-setup.md)用 `--index-url https://download.pytorch.org/whl/cu128` 重装 |
| `torch.cuda.is_available()` 为 `False`，`torch.__version__` 带 `+cpu` | pip 解析到了 CPU wheel | 同上；注意必须用 `--index-url` 而不是 `--extra-index-url`，后者会让 pip 静默保留 CPU wheel |
| bitsandbytes 报 CUDA 版本不匹配，或 4-bit 加载时崩溃 | 装了 11.8–12.6 线的 wheel | 换 12.8–12.9 线的官方 Windows x86-64 wheel |
| `ImportError: cannot import name 'flash_attn'` / 编译失败 | 试图在 Windows 上装 flash-attn | 不装。改用 `attn_implementation="sdpa"` |
| Unsloth 首次运行报 Triton 相关错误 | 原生 Windows 无 Triton，Unsloth 的 fast path 依赖自定义 Triton kernel | 把 Unsloth 当可选加速器；不可用时走 TRL+PEFT+bitsandbytes。不要为它改核心流程 |
| 训练 loss 一开始就是 `nan` | bf16 与部分 kernel 组合、lr 过高、或数据里有 NaN 目标 | 确认 `bf16=true` 且 `fp16=false`；lr 降到 5e-5；检查 SFT JSONL 的 JSON 可解析性 || valid loss 单调上升、train loss 下降 | 小样本过拟合 | 减到 2 epoch、`lora_dropout` 提到 0.1、或减 `lora_r` 到 16 |
| valid 指标剧烈波动 | 事件稀有导致 valid 正样本只有个位数 | 这是本质困难，不是 bug。改用更长的 valid 区间，并依赖滚动评估（见[评测](05-evaluation.md)的陷阱 1）而非单点指标 |
| `DataLoader` 抛 pickle/spawn 相关错误 | `num_workers > 0` | 设 `dataloader_num_workers=0` |
| 读 JSONL 时 `UnicodeDecodeError` | 未指定编码 | 显式 `encoding="utf-8"` |
| `OSError: [Errno 206] Filename too long` | HF 缓存路径超 MAX_PATH | 启用长路径，或把 `HF_HOME` 移到短路径 |
| 报告里的 `quote` 校验全部失败 | CRLF/LF 不一致导致子串不匹配 | 规范化文本行尾后再做 `in` 校验；`.gitattributes` 统一 LF |
| 校准后的概率系统性偏高 | 负样本下采样后未回填 `sample_weight` | 检查 `sample_weight` 是否传入训练与校准；评估集必须不降采样 |
| 构造 `SFTConfig` 报 `TypeError: unexpected keyword argument 'warmup_ratio'` | `transformers` v5 移除了 `warmup_ratio`，YAML 里仍是旧键 | 已修：运行时折算为 `warmup_steps`（见 2.6）。同类错误现在会在**下载模型之前**被 `assert_trainer_arguments_supported` 拦下，并列出所有不认识的键 |
| 参数错误发生在模型下载/量化**之后** | 参数校验曾经位于 `from_pretrained` 之后 | 校验已前置到模型加载之前。用 `python scripts/smoke_train.py` 在约一分钟内复现整条链路 |

## 5. 尚未验证的部分

以下条目必须读作"未验证"，不得在对外材料中当作已完成工作：

- **14B 训练已产出 checkpoint，但它证明的是链路，不是能力。** `artifacts/lora-contract-v2/adapter/`（Qwen3-14B、108 步、48:08、`run.json` 记下 `warmup_steps=3` 与六个训练库版本）是**当前**适配器；`artifacts/lora/` 是同日上午在**旧指令契约**下跑出的同规格 adapter，原地保留（它现在会被 `--verify-prompts` 拒绝打分，而那是**正确行为**而非损坏）。两个 adapter 的语料都来自 `configs/default.yaml` 的默认 `sources: [synthetic]`（`sample_id` 形如 `SXAA-20100101`），所以“训练链路在 32 GB 单卡上端到端可用”是事实，“训出了金融风险模型”不是。
- **同分布评测显示这个 adapter 的输出是常量，不是“弱”。** `shingan eval lora` 在合成 test 块上得到 AUC **0.5000** / KS **0.0000**，64 行全部 `score: 0.0`、`severity` 全为 `low`。**而同一次运行里的零样本臂不是常量**（AUC 0.5172 / KS 0.1273，两臂都 64/64 解析），所以“基座模型只会输出一个数”已被排除。配对差值 `lora − zero_shot` 为负、**区间跨零**：可写“没有可测量的正增量”，不可写“显著变差”。该样本仍不能作为 Track B 的性能引用。完整记录见 [09 LoRA 评测接入](09-lora-evaluation.md)。
- **合成数据上的"文本轨有增益"是生成器构造出来的**，不是实验发现。融合增益在合成数据上为正只说明实现与设计一致，不构成真实世界证据。
- **文本基线（`text_baseline.py`）与 14B LoRA 的相对表现**：在合成数据上 TF-IDF 明显更好，因为 LoRA 那一列是常量；**在真实数据上仍是未知**——真实 `tail_risk` 的 train 切分只有 4 个正样本，训 14B 只会再学出一个常量。瓶颈是正样本数，不是数据 provenance。
- **三个底座（8B / 14B / 30B-A3B）之间没有做过对比实验**，选型理由是显存与任务性质的推理，不是实测。
- **校准器的选择阈值 `min_pos_for_isotonic=50` 是经验值**，未在真实基率下验证过。`fraud_risk` 在 POC 规模下大概率只能走 Platt，其校准质量未知。
- **超参数未调**。第 2.2 节的配置来自常用取值范围与常见实践，不是搜索结果。
- **`evidence[].quote` 的逐字校验在真实 EDGAR 文本上的余量未知。** 校验本身是硬门（`sft_examples` 里任何一条引用不是逐字命中就 `ValueError` 拒绝写出训练文件），所以"语料存在"等价于"当时全部通过"。但通过得有多勉强——比如有多少条是从超长段落里切出来的、有多少条接近长度下限——没有被记录，因此真实 10-K/10-Q 上这道门的脆弱程度读不出来。
- **适配器从未在真实文本上被评分过。** 真实数据上的 `text_only_lora` 需要 GPU 推理；按 2026-09-23 的决定，在正样本扩容之前不再为这个常量花 GPU（见 [09](09-lora-evaluation.md) 第 7、9 节）。
- **价格与 EDGAR 适配器已对 live 端点验证，其余没有。** `scripts/fetch_real.py` 已真实跑出 `data/raw/real/`（prices 114,424 行、fundamentals 2,189、filings 2,879）。**新闻（FNSPID）与事件（评级/执法）两个源仍未接入**，所以 `n_news_*` / `sent_*` 在真实面板上是零、`default_risk` 与 `fraud_risk` 无法评测；"真实数据训练"这条路径的整体可行性因此只对 `tail_risk` 成立（见[数据](02-data.md)）。

## 6. 训练环境安装

完整步骤、可复制的 PowerShell 命令、验证脚本与排错表见 [06 Windows 环境](06-windows-setup.md)。核心要点：

```powershell
# PyTorch cu128（sm_120 必需），注意用 --index-url
pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7.0" torchvision torchaudio

# bitsandbytes：选 12.8-12.9 线的官方 Windows x86-64 wheel
pip install bitsandbytes

# 训练栈
pip install -e ".[train]"
```

不要在 Windows 上尝试 `flash-attn`。不要在原生 Windows 上依赖 Triton。若上述任一步在本机无法完成，回退路径是 WSL2 + Ubuntu，该路径是**一等替代方案**，见 [06 Windows 环境](06-windows-setup.md) 与 [ADR-0002](adr/0002-training-stack-windows.md)。

## 7. 相关文档

- 特征与标签的来源：[02 数据](02-data.md)、[03 标注](03-labeling.md)
- 校准与融合为何只能在 valid 上拟合：[05 评测](05-evaluation.md)
- 平台决策的完整理由：[ADR-0002](adr/0002-training-stack-windows.md)
- 发布模型时的局限声明：[模型卡模板](../templates/model_card.md)
