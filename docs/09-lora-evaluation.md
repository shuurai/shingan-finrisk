# 09 LoRA 评测接入：分步记录

本文档是一份**步骤日志**，不是设计文档。它记录把「已训好的 LoRA 适配器接进评测」这件事的每一步：目标、决策与理由、实际执行的命令、产出的 artifact、验证了什么、以及**没验证什么**。权威的架构与口径仍以 [01 架构](01-architecture.md) 与 [05 评测](05-evaluation.md) 为准；本文档只回答"这一步做了什么、凭什么说它成立"。

## 0. 为什么要做这件事

[ADR-0003](adr/0003-dual-track-over-single-lora.md) 把 `text_only_lora` 列为任何性能声明的**强制**消融行之一。但截至 2026-09-23，代码库里**不存在这条路径**：

- `shingan eval run` 只拟合 `structured`、`text_baseline`（TF-IDF）、`fused` 三路。全仓搜索 `text_only_lora` 为零命中，"adapter" 一词在代码里全部指**数据适配器**。
- `models/lora.py` 只负责训练，没有任何推理入口。
- 后果：**项目最核心的科学问题（文本轨在结构化信号之外有没有可度量的增量）按设计无法从现有产物回答**。这不是数据问题，是缺一条评测路径。

已训好的适配器见 `artifacts/lora/adapter/`（Qwen3-14B、108 步、train_loss 0.3658）。**它是在合成语料上训的**，因此它证明的是"训练链路在 RTX 5090 上端到端可用"，不是预测能力。

## 1. 动手前的四项侦察结论

这四条改变了本步的做法，先记录，再据此定方案。

**1.1 这个适配器不是 text-only。** 训练 prompt 的 `<STRUCTURED_SIGNALS>` 块包含 `PROMPT_SIGNAL_COLUMNS` 的 **12 个结构化信号**（`debt_to_equity`、`interest_coverage`、`current_ratio`、`altman_z`、`accruals_ratio`、`revenue_yoy`、`vol_60d`、`max_drawdown_60d`、`dist_52w_high`、`abnormal_volume_20d`、`n_news_30d`、`sent_mean_30d`）。

直接拿它的 AUC 与 `structured`（全特征，约 40 列）相减，**两头都不公平**：既高估了文本贡献（模型的输入里已有 12 个结构化信号），又低估了（结构化轨拿到的信息更多）。

这也与 ADR-0003 对 Track B 的定义（"输入为 `as_of` 前已公开的披露文本与新闻"）存在张力。**降级处理见 3.1**；真正的 text-only 适配器需要一次不带信号块的重训。

**1.2 解析器早已写好，缺的是调用方。** `prompts.py` 里 `extract_json_object` / `parse_assessment` / `score_to_risk` 均已实现，含 markdown 围栏、前置散文、schema 容错（severity 与 score 不一致、evidence 写成裸字符串）。全仓搜索显示 **`parse_assessment` 从未被任何模块调用**。`pipeline.py` 的 `build_prompt_contexts` 也可直接复用。推理路径缺的是"接线"，不是能力。

**1.3 `text_inputs()` 有一个会污染对比的真 bug。** `pipeline.py` 在进入标签循环**之前**只调用一次 `text_inputs(panel, build, config)`，而该函数内部把 label **硬编码为 `"default_risk"`**（`build_prompt_contexts(panel, result, config, "default_risk", budget)`）。`USER_TEMPLATE` 会把 label 与 horizon 写进 prompt 的 `<TASK>` 块，因此：

- 让 TF-IDF 基线去预测 `tail_risk`（horizon 30 交易日）时，它看到的 prompt 写的是 `label=default_risk horizon_days=365` —— **问题问错了**。
- LoRA 的**训练**语料是按每个 label 正确写的（`sft_examples` 逐 label 构建），而评测时的 TF-IDF 不是。

不修，LoRA 与 TF-IDF 的对比天然不公平；修，则 `tail_risk`/`fraud_risk` 的 `text_baseline` 历史数字作废。**已选择修，作废范围见 3.2。**

**1.4 评测数据集是个真分岔。** 适配器在合成语料上训成，因此：

| 评测集 | 行数 / 正样本 | 同分布？ | 能回答什么 |
| --- | --- | --- | --- |
| 合成 test 块 | 64 / 9 | 是 | 推理路径是否正确、模型有没有学到读文本。**不是市场结论** |
| 真实 test 块 | 492 / 5 | 否（跨分布） | 对外唯一有意义的数字；但按项目自己的门禁标准，5 个正样本**不足以支撑任何结论** |

## 2. 已批准的决策

| # | 决策 | 理由 |
| --- | --- | --- |
| D1 | 按训练原样给适配器打分，**并另加一条同信息量基线**（只用同样 12 个信号），`lora − structured_matched` 才是干净的文本增量 | 不需要重训、不需要额外 GPU 小时；把"信息量不匹配"这个混杂因素显式消掉 |
| D2 | 合成 test 与真实 test **两个都跑** | 前者验证路径与"是否学到"，后者如实按 insufficient evidence 呈现 |
| D3 | `text_inputs` 的 label 硬编码**本次一并修** | 否则对比不公平；历史数字作废的范围在 3.2 明确标注 |
| D4 | 每步的文档**放进仓库**（即本文档） | 推理链随代码演进、外部读者可见 |

未采纳的选项（留档，避免以后重新讨论）：**重训一个真正不带结构化信号的 text-only 适配器**（约 +42 min GPU）。它给出教科书式干净的 `text_only_lora` 行，但会多一个需要维护的 adapter，且不解决"当前这个已训好的适配器值不值得评"的问题。**若 Step 4/5 显示文本增量可疑，这是下一步应当做的事。**

## 3. Step 1 — 修复 prompt 的 label 硬编码

**目标**：让每一条被评测的路径看到**与它被问的问题一致**的 prompt。

**改动**：

- `pipeline.text_inputs()` 增加必填参数 `label`，不再在函数体内写死。
- `run_pipeline()` 把 `text = text_inputs(...)` 从标签循环外**移进循环内**，按 label 分别渲染。
- 因为 `build_prompt_contexts` 逐行渲染 12 个信号与文本块，这一步对每个 label 各做一次；行数规模下开销可忽略。

**验证**：见 3.3 的测试项。**这是本步唯一一处会改变历史数字的改动**，其余都是新增。

### 3.1 命名与口径（D1 的落地方式）

评测行仍叫 `text_only_lora`（ADR-0003 与模型卡模板都用这个名字），但 artifact 里**强制**携带：

- `prompt_includes_structured_signals: true`
- `structured_signals_in_prompt: [12 个列名]`
- `matched_baseline: structured_matched`（即 `lora − structured_matched` 才是文本增量）

不加这三个字段的渲染器无法把这一行读成"纯文本模型"。

### 3.2 作废范围（D3 的代价）

| 产物 | 受影响的行 | 处理 |
| --- | --- | --- |
| `artifacts/reports/20260923T055845Z.*`（合成） | `tail_risk` 的 `text_baseline`（AUC 0.4404） | **作废**，已被 `20260923T065029Z.*` 取代 |
| `artifacts/stage2/20260921T073120Z.*`（真实） | **全部行**，不止 `text_baseline` | **作废**，已被 `20260923T065911Z.*` 取代。第二个、独立的作废理由见 **7.3.1**（那次运行的面板缺少整个披露文本层） |

`structured` 与 `fused` 两行**在合成数据上**不受 prompt 修复影响（它们不吃 prompt）；但在**真实数据**上它们同样作废，因为旧面板没有文本层——这一条与 prompt 修复无关，是 7.3.1 查出来的。`default_risk` 的 prompt 恰好本来就写对了，故其历史数字不受本步影响。

### 3.3 Step 1 结果

- 状态：**已完成**
- 改动：`pipeline.text_inputs()` 增加必填 `label`；`run_pipeline()` 的调用从标签循环外移入循环内。
- 测试：`tests/test_prompt_labels.py`（新增 5 项，0.31 s）。含一条**接缝测试**：桩掉面板构造与标签拟合，只观测 `text_inputs` 的调用，断言 `seen == ['default_risk', 'tail_risk']`。旧代码下这条测试会失败（旧签名没有 `label` 参数，且调用发生在循环之外），因此它是真守卫。整套 **198 passed / 覆盖率 38%**。
- 未验证：无。本步不涉及 GPU。
- 注意：本步**只**改变 `text_baseline` 一行。`structured` / `fused` 不吃 prompt，不受影响。

## 4. Step 2 — LoRA 推理与打分路径

**目标**：存在一条命令，能对给定适配器与给定数据配置产出可复核的 `text_only_lora` 行。

**设计要点**：

1. **必须复用训练时的 prompt 构造**（`build_prompt_contexts` + `build_chat_messages`），不得另写一份。另写一份 = 训练与评测的输入不一致，而且是静默不一致。
2. **tokenizer 取适配器目录里的那一份**。训练日志出现过 `Updated tokens: {'bos_token_id': None}`，说明训练时的 tokenizer 覆盖了底座 config 的 bos。推理若另起 `Qwen/Qwen3-14B` 自带 tokenizer（bos=151643），token 口径与训练不一致。此条**由 Step 2 的产物记录实际值**，不再停留在"未验证"。
3. **解析失败不是异常**。`parse_assessment` 返回 `None`，调用方统计**解析失败率**并写入 artifact。失败率过高时该行结论无效，这一点必须在产物里可见。
4. **溯源强制写入**：`is_synthetic`、数据配置路径、`split_definition`、adapter 目录哈希、逐行 prompt 哈希、解析失败率、生成参数（temperature/max_new_tokens）。缺一项则拒绝写出——沿用 `publish hf` 的占位符拒绝门思路。

**不含 GPU 的完整性校验**（本步最有价值的一项）：训练集的行**同时**存在于面板与 SFT 文件里，因此可以重建这些行的 prompt 与 SFT 文件里的 user turn **逐字节比对**。相等即证明推理路径精确复现了训练输入。此校验可在无 GPU 机器上运行，也就能进 CI。

### 4.1 Step 2 结果

- 状态：**已完成**
- 新增 `src/shingan/models/lora_inference.py`：`adapter_facts()`（权重与 config 的 SHA-256、底座名、rank/alpha/target_modules、tokenizer 来源）、`load_for_inference()`、`generate_texts()`、`score_generation()`。`torch` / `transformers` / `peft` **只在函数内导入**，所以没有训练栈的机器也能导入这个模块（已核验：模块级 import 行中不含任何训练依赖）。
- 新增 `src/shingan/eval/lora.py`：`compare_prompts()`、`score_from_attempts()`、`build_payload()`、`render_markdown()`、`write_artifact()`。
- `prompts.parse_assessment()` 新增 `failure_reason` 收集口。原因字符串**只产生一次**，同时进 debug log 与调用方的计数——两处不可能对不上。
- 新增 CLI：`shingan eval lora`，参数 `--adapter --label --split --limit --out --max-new-tokens --batch-size --temperature --verify-prompts/--no-verify-prompts --dry-run`。
- 测试：`tests/test_lora_eval.py`（新增 28 项，0.48 s）。

**实测验证（两条，都不依赖人工判断）**

1. `shingan eval lora --dry-run`（不加载模型）：
   `prompt check: 260/260 rows for tail_risk rebuilt byte-identically to the SFT file (782 records scanned, 522 for other labels)`。**训练集 260 条 tail_risk 行的 prompt 被逐字节重建**，因此"评测输入 = 训练输入"是已验证事实而非假设。
2. `shingan eval lora --limit 4`（真实 GPU）：加载适配器耗时约 23 s，产出 `parse: 4/4 usable (failure rate 0.0000)`，artifact 严格 JSON 可解析。4 行中 0 个正样本，故 AUC/PR-AUC 按设计报 `not measured`。

**这一跑解掉了一个此前的"未验证"**：训练日志里的 `Updated tokens: {'bos_token_id': None}` 曾让人担心推理会用错 tokenizer。产物显示 `tokenizer_source: adapter directory` —— **适配器目录自带 tokenizer，推理正是从那里加载的**，与训练口径一致。

**顺带发现的一个溯源事实**：`sft/manifest.json` 与 `train.jsonl` 里的 `meta.sample_id` **在三个标签之间重复**（`sft_examples` 是每行 × 每标签一条样本）。因此 `sample_id` 不是唯一键，提示词比对必须用 `(sample_id, label)`。用 `sample_id` 单独作键时，一个完全正确的三标签文件会有三分之二的行被判为不匹配——这正是本次被这道检查抓出来的第一个错。这也意味着**任何以 `sample_id` 为行标识的下游工具都会混淆三个标签**，已列入第 8 节待办。

- 未验证：全量合成 test 与真实 test 的实测耗时（Step 4/5 记录）。

## 5. Step 3 — 同信息量基线 `structured_matched`

**目标**：把"文本增量"从"信息量差异"里分离出来。

`structured_matched` = 结构化模型，但**特征列限定为 `PROMPT_SIGNAL_COLUMNS` 那 12 个**，同一 train/valid/test、同一标签、同一校准区间。

对比表按 ADR-0003 的要求给出全部行：

| 行 | 输入 | 作用 |
| --- | --- | --- |
| `structured` | 全特征（约 40 列） | Track A 正式结果 |
| `structured_matched` | 同样 12 个信号 | **同信息量基线**，`lora − 本行` 才是文本增量 |
| `text_baseline` | TF-IDF，与 LoRA 同一 prompt | 非 LLM 文本基线 |
| `text_only_lora` | 适配器生成 | Track B 正式结果 |
| `fused` | structured + text_baseline 的分数 | 现有融合层 |

差值用 **paired bootstrap** 给区间，不用点估计。

### 5.1 Step 3 结果

- 状态：**已完成**
- `pipeline.matched_signal_report()`（新增）：同一 `StructuredRiskModel`、同一 12 列、同一 train/valid/test 与校准 fold。任何请求的信号列若不在面板中，**直接拒绝**而不是退化成"有什么用什么"——否则这一行仍叫 `structured_matched` 却已不是那个意思。
- `eval/lora.paired_differences()`（新增）：`candidate - baseline` 的配对 bootstrap，指标为 PR-AUC 与 AUC。任何一条路径缺分数的行**对所有臂一起剔除**（`n_rows` 记录实际行数）；样本只有单一类别时 `estimate` 写 `None` 并附原因，**不写 0**——"没测"与"测到零"必须可区分。
- 测试：`tests/test_lora_eval.py` 增加 9 项。

## 6. Step 4 — 在合成 test 上跑完整评测

**目标**：一次完整（不带 `--limit`）的运行，回答两件事——新增的推理路径在整块数据上是否可用；这个适配器到底有没有学会排序。**这两件事都与真实数据无关**，所以先在同分布的数据上问，问题更干净。

**为什么是合成数据**：适配器在合成语料上训成，合成 test 与它同分布。先在跨分布的真实数据上跑，会把"模型没学到"与"分布不匹配"两个解释混在一起，之后无法拆开。

### 6.1 Step 4 结果

- 状态：**已完成**
- 命令：`shingan eval lora`（默认 `configs/default.yaml`，合成面板 553 行 / 10 家公司 / test 64 行 9 正）
- artifact：`artifacts/lora-eval/20260923T062853Z/{lora_eval.json,lora_eval.md}`
- 解析：**64/64 可用，失败率 0.0000**

| path | AUC | KS | PR-AUC | 正样本 | n |
| --- | --- | --- | --- | --- | --- |
| `structured` | 0.5434 | 0.2323 | 0.1670 | 9 | 64 |
| `structured_matched` | 0.5091 | 0.2727 | 0.1492 | 9 | 64 |
| `text_baseline` | 0.4424 | 0.3374 | 0.2956 | 9 | 64 |
| `fused` | 0.4525 | 0.3152 | 0.1538 | 9 | 64 |
| `text_only_lora` | **0.5000** | **0.0000** | **0.1406** | 9 | 64 |

六项配对差值（`lora − matched`、`lora − text_baseline`、`lora − structured`，各 PR-AUC 与 AUC）**全部区间跨零**。

#### 这个结果的含义：适配器是退化的，不是"弱"

`text_only_lora` 的 AUC 恰好 `0.5000`、KS 恰好 `0.0000` 不是巧合。逐行核对 artifact 的 `predictions`：

- **64 行全部输出 `score: 0.0`**，9 个正样本与 55 个负样本拿到的分数完全相同。
- `severity` 64 行全部是 `low`。
- 首个 `reasons` 模板只有 **1 种**，64 行逐字相同（`no tail_risk event was recorded in the 30 days after <date>`）。

也就是说：**模型学会了输出格式，没有学会排序。** AUC 0.5 不是"模型很弱"，是"这一列不携带任何排序信息"。这与训练语料的算术一致——566 行里只有 18 个正样本，最小化损失的最优解就是恒定输出基率。因此：

- **`text_only_lora` 这一行今天不能被引用为 Track B 的性能**，也不能用来回答"文本有没有增量"。它现在是一个常数。
- 这一条**不是评测管道的问题**。Step 4 的既定目的是"确认推理路径可用、并看模型有没有学到东西"，两个问题现在都有明确答案：路径可用（64/64 解析、产物可复算），模型没学到排序。
- **AC 阶段的性质**由此改变：瓶颈从"缺一条评测路径"（已解决）转为"监督信号太薄"（见第 8 节待办 5）。

#### 顺带验证：Step 1 的修复确实改变了数字

`text_baseline` 从被取代那次运行的 PR-AUC **0.2400** 变为 **0.2956**（AUC 0.4404 → 0.4424）。差异的来源正是 prompt 里的任务标签：修好之前它是 `label=default_risk horizon_days=365`。这就是 3.2 节所声明作废范围的实际落地。

## 7. Step 5 — 在真实 test 上跑

本步原定目标是"拿到真实数据上的 `text_only_lora` 一行，使对外数字有一个真实语料来源"。Step 4 之后这个目标需要重新判断（见 7.1），确认后的执行与结果见 7.2–7.4。

### 7.1 开工前的两项事实（背景）

在合成数据上跑完后，真实数据这一跑的价值需要重新判断：它原本的目的（拿到一个真实数据上的 LoRA 数字）已经不成立，同时出现了一个新事实。判断在第 9 节提问并得到答复，**定为选项 A（不加载适配器、不用 GPU）**，执行结果见 7.2。

**新事实**：对真实面板（2221 行）干跑（`--dry-run`，不花 GPU）显示 **2139 / 2221 条 prompt（96.3%）超出 13,245 字符预算而被截断**，prompt 长度中位数 13,384 字符。真实 10-K/10-Q 的文本远长于合成语料，所以真实数据上的 prompt 几乎全部是被裁剪过的。截断计数已进入 artifact（`prompt.n_truncated`），不是静默行为。

不要把截断读成"只影响 LoRA 那一路"。已核验：`text_inputs()` 返回的就是**裁剪后**渲染出的 prompt，而 TF-IDF 基线消费的正是这个返回值（`cli.py` 把它整行传进 `evaluate_label`），SFT 语料的构建也走同一个 `chars_budget_for_seqlength()`（`pipeline.py:223` 与 `pipeline.py:449` 是同一预算的唯二两个出口）。因此：**预算是一个单一真相源，它同时决定训练输入、LoRA 评测输入、TF-IDF 基线输入三者**。这条事实也是问题 2 里"能不能只给评测放大预算"的答案依据。

**为什么原来的目的不再成立**：适配器是同一个，其恒定输出由训练语料决定，与输入来自合成还是真实无关；在真实 test 块上它会再次给出恒定分数，AUC 仍会是 0.5。

**仍然有价值的、且不需要 GPU 的部分**：真实数据上的 `structured` / `structured_matched` / `text_baseline` / `fused` 四行。它们会因为 Step 1 的修复与问题 3 的落地而刷新，并且它们才是项目对外唯一有意义的数字。这部分是 CPU 计算。

### 7.2 执行与结果

- 状态：**已完成**
- 命令：`shingan eval run --data-config configs/data/stage2_real.yaml`（按答复选项 A，不加载适配器）
- artifact：`artifacts/reports/20260923T065911Z.{json,md}`（首次；同一次运行覆写 `data/processed/panel.csv`，见 7.3.1）。第 11 节的改动落地后**用同一条命令重跑**，得到 `20260923T071547Z.{json,md}`，四行数字逐位相同（0.6294 / 0.7497 / 0.3281 / 0.6924），因此这一跑是确定性的，不是抽签。
- test 块 492 行 / 5 正 / 基率 0.0102；对比表 **4 行**（问题 3 的落地效果）

| path | AUC | KS | PR-AUC | 正样本 | n |
| --- | --- | --- | --- | --- | --- |
| `structured` | 0.6294 | 0.4407 | 0.0160 | 5 | 492 |
| `structured_matched` | 0.7497 | 0.5072 | 0.0248 | 5 | 492 |
| `text_baseline` | 0.3281 | 0.5692 | 0.0110 | 5 | 492 |
| `fused` | 0.6924 | 0.5606 | 0.0190 | 5 | 492 |

同一次运行的门禁：`headline_auc`（目标 `> 0.75`）实测 0.6924，**未通过**；`fusion_gain_pr_auc`（项目中心命题）`delta = +0.0030 [-0.0008, 0.0157]`，**区间跨零，未通过**；`brier_beats_base_rate` 为 −0.0053，**未通过**；`calibration_ece` 0.0112 通过；`stability_enough_usable_windows` 为 **1/15**，未通过。

**这张表仍然不能被引用为性能。** 5 个正样本下有三处互相矛盾的现象，都无法与抽样噪声区分：

1. `text_baseline` 的 AUC 是 **0.3281**，低于随机——它把负样本排得比正样本高。
2. 同一行的 KS 是 **0.5692**，高于 `structured` 的 0.4407。KS 是**不分方向**的最大 CDF 间距（`metrics.ks_statistic` 取的是 `np.abs`），方向另由 `ks_direction` 报出；本行的方向正是 `negatives_higher`。所以第 1 条与第 2 条不是互相印证，而是互相抵消。（当时这张表只印 KS、不印方向——这是第 11 节修掉的。）
3. `structured_matched`（只有 12 列）的 AUC **高于** `structured`（全特征约 40 列）。限制信息量反而变好，在这个样本量下只可能是噪声。

报告自己的第一条 caveat 就是"样本太小到不足以支撑性能声明"。**本步的结论是：真实数据上有一条可复算的四行基线与一份带区间的融合增益，其中融合增益仍跨零。**

### 7.3 两处必须与数字同时出现的发现

#### 7.3.1 被作废的不是一行，是旧的整张真实对比表

3.2 只把 `tail_risk` 的 `text_baseline` 标为作废。核验后范围要扩大：**旧真实报告 `artifacts/stage2/20260921T073120Z.*` 的每一行都不可比**，而且原因是**第二个、与 Step 1 无关**的原因。

- 那次运行留下的面板快照（`artifacts/stage2/panel.csv`，mtime 与报告生成时间只差 1 秒，因此确定是同一跑的输入）里，**7 个披露文本特征全部 2221/2221 为 NaN**：`disclosure_len_tokens`、`risk_factor_token_share`、`uncertainty_hits`、`going_concern_hits`、`restatement_hits`、`disclosure_len_chg`、`neg_kw_density_mdna`。同时 `n_sources` **在每一行上都比现在少 1**。
- 旧报告自己把这写进了 caveat：**"15 configured feature(s) held no observation anywhere in the training block and were dropped before fitting"**，列出的名单里 7 个文本特征全在，原文接着写 "nothing in the data layer supplies them"。
- 新报告同一位置的名单只剩 **8 个**（news 3 个、VIX 2 个、`credit_spread_chg_20d`、`beta_252d`、`turnover_20d`），**7 个文本特征已不在其中**。

即：旧的 `structured` 是在**完全没有披露文本层**的特征矩阵上拟合的，新的 `structured` 有这一层。AUC 0.7686 → 0.6294 由此解释。分界是提交 `8265756`（2026-09-21 19:37，"Ingest the SEC filing corpus, fix the defects it exposed and add CI"），旧报告 15:31 生成，**早于它 4 小时**。

**结论**：3.2 的作废范围由"一行"改为"旧真实报告的对比表整体"，并新增一条独立于 Step 1 的作废理由（特征矩阵缺层）。旧数字今后只在本文件里作为记录保留。

#### 7.3.2 真实正样本的多数落在切分的空档里

真实面板（34 家公司 / 2010–2024）的 `tail_risk` 共有 **39 个可观测正样本 / 1,778 个可观测行**（报告 `stability.coverage` 的 `total_positives` / `total_rows` 就是这两个数）。按 split 拆开：

| split | 可观测行 | 正样本 |
| --- | --- | --- |
| train | 779 | 4 |
| valid | 354 | 1 |
| test | 492 | 5 |
| **excluded** | **124** | **27** |
| **purged** | **29** | **2** |
| 合计（可观测） | 1,778 | 39 |

- `excluded` 的 124 行**全部是 2020 年**，含 27 个正样本。原因是切分几何：`valid` 到 2019-12-31 结束、`test` 从 2021-01-01 开始，**2020 整年夹在两者之间**。而按年份看，2020 是这 34 家里唯一出现系统性下跌的年份（39 个正样本里 27 个），2021 年 0 个、2022 年 4 个、2023 年 1 个。
- `purged` 的 29 行是 2016 年（2 个正样本），被 `purge_days=60` 的边界缓冲吃掉。

**这条改写了待办 5 的形状。** 原表述是"真实 `tail_risk` 的 train 只有 779 行 / 4 个正样本，瓶颈是正样本数"。准确的说法是：**已有的 39 个正样本里有 29 个（74.4%）被切分排除，而被排除的恰好是唯一包含系统性尾部事件的那一年。** 所以"扩股票池"不是唯一解，甚至不该是第一解——先问 2020 能不能进入训练/评测几何。

这不是纸面建议：**滚动窗口机制本来就把 2020 覆盖在内**。新报告的 walk-forward 表里 W1 的 test 窗口是 `2020-01-01 .. 2022-12-30`；按年计的标签覆盖里 `n_usable = 4`，而 2020 **不在**那 11 个 `insufficient` 年份名单里。也就是说同一份数据、同一套代码，换一种切分就能用上那 27 个正样本，而固定切分的 `test` 块把它们整体排除了。

### 7.4 本步没有验证什么

- **真实数据上的 `text_only_lora` 行没有测**（按答复选项 A）。因此"这个适配器在真实数据上也是常量"至今是**推断**而非实测；推断依据是常量由训练语料决定，与 7.1 的截断事实无关。
- **截断的后果没有量化**。96.3% 的真实 prompt 被裁到前 13,245 字符；本步只是让四条基线在这个边界下产出数字，没有测"多看到后面的文字会改变什么"。这条边界必须与数字同时出现，计数器是各产物里的 `n_truncated`。
- **`excluded` / `purged` 能否用别的切分几何救回来没有验证**（7.3.2）。验证它要改 `split` 配置并重跑，属于待办 5 的范围。
- **没有重跑 `eval lora`**。`artifacts/lora-eval/20260923T062853Z` 仍是合成语料的产物，本步未触碰。

## 8. 本组步骤之外的待办（明确不做，避免被读成已覆盖）

1. **真正不带结构化信号的 text-only 适配器**（D1 的未采纳选项）。ADR-0003 意义上的 `text_only_lora` 需要它。
2. **`text_only_zero_shot`**：底座模型不带适配器的 zero-shot 行，用于回答"微调是否必要"。
3. **打乱文本的 placebo 对照**：ADR-0003 明确要求。
4. **P0 溯源缺口（5 项）**：`run.json` 不含语料溯源（无 source / panel 路径 / 哈希 / `is_synthetic`）；`sft/manifest.json` 同样无 `is_synthetic`；模型卡模板把数据源**硬编码**在 `{{data_sources}}` 旁；模板要求合成训练必须标 `trained_on=synthetic` 但**没有对应占位符**；模板写 `warmup ratio {{warmup}}` 而实际产出的是 `warmup_steps=3`。
5. **正样本扩容，但要先切分后扩池。** 真实面板 `tail_risk` 的 39 个可观测正样本里有 **29 个（74.4%）落在 `excluded` / `purged`**，且被排除的正是唯一有系统性下跌的 2020 年（见 **7.3.2**）。所以顺序是：先问能不能改切分几何把 2020 用上（同一份数据、同一套代码，滚动窗口已经覆盖它），再决定要不要为更大的股票池付 EDGAR 的下载成本。用 4 个正样本训 14B 只会学到恒定输出 0——**瓶颈是监督密度与切分几何，不是数据来源。**
6. **`sample_id` 不是唯一键。** `sft_examples` 是每行 × 每标签一条样本，所以 `sample_id` 在三个标签之间重复；任何以它为行标识的下游工具都会把三条样本混成一条。提示词比对必须用 `(sample_id, label)`（本步第一版就踩了这个坑，被 `--verify-prompts` 抓出 260/782 的假不匹配）。尚未审查**其余**以 `sample_id` 为键的下游用途。
7. **发布模型卡**：必须带 `trained_on=synthetic`，且 Evaluation 段写 `no real-data evaluation has been performed`；而这两条依赖待办 4 的模板修复。

**已在第 11 节完成、不再挂在待办里的一项**：对比表只印 `ks` 不印方向。`ks_direction` 现在进入 `eval run` 的对比表（JSON / Markdown / 控制台三处），`report.py` 第 3 节加了说明；`tests/test_matched_in_report.py` 有 3 项守着。

## 9. 问题与答复（2026-09-23）

我在 Step 4 结束时**停下**，没有开始 Step 5。这不是技术阻塞，而是 Step 4 的结果改变了 Step 5 的性价比，而"这一步产出的数字将来怎么被引用"属于口径决定，不该由我代你定。以下三问的答案会直接改写 Step 5 的定义。三问都已在动手前提出并得到答复，答复与据此做的动作记录在每问末尾。

### 问题 1 — 还要不要花 GPU，给这个适配器在真实数据上打分？

**背景**：原定的理由是"至少拿到一个真实数据上的 LoRA 数字"。现在已知适配器的输出是常量，而常量是**训练语料**的属性，与输入来自合成还是真实基本无关，所以这一跑大概率只是把同一个常量在第二个数据集上再产一次。

诚实地说，"基本无关"不等于"已验证"：真实 10-K 文本与合成语料不同，理论上仍可能激出方差，只是按 566 行 / 18 正样本这个算术，预期不会。所以选项 B 不是零信息，它是"给常量加一份第二数据集的证据"。

| 选项 | 代价 | 得到什么 | 失去什么 |
| --- | --- | --- | --- |
| **A（推荐）不用 GPU**：真实数据只跑四条 CPU 基线；`text_only_lora` 行引用已测的合成结果，并在 artifact 与文档里显式标注语料来源 | ≈0 GPU | 刷新后的真实数字（其中 `text_baseline` 因 Step 1 修复而变化） | 一个可复算的"真实数据上也是常量"的书面证据 |
| **B 仍用 GPU 跑**：`shingan eval lora --data-config configs/data/stage2_real.yaml`（492 行） | 十几分钟 GPU（含约 23 s 加载） | 结论从"预期恒定"变成"实测恒定"，对外可写"在真实数据上也实测过" | 十几分钟 GPU；以及仍需向读者解释为什么报告里有一行常量 |

**建议 A 并附条件**：如果你打算近期把这件事对外说（模型卡、README、HF 卡片），那 B 的"实测过"值这十几分钟，选 B。

**你的答复：A（不用 GPU）。** 据此 Step 5 = 真实数据上四条 CPU 基线；同一 artifact 里**不出现** `text_only_lora` 行，改由本文档与 §7.2 指向合成 artifact，并标注语料来源。

### 问题 2 — 96.3% 的真实 prompt 被截断：接受并声明，还是提高预算并重训？

**先说什么是无效选项**：只给评测放大预算、不重训。这会被 `--verify-prompts` 的逐字节门直接拒绝（重建的 prompt 不再等于 SFT 文件里的 user turn），**而这个门失败是正确行为**——门失败正说明评测输入与训练输入已经不同。即使绕过门，把一个只见过 4k 上下文窗口内容的适配器放进 8k prompt 里评，测的仍是训练分布外的输入形状。

| 选项 | 代价 | 说明 |
| --- | --- | --- |
| **A（推荐）接受，并写进结论边界**：所有真实数据的文本行（**含 `text_baseline`**，见 7.1）只在"每份文档的前 13,245 字符"上成立；artifact 与报告带 `n_truncated`。把"提高预算"并入待办 5，与正样本扩容一起做 | 0 | 那时无论如何都要重训一次，省一次 42 min |
| **B 现在就提高 `lora.max_seq_length` 并重训** | +42 min GPU 与更高显存 | 真实语料可见内容从约 13k 升到约 26k 字符；但**在当前正样本算术下重训仍只会产出常量**，等于把 GPU 花在一个已知结论上 |

**建议 A**。

**你的答复（原文）：** *"I am happy to do this properly and achieve a better result model. So based on this decide if 1 or retrain."* —— 即：以"能否得到更好的模型"为准来定 A 还是 B。这个判断在 §10 展开，结论是**接受截断（A），并把提高预算并入正样本扩容那一次重训**。

### 问题 3 — 把 `structured_matched` 提升为 `shingan eval run` 的标准一行？

**背景**：`structured_matched` 目前只存在于新命令 `eval lora` 的产物里。标准报告 `shingan eval run` 依旧只有 `structured` / `text_baseline` / `fused` 三行。若问题 1 选 A（不用 GPU），真实数据的同信息量基线就只能通过新命令获得，而新命令的产物**同时**包含一个常量 LoRA 行——四行基线与一个常量被渲染进同一张表，是很容易被误读的形态。

| 选项 | 代价 | 收益 |
| --- | --- | --- |
| **A（推荐）加进 `eval run`** | 标准报告的对比表多一行；`05` 需标注"此前的报告没有这一行" | 真实数据的基线可纯 CPU 刷新；"同信息量基线"从此是每份标准报告自带的，不依赖谁记得跑新命令 |
| **B 维持现状** | 需要明确说明 matched 行仅在新命令里可得 | 不动评测契约 |

**建议 A**。这一条与问题 1 是耦合的：选 A + A 才是干净的组合。

**你的答复：A（加进 `eval run`）。已实施并验证。** 落地细节：

- `pipeline.COMPARISON_ORDER = (structured, structured_matched, text_baseline, fused)` —— 报告打印 **4 行**，matched 排第二，紧贴它要为之做减法的两行。
- **`PATH_ORDER` 仍是三条轨道**（`structured` / `text_baseline` / `fused`）。这一条是刻意的：`PATH_ORDER` 决定哪些 `score_*` 列写进面板、哪两条序列进融合，把控制项放进去会让它经漂移、滚动稳定性、压力测试与消融表外溢。三条轨道，四行对比。
- `matched_signal_report()` 拆出内部 `_matched_signal_scores()`，`evaluate_label()` 与 `eval lora` 共用同一个实现。此前 `eval lora` 自己拟合一次、`eval run` 完全不产出——同一个名字下迟早会出现两个数。
- 面板没有那 12 列时**不静默丢弃**：`outcome.matched_reason` 记录原因，`run_pipeline` 把它写进报告的 notes 通道（表格少一行而不说明，读起来仍是一张完整对比表）。
- `_label_table()`（控制台）也改用 `COMPARISON_ORDER`，否则 dict 顺序会把 matched 排到末行，与 artifact 的行序不一致。
- **交叉验证**：合成数据上两条命令给出的 matched 行完全相同 —— `eval lora` 的 artifact（`20260923T062853Z`）与 `eval run` 的新报告（`20260923T065029Z`）都是 AUC `0.5091` / KS `0.2727` / PR-AUC `0.1492`。
- 测试：`tests/test_matched_in_report.py`（8 项），含"控制项不得进入 `PATH_ORDER`"、"少一行时不打印空行"、"控制台行序 = artifact 行序"、"被拒绝时记录原因而不抛出"、"两个调用方走同一实现"。全量 **242 passed**。

### 已按以下默认处理（如无异议我直接做，不再单独确认）

1. **旧 artifact 的作废标记**。`artifacts/` 整个被 `.gitignore` 忽略（已核验：`git ls-files artifacts` 为空），所以"作废"不是 git 动作，而是本地可读性问题。计划在 `artifacts/reports/20260923T055845Z/` 与 `artifacts/stage2/20260921T073120Z/` 各写一个 `SUPERSEDED.md`，写明哪一行作废、被哪个目录取代。与本仓既有约定一致——让一个数字难以被误读。
2. **3.2 表中列出的两个旧 `text_baseline` 数字**在结论文档里一律不再出现，只在本文件里作为"作废记录"保留。
3. **修正 `docs/04-training.md` 第 5 节**（"14B 真实训练尚未产出 checkpoint"已被本次运行证伪）归入 Step 6。**已完成**：第 5 节现在写"已产出 checkpoint，但它证明的是链路，不是能力"，并新增"同分布评测显示输出是常量"与"适配器从未在真实文本上被评分过"两条；`docs/index.md` 的状态表、`README.md` 的状态表与诚实性声明、`docs/05` 开头"不存在任何真实数据评测结果"一句同批修正。

### 为什么这三问不代你定

它们都不是"哪种实现更对"，而是"这份产物将来给谁看、看到哪个数字"。问题 1 决定要不要产出一个已知为常量的证据；问题 2 决定真实文本行的适用边界怎么写进结论；问题 3 决定同信息量基线是标准件还是特例。三者都会改变别人 clone 这个仓库后能得出的结论，因此按你的要求先确认再动手。

## 10. 问题 2 的决定：接受截断，把提高预算并进下一次重训

**你的指示**：以"能否得到更好的模型"为准决定 A 还是 B。

**结论：选 A（接受并写进结论边界），不现在为提预算而重训。** 理由按权重排序。

### 10.1 提预算不会让模型变好——它不是当前的瓶颈

同一份语料、同一个目标，在 4k 窗口与 8k 窗口下的最小损失解都是"输出基率"。合成语料 566 行 / 18 正样本给出 AUC 恰 `0.5000`、KS 恰 `0.0000`，这是**证据**：模型连训练集里那 18 个正样本都没有形成排序。窗口变大只让模型多看到一些未标注文本，不改变"它没有可学的排序信号"这一点。所以 B 的 42 分钟买到的是"更长的输入 + 同一个常量"。

### 10.2 卡住"更好的模型"的是监督密度，量级差距很大

真实面板（2,221 行 / 34 家公司 / 2010–2024）的 `tail_risk` 正样本：train **4**、valid **1**、test **5** —— 全样本 **10 个**，在 1,625 个可观测行上正率 0.6%。要让 QLoRA 学出排序而不是学出基率，正样本得从"个位数"变成"几百个"。

| 股票池 | 预计行数 | 预计正样本（按 0.6%） | 文本侧代价 |
| --- | --- | --- | --- |
| 34（现状） | 2,221 | 10 | 已下载（2,879 条 filing / 285 MB） |
| ~200 | ~13,000 | ~80 | EDGAR + XBRL 约 6 倍下载量与速率限制 |
| ~600 | ~40,000 | ~240 | 约 17 倍 |

**这是推算，不是实测。** 它假设事件率在不同股票池上稳定，而现有 34 家是刻意挑过"确实经历过压力"的名字（见 `configs/data/stage2_real.yaml` 的注释），所以扩池后的真实正率**大概率更低**。

要把它变成实测其实很便宜：`tail_risk` 的标签**纯由价格算出**，不需要文本。因此可以先用 `sources: [prices_yfinance]` 在一份**独立缓存目录**里扫一遍更大的股票池，数出真实正样本数，**再**决定为哪些公司付 EDGAR 的下载成本。独立目录这一步是硬要求——`data/raw/real/` 里的 prices 表一变，刚跑出来的真实报告就不再能从缓存复现了。

### 10.3 时间顺序上，提预算应当与扩池合并

扩池之后无论如何都要重跑一次 `data sft` + `train lora`；把 `lora.max_seq_length` 一起提高，成本相同而收益叠加。**现在单独重训是纯粹的重复支出。**

### 10.4 有一条不需要训练、也不需要正样本的路

`text_only_zero_shot`（底座模型不带适配器、直接用同一条 prompt 去问）**不需要任何训练数据**，因此恰好绕开 10.2 那个瓶颈。它有两个价值：

1. 它是微调的直接对照物——ADR-0003 让 `text_only_zero_shot` 存在的意义就是回答"微调是否必要"。
2. 它是**目前最有希望成为第一个可引用的真实 Track B 行**。底座 Qwen3-14B 没有被 18 个正样本压成常量，它的输出大概率有方差（**这是预测，未实测**）。若真有方差，AUC 无论高低都是真数字；若它也是常量、或解析失败率极高，那同样是答案——关于"14B 底座能不能做这件事"的答案。

代价：给 `shingan eval lora` 加一个"不带适配器"的模式（复用同一条 prompt 构造、同一套解析器、同一份 artifact 契约），然后在真实 test 块（492 行）上跑一次 GPU 推理，约十几分钟。

### 10.5 两个候选下一步

**注意命名**：这里不再用 A / B，因为本节标题里的 A / B（问题 1 的选项、问题 2 的选项）已经各自用过一轮字母，而两个问题的 A 含义不同。以下按内容命名。

| 候选 | 代价 | 产出 |
| --- | --- | --- |
| **零样本真实数据行**（建议先做） | 小改 `eval lora` 加一个"不带适配器"模式 + 约十几分钟 GPU | 一个真实的 `text_only_zero_shot` 数字（好坏都是答案），以及真实的解析失败率 |
| **扩股票池** | yfinance 扫描（分钟级）+ EDGAR 下载（数百家、GB 级、受速率限制）+ 重建面板 + 一次重训（含提高 `max_seq_length`） | 一个**可能**有排序能力的适配器；至少得到"正样本仍不够"的实测边界 |

**建议顺序：先做零样本行。** 它只要十几分钟、不改数据缓存，而且是扩池的对照前提——不知道零样本基线，就无法判断微调有没有用。扩池的股票池规模需要事先定，因为它决定下载量与磁盘占用。

**问题 2 选择"接受截断"的即时后果**：真实数据的文本行（含 `text_baseline`）只在"每份文档的前 13,245 字符"上成立。这条边界必须与数字一起出现，artifact 里的 `n_truncated` 是它的计数器。

## 11. 顺带修掉的一处渲染缺陷：对比表有 KS、没有方向

**发现方式**：7.2 的表里 `text_baseline` 的 AUC 是 **0.3281**、KS 是 **0.5692**。两个数印在同一行，读起来是"文本基线区分度最强"。

**为什么那是误读**：`metrics.ks_statistic` 取的是 `np.max(np.abs(cdf_pos - cdf_neg))` —— **不分方向**。方向由另一个函数 `metrics.ks_direction` 报出（`positives_higher` / `negatives_higher` / `tied` / `undefined`），它的 docstring 写着：

> A model whose negatives score higher than its positives has a KS with the right magnitude and the wrong sign. That is not a numerical curiosity; it means a feature or a score is inverted, and reporting KS alone hides it.

本次真实运行正好命中这句话：`text_baseline` 的方向是 `negatives_higher`，即排序是反的。

**改了什么（三处，都是加法）**：

| 位置 | 改动 |
| --- | --- |
| `pipeline._comparison_table()` | 每行增加 `ks_direction`。JSON 走 `comparison.to_dict()`、Markdown 走 `_frame_to_markdown()`，两处自动带上，不需要各自改 |
| `cli._label_table()`（控制台） | 增加 `KS dir` 列，紧跟 `KS` |
| `report.py` §3 Path comparison 的表下 | 加一句说明：`ks` 是 CDF 最大间距、不带方向；`ks_direction` 是符号、由两类的均值比较得出。**两者回答的不是同一个问题，所以可以不一致** |

**为什么算缺陷而不算新功能**：`eval lora` 的产物**一直带着**这个字段——它的行来自 `ClassificationReport.as_dict()`（`metrics.py`）。也就是说同一个指标在同一个项目的两份产物里有**两种保真度**，而标准报告恰好是保真度更低的那一份。

**加上这一列之后立刻又暴露出一件事**（这一条是本节的副产物，值得单独记下）。同一次运行的四行现在读作：

| path | AUC | KS | ks_direction |
| --- | --- | --- | --- |
| `structured` | 0.6294 | 0.4407 | **`negatives_higher`** |
| `structured_matched` | 0.7497 | 0.5072 | `positives_higher` |
| `text_baseline` | 0.3281 | 0.5692 | `negatives_higher` |
| `fused` | 0.6924 | 0.5606 | `positives_higher` |

`structured` 的 AUC 高于 0.5，但它的方向是 `negatives_higher`。这不是列算错了，已用面板独立复核：test 块上 `score_structured_tail_risk` 的正样本均值 0.002937、负样本均值 0.003243，负样本确实更高。原因是两个统计量问的不是同一件事——`ks_direction` 比的是**两类均值**，KS 量的是**CDF 最大间距**；5 个正样本里有几个负样本拿到高分，就足以把负类均值抬到正类之上，同时 AUC 仍然大于 0.5。

**因此这一列不解决噪声，它让噪声可见。** 这符合本步的整体口径：5 个正样本下这些数字都不能被引用为性能（7.2），而把不一致印出来，比让表格看起来自洽要好。

**测试**：`tests/test_matched_in_report.py` 增加 3 项（该文件共 11 项）：对比表每行都带方向；用算术（负样本分数全部高于正样本）钉住 `ks_statistic == 1.0` 与 `ks_direction == "negatives_higher"` 这对定义，避免以后有人把方向合并进 KS 本身；控制台把方向印在 KS 旁边。全量 **245 passed**。

**没有改什么**：**门禁**。`headline_ks` 门本来就单独报 `direction=positives_higher`（`report.py`），判定一直是对的；这次修的是**表格的读法**，不是判定。


