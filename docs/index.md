# Shingan 文档地图

**Shingan**（心眼）是一个开源研究项目，目标是验证"文本信号能否在结构化财务/量价信号之外，为上市公司风险预测提供可度量的增量信息"。它采用双轨架构：Track A 用 gradient-boosted trees 拟合结构化信号（财务比率 + 市场微观结构/技术特征），Track B 用 QLoRA 指令微调的 LLM 读取非结构化文本（10-K/10-Q/8-K 摘录 + 新闻），再由一层轻量 fusion 层把两条轨的分数融合。项目当前处于 POC 阶段，以合成数据端到端跑通流水线，真实数据上的预测能力**尚未得到任何验证**。

- 仓库：`shuurai/shingan-finrisk`
- 维护者：Shane（GitHub `shuurai`），Windows 11 + RTX 5090 (32 GB)
- License：Apache-2.0

## 核心论点：为什么是双轨

项目最初的设想是"把 20 年年报 + 新闻 + 交易信号拼成一个大文本，训一个 LoRA 去预测风险"。本项目**明确否决了这个做法**，理由有两条：

1. **信号形态不同，需要不同的归纳偏置。** 财务比率与量价特征是有量纲、近似平稳、样本量大的数值面板，gradient-boosted trees 在这类数据上稳定、可校准、可归因；把数字序列化成 token 交给 LLM，等于用一个高方差、样本效率低的函数逼近器去做一件树模型做得更好的事。
2. **可验证性依赖分离。** 双轨把"文本贡献了什么"变成一个可以消融的实验问题：structured-only 与 text-only 两条基线各自独立存在，fused 只有在 PR-AUC 上显著超过 structured-only 时才说明文本轨有价值。单一 LoRA 把两种信号混在一个不可分解的权重里，无法回答这个问题。

因此本项目有一条硬性评测要求：**任何声称有效的模型，都必须同时给出 structured-only、text-only、fused 三路结果，外加无模型基线**。详见[评测框架](05-evaluation.md)。

## 文档目录

| 文档 | 内容 | 适合谁读 |
| --- | --- | --- |
| [本页](index.md) | 文档地图、状态表、术语起点 | 所有人 |
| [01 架构](01-architecture.md) | 双轨架构与理由、各边界输入契约、证据化输出契约、明确非目标 | 想理解设计取舍的人 |
| [02 数据](02-data.md) | 三条数据腿与真实来源、as-of/no-lookahead 纪律、去重与对齐规则、合成数据生成器、处理后面板数据字典、许可与署名 | 要接数据的人 |
| [03 标注](03-labeling.md) | 风险分类学、三个 in-scope 标签的精确 horizon 与阈值、JSONL schema、正负样本示例、基率现实、降采样规则、标注溯源 | 要造标签的人 |
| [04 训练](04-training.md) | 结构化轨训练与校准、底座选型（4B/8B/14B/30B-A3B vs 32 GB）、QLoRA 配置表、指令格式、显存预算、Windows/Blackwell 安装配方、失败模式 | 要跑训练的人 |
| [05 评测](05-evaluation.md) | 指标与门槛、purged walk-forward + embargo 切分、压力测试、基线与消融、漂移监控、可解释性、验收门槛、证伪条件 | 要判断这个项目是否成立的人 |
| [06 Windows 环境](06-windows-setup.md) | Windows 11 + RTX 5090 的逐步搭建、cu128 安装命令、验证脚本、`shingan doctor` 期望输出、WSL2 回退、排错表 | 要在这台机器上跑起来的人 |
| [07 路线图](07-roadmap.md) | Stage 0-5 分阶段计划、每阶段的风险与推进条件 | 规划节奏的人 |
| [08 命名决策](08-naming.md) | 四个候选名的比较、评分标准、Shingan 胜出理由、命名族、改名流程 | 关心命名与品牌一致性的人 |
| [09 LoRA 评测接入](09-lora-evaluation.md) | 把已训好的适配器接进评测的**步骤日志**：目标、决策与理由、实际命令、产物、验证了什么与没验证什么（含零样本对照臂 `text_only_zero_shot` 与它的指令契约缺口） | 想知道"文本轨有没有增量"这个问题的答案从哪来的人 |
| [ADR-0001](adr/0001-project-name.md) | 项目命名决策记录 | — |
| [ADR-0002](adr/0002-training-stack-windows.md) | cu128/Blackwell 与训练栈选择（TRL+PEFT+bitsandbytes） | — |
| [ADR-0003](adr/0003-dual-track-over-single-lora.md) | 为什么拒绝"单一 LoRA 吞下全部输入" | — |
| [模型卡模板](../templates/model_card.md) | Hugging Face model card 模板 | 发布模型的人 |
| [数据集卡模板](../templates/dataset_card.md) | Hugging Face dataset card 模板 | 发布数据集的人 |

## 状态表：今天能跑什么，哪些还没有

状态口径：

- `已实现·已验证` — 代码存在，且在合成数据上端到端跑通并有输出产物。
- `已实现·未验证` — 代码存在，但**尚未**对真实数据或实时端点执行过，行为未经证实。
- `已产出·不可引用` — 数字已经产出且可复算，但按本项目的门禁标准它**不能被引用为能力声明**（正样本太少，或该路是常量输出）。"有产物"与"能下结论"是两件事，这个状态专门用来分开它们。
- `接口已定义·未验证` — live 端点适配器（网络调用）已写好，从未真正调用过成功路径。
- `设计已定·未实现` — 设计在本套文档中已固定，代码尚未写。
- `超出 POC 范围` — 有意不做。

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| 包结构 / `pyproject.toml` / `src/shingan/` 模块划分 | 已实现·已验证 | hatchling，版本单一来源于 `src/shingan/__about__.py` |
| CLI 命令面（`doctor` / `data` / `train` / `eval` / `demo` / `publish`） | 已实现·已验证 | 见 `src/shingan/cli.py`，Typer + Rich |
| 合成数据生成器（`data/synthetic.py`） | 已实现·已验证 | 确定性、离线，`shingan data synth` |
| `shingan demo`（CPU 端到端出 markdown 报告） | 已实现·已验证 | 证明流水线连通，**不证明预测能力** |
| as-of / leakage 断言（`leakage.py`） | 已实现·已验证 | 合成数据上的断言通过 |
| 时序切分 + purge/embargo（`eval/splits.py`） | 已实现·已验证 | 合成数据上的折数与区间已验证 |
| 结构化轨（`models/structured.py`） | 已实现·已验证 | 合成数据可训练，validation fold 上做校准 |
| 文本基线（`models/text_baseline.py`） | 已实现·已验证 | 词频/TF-IDF 类基线，供 text-only 一路对比 |
| 评测指标与报告（`eval/metrics.py`、`report.py`） | 已实现·已验证 | 合成数据上有输出 |
| SEC EDGAR 客户端（`data/edgar.py`） | 已实现·已验证 | 已对 live EDGAR 跑通：`data/raw/real/` 下 filings 2,879 / fundamentals 2,189。踩过的坑是 **UA 里的联系邮箱域名**决定 200 还是 403，不是 UA 的形状 |
| 价格数据适配器（`data/prices.py`） | 已实现·已验证 | 已对 live 端点跑通：prices 114,424 行 / 34 家公司 / 2010–2024（`scripts/fetch_real.py`） |
| 新闻适配器（`data/news.py`） | 接口已定义·未验证 | FNSPID 是离线快照、未接入，所以真实面板上 `n_news_30d` / `sent_*` 全为零，`default_risk` 与 `fraud_risk` 无法评测 |
| QLoRA 训练（`models/lora.py`） | 已实现·已验证 | `artifacts/lora-contract-v2/adapter/` 是**当前**适配器（Qwen3-14B、108 步 / 48:08、train_loss 0.3624，语料仍为合成数据）。`artifacts/lora/` 是**旧契约下**的 108 步适配器，原地保留，但 prompt 契约变更后它已被门禁**正确地**拒绝打分。**当前适配器的输出在同分布评测上是常量**（AUC 恰 0.5、KS 恰 0.0），因此证明的是链路可用，不是预测能力——见 [09](09-lora-evaluation.md) 第 13 节 |
| LoRA 推理与评测路径（`models/lora_inference.py`、`eval/lora.py`、`shingan eval lora`） | 已实现·已验证 | 三档 `--mode`：`adapter`（默认，行为不变）/ `zero_shot` / `both`（同进程依次跑两臂，**差值才是配对测量**）。训练输入的 prompt 被逐字节重建（260/260 user 轮 **+ 782/782 system 轮**，`--verify-prompts`）；完整合成 test 块 64/64 解析成功 |
| 零样本臂 `text_only_zero_shot`（`shingan eval lora --mode both`） | **已实现·已验证** | 两臂同进程配对。合成 test 块上 **64/64 解析成功**（失败率 0.0000），AUC 0.5172 / KS 0.1273 / 方向 `positives_higher`（表里唯一方向正确的一行）。命令与配对差值都能跑；**但这一行不能引作能力声明**，因为面板是合成的、文本信号是生成器植入的。真实面板（492 行 / 5 正样本）**479/492 解析，AUC 0.7542 / KS 0.5451 / PR-AUC 0.0960、方向 `positives_higher`**；对 `structured_matched` 的配对差 AUC +0.0034、PR-AUC +0.0702，区间均跨零——值得跟踪的信号，不是性能结论；产物早于引用审计。见 [09](09-lora-evaluation.md) §13.9.1 |
| 同信息量基线 `structured_matched` | 已实现·已验证 | 标准报告自 2026-09-23 起有 4 行；`text_only_* − structured_matched` 才是文本增量，见 [05](05-evaluation.md) 第 5.1 节 |
| Fusion 层（`models/fusion.py`） | 已实现·未验证 | logistic stacker / rank-average 可跑；真实 `tail_risk` 的 valid 折只有 **1 个正样本**，报告的融合增益区间跨零 |
| 真实数据结果（AUC/KS/PR-AUC 等数字） | 已产出·不可引用 | `artifacts/reports/` 里有真实面板上的四行对比。test 块只有 **5 个正样本**，且 39 个可观测正样本里 29 个落在切分空档（[09](09-lora-evaluation.md) 第 7.3.2 节）。这些数字描述这份样本，不描述模型能力 |
| 三个 in-scope 标签的真实事件标注 | 设计已定·未实现 | 需要评级历史、EDGAR 重述/执法、审计意见等数据源接入 |
| `publish hf` 到 Hugging Face Hub | **已实现·已验证**（数据集部分） | 数据集卡 + `panel.parquet` 已发布至 `shuurai2000/shingan-finrisk-labels`，`load_dataset` 端到端验证通过（2221 行 / 68 列 / tail_risk 39 正样本，与卡片声明一致）。`--data-file` 上传前按卡片 `n_rows` 预检，矛盾则拒绝发布。模型卡仍有 61 个占位符（需适配器评测数字），按设计被门禁拦住 |
| `liquidity_risk` / `event_driven_risk` / `macro_contagion_risk` | 超出 POC 范围 | 定义见[标注](03-labeling.md)，POC 不实现 |

### 诚实性声明

本仓库的合成数据 demo 只能证明**流水线可运行**——数据生成、as-of 检查、特征构造、两条轨训练、融合、切分、指标计算、报告渲染都接通了。它**不能**证明模型具有真实预测能力。合成数据里的信号是生成器人为植入的，指标高只反映实现与设计一致。

真实数据结果**已经产出，但它读不出能力**。`artifacts/reports/` 里有真实面板（2,221 行 / 34 家公司 / 2010–2024）上的四行对比，EDGAR 与价格两条腿都已对 live 端点跑通并有落盘数据。但那个 test 块只有 **5 个正样本**，而且真实 `tail_risk` 的 39 个可观测正样本里有 **29 个落在切分的空档里**（2020 整年夹在 `valid` 与 `test` 之间，被排除的正是唯一出现系统性下跌的那一年）——详见 [09](09-lora-evaluation.md) 第 7.3.2 节。

因此这套文档中出现的所有门槛值仍应读作**目标值（target/gate）**，而不是达成值；真实数据上的融合增益至今区间跨零。文本轨的贡献仍然是**未知**——同分布评测已证实那个适配器的输出是常量；零样本臂在修掉指令契约缺口后**已经能测且不是常量**，且真实面板上有了第一个实测基线（AUC 0.7542、方向正确，配对区间跨零，见 [09](09-lora-evaluation.md) §13.9.1）；真正不带结构化信号的 text-only 适配器仍不存在。
