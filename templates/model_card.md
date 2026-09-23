---
license: apache-2.0
base_model: Qwen/Qwen3-14B
library_name: peft
language:
  - en
tags:
  - financial-risk
  - risk-assessment
  - qlora
  - lora
  - peft
  - qwen3
  - text-classification
  - evidence-grounded
pipeline_tag: text-classification
pretty_name: Shingan Qwen3-14B FinRisk
---

<!--
模板使用方式：
  1. 把 {{...}} 占位符替换为实测值。
  2. 未实测的字段不要猜数字，写 "not evaluated" 并说明原因。
  3. Evaluation 段只能包含已有可复现报告的数字；没有报告就写 "no real-data evaluation has been performed"。
  4. 发布前删除本注释块与所有未替换的占位符。
  5. data_provenance / data_sources / training_data_ref 必须取自本次训练 run.json 的
     data 段（训练与验证文件的 sha256、SFT manifest 及其 raw_files 哈希），不得手工
     填写来源描述：模板不替任何一次训练断言它用了什么数据。
-->

# Shingan Qwen3-14B FinRisk

Shingan（心眼，"the mind's eye"）是一个上市公司**风险评分**研究项目。本模型是其中的文本轨（Track B）：在 `{{base_model_revision}}` 上以 QLoRA 指令微调的适配器，读取 `as_of` 时点之前已公开的披露文本与新闻，输出一个证据化的风险 JSON。

本模型的分数是 Shingan 双轨架构中的一路。与之搭配的结构化轨（GBDT）与融合层不在本仓库中；融合结果见 [Shingan 项目文档]({{repo_url}})。**单独使用本适配器不构成完整系统。**

## Model Details

| 项 | 值 |
| --- | --- |
| 项目 | Shingan（心眼） |
| 本 artifact 的角色 | 文本轨（Track B）LoRA 适配器 |
| Base model | `Qwen/Qwen3-14B` |
| Base model revision | `{{base_model_revision}}` |
| 微调方式 | QLoRA（4-bit nf4，double quant），LoRA r=32 / alpha=64 / dropout=0.05 |
| 目标模块 | `q_proj`、`k_proj`、`v_proj`、`o_proj` |
| 注意力实现 | `sdpa`（未使用 flash-attn） |
| 可训练参数量 | `{{trainable_params}}` |
| 序列长度 | `{{max_seq_length}}` |
| 训练框架 | TRL + PEFT + bitsandbytes |
| 精度 | bfloat16 compute，4-bit nf4 权重 |
| License | Apache-2.0 |
| 训练日期 | `{{training_date}}` |
| 代码版本 | `{{git_commit}}` |
| 训练配置快照 | `{{config_path}}` |

### 输入输出

- **输入**：一段包含 `<AS_OF>`、`<STRUCTURED_SIGNALS>`、`<FILING_EXCERPTS>`、`<NEWS>` 四个块的 prompt，并要求评估三个风险标签之一。
- **输出**：单个 JSON 对象，字段为 `label`、`severity`、`score`、`horizon_days`、`reasons[]`、`evidence[]`。
- `evidence[].quote` 必须是输入文档中的**逐字子串**。使用者应当自行做子串校验，不能假定模型一定遵守（见 Limitations）。

### 风险标签

| 标签 | 定义 | horizon |
| --- | --- | --- |
| `default_risk` | 评级下调 >= 2 档，或进入破产/违约程序 | 365 日历日 |
| `fraud_risk` | 财务重述、SEC 执法行动、或非标审计意见 | 730 日历日 |
| `tail_risk` | 30 个交易日内累计回撤低于 -30% | 30 交易日 |

`liquidity_risk`、`event_driven_risk`、`macro_contagion_risk` 在项目中有定义但不在 POC 范围内，本模型**未**针对它们训练。

## Intended Use

**本模型仅用于研究与教育目的。**

允许的用途：

- 学术研究与方法论探索，特别是"文本信号能否在结构化财务/量价信号之外为风险预测提供增量"这一问题。
- 教学与研究复现。
- 作为更完整流水线的组件进行评测与消融。

**明确不适用（Out-of-scope Use）：**

- **不构成投资建议。** 输出不是推荐，不针对任何特定投资者的财务状况、目标或风险承受能力，不得作为任何投资决策的依据。
- **不是交易系统。** 不得用于自动化下单、仓位决策、或任何实盘策略的信号生成。项目中的回测只用于验证分数的排序稳定性与经济意义，不是可交易策略的证据。
- **不能替代信用或审计尽职调查。** 不得作为信贷审批、授信额度、债券定价、评级、审计程序、内控评价或任何受监管的风险管理流程的输入或输出。
- **不得用于对个人做出不利决定。** 模型输出是关于发行实体的，不涉及自然人；任何将该输出用于个人征信、雇佣、保险或类似用途的做法都超出设计意图。
- **不得用于合规或法律用途。** 输出不得作为监管报送、法律意见或争议解决的材料。
- 不得在延迟敏感或高可用的生产服务中部署。
- 不得用于任何自动化的对外沟通（例如自动发送风险提示给客户）。

任何使用者都在自行承担全部责任的前提下使用本模型。使用者应自行评估其适用性、准确性，并遵守所在地的法律法规。

## Training Data

| 项 | 值 |
| --- | --- |
| 数据类型 | `{{data_provenance}}` |
| 数据来源 | `{{data_sources}}` |
| 训练数据 provenance | `{{training_data_ref}}`（run.json 的 `data` 段：训练/验证文件哈希与 SFT manifest） |
| 数据集 artifact | `{{dataset_repo}}` |
| 训练样本数 | `{{n_train_samples}}`（其中正样本 `{{n_train_positives}}`） |
| 验证样本数 | `{{n_valid_samples}}`（其中正样本 `{{n_valid_positives}}`） |
| 样本构造 | 每个 `(ticker, as_of)` 对应一条样本；三个标签各自展开 |
| 时间切分 | 见 [05 评测]({{repo_url}}/blob/main/docs/05-evaluation.md)，含 purged walk-forward 与 embargo |
| 数据版本 | `{{data_version}}` |

**关于合成数据**：若本版本在合成数据上训练，必须以 `trained_on=synthetic` 标注。合成数据由确定性生成器构造，其"文本含增量信号"的结论是**生成器假设进去的**，不是从真实世界观察到的。合成数据上的任何指标只能证明流水线可运行，不能证明模型具有真实预测能力。不得将其任何数字用于对外说明模型效果。

**数据净化**：所有输入文本严格限制为 `as_of` 之前已公开的内容。`as_of` 之后的重述、执法行动、以及回顾性新闻均被排除。详见数据集卡。

## Training Procedure

- **目标**：指令微调。输入如上，监督信号为证据化 JSON（含标签、严重度、分数、理由与逐字引文）。
- **目标构造**：引文从已抓取的源文本中直接截取，因此逐字约束在训练数据上天然成立。文本未被改写。
- **训练配置**：见 [04 训练]({{repo_url}}/blob/main/docs/04-training.md) 的 QLoRA 配置表；本次运行的完整快照在 `{{config_path}}`。
- **QloRA 细节**：4-bit nf4、double quant、`bnb_4bit_compute_dtype=bfloat16`、`optim=paged_adamw_8bit`、`gradient_checkpointing=true`。
- **超参数**：learning rate `{{lr}}`、cosine schedule、warmup ratio `{{warmup}}`、epochs `{{epochs}}`、per-device batch `{{bs}}`、gradient accumulation `{{grad_accum}}`。
- **长度处理**：超长样本按"先丢最早新闻、再丢低优先级段落、最后截断结构化摘要"的顺序处理。
- **随机种子**：`{{seed}}`；训练次数 `{{n_seeds}}`。
- **硬件**：`{{hardware}}`（例如 1x NVIDIA RTX 5090 32 GB, Windows 11）。
- **软件**：torch `{{torch_version}}`、CUDA `{{cuda_version}}`、transformers `{{transformers_version}}`、peft `{{peft_version}}`、trl `{{trl_version}}`、bitsandbytes `{{bnb_version}}`。

## Evaluation

**本模型的真实数据评测结果：`{{real_data_eval_status}}`**（若尚未进行，写 `no real-data evaluation has been performed`）。

评测框架（指标定义、切分设计、验收门槛、证伪条件）见 [05 评测]({{repo_url}}/blob/main/docs/05-evaluation.md)。任何数字都必须与切分定义、置信区间、正样本数一起呈现；**不得单独引用任何一个指标值**。

| 指标 | 目标门槛 | 实测值 | 95% 区间 | 备注 |
| --- | --- | --- | --- | --- |
| AUC-ROC | > 0.75（gate） | `{{auc}}` | `{{auc_ci}}` | `{{label}}` |
| KS | > 0.3（gate） | `{{ks}}` | `{{ks_ci}}` | — |
| PR-AUC | 高于 structured-only | `{{pr_auc}}` | `{{pr_auc_ci}}` | 核心判据 |
| IC（Spearman，对 `fwd_realized_vol_21d`） | > 0.05 | `{{ic}}` | `{{ic_ci}}` | 逐日横截面 |
| ICIR | > 0.5 | `{{icir}}` | — | — |
| ECE | < 0.05 | `{{ece}}` | — | 未降采样的 test |
| Brier | 优于常数基率基线 | `{{brier}}` | — | — |

对比路径（缺任一即视为结果不完整）：

| 路径 | PR-AUC | AUC | KS | 说明 |
| --- | --- | --- | --- | --- |
| `random` | `{{...}}` | 0.5（理论） | 0 | 下界 |
| `structured_only` | `{{...}}` | `{{...}}` | `{{...}}` | 融合判据的对照 |
| `text_only_tfidf` | `{{...}}` | `{{...}}` | `{{...}}` | 非 LLM 文本基线 |
| `text_only_zero_shot` | `{{...}}` | `{{...}}` | `{{...}}` | 未微调基线 |
| `text_only_lora`（本模型） | `{{...}}` | `{{...}}` | `{{...}}` | — |
| `fused_stacker` | `{{...}}` | `{{...}}` | `{{...}}` | 主结果 |
| 打乱文本（placebo） | `{{...}}` | `{{...}}` | `{{...}}` | 必须显著下降 |

其他必报项：

- 测试区间：`{{test_window}}`；该区间的市场 regime 描述（如 VIX 均值、指数最大回撤）：`{{regime_note}}`。
- 校准器：`{{calibrator_type}}`，在 `{{calibration_fold}}` 上拟合，样本数 `{{n_calib}}`，正样本数 `{{n_calib_pos}}`。
- walk-forward 折数 `{{n_folds}}`，各折指标的分布与置信区间：`{{fold_table}}`。
- 压力测试：2008 `{{stress_2008}}`（若数据不可用，写 `not available: data span does not cover 2008`）、2020 `{{stress_2020}}`、2022 `{{stress_2022}}`。
- 正样本数若 < 20，必须标注为"证据不足"，不得报告为通过验收。

## Limitations and Risks

**关于本模型自身的严重限制：**

1. **合成数据只验证流水线。** 若本版本在合成数据上训练，则它证明的是数据生成、as-of 检查、特征构造、训练、切分、指标计算与报告渲染这条链路可以跑通。它**不证明**模型有任何真实预测能力。合成数据中的"文本有信号"是生成器人为植入的，是构造保真的检查，不是发现。
2. **真实数据结果可能根本不存在。** 若 Evaluation 段为 `no real-data evaluation has been performed`，则本模型没有任何关于真实世界的性能证据。不要在缺少这一行的情况下引用本模型为"有效"。
3. **稀有事件的结构性困难。** 三个标签的基率都是个位数百分比。在这个基率下，accuracy 完全不可用（全部预测为 0 就能得到 95% 以上的 accuracy），且小样本下的 AUC/PR-AUC 置信区间极宽。任何数字都必须带区间与正样本数。
4. **`fraud_risk` 的证据最弱。** 730 天的 horizon 与标签可观测性截断共同导致它的有效验证与测试窗口最短，正样本最少。其结论应视为探索性的。
5. **校准依赖拟合区间。** `score` 只有在配置指定的校准器于验证折上拟合之后才具备概率含义。未校准的原始输出不得当作概率使用。若输出 JSON 中 `calibrated` 为 `false`，必须按分数排序处理，不得解释为概率。
6. **引文可能不忠实。** 模型被训练为逐字引用，但生成模型无法保证一定遵守。使用者**必须**自行对每条 `evidence[].quote` 做子串校验，丢弃无法在源文档中定位的条目。
7. **不稳定的输出格式。** 指令微调后的模型仍可能输出非法 JSON、字段缺失或 `horizon_days` 与请求不一致。生产式使用需要外部 schema 校验与重试。
8. **提示敏感性。** 对 prompt 措辞、段落顺序、`<STRUCTURED_SIGNALS>` 的呈现格式敏感。项目要求把敏感性作为一项鲁棒性测试报告；本卡片未声称该测试已通过。
9. **注意力不是证据。** 本模型的注意力分布与因果贡献之间没有确定的对应关系。可解释性依赖输出中的显式引文，而不是注意力权重。
10. **语言与地域限制。** 训练数据为英文的美国上市公司披露文本。非英文披露、非美上市发行人不在适用范围内。
11. **数据源限制。** `default_risk` 的原始定义依赖评级机构历史，该数据在 POC 中不可得时须以降级代理事件代替，这会改变标签语义。具体降级方式见数据集卡；使用者必须知晓这一差异。
12. **市场 regime 依赖。** 模型在特定市场环境下训练，可能把当时的披露惯例、会计政策与市场结构当作一般规律。极端 regime（系统性冲击、流动性枯竭）下的行为未被充分验证。
13. **已知失效边界。** 无历史先例的黑天鹅事件、内部操作风险（未披露时不在数据中）、主权/政治风险、流动性黑洞下历史价量关系失效——这些是任务本身的性质，不是可修复的缺陷。

## Environmental Impact

| 项 | 值 |
| --- | --- |
| 硬件 | `{{hardware}}` |
| 训练时长 | `{{train_hours}}` GPU 小时 |
| 硬件类型 / TDP | `{{gpu_tdp}}` W |
| 估算能耗 | `{{energy_kwh}}` kWh（= GPU 小时 x TDP x PUE 假设 `{{pue}}`） |
| 估算碳排放 | `{{co2_kg}}` kg CO2eq（依据 `{{grid_intensity}}` gCO2/kWh 与 `{{region}}` 电网） |
| 云服务商 | `{{cloud_provider_or_local}}` |
| 计算碳抵消 | `{{offsets}}`（无则写 `none`） |
| 备注 | QLoRA 4-bit 相比全参微调显著降低显存与能耗；排放估算基于以上假设，非实测 |

## Citation

```bibtex
@misc{shingan_{{artifact_slug}},
  title        = {Shingan: a dual-track financial risk model (text track: {{model_name}})},
  author       = {Shane},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/{{model_repo}}}},
  note         = {Research and education only. Not investment advice.}
}
```

项目仓库：`{{repo_url}}`
数据集：`{{dataset_repo}}`
评测基准：`{{bench_repo}}`
存档 DOI：`{{zenodo_doi}}`

## 联系与反馈

问题、复现困难或发现输出问题，请通过项目仓库的 issue 提交：`{{repo_url}}/issues`。如果发现模型产出了不忠实或有害的输出，请一并提供触发它的 prompt 与输入文档（可脱敏）。
