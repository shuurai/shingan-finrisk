---
license: apache-2.0
language:
  - en
task_categories:
  - text-classification
  - tabular-classification
tags:
  - financial-risk
  - credit-risk
  - fraud-detection
  - tail-risk
  - sec-edgar
  - fnspid
  - risk-labels
  - as-of
pretty_name: Shingan FinRisk Labels
size_categories:
  - {{size_category}}
---

<!--
模板使用方式：
  1. 替换所有 {{...}} 占位符为实测值；未实测的字段写 "not measured" 并说明原因。
  2. 基率、样本数、正样本数必须是实测值，不得估算后当作实测写入。
  3. 若数据含合成样本，必须保留 Synthetic Data 一节并标注比例。
  4. 发布前删除本注释块与所有未替换的占位符。
-->

# Shingan FinRisk Labels

Shingan（心眼）项目的**标注数据集**：上市公司在具体 `as_of` 时点的风险标签与派生特征，用于训练与评测一个双轨（结构化 GBDT + 文本 QLoRA）风险模型。

**本数据集不包含原始新闻正文，也不包含原始价格面板。** 文本轨训练所需的原文由使用者自行从原始来源获取（见 Licensing）。这是刻意的设计，用于遵守各源的再分发条款。

## Dataset Description

| 项 | 值 |
| --- | --- |
| 项目 | Shingan（心眼） |
| 数据版本 | `{{data_version}}` |
| 主键 | `(ticker, as_of)` |
| 行数 | `{{n_rows}}` |
| 公司数 | `{{n_companies}}` |
| 时间跨度 | `{{date_range}}` |
| 标签数 | 3（`default_risk` / `fraud_risk` / `tail_risk`） |
| 含合成样本 | `{{contains_synthetic}}`（比例 `{{synthetic_share}}`） |
| License | Apache-2.0（仅覆盖本项目生成的标签与派生特征；见 Licensing） |
| 项目文档 | `{{repo_url}}/tree/main/docs` |

### 标签

| 标签 | 定义 | horizon | 正样本数 | 正样本率 |
| --- | --- | --- | --- | --- |
| `default_risk` | 评级下调 >= 2 档，或进入破产/违约程序 | 365 日历日 | `{{pos_default}}` | `{{rate_default}}` |
| `fraud_risk` | 财务重述、SEC 执法行动、或非标审计意见 | 730 日历日 | `{{pos_fraud}}` | `{{rate_fraud}}` |
| `tail_risk` | 30 个交易日内累计回撤低于 -30% | 30 交易日 | `{{pos_tail}}` | `{{rate_tail}}` |

**基率是本数据集最重要的属性。** 三个标签都是稀有事件，正样本率在个位数百分比量级（`fraud_risk` 通常最低）。这直接决定了评估口径：accuracy 不可用，headline 指标必须是 PR-AUC、KS 与校准误差。使用本数据集时若报告 accuracy，等于报告了一个无意义的数字。

三个标签**互不排斥**，可以同时为 1。它们是三个独立的二分类目标，不是 softmax 多分类。

`liquidity_risk`、`event_driven_risk`、`macro_contagion_risk` 在项目文档中有定义，但**不包含在本数据集中**。

### 字段

| 字段组 | 字段 | 说明 |
| --- | --- | --- |
| 标识 | `id`, `ticker`, `cik`, `company_name`, `sector` | `id` 为 `{ticker}-{as_of}`，全表唯一 |
| 时间 | `as_of` | 该行信息的截止交易日。所有特征在该时点可用 |
| 切分 | `split` | `train` / `valid` / `test` / `purged`。`purged` 行为被 purge/embargo 剔除的样本，保留以便审计 |
| 标签 | `labels.{default_risk,fraud_risk,tail_risk}` | 0/1 |
| 标签掩码 | `label_masks.*` | `false` 表示该行在该标签上不可判定（窗口被右端截断，或事件源未覆盖）。掩码为 `false` 时标签值无意义 |
| 标签元数据 | `horizon_days.*`, `event_date.*`, `source_of_record.*` | `event_date` 与 `source_of_record` 仅供审计，**不得**作为模型特征 |
| 前瞻量 | `fwd_ret_21d`, `fwd_realized_vol_21d`, `fwd_max_drawdown_30d` | 连续前瞻量，供 IC 与回测使用。**不得**作为模型特征 |
| 结构化特征 | 财务比率列（约 13 列） | 见项目文档的数据字典 |
| 结构化特征 | 技术/微观结构列（约 20 列） | 同上 |
| 文本计数量 | 新闻条数、情感聚合、词典命中（约 10 列） | 仅统计量，不含原文 |
| 权重 | `sample_weight` | 负样本降采样后用于恢复真实基率的权重 |
| 元数据 | `is_synthetic`, `data_version`, `n_sources` | — |

完整字段定义见 `{{repo_url}}/blob/main/docs/02-data.md` 的数据字典一节。

## 构造方式（Provenance）

| 标签 / 数据 | 来源记录 | `source_of_record` 取值 |
| --- | --- | --- |
| `default_risk` — 评级下调 | 评级机构历史记录 | `rating_history:<agency>` |
| `default_risk` — 破产/违约 | 破产记录、8-K Item 1.03、交易所公告 | `bankruptcy_record` / `sec_8k_item_1.03` / `exchange_notice` |
| `fraud_risk` — 财务重述 | 8-K Item 4.02、财报中的重述披露 | `sec_item_4.02_8k` / `filing_restatement_note` |
| `fraud_risk` — SEC 执法 | SEC enforcement 列表 | `sec_enforcement` |
| `fraud_risk` — 非标审计意见 | 10-K 审计报告段文本抽取 | `audit_opinion_extracted` |
| `tail_risk` | 复权价格面板 | `price_panel:adjusted_close` |
| 财务比率 | SEC EDGAR inline XBRL | `edgar_xbrl` |
| 技术/微观结构 | 价量面板 | `price_panel` |
| 文本计数量 | FNSPID 新闻 + EDGAR 全文 | `fnspid` / `edgar_fulltext` |

**代理事件声明**：`default_risk` 的原始定义依赖评级机构历史，该项需要商业数据许可。在本次构建中，评级历史的状态是：`{{rating_history_status}}`。

- 若为 `available`，标签按原定义标注。
- 若为 `proxy`，则使用公开可核对的替代事件（如 8-K Item 2.04 触发事件、交易所退市通知）作为代理，`source_of_record` 前缀为 `proxy:`。**代理事件与"评级下调 >= 2 档"不是同一个事实**，正样本集合会因此改变，基率与可比性都受影响。使用者必须知晓这一差异。
- 若为 `unavailable`，`label_masks.default_risk` 全为 `false`，该标签不可用。

### As-of 纪律

每一行的**全部**特征满足"在 `as_of`（含当日收盘后）已经公开"：

| 数据 | 可用性判定 |
| --- | --- |
| 10-K/10-Q/8-K 全文 | `filing_date <= as_of` |
| inline XBRL 数值事实 | `filed <= as_of`（不是 `end <= as_of`） |
| 新闻 | 发布时间戳（转 UTC 日期）`<= as_of` |
| 价格/成交量 | 交易日 `<= as_of`，只用当日收盘及之前 |
| 事件（标签来源） | `event_date > as_of`，事件本身绝不进入任何特征 |

明确排除的内容：

- `as_of` 之后的重述。t 时点的模型只能看到当时披露的数字，即使后来被证明是错的。
- `as_of` 之后的执法行动与审计意见变更。
- `as_of` 之后的新闻，**特别包括回顾性报道**（它们可能直接总结"该公司最终破产"，等于把标签写进输入）。
- 任何由 `event_date` 派生的字段（如 `days_to_event`）。
- `ticker` 不作为特征；`sector` 只做 one-hot 或原生 categorical，**不做 target encoding**（会因为折内包含未来样本而泄漏）。

关于 as-of 违约风险的一处诚实说明：as-of 纪律由构建代码与断言保证，但**未对本数据集的每一行做人工逐行核验**。使用者若发现任何一行的特征引用了 `as_of` 之后的信息，请提交 issue。

## 已知偏差（Known Biases）

| 偏差 | 描述 | 影响 | 状态 |
| --- | --- | --- | --- |
| **幸存者偏差（S&P 500 成分股）** | 若公司集合取自"当前"的 S&P 500 成分股，则已破产、被收购、被剔除的公司不会出现在样本中 | 系统性压低正样本率，并高估模型表现。这是最严重的偏差 | `{{pit_membership_status}}`。点内成分股（point-in-time membership）是必需项；若为 `not applied`，本数据集的基率与指标均不可与其他数据集比较 |
| **重述前视（vendor 数据）** | 第三方财务数据供应商常回填重述后的数值，使 t 时点的"当时数字"实际是修订值 | 直接泄漏 `fraud_risk` 的信息 | 已通过使用 `filed <= as_of` 的 XBRL 版本缓解；若使用了回填型供应商数据，状态为 `{{vendor_lookahead_status}}` |
| **仅英文** | 训练与标注文本为英文的美国上市公司披露 | 不适用于非英文披露或非美上市发行人 | 设计限制 |
| **仅美上市** | 覆盖 US-listed 发行人 | 不适用于其他司法管辖区的会计与披露制度 | 设计限制 |
| **大市值倾斜** | Stage 2 的公司集合以大市值龙头为主 | 大市值公司的风险事件率显著低于全市场，基率偏低；模型在小市值上的表现未经检验 | `{{market_cap_scope}}` |
| **时间覆盖不均** | FNSPID 的新闻密度与 EDGAR 的 XBRL 覆盖率随年份变化（早期明显更低） | 跨期比较不可直接进行；必须按年分组评估 | 按时间分组报告指标 |
| **评级数据缺口** | 见上方的代理事件声明 | 标签语义可能被改变 | `{{rating_history_status}}` |
| **基率随 regime 变化** | `tail_risk` 的基率在市场高波动期显著抬升 | 全局固定阈值口径下的基率不稳定，影响校准 | 按年报告基率 |
| **标签噪声** | 事件判定依赖文本抽取（审计意见、重述识别），存在误判 | 引入标签噪声，压低可达到的性能上限 | 人工抽查 `{{audit_sample_size}}` 条，不一致率 `{{audit_disagreement_rate}}` |
| **负样本降采样** | `train` 切分的负样本按 `neg_pos_ratio` 降采样 | 训练分布不再对应真实基率 | 用 `sample_weight` 补偿；**评估必须在未降采样的 `valid`/`test` 上进行** |

## 合成数据（Synthetic Data）

本数据集合成样本占比：`{{synthetic_share}}`。若占比较大，以下内容必须被理解：

- 合成数据由 `data/synthetic.py` 确定性生成，固定种子，可完全复现。
- 合成数据中存在一个**人为植入的**潜在结构：文本携带结构化特征无法观测到的风险分量。它的目的是让"fused 应当优于 structured-only"成为一条**构造保真的回归检查**——若融合增益在合成数据上为零或负，说明实现有 bug。
- **这不是发现。** 合成数据上的融合增益是生成器假设进去的，不能用来论证真实世界中的文本有用。
- 任何在合成数据上得出的指标（AUC、KS、PR-AUC 等）只能说明流水线连通，**不得**作为模型能力的证据，不得对外引用。
- 合成行的 `is_synthetic` 为 `true`。真实数据与合成数据**不得**混进同一份 processed 表；混用会被构建器拒绝。

## 用法

```python
from datasets import load_dataset

ds = load_dataset("{{dataset_repo}}", split="train")

# 稀有事件：必须看基率，而不是 accuracy
import numpy as np

y = np.array(ds["labels.default_risk"])
print("positive rate:", y.mean())

# 评估时只使用 label_mask 为 true 的行，且不要让降采样影响评估集
mask = np.array(ds["label_masks.default_risk"])
y = y[mask]

# 特征矩阵：剔除所有标签与前视列
BANNED = ("label_", "labels.", "fwd_", "event_", "source_of_record")
feature_cols = [c for c in ds.column_names if not c.startswith(BANNED)]
```

三条使用纪律：

1. **不要用 `fwd_*` 列做特征。** 它们是评估目标（连续前瞻量），作为特征会立刻造成泄漏。
2. **不要在评估集上使用降采样数据。** 校准误差与 PR-AUC 只有在真实基率下才有意义。
3. **时间切分不可随机化。** 面板数据的样本不独立；且标签 horizon 长达 730 天，切分边界必须做 purge 与 embargo（见项目文档 05-evaluation.md）。

## Licensing 与署名

本数据集的 Apache-2.0 许可**只覆盖本项目生成的标签与派生特征**，不覆盖任何上游数据的原始内容。

| 上游来源 | 许可 / 条款 | 本数据集的再分发状况 |
| --- | --- | --- |
| SEC EDGAR | 美国联邦政府公开信息，无版权限制 | 仅分发由公开 filing 派生的比率与统计量，不包含 filing 全文 |
| FNSPID | 由数据集作者在 Hugging Face 发布，遵循其数据集卡声明的许可 | **不再分发新闻正文**。仅包含聚合统计量（条数、情感均值/标准差、负面占比）。使用者需自行获取原文 |
| 价量数据（yfinance / Stooq） | 各有使用条款；Stooq 再分发受限 | **不再分发原始价格面板**。仅包含由价格派生的技术特征与标签 |
| 评级历史 / WRDS（若使用） | 商业订阅，本项目无再分发许可 | 不包含原始记录；仅包含派生的 0/1 标签与来源标识 |
| `Qwen/Qwen3-14B` | Apache-2.0 | 与本数据集无关（模型侧的 base model） |

**如果使用者要复现文本轨训练**，需要自行取得 FNSPID 与 EDGAR 数据，然后按项目文档的 as-of 规则重建 prompt。本数据集提供 `id`（`{ticker}-{as_of}`）作为与外部文本对齐的键。

引用要求：

```bibtex
@misc{shingan_finrisk_labels,
  title        = {Shingan FinRisk Labels: an as-of panel of listed-company risk labels},
  author       = {Shane},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/{{dataset_repo}}}},
  note         = {Research and education only. Not investment advice.}
}
```

使用本数据集时请一并引用 FNSPID（若使用了其派生特征）与 SEC EDGAR 作为数据来源。

## 个人信息说明

- 本数据集的主体是**发行实体（公司）**，不是自然人。
- 数据集**不包含**任何自然人的个人信息：不含高管姓名、薪酬、持股明细或任何可识别到个人的字段。
- 事件标签涉及的是公司层面的披露与监管行为。若某条事件的公开记录中不可避免地附带人名（例如执法行动的对象包括个人），本项目在派生特征与标签中**不保留这些人名**，只保留公司级标识与事件类型。
- 若你在使用中发现任何可识别到自然人的信息，请提交 issue，我们会将其移除。

## 维护与版本

| 项 | 值 |
| --- | --- |
| 构建脚本版本 | `{{git_commit}}` |
| 构建日期 | `{{build_date}}` |
| 上游数据快照日期 | `{{source_snapshot_date}}` |
| 已知问题清单 | `{{known_issues_url}}` |
| 变更日志 | `{{changelog_url}}` |

版本变更规则：任何字段增删、标签定义调整、时间跨度变化、或 as-of 规则变化，都必须提升 `data_version` 并在此卡片中记录。**字段含义的静默变化是不允许的。**

## 联系与反馈

数据错误、as-of 违规、偏差遗漏，请通过 `{{repo_url}}/issues` 提交。
