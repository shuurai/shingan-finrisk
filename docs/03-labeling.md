# 03 标注

风险预测没有现成标签，标签必须自己构造。本文定义完整风险分类学、三个 in-scope 标签的精确判定规则、JSONL 记录 schema、正负样本的构造方式、基率现实，以及每条标签事实的来源溯源规则。

判定规则的唯一权威是本文。

## 1. 风险分类学

| 标签 | 范围 | horizon | 判定核心 | 主要数据源 |
| --- | --- | --- | --- | --- |
| `default_risk` | **in scope** | 365 日历日 | 评级下调 >= 2 档，或进入破产/违约程序 | 评级历史（待接入）、破产/违约记录、8-K |
| `fraud_risk` | **in scope** | 730 日历日 | 财务重述、SEC 执法行动、或非标审计意见 | EDGAR 8-K Item 4.02、SEC enforcement、10-K 审计报告段 |
| `tail_risk` | **in scope** | 30 交易日 | 累计回撤低于 -30% | 复权价格面板 |
| `liquidity_risk` | 超出 POC 范围 | — | 买卖价差扩大、成交量骤降、融资流动性枯竭 | 价量面板（订单簿深度不可得） |
| `event_driven_risk` | 超出 POC 范围 | — | 并购失败、监管否决、诉讼败诉、高管离职等事件后的异常波动 | 8-K、新闻 |
| `macro_contagion_risk` | 超出 POC 范围 | — | 个股对行业/宏观冲击的敏感度 | 宏观因子、行业 ETF、新闻 |
| ~~未来涨跌方向~~ | **禁止** | — | — | — |

horizon 的差异是设计本身的一部分：`tail_risk` 是 30 个交易日的市场风险，`default_risk` 是 1 年的信用风险，`fraud_risk` 是 2 年的披露风险。三者量级相差最大 17 倍，因此**不能**被合并成单一分类任务，也不能共用一个固定的评估窗口。

### 1.1 为什么三个标签同时存在

三个标签捕捉不同机制，且都能被本次设计的输入覆盖：

- `default_risk`：信用恶化。财务比率（`interest_coverage`、`debt_short_term_ratio`、`debt_to_equity`）是它的天然结构化信号；文本侧则体现在 MD&A 中的流动性措辞与评级展望表述。
- `fraud_risk`：披露质量恶化。`accruals_ratio`、`goodwill_to_assets` 是数值侧的代理；文本侧的风险因子章节篇幅变化、模糊语义密度、重述措辞是主要来源。这是**文本轨最可能体现增量价值**的标签——舞弊的早期痕迹通常以措辞形式出现，而不是数字形式。
- `tail_risk`：市场风险。技术/微观结构特征是主力（波动率、偏度、峰度、非流动性）；新闻的恐慌情绪与财报中的宏观不确定性提及是文本侧的补充。

### 1.2 超出 POC 范围的三类，为什么不删掉

`liquidity_risk`、`event_driven_risk`、`macro_contagion_risk` 三类的定义已记录在本节，但**不在 POC 中实现**，原因分别是：

- `liquidity_risk` 需要订单簿深度或有效价差数据，这些数据在 POC 的免费来源中不可得。仅有成交量与 Amihud 指标会得到一个与 `tail_risk` 高度重合的标签，没有独立价值。
- `event_driven_risk` 的事件类型（并购失败、监管否决、诉讼结果）需要逐事件人工或规则化的结果判定，成本高且事件定义边界模糊。
- `macro_contagion_risk` 本质上是因子暴露估计，不是公司级风险预测；它与前两个标签的评测口径（横截面排序）不同，混在一起会让验收门槛失去意义。

保留定义的意义在于：数据字典与 schema 预留了扩展位，后续实现不需要重构标签层。

### 1.3 明确禁止的标签：未来涨跌

不要用"未来涨跌"当风险标签：那是 alpha，不是 risk，且信噪比极低。本项目把它固化为禁止项，理由如下：

1. **语义错误**：风险预测的目标是识别下行损失的暴露，不是预测价格方向。价格上涨本身不降低已发生的风险暴露。
2. **信噪比**：日频收益的可预测部分极低，方向准确率长期在 50% 附近。用它做监督信号会让模型拟合噪声，并且这种噪声在样本外不可复现。
3. **不可校准**：涨跌方向是二值事件，没有内在概率语义。风控需要的"未来一年违约概率是 8%"这样的量，无法从方向标签中产生。
4. **horizon 语义塌陷**：涨跌标签必须绑定一个具体持有期，而风险标签绑定的是事件窗口，两者的时间语义不同。

因此：任何形如 `label_future_up`、`label_future_return_sign`、`target_direction` 的列都不得出现在标签表中。数据字典中的 `fwd_ret_21d` 与 `fwd_realized_vol_21d` 是**连续的前瞻量**，只用于 IC 计算与回测（见[评测](05-evaluation.md)），也不是标签。

## 2. 三个 in-scope 标签的判定规则

### 2.1 `default_risk`（horizon = 365 日历日）

**触发条件**：在 `[as_of + 1 天, as_of + 365 天]` 窗口内，满足以下任一项：

1. **评级下调 >= 2 档**：任一被纳入来源的评级机构对该发行人（或其在来源中可唯一对应的主体）的主体信用评级发生下调，且**累计下调档数 >= 2**。档数按该机构自己的评级刻度计数（例如从 A- 到 BBB- 是 3 档）。
2. **破产 / 违约程序**：进入破产保护（如 Chapter 11）、破产清算、或发生债务违约（未按约定偿付本息、债务置换导致实质减债、cross-default 触发）。

判定细节：

- **多家机构同时下调不叠加**：以单一机构的最大累计下调档数为准，不跨机构求和，避免同一事件被重复计数。
- **窗口内多次下调累计**：若 365 天内发生两次各下调 1 档，累计为 2，触发。
- **下调后上调**：若窗口内先下调 2 档再上调 1 档，判定以窗口内累计下调档数为准（本例触发）。理由：风险已实现。
- **`as_of` 当日的事件不算未来事件**：窗口从 `as_of + 1` 天开始。
- **窗口右端截断**：若 `as_of + 365 天` 超过数据可用区间的末端，该行的 `label_mask_default_risk` 置 `false`，不参与训练与评估。这是 censoring，不是负样本。

### 2.2 `fraud_risk`（horizon = 730 日历日）

**触发条件**：在 `[as_of + 1 天, as_of + 730 天]` 窗口内，满足以下任一项：

1. **财务重述**：公司披露对已发布财务报表的重述（含 8-K Item 4.02 的"非依赖先前发布的财报"披露），或对前期报表做追溯调整且性质为纠错（而非会计政策变更的追溯应用）。
2. **SEC 执法行动**：SEC 对该公司或其高管因财务披露相关事由提起的执法行动（administrative proceeding、civil action、或和解令）。
3. **非标审计意见**：审计师对该公司财务报表出具非无保留意见（qualified、adverse、disclaimer of opinion），或在对持续经营能力存在重大疑虑的报告中出具 going concern 段落。

判定细节：

- **会计政策变更的追溯应用不算重述**：例如存货计价方法变更的追溯调整属于合规行为，不触发。判定依据是披露文本中是否明确说明"先前发布的财务报表不应再被依赖"或等义表述。
- **仅高管个人被执法、且与公司财务披露无关**（如个人内幕交易）**不触发**。这条边界必须在标注记录中留痕，便于事后复核。
- **going concern 段落的判定**依赖 10-K 审计报告段的文本抽取，属于未实现的抽取任务（见[数据](02-data.md)第 2.4 节的缺口表）。
- 730 天的 horizon 显著长于 POC 的样例年份区间，因此 Stage 2 的正样本数会非常少。这一点在验收门槛设计中被显式考虑（见[评测](05-evaluation.md)）。

### 2.3 `tail_risk`（horizon = 30 交易日）

**触发条件**：从 `as_of` 之后第一个交易日开始的 30 个交易日内，**累计回撤**低于 -30%。

定义：

```
peak_t   = max(close_s) for s in [t0, t]      # t0 为 as_of 后的第一个交易日
trough_t = close_t
drawdown_t = trough_t / peak_t - 1
tail_risk = 1  if  min(drawdown_t for t in [t0, t0+29]) < -0.30
```

- 使用**回撤**（相对窗口内峰值）而不是"30 日累计跌幅"（相对起点）。两者在单边下跌时接近，但在先涨后跌的路径上差异显著：一只股票先涨 20% 再跌 25%，起点到终点跌幅为 -10%，但回撤为 -37.5%。风险语境下关注的是从高点的损失，因此取回撤。
- 严格不等号：`< -0.30` 触发，恰好 -30.00% 不触发。
- 使用**复权后**收盘价。未复权价格会在除权日产生虚假跳变。
- 30 个交易日不足时（接近数据末端），`label_mask_tail_risk` 置 `false`。

**口径说明**：这一口径有两种常见替代写法，都不采用。"未来 30 个交易日累计跌幅 > 30%"用起点口径而非回撤口径，缺陷见上一条；"未来 N 日（如 20 日）跌幅超过阈值（如 -15%）"的 horizon 与阈值均与本文三个标签的设计不一致。本文的唯一口径是 **30 交易日 / -30% 回撤**。

### 2.4 三个标签的关系

三个标签**可以同时为 1**。一家公司可能在 30 日内大幅回撤、并在 365 日内被下调评级、并在 730 日内发生重述。标签之间不是互斥关系，也不是层级关系：

| 情形 | default | fraud | tail |
| --- | --- | --- | --- |
| 财务造假曝光导致股价崩塌并最终破产 | 1 | 1 | 1 |
| 仅市场系统性下跌导致个股回撤 | 0 | 0 | 1 |
| 仅被下调 1 档评级，无其他事件 | 0 | 0 | 0 |
| 重述但不破产、股价未大跌 | 0 | 1 | 0 |

因此实现上是**三个独立的二分类头**，不是 softmax 多分类。`data/schema.py` 对三个标签列分别做取值范围校验，不施加"和为 1"的约束。

## 3. JSONL 记录 schema

项目产出两类 JSONL，不要混淆：

| 文件 | 生产者 | 消费者 | 内容 |
| --- | --- | --- | --- |
| 标签数据集 | `shingan data build` | 评测、分析、HF 数据集 | 一行 = 一个 `(ticker, as_of)`，含标签与元数据，**不含**原始文本 |
| SFT 指令集 | `shingan data sft` | `shingan train lora` | 一行 = 一条训练样本，含渲染好的 prompt 与目标输出 |

### 3.1 标签数据集记录

```json
{
  "id": "JPM-2020-03-02",
  "ticker": "JPM",
  "cik": "0000019617",
  "as_of": "2020-03-02",
  "split": "train",
  "is_synthetic": false,
  "labels": {
    "default_risk": 1,
    "fraud_risk": 0,
    "tail_risk": 1
  },
  "label_masks": {
    "default_risk": true,
    "fraud_risk": false,
    "tail_risk": true
  },
  "horizon_days": {
    "default_risk": 365,
    "fraud_risk": 730,
    "tail_risk": 30
  },
  "event_date": {
    "default_risk": "2020-06-11",
    "fraud_risk": null,
    "tail_risk": "2020-03-23"
  },
  "source_of_record": {
    "default_risk": "rating_history:SP",
    "fraud_risk": null,
    "tail_risk": "price_panel:adjusted_close"
  },
  "forward": {
    "fwd_ret_21d": -0.184,
    "fwd_realized_vol_21d": 0.71,
    "fwd_max_drawdown_30d": -0.352
  },
  "sample_weight": 1.0,
  "data_version": "labels-0.1.0"
}
```

字段说明：

| 字段 | 约束 |
| --- | --- |
| `id` | `{ticker}-{as_of}`，全表唯一 |
| `labels` | 三个 int，取值 0/1；`label_masks` 为 `false` 时该标签的值无意义，必须忽略 |
| `label_masks` | censoring 标记。右端截断（窗口超出数据末端）或缺少事件源覆盖时为 `false` |
| `event_date` | 触发该标签的**首个**事件日期；为 0 时该项为 `null`。**不得**进入特征矩阵 |
| `source_of_record` | 该标签事实的来源标识，见第 6 节 |
| `forward` | 连续前瞻量，供 IC 与回测使用；**不得**进入特征矩阵 |
| `sample_weight` | 降采样后用于恢复原始基率的权重，见第 5 节 |

`fwd_*` 与 `event_date`、`labels` 都带前缀保护，`leakage.py` 在构建模型输入矩阵时按前缀黑名单剔除。

### 3.2 SFT 指令集记录

```json
{
  "id": "JPM-2020-03-02-sft",
  "source_id": "JPM-2020-03-02",
  "label": "default_risk",
  "horizon_days": 365,
  "messages": [
    {
      "role": "system",
      "content": "You are a financial risk analyst. Output a single JSON object with keys: label, severity, score, horizon_days, reasons, evidence. Cite only spans that appear verbatim in the provided documents."
    },
    {
      "role": "user",
      "content": "<AS_OF>2020-03-02</AS_OF>\n<STRUCTURED_SIGNALS>...</STRUCTURED_SIGNALS>\n<FILING_EXCERPTS>...</FILING_EXCERPTS>\n<NEWS>...</NEWS>\nAssess default_risk over the next 365 days."
    },
    {
      "role": "assistant",
      "content": "{\"label\":\"default_risk\",\"severity\":\"high\",\"score\":0.37,\"horizon_days\":365,\"reasons\":[\"...\"],\"evidence\":[{\"source_type\":\"10-Q\",\"source_ref\":\"CIK 0000019617, 10-Q, Item 1A\",\"quote\":\"...\"}]}"
    }
  ],
  "meta": {
    "n_filing_tokens": 2841,
    "n_news_items": 7,
    "n_structured_signals": 42,
    "truncated": false
  }
}
```

约束：

- 每条样本只训练**一个**标签（`label` 字段），因为 horizon 不同，prompt 中的时间表述必须与 horizon 一致。三个标签按 3 倍数据量展开。
- `assistant` 的 `content` 必须是可解析的 JSON 单对象。`score` 是**训练目标的一部分**，但要注意：合成标签只在 0/1 上可靠，`score` 在真实数据上应由事件离 `as_of` 的远近与严重度派生，派生规则必须记录在数据集卡的 `labeling` 段落中，且**不得**使用 `as_of` 之后才知道的信息来构造其数值大小。
- `evidence[].quote` 在渲染时从已抓取的源文本中截取，因此天然满足"逐字子串"约束。构造样本时不做改写。
- `truncated` 标记序列长度截断是否发生。截断策略与顺序见[训练](04-training.md)。
- prompt 中出现的日期字符串必须全部 `<= as_of`。`prompts.py` 在渲染后做一次扫描，命中未来日期即报错。

## 4. 正负样本示例（示意）

以下示例为**构造的示意样本**，用于说明判定流程，不对应任何具体的真实事件。所有可核对的真实事件必须来自第 6 节的来源。

### 4.1 `default_risk` 正例（示意）

```
ticker: ACME
as_of:  2020-03-02
窗口:   (2020-03-03, 2021-03-01]
事件:   2020-06-11 评级由 A- 下调至 BBB-（3 档）
        2020-09-30 披露 2020 年债券利息未按期支付
判定:   下调档数 3 >= 2 → default_risk = 1
        source_of_record = rating_history:<agency>
        event_date = 2020-06-11（首个触发事件）
sample_weight: 8.3（降采样后的重加权系数）
```

### 4.2 `default_risk` 负例（示意）

```
ticker: ACME
as_of:  2021-03-02
窗口:   (2021-03-03, 2022-03-01]
事件:   窗口内无评级下调，无破产/违约记录
判定:   default_risk = 0
label_mask: true（窗口完整，未被右端截断）
```

### 4.3 `fraud_risk` 正例（示意）

```
ticker: BOREAL
as_of:  2018-08-15
窗口:   (2018-08-16, 2020-08-14]
事件:   2019-02-20 提交 8-K Item 4.02，声明先前发布的季度财报不应再被依赖
判定:   财务重述 → fraud_risk = 1
        source_of_record = sec_item_4.02_8k
event_date: 2019-02-20
```

### 4.4 `fraud_risk` 负例（示意，边界情形）

```
ticker: BOREAL
as_of:  2017-08-15
事件:   2017-11-02 会计政策变更（存货计价方法），做追溯调整
判定:   fraud_risk = 0 —— 会计政策变更的追溯应用不属于纠错型重述
标注备注: 必须写明"policy change, not error correction"，便于事后复核
```

### 4.5 `tail_risk` 正例（示意）

```
ticker: CENTAUR
as_of:  2020-03-02
窗口:   30 个交易日，2020-03-03 起
价格路径: 窗口内峰值 100（3 月 4 日），随后最低收盘 66（3 月 23 日）
drawdown: 66 / 100 - 1 = -0.34 < -0.30 → tail_risk = 1
source_of_record = price_panel:adjusted_close
mark:   注意起点到终点的跌幅是 -34%，与回撤在此例中巧合地接近；
        路径不同时两者会显著分离，因此实现必须用回撤口径。
```

### 4.6 `tail_risk` 负例（示意，先涨后跌）

```
ticker: DUNLIN
as_of:  2021-06-01
窗口:   30 个交易日
价格路径: 起点 100，第 10 日 120，第 30 日 90
起点跌幅: 90 / 100 - 1 = -0.10（未触发 -30% 的"跌幅"口径）
回撤:     90 / 120 - 1 = -0.25（仍未触发 -30%）
判定:     tail_risk = 0
```

### 4.7 三标签共存的复合例（示意）

```
ticker: EGRET
as_of:  2019-10-01
default_risk = 1  窗口内评级下调 2 档
fraud_risk   = 1  窗口内提交 8-K Item 4.02
tail_risk    = 1  窗口内 30 日回撤 -41%
三者互不冲突，证明标签是三个独立头而非互斥分类。
```

## 5. 基率现实与负样本构造

### 5.1 基率

三个标签都是稀有事件，预期正样本率为**个位数百分比**，且 `fraud_risk` 大概率处于最低端（730 天 horizon 与 US-listed 大市值公司的实际重述率共同决定）。

具体数值必须在真实数据到位后**实测并写入数据集卡**，不得预先声明。当前可以确定的只有量级判断：

| 标签 | 预期量级 | 决定因素 |
| --- | --- | --- |
| `default_risk` | 低个位数百分比 | S&P 500 大市值公司的年度降级 >= 2 档概率 |
| `fraud_risk` | 更低 | 730 天窗口内重述/执法/非标意见的联合概率，大市值公司尤其低 |
| `tail_risk` | 相对最高，仍为个位数到低两位数 | 30 日 -30% 回撤在大市值股上的频率，随市场 regime 波动剧烈 |

这个现实直接决定评估口径：**accuracy 完全不可用**（全预测为 0 就能得到 95% 以上的 accuracy），因此 headline 指标是 PR-AUC、KS 与校准误差，而不是 accuracy。见[评测](05-evaluation.md)。

### 5.2 负样本如何采样

朴素做法是"所有未触发事件的行都是负样本"，但这会产生两个问题：

1. **正负比极端失衡**（可能 1:99 或更差），树模型与校准器都会被未被解释的负样本主导。
2. **标签噪声**：一个"未触发"的行可能只是事件发生在 horizon 之外，或者事件源没有覆盖到，这类行与"真正安全"的行是不同性质的。

本项目的处理规则：

| 规则 | 内容 |
| --- | --- |
| 保留全部可用正样本 | 不做正样本下采样。稀有事件的每一个样本都有信息量。 |
| 负样本下采样 | 按标签分别做。默认负正比目标 `neg_pos_ratio: 20`，即每 1 个正样本保留 20 个负样本。取值放在 `configs/data/*.yaml` 中可调。 |
| 分层采样 | 负样本按 `(year, sector)` 分层随机采样，保证时间分布与行业分布不被压缩。**不按标签分层**（那会泄漏）。 |
| 只使用 `label_mask = true` 的行 | 被 censoring 的行既不是正样本也不是负样本，直接从训练集中排除。 |
| 权重回填 | 下采样后的样本带 `sample_weight = N_neg_total / N_neg_kept`，在训练与评估中生效，使校准概率仍然对应真实基率。**评估必须在未下采样的完整负样本集上进行**，否则 PR-AUC 与校准都会被扭曲。 |
| 评估集不下采样 | valid 与 test 保持原始基率。这是校准误差（ECE）有意义的前提。 |
| 固定种子 | 采样用 `seed.py` 的统一种子，采样结果按 `data_version` 可复现。 |

这套规则的关键点是**训练时降采样、评估时不降采样、用权重对齐两者**。任何在评估阶段使用降采样数据的报告都视为无效。

### 5.3 不要用"未来收益符号"当负样本判据

一个常见的错误捷径是"未来收益为正 = 负样本"，以扩增负样本。这违反第 1.3 节的禁止项：它把 alpha 标签引入了 risk 模型，且会让标签与 `fwd_ret_21d` 高度耦合，使 IC 与回测结果失去意义（等于用同一个量同时做标签和评估目标）。

## 6. 标注溯源规则

每条标签事实必须能追溯到**一个指定的来源记录**。这条规则的目的是：当模型判错时，能区分"信息不存在"与"模型没读懂"；当评测被质疑时，能逐条复核。

`source_of_record` 的取值与优先级：

| 标签 | 事实 | 来源 | `source_of_record` 取值 |
| --- | --- | --- | --- |
| `default_risk` | 评级下调档数 | 评级机构历史记录 | `rating_history:<agency>` |
| `default_risk` | 破产 / 违约 | 破产法院记录、公司 8-K、交易所公告 | `bankruptcy_record` / `sec_8k_item_1.03` / `exchange_notice` |
| `fraud_risk` | 财务重述 | EDGAR 8-K Item 4.02，或 10-K/10-Q 的重述披露 | `sec_item_4.02_8k` / `filing_restatement_note` |
| `fraud_risk` | SEC 执法行动 | SEC enforcement 列表 | `sec_enforcement` |
| `fraud_risk` | 非标审计意见 | 10-K 审计报告段文本抽取 | `audit_opinion_extracted` |
| `tail_risk` | 30 日回撤 | 复权价格面板 | `price_panel:adjusted_close` |

规则：

1. **一个标签一个来源**：同一标签的每次判定只引用一个 `source_of_record`。多源冲突时按上表从上到下取优先级最高的，并在标注记录中留 `conflict_note`。
2. **来源必须在数据侧真实存在**：`source_of_record` 引用的记录必须能在 `data/interim/` 中找到对应的原始对象。引用一个不存在的来源视为标注错误。
3. **不可用商业数据时显式降级**：`rating_history` 目前在 POC 中不可得。降级方案是用**公开可核对的事件**（8-K Item 2.04 触发事件、交易所退市通知）作为 `default_risk` 的代理，并在 `source_of_record` 中写出代理标识（如 `proxy:sec_8k_item_2.04`），同时在数据集卡的 known limitations 中声明代理与原定义的差异。**不得**把代理事件悄悄当成评级下调。
4. **标注集本身要有人工抽查**：至少抽取 50 条（或全部正样本，若少于 50 条）做人工复核，复核结果与不一致率写入数据集卡。未见复核的标注集不得用于 headline 结果。
5. **来源版本化**：`data_version` 变更时，所有 `source_of_record` 需重新校验一遍，日志保留。

## 7. 相关文档

- 事件在流水线中如何被 as-of 化：[02 数据](02-data.md)
- 标签如何进入两条轨的训练：[04 训练](04-training.md)
- 稀有事件下的指标选择、IC 的连续前瞻量、切分与 embargo：[05 评测](05-evaluation.md)
- 发布标注集时的溯源与偏差声明：[数据集卡模板](../templates/dataset_card.md)
