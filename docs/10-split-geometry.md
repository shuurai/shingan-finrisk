# 10 切分几何修订：方案对比（决策文档，2026-09-26）

**状态：等待拍板。本文档只分析与推荐，不改任何配置。** 拍板后重跑
`data build`（CPU 分钟级）即可让新切分生效；本文档的全部数字用仓库自己的
`assign_split_column` 模拟，与实现零漂移（模拟脚本见 §6）。

## 1. 问题：39 个真实正样本，评测只用了 10 个

真实面板（2221 行、34 家公司、2009-07 → 2026-09）的 `tail_risk` 正样本
按 `as_of` 年份分布（mask-true 共 39 个）：

| 年份 | 2010 | 2011 | 2014 | 2016 | 2018 | **2020** | 2022 | 2023 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 正样本 | 1 | 2 | 1 | 2 | 1 | **27** | 4 | 1 |

**2020 一年占 27 个（69%）**——疫情崩盘正是 `-30%/30 交易日` 事件的集中爆发期。
而现行切分下这 27 个全部落在 `excluded`：不是 purge 吞掉的，是 **test 名义窗口
从 2021-01-01 起，2020 年根本不在任何名义窗口里**。

现行配置（`configs/default.yaml`）名义窗口 train 2010–2016 / valid 2017–2019 /
test 2021–2024，`purge_days=730` + `embargo_days=30`（交易日）。purge 从**每块
末尾**扣除 `purge+embargo ≈ 772 天`，所以：

| 方案 | train | valid | test | purged | excluded |
| --- | --- | --- | --- | --- | --- |
| 现状 S0（行数） | 617 | 253 | 544 | 396 | 411 |
| 现状 S0（正样本） | **4** | **1** | **5** | 2 | **27** |

三重后果：

1. **test 只有 5 个正样本**，低于项目自己的门禁 `min_positives_for_metrics=20`
   ——所有真实 test 指标（AUC 0.7686 等）按本仓库的评价标准属于"样本不足、
   不可判定"，这正是它们只能标注"不构成性能结论"的结构性原因。
2. **2020 的 27 个正样本（69% 的监督信号）被浪费**——不是数据不够，是几何切错。
3. 训练集有效窗口被 730 天 purge 砍到 2014-11，2016 年的 2 个正样本掉进 `purged`。

## 2. 为什么 purge 是 730 天：一个为 fraud 设计的参数绑架了 tail

`per_label=False` 时，purge margin 必须覆盖**最长**标签 horizon——
`fraud_risk` 的 730 天。但 `tail_risk` 的 horizon 只有 30 交易日 ≈ 42 日历天，
却同样承受 730 天的块间隔离。

配置里为此预留了 `split.per_label: true`：每个标签用自己的 horizon 定 margin。
对 tail_risk：`purge_days=45`（≥42 即可）+ embargo 42 日历天 → margin **87 天**。
泄漏防护不被削弱（隔离仍 ≥ 2 倍 horizon），砍掉的只是 fraud 强加给 tail 的两年。
**这是所有方案的共同前置**；下面的方案对比全部在 `per_label=true, purge_days=45`
上进行（方案 C 证明只做这一步不够）。

## 3. 方案对比（全部用 `assign_split_column` 实测）

| | S0 现状 | **A：2020 归 test** | **B'：2020 归 valid** | C：只换 margin |
| --- | --- | --- | --- | --- |
| 名义窗口 | 2010–16 / 17–19 / 21–24 | 2010–16 / 17–19 / **2020–24** | 2010–16 / 17–**21** / **2022–24** | 不变 |
| purge | 730（全局） | 45（per_label） | 45（per_label） | 45（per_label） |
| train 正样本 | 4 | 4 | 4 | 4 |
| valid 正样本 | 1 | 1 | **28** | 1 |
| **test 正样本** | 5 | **32** | 5 | 5 |
| excluded 正样本 | 27 | **0** | 27 | 27 |
| 39 个正样本用上 | 10 | **37** | 37 | 11 |
| test ≥ 20 正门禁 | ✗ | **✓** | ✗ | ✗ |
| 评测回答的问题 | 无法回答 | "崩盘 regime 下模型排序如何" | —— | 无法回答 |

方案 C（只把 purge 换成 45、窗口不动）实测：excluded 仍是 411 行 / 27 正——
**光换 margin 没用，2020 必须被某个名义窗口覆盖**。A 与 B' 不可兼得：
27 个正样本集中在一年，只能整体划给一个块。

## 4. 推荐：方案 A，代价如实列出

**推荐 A**（`per_label: true, purge_days: 45, test.start: 2020-01-01`），理由：

1. **test 从 5 正 → 32 正，首次越过 `min_positives_for_metrics=20`**。所有真实
   指标从"样本不足、不可判定"变成"可判定、需带区间报告"——这是本次修订唯一
   能用数字衡量的收益。
2. tail_risk 的本质就是崩盘事件。把 2020 排除在 test 外，等于评测一个
   "没有 tail risk 的 tail risk 模型"。
3. 2022 年的 4 个正样本提供第二个、较弱的不同 regime 证据；**报告必须按年
   分组呈现**（dataset 卡已承诺此口径），2020 主导总量但不是唯一 regime。

**必须与方案同时接受的代价（防误读清单）：**

- **test 被 2020 regime 主导**：32 个正样本里 27 个来自 2020-03 前后。test 指标
  首先度量"疫情式急跌下的排序"，对慢熊/阴跌型 tail 事件的外推没有证据。
- **所有真实数字全部重算作废**：structured AUC 0.7686、KS 0.5873、PR-AUC 0.0258、
  零样本 AUC 0.7542 等全部随切分失效重跑。`eval run`（CPU）分钟级；LoRA 评测
  在 bf16 选项下从 69.5h 缩到约 1/5；**重训适配器必须与 SFT 引用形状修复同批**。
- **valid 仍然只有 1 个正样本**：epoch 选择与校准依旧瘸腿。这是 A 与 B' 的
  交换：2020 给 test（评测效力）还是给 valid（选模/校准效力）。折中不存在。
- train 正样本维持 4 个不变——训练监督密度问题不因本方案解决，扩池（下一步）
  才是那条路。

**什么情况下选 B' 而不是 A**：近期目标若是从真实数据训出一个可信的适配器
（需要校准与 epoch 选择有足够正样本），先给 valid 吃 2020、test 留到扩池后
再扩；评测效力牺牲到扩池之后补。两个方向都合法，取决于先要"能评测"还是先要
"能选模"。

## 5. 决定后的执行清单（拍板后自动依次做）

1. `configs/default.yaml`：`per_label: true`、`purge_days: 45`、
   `test.start: 2020-01-01`（A）或对应 B' 变体；`docs/02`、`docs/05` 同步切分定义。
2. 重跑 `data build --data-config configs/data/stage2_real.yaml`，核对
   split 报告与本文档 §3 的预测逐项一致（不一致 = 停下来查）。
3. 重跑 `eval run`（结构化/TF-IDF/fusion 四行 + matched 基线），全部带区间。
4. CHANGELOG 标记"真实数字以 2026-09-xx 切分为准"，旧数字在 docs/09 标注失效。
5. LoRA 重训与 SFT 引用形状修复同批；评测用 `--no-load-in-4bit`（§bf16 已上线）。

## 6. 复现

本文档全部数字来自以下模拟（用仓库实现，非手算）：

```python
import copy
from datetime import date
import pandas as pd
from shingan.config import load_config
from shingan.eval.splits import assign_split_column, TimeWindow

df = pd.read_parquet("data/processed/panel.parquet")
df["as_of"] = pd.to_datetime(df["as_of"])
mask = df["label_mask_tail_risk"]
data_end = df["as_of"].max().date()
cfg = load_config("configs/default.yaml")
cfg.split.per_label, cfg.split.purge_days = True, 45
cfg.split.test = TimeWindow(start=date(2020, 1, 1), end=date(2024, 12, 31))
out, report = assign_split_column(df, cfg.split, cfg.labels, data_end, label="tail_risk")
print(report.counts)  # 与 §3 方案 A 行数一致
```
