# 02 数据

本文定义三条数据腿的真实来源、处理纪律、合成数据生成器的存在理由，以及处理后面板数据的字段字典。文中所有"未验证"标记均为字面含义：相关适配器代码存在，但尚未对 live 端点成功执行过。

## 1. 三条数据腿

| 腿 | 模态 | 来源 | 覆盖 | 在架构中的去向 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| 披露文本 | 非结构化全文 | SEC EDGAR 10-K / 10-Q / 8-K | 2001 年起全文可检索 | Track B 输入；`fraud_risk` 与 `default_risk` 的事件源 | 适配器 `data/edgar.py` 已写好·未验证 |
| 披露数值 | 结构化 | EDGAR inline XBRL financial facts | 2009 年起较可靠 | `features/ratios.py` | 同上 |
| 新闻 | 非结构化 + 情感 | FNSPID | 约 15.7M 篇文章，1999–2023，覆盖 4,775 家 S&P 500 公司，带情感分 | Track B 输入；`features/text.py` 的计数类特征 | 适配器 `data/news.py` 已写好·未验证 |
| 价量 | 结构化面板 | 日频价格/成交量（yfinance、Stooq） | 日频 | `features/technical.py`；`tail_risk` 标签的价格源 | 适配器 `data/prices.py` 已写好·未验证 |
| 合成 | 全部 | `data/synthetic.py` | 确定性生成 | 四条腿的离线替代，供 demo / CI | 已实现·已验证 |

**"薄适配器"的含义**：`data/edgar.py`、`data/news.py`、`data/prices.py` 只做请求构造、分页、限速、重试（`tenacity`）、原始响应落盘与最小字段映射。它们**没有**对真实端点跑通过完整成功路径，也没有处理过真实数据里的畸形记录。把它们当成"待验证的接口"，不是"已可用的数据源"。

## 2. 各腿的数据细节

### 2.1 SEC EDGAR

- **拉取范围**：10-K（年报）、10-Q（季报）、8-K（重大事件）。8-K 是 `event_driven_risk` 与部分 `fraud_risk` 事件的关键来源，即使该标签超出 POC 范围，文本仍进入 Track B 的输入。
- **必须的请求头**：SEC 的访问政策要求声明身份，所有请求必须携带 `User-Agent`，形如 `shingan-research/0.1 (contact: <email>)`。缺这个头会收到 403，而不是一个可读的错误。
- **限速**：遵守 SEC 公布的速率上限（10 req/s 量级），`edgar.py` 用 token-bucket + `tenacity` 退避实现。POC 阶段另行把速率调低，宁可慢。
- **全文与数值分开取**：
  - 全文（HTML/txt）→ 清洗为纯文本后按 Item 切段（Item 1A Risk Factors、Item 7 MD&A、Item 8 Financial Statements 是主要取材段）。
  - inline XBRL facts → 通过 `companyfacts`/`companyconcept` 接口取，保留 `frame` / `end` / `filed` 三个时间字段。**这三个字段是 as-of 语义的物理基础**：一份 FY2019 的 10-K 里 2019 年数据，直到 `filed` 日期才可用。
- **重述的处理**：同一 `(cik, concept, end)` 可能出现多个 `filed` 版本。builder 必须按 `filed <= as_of` 取**当时可见的那个版本**，而不是取最新值。这是与 `fraud_risk` 标签最直接相关的穿越风险点。
- **POC 折衷**：不做全量抓取，只取 Stage 2 指定公司集合与年份区间（见[路线图](07-roadmap.md)）。

### 2.2 FNSPID（新闻）

- **内容**：约 15.7M 篇金融新闻，1999–2023，覆盖 4,775 家 S&P 500 公司，自带情感分数。
- **用途**：Track B 的第二类文本；`features/text.py` 从中派生计数类聚合特征（条数、情感均值、情感波动、极端负面占比）。
- **对齐**：新闻发布时间戳是**带时分的**，必须映射到交易日。规则见第 4.3 节。
- **已知问题**：
  - 情感分是数据集提供的，来源模型未知、未经本项目校准。它作为**特征**可用，但**不能**直接当作标签或真值。
  - 覆盖度随年份变化，早期年份的文章密度明显低于近年。跨期比较时必须按年份分组评估（`eval/splits.py` 按时间切分，天然做到这一点）。
  - 同一事件常被多家转载，需要去重（第 4.2 节）。
- **状态**：FNSPID 本身是离线数据集，`data/news.py` 是本地快照的读取适配器，尚未在真实快照上运行过。

### 2.3 价格/成交量面板

- **用途**：
  1. 技术/微观结构特征（`features/technical.py`）；
  2. `tail_risk` 标签的计算基础（30 个交易日累计回撤）；
  3. 回测中的未来收益与未来已实现波动（IC 与分位回测需要，见[评测](05-evaluation.md)）。
- **来源**：`data/prices.py` 提供 yfinance 与 Stooq 两个适配器。选两个是为了在其中一个限流或数据缺口时能交叉核对。
- **必须处理**：
  - **复权**：分红与拆股必须调整，否则回撤与收益计算会注入虚假跳变。适配器统一抓取调整后价格，并在元数据中记录复权方式。
  - **交易日历**：所有面板索引来自同一交易日历（NYSE）。非交易日不得生成特征行。
  - **停牌与退市**：`tail_risk` 与 `default_risk` 的真实正样本经常伴随停牌或退市，价格序列会出现缺口。规则见第 4.4 节。
- **状态**：未对 live 端点验证。POC 的 `demo` 路径完全走合成数据，不触发网络。

### 2.4 其他来源（设计已定·未实现）

以下来源本项目承认其必要性但**尚未接入**：

| 来源 | 用于 | 缺口 |
| --- | --- | --- |
| 评级机构历史（S&P / Moody's） | `default_risk` 的"下调 >= 2 档"判定 | 需商业订阅或 WRDS；POC 无许可，必须替换为可公开核对的代理事件 |
| SEC enforcement actions 列表 | `fraud_risk` 的事件源 | 可从 EDGAR 公开页面解析，尚未实现 |
| 审计意见类型 | `fraud_risk` 的"非标意见"判定 | 需从 10-K 的审计报告段解析，属文本抽取任务，未实现 |
| 重述公告（Item 4.02 8-K） | `fraud_risk` 的事件源 | 依赖 8-K 全文解析，未实现 |

这些缺口直接影响[路线图](07-roadmap.md) Stage 2 的可执行性：在评级历史缺失的情况下，`default_risk` 的标注只能退化用代理事件，且必须在数据集卡中显式说明。

## 3. 合成数据生成器

- 代码：`src/shingan/data/synthetic.py`
- 命令：`shingan data synth`（离线，确定性，固定种子）

### 3.1 为什么存在

1. **可复现**：真实数据会变（EDGAR 补报、供应商修订、yfinance 接口变更）。一个随源数据漂移的 demo 不是回归测试。
2. **离线**：CI 与 `shingan demo` 不允许依赖网络。合成数据让整条链路（含 BERT/LLM 之外的所有步骤）在无网络环境跑通。
3. **CI 友好**：生成快（秒级）、体积小、CPU 可跑，因此可以对每个 PR 跑端到端。
4. **测试边界条件**：真实数据里极稀有的情形（停牌、重述、极短历史、全缺失列）在合成数据里可以被主动注入，用来验证 `leakage.py` 的断言确实会抛错。
5. **使融合对比有意义**——见下节。

### 3.2 受控的潜在结构

合成数据的核心设计是：**植入一个只出现在文本里、结构化特征无法观测的潜在因子**。

生成过程：

1. 为每家公司生成一个隐藏的风险状态轨迹（latent factor），它决定事件是否在 horizon 内发生。
2. **结构化特征**由潜在因子经一个**刻意加噪、且有信息损失**的映射生成。具体做法：把潜在因子投影到低维，丢弃若干分量，再叠加相关噪声。结果是与标签有正相关、但相关性有上限的关系。
3. **文本 token**由完整的潜在因子生成（包括被结构化侧丢弃的那些分量），并加入金融语域的模板、措辞噪声与无关段落。
4. 标签由潜在因子与事件时间决定，严格遵守第 5 节的 as-of 规则。

这样设计的效果是：structured-only 的 AUC 会稳定在一个有明显天花板的值，而 text-only 与 fused 在理论上界更高。**如果 fused 不能超过 structured-only，说明实现有问题，而不是"文本没用"**。这正是把合成数据当回归测试的用法。

必须诚实说明的局限：这个"文本有增量"的结论是生成器**假设进去的**，不是发现的。合成数据上融合增益大于零是构造保真的检查，**不构成任何关于真实世界文本是否有用的证据**。真实结论只能来自 Stage 2 之后的真实数据评测。

### 3.3 与真实数据的隔离

- 合成数据行的 `is_synthetic` 列恒为 `true`，真实数据恒为 `false`。
- `data/builder.py` 拒绝把两种来源混进同一份 processed 表；混用即中止。
- 报告（`eval/report.py`）在页首打印数据来源，合成数据报告带醒目标记，防止截图被误读成真实结果。
- 测试集只用真实数据，不用合成数据；在真实数据到位前，合成数据的 split 被称为 `smoke split`，不称为 test。

## 4. 处理纪律

### 4.1 as-of / no-lookahead

**核心不变量**：`processed` 表中的每一行由一个 `(ticker, as_of)` 标识，该行的**全部**特征必须满足"在 `as_of`（含当日收盘后）已经公开"。

具体规则：

| 数据 | 可用性判定 |
| --- | --- |
| 10-K/10-Q/8-K 全文 | `filing_date <= as_of` |
| inline XBRL 数值事实 | `filed <= as_of`（不是 `end <= as_of`） |
| 新闻 | 发布时间戳（转 UTC 日期）`<= as_of` |
| 价格/成交量 | 交易日 `<= as_of`，且只用当日收盘及之前的数据 |
| 财务事件（原标签事件） | `event_date > as_of`（事件必须在未来；事件本身绝不进特征） |

必须被排除的内容：

- `as_of` 之后发生的**重述**。一家公司在 t 时点披露的数字，即使后来被证明是错的，t 时点的模型也只能看到原数字。
- `as_of` 之后公布的**执法行动**与**审计意见变更**。
- `as_of` 之后的**新闻**，包括回顾性文章。回顾性文章尤其危险：它可能直接总结"该公司最终破产"，等于把标签写进输入。
- 任何由 `event_date` 派生的字段（`days_to_event` 之类）——这是最直白的标签泄漏。

实施方式：

- `data/builder.py` 的 join 全部是 as-of join（pandas `merge_asof` 语义），不是普通 join。
- `src/shingan/leakage.py` 提供断言：列名黑名单（`label_`、`fwd_`、`event_` 前缀不得出现在 X 中）、时间戳单调性检查、以及"特征列与标签列的相关性异常高"的启发式告警。
- 断言在构建时执行，失败即抛错中止，不降级为警告。

### 4.2 多源去重

| 层次 | 规则 |
| --- | --- |
| 同一文档多次抓取 | 以 `(source, doc_id, filed/published_timestamp)` 去重，保留首次抓取 |
| 同一 XBRL fact 多版本 | 按 `filed` 取 `filed <= as_of` 的最新版本；不合并、不取均值 |
| 新闻转载 | 规范化后（小写、去标点、压缩空白）对标题做精确匹配去重；正文前 256 字符做 shingle 相似度 >= 0.9 的近似去重，同簇保留最早发布的一条 |
| 跨源同事件 | EDGAR 与新闻描述同一事件时保留两条记录（模态不同），但在 `features/text.py` 的计数特征中按事件簇去重后再计数，避免一次事件被计成十次 |

去重的所有决策记录到 `data/interim/dedup_log.parquet`，便于回溯"某条新闻为什么没进模型"。

### 4.3 交易日对齐

- **索引**：统一使用 NYSE 交易日历。非交易日不产生特征行。
- **新闻 → 交易日**：发布时间在交易日 00:00–16:00 ET 之间 → 归到当日；16:00 之后或非交易日 → 归到**下一个**交易日。收盘后发布的新闻不能被当成当日已知信息。
- **财报 → 交易日**：`filing_date` 为交易日则归当日，否则归下一个交易日。10-K/10-Q 常在盘后提交，因此这条规则与上一行一致。
- **季度财务数据 → 日频面板**：XBRL 的季度事实向前填充（ffill），**绝不向后填充**。ffill 的上界是 `as_of`，天然无穿越。
- **时区**：所有时间戳以 UTC 存储，展示层转 ET。日界以 ET 收盘为界，因为数据的语义边界是市场收盘。

### 4.4 停牌、退市与缺口

- **停牌**：价格序列保留 NaN 行，不用前收盘填充。特征侧对连续 NaN 有上限（`max_nan_run`，默认 10 个交易日），超限则该窗口特征置 NaN 并让树模型处理缺失。
- **退市**：退市后的行不存在。`tail_risk` 在退市前的窗口若能算出回撤则正常标注；`default_risk` 的破产/违约事件由事件源提供，不依赖价格序列延续。
- **幸存者偏差**：只取"当前仍是 S&P 500 成分股"的公司集合会系统性丢掉失败公司的历史，直接压低正样本率并高估模型表现。**必须使用历史成分股名单（point-in-time membership）**。POC 阶段的 Stage 2 只取 5–10 家指定公司，该问题不显著，但必须在数据集卡中记录；Stage 4 扩到全 S&P 500 时这是阻塞项。详见[数据集卡模板](../templates/dataset_card.md)的 known biases 一节。

## 5. 数据字典

`data/processed/` 下的面板数据集，一行 = 一个 `(ticker, as_of)`。列类型为写入 parquet 时的物理类型。

### 5.1 标识与元数据

| 字段 | 类型 | 来源 | 说明 |
| --- | --- | --- | --- |
| `ticker` | string | 价格源 | 交易所代码，对应 `as_of` 时点的代码（历史代码变更以 PIT 名单为准） |
| `cik` | string | EDGAR | 10 位零填充 CIK，EDGAR 引用的主键 |
| `company_name` | string | EDGAR | 用于展示，不参与建模 |
| `sector` | category | GICS 或 EDGAR SIC 映射 | 用于跨行业分组评估；**不得**用于 target encoding |
| `as_of` | date | 构建生成 | 该行信息的截止交易日 |
| `split` | category | `eval/splits.py` | `train` / `valid` / `test` / `purged`；`purged` 行为被 purge/embargo 剔除的行，保留在表中以便审计 |
| `is_synthetic` | bool | 生成器 | 合成数据为 `true` |
| `data_version` | string | 构建生成 | processed 数据集版本号，写入数据集卡 |
| `n_sources` | int | 构建生成 | 该行贡献过信息的源数量，用于发现"只有一条新闻就建了一行"的薄弱样本 |

### 5.2 结构化特征 — 财务比率（`features/ratios.py`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `debt_to_equity` | float | 总负债 / 股东权益 |
| `debt_short_term_ratio` | float | 短期负债 / 总负债，衡量负债结构短期化 |
| `current_ratio` | float | 流动资产 / 流动负债 |
| `interest_coverage` | float | EBIT / 利息费用 |
| `net_margin` | float | 净利润 / 营收 |
| `roa` / `roe` | float | 资产 / 权益回报率 |
| `fcf_margin` | float | 自由现金流 / 营收 |
| `altman_z` | float | Altman Z-score（破产风险的经典线性代理） |
| `accruals_ratio` | float | 应计项目 / 总资产，舞弊文献中的常用信号 |
| `revenue_yoy` | float | 营收同比 |
| `asset_growth` | float | 总资产同比 |
| `goodwill_to_assets` | float | 商誉 / 总资产，减值风险代理 |
| `ratios_missing_frac` | float | 该行比率列的缺失比例，作为显式特征提供给树模型 |

所有比率在计算前必须使用 `filed <= as_of` 的 XBRL 版本，且财报口径变更（如会计政策变更）不做跨期平滑——平滑会引入未来信息。

### 5.3 结构化特征 — 技术/微观结构（`features/technical.py`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `ret_1d` / `ret_5d` / `ret_20d` / `ret_60d` / `ret_252d` | float | 对数收益率的多个回看窗口 |
| `vol_20d` / `vol_60d` | float | 已实现波动率（年化） |
| `downside_vol_60d` | float | 下行波动率，只计负收益 |
| `skew_60d` / `kurt_60d` | float | 收益分布偏度与峰度，尾部特征 |
| `max_drawdown_60d` | float | 60 日滚动最大回撤，用负值表示 |
| `dist_52w_high` | float | 距 52 周高点的百分比距离 |
| `rsi_14` | float | RSI |
| `macd_hist` | float | MACD 柱状值 |
| `atr_14` | float | ATR |
| `adx_14` | float | ADX，趋势强度 |
| `amihud_illiq_20d` | float | Amihud 非流动性指标，`|ret| / 成交额` 的 20 日均值 |
| `turnover_20d` | float | 换手率 20 日均值 |
| `beta_252d` | float | 对市场指数的 252 日 beta |
| `abnormal_volume_20d` | float | 当前成交量 / 20 日均量的比值，异常成交量信号 |
| `vix_level` / `vix_chg_5d` | float | VIX 水平与 5 日变化，市场 regime 变量 |
| `credit_spread_chg_20d` | float | 信用利差 20 日变化（宏观 regime 变量；数据源待接入） |

技术特征的设计原则：全部只用 `as_of` 及之前的价格，滚动窗口必须有足够的非 NaN 历史才输出，否则整行特征置 NaN 并把该行标记为 `insufficient_history`。

### 5.4 文本派生特征（`features/text.py`）

这些是**计数/统计类**特征，属于结构化侧；语义理解由 Track B 承担。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `n_news_30d` / `n_news_90d` | int | 近 30/90 日新闻条数（去重后） |
| `sent_mean_30d` / `sent_std_30d` | float | FNSPID 情感分的均值与标准差 |
| `sent_neg_share_30d` | float | 情感分低于阈值的占比 |
| `neg_kw_density_mdna` | float | MD&A 段中负面词典词密度 |
| `risk_factor_token_share` | float | Item 1A 占全文 token 比例，篇幅变化的代理 |
| `disclosure_len_tokens` | int | 本期披露文本 token 数 |
| `disclosure_len_chg` | float | 与上期的 token 数变化率 |
| `going_concern_hits` | int | going concern 相关短语命中次数 |
| `uncertainty_hits` | int | 不确定性/模糊语义词命中次数 |
| `restatement_hits` | int | 重述相关短语命中次数 |

词典与规则定义在 `features/text.py`，词表随数据版本一起记录，避免"同一版本号下词表漂移"。

### 5.5 标签与前瞻目标

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `label_default_risk` | int8 | 0/1，定义见[标注](03-labeling.md) |
| `label_fraud_risk` | int8 | 0/1 |
| `label_tail_risk` | int8 | 0/1 |
| `label_mask_default_risk` 等 | bool | 该行在该标签上是否可判定（缺少事件源覆盖时为 `false`） |
| `event_date_default_risk` 等 | date | 事件发生日；**仅用于构建与审计，绝不出现在特征矩阵中** |
| `source_of_record_default_risk` 等 | string | 该标签事实的来源（如 `sec_item_4.02_8k`、`price_panel`） |
| `horizon_days_default_risk` | int | 365；`fraud_risk` 为 730；`tail_risk` 为 30（交易日） |
| `fwd_ret_21d` | float | 未来 21 个交易日收益。**仅用于 IC 与回测，是连续量，不是标签** |
| `fwd_realized_vol_21d` | float | 未来 21 个交易日已实现波动率。**IC 的首选输入**，理由见[评测](05-evaluation.md#陷阱-3ic-必须用连续的前瞻量) |
| `fwd_max_drawdown_30d` | float | 未来 30 个交易日最大回撤（负值），`tail_risk` 的连续版本，供回归式评估与排序检验 |
| `sample_weight` | float | 标签的样本权重（用于降采样后的重加权，见[标注](03-labeling.md)） |

前缀 `fwd_` 与 `label_`、`event_` 一起构成 `leakage.py` 的列名黑名单：这些列不得出现在任何模型的输入矩阵 X 中。

## 6. 许可与署名

| 来源 | 许可 / 条款 | 必须做的事情 |
| --- | --- | --- |
| SEC EDGAR | 美国联邦政府公开信息，无版权限制 | 声明请求头身份；不声称 SEC 对本项目的背书；注明数据取自 EDGAR |
| FNSPID | 由数据集作者在 Hugging Face 发布，遵循其数据集卡声明的许可 | 引用原论文与数据集卡；**不得**再分发原始新闻正文，衍生物（聚合特征、抽取片段）在数据集卡中说明来源 |
| yfinance / Stooq | 免费接口，各自有使用条款；Stooq 数据再分发受限 | POC 与本地研究用途；**不得**随仓库或数据集分发原始价格面板 |
| Qwen3 权重 | Apache-2.0（`Qwen/Qwen3-14B`） | 在模型卡中标注 base model 与许可 |
| 第三方评级 / WRDS | 商业订阅，本项目 POC 无许可 | 不引入、不再分发；Stage 2 需以公开代理事件替换 |

再分发规则（本项目对外发布的所有 artifact 都受此约束）：

1. **可再分发**：合成数据、代码、特征列定义、聚合统计量、模型权重（受 base model 许可约束）、抽取出的短引文（用于 evidence 展示，按 fair use 控制长度）。
2. **不可再分发**：新闻正文全文、EDGAR 之外的价格面板原始数据、任何商业数据源的原始记录。
3. HF 上发布的 `shuurai/shingan-finrisk-labels` 只包含**标签与派生特征**，不含原始新闻正文；文本轨训练所需原文由使用者自行从原始来源获取。详见[数据集卡模板](../templates/dataset_card.md)。

## 7. 相关文档

- 标签如何从这些事件定义出来：[03 标注](03-labeling.md)
- 特征如何进入两条轨：[01 架构](01-architecture.md)、[04 训练](04-training.md)
- as-of 与切分如何在评测中被验证：[05 评测](05-evaluation.md)
- 数据来源的许可如何在 HF 卡片中呈现：[数据集卡模板](../templates/dataset_card.md)
