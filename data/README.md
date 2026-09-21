# data 目录布局

本仓库的原则是**代码入库、数据不入库**。`data/` 只保留目录骨架与本说明,**不提交任何数据文件**。
数据按处理阶段分四层,每层的可重生成性不同:越往下游越应该"删了就能重建"。

## 目录说明

| 目录 | 内容 | 可重生成 | 入库 |
| --- | --- | --- | --- |
| `raw/` | 原始下载件:EDGAR filing、新闻、行情、评级与处罚名单。**只读,永不手工编辑** | 否(需重新下载) | 否 |
| `interim/` | 中间产物:文本切片、事件对齐到交易日后的表 | 是 | 否 |
| `processed/` | model-ready 的宽表与标签表,训练与评估直接读这里 | 是 | 否 |
| `external/` | 第三方参考数据:评级动作、SEC enforcement 名单等 | 否(外部导入) | 否 |

约定:`raw/` 中的文件一旦落下就不再修改。需要修正时,把修正逻辑写在 `interim/` 的转换里,
而不是回头改原始文件 —— 这样才能保证整条链路可以从头重放。

## 为什么看不到数据文件

各层目录的**内容**都被 `.gitignore` 排除;目录本身通过每层里的 `.gitkeep`(空文件)保留,
这样 `git clone` 之后目录结构依然完整,下游脚本不需要额外 `mkdir`。

因此:不要为了让某个路径存在而提交占位数据文件,也不要用 `git add -f` 强推数据。

## 重新生成

```bash
# 从已下载的 raw 数据构建 interim / processed(确定性、可重复)
shingan data build

# 完全离线的合成路径:不下载任何外部数据,生成确定性合成数据
shingan data synth
```

`shingan data build` 要求本地已存在 `raw/` 数据;`shingan data synth` 不需要网络,是在 clean checkout 上
验证 pipeline 的推荐入口,与 `shingan demo` 使用同一套确定性合成数据。

采集相关的具体子命令以 `shingan data --help` 为准,本文档不复制其参数以免过期。

## 预期产物

| 路径 | 内容 | 生成方式 |
| --- | --- | --- |
| `data/raw/**` | 原始下载件,按来源分子目录,保留下载当日的快照 | 外部采集流程(见 `shingan data --help`) |
| `data/interim/**` | 清洗、切片、时间对齐后的中间表 | `shingan data build` |
| `data/processed/**` | model-ready 特征宽表与三类标签表(`default_risk` / `fraud_risk` / `tail_risk`) | `shingan data build` |
| `data/processed/synth/**` | 确定性合成数据集及其标签分布报告 | `shingan data synth` |
| `data/external/**` | 评级动作、SEC enforcement 等第三方参考表 | 外部导入 |

## 数据来源与使用条款

- **SEC EDGAR**(10-K / 10-Q / 8-K 等):内容属 public domain,可自由使用。但访问需遵守 SEC 的
  access policy:必须提供**可联系的 `User-Agent`**(形如 `shingan-research/0.1 (contact@example.com)`)
  并**限速**(不超过 10 req/s)。请勿用匿名默认 UA 抓取。
- **FNSPID**(新闻与股价数据集):声明的用途是 research use。使用时需**引用其论文**,并且在任何形式的
  再分发之前,回到其自有仓库**核对当时的 licence**;不同版本条款可能不同,不要凭记忆判断。
- **行情数据(yfinance / Stooq 等)**:仅供本仓库**本地研究使用,不可再分发**。仓库任何位置都不得出现
  这些价格序列本身(包括示例、测试 fixture 与 notebook 输出)。
- **合成数据生成器**(`shingan data synth`):随仓库分发,CC0 / 可自由使用。它是离线验证与 CI 的基础,
  不依赖任何外部数据源。

## 绝不提交

- 任何个人可识别信息(PII):姓名、邮箱、身份证件、联系方式等。
- 任何 vendor 的行情价格序列(见上文,不可再分发)。
- 付费墙内或需要授权才能访问的 filing、研究报告。
- 任何处于 NDA 或其他保密协议下的材料。
