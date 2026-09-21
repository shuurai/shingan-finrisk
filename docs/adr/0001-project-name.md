# ADR-0001: 项目名称定为 Shingan

- 状态：Accepted
- 日期：2026-09-21
- 决策者：Shane（`shuurai`）

## Context

项目需要一个名称，它要同时作为 GitHub 仓库名、Python 包名、CLI 二进制名、Hugging Face 模型/数据集/benchmark 的共享词根，并出现在文档标题中。改名成本随发布进度增长，因此需要在发布任何 artifact 之前一次定下来。

候选四个：Shingan（心眼）、Presage、Shura（修羅）、Kizashi（兆し）。评分标准为双语可读性、唯一性、语义契合（"提前看见隐藏的风险"）、可扩展性（能否派生家族命名）。

## Decision

采用 **Shingan**（心眼，"the mind's eye"），Apache-2.0。命名家族：

| artifact | 名称 |
| --- | --- |
| GitHub 仓库 | `shuurai/shingan` |
| Python 包与 CLI | `shingan` |
| Hugging Face 模型 | `shuurai/shingan-qwen3-14b-finrisk` |
| Hugging Face 数据集 | `shuurai/shingan-finrisk-labels` |
| Benchmark | `shuurai/shingan-bench` |

否决理由：

- **Presage** — 通用英文词，唯一性不足（搜索污染、已有同名项目），且 `/prɪˈseɪdʒ/` 的拼写发音关系对中文受众不直观。
- **Shura** — 语义指向"战斗/惨烈"，与风险预测需二次关联；命名空间在动漫/游戏领域高度饱和。
- **Kizashi** — 语义契合，但 `Kizashi` 是 Suzuki 的车型名，搜索污染最严重；且对中文读者无语感亲和。

## Consequences

正面：

- 词根短（7 字符）、纯 ASCII，可安全用于包名、CLI 名、目录名与 URL 片段。
- "心眼"的语义同时覆盖"洞察力"与"看见隐藏之物"，与项目论点（读出数字未编码的风险）一致，名称本身可承担一句简介。
- 家族命名模式（`shingan-<base>-<task>`、`shingan-<task>-<content>`）一致，可扩展。

代价与约束：

- 名称在中文语境下自然，但英文语境下需要一次解释（`Shingan` 非英语词）。所有对外材料必须在首次出现时给出 `心眼 / "the mind's eye"` 的注解。
- 需一次性确认 `shingan` 在 PyPI 与目标 HF 命名空间下的可用性；若 PyPI 包名不可用，采用 `shingan-ml` 作为分发名而保持 import 路径为 `shingan`。
- 改名清单与流程固化在 [08 命名决策](../08-naming.md) 第 6 节，以备将来发生冲突。

补充记录：`redfruit` 仅为工作目录的占位文件夹名，从未作为项目名，也不对应任何 artifact。
