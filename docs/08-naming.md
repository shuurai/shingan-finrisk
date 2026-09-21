# 08 命名决策

本文记录项目命名的比较过程与最终决定。命名在这里不是品牌问题，而是工程问题：名字会出现在包名、CLI 二进制名、Hugging Face 仓库名、Python import 路径、环境变量前缀和文档标题里，改名的成本随发布进度增长。因此把决策一次做完，并把"如何改名"写清楚。

## 1. 命名家族（已决定）

| artifact | 名称 |
| --- | --- |
| 项目名 | Shingan（心眼，"the mind's eye"） |
| GitHub 仓库 | `shuurai/shingan-finrisk` |
| Python 包与 CLI | `shingan` |
| Hugging Face 模型 | `shuurai/shingan-qwen3-14b-finrisk` |
| Hugging Face 数据集 | `shuurai/shingan-finrisk-labels` |
| Benchmark | `shuurai/shingan-bench` |
| License | Apache-2.0 |

命名模式：`shingan` 为不可变的词根；`-qwen3-14b-finrisk` 描述基座与任务；`-finrisk-labels` 描述数据集内容；`-bench` 为评测集。词根与后缀之间统一用连字符（`-`），后缀内部用小写与数字，不使用下划线。

## 2. 四个候选

| 候选 | 来源 / 含义 | 语言 | 音节 |
| --- | --- | --- | --- |
| **Shingan** | 心眼 — "the mind's eye"；洞察力，也指"看见隐藏之物" | 中文 / 日文 | shin-gan |
| Presage | 英文名词/动词，预示、前兆（尤指凶兆） | 英文（源自拉丁 praesagium） | pre-sage |
| Shura | 修羅（梵语 asura 的汉译/日语音读）；战斗、厮杀；日语 `修羅場` 指惨烈场面 | 中文 / 日文 / 梵语 | shu-ra |
| Kizashi | 兆（し），日语"征兆、前兆、苗头" | 日文 | ki-za-shi |

## 3. 评分标准

四个维度，各 1–5 分。权重反映本项目的实际使用场景（开源仓库、双语受众、需要形成家族）。

| 维度 | 含义 | 权重 | 为什么重要 |
| --- | --- | --- | --- |
| 双语可读性 | 中英文使用者都读得对、记得住，无歧义音节 | 25% | 项目文档与讨论以中文为主，代码与发布以英文为主，两边都要顺畅 |
| 唯一性 | 作为搜索词与仓库名不易与既有项目/品牌冲突 | 25% | HF 仓库名与 PyPI 包名必须可注册，搜索时不能被无关内容淹没 |
| 语义契合 | 表达"提前看见隐藏的风险" | 30% | 名字是唯一必须在一行内解释清项目论点的地方 |
| 可扩展性 | 能否自然派生出模型/数据/benchmark 等家族命名 | 20% | 至少五个 artifact 需要共享词根 |

评分：

| 候选 | 双语可读性 | 唯一性 | 语义契合 | 可扩展性 | 加权总分 |
| --- | --- | --- | --- | --- | --- |
| **Shingan** | 5 | 4 | 5 | 5 | **4.75** |
| Presage | 2 | 2 | 4 | 4 | 3.00 |
| Kizashi | 3 | 2 | 4 | 4 | 3.25 |
| Shura | 4 | 1 | 3 | 4 | 2.90 |

各维度的评注：

- **双语可读性**：`Shingan` 对中文读者直接是"心眼"，不需要翻译；对英文读者按 romanization 读 shin-gan 即可，无歧义音节。`Presage` 对中文读者是陌生拼写，且发音 `/prɪˈseɪdʒ/` 与拼写的对应关系不直观（`-sage` 读 /sɪdʒ/）。`Kizashi` 的重音位置对非日语使用者不确定（ki-ZA-shi），且英语语境下常被读成 /kɪˈzɑːʃi/。
- **唯一性**：`Presage` 是通用英文词，搜索污染严重，且已有多个同名项目。`Kizashi` 是 Suzuki 的一款车型名，商业品牌冲突明显，搜索结果被车型占据。`Shura` 在动漫、游戏、音乐的命名中高度饱和。`Shingan` 短且无常见英文同形词，搜索污染最低。
- **语义契合**：`Shingan` 与 `Kizashi` 都直接是"预兆/洞察"语义，契合度高。`Presage` 语义也对（预示凶兆），但偏"预言"而非"洞察"，与项目"从已有文本中读出未被编码的信息"的机制略有距离。`Shura` 语义是"战斗/惨烈"，需要二次解释才能与风险关联，契合度最低。
- **可扩展性**：四者都能加连字符后缀。`Shingan` 额外占优之处在于词根短（7 字符）且不含连字符，作为 CLI 二进制名不与常见命令冲突；`presage` 有被误读为动词 `pre-sage` 的风险。

## 4. 为什么是 Shingan

1. **语义即论点**。"心眼"有两层意思，两层都正好是项目要做的事：一是洞察力（从噪声里分辨信号），二是"看见隐藏之物的眼睛"（看见数字里没有编码进去的风险）。名字本身就能承担一句项目简介的功能。
2. **中文原生，英文可用**。文档以中文为主，`心眼` 对中文读者是零成本理解；同时 `shingan` 是一个干净的 ASCII 词，可以安全地作为 Python 包名、CLI 名、目录名与 URL 片段，不需要转写或缩写。
3. **唯一性足够**。没有常见英文同形词，没有显著商业品牌冲突，作为 GitHub 仓库名与 HF 命名空间下的前缀都可用。
4. **家族扩展自然**。`shingan-qwen3-14b-finrisk`、`shingan-finrisk-labels`、`shingan-bench` 三个名字都能被一眼理解为同一家族，不需要额外说明命名规则。
5. **长度合适**。7 个字符，在 `src/shingan/`、`import shingan`、`shingan doctor` 里都不会造成视觉或书写负担。

被否掉的关键原因：

- **Presage**：通用英文词导致唯一性不足，且发音拼写不直观，对主要的中文受众不友好。
- **Shura**：语义指向"战斗"而非"风险"，需要额外解释；且命名空间饱和。
- **Kizashi**：语义契合但唯一性最差（车型名冲突），且对中文读者没有语感亲和（不像"心眼"能被直接理解），在双语场景下损失了它最大的优点。

## 5. `redfruit` 的说明

`redfruit` **只是本项目工作目录的占位文件夹名**，从来不是项目名，也不是任何 artifact 的一部分。

需要明确的几点：

- 不存在也不应该存在名为 `redfruit` 的 Python 包、CLI 命令、Hugging Face 仓库或数据集。
- 目录名 `redfruit-model` 与项目名 `Shingan` 之间没有映射关系；不要因为它产生"项目曾叫 redfruit"的推断，也不要在代码、文档、注释或提交信息中使用 `redfruit` 作为项目指代。
- 若工作目录被重命名为 `shingan`，不需要改动任何代码或配置：仓库内的所有路径引用都是相对路径（`paths.py` 以包所在位置为基准解析），不依赖工作目录名。
- 凡是发现文档或代码里出现 `redfruit`，都应当视为待清理的占位残留。

## 6. 如果决定改名

假设未来因商标冲突或命名空间问题必须改名。改名的完整清单与其影响范围：

| # | 位置 | 当前值 | 是否影响用户 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | GitHub 仓库名 | `shuurai/shingan-finrisk` | 是 | GitHub 会自动重定向旧 URL，但 HF 卡片与文档中的链接需要更新。词根仍是 `shingan`，后缀 `-finrisk` 仅用于区分 GitHub 仓库 |
| 2 | Python 包目录 | `src/shingan/` | 是 | **破坏性变更**，import 路径改变 |
| 3 | CLI 命令名 | `shingan` | 是 | **破坏性变更** |
| 4 | `pyproject.toml` 的 `name`、`[project.scripts]` 入口 | `shingan` | 是 | 与 2、3 同步改 |
| 5 | `src/shingan/__about__.py` 的版本与项目名常量 | — | 否 | 单一版本来源，见下 |
| 6 | 环境变量前缀（若有） | `SHINGAN_*` | 是 | 需要提供过渡期的兼容读取 |
| 7 | HF 模型仓库 | `shuurai/shingan-qwen3-14b-finrisk` | 是 | 可以新建仓库而非改名，旧仓库保留并加 deprecated 说明 |
| 8 | HF 数据集仓库 | `shuurai/shingan-finrisk-labels` | 是 | 同上 |
| 9 | Benchmark 仓库 | `shuurai/shingan-bench` | 是 | 同上 |
| 10 | `templates/model_card.md` 的 frontmatter 与正文 | 含模型名 | 否 | 模板，改起来零成本 |
| 11 | `templates/dataset_card.md` | 含数据集名 | 否 | 同上 |
| 12 | `docs/` 全部文档的标题与正文 | `Shingan` / `shingan` | 否 | 机械替换，但需人工复核语义通顺 |
| 13 | `README.md`（仓库根，本套文档维护） | 含项目名与仓库链接 | 是 | 与 12 一并机械替换；同时更新 `pyproject.toml` 的 `readme` 字段指向的文件名 |
| 14 | Zenodo DOI 元数据 | 与 release tag 绑定 | 是 | 旧 DOI 不可改，需发新版本并保留旧版记录 |
| 15 | 已发布的报告与 runs 目录中的元数据 | — | 否 | 历史记录不追溯修改，保留原名并加注释 |

改名的推荐顺序：

1. 新建 HF 仓库（不改名旧仓库），旧仓库加 deprecated 说明与指向新仓库的链接。
2. 在 GitHub 上重命名仓库（保留自动重定向）。
3. 在**一个 PR 内**同时改 2、3、4、5，避免中间状态不可用。必要时提供一个过渡期的兼容 shim（旧命令名转发到新命令名并打印弃用警告），并在下一个小版本移除。
4. 批量更新文档与模板。
5. 发 release、请求 Zenodo 生成新版本 DOI，在旧 DOI 记录中注明后继版本。

避免的中间状态：只改了文档没改包名（文档指向不存在的命令），或只改了包名没改 `pyproject.toml`（`pip install -e .` 后 CLI 入口失效）。

## 7. 相关文档

- 命名决策的正式记录：[ADR-0001](adr/0001-project-name.md)
- 发布时的 artifact 清单：[07 路线图](07-roadmap.md) Stage 5
- 卡片模板中的名称位置：[模型卡模板](../templates/model_card.md)、[数据集卡模板](../templates/dataset_card.md)
