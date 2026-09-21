# Shingan 心眼

[![ci](https://github.com/shuurai/shingan-finrisk/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/shuurai/shingan-finrisk/actions/workflows/ci.yml)

**Evidence-grounded financial risk modelling.** An open-source proof of concept that asks whether *text* signals add measurable information beyond structured financial and market signals — and refuses to claim an answer until the ablation says so.

> Shingan is an open-source proof of concept for a financial risk model built as two cooperating tracks. **Track A** is a calibrated gradient-boosted model over structured financial and market signals. **Track B** is a QLoRA instruction-tuned language model reading SEC filings and financial news. A thin **fusion** layer combines both and emits evidence-grounded risk assessments for three label families: credit, fraud/misstatement, and tail risk. The design target is ranking quality and calibration under strict point-in-time discipline with purged, embargoed time-series validation — **not** return forecasting.

**Status: POC with a first real-data result, and that result is provisional.** The pipeline runs end to end on **synthetic data**, which proves it is connected and nothing more. Stage 2 has now also been executed once against real SEC and market data — 34 companies, one label (`tail_risk`), 39 positives of which 5 fall in the test block. The numbers below are real measurements on real filings and prices, but the sample is far too small to support a performance claim. The SEC filing *text* corpus has since been downloaded (2,222 primary documents) and the panel rebuilt with live text features, but the three-track evaluation has **not** been re-run on that panel yet — so every number quoted below still comes from the text-free run. Everywhere else, the numbers in the documentation are still **targets or gates**, not achieved values.

- Repository: [`shuurai/shingan-finrisk`](https://github.com/shuurai/shingan-finrisk)
- Maintainer: Shane (GitHub [`shuurai`](https://github.com/shuurai))
- License: Apache-2.0

---

## 核心论点：为什么是双轨

项目最初的设想是"把 20 年年报 + 新闻 + 交易信号拼成一个大文本，训一个 LoRA 去预测风险"。本项目**明确否决了这个做法**，理由有两条：

1. **信号形态不同，需要不同的归纳偏置。** 财务比率与量价特征是有量纲、近似平稳、样本量大的数值面板，gradient-boosted trees 在这类数据上稳定、可校准、可归因；把数字序列化成 token 交给 LLM，等于用一个高方差、样本效率低的函数逼近器去做一件树模型做得更好的事。
2. **可验证性依赖分离。** 双轨把"文本贡献了什么"变成一个可以消融的实验问题：structured-only 与 text-only 两条基线各自独立存在，fused 只有在 PR-AUC 上显著超过 structured-only 时才说明文本轨有价值。单一 LoRA 把两种信号混在一个不可分解的权重里，无法回答这个问题。

因此本项目有一条硬性评测要求：**任何声称有效的模型，都必须同时给出 structured-only、text-only、fused 三路结果，外加无模型基线。**

## 三个标签

三个标签都在同一套 as-of / 无前视纪律下计算。右边界截断的样本（`as_of + horizon` 超出数据末端）标记为 `label_mask=false`，**绝不**记为负样本。

| 标签 | 事件定义 | horizon |
| --- | --- | --- |
| `default_risk` | 评级下调 ≥ 2 档，或破产 | 365 个日历日 |
| `fraud_risk` | 财报重述 / 监管执法行动 / 非标准审计意见 | 730 个日历日 |
| `tail_risk` | 峰谷回撤劣于 −30% | 30 个交易日 |

`liquidity_risk`、`event_driven_risk`、`macro_contagion_risk` 已定义但**超出 POC 范围**，不实现。

## 快速开始

无需 GPU、无需网络即可跑通 `demo` 与测试套件。完整训练需要 NVIDIA GPU（开发机为 Windows 11 + RTX 5090 32 GB）。

```powershell
# Windows 11（支持的训练路径）
powershell -ExecutionPolicy ByPass -File scripts\bootstrap.ps1
powershell -ExecutionPolicy ByPass -File scripts\bootstrap.ps1 -Train   # 训练 LoRA 前
powershell -ExecutionPolicy ByPass -File scripts\run_poc.ps1
```

```bash
# macOS / Linux
bash scripts/bootstrap.sh
bash scripts/run_poc.sh
```

`run_poc` 依次执行 `data build` → `train structured` → `eval run` → `eval report`。若本机装有 GNU Make（Windows 默认没有），等价命令是：

```bash
make bootstrap
make build train-structured eval report
make doctor          # 环境自检
make lint typecheck test
```

单独跑最短的 CPU 端到端验证：

```bash
python -m shingan demo                      # 合成数据，CPU，输出 markdown 报告
python -m shingan doctor                    # 打印环境与依赖版本
```

### 关于默认数据配置

`run_poc` 默认使用 `configs/data/poc_largecap10.yaml`（10 只大盘股，全时间跨度）。这个配置下**三个标签里只有一个会被评测**——10 只股票、3 年验证块、1 年 horizon 不足以积累出可用于校准的下调样本。报告不会留空白，而是在 Caveats 段落里写明原因并给出正样本计数。

要端到端跑通全部三个标签，使用演示叠加层（事件率高于标注文档规定的真实基率，因此单独成一个文件而非默认值）：

```bash
bash scripts/run_poc.sh --data-config configs/data/demo.yaml --out artifacts/demo
bash scripts/run_poc.sh --labels tail_risk        # 只评测一个标签
```

更多细节见 [`scripts/README.md`](scripts/README.md)。

## 仓库结构

```
src/shingan/        包本体：CLI、数据、特征、模型、评测、报告
  cli.py            Typer 命令面
  data/             合成数据生成器 + EDGAR / 价格 / 新闻适配器
  features/         结构化特征与文本特征
  models/           structured（GBDT）、text_baseline（TF-IDF）、lora（QLoRA）、fusion
  eval/             切分、purge/embargo、指标、稳定性、压力测试、报告
configs/            data / train / eval 三组 YAML 配置
scripts/            平台包装脚本（bootstrap、run_poc）与说明
docs/               设计文档（架构、数据、标注、训练、评测、Windows 环境、路线图、命名）
templates/          Hugging Face model card / dataset card 模板
notebooks/          探索性 notebook（默认剥离输出）
tests/              pytest 测试套件
```

## 命令面

| 命令 | 作用 |
| --- | --- |
| `shingan doctor` | 环境与依赖自检 |
| `shingan demo` | 合成数据 CPU 端到端，输出 markdown 报告 |
| `shingan data synth` | 生成确定性合成面板 |
| `shingan data build` | 构建特征面板 |
| `shingan data sft` | 导出指令微调数据集 |
| `shingan train structured` | 训练并校准结构化轨 |
| `shingan train lora` | QLoRA 微调文本轨（需 GPU 与 `train` extra） |
| `shingan eval run` | 跑评测，写出 JSON 记录 |
| `shingan eval report` | 从 JSON 记录渲染报告，并与存储指标比对 |
| `shingan publish hf` | 推送到 Hugging Face Hub |
| `shingan version` | 打印版本 |

## 状态表

状态口径：

- `已实现·已验证` — 代码存在，且在合成数据上端到端跑通并有输出产物。
- `已实现·未验证` — 代码存在，但**尚未**对真实数据或实时端点执行过。
- `接口已定义·未验证` — live 端点适配器已写好，从未调用过成功路径。
- `设计已定·未实现` — 设计已固定，代码尚未写。
- `超出 POC 范围` — 有意不做。

| 组件 | 状态 |
| --- | --- |
| 包结构 / `pyproject.toml` / `src/shingan/` 模块划分 | 已实现·已验证 |
| CLI 命令面 | 已实现·已验证 |
| 合成数据生成器 | 已实现·已验证 |
| `shingan demo`（CPU 端到端出报告） | 已实现·已验证 |
| as-of / leakage 断言 | 已实现·已验证 |
| 时序切分 + purge/embargo | 已实现·已验证 |
| 结构化轨（含校准） | 已实现·已验证 |
| 文本基线（TF-IDF 类） | 已实现·已验证 |
| 评测指标与报告渲染 | 已实现·已验证 |
| SEC EDGAR 客户端 | 已实现·已验证（submissions、companyfacts、全文检索端点与 `/Archives` 正文均已在真实数据上跑通；正文语料尚未入库，见下） |
| 价格数据适配器（yfinance / Stooq） | 已实现·已验证（yfinance 拿到 31/34 只的日线） |
| 新闻适配器（FNSPID，离线数据集） | 接口已定义·未验证 |
| QLoRA 训练 | 已实现·未验证 |
| Fusion 层 | 已实现·已验证（真实数据上相对 structured-only 增益为 **负**） |
| **真实数据结果（Stage 2 / `tail_risk` / 34 家公司 / 2221 行面板）** | **structured AUC 0.7686、KS 0.5873、PR-AUC 0.0258；text 基线 0.4074；fused 0.6554。test 仅 5 个正样本，不构成性能结论** |
| 标签复核（`tail_risk` 全部 39 个正样本） | 已实现·已验证（独立重算，不一致率 0.00%） |
| 三个 in-scope 标签的真实事件标注 | 设计已定·未实现（`default_risk` / `fraud_risk` 的事件源未接入） |
| `publish hf` | 已实现·未验证 |
| `liquidity_risk` / `event_driven_risk` / `macro_contagion_risk` | 超出 POC 范围 |

### 诚实性声明

本仓库的合成数据 demo 只能证明**流水线可运行**——数据生成、as-of 检查、特征构造、两条轨训练、融合、切分、指标计算、报告渲染都接通了。它**不能**证明模型具有真实预测能力。合成数据里的信号是生成器人为植入的，指标高只反映实现与设计一致。

Stage 2 已在真实数据上跑过一次，产物在 `artifacts/stage2/`。读那些数字时请注意以下几点，它们不是免责套话，而是当前结果的真实边界：

- **test 块只有 5 个正样本。** AUC 0.7686 建立在 5 个正例上，任何一个正例换位都会显著改变它。按[评测](docs/05-evaluation.md)第 10 节，这触发 F8：不作为性能结论，只作为"管线在真实数据上能算出带区间（或明确无法给出区间）的指标"的证据。
- **融合层没有增益。** fused 的 PR-AUC 比 structured-only 低 0.0075，区间跨零。文本轨目前没有提供可度量的增量，这正是本项目要回答的问题——答案目前是"没有"，而不是"有"。
- **SEC 申报正文：已入库，但尚未在新面板上重跑评测。** Stage 2 运行期间 `www.sec.gov/Archives` 对本网络的任何 User-Agent 都返回 403（"Undeclared Automated Tool"），事后定位到真正的原因**不是 UA 的形状而是 UA 里联系邮箱的域名**：`research@shingan.dev` 之类的普通域名正常返回 200，`a@github.com` 以及浏览器 UA、`curl/8.4.0` 一律 403。下载器 `scripts/fetch_sec_docs.py`（3 并发、磁盘缓存、断点续传、原子写）随后把 2,222 份正文全部取回（0 失败，约 10 GB，56 分钟），并从缓存重建了 `filings.parquet`。面板已用真文本重建：`risk_factor_token_share` 在 2,221 行中有 1,774 行为非零、`neg_kw_density_mdna` 有 1,372 行非零（修复前这两列**全零**）。但 49 个配置特征里仍有 15 个零覆盖、在拟合时被剔除，且**这一次的三路评测还没有跑**——所以上表那个 fused 对比仍然是"文本为空"时的对比。可达性会再次变化，抓取器必须带磁盘缓存与断点续传。
- **`default_risk` 与 `fraud_risk` 没有实现。** 它们的真实事件源（评级历史、执法行动）尚未接入，Stage 2 只评测 `tail_risk`。
- **价格表缺 MRO / WBA / X。** 这三家有申报但没有价格序列，其行现在被正确标为不可观测（见 `artifacts/stage2/label_review.md` 第 5 节：修复前它们曾被当作负样本，并因此抬高过 structured 的 AUC）。

除 Stage 2 列出的这几个数字外，本套文档中出现的其余数字都应读作**目标值**，而不是达成值。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/index.md](docs/index.md) | 文档地图、完整状态表、术语起点 |
| [01 架构](docs/01-architecture.md) | 双轨架构与理由、各边界输入契约、证据化输出契约、明确非目标 |
| [02 数据](docs/02-data.md) | 三条数据腿与真实来源、as-of/no-lookahead 纪律、去重与对齐、处理后面板数据字典 |
| [03 标注](docs/03-labeling.md) | 风险分类学、三个标签的精确 horizon 与阈值、JSONL schema、基率现实 |
| [04 训练](docs/04-training.md) | 结构化轨训练与校准、底座选型、QLoRA 配置、显存预算、Windows/Blackwell 配方 |
| [05 评测](docs/05-evaluation.md) | 指标与门槛、purged walk-forward + embargo、压力测试、基线消融、证伪条件 |
| [06 Windows 环境](docs/06-windows-setup.md) | Windows 11 + RTX 5090 逐步搭建、cu128 安装、排错表 |
| [07 路线图](docs/07-roadmap.md) | Stage 0-5 分阶段计划与推进条件 |
| [08 命名决策](docs/08-naming.md) | 候选名比较、命名族、改名流程 |
| [ADR](docs/adr/) | 架构与命名决策记录 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 开发环境、提交规范、测试要求 |

## 引用

若使用本项目的代码、模型或标签数据集，请按 [`CITATION.cff`](CITATION.cff) 引用。

## License

Apache-2.0，见 [`LICENSE`](LICENSE)。
