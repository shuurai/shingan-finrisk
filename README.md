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

## 测试与覆盖率

[![ci](https://github.com/shuurai/shingan-finrisk/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/shuurai/shingan-finrisk/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-160%20passed-brightgreen)
![coverage](https://img.shields.io/badge/coverage-34%25-yellow)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

> `ci` 是唯一自动更新的徽章;其余四个是静态的,由下面的命令复现。**门禁表如实报红,不是全绿**——原因见本节末尾。

**CI 跑的是 Linux(`ubuntu-latest`)、Python 3.11、单个 job。** 无 matrix、无 secret、无外部网络——除 `pyproject.toml` 声明的 `requires-python` 与 `data/` 目录名之外,本仓库没有大小写敏感或路径分隔符相关的平台假设。

```bash
python -m pytest -m "not network and not gpu and not slow" -q --cov=shingan
```

### 测试套件

| 指标 | 值 |
| --- | --- |
| 测试数 | **171 全部通过** |
| 套件耗时 | **3.7 s**(纯测试) |
| 测试文件 | 10 |
| 被跳过的分支 | `network` / `gpu` / `slow` 三个 marker,当前无测试落在其中 |

| 文件 | 测试数 | 覆盖的对象 |
| --- | --- | --- |
| [`tests/test_metrics.py`](tests/test_metrics.py) | 43 | AUC / KS / PR-AUC / ECE / capture,含并列分数的行为与 sklearn 交叉验证 |
| [`tests/test_caveats.py`](tests/test_caveats.py) | 22 | 缺特征诊断、按来源分组、可执行建议、窗口未覆盖年份的披露 |
| [`tests/test_models.py`](tests/test_models.py) | 22 | 三路模型的持久化契约与往返、消融表的完整性 |
| [`tests/test_sec_docs.py`](tests/test_sec_docs.py) | 22 | EDGAR 归档地址拼装、UA 校验、部分写入防护、断点续传 |
| [`tests/test_text_features.py`](tests/test_text_features.py) | 15 | 章节分段、标题识别、空文档与"无负面词"的区别 |
| [`tests/test_report_gates.py`](tests/test_report_gates.py) | 11 | 门禁行的目标/达成/判定三者自洽 |
| [`tests/test_labeling.py`](tests/test_labeling.py) | 9 | 三个标签的事件定义与右边界截断 |
| [`tests/test_splits.py`](tests/test_splits.py) | 9 | purge / embargo、滚动窗口的可用性判定 |
| [`tests/test_builder_text.py`](tests/test_builder_text.py) | 7 | 面板拼接与文本列的接线 |

### 覆盖率:34%,并且不是一个门槛

| 范围 | 语句覆盖 |
| --- | --- |
| **总体** | **34%**(5850 条语句,3591 条未执行) |
| `models/persistence.py` | 100% |
| `features/text.py` | 86% |
| `models/fusion.py` / `models/text_baseline.py` | 79% |
| `models/structured.py` | 73% |
| `eval/metrics.py` | 49% |
| `cli.py` | 26%(发布与 doctor 路径已覆盖) |
| `data/prices.py` / `data/news.py` | 0% |

**这个数字低,而且低得有明确原因。** 覆盖率不是本项目的验收标准——README 里唯一有资格充当门槛的是[评测](docs/05-evaluation.md)第 9 节那七个门禁。补测试的唯一理由是防回归,不是把百分比推高:

- **被覆盖的恰好是"错了看不出来"的那些。** `features/text.py`(86%)与三个模型类(73–79%)之上的测试,针对的是四个真实缺陷:章节标题识别在 1722/2222 份文档上失效而三列特征全零、`average_precision` 对并列分数按行序取整(无信号模型可得 1.0)、`expected_calibration_error` 用 rank 分箱(完美校准反而得 0.5)、模型 `save()` 与 `load()` 的路径契约互相矛盾。这些东西**不会**在 demo 里报错,只会安静地把数字变好看。
- **未覆盖的大多是"要么跑起来、要么连不上"的接线。** `cli.py`、`data/prices.py`、`data/news.py` 是三个 live 端点适配器与命令面。它们的失败模式是 403、超时、schema 变了——靠 mock 断言不了,靠真跑才有意义,而在 CI 里真跑会既贵又不稳。这部分刻意留在 `network` marker 之下。
- **`data/synthetic.py`(15%)是测试夹具,不是被测对象。** 它的正确性由"跑出来的面板恰好有三个标签且事件率落在预期区间"来证明。

等你看到这里,如果想问"为什么不加门槛":加一份门槛,接下来会出现的是一批为了让数字变绿而写的测试,那比没有测试更糟。

### 门禁:7 项中 1 项通过

这是合成数据 demo(`shingan demo`)的实测结果,原始记录在 [`artifacts/_readme_demo/`](artifacts/_readme_demo/)。**这不是性能结论**,理由见[诚实性声明](#诚实性声明):合成数据的信号是生成器植入的,指标只反映实现与设计一致。列在这里是为了证明门禁会**如实报红**。

| 门禁 | 目标 | 实测 | 判定 |
| --- | --- | --- | --- |
| `headline_auc` | > 0.75 | 0.4759 | 未通过 |
| `headline_ks` | > 0.30 | 0.1192 | 未通过 |
| `fusion_gain_pr_auc` | fused PR-AUC > structured-only,且区间不含零 | −0.0952 [−0.0952, −0.0767] | 未通过 |
| `calibration_ece` | < 0.05 | 0.1416 | 未通过 |
| `brier_beats_base_rate` | Brier skill > 0 | −0.0195 | 未通过 |
| `stability_has_multiple_windows` | 窗口数足够 | 14 个配置窗口 | **通过** |
| `stability_enough_usable_windows` | ≥ 3 个窗口同时带标签与分数 | 2/14 | 未通过 |

七项里只有一项背景性检查通过。**"PR-AUC 区间不含零但方向为负"是本项目要回答的核心问题目前的答案:文本轨没有提供增量,融合反而更差。** 这个结论写在门禁里,而不是留在某个被挑出来的数字里。

### 三路消融:文本轨当前没有增量

`shingan demo` 的三路对比,同一测试块、同一标签定义,唯一变量是用了哪条轨。基础率与正样本数一并列出,因为一个 64 行、9 个正例的测试块上的 AUC 不该被单独引用。

| 标签 | 路径 | AUC | KS | PR-AUC | ECE | 基础率 / 正例 |
| --- | --- | --- | --- | --- | --- | --- |
| `default_risk` | structured | 0.5931 | 0.1931 | 0.5317 | 0.0496 | 0.4531 / 29 |
| | text-only | 0.3773 | 0.2335 | 0.3755 | 0.1191 | |
| | fused | 0.4759 | 0.1192 | 0.4365 | 0.1416 | |
| `fraud_risk` | structured | 0.6904 | 0.3568 | 0.2968 | 0.0534 | 0.1719 / 11 |
| | text-only | **0.7684** | **0.5043** | **0.6091** | 0.1026 | |
| | fused | 0.7084 | 0.3756 | 0.4152 | 0.1062 | |
| `tail_risk` | structured | 0.5434 | 0.2323 | 0.1670 | 0.1335 | 0.1406 / 9 |
| | text-only | 0.4404 | 0.3374 | 0.2400 | 0.1403 | |
| | fused | 0.4505 | 0.3152 | 0.1530 | 0.1285 | |

值得记录的观察:`fraud_risk` 上文本轨单独跑出全场最好的 AUC(0.7684),而在另外两个标签上它低于随机。**这个方向的不一致本身就是结果**——在 9–29 个正样本上,三条轨之间的差距没有一个具备统计意义,而融合层在三个标签上全部没有超过 structured-only。诚实的表述是"当前样本量下无法区分",不是"文本有用"或"文本没用"。

### 复现

```bash
python -m shingan demo --out artifacts/_readme_demo    # 合成数据,CPU,无需网络
python -m pytest -m "not network and not gpu and not slow" -q --cov=shingan
```

上面的表格全部来自这两条命令的输出,没有手工转录。`--run-dir` 下同时留下 `.json`(机器可读的记录)与 `.md`(渲染稿),`shingan eval report` 会重跑一遍并逐项比对两者——报告与它自己的载荷不一致时报错,而不是等读者发现。

## 发布到 Hugging Face

发布路径已经写好,但**默认拒绝上传不完整的卡片**。这不是仪式:模型卡的每一个 `{{...}}` 槽位最终都会变成页面上的一句话,而没人核对过的数字一旦发布,就再也收不回来。所以 `--dry-run` 是常规工作流,真实上传会在还有占位符时直接失败。

卡片值有两条注入通道:运行报告自动提供 `run_id` 与 fused 的 AUC;`--values-file` 注入人工核对过的卡片值(数据统计、审计结果、状态声明),**文件值优先于报告值**。生成数据卡值的脚本从面板与标签复核产物里读取,不手抄:

```bash
python scripts/card_values.py    # -> artifacts/stage2/card_values.json
python -m shingan publish hf --run-dir artifacts/stage2 \
    --values-file artifacts/stage2/card_values.json --only dataset --dry-run
```

当前真实状态(`--only` 存在的原因:两张卡的完成时间不同):

```
publish plan
┌──────────────┬───────────────────────────────────┬───────────────────┐
│ artifact     │ destination                       │ placeholders left │
├──────────────┼───────────────────────────────────┼───────────────────┤
│ model card   │ shuurai2000/shingan-qwen3-14b-finrisk │ 62                │
│ dataset card │ shuurai2000/shingan-finrisk-labels    │ 0                 │
└──────────────┴───────────────────────────────────┴───────────────────┘
```

**数据卡今天就可以发布,模型卡不行。** 两者的差距与原因:

- **数据卡:0 占位符,可以发布。** 23 个值全部来自实测产物:面板统计(`n_rows`=2221、`n_companies`=34、`pos_tail`=39)、独立重算(`audit_sample_size`=1778、不一致率 0.00%)、仓库事实(`git_commit`、`changelog_url`)。`default_risk`/`fraud_risk` 的正样本数**没有数可填**——事件源未接入,面板里连标签列都不存在——所以按模板规则如实写 `not measured (event source not connected)`,不估一个数。
- **模型卡:62 个未填槽位,全部被 `train lora` 阻塞**——`epochs`、`bs`、`cuda_version`、`bnb_version`、`energy_kwh`/`co2_kg` 这些训练元数据只有真正跑过 QLoRA 才会有(见 [`docs/04-training.md`](docs/04-training.md) 第 5 节)。在适配器存在之前发布模型卡,等于发布一个不存在的模型的说明书。
- 两处 `changelog_url`/`known_issues_url` 已指向本仓库的 `CHANGELOG.md` 与 issues 页;模型卡还缺 `bench_repo` 与 `zenodo_doi`,等评测页与 DOI 就绪。

### 真正上传前需要做的三件事

1. **把所选卡片填到零占位符。** 数据卡已经清零;模型卡等 LoRA 跑完。填不了的字段**诚实地写"未测量"而不是估一个数**——模板把未填槽位渲染成 `not measured`,这个措辞是刻意选的。
2. **装依赖**:`pip install huggingface_hub`。当前环境未安装。
3. **目标仓库必须已经存在。** `publish hf` 走的是 `HfApi.upload_file`,它**不会**自动创建仓库;repo id 写错时会报缺权限,而不是替你新建一个。

### Token 处理

CLI 只从**环境变量** `HF_TOKEN` 读取 token,不读任何配置文件——这一点写死在 `publish_hf` 里,是有意的。`.env.local` 已经被 `.gitignore` 忽略,但代码**不会**自动加载它,所以需要显式注入:

```bash
# Linux / macOS
export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' .env.local | cut -d= -f2-)
python -m shingan publish hf --run-dir artifacts/<run>
```

```powershell
# Windows PowerShell
$env:HF_TOKEN = (Select-String -Path .env.local -Pattern '^HF_TOKEN=' |
    Select-Object -First 1).Line -replace '^HF_TOKEN=', ''
python -m shingan publish hf --run-dir artifacts/<run>
```

`.env.local` 是 `KEY=value` 单行纯文本,上面的写法与它匹配。**不要**把 token 贴进对话、命令历史或 commit message;`.gitignore` 已经挡住 `.env` / `.env.*` / `*.token`,但那条规则挡不住终端历史。

上游还有一条独立的手动路径:`.github/workflows/publish-hf.yml`(`workflow_dispatch`,输入 `repo_id` / `repo_type` / `path`,默认 `dry_run: true`)。它走 GitHub Secret 里的 `HF_TOKEN`,适合把整个产物目录推上去而不是只推一张卡;使用前需要先在仓库 Settings → Secrets 里配置该 secret。

### 上传前应该先确认的边界

**当前没有可上传的模型权重。** `train lora` 从未运行,`artifacts/` 里只有结构化模型(`.joblib`)与评测记录。所以现在能发布的是数据卡(已清零)与模型卡的**渲染稿**——模型卡在占位符清零之前会被拒绝,而占位符清零的路径只有把训练真的跑完。这个顺序是故意的:先把卡片上声称的东西做出来,再把卡片发出去。

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
