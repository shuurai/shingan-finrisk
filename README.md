# Shingan 心眼

[![ci](https://github.com/shuurai/shingan-finrisk/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/shuurai/shingan-finrisk/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-171%20passed-brightgreen)
![coverage](https://img.shields.io/badge/coverage-36%25-yellow)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

**Evidence-grounded financial risk modelling.** An open-source proof of concept that asks whether *text* signals add measurable information beyond structured financial and market signals — and refuses to claim an answer until the ablation says so.

> Shingan is an open-source proof of concept for a financial risk model built as two cooperating tracks. **Track A** is a calibrated gradient-boosted model over structured financial and market signals. **Track B** is a QLoRA instruction-tuned language model reading SEC filings and financial news. A thin **fusion** layer combines both and emits evidence-grounded risk assessments for three label families: credit, fraud/misstatement, and tail risk. The design target is ranking quality and calibration under strict point-in-time discipline with purged, embargoed time-series validation — **not** return forecasting.

**Status: POC with a first real-data result, and that result is provisional.** The pipeline runs end to end on **synthetic data**, which proves it is connected and nothing more. Stage 2 has now also been executed once against real SEC and market data — 34 companies, one label (`tail_risk`), 39 positives of which 5 fall in the test block. The numbers below are real measurements on real filings and prices, but the sample is far too small to support a performance claim. The SEC filing *text* corpus has since been downloaded (2,222 primary documents) and the panel rebuilt with live text features, but the three-track evaluation has **not** been re-run on that panel yet — so every number quoted below still comes from the text-free run. Everywhere else, the numbers in the documentation are still **targets or gates**, not achieved values.

- Repository: [`shuurai/shingan-finrisk`](https://github.com/shuurai/shingan-finrisk)
- Dataset card: [`shuurai2000/shingan-finrisk-labels`](https://huggingface.co/datasets/shuurai2000/shingan-finrisk-labels) (published)
- Maintainer: Shane (GitHub [`shuurai`](https://github.com/shuurai))
- License: Apache-2.0

---

## 中文简介

**Shingan（心眼）** 是一个开源的金融风险建模概念验证，核心问题是：**文本信号（财报、新闻）能否在结构化财务与市场信号之外提供可度量的增量信息？** 项目拒绝在消融实验给出答案之前宣称答案。

- **双轨架构**：Track A 是结构化信号上的校准梯度提升模型；Track B 是读取 SEC 文件与新闻的 QLoRA 指令微调模型；一层薄 fusion 融合两路。目标是在严格 point-in-time 纪律与 purge/embargo 时序验证下的排序质量与校准，**不是收益预测**。
- **三个标签**：`default_risk`（评级下调/破产，365 天）、`fraud_risk`（重述/执法/非标审计意见，730 天）、`tail_risk`（30 交易日回撤劣于 −30%），均在同一套 as-of/无前视纪律下计算。
- **硬性评测要求**：任何声称有效的模型必须同时给出 structured-only、text-only、fused 三路结果加无模型基线。
- **当前状态**：流水线在合成数据上端到端可跑；Stage 2 已在真实数据（34 家公司、`tail_risk`、2221 行面板）上跑过一次——structured AUC 0.7686 / KS 0.5873 / PR-AUC 0.0258，但 test 块仅 5 个正样本，**不构成性能结论**；三路消融显示文本轨当前没有可度量的增量。QLoRA 训练正在进行中。
- **已发布**：数据卡 [`shuurai2000/shingan-finrisk-labels`](https://huggingface.co/datasets/shuurai2000/shingan-finrisk-labels)（英文，0 占位符，全部为实测值）。
- 设计文档在 [`docs/`](docs/)（中文），标注数据集卡模板在 [`templates/`](templates/)。

---

## The core claim: why two tracks

The project's original idea was "concatenate 20 years of filings + news + trading signals into one large text and train a LoRA to predict risk". This project **explicitly rejects that approach**, for two reasons:

1. **Different signal shapes need different inductive biases.** Financial ratios and price-volume features are dimensional, approximately stationary, large-sample numeric panels — gradient-boosted trees are stable, calibratable and attributable on exactly this kind of data. Serialising numbers into tokens for an LLM hands the job to a high-variance, sample-inefficient function approximator to do what a tree model does better.
2. **Verifiability depends on separation.** The two-track design turns "what does text contribute" into an ablatable experimental question: structured-only and text-only baselines exist independently, and fused only demonstrates text-track value if it beats structured-only significantly on PR-AUC. A single LoRA mixes both signals into one indecomposable weight matrix and cannot answer the question.

Hence the project's hard evaluation requirement: **any model claiming effectiveness must report structured-only, text-only and fused results side by side, plus a no-model baseline.**

## The three labels

All three labels are computed under the same as-of / no-lookahead discipline. Samples whose right edge is truncated (`as_of + horizon` beyond the end of data) are marked `label_mask=false` and are **never** counted as negatives.

| Label | Event definition | Horizon |
| --- | --- | --- |
| `default_risk` | Rating downgrade ≥ 2 notches, or bankruptcy | 365 calendar days |
| `fraud_risk` | Restatement / enforcement action / non-standard audit opinion | 730 calendar days |
| `tail_risk` | Peak-to-trough drawdown worse than −30% | 30 trading days |

`liquidity_risk`, `event_driven_risk` and `macro_contagion_risk` are defined but **out of POC scope** and are not implemented.

## Quick start

The `demo` and the test suite run with no GPU and no network. Full training needs an NVIDIA GPU (the dev machine is Windows 11 + RTX 5090 32 GB).

```powershell
# Windows 11 (supported training path)
powershell -ExecutionPolicy ByPass -File scripts\bootstrap.ps1
powershell -ExecutionPolicy ByPass -File scripts\bootstrap.ps1 -Train   # before training the LoRA
powershell -ExecutionPolicy ByPass -File scripts\run_poc.ps1
```

```bash
# macOS / Linux
bash scripts/bootstrap.sh
bash scripts/run_poc.sh
```

`run_poc` executes `data build` → `train structured` → `eval run` → `eval report` in sequence. If GNU Make is installed (it is not on Windows by default), the equivalent is:

```bash
make bootstrap
make build train-structured eval report
make doctor          # environment self-check
make lint typecheck test
```

Shortest CPU end-to-end check on its own:

```bash
python -m shingan demo                      # synthetic data, CPU, markdown report
python -m shingan doctor                    # print environment and dependency versions
```

### About the default data config

`run_poc` defaults to `configs/data/poc_largecap10.yaml` (10 large caps, full time span). Under that config **only one of the three labels gets evaluated** — 10 companies, a 3-year validation block and a 1-year horizon do not accumulate enough downgrade events to calibrate against. The report does not leave the cell blank: the Caveats section states the reason and the positive count.

To run all three labels end to end, use the demo overlay (its event rates are higher than the labelled docs' real base rates, which is why it is a separate file rather than the default):

```bash
bash scripts/run_poc.sh --data-config configs/data/demo.yaml --out artifacts/demo
bash scripts/run_poc.sh --labels tail_risk        # evaluate one label only
```

More detail in [`scripts/README.md`](scripts/README.md).

## Repository layout

```
src/shingan/        the package: CLI, data, features, models, evaluation, reporting
  cli.py            Typer command surface
  data/             synthetic generator + EDGAR / price / news adapters
  features/         structured and text features
  models/           structured (GBDT), text_baseline (TF-IDF), lora (QLoRA), fusion
  eval/             splits, purge/embargo, metrics, stability, stress tests, reporting
configs/            data / train / eval YAML configs
scripts/            platform wrapper scripts (bootstrap, run_poc) and docs
docs/               design docs (architecture, data, labelling, training, evaluation, Windows, roadmap, naming) — in Chinese
templates/          Hugging Face model card / dataset card templates
notebooks/          exploratory notebooks (outputs stripped by default)
tests/              pytest suite
```

## Command surface

| Command | Purpose |
| --- | --- |
| `shingan doctor` | environment and dependency self-check |
| `shingan demo` | synthetic-data CPU end-to-end, markdown report |
| `shingan data synth` | deterministic synthetic panel |
| `shingan data build` | build the feature panel |
| `shingan data sft` | export the instruction-tuning dataset |
| `shingan train structured` | train and calibrate the structured track |
| `shingan train lora` | QLoRA fine-tune the text track (needs GPU + `train` extra) |
| `shingan eval run` | run evaluation, write JSON records |
| `shingan eval report` | render a report from JSON records, cross-check against stored metrics |
| `shingan publish hf` | push cards to the Hugging Face Hub |
| `shingan version` | print version |

## Testing and coverage

> `ci` is the only auto-updating badge; the other four are static and reproduced by the commands below. **The gate table reports red honestly, it is not all-green** — see the end of this section for why.

**CI runs Linux (`ubuntu-latest`), Python 3.11, a single job.** No matrix, no secrets, no external network. Apart from the `requires-python` declared in `pyproject.toml` and the `data/` directory names, this repo makes no case-sensitivity or path-separator platform assumptions.

```bash
python -m pytest -m "not network and not gpu and not slow" -q --cov=shingan
```

### Test suite

| Metric | Value |
| --- | --- |
| Tests | **171 passing** |
| Suite time | **~4 s** (tests only) |
| Test files | 10 |
| Skipped branches | the `network` / `gpu` / `slow` markers — no tests currently live under them |

| File | Tests | Covers |
| --- | --- | --- |
| [`tests/test_metrics.py`](tests/test_metrics.py) | 43 | AUC / KS / PR-AUC / ECE / capture, tied-score behaviour, sklearn cross-check |
| [`tests/test_caveats.py`](tests/test_caveats.py) | 22 | missing-feature diagnostics, grouping by source, actionable advice, disclosure of years a window does not cover |
| [`tests/test_models.py`](tests/test_models.py) | 22 | persistence contracts and round-trips for the three tracks, ablation-table completeness |
| [`tests/test_sec_docs.py`](tests/test_sec_docs.py) | 22 | EDGAR archive URL assembly, UA validation, partial-write protection, resume |
| [`tests/test_text_features.py`](tests/test_text_features.py) | 15 | section splitting, heading recognition, empty document vs "no negative words" |
| [`tests/test_publish_hf.py`](tests/test_publish_hf.py) | 11 | values injection, selective refusal, template-link integrity |
| [`tests/test_report_gates.py`](tests/test_report_gates.py) | 11 | a gate row's target/achieved/verdict staying mutually consistent |
| [`tests/test_labeling.py`](tests/test_labeling.py) | 9 | the three label event definitions and right-edge truncation |
| [`tests/test_splits.py`](tests/test_splits.py) | 9 | purge / embargo, rolling-window usability |
| [`tests/test_builder_text.py`](tests/test_builder_text.py) | 7 | panel assembly and text-column wiring |

### Coverage: 36%, and it is not a threshold

| Scope | Statement coverage |
| --- | --- |
| **Total** | **36%** (5,888 statements, 3,449 missed) |
| `models/persistence.py` | 100% |
| `features/text.py` | 86% |
| `models/fusion.py` / `models/text_baseline.py` | 79% |
| `models/structured.py` | 72% |
| `eval/metrics.py` | 49% |
| `cli.py` | 26% (publish and doctor paths covered) |
| `data/prices.py` / `data/news.py` | 0% |

**The number is low, and low for a stated reason.** Coverage is not this project's acceptance criterion — the only thing qualified to act as a gate is the seven gates in section 9 of the [evaluation doc](docs/05-evaluation.md). The only legitimate reason to add a test is regression protection, not pushing a percentage up:

- **What is covered is exactly the "wrong but silent" code.** The tests over `features/text.py` (86%) and the three model classes (72–79%) target four real defects: section-heading recognition failing on 1,722 of 2,222 documents while all three text features silently went to zero; `average_precision` resolving tied scores by row order (a no-signal model could score 1.0); `expected_calibration_error` using rank binning (perfect calibration scoring 0.5); and model `save()`/`load()` path contracts contradicting each other. None of these **error out** in the demo — they just quietly make the numbers look better.
- **What is uncovered is mostly "either runs or doesn't connect" wiring.** `cli.py`, `data/prices.py` and `data/news.py` are live-endpoint adapters and the command surface. Their failure modes are 403s, timeouts and schema changes — not assertable with mocks, only meaningful when really run, and really running them in CI is both expensive and flaky. They are deliberately kept under the `network` marker.
- **`data/synthetic.py` (15%) is a test fixture, not a test target.** Its correctness is demonstrated by the panel it produces having exactly three labels with event rates in the expected ranges.

If you got this far and want to ask "why no threshold": add a threshold, and the next thing that appears is a batch of tests written to make the number green — which is worse than no tests.

### Gates: 1 of 7 passing

These are the measured results of the synthetic-data demo (`shingan demo`), with raw records in [`artifacts/_readme_demo/`](artifacts/_readme_demo/). **This is not a performance claim** — see the [honesty statement](#honesty-statement): the signal in synthetic data is planted by the generator, so the metrics only reflect implementation matching design. They are listed here to prove the gates **report red honestly**.

| Gate | Target | Measured | Verdict |
| --- | --- | --- | --- |
| `headline_auc` | > 0.75 | 0.4759 | fail |
| `headline_ks` | > 0.30 | 0.1192 | fail |
| `fusion_gain_pr_auc` | fused PR-AUC > structured-only, interval excluding zero | −0.0952 [−0.0952, −0.0767] | fail |
| `calibration_ece` | < 0.05 | 0.1416 | fail |
| `brier_beats_base_rate` | Brier skill > 0 | −0.0195 | fail |
| `stability_has_multiple_windows` | enough windows configured | 14 configured windows | **pass** |
| `stability_enough_usable_windows` | ≥ 3 windows with both labels and scores | 2/14 | fail |

Only one background check passes out of seven. **"The PR-AUC interval excludes zero but points negative" is the current answer to the project's core question: the text track provides no increment, and fusion makes things worse.** That conclusion lives in the gate table, not in some cherry-picked number.

### Three-track ablation: the text track currently adds nothing

The three-track comparison from `shingan demo` — same test block, same label definitions, the only variable being which track is used. Base rate and positive counts are listed alongside, because an AUC on a 64-row, 9-positive test block should never be quoted alone.

| Label | Track | AUC | KS | PR-AUC | ECE | Base rate / positives |
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

An observation worth recording: on `fraud_risk` the text track alone posts the best AUC of the whole table (0.7684), while on the other two labels it is below random. **That directional inconsistency is itself the result** — on 9–29 positives, no difference between the three tracks is statistically significant, and the fusion layer fails to beat structured-only on all three labels. The honest statement is "indistinguishable at the current sample size", not "text helps" or "text doesn't help".

### Reproducing

```bash
python -m shingan demo --out artifacts/_readme_demo    # synthetic data, CPU, no network
python -m pytest -m "not network and not gpu and not slow" -q --cov=shingan
```

Every table above comes from the output of these two commands — nothing was hand-transcribed. The `--out`/`--run-dir` directory keeps both `.json` (machine-readable records) and `.md` (rendered report), and `shingan eval report` re-renders and diffs the two — a report inconsistent with its own payload is an error, not something a reader discovers.

## Publishing to Hugging Face

The publishing path is built and **refuses by default to upload an incomplete card**. That is not ceremony: every `{{...}}` slot in the model card eventually becomes a sentence on a public page, and a number nobody checked, once published, cannot be retracted. So `--dry-run` is the normal workflow, and a real upload fails while placeholders remain.

Card values have two injection channels: the run report automatically supplies `run_id` and the fused AUC; `--values-file` injects human-verified card values (dataset statistics, audit results, status declarations), **file values taking precedence over report values**. The script that generates the dataset-card values reads from the panel and label-review artifacts — nothing hand-copied:

```bash
python scripts/card_values.py    # -> artifacts/stage2/card_values.json
python -m shingan publish hf --run-dir artifacts/stage2 \
    --values-file artifacts/stage2/card_values.json --only dataset --dry-run
```

Current real state (the reason `--only` exists: the two cards complete at different times):

| Artifact | Destination | Placeholders | State |
| --- | --- | --- | --- |
| dataset card | [`shuurai2000/shingan-finrisk-labels`](https://huggingface.co/datasets/shuurai2000/shingan-finrisk-labels) | 0 | **published (English)** |
| model card | `shuurai2000/shingan-qwen3-14b-finrisk` | 62 | blocked on `train lora` |

- **Dataset card: 0 placeholders, published.** All 23 values come from measured artifacts: panel statistics (`n_rows`=2221, `n_companies`=34, `pos_tail`=39), independent recomputation (`audit_sample_size`=1778, disagreement 0.00%), repo facts (`git_commit`, `changelog_url`). The `default_risk`/`fraud_risk` positive counts have **no number to fill** — the event sources are not connected and the label columns do not exist in the panel — so the card honestly reads `not measured (event source not connected)` rather than estimating.
- **Model card: 62 unfilled slots, all blocked on `train lora`** — `epochs`, `bs`, `cuda_version`, `bnb_version`, `energy_kwh`/`co2_kg` exist only after a real QLoRA run (see section 5 of [`docs/04-training.md`](docs/04-training.md)). The first training run is currently in progress; until an adapter completes, publishing the model card would be a manual for a model that does not exist.

### Before any real upload

1. **Fill the selected card to zero placeholders.** The dataset card is done; the model card waits for LoRA. Fields that cannot be filled are written **honestly as "not measured" rather than estimated** — the template renders unfilled slots as `not measured`, and that wording is deliberate.
2. **Install the dependency**: `pip install huggingface_hub`.
3. **The target repo must already exist.** `publish hf` uses `HfApi.upload_file`, which does **not** auto-create repos; a wrong repo id surfaces as a permission error, not a new repo.

### Token handling

The CLI reads the token only from the **environment variable** `HF_TOKEN`, never from a config file — hard-coded in `publish_hf`, deliberately. `.env.local` is gitignored, but the code does **not** auto-load it, so inject it explicitly:

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

`.env.local` is single-line `KEY=value` plain text, which both snippets match. **Do not** paste the token into chats, shell history or commit messages; `.gitignore` blocks `.env` / `.env.*` / `*.token`, but no gitignore rule blocks terminal history.

There is also a separate manual path upstream: `.github/workflows/publish-hf.yml` (`workflow_dispatch`, inputs `repo_id` / `repo_type` / `path`, default `dry_run: true`). It uses an `HF_TOKEN` GitHub secret and suits pushing a whole artifact directory rather than one card; configure the secret in Settings → Secrets first.

## Status table

Status vocabulary:

- `Implemented · verified` — the code exists and has run end-to-end on real inputs with output artifacts.
- `Implemented · not verified` — the code exists but has **not yet** been executed against real data or live endpoints.
- `Interface defined · unverified` — the live-endpoint adapter is written; its success path has never been called.
- `Designed · not implemented` — the design is fixed; the code is not written.
- `Out of POC scope` — deliberately not done.

| Component | Status |
| --- | --- |
| Package layout / `pyproject.toml` / `src/shingan/` module split | Implemented · verified |
| CLI command surface | Implemented · verified |
| Synthetic data generator | Implemented · verified |
| `shingan demo` (CPU end-to-end with report) | Implemented · verified |
| as-of / leakage assertions | Implemented · verified |
| Time-series splits + purge/embargo | Implemented · verified |
| Structured track (with calibration) | Implemented · verified |
| Text baseline (TF-IDF class) | Implemented · verified |
| Evaluation metrics and report rendering | Implemented · verified |
| SEC EDGAR client | Implemented · verified (submissions, companyfacts, full-text search and `/Archives` bodies all exercised on real data; the body corpus is ingested, see below) |
| Price data adapter (yfinance / Stooq) | Implemented · verified (yfinance returned daily bars for 31/34 tickers) |
| News adapter (FNSPID, offline dataset) | Interface defined · unverified |
| QLoRA training | **First run in progress** — no completed adapter yet |
| Fusion layer | Implemented · verified (gain over structured-only on real data is **negative**) |
| **Real-data result (Stage 2 / `tail_risk` / 34 companies / 2,221-row panel)** | **structured AUC 0.7686, KS 0.5873, PR-AUC 0.0258; text baseline 0.4074; fused 0.6554. Only 5 test positives — not a performance claim** |
| Label review (all 39 `tail_risk` positives) | Implemented · verified (independent recomputation, disagreement 0.00%) |
| Real event labelling for the three in-scope labels | Designed · not implemented (`default_risk` / `fraud_risk` event sources not connected) |
| `publish hf` | Implemented · verified for the dataset card (published live); model card pending training |
| `liquidity_risk` / `event_driven_risk` / `macro_contagion_risk` | Out of POC scope |

### Honesty statement

The synthetic-data demo in this repo proves exactly one thing: **the pipeline runs** — data generation, as-of checks, feature construction, both track trainings, fusion, splitting, metric computation and report rendering are all wired together. It does **not** prove the models have real predictive power. The signal in the synthetic data is planted by the generator; good metrics only reflect the implementation matching the design.

Stage 2 has run once on real data; artifacts are in `artifacts/stage2/`. When reading those numbers, note the following — not boilerplate disclaimers, but the actual boundaries of the current result:

- **The test block has 5 positives.** AUC 0.7686 rests on 5 positive examples; swapping any one would move it materially. Per section 10 of the [evaluation doc](docs/05-evaluation.md) this triggers F8: it is not a performance claim, only evidence that the pipeline can compute interval-carrying metrics (or state clearly why it cannot) on real data.
- **The fusion layer adds nothing.** Fused PR-AUC is 0.0075 *below* structured-only, with an interval crossing zero. The text track currently provides no measurable increment — and that is exactly the question this project set out to answer. The current answer is "no", not "yes".
- **SEC filing bodies: ingested, but the evaluation has not been re-run on the new panel.** During the Stage 2 run, `www.sec.gov/Archives` returned 403 ("Undeclared Automated Tool") to every User-Agent from this network. The real cause turned out to be not the UA's shape but the domain of the contact e-mail inside it: plain domains like `research@shingan.dev` got 200, while `a@github.com`, browser UAs and `curl/8.4.0` all got 403. The downloader `scripts/fetch_sec_docs.py` (3-way concurrency, disk cache, resume, atomic writes) then retrieved all 2,222 bodies (0 failures, ~10 GB, 56 minutes) and `filings.parquet` was rebuilt from cache. The panel has been rebuilt with real text: `risk_factor_token_share` is non-zero for 1,774 of 2,221 rows and `neg_kw_density_mdna` for 1,372 (before the fix both columns were **all zero**). But 15 of the 49 configured features still have zero coverage and were dropped at fit time, and **the three-track evaluation has not been re-run** — so the fused comparison in the table above is still the text-free comparison. Availability will change again; any fetcher must carry a disk cache and resume.
- **`default_risk` and `fraud_risk` are not implemented.** Their real event sources (rating history, enforcement actions) are not connected; Stage 2 evaluated `tail_risk` only.
- **The price table is missing MRO / WBA / X.** These three have filings but no price series; their rows are now correctly marked unobservable (see section 5 of `artifacts/stage2/label_review.md`: before the fix they were counted as negatives, which had inflated the structured AUC).

Apart from the Stage 2 numbers listed here, every other number in this documentation set should be read as a **target**, not an achievement.

## Documentation

The design documents are written in Chinese; each is linked with an English gloss.

| Document | Contents |
| --- | --- |
| [docs/index.md](docs/index.md) | documentation map, full status table, glossary entry point |
| [01 Architecture](docs/01-architecture.md) | the two-track architecture and rationale, boundary input contracts, evidence-grounded output contract, explicit non-goals |
| [02 Data](docs/02-data.md) | the three data legs and real sources, as-of/no-lookahead discipline, dedup and alignment, processed-panel data dictionary |
| [03 Labelling](docs/03-labeling.md) | risk taxonomy, exact horizons and thresholds for the three labels, JSONL schema, base-rate realism |
| [04 Training](docs/04-training.md) | structured-track training and calibration, base-model selection, QLoRA config, VRAM budget, Windows/Blackwell recipe |
| [05 Evaluation](docs/05-evaluation.md) | metrics and thresholds, purged walk-forward + embargo, stress tests, baseline ablation, falsification conditions |
| [06 Windows setup](docs/06-windows-setup.md) | step-by-step Windows 11 + RTX 5090 setup, cu128 install, troubleshooting table |
| [07 Roadmap](docs/07-roadmap.md) | Stage 0–5 plan and promotion conditions |
| [08 Naming decisions](docs/08-naming.md) | candidate-name comparison, naming family, rename procedure |
| [ADR](docs/adr/) | architecture and naming decision records |
| [CONTRIBUTING.md](CONTRIBUTING.md) | dev environment, commit conventions, test requirements |

## Citation

If you use this project's code, models or label dataset, please cite it via [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
