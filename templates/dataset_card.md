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
How to use this template:
  1. Replace every {{...}} placeholder with a measured value; for anything not
     measured, write "not measured" and say why.
  2. Base rates, sample counts and positive counts must be measured values.
     Never estimate a number and present it as measured.
  3. If the data contains synthetic rows, keep the Synthetic Data section and
     state the share.
  4. Delete this comment block and every unfilled placeholder before publishing.
-->

# Shingan FinRisk Labels

The **labelled dataset** of the Shingan (心眼) project: risk labels and derived
features for listed companies at specific `as_of` dates, built to train and
evaluate a dual-track (structured GBDT + text QLoRA) risk model.

**This dataset contains neither raw news text nor the raw price panel.** To
train the text track you must obtain the underlying text yourself from the
original sources (see Licensing). That is a deliberate design decision made to
comply with each source's redistribution terms.

## Dataset Description

| Item | Value |
| --- | --- |
| Project | Shingan (心眼) |
| Data version | `{{data_version}}` |
| Primary key | `(ticker, as_of)` |
| Rows | `{{n_rows}}` |
| Companies | `{{n_companies}}` |
| Date range | `{{date_range}}` |
| Labels | 3 (`default_risk` / `fraud_risk` / `tail_risk`) |
| Contains synthetic rows | `{{contains_synthetic}}` (share `{{synthetic_share}}`) |
| License | Apache-2.0 (covers only the labels and derived features produced by this project; see Licensing) |
| Project documentation | `{{repo_url}}/tree/main/docs` |

### Labels

| Label | Definition | Horizon | Positives | Positive rate |
| --- | --- | --- | --- | --- |
| `default_risk` | Rating downgrade ≥ 2 notches, or entry into bankruptcy/default proceedings | 365 calendar days | `{{pos_default}}` | `{{rate_default}}` |
| `fraud_risk` | Financial restatement, SEC enforcement action, or going-concern/qualified audit opinion | 730 calendar days | `{{pos_fraud}}` | `{{rate_fraud}}` |
| `tail_risk` | Cumulative drawdown below -30% within 30 trading days | 30 trading days | `{{pos_tail}}` | `{{rate_tail}}` |

**Base rate is the single most important property of this dataset.** All three
labels are rare events, with positive rates in the low single digits
(`fraud_risk` is typically the lowest). That dictates the evaluation protocol:
accuracy is meaningless here — the headline metrics must be PR-AUC, KS, and
calibration error. Reporting accuracy on this dataset is reporting a number
that carries no information.

The three labels are **not mutually exclusive**; a row can have several set to
1. They are three independent binary targets, not a softmax multi-class problem.

`liquidity_risk`, `event_driven_risk` and `macro_contagion_risk` are defined in
the project docs but are **not included in this dataset**.

### Fields

| Field group | Fields | Notes |
| --- | --- | --- |
| Identifiers | `id`, `ticker`, `cik`, `company_name`, `sector` | `id` is `{ticker}-{as_of}`, unique across the table |
| Time | `as_of` | The cut-off trading day for the row. Every feature was available at this point |
| Split | `split` | `train` / `valid` / `test` / `purged`. `purged` rows were removed by the purge/embargo rules and are kept for auditability |
| Labels | `labels.{default_risk,fraud_risk,tail_risk}` | 0/1 |
| Label masks | `label_masks.*` | `false` means the label is undecidable for that row (window truncated by the right edge, or the event source does not cover it). Where the mask is `false`, the label value is meaningless |
| Label metadata | `horizon_days.*`, `event_date.*`, `source_of_record.*` | `event_date` and `source_of_record` are for audit only and must **not** be used as model features |
| Forward quantities | `fwd_ret_21d`, `fwd_realized_vol_21d`, `fwd_max_drawdown_30d` | Continuous forward-looking quantities for IC and backtesting. Must **not** be used as model features |
| Structured features | financial-ratio columns (about 13) | See the data dictionary in the project docs |
| Structured features | technical / microstructure columns (about 20) | Same |
| Text counts | news counts, sentiment aggregates, lexicon hits (about 10) | Counts only — no raw text |
| Weight | `sample_weight` | Compensates for negative downsampling so the real base rate can be recovered |
| Metadata | `is_synthetic`, `data_version`, `n_sources` | — |

Full field definitions are in the data dictionary section of
`{{repo_url}}/blob/main/docs/02-data.md`.

## Provenance

| Label / data | Source of record | `source_of_record` value |
| --- | --- | --- |
| `default_risk` — rating downgrade | Rating agency history | `rating_history:<agency>` |
| `default_risk` — bankruptcy/default | Bankruptcy records, 8-K Item 1.03, exchange notices | `bankruptcy_record` / `sec_8k_item_1.03` / `exchange_notice` |
| `fraud_risk` — financial restatement | 8-K Item 4.02, restatement disclosures in filings | `sec_item_4.02_8k` / `filing_restatement_note` |
| `fraud_risk` — SEC enforcement | SEC enforcement list | `sec_enforcement` |
| `fraud_risk` — audit opinion | Text extraction from the 10-K audit report section | `audit_opinion_extracted` |
| `tail_risk` | Adjusted price panel | `price_panel:adjusted_close` |
| Financial ratios | SEC EDGAR inline XBRL | `edgar_xbrl` |
| Technical / microstructure | Price-volume panel | `price_panel` |
| Text counts | FNSPID news + EDGAR full text | `fnspid` / `edgar_fulltext` |

**Proxy event disclosure**: the original definition of `default_risk` depends
on rating agency history, which requires a commercial data licence. In this
build, the status of rating history is: `{{rating_history_status}}`.

- If `available`, labels follow the original definition.
- If `proxy`, publicly verifiable substitute events (e.g. 8-K Item 2.04 trigger
  events, exchange delisting notices) are used instead, with a `proxy:` prefix
  on `source_of_record`. **A proxy event is not the same fact as "downgraded
  ≥ 2 notches"** — the positive set, the base rate and comparability all change.
  Users must be aware of this difference.
- If `unavailable`, `label_masks.default_risk` is `false` for every row and the
  label is unusable.

### As-of discipline

**Every** feature in every row satisfies "publicly available as of `as_of`
(inclusive of that day's close)":

| Data | Availability rule |
| --- | --- |
| 10-K/10-Q/8-K full text | `filing_date <= as_of` |
| inline XBRL numeric facts | `filed <= as_of` (not `end <= as_of`) |
| News | publication timestamp (converted to UTC date) `<= as_of` |
| Price / volume | trading day `<= as_of`, using that day's close and earlier only |
| Events (label sources) | `event_date > as_of` — events never enter any feature |

Explicitly excluded:

- Restatements published after `as_of`. A model at time t only sees the numbers
  as disclosed at the time, even if later shown to be wrong.
- Enforcement actions and audit-opinion changes occurring after `as_of`.
- News published after `as_of`, **including retrospective coverage in
  particular** — a look-back piece that sums up "the company eventually went
  bankrupt" writes the label into the input.
- Anything derived from `event_date` (e.g. `days_to_event`).
- `ticker` is never a feature; `sector` is one-hot or native categorical only,
  **never target-encoded** (target encoding leaks future rows into a fold).

One honest caveat on as-of risk: the discipline is enforced by construction
code and assertions, but **not verified row-by-row by a human** for this
dataset. If you find any row whose features reference information from after
its `as_of`, please open an issue.

## Known Biases

| Bias | Description | Effect | Status |
| --- | --- | --- | --- |
| **Survivorship (S&P 500 membership)** | If the company set is drawn from *today's* S&P 500 constituents, bankrupted, acquired and delisted firms never enter the sample | Systematically lowers the base rate and inflates model performance. The most serious bias here | `{{pit_membership_status}}`. Point-in-time membership is required; while it is `not applied`, this dataset's base rates and metrics are not comparable with other datasets |
| **Restatement look-ahead (vendor data)** | Third-party financial-data vendors routinely backfill restated figures, so the "number as of t" is actually the revised one | Leaks `fraud_risk` information directly | Mitigated by reading XBRL with `filed <= as_of`; if backfilled vendor data had been used, the status would be `{{vendor_lookahead_status}}` |
| **English only** | Training and labelling text is English-language disclosure by US-listed issuers | Not applicable to non-English disclosure or non-US issuers | Design limitation |
| **US-listed only** | Covers US-listed issuers | Not applicable to other jurisdictions' accounting and disclosure regimes | Design limitation |
| **Large-cap skew** | The Stage 2 company set is mostly large-cap leaders | Large caps have markedly lower event rates than the whole market, so the base rate is low; small-cap behaviour is untested | `{{market_cap_scope}}` |
| **Uneven time coverage** | FNSPID news density and EDGAR XBRL coverage vary by year (materially thinner early on) | Cross-period comparison is not valid; metrics must be grouped by year | Metrics reported grouped over time |
| **Rating data gap** | See the proxy event disclosure above | Label semantics may be altered | `{{rating_history_status}}` |
| **Regime-dependent base rate** | The `tail_risk` base rate rises sharply in high-volatility regimes | Base rate is unstable under a single global threshold, affecting calibration | Base rate reported per year |
| **Label noise** | Event determination relies on text extraction (audit opinions, restatement identification) and can misjudge | Adds label noise and lowers the achievable performance ceiling | Independent recomputation of `{{audit_sample_size}}` rows, disagreement rate `{{audit_disagreement_rate}}` |
| **Negative downsampling** | Negatives in the `train` split are downsampled by `neg_pos_ratio` | The training distribution no longer matches the real base rate | Compensated via `sample_weight`; **evaluation must use the un-downsampled `valid`/`test` splits** |

## Synthetic Data

Synthetic share of this dataset: `{{synthetic_share}}`. If the share were
substantial, the following would have to be understood:

- Synthetic data is generated deterministically by `data/synthetic.py` with a
  fixed seed and is fully reproducible.
- It contains one **deliberately planted** latent structure: the text carries a
  risk component invisible to the structured features. Its purpose is to make
  "fused should beat structured-only" a **construct-valid regression check** —
  if the fusion gain on synthetic data were zero or negative, the
  implementation has a bug.
- **That is not a finding.** The fusion gain on synthetic data is an assumption
  baked into the generator and cannot be used to argue that text helps in the
  real world.
- Any metric computed on synthetic data (AUC, KS, PR-AUC, …) only demonstrates
  that the pipeline runs end to end. It must **not** be cited as evidence of
  model capability.
- Synthetic rows have `is_synthetic = true`. Real and synthetic data must
  **never** be mixed in one processed table; the builder rejects a mix.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{{dataset_repo}}", split="train")

# Rare events: look at the base rate, not accuracy
import numpy as np

y = np.array(ds["labels.default_risk"])
print("positive rate:", y.mean())

# Evaluate only rows whose label mask is true, and never let downsampling
# contaminate the evaluation set
mask = np.array(ds["label_masks.default_risk"])
y = y[mask]

# Feature matrix: drop every label and forward-looking column
BANNED = ("label_", "labels.", "fwd_", "event_", "source_of_record")
feature_cols = [c for c in ds.column_names if not c.startswith(BANNED)]
```

Three rules of use:

1. **Never use the `fwd_*` columns as features.** They are the evaluation
   targets (continuous forward quantities); as features they leak immediately.
2. **Never evaluate on downsampled data.** Calibration error and PR-AUC are
   only meaningful at the real base rate.
3. **Never randomise the temporal split.** Panel samples are not independent,
   and label horizons run up to 730 days — split boundaries require purge and
   embargo (see 05-evaluation.md in the project docs).

## Licensing and Attribution

The Apache-2.0 licence on this dataset **covers only the labels and derived
features produced by this project**. It does not cover the underlying content
of any upstream source.

| Upstream source | Licence / terms | Redistribution status in this dataset |
| --- | --- | --- |
| SEC EDGAR | US federal government public information, no copyright | Only ratios and statistics derived from public filings are redistributed; no filing full text |
| FNSPID | Published on Hugging Face by its authors under their dataset card's licence | **No news text is redistributed.** Only aggregate statistics (counts, sentiment mean/std, negative share). Users must obtain the text themselves |
| Price-volume data (yfinance / Stooq) | Each has its own terms; Stooq restricts redistribution | **No raw price panel is redistributed.** Only price-derived technical features and labels |
| Rating history / WRDS (if used) | Commercial subscription; this project has no redistribution licence | No raw records; only derived 0/1 labels and source identifiers |
| `Qwen/Qwen3-14B` | Apache-2.0 | Unrelated to this dataset (the model-side base model) |

**To reproduce the text-track training**, users must obtain FNSPID and EDGAR
data themselves and rebuild the prompts under the project's as-of rules. The
`id` field (`{ticker}-{as_of}`) is the join key for aligning external text.

Citation request:

```bibtex
@misc{shingan_finrisk_labels,
  title        = {Shingan FinRisk Labels: an as-of panel of listed-company risk labels},
  author       = {Shane},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/{{dataset_repo}}}},
  note         = {Research and education only. Not investment advice.}
}
```

If you use this dataset, please also cite FNSPID (if its derived features are
used) and SEC EDGAR as data sources.

## Personal Information

- The subjects of this dataset are **issuing entities (companies)**, not
  natural persons.
- The dataset **contains no** personal information about individuals: no
  executive names, compensation, holdings, or any person-identifiable field.
- Event labels concern company-level disclosure and regulatory actions. Where a
  public record inevitably names individuals (e.g. enforcement actions
  targeting persons), this project does **not** carry those names into derived
  features or labels — only company-level identifiers and event types.
- If you find any person-identifiable information, please open an issue and we
  will remove it.

## Maintenance and Versioning

| Item | Value |
| --- | --- |
| Build script version | `{{git_commit}}` |
| Build date | `{{build_date}}` |
| Upstream data snapshot date | `{{source_snapshot_date}}` |
| Known issues | `{{known_issues_url}}` |
| Changelog | `{{changelog_url}}` |

Versioning rule: any field added or removed, any label definition change, any
change of the date span, or any change to the as-of rules must bump
`data_version` and be recorded in this card. **Silently changing what a field
means is not allowed.**

## Contact

Data errors, as-of violations, or missed biases: please file at
`{{repo_url}}/issues`.
