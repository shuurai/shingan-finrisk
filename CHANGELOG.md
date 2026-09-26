# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, the public Python API may change between minor releases.
The dataset schema and the label definitions are versioned separately via the
`data_version` field written into every processed dataset.

## [Unreleased]

### Added
- `shingan publish hf --data-file` uploads data files into the dataset
  repository alongside the card (repeatable). Every file is pre-flighted
  locally before anything reaches the network: a parquet whose row count
  contradicts the card's `n_rows` refuses the whole publish. The labelled
  panel is now published to `shuurai2000/shingan-finrisk-labels` and verified
  end to end with `load_dataset` (2,221 rows, tail_risk 39 positives in 1,778
  mask-true rows — matching the card).
- The dataset card's label inventory is now a rendered value
  (`{{label_columns}}`, derived from the panel's actual label columns) instead
  of a hardcoded "3 labels"; the fields table and the usage example name the
  real flat columns (`label_tail_risk`, `label_mask_tail_risk`) rather than a
  hierarchical naming the panel never used.
- `shingan eval lora` accepts `--load-in-4bit/--no-load-in-4bit`, overriding the
  config default. Unquantised bf16 loading decodes several times faster than
  bitsandbytes 4-bit on Blackwell (the 4-bit decode kernels are unoptimised
  there); the resolved value is recorded in the artifact's `quantization` field,
  and scoring an adapter arm on a different substrate from its training config
  prints a warning — the delta is then not a paired measurement.
- `scripts/placebo_corpus.py` writes a deterministic `manifest.json` (seed,
  source and output SHA-256s, per-split counts) into the placebo corpus
  directory, so a placebo training run's `run.json` embeds the same provenance
  chain as a real one. Rebuilding with the same seed is byte-identical; the
  already-produced placebo `run.json` keeps its "no manifest" note rather than
  being rewritten.
- Data provenance that travels with the artifacts: `sft/manifest.json` and the
  training `run.json` now carry a `data` block (source set, `is_synthetic`,
  panel shape, raw-table SHA-256s, and — for training — the hashes of the SFT
  files plus the embedded SFT manifest). The model card template no longer
  asserts a source list in prose; its provenance slots must be filled from the
  run's own record.
- A citation audit on every `eval lora` arm: each `source_ref` a parsed
  assessment emits is checked against the documents actually rendered into
  *that row's* prompt. It is a disclosure, not a gate — unresolved citations
  never drop rows — reported per arm in `lora_eval.json` and per row in every
  prediction record.

### Planned
- Split-geometry revision so 2020 sits inside the evaluation blocks, then
  positive-sample expansion — in that order: the bottleneck is supervision
  density plus split geometry, not data provenance.
- Model card publication, now that the provenance fields exist in the template.
- SEC enforcement-action and Item 4.02 restatement parsers to replace the
  proxy events currently used for `fraud_risk`.
- Point-in-time S&P 500 membership, which is a hard blocker for Stage 4.

## [0.2.0] - 2026-09-23

Second release. The pipeline has run once against real SEC and market data,
the LoRA track has an in-distribution scoring path with a paired zero-shot
control, and the instruction-contract gap that made the first zero-shot
attempt unmeasurable is fixed and pinned by tests. The real-data result rests
on 5 test positives: it is a measurement of the pipeline, not a performance
claim, and every table that carries it says so.

### Added
- **Real-data Stage 2.** A 34-company universe and a 2,221-row panel, with the
  SEC filing body corpus (2,222 primary documents, ~10 GB) fetched by a
  caching, resuming downloader after a `403 Undeclared Automated Tool` whose
  root cause turned out to be the contact e-mail's domain rather than the
  User-Agent's shape. First three-track comparison on real data —
  `structured` 0.6294, `structured_matched` 0.7497, `text_baseline` 0.3281,
  `fused` 0.6924 on 492 test rows holding 5 positives — published with the
  sample-size statement attached, because a number that cannot support a
  performance claim is still a number that needs its boundary printed next
  to it.
- **Same-information control `structured_matched`**: the same estimator
  restricted to the twelve structured signals the text prompt actually
  carries. Subtracting text-only from the full-feature `structured` mixes
  information sets with model classes, so it looks like an answer without
  being one; `text-only − structured_matched` is the subtraction that means
  something. Fourth row of the standard comparison as of this release.
- **`shingan eval lora`** with three `--mode` values (`adapter` default,
  `zero_shot`, `both`). The two arms run in one process, on one set of base
  weights, one tokenizer and one prompt set, because the difference between
  them is the measurement; `--verify-prompts` rebuilds the training prompts
  byte-identically (260/260 user turns and 782/782 system turns) before any
  scoring happens.
- **Trainer-argument boundary.** The config→trainer mapping is a pure
  function, validated against the installed trainer's signature *before* the
  model loads — `transformers` v5 removed `warmup_ratio`, and the previous
  failure mode was a `TypeError` after a 25 GB download. Removed keys are
  translated at the boundary (`warmup_steps`), and `run.json` records the
  translation plus the six training-library versions.
- **Instruction-contract tests** (`tests/test_instruction_contract.py`): every
  closed set the prompt requires is compared for equality against the schema
  enums in both directions; the `source_ref` examples are pinned to what the
  renderers actually emit; the truncation budget's overhead reserve is pinned
  by an inequality, with a test proving the previous reserve silently
  overflowed.
- **HF publisher values channel** (`--values-file`) and a live-published
  dataset card. `--only` lets the dataset card ship while the model card
  waits for an evaluation section worth publishing.
- `ModelIdentity` on every artifact: base model, tokenizer source and adapter
  weights sha256 travel with the numbers.

### Changed
- Text prompts interpolate the label being asked (`text_inputs(..., label)`).
  Previously every label's prompt asked about `default_risk` over 365 days, so
  the TF-IDF baseline was scored on `tail_risk` while reading a `default_risk`
  question (`tail_risk` text-only PR-AUC moved 0.2400 → 0.2956).
- Comparison tables print `ks_direction` beside `ks`. An inverted ranking
  (AUC 0.3281) with a direction-agnostic KS (0.5692) read as the strongest
  separation in the table.
- Truncation budget `chars_budget_for_seq_length(4096)` is 12,945 characters:
  the overhead reserve rose 1,500 → 1,800 because the expanded instruction
  contract had quietly overflowed it — a quiet overflow, not an error, which
  is why it is now pinned by a test rather than by a comment.
- Documentation is self-contained and English-primary. The internal design
  notes that preceded the repository are excluded, every statement carries
  its own reason, and every number carries its sample size.
- Test suite: 286 tests across 16 files (~6 s), coverage 44% (`prompts.py`
  73%). CI remains Linux-only, single job.

### Notes
- **The 14B adapter is trained on synthetic data only**, and its
  in-distribution output is a constant (AUC exactly 0.5000; 64 rows, 1
  distinct score). With the contract fixed, the paired difference
  `lora − zero_shot` is −0.0172 AUC with an interval crossing zero: no
  measurable positive increment from fine-tuning on this sample, and
  fine-tuning turned a non-degenerate ranking into a constant. That is
  evidence the training stack works on 32 GB, not evidence of predictive
  power, and the adapter artifact says so on its face.
- **74% of the real panel's 39 observable `tail_risk` positives fall in the
  split gap** — the whole of 2020 sits between the validation and test
  blocks. The walk-forward configuration covers that year, so this is a
  split-geometry choice, and revising it is the next work item, ahead of
  expanding the stock pool.


## [0.1.0] - 2026-09-21

First public scaffold. This release establishes the structure, the evaluation
contract and a runnable offline pipeline. It does **not** yet contain real-data
results, and no claim of predictive skill on real companies is made.

### Added
- Project structure, packaging (`hatchling` + `pyproject.toml`) and a
  CPU-only core dependency set so the pipeline runs without a GPU or network.
- `shingan` CLI with `doctor`, `data synth`, `data build`, `data sft`,
  `train structured`, `train lora`, `eval run`, `eval report`, `demo`,
  `publish hf` and `version`.
- Dual-track architecture: a calibrated gradient-boosted structured track, a
  text track, and an explicit fusion layer — see
  `docs/adr/0003-dual-track-over-single-lora.md`.
- Risk label definitions for `default_risk`, `fraud_risk` and `tail_risk`, with
  explicit `label_mask_*` observability flags so a sample whose label window has
  not finished is never silently treated as a negative.
- Point-in-time discipline: as-of joins in the builder, a `leakage.py` guard
  with a column-name blacklist (`label_` / `fwd_` / `event_`), timestamp
  monotonicity checks and a high-correlation heuristic.
- Purged walk-forward cross-validation with an embargo
  (`eval/splits.py`), configured by `purge_days` / `embargo_days` /
  `test_block_years` / `step_years`, plus per-label split support.
- Evaluation suite: AUC, KS, PR-AUC and its random baseline, F-beta, CAP/AR,
  risk-decile monotonicity, IC/ICIR (against a continuous forward quantity,
  never a binary label), ECE / MCE / Brier with reliability curves, per-date
  cross-sectional quantile backtests with Sharpe / Sortino / Calmar / max
  drawdown, PSI/CSI drift, and a markdown report generator.
- Explicit validation-fold-only calibration (isotonic or Platt) and an
  explicit validation-fold-only fusion stacker, avoiding the deprecated
  `prefit` calibration API.
- Deterministic synthetic data generator with a controlled latent structure
  that puts signal in the text that the structured features cannot observe, so
  the fusion comparison is a meaningful regression test rather than a
  tautology. Synthetic rows are always flagged `is_synthetic = true`.
- Test suite covering known-value metrics, split/embargo correctness, label
  window boundaries, feature look-ahead, configuration loading, a static
  Windows-compatibility scan and a full end-to-end demo run.
- Documentation set under `docs/`, including the consolidated design docs, ADRs,
  a Windows + RTX 5090 setup guide, and Hugging Face model/dataset card templates.

### Notes
- The SEC EDGAR, news and price adapters are thin, unvalidated adapters. They
  implement request construction, pagination, rate limiting and field mapping,
  but have not been exercised against live endpoints. Treat them as interfaces,
  not as working data sources.
- The highest-fidelity local configuration this was designed for is Windows 11
  with an RTX 5090 (32 GB, sm_120), which requires PyTorch built for CUDA 12.8
  and a bitsandbytes build from the 12.8-12.9 line.

[Unreleased]: https://github.com/shuurai/shingan-finrisk/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shuurai/shingan-finrisk/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shuurai/shingan-finrisk/releases/tag/v0.1.0
