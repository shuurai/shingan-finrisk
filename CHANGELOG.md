# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, the public Python API may change between minor releases.
The dataset schema and the label definitions are versioned separately via the
`data_version` field written into every processed dataset.

## [Unreleased]

### Changed
- Documentation is self-contained. `docs/` no longer cites the internal design
  notes that preceded the project: every statement now carries its own reason,
  and cross-references point at this repository's own documents. Section 4 of
  `docs/05-evaluation.md` is reframed from "corrections to a draft script" to
  six implementation pitfalls, which is what it always described.
- The internal design notes are excluded from the repository (`.gitignore`), so
  a checkout exposes the official setup and files only. Nothing in the shipped
  code or documentation depends on them.

### Planned
- Stage 2 real-data POC on a 5-10 company universe (see `docs/07-roadmap.md`).
- SEC enforcement-action and Item 4.02 restatement parsers to replace the
  proxy events currently used for `fraud_risk`.
- Point-in-time S&P 500 membership, which is a hard blocker for Stage 4.
- Released QLoRA adapter and label dataset on the Hugging Face Hub.

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

[Unreleased]: https://github.com/shuurai/shingan/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shuurai/shingan/releases/tag/v0.1.0
