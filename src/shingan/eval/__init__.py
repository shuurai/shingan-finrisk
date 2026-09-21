"""Evaluation: splits, metrics, backtest, stability and reporting.

The four modules form a pipeline and are meant to be read in that order:

1. :mod:`shingan.eval.splits` — decides **which rows may be used for what**. Purged
   walk-forward with an embargo, plus the rolling time buckets. Every other module
   here takes the split as given; none of them re-derives it.
2. :mod:`shingan.eval.metrics` — ranking, calibration and significance. Degenerate
   folds return NaN rather than raising, because a walk-forward run over dozens of
   folds will contain folds with four positives and those must be reportable.
3. :mod:`shingan.eval.backtest` — cross-sectional quantile returns. Requires a date
   column by construction, so the pooled-quantile mistake is not expressible.
4. :mod:`shingan.eval.stability` — drift, rolling metrics grouped by date, and the
   stress windows.
5. :mod:`shingan.eval.report` — assembles the above, enforces the required metadata,
   and evaluates the falsification conditions.

Two invariants hold across the package and are worth stating once:

* Nothing here fits a parameter on test data. Calibrators and the fusion stacker are
  fitted on the validation fold; the modules that consume them only transform.
* Nothing here treats a synthetic-data result as evidence about markets. The stress
  helpers label their output ``is_synthetic=True``, and the report separates the two
  tables.

Nothing in this package imports :mod:`shingan.models`, so evaluating a run does not
require the training stack to be installed.
"""

from __future__ import annotations

from shingan.eval.backtest import (
    SHARPE_TARGET,
    PerformanceStats,
    QuantileBacktestResult,
    assign_cross_sectional_quantiles,
    benchmark_returns,
    buy_and_hold_returns,
    max_drawdown,
    performance_stats,
    quantile_returns,
    sector_breakdown,
    subsample_every,
)
from shingan.eval.metrics import (
    GATES,
    BootstrapCI,
    CalibrationResult,
    ClassificationReport,
    ICResult,
    average_precision,
    average_precision_lift,
    block_bootstrap_ci,
    brier_score,
    brier_skill_score,
    calibration_curve_table,
    capture_at_fraction,
    evaluate_classification,
    expected_calibration_error,
    fbeta,
    information_coefficient,
    ks_statistic,
    metric_table,
    newey_west_tstat,
    paired_bootstrap_difference,
    risk_decile_table,
    roc_auc,
    select_fbeta_threshold,
)
from shingan.eval.report import (
    NOT_AVAILABLE,
    NOT_MEASURED,
    EvaluationReport,
    ReportValidationError,
    RunMetadata,
    build_run_metadata,
    default_caveats,
    describe_regime,
    falsification_table,
    gate_table,
    render_dataset_card,
    render_model_card,
)
from shingan.eval.splits import (
    EffectiveWindows,
    RollingWindow,
    SplitReport,
    WalkForwardFold,
    assign_split_column,
    effective_windows,
    fold_masks,
    resolve_purge_days,
    rolling_window_report,
    rolling_windows,
    walk_forward_folds,
)
from shingan.eval.stability import (
    SYNTHETIC_STRESS_KINDS,
    DriftReport,
    RollingStabilityResult,
    TrendSummary,
    characteristic_stability_index,
    drift_report,
    historical_stress_suite,
    population_stability_index,
    rolling_stability,
    stress_window,
    synthetic_stress,
    synthetic_stress_sweep,
    theil_sen_trend,
)

__all__ = [
    "GATES",
    "NOT_AVAILABLE",
    "NOT_MEASURED",
    "SHARPE_TARGET",
    "SYNTHETIC_STRESS_KINDS",
    "BootstrapCI",
    "CalibrationResult",
    "ClassificationReport",
    "DriftReport",
    "EffectiveWindows",
    "EvaluationReport",
    "ICResult",
    "PerformanceStats",
    "QuantileBacktestResult",
    "ReportValidationError",
    "RollingStabilityResult",
    "RollingWindow",
    "RunMetadata",
    "SplitReport",
    "TrendSummary",
    "WalkForwardFold",
    "assign_cross_sectional_quantiles",
    "assign_split_column",
    "average_precision",
    "average_precision_lift",
    "benchmark_returns",
    "block_bootstrap_ci",
    "brier_score",
    "brier_skill_score",
    "build_run_metadata",
    "buy_and_hold_returns",
    "calibration_curve_table",
    "capture_at_fraction",
    "characteristic_stability_index",
    "default_caveats",
    "describe_regime",
    "drift_report",
    "effective_windows",
    "evaluate_classification",
    "expected_calibration_error",
    "falsification_table",
    "fbeta",
    "fold_masks",
    "gate_table",
    "historical_stress_suite",
    "information_coefficient",
    "ks_statistic",
    "max_drawdown",
    "metric_table",
    "newey_west_tstat",
    "paired_bootstrap_difference",
    "performance_stats",
    "population_stability_index",
    "quantile_returns",
    "render_dataset_card",
    "render_model_card",
    "resolve_purge_days",
    "risk_decile_table",
    "roc_auc",
    "rolling_stability",
    "rolling_window_report",
    "rolling_windows",
    "sector_breakdown",
    "select_fbeta_threshold",
    "stress_window",
    "subsample_every",
    "synthetic_stress",
    "synthetic_stress_sweep",
    "theil_sen_trend",
    "walk_forward_folds",
]
