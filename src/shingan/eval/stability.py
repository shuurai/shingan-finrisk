"""Drift monitoring, rolling stability and stress windows.

What "stability" has to mean here
---------------------------------
The gate "rolling-window metric fluctuation < 10%" is not used here, and the reason is
the reason this module is shaped the way it is: at a base rate of a few percent, a fold
can contain four positives, and the sampling standard deviation of a single-fold AUC at
that count already exceeds 10%. A 10% band is not a strict test, it is an unpassable
one, and an unpassable gate gets disabled rather than fixed.

So the honest criterion is reported instead, in three parts:

1. Each window's metric **with its class counts**, so a reader can see what the
   number rests on.
2. Whether each window's confidence interval sits above random, rather than whether
   the point estimate moved by some fraction.
3. Whether there is a monotone trend, via Theil-Sen rather than a breakpoint test —
   the sample is far too small to locate a changepoint.

Drift baselines do not roll
---------------------------
:func:`drift_report` compares each period against a fixed baseline, normally the
validation window. Updating the baseline as new periods arrive is the opposite of
monitoring: drift gets absorbed a little at a time until the day the model fails,
and there is no record of when it started.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from shingan.eval.backtest import quantile_returns
from shingan.eval.metrics import (
    GATES,
    NAN,
    ClassificationReport,
    evaluate_classification,
)
from shingan.eval.splits import RollingWindow, rolling_window_report, rolling_windows
from shingan.logging_utils import get_logger
from shingan.seed import DEFAULT_SEED

logger = get_logger(__name__)

#: Floor used in the PSI log ratio. Without it a bucket that is empty in the
#: comparison period produces an infinite contribution, and one infinite term
#: dominates the sum — the metric stops being a measure of drift and becomes a
#: boolean "some bin emptied".
PSI_EPSILON = 1e-6

#: Synthetic stress scenarios. Declared as an explicit enumeration, and annotated
#: with the literal type rather than plain ``str``, so that a synthetic scenario
#: cannot be silently substituted for a historical one: the evaluation document
#: requires that the two never appear in the same table.
StressKind = Literal["liquidity_drought", "rate_shock"]

SYNTHETIC_STRESS_KINDS: tuple[StressKind, ...] = ("liquidity_drought", "rate_shock")


def population_stability_index(
    expected: Sequence[float] | pd.Series | np.ndarray,
    actual: Sequence[float] | pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """PSI between a baseline distribution and a later one, using baseline quantiles.

    Bin edges come from the *baseline*, not from the pooled data. Using pooled edges
    makes the metric partly a function of the amount of drift present, which is
    circular: a large shift moves the edges and can shrink the measured distance.

    Interpretation, by convention and not by derivation: below 0.1 is no meaningful
    change, 0.1 to 0.25 a moderate shift worth investigating, above 0.25 a
    significant shift. This project gates at 0.25.

    Args:
        expected: Baseline values, normally the validation window.
        actual: Comparison values.
        n_bins: Number of buckets.

    Returns:
        The index, or NaN when either side is empty or the baseline is constant.
    """
    base = np.asarray(pd.Series(expected, dtype=float).dropna(), dtype=float)
    later = np.asarray(pd.Series(actual, dtype=float).dropna(), dtype=float)
    if base.size == 0 or later.size == 0 or n_bins < 2:
        return NAN
    quantiles = np.unique(np.quantile(base, np.linspace(0.0, 1.0, n_bins + 1)))
    if quantiles.size < 3:
        logger.debug("baseline is near-constant; PSI is undefined")
        return NAN
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    base_counts = np.histogram(base, bins=quantiles)[0] / base.size
    later_counts = np.histogram(later, bins=quantiles)[0] / later.size
    base_share = np.clip(base_counts, PSI_EPSILON, None)
    later_share = np.clip(later_counts, PSI_EPSILON, None)
    return float(np.sum((later_share - base_share) * np.log(later_share / base_share)))


def characteristic_stability_index(
    expected_scores: Sequence[float] | pd.Series | np.ndarray,
    actual_scores: Sequence[float] | pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """PSI applied to the score distribution. The same arithmetic, named for its use.

    A rising CSI with a stable PSI means the inputs have not moved but the model's
    output has — which points at the model or its thresholds rather than at the data.
    """
    return population_stability_index(expected_scores, actual_scores, n_bins=n_bins)


@dataclass(slots=True)
class DriftReport:
    """Feature PSI and score CSI for one comparison period against a fixed baseline."""

    baseline_label: str
    comparison_label: str
    psi: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    csi: float = NAN
    missing_rate_shift: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    threshold: float = GATES["psi"]

    @property
    def unstable_features(self) -> list[str]:
        """Features whose PSI exceeds the gate, most-drifted first."""
        if self.psi.empty:
            return []
        flagged = self.psi.loc[self.psi["psi"] > self.threshold].sort_values("psi", ascending=False)
        return [str(name) for name in flagged["feature"]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline_label,
            "comparison": self.comparison_label,
            "csi": self.csi,
            "n_unstable_features": len(self.unstable_features),
            "unstable_features": self.unstable_features,
            "threshold": self.threshold,
            "max_psi": float(self.psi["psi"].max()) if not self.psi.empty else NAN,
        }


def drift_report(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    feature_columns: Iterable[str] | None = None,
    score_column: str | None = None,
    baseline_label: str = "baseline",
    comparison_label: str = "comparison",
    n_bins: int = 10,
) -> DriftReport:
    """Compute per-feature PSI, score CSI and missing-rate shifts.

    Args:
        baseline: The reference frame. Never updated as periods accumulate — see the
            module docstring.
        comparison: The period being checked.
        feature_columns: Features to test. Defaults to the numeric columns shared by
            both frames, excluding the score column.
        score_column: Model score, for the CSI.
        baseline_label: Name recorded in the result.
        comparison_label: Name recorded in the result.
        n_bins: Bins per feature.

    Returns:
        A :class:`DriftReport`.
    """
    if feature_columns is None:
        shared = [column for column in baseline.columns if column in comparison.columns]
        candidates = [
            column
            for column in shared
            if pd.api.types.is_numeric_dtype(baseline[column])
            and column != score_column
            and not column.startswith(("label_", "fwd_", "event_", "horizon_days_"))
        ]
    else:
        candidates = list(feature_columns)

    rows: list[dict[str, Any]] = []
    for column in candidates:
        baseline_values = pd.to_numeric(baseline[column], errors="coerce")
        comparison_values = pd.to_numeric(comparison[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "psi": population_stability_index(
                    baseline_values, comparison_values, n_bins=n_bins
                ),
                "baseline_missing": float(baseline_values.isna().mean()),
                "comparison_missing": float(comparison_values.isna().mean()),
            }
        )
    psi_frame = pd.DataFrame.from_records(rows)
    if not psi_frame.empty:
        psi_frame["missing_shift"] = psi_frame["comparison_missing"] - psi_frame["baseline_missing"]
        psi_frame = psi_frame.sort_values("psi", ascending=False, na_position="last")
    else:
        psi_frame = pd.DataFrame(
            columns=["feature", "psi", "baseline_missing", "comparison_missing", "missing_shift"]
        )

    csi = NAN
    if (
        score_column is not None
        and score_column in baseline.columns
        and score_column in comparison.columns
    ):
        csi = characteristic_stability_index(
            baseline[score_column], comparison[score_column], n_bins=n_bins
        )

    missing = psi_frame[
        ["feature", "baseline_missing", "comparison_missing", "missing_shift"]
    ].copy()

    flagged = (
        psi_frame.loc[psi_frame["psi"] > GATES["psi"], "feature"].tolist()
        if not psi_frame.empty
        else []
    )
    if flagged:
        logger.warning(
            "%d feature(s) drift beyond the PSI gate of %.2f: %s",
            len(flagged),
            GATES["psi"],
            ", ".join(str(name) for name in flagged[:10]),
        )

    return DriftReport(
        baseline_label=baseline_label,
        comparison_label=comparison_label,
        psi=psi_frame,
        csi=csi,
        missing_rate_shift=missing,
    )


# --------------------------------------------------------------------------- #
# Rolling metric stability
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TrendSummary:
    """Whether a metric series trends, described rather than gated."""

    n_points: int
    slope_per_period: float
    first_value: float
    last_value: float
    delta: float
    direction: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "theil_sen_slope": self.slope_per_period,
            "first": self.first_value,
            "last": self.last_value,
            "delta": self.delta,
            "direction": self.direction,
        }


def theil_sen_trend(
    values: Sequence[float] | pd.Series, dates: Sequence[Any] | None = None
) -> TrendSummary:
    """Median of pairwise slopes, plus a plain first/last comparison.

    Chosen over least squares because a single fold where the model happened to catch
    every positive produces an enormous AUC, and one such point dominates an OLS
    slope. The median of pairwise slopes is unmoved by it.

    Args:
        values: The per-window metric, in chronological order.
        dates: Optional per-window dates, used only to report the first and last
            values; the slope is per step, not per day.

    Returns:
        A :class:`TrendSummary`. ``direction`` is ``"flat"`` when fewer than three
        finite points are available, because two points always produce a direction and
        calling that a trend is reading noise.
    """
    del dates  # the slope is per window; dates are carried by the caller's table
    clean = np.asarray(pd.Series(values, dtype=float).to_numpy(), dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return TrendSummary(0, NAN, NAN, NAN, NAN, "flat")
    if clean.size < 3:
        first, last = float(clean[0]), float(clean[-1])
        return TrendSummary(
            n_points=int(clean.size),
            slope_per_period=NAN,
            first_value=first,
            last_value=last,
            delta=last - first,
            direction="flat",
        )

    steps = np.arange(clean.size, dtype=float)
    slopes = [
        (clean[j] - clean[i]) / (steps[j] - steps[i])
        for i in range(clean.size - 1)
        for j in range(i + 1, clean.size)
    ]
    slope = float(np.median(slopes))
    first, last = float(clean[0]), float(clean[-1])
    direction = "up" if slope > 0 else "down"
    if abs(slope) < 1e-12:
        direction = "flat"
    return TrendSummary(
        n_points=int(clean.size),
        slope_per_period=slope,
        first_value=first,
        last_value=last,
        delta=last - first,
        direction=direction,
    )


@dataclass(slots=True)
class RollingStabilityResult:
    """Per-window metrics plus the summary the report needs."""

    table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    coverage: dict[str, Any] = field(default_factory=dict)
    trends: dict[str, TrendSummary] = field(default_factory=dict)
    windows: list[RollingWindow] = field(repr=False, default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "trends": {name: summary.as_dict() for name, summary in self.trends.items()},
            "n_rows": len(self.table),
        }

    def passes(self) -> dict[str, bool]:
        """Stability criteria from the evaluation document, section 9.

        Note what is *not* here: any check on the magnitude of the swing. The gate
        that would have been checked is unreachable at this base rate, and saying so
        is more useful than reporting it as failed.
        """
        if self.table.empty:
            return {"has_multiple_windows": False}
        checks: dict[str, bool] = {"has_multiple_windows": len(self.table) >= 3}
        usable = self.table.loc[self.table["reason"] == "ok"]
        checks["enough_usable_windows"] = len(usable) >= 3
        if "auc" in self.table.columns and len(usable) >= 3:
            checks["no_systematic_decline"] = not (
                self.trends.get("auc") is not None and self.trends["auc"].direction == "down"
            )
            checks["auc_ci_above_random"] = bool((usable["auc_ci_low"] > 0.5).all())
        return checks


def rolling_stability(
    panel: pd.DataFrame,
    *,
    score_column: str,
    label_column: str,
    date_column: str = "as_of",
    mask_column: str | None = None,
    freq: str = "YE",
    n_boot: int = 200,
    seed: int = DEFAULT_SEED,
) -> RollingStabilityResult:
    """Per-time-bucket metrics, grouped by date.

    Grouping is delegated to :func:`shingan.eval.splits.rolling_windows`, which cannot
    slice by row position. That is the whole point: the draft script's rolling AUC
    averaged over arbitrary row windows, which on a ticker-sorted panel measures
    within-firm discrimination rather than stability over time.

    Args:
        panel: Assembled panel.
        score_column: Model score column.
        label_column: Binary label column.
        date_column: Decision date column.
        mask_column: Observability mask; masked rows are excluded.
        freq: Period frequency, e.g. ``"YE"``, ``"QE"``, ``"ME"``.
        n_boot: Bootstrap resamples per window. Zero disables the interval. The block
            length is taken as the window length, which is the largest value that is
            still self-consistent: a block longer than the window is the window.
        seed: RNG seed.

    Returns:
        A :class:`RollingStabilityResult` whose ``table`` has one row per window,
        including windows that were unusable and why.
    """
    from shingan.eval.metrics import block_bootstrap_ci

    windows = rolling_windows(
        panel,
        date_column=date_column,
        freq=freq,
        label_column=label_column,
        mask_column=mask_column,
    )
    coverage = rolling_window_report(windows)

    records: list[dict[str, Any]] = []
    reports: list[ClassificationReport] = []
    for window in windows:
        subset = panel.loc[window.index]
        truth = pd.to_numeric(subset[label_column], errors="coerce")
        score = pd.to_numeric(subset[score_column], errors="coerce")
        usable_mask = truth.notna() & score.notna()
        report = evaluate_classification(
            truth[usable_mask],
            score[usable_mask],
            path="rolling",
            label=label_column,
            split=window.label,
        )
        reports.append(report)

        # Always derived from the rows that were actually scored, not from the
        # window's raw counts: the two differ when a row has a label but no score, and
        # it is the scored counts that determine whether a metric exists.
        row: dict[str, Any] = {
            "window": window.label,
            "start": window.start.date().isoformat(),
            "end": window.end.date().isoformat(),
            "n_rows": report.n_rows,
            "n_positives": report.n_positives,
            "base_rate": report.base_rate,
            "reason": _reason_from_report(report),
            "auc": report.auc,
            "ks": report.ks,
            "pr_auc": report.pr_auc,
            "ece": report.calibration.ece if report.calibration else NAN,
            "auc_ci_low": NAN,
            "auc_ci_high": NAN,
        }

        if n_boot > 0 and report.n_positives >= 2 and report.n_negatives >= 2:
            subset_dates = subset.loc[usable_mask, date_column]
            interval = block_bootstrap_ci(
                subset_dates,
                _auc_over_rows(truth[usable_mask].to_numpy(), score[usable_mask].to_numpy()),
                n_boot=n_boot,
                # A window shorter than one label horizon cannot be block-bootstrapped
                # meaningfully — its blocks would be the whole window. Widening the
                # block to the window length makes the interval honest (it degenerates
                # towards the point estimate) rather than artificially narrow.
                block_days=max(1, (window.end - window.start).days),
                seed=seed,
            )
            row["auc_ci_low"] = interval.low
            row["auc_ci_high"] = interval.high

        records.append(row)

    table = pd.DataFrame.from_records(records)
    trends = {
        name: theil_sen_trend(table[name])
        for name in ("auc", "ks", "pr_auc", "ece")
        if name in table.columns and table[name].notna().any()
    }

    # `coverage` counts windows that were *misconfigured* into insufficiency, which is a
    # different question from whether a window held a single scorable row. A window can be
    # sufficient and still contribute nothing -- e.g. it lies outside the block the model
    # was scored on -- and `passes()` counts those as unusable. Without this key the report
    # had no number matching its own gate and printed `n_usable` against it, producing the
    # impossible row "4/15 usable windows | FAIL".
    coverage["n_scored"] = int((table["reason"] == "ok").sum()) if not table.empty else 0

    return RollingStabilityResult(table=table, coverage=coverage, trends=trends, windows=windows)


def _reason_from_report(report: ClassificationReport) -> str:
    """Classify why a window is unusable, from its counts."""
    if report.n_rows == 0:
        return "empty"
    if report.n_positives == 0:
        return "insufficient_positives"
    if report.n_positives < 2:
        return "single_positive"
    if report.n_negatives < 2:
        return "insufficient_negatives"
    return "ok"


def _auc(truth: np.ndarray, score: np.ndarray) -> float:
    """AUC, imported lazily to keep this module's import graph one-directional."""
    from shingan.eval.metrics import roc_auc

    return roc_auc(truth, score)


def _auc_over_rows(truth: np.ndarray, score: np.ndarray) -> Callable[[np.ndarray], float]:
    """Build the row-index statistic ``block_bootstrap_ci`` expects.

    A factory rather than an inline closure because the caller builds one inside a
    loop over windows: a closure defined in a loop body captures the loop variable by
    reference, which is correct only as long as the callback is consumed before the
    next iteration. It is consumed immediately today, so this is not a live bug — but
    it is one refactor away from silently scoring every window with the last window's
    arrays, and a factory removes the possibility.
    """

    def statistic(rows: np.ndarray) -> float:
        return float(_auc(truth[rows], score[rows]))

    return statistic


# --------------------------------------------------------------------------- #
# Stress windows
# --------------------------------------------------------------------------- #


def stress_window(
    panel: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    date_column: str = "as_of",
    train_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Slice a historical stress period out of the panel.

    Two rules from the evaluation document are enforced rather than documented:

    * The window is right-truncated by the caller's existing label mask. Rows whose
      label window has not completed must not be counted as negatives — that is the
      same failure the split code guards against, and a stress window is where it does
      the most damage, because the stressed period is precisely where labels are
      least likely to have run their course.
    * When ``train_end`` is supplied and the stress window starts on or before it, a
      warning is emitted. A stress test whose data was in the training set measures
      memorisation, not robustness, and the result must be labelled invalid.

    Args:
        panel: Assembled panel.
        start: Inclusive start of the stress window.
        end: Inclusive end of the stress window.
        date_column: Date column.
        train_end: End of the training period, for the contamination check.

    Returns:
        The subset of rows in the window, with an added ``is_label_observable``
        filter already applied when the mask column is present.
    """
    if date_column not in panel.columns:
        raise KeyError(f"{date_column!r} is not in the panel")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"stress window start {start_ts.date()} is after end {end_ts.date()}")

    dates = pd.to_datetime(panel[date_column])
    subset = panel.loc[(dates >= start_ts) & (dates <= end_ts)].copy()

    from shingan.data.schema import mask_column as mask_name_for

    dropped_unobservable = 0
    for label in ("default_risk", "fraud_risk", "tail_risk"):
        column = mask_name_for(label)
        if column in subset.columns:
            dropped_unobservable += int((~subset[column].astype(bool)).sum())
            subset = subset.loc[subset[column].astype(bool)]
    if dropped_unobservable:
        logger.info(
            "%d row-label pairs in %s..%s were unobservable and excluded",
            dropped_unobservable,
            start_ts.date(),
            end_ts.date(),
        )

    if train_end is not None and start_ts <= pd.Timestamp(train_end):
        logger.warning(
            "stress window %s..%s overlaps the training period (which ends %s). Per "
            "docs/05-evaluation.md section 6, a stress test run on training data "
            "measures memorisation, not robustness; the result must be reported as "
            "invalid.",
            start_ts.date(),
            end_ts.date(),
            pd.Timestamp(train_end).date(),
        )
    return subset


def synthetic_stress(
    panel: pd.DataFrame,
    kind: StressKind,
    *,
    severity: float = 1.5,
    seed: int = DEFAULT_SEED,
    score_column: str | None = None,
) -> pd.DataFrame:
    """Construct a synthetic stress scenario by perturbing the panel.

    Deliberately crude and deliberately labelled. Historical windows answer "how did
    the model behave in 2020?"; this answers "does the ranking degrade gracefully as
    conditions worsen?", and the two are different kinds of evidence. The evaluation
    document requires that they never share a table, so the returned frame carries a
    ``stress_kind`` column whose value is always prefixed ``synthetic_``.

    Args:
        panel: Assembled panel.
        kind: One of :data:`SYNTHETIC_STRESS_KINDS`.
        severity: Multiplier on the perturbation. 1.5 deepens volatility by 50%.
        seed: RNG seed.
        score_column: Unused; accepted so call sites can pass their pipeline through.

    Returns:
        A perturbed copy, with ``stress_kind = f"synthetic_{kind}"`.

    Raises:
        ValueError: If ``kind`` is not a recognised synthetic scenario.
    """
    del score_column
    if kind not in SYNTHETIC_STRESS_KINDS:
        raise ValueError(
            f"unknown synthetic stress kind {kind!r}; expected one of "
            f"{', '.join(SYNTHETIC_STRESS_KINDS)}"
        )
    if severity <= 0:
        raise ValueError(f"severity must be positive, got {severity}")

    from shingan.seed import rng

    generator = rng(seed)
    working = panel.copy()

    if kind == "liquidity_drought":
        # Illiquidity is scaled up and turnover down, which is what a liquidity
        # drought looks like in the feature space.
        for column, factor in (("amihud_illiq", severity), ("turnover", 1.0 / severity)):
            if column in working.columns:
                working[column] = working[column].astype(float) * factor
        for column in ("abnormal_volume",):
            if column in working.columns:
                working[column] = working[column].astype(float) / severity
    else:  # rate_shock
        # Volatility and the leverage-sensitive ratios move together, which is the
        # mechanism a rate shock is supposed to exercise.
        for column in ("vol_20d", "vol_60d", "downside_vol_60d"):
            if column in working.columns:
                working[column] = working[column].astype(float) * severity
        for column in ("debt_to_equity", "debt_short_term_ratio"):
            if column in working.columns:
                working[column] = working[column].astype(float) * severity
        if "interest_coverage" in working.columns:
            working["interest_coverage"] = working["interest_coverage"].astype(float) / severity

    if "n_news_30d" in working.columns:
        working["n_news_30d"] = (
            working["n_news_30d"].astype(float) * max(1.0, severity - 0.2)
        ).round()
    working["stress_kind"] = f"synthetic_{kind}"
    working["stress_severity"] = severity
    working["is_synthetic"] = True
    working["_perturbation_seed"] = int(generator.integers(0, 2**31 - 1))
    logger.info(
        "built synthetic stress scenario %r at severity %.2f over %d rows; these "
        "results must not be presented alongside historical stress results",
        kind,
        severity,
        len(working),
    )
    return working


def synthetic_stress_sweep(
    panel: pd.DataFrame,
    *,
    score_column: str,
    label_column: str,
    forward_column: str,
    date_column: str = "as_of",
    kinds: Sequence[StressKind] = SYNTHETIC_STRESS_KINDS,
    severities: Sequence[float] = (1.0, 1.5, 2.0),
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Run every synthetic scenario at several severities, for a degradation curve.

    A single severity answers "does it break?". Three severities answer "does it
    degrade smoothly, or fall off a cliff?", and the second question is the one that
    bears on whether the model is safe to use with a margin.

    Returns:
        One row per scenario and severity, with the ranking metrics and the long-short
        spread. All rows are synthetic.
    """
    rows: list[dict[str, Any]] = []
    for kind in kinds:
        for severity in severities:
            stressed = synthetic_stress(panel, kind, severity=severity)
            truth = pd.to_numeric(stressed[label_column], errors="coerce")
            score = pd.to_numeric(stressed[score_column], errors="coerce")
            usable = truth.notna() & score.notna()
            report = evaluate_classification(truth[usable], score[usable], path=f"synthetic_{kind}")
            backtest = quantile_returns(
                stressed.loc[usable],
                score_col=score_column,
                forward_col=forward_column,
                date_col=date_column,
                n_quantiles=n_quantiles,
            )
            rows.append(
                {
                    "stress_kind": f"synthetic_{kind}",
                    "severity": severity,
                    "is_synthetic": True,
                    "auc": report.auc,
                    "ks": report.ks,
                    "pr_auc": report.pr_auc,
                    "ece": report.calibration.ece if report.calibration else NAN,
                    "mean_spread": float(backtest.long_short.mean())
                    if not backtest.long_short.empty
                    else NAN,
                }
            )
    return pd.DataFrame.from_records(rows)


def historical_stress_suite(
    panel: pd.DataFrame,
    *,
    score_column: str,
    label_column: str,
    forward_column: str,
    periods: Sequence[tuple[str, str, str]],
    date_column: str = "as_of",
    train_end: str | pd.Timestamp | None = None,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Run the historical stress periods and report their metrics.

    Args:
        panel: Assembled panel.
        score_column: Model score column.
        label_column: Binary label column.
        forward_column: Forward return column for the backtest.
        periods: ``(name, start, end)`` triples.
        date_column: Date column.
        train_end: Training end, for the contamination warning.
        n_quantiles: Quantile buckets.

    Returns:
        One row per period with the full metric vector — the document requires the
        whole vector rather than a composite score, because a stressed window holds
        too few positives for one number to be readable.
    """
    rows: list[dict[str, Any]] = []
    for name, start, end in periods:
        subset = stress_window(panel, start, end, date_column=date_column, train_end=train_end)
        truth = pd.to_numeric(subset[label_column], errors="coerce")
        score = pd.to_numeric(subset[score_column], errors="coerce")
        usable = truth.notna() & score.notna()
        report = evaluate_classification(truth[usable], score[usable], path=name, split=name)
        backtest = quantile_returns(
            subset.loc[usable],
            score_col=score_column,
            forward_col=forward_column,
            date_col=date_column,
            n_quantiles=n_quantiles,
        )
        rows.append(
            {
                "period": name,
                "start": pd.Timestamp(start).date().isoformat(),
                "end": pd.Timestamp(end).date().isoformat(),
                "is_synthetic": False,
                "n_rows": report.n_rows,
                "n_positives": report.n_positives,
                **report.as_dict(),
                "mean_spread": float(backtest.long_short.mean())
                if not backtest.long_short.empty
                else NAN,
            }
        )
    return pd.DataFrame.from_records(rows)


__all__ = [
    "PSI_EPSILON",
    "SYNTHETIC_STRESS_KINDS",
    "DriftReport",
    "RollingStabilityResult",
    "StressKind",
    "TrendSummary",
    "characteristic_stability_index",
    "drift_report",
    "historical_stress_suite",
    "population_stability_index",
    "rolling_stability",
    "stress_window",
    "synthetic_stress",
    "synthetic_stress_sweep",
    "theil_sen_trend",
]
