"""Ranking, calibration, stability and significance metrics.

Three rules shape every function here.

**Degenerate input returns NaN, it does not raise.** A walk-forward run evaluates
dozens of folds. Some of them — especially the `fraud_risk` folds, whose valid
window is about 22 months at a low base rate — will contain one positive, or none.
A metric that raises turns a reportable "this fold has insufficient positives"
into a crashed run, which is strictly less informative. Every function therefore
returns ``nan`` for an undefined quantity and logs at debug level. Callers surface
the count of undefined folds rather than hiding them.

**Ranking precedes thresholding.** Risk models spend a scarce review budget, so
AUC / KS / PR-AUC / IC are the primary metrics and threshold-based scores
(F-beta, precision at a cut) are secondary and always reported next to the
threshold that produced them. A threshold chosen on the test set is a fitted
parameter; :func:`select_fbeta_threshold` exists to make that choice on the
validation fold instead, and its docstring says so.

**Significance needs a time-block bootstrap, not an i.i.d. p-value.** Panel rows
are not independent: rows share dates, tickers overlap across dates, and labels
at a 730-day horizon overlap almost completely. :func:`block_bootstrap_ci`
resamples contiguous *date* blocks with a block length of at least the label
horizon, which is the weakest assumption that still yields an interval.

All randomness is seeded through :func:`shingan.seed.rng` so that a reported
interval is reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from shingan.logging_utils import get_logger
from shingan.seed import DEFAULT_SEED, rng

logger = get_logger(__name__)

#: Below these counts a metric is not computed at all. Two is the minimum for
#: which a rank-based statistic is defined; the thresholds are deliberately not
#: higher, because hiding a fold with three positives behind a NaN is worse than
#: reporting a very wide interval.
MIN_POSITIVES = 2
MIN_NEGATIVES = 2

#: Fewest calendar blocks a block bootstrap needs before its resample distribution
#: means anything. With one block, every draw selects that block and the "interval"
#: has zero width — it reports the point estimate as though it were exact. Two is the
#: smallest count that produces any variation at all; it is still very coarse, and the
#: reported ``n_blocks`` lets a reader see that.
MIN_BLOCKS_FOR_INTERVAL = 2

#: Gate values from docs/05-evaluation.md section 2 and section 9. Kept as data so
#: the report can print "target vs achieved" without duplicating the numbers.
GATES: Mapping[str, float] = {
    "auc": 0.75,
    "ks": 0.30,
    "ic": 0.05,
    "icir": 0.50,
    "ece": 0.05,
    "psi": 0.25,
    "csi": 0.25,
}

NAN = float("nan")

#: Anything that reads as a one-dimensional float sequence. Widened to include
#: ndarray and Index as well as Series, because the callers here hold results from
#: ``np.asarray`` and ``groupby`` in equal measure and a narrower annotation would
#: only force conversion calls whose sole purpose is to satisfy the checker.
ArrayLike = Sequence[float] | pd.Series | np.ndarray | pd.Index


def _clean(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays and drop rows where either value is not finite.

    Dropping is the right call rather than imputing: a row with a missing score
    was not predicted, and a row with a missing label is a masked sample that
    should not have reached the metric in the first place. Both cases are counted
    by the caller through the returned length.
    """
    truth = np.asarray(y_true, dtype=float).ravel()
    score = np.asarray(y_score, dtype=float).ravel()
    if truth.shape != score.shape:
        raise ValueError(
            f"y_true and y_score must have the same length, got {truth.size} and {score.size}"
        )
    finite = np.isfinite(truth) & np.isfinite(score)
    return truth[finite], score[finite]


def _class_counts(y_true: np.ndarray) -> tuple[int, int]:
    positives = int((y_true >= 0.5).sum())
    return positives, int(y_true.size - positives)


def _usable(y_true: np.ndarray, *, minimum_positives: int = MIN_POSITIVES) -> bool:
    """Whether a rank metric is defined on this fold."""
    positives, negatives = _class_counts(y_true)
    if positives < minimum_positives or negatives < MIN_NEGATIVES:
        logger.debug(
            "metric undefined: %d positives and %d negatives (needs at least %d and %d)",
            positives,
            negatives,
            minimum_positives,
            MIN_NEGATIVES,
        )
        return False
    return True


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def roc_auc(y_true: Any, y_score: Any) -> float:
    """Area under the ROC curve, computed from ranks.

    Ranks rather than `sklearn.metrics.roc_auc_score`, for two reasons: the
    Mann-Whitney form makes ties explicit (`average` rank, which is what the AUC
    means for a score with ties — and a tree model at a low base rate produces many
    ties), and it does not silently accept a single-class input.

    Returns:
        The AUC, or NaN when the fold has too few of either class.
    """
    truth, score = _clean(y_true, y_score)
    if not _usable(truth):
        return NAN
    ranks = pd.Series(score).rank(method="average").to_numpy()
    positives = truth >= 0.5
    n_pos = int(positives.sum())
    n_neg = int(truth.size - n_pos)
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y_true: Any, y_score: Any) -> float:
    """Area under the precision-recall curve, using the step-wise estimator.

    This is the headline metric for the fusion comparison. At a base rate of a few
    percent the absolute value is small and easy to misread, which is why
    :func:`average_precision_lift` exists and why the report always prints the
    random baseline next to it.

    Returns:
        Average precision, or NaN when undefined for this fold.
    """
    truth, score = _clean(y_true, y_score)
    if not _usable(truth, minimum_positives=1):
        return NAN
    order = np.argsort(-score, kind="stable")
    ordered_truth = (truth[order] >= 0.5).astype(float)
    cumulative = np.cumsum(ordered_truth)
    positions = np.arange(1, ordered_truth.size + 1)
    precision = cumulative / positions
    recall = cumulative / cumulative[-1]
    # Sum the precision at each positive, weighted by the recall step it adds.
    recall_step = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * recall_step))


def average_precision_lift(y_true: Any, y_score: Any) -> float:
    """PR-AUC divided by the base rate, i.e. the value of a random ranker.

    A PR-AUC of 0.04 on a 1% base rate is four times random and worth reporting; a
    PR-AUC of 0.04 on a 10% base rate is worse than random. The absolute number
    cannot distinguish those, so the lift is what gets compared.
    """
    truth, _ = _clean(y_true, y_score)
    if truth.size == 0:
        return NAN
    base_rate = float((truth >= 0.5).mean())
    if base_rate <= 0.0:
        return NAN
    score = average_precision(y_true, y_score)
    if not np.isfinite(score):
        return NAN
    return score / base_rate


def ks_statistic(y_true: Any, y_score: Any) -> float:
    """Kolmogorov-Smirnov statistic: the largest gap between the two score CDFs.

    Computed directly rather than through ``scipy.stats.ks_2samp`` so that the
    direction (which class is higher) can be reported too and so that ties are
    handled on the score rather than on the sample.
    """
    truth, score = _clean(y_true, y_score)
    if not _usable(truth):
        return NAN
    positives = np.sort(score[truth >= 0.5])
    negatives = np.sort(score[truth < 0.5])
    grid = np.union1d(positives, negatives)
    cdf_pos = np.searchsorted(positives, grid, side="right") / positives.size
    cdf_neg = np.searchsorted(negatives, grid, side="right") / negatives.size
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def ks_direction(y_true: Any, y_score: Any) -> str:
    """Which class sits higher, or ``"undefined"``.

    A model whose negatives score higher than its positives has a KS with the right
    magnitude and the wrong sign. That is not a numerical curiosity; it means a
    feature or a score is inverted, and reporting KS alone hides it.
    """
    truth, score = _clean(y_true, y_score)
    if not _usable(truth):
        return "undefined"
    pos_mean = float(score[truth >= 0.5].mean())
    neg_mean = float(score[truth < 0.5].mean())
    if pos_mean > neg_mean:
        return "positives_higher"
    if pos_mean < neg_mean:
        return "negatives_higher"
    return "tied"


def capture_at_fraction(y_true: Any, y_score: Any, fraction: float = 0.05) -> float:
    """Fraction of all positives captured in the highest-scoring ``fraction`` of rows.

    This is the metric that maps onto the actual workflow: "if an analyst reviews the
    top 5% of names by score, how many of the eventual problem cases do they see?"
    A random ranker captures ``fraction``.

    Returns:
        The capture rate in [0, 1], or NaN when the fold is too small or degenerate.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    truth, score = _clean(y_true, y_score)
    if not _usable(truth, minimum_positives=1):
        return NAN
    n_keep = max(1, int(np.ceil(fraction * truth.size)))
    order = np.argsort(-score, kind="stable")[:n_keep]
    total_positives = float((truth >= 0.5).sum())
    if total_positives <= 0:
        return NAN
    return float((truth[order] >= 0.5).sum() / total_positives)


def accuracy_ratio(y_true: Any, y_score: Any, fraction: float = 0.05) -> float:
    """CAP curve accuracy ratio: ``(capture - fraction) / (1 - fraction)``.

    1.0 is a perfect ranker, 0.0 is random, negative is worse than random. It is
    the standard AR and it is the one number from the CAP curve worth tabulating.
    """
    capture = capture_at_fraction(y_true, y_score, fraction)
    if not np.isfinite(capture):
        return NAN
    return (capture - fraction) / (1.0 - fraction)


def risk_decile_table(y_true: Any, y_score: Any, n_buckets: int = 10) -> pd.DataFrame:
    """Positive rate per score bucket, highest score first.

    Returns:
        A frame with ``bucket``, ``n``, ``n_positive``, ``positive_rate`` and
        ``mean_score``. Empty when the fold is degenerate — which includes any fold
        with fewer than two rows, since a boundary cannot be drawn through a single
        observation.

    Note:
        The guard is ``size < 2`` rather than ``size == 0``, and that distinction cost a
        bug. A one-row slice used to pass the guard, ``pd.qcut`` was then asked for two
        buckets from one distinct value, every row landed in a dropped bin, and the
        resulting ``NaN`` bucket label surfaced three frames later as an
        ``IntCastingNaNError`` from the ``astype`` below — inside ``evaluate_classification``,
        far from the slice that caused it. A single-row stress window was enough to
        trigger it.
    """
    truth, score = _clean(y_true, y_score)
    empty = pd.DataFrame(columns=["bucket", "n", "n_positive", "positive_rate", "mean_score"])
    if truth.size < 2:
        return empty
    ranks = pd.Series(score).rank(method="first")
    buckets = pd.qcut(ranks, min(n_buckets, max(2, truth.size)), labels=False, duplicates="drop")
    if buckets.isna().any():
        # qcut's `duplicates="drop"` can leave values unassigned when the requested bin
        # count collapses. Since the table's contract is "empty when degenerate", return
        # empty rather than dropping the unassigned rows, which would silently shrink `n`
        # and make the rates that remain describe a different sample than the caller
        # passed in.
        logger.warning(
            "score buckets could not be formed from %d rows (%d distinct scores); "
            "returning an empty decile table rather than a partial one",
            truth.size,
            int(np.unique(score).size),
        )
        return empty
    frame = pd.DataFrame(
        {"bucket": buckets.astype(int), "y": (truth >= 0.5).astype(int), "s": score}
    )
    grouped = frame.groupby("bucket", as_index=False).agg(
        n=("y", "size"), n_positive=("y", "sum"), mean_score=("s", "mean")
    )
    grouped["positive_rate"] = grouped["n_positive"] / grouped["n"]
    # Renumber so bucket 0 is the highest-scoring group. The report reads top-down.
    grouped = grouped.sort_values("bucket", ascending=False).reset_index(drop=True)
    grouped["bucket"] = np.arange(len(grouped))
    return grouped[["bucket", "n", "n_positive", "positive_rate", "mean_score"]]


def monotonicity(table: pd.DataFrame, *, tolerance: float = 0.0) -> dict[str, Any]:
    """Whether the positive rate declines as the bucket index increases.

    ``tolerance`` allows end-of-tail reversals, which are expected and should be
    reported rather than flattened: with a handful of positives in the bottom
    bucket, its rate is dominated by sampling noise. A run of small reversals
    followed by a large one at the very end is normal; a reversal in the middle is
    not.
    """
    if table.empty or len(table) < 2:
        return {
            "monotonic": None,
            "n_reversals": 0,
            "max_reversal": NAN,
            "detail": "table too small",
        }
    rates = table["positive_rate"].to_numpy(dtype=float)
    deltas = np.diff(rates)
    reversals = deltas > tolerance
    return {
        "monotonic": bool(not reversals.any()),
        "n_reversals": int(reversals.sum()),
        "max_reversal": float(deltas.max()) if deltas.size else NAN,
        "detail": "rate should fall as bucket index rises (0 = highest scoring group)",
    }


# --------------------------------------------------------------------------- #
# Threshold-based scores
# --------------------------------------------------------------------------- #


def fbeta(
    y_true: Any, y_score: Any, *, beta: float = 2.0, threshold: float = 0.5
) -> dict[str, float]:
    """F-beta at an explicit threshold.

    ``beta > 1`` weights recall above precision, which is the right asymmetry for
    risk review: a missed default is more expensive than a reviewed name that turns
    out fine. The threshold is a parameter, not a search — a threshold chosen on the
    evaluation fold is a fitted parameter, and :func:`select_fbeta_threshold` exists
    so that the search happens on validation data instead.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    truth, score = _clean(y_true, y_score)
    if truth.size == 0:
        return {"fbeta": NAN, "precision": NAN, "recall": NAN, "threshold": threshold}
    predicted = score >= threshold
    actual = truth >= 0.5
    true_positive = float((predicted & actual).sum())
    precision = true_positive / float(predicted.sum()) if predicted.any() else NAN
    recall = true_positive / float(actual.sum()) if actual.any() else NAN
    if not (np.isfinite(precision) and np.isfinite(recall)) or (precision + recall) == 0:
        value = NAN
    else:
        beta_sq = beta * beta
        value = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return {"fbeta": value, "precision": precision, "recall": recall, "threshold": threshold}


def select_fbeta_threshold(
    y_true: Any,
    y_score: Any,
    *,
    beta: float = 2.0,
    n_grid: int = 200,
) -> float:
    """Pick the F-beta-maximising threshold **on validation data**.

    Returns 0.5 when the fold is degenerate, which is a defensible neutral default
    and is logged. Callers must record which fold this was fitted on; using this on
    the test fold produces an optimistic F-beta and the report says so.
    """
    truth, score = _clean(y_true, y_score)
    if truth.size == 0 or not _usable(truth, minimum_positives=1):
        logger.debug("cannot select an F-beta threshold on a degenerate fold; using 0.5")
        return 0.5
    candidates = np.unique(np.quantile(score, np.linspace(0.01, 0.99, n_grid)))
    best_value = -np.inf
    best_threshold = 0.5
    for threshold in candidates:
        value = fbeta(truth, score, beta=beta, threshold=float(threshold))["fbeta"]
        if np.isfinite(value) and value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CalibrationResult:
    """Calibration diagnostics, all measured on the **un-downsampled** data."""

    ece: float
    mce: float
    brier: float
    brier_skill: float
    base_rate: float
    n_bins: int
    reliability: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    tail_ece: float = NAN
    tail_n: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "brier_skill": self.brier_skill,
            "base_rate": self.base_rate,
            "n_bins": self.n_bins,
            "tail_ece": self.tail_ece,
            "tail_n": self.tail_n,
        }

    def passes(self) -> dict[str, bool]:
        """Gate evaluation, with `None`-like entries as False rather than hidden."""
        return {
            "ece_below_gate": bool(np.isfinite(self.ece) and self.ece < GATES["ece"]),
            "brier_beats_base_rate": bool(np.isfinite(self.brier_skill) and self.brier_skill > 0.0),
        }


def brier_score(y_true: Any, y_score: Any) -> float:
    """Mean squared error of the predicted probabilities."""
    truth, score = _clean(y_true, y_score)
    if truth.size == 0:
        return NAN
    return float(np.mean((score - (truth >= 0.5).astype(float)) ** 2))


def brier_skill_score(y_true: Any, y_score: Any) -> float:
    """``1 - Brier / Brier_of_the_constant_base_rate``.

    Positive means the model beats always predicting the base rate. Zero or below
    means it does not, no matter how good the AUC looks — which is the whole point
    of reporting it: a ranker can order correctly and still be a worse probability
    forecast than a constant.
    """
    truth, score = _clean(y_true, y_score)
    if truth.size == 0:
        return NAN
    outcome = (truth >= 0.5).astype(float)
    reference = float(np.mean((float(np.mean(outcome)) - outcome) ** 2))
    if reference <= 0:
        return NAN
    return 1.0 - brier_score(truth, score) / reference


def calibration_curve_table(
    y_true: Any, y_score: Any, *, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Bucketed predicted probability against observed frequency.

    Quantile bins by default rather than equal-width: at a 1% base rate almost every
    prediction falls in the first equal-width bin, and the resulting "curve" is one
    point. Quantile bins guarantee the low buckets are still populated.

    Returns:
        One row per non-empty bin with ``bin``, ``n``, ``mean_predicted``,
        ``observed_rate``, ``abs_gap``.
    """
    truth, score = _clean(y_true, y_score)
    if truth.size == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_predicted", "observed_rate", "abs_gap"])
    frame = pd.DataFrame({"y": (truth >= 0.5).astype(int), "p": score})
    if strategy == "quantile":
        frame["bin"] = pd.qcut(
            frame["p"].rank(method="first"),
            min(n_bins, max(2, len(frame))),
            labels=False,
            duplicates="drop",
        )
    elif strategy == "uniform":
        frame["bin"] = pd.cut(frame["p"], min(n_bins, max(2, len(frame))), labels=False)
    else:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")
    grouped = (
        frame.dropna(subset=["bin"])
        .groupby("bin", as_index=False)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    )
    grouped["abs_gap"] = (grouped["mean_predicted"] - grouped["observed_rate"]).abs()
    return grouped.rename(columns={"bin": "bin"})[
        ["bin", "n", "mean_predicted", "observed_rate", "abs_gap"]
    ]


def expected_calibration_error(
    y_true: Any,
    y_score: Any,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
    tail_fraction: float = 0.1,
) -> CalibrationResult:
    """ECE, MCE, Brier, Brier skill and the reliability table in one pass.

    The tail metric is reported separately because pooled ECE hides the region that
    matters: a model can be well calibrated on average while being badly
    over-confident in its top decile, which is exactly the region a risk process
    acts on.

    Args:
        y_true: Binary outcomes.
        y_score: Predicted probabilities. These must already be calibrated if the
            result is to mean anything — measuring ECE on a raw tree score measures
            the absence of calibration, not its quality.
        n_bins: Number of buckets.
        strategy: ``"quantile"`` (default) or ``"uniform"``.
        tail_fraction: Fraction of the highest-scoring rows treated as the tail.

    Returns:
        A :class:`CalibrationResult`. Empty-degenerate input yields NaN fields.
    """
    truth, score = _clean(y_true, y_score)
    if truth.size == 0:
        return CalibrationResult(NAN, NAN, NAN, NAN, NAN, n_bins)
    outcome = (truth >= 0.5).astype(float)
    table = calibration_curve_table(truth, score, n_bins=n_bins, strategy=strategy)
    if table.empty:
        return CalibrationResult(NAN, NAN, NAN, NAN, float(outcome.mean()), n_bins)
    weights = table["n"] / table["n"].sum()
    ece = float((table["abs_gap"] * weights).sum())
    mce = float(table["abs_gap"].max())

    tail_n = max(1, int(np.ceil(tail_fraction * truth.size)))
    tail_index = np.argsort(-score, kind="stable")[:tail_n]
    tail_ece = float(abs(float(score[tail_index].mean()) - float(outcome[tail_index].mean())))

    return CalibrationResult(
        ece=ece,
        mce=mce,
        brier=brier_score(truth, score),
        brier_skill=brier_skill_score(truth, score),
        base_rate=float(outcome.mean()),
        n_bins=len(table),
        reliability=table,
        tail_ece=tail_ece,
        tail_n=tail_n,
    )


# --------------------------------------------------------------------------- #
# Cross-sectional information coefficient
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ICResult:
    """Cross-sectional rank correlation against a continuous forward quantity."""

    ic: float
    icir: float
    t_stat: float
    n_periods: int
    n_obs: int
    periods_used: int
    mean_obs_per_period: float
    series: pd.Series = field(repr=False, default_factory=pd.Series)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ic": self.ic,
            "icir": self.icir,
            "t_stat": self.t_stat,
            "n_periods": self.n_periods,
            "periods_used": self.periods_used,
            "n_obs": self.n_obs,
            "mean_obs_per_period": self.mean_obs_per_period,
        }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, NaN when either side is constant."""
    if a.size < 2:
        return NAN
    ranks_a = pd.Series(a).rank(method="average").to_numpy()
    ranks_b = pd.Series(b).rank(method="average").to_numpy()
    if ranks_a.std() == 0 or ranks_b.std() == 0:
        return NAN
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def newey_west_tstat(series: ArrayLike, *, lags: int | None = None) -> float:
    """t-statistic of the mean, with Newey-West correction for autocorrelation.

    An IC series is autocorrelated — the same firm appears on consecutive dates, and
    a 730-day label overlaps almost completely across a year of rows. The naive
    ``mean / (std / sqrt(n))`` therefore overstates significance, sometimes by a
    factor of three, and that is the single easiest way to manufacture a result here.

    Args:
        series: The per-period statistic.
        lags: Bartlett kernel bandwidth. Defaults to the standard
            ``floor(4 * (n / 100) ** (2/9))`` rule.

    Returns:
        The corrected t-statistic, or NaN when there are fewer than three
        observations or the series is constant.
    """
    values = np.asarray(pd.Series(series).dropna(), dtype=float)
    n = values.size
    if n < 3:
        return NAN
    demeaned = values - values.mean()
    variance = float(np.dot(demeaned, demeaned) / n)
    if variance <= 0:
        return NAN
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lags = max(0, min(lags, n - 1))

    long_run = variance
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)  # Bartlett
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run += 2 * weight * covariance
    if long_run <= 0:
        return NAN
    standard_error = np.sqrt(long_run / n)
    return float(values.mean() / standard_error)


def information_coefficient(
    dates: Any,
    scores: ArrayLike,
    forward: ArrayLike,
    *,
    min_per_period: int = 3,
) -> ICResult:
    """Compute IC per period, then summarise the resulting series.

    The comparison is against a **continuous** forward quantity — realised
    volatility, drawdown, or return — never against the binary label. Correlating
    against a 0/1 outcome collapses to "are positives ranked above negatives", which
    is the AUC again, and discards the severity dimension the ranking is supposed to
    capture.

    The correlation is computed **within each date**, not on the pooled sample. A
    pooled Spearman is dominated by the market-wide drift in scores — in a
    high-volatility regime every name scores higher — so it measures the regime, not
    the cross-sectional ordering.

    Args:
        dates: Decision dates, one per row.
        scores: Model scores.
        forward: Continuous forward quantity, aligned row-wise with ``scores``.
        min_per_period: Minimum usable rows for a period's correlation to count.

    Returns:
        An :class:`ICResult` whose ``ic`` is the mean per-period correlation and
        whose ``t_stat`` is Newey-West corrected.
    """
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(dates).to_numpy()),
            "score": np.asarray(scores, dtype=float).ravel(),
            "forward": np.asarray(forward, dtype=float).ravel(),
        }
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return ICResult(NAN, NAN, NAN, 0, 0, 0, NAN)

    per_period: dict[Any, float] = {}
    for period, group in frame.groupby("date", sort=True):
        if len(group) < min_per_period:
            continue
        value = _spearman(group["score"].to_numpy(), group["forward"].to_numpy())
        if np.isfinite(value):
            per_period[period] = value

    series = pd.Series(per_period, dtype=float).sort_index()
    series.index.name = "date"
    if series.empty:
        logger.debug("no period had %d usable rows; IC undefined", min_per_period)
        return ICResult(NAN, NAN, NAN, int(frame["date"].nunique()), len(frame), 0, NAN)

    ic = float(series.mean())
    dispersion = float(series.std(ddof=1)) if series.size > 1 else NAN
    icir = ic / dispersion if np.isfinite(dispersion) and dispersion > 0 else NAN
    return ICResult(
        ic=ic,
        icir=icir,
        t_stat=newey_west_tstat(series),
        n_periods=int(frame["date"].nunique()),
        n_obs=len(frame),
        periods_used=int(series.size),
        mean_obs_per_period=float(len(frame) / max(1, frame["date"].nunique())),
        series=series,
    )


# --------------------------------------------------------------------------- #
# Significance
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class BootstrapCI:
    """A block-bootstrap confidence interval for one metric."""

    estimate: float
    low: float
    high: float
    n_boot: int
    block_days: int
    alpha: float
    n_failed: int = 0
    #: How many contiguous calendar blocks the sample was divided into. Fewer than
    #: :data:`MIN_BLOCKS_FOR_INTERVAL` means the resample distribution was degenerate
    #: and the interval is NaN; recorded so a caller can say why.
    n_blocks: int = 0

    @property
    def crosses_zero(self) -> bool | None:
        """Whether the interval includes zero, or None when no interval was computed.

        Returning False for a degenerate interval would be the most misleading answer
        available: "the difference is significant" is exactly the reading a reader
        takes from a CI that excludes zero, and a CI that was never formed excludes
        nothing.
        """
        if not (np.isfinite(self.low) and np.isfinite(self.high)):
            return None
        return bool(self.low <= 0 <= self.high)

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_low": self.low,
            "ci_high": self.high,
            "ci_level": 1 - self.alpha,
            "n_boot": self.n_boot,
            "block_days": self.block_days,
            "n_failed": self.n_failed,
            "n_blocks": self.n_blocks,
            "crosses_zero": self.crosses_zero,
        }


def _date_blocks(unique_dates: np.ndarray, block_days: int) -> list[np.ndarray]:
    """Partition sorted unique dates into contiguous blocks of at most ``block_days``.

    Block membership is decided on the calendar, not on the row count, because the
    dependence being corrected for runs along calendar time: two rows a week apart
    share news and prices regardless of how many other rows sit between them.
    """
    if unique_dates.size == 0:
        return []
    blocks: list[np.ndarray] = []
    start = unique_dates[0]
    current: list[Any] = []
    for value in unique_dates:
        if current and (value - start) > np.timedelta64(block_days, "D"):
            blocks.append(np.asarray(current, dtype=unique_dates.dtype))
            current = []
            start = value
        current.append(value)
    if current:
        blocks.append(np.asarray(current, dtype=unique_dates.dtype))
    return blocks


def block_bootstrap_ci(
    dates: Any,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 500,
    block_days: int = 730,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
    index: np.ndarray | None = None,
) -> BootstrapCI:
    """Bootstrap a statistic by resampling contiguous blocks of dates.

    ``block_days`` must be at least the label horizon. Shorter blocks understate the
    interval, because rows within a horizon share their label outcome — a 730-day
    label means two rows a year apart are not independent draws, and no amount of
    resampling makes them so.

    Args:
        dates: One date per row, aligned with ``index``.
        statistic: Takes an array of row positions and returns a scalar. Rows, not
            values, so the caller can compute differences between two score columns
            on the same resampled rows.
        n_boot: Number of resamples.
        block_days: Calendar length of each resampled block.
        alpha: Two-sided level; 0.05 gives a 95% interval.
        seed: RNG seed.
        index: Optional precomputed positional index.

    Returns:
        A :class:`BootstrapCI`. ``n_failed`` counts resamples whose statistic was
        NaN — a bootstrap that silently drops half its draws produces an interval
        that is too narrow, so the count is part of the result.
    """
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if block_days <= 0:
        raise ValueError(f"block_days must be positive, got {block_days}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    positions = np.arange(len(dates)) if index is None else np.asarray(index)
    date_values = pd.to_datetime(pd.Series(np.asarray(dates)).to_numpy()).to_numpy()
    unique_dates = np.unique(date_values)
    blocks = _date_blocks(unique_dates, block_days)
    if not blocks:
        return BootstrapCI(NAN, NAN, NAN, 0, block_days, alpha, n_boot, n_blocks=0)

    # A block bootstrap needs more than one block to have anything to resample. With a
    # single block, every draw selects that one block and the resample distribution is a
    # point mass: the "interval" comes out with zero width and reads as infinite
    # confidence in the estimate. Returning NaN instead makes the caller report the
    # interval as not computed, which is true, rather than as a narrow one, which is the
    # opposite of the truth.
    #
    # This is not a corner case at POC scale. `block_days` must cover the label horizon
    # (730 days for fraud_risk), and a 15-year panel cannot give the test block more than
    # one such block once the split margins have taken their share.
    if len(blocks) < MIN_BLOCKS_FOR_INTERVAL:
        logger.warning(
            "the sample spans %d unique dates in %d block(s) of %d days; a block "
            "bootstrap cannot form a resample distribution from %d block(s), so no "
            "interval is reported. Widen the sample, shorten the label horizon, or "
            "report the point estimate alone.",
            unique_dates.size,
            len(blocks),
            block_days,
            len(blocks),
        )
        return BootstrapCI(NAN, NAN, NAN, n_boot, block_days, alpha, 0, n_blocks=len(blocks))

    rows_by_date: dict[Any, np.ndarray] = {
        date: positions[date_values == date] for date in unique_dates
    }

    generator = rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n_failed = 0
    for draw in range(n_boot):
        chosen = generator.integers(0, len(blocks), size=len(blocks))
        sampled = np.concatenate([rows_by_date[date] for pick in chosen for date in blocks[pick]])
        try:
            value = float(statistic(sampled))
        except (ValueError, ZeroDivisionError, IndexError):  # pragma: no cover - degenerate draw
            value = NAN
        if np.isfinite(value):
            draws[draw] = value
        else:
            draws[draw] = NAN
            n_failed += 1

    valid = draws[np.isfinite(draws)]
    if valid.size < max(10, n_boot // 10):
        logger.warning(
            "only %d of %d bootstrap resamples produced a statistic; the interval is not reported",
            valid.size,
            n_boot,
        )
        return BootstrapCI(NAN, NAN, NAN, n_boot, block_days, alpha, n_failed, n_blocks=len(blocks))

    estimate = float(np.mean(valid))
    low, high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(
        estimate=estimate,
        low=float(low),
        high=float(high),
        n_boot=n_boot,
        block_days=block_days,
        alpha=alpha,
        n_failed=n_failed,
        n_blocks=len(blocks),
    )


def paired_bootstrap_difference(
    dates: Any,
    score_a: ArrayLike,
    score_b: ArrayLike,
    y_true: ArrayLike,
    metric: Callable[[Any, Any], float],
    *,
    n_boot: int = 500,
    block_days: int = 730,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Bootstrap ``metric(A) - metric(B)`` on the **same** resampled rows.

    Two independent intervals that happen to overlap is not a significance test, and
    two independent intervals that do not overlap is not one either. Resampling both
    arms on identical rows cancels the shared sampling noise, which is the difference
    between being able to say "fused beats structured-only" and only being able to
    say "they were measured".

    Intersecting zero means no demonstrated improvement — the falsification
    condition F1 in the evaluation document.

    Args:
        dates: Decision dates, one per row.
        score_a: The candidate, typically the fused score.
        score_b: The baseline, typically structured-only.
        y_true: Binary outcomes.
        metric: ``(y_true, y_score) -> float``.
        n_boot: Number of resamples.
        block_days: Block length, at least the label horizon.
        alpha: Two-sided level.
        seed: RNG seed.

    Returns:
        A :class:`BootstrapCI` whose ``estimate`` is the observed difference.
    """
    truth = np.asarray(y_true, dtype=float).ravel()
    a = np.asarray(score_a, dtype=float).ravel()
    b = np.asarray(score_b, dtype=float).ravel()
    for name, array in (("y_true", truth), ("score_a", a), ("score_b", b)):
        if array.size != truth.size:
            raise ValueError(f"{name} has length {array.size}, expected {truth.size}")

    observed = float(metric(truth, a)) - float(metric(truth, b))

    def statistic(rows: np.ndarray) -> float:
        return float(metric(truth[rows], a[rows])) - float(metric(truth[rows], b[rows]))

    interval = block_bootstrap_ci(
        dates,
        statistic,
        n_boot=n_boot,
        block_days=block_days,
        alpha=alpha,
        seed=seed,
    )
    # Report the observed difference as the point estimate; the bootstrap mean is
    # biased low for AUC-like statistics, which are bounded above.
    return BootstrapCI(
        estimate=observed,
        low=interval.low,
        high=interval.high,
        n_boot=interval.n_boot,
        block_days=interval.block_days,
        alpha=interval.alpha,
        n_failed=interval.n_failed,
        n_blocks=interval.n_blocks,
    )


# --------------------------------------------------------------------------- #
# Aggregate entry point
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ClassificationReport:
    """The full metric vector for one path on one evaluation slice.

    Every field is optional in the sense that it is NaN when undefined, and the
    ``n_*`` fields say how much data the number rests on. A report without those
    counts cannot be judged.
    """

    path: str
    label: str
    split: str
    n_rows: int
    n_positives: int
    n_negatives: int
    base_rate: float
    auc: float = NAN
    ks: float = NAN
    ks_direction: str = "undefined"
    pr_auc: float = NAN
    pr_auc_lift: float = NAN
    capture_top5: float = NAN
    accuracy_ratio: float = NAN
    calibration: CalibrationResult | None = None
    monotonic: bool | None = None
    n_reversals: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "label": self.label,
            "split": self.split,
            "n_rows": self.n_rows,
            "n_positives": self.n_positives,
            "n_negatives": self.n_negatives,
            "base_rate": self.base_rate,
            "auc": self.auc,
            "ks": self.ks,
            "ks_direction": self.ks_direction,
            "pr_auc": self.pr_auc,
            "pr_auc_lift": self.pr_auc_lift,
            "capture_top5": self.capture_top5,
            "accuracy_ratio": self.accuracy_ratio,
            "monotonic": self.monotonic,
            "n_reversals": self.n_reversals,
        }
        payload.update(self.calibration.as_dict() if self.calibration else {})
        return payload

    def gates(self, *, require_calibration: bool = True) -> dict[str, bool]:
        """Which section-9 gates this slice meets.

        Returns:
            A mapping of gate name to pass/fail. Failing entries are present and
            false rather than omitted: a gate that is not reported reads as "not
            applicable", which is the opposite of what a miss means.
        """
        checks = {
            "auc_above_gate": bool(np.isfinite(self.auc) and self.auc > GATES["auc"]),
            "ks_above_gate": bool(np.isfinite(self.ks) and self.ks > GATES["ks"]),
            "ks_direction_correct": self.ks_direction == "positives_higher",
            "pr_auc_above_random": bool(np.isfinite(self.pr_auc_lift) and self.pr_auc_lift > 1.0),
            "capture_above_random": bool(
                np.isfinite(self.capture_top5) and self.capture_top5 > 0.05
            ),
            "monotonic_or_minor_tail": self.monotonic is not False or self.n_reversals <= 1,
        }
        if require_calibration:
            checks["calibration"] = bool(
                self.calibration and all(self.calibration.passes().values())
            )
        return checks


def evaluate_classification(
    y_true: Any,
    y_score: Any,
    *,
    path: str = "unknown",
    label: str = "unknown",
    split: str = "unknown",
    calibration_bins: int = 10,
) -> ClassificationReport:
    """Compute the whole metric vector for one path on one slice.

    Calibration is computed on whatever rows are passed in. Passing a downsampled
    frame here silently invalidates ECE, Brier and PR-AUC while leaving AUC and KS
    roughly intact, which is a particularly confusing failure: the ranking metrics
    look fine and the probability metrics are wrong. Evaluation frames must not be
    downsampled.
    """
    truth, score = _clean(y_true, y_score)
    n_positives, n_negatives = _class_counts(truth)
    monotonic_flag: bool | None = None
    n_reversals = 0
    if truth.size:
        table = risk_decile_table(truth, score, n_buckets=10)
        summary = monotonicity(table)
        monotonic_flag = summary["monotonic"]
        n_reversals = int(summary["n_reversals"])

    return ClassificationReport(
        path=path,
        label=label,
        split=split,
        n_rows=int(truth.size),
        n_positives=n_positives,
        n_negatives=n_negatives,
        base_rate=float(n_positives / truth.size) if truth.size else NAN,
        auc=roc_auc(truth, score),
        ks=ks_statistic(truth, score),
        ks_direction=ks_direction(truth, score),
        pr_auc=average_precision(truth, score),
        pr_auc_lift=average_precision_lift(truth, score),
        capture_top5=capture_at_fraction(truth, score, 0.05),
        accuracy_ratio=accuracy_ratio(truth, score, 0.05),
        calibration=expected_calibration_error(truth, score, n_bins=calibration_bins),
        monotonic=monotonic_flag,
        n_reversals=n_reversals,
    )


def metric_table(reports: Sequence[ClassificationReport]) -> pd.DataFrame:
    """One row per report, ordered as given. Convenience for the report writer."""
    if not reports:
        return pd.DataFrame()
    return pd.DataFrame.from_records([report.as_dict() for report in reports])


__all__ = [
    "GATES",
    "MIN_NEGATIVES",
    "MIN_POSITIVES",
    "ArrayLike",
    "BootstrapCI",
    "CalibrationResult",
    "ClassificationReport",
    "ICResult",
    "accuracy_ratio",
    "average_precision",
    "average_precision_lift",
    "block_bootstrap_ci",
    "brier_score",
    "brier_skill_score",
    "calibration_curve_table",
    "capture_at_fraction",
    "evaluate_classification",
    "expected_calibration_error",
    "fbeta",
    "information_coefficient",
    "ks_direction",
    "ks_statistic",
    "metric_table",
    "monotonicity",
    "newey_west_tstat",
    "paired_bootstrap_difference",
    "risk_decile_table",
    "roc_auc",
    "select_fbeta_threshold",
]
