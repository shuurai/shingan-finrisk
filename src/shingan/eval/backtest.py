"""Cross-sectional quantile backtest.

The distinction this module exists to enforce
------------------------------------------------
A risk score that behaves well in-sample can still be useless for allocation, and
the difference shows up in one place: whether the score orders names **against each
other on the same day**.

The tempting implementation pools the whole panel and calls ``pd.qcut`` once:

    df["quantile"] = pd.qcut(df["pred"], n_quantiles, labels=False)

It is wrong twice over.

1. **It mixes time series with cross-section.** Whether a 2020 row lands in the
   top quintile then depends on how it ranks against 2023 rows. Scores drift with
   the regime — in a stressed market every name scores higher — so "quintile 5"
   describes a different kind of company in each period. Early on it may be
   entirely ordinary firms; during a crash, entirely distressed ones. The bucket
   label stops meaning anything.
2. **One period dominates.** A pooled mean weights each period by its row count.
   Disclosure density and universe size both change over time, so the result is
   usually driven by whichever period has the most rows.

:func:`quantile_returns` therefore *requires* a date column and ranks within each
date. There is no code path that skips the grouping, which is the only reliable way
to prevent the mistake: a docstring saying "remember to group by date" gets
forgotten, an API that cannot be called without a date does not.

Overlapping windows are the second trap
---------------------------------------
With a 21-trading-day forward return sampled every day, consecutive observations
share 20 of their 21 days. Return series built that way are extremely
autocorrelated, and the usual ``mean / std * sqrt(252)`` Sharpe overstates the truth
by roughly ``sqrt(21)``. :func:`performance_stats` reports the naive figure and a
Newey-West corrected t-statistic side by side, and
:func:`subsample_every` exists to build a non-overlapping series when that is what
is wanted. The naive number alone is not reportable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from shingan.eval.metrics import (
    NAN,
    ArrayLike,
    BootstrapCI,
    block_bootstrap_ci,
    newey_west_tstat,
)
from shingan.logging_utils import get_logger
from shingan.seed import DEFAULT_SEED

logger = get_logger(__name__)

#: Trading days per year, for annualising a daily series.
TRADING_DAYS_PER_YEAR = 252

#: Sharpe aspiration rather than a gate. Deliberately *not* an entry in
#: ``metrics.GATES``: the evaluation document lists it as a target that this sample
#: size is unlikely to reach, so it is reported as an aspiration and never as a
#: pass/fail gate.
SHARPE_TARGET = 1.0

#: Benchmark labels the report must include alongside the model paths
#: (docs/05-evaluation.md section 5.1).
BENCHMARK_PATHS: tuple[str, ...] = ("random", "buy_and_hold", "equal_weight")


@dataclass(slots=True)
class PerformanceStats:
    """Risk-adjusted performance of one return series."""

    n_periods: int
    total_return: float
    mean_period_return: float
    volatility: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    t_stat: float
    periods_per_year: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_periods": self.n_periods,
            "total_return": self.total_return,
            "mean_period_return": self.mean_period_return,
            "volatility": self.volatility,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "hit_rate": self.hit_rate,
            "t_stat_newey_west": self.t_stat,
            "periods_per_year": self.periods_per_year,
        }


def max_drawdown(returns: ArrayLike) -> float:
    """Largest peak-to-trough decline of the compounded equity curve.

    ``returns`` are per-period *simple* returns, not log returns: the drawdown of a
    log-return series is not what an investor experiences. The result is negative or
    zero, matching the sign convention of the forward drawdown columns so the two can
    be compared without a mental sign flip.
    """
    values = np.asarray(pd.Series(returns, dtype=float).dropna(), dtype=float)
    if values.size == 0:
        return NAN
    equity = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = equity / peak - 1.0
    return float(np.min(drawdown))


def performance_stats(
    returns: ArrayLike,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceStats:
    """Annualised performance statistics for a per-period return series.

    Args:
        returns: Simple (not log) per-period returns, one per rebalance.
        periods_per_year: Annualisation factor. 252 for a daily series. Passing 52
            for a weekly series is the caller's responsibility and is the most
            common way to produce a Sharpe that is wrong by a factor of two.

    Returns:
        A :class:`PerformanceStats`. Degenerate input yields NaN statistics rather
        than raising, so one flat quantile bucket does not abort a backtest.
    """
    values = np.asarray(pd.Series(returns, dtype=float).dropna(), dtype=float)
    n = values.size
    if n == 0:
        return PerformanceStats(
            0, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, periods_per_year
        )

    mean = float(values.mean())
    volatility = float(values.std(ddof=1)) if n > 1 else NAN
    downside = values[values < 0]
    downside_deviation = float(np.sqrt(np.mean(downside**2))) if downside.size else NAN
    annual_return = mean * periods_per_year
    annual_volatility = volatility * np.sqrt(periods_per_year) if np.isfinite(volatility) else NAN
    sharpe = (
        annual_return / annual_volatility if annual_volatility and annual_volatility > 0 else NAN
    )
    sortino = (
        annual_return / (downside_deviation * np.sqrt(periods_per_year))
        if np.isfinite(downside_deviation) and downside_deviation > 0
        else NAN
    )
    drawdown = max_drawdown(values)
    calmar = annual_return / abs(drawdown) if np.isfinite(drawdown) and drawdown < 0 else NAN

    return PerformanceStats(
        n_periods=n,
        total_return=float(np.prod(1.0 + values) - 1.0),
        mean_period_return=mean,
        volatility=volatility,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=drawdown,
        calmar=calmar,
        hit_rate=float((values > 0).mean()),
        # The Newey-West t-statistic is the one that can be quoted. The naive
        # `mean / (std / sqrt(n))` assumes independence that overlapping forward
        # windows destroy.
        t_stat=newey_west_tstat(values),
        periods_per_year=periods_per_year,
    )


def subsample_every(
    frame: pd.DataFrame,
    *,
    date_col: str,
    every: int,
) -> pd.DataFrame:
    """Keep every ``every``-th date, to build a non-overlapping return series.

    Use this when the forward horizon is 21 days and you want a daily Sharpe that is
    not triple-counted. ``every=21`` on trading-day data gives a non-overlapping
    series.

    Args:
        frame: Frame containing ``date_col``.
        date_col: Column of dates.
        every: Stride over the sorted unique dates.

    Returns:
        The subset of rows whose date is in the retained set.
    """
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}")
    if every == 1:
        return frame
    unique_dates = np.sort(pd.Series(frame[date_col]).unique())
    keep = unique_dates[::every]
    return frame.loc[frame[date_col].isin(keep)].copy()


@dataclass(slots=True)
class QuantileBacktestResult:
    """Everything a quantile backtest produces."""

    n_quantiles: int
    per_date: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    mean_by_quantile: pd.Series = field(repr=False, default_factory=pd.Series)
    long_short: pd.Series = field(repr=False, default_factory=pd.Series)
    stats_by_quantile: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    long_short_stats: PerformanceStats | None = None
    long_short_ci: BootstrapCI | None = None
    spread_monotonic: bool | None = None
    n_dates: int = 0
    n_rows: int = 0
    insufficient_dates: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_quantiles": self.n_quantiles,
            "n_dates": self.n_dates,
            "n_rows": self.n_rows,
            "insufficient_dates": self.insufficient_dates,
            "spread_monotonic": self.spread_monotonic,
            "mean_return_by_quantile": {
                str(key): float(value) for key, value in self.mean_by_quantile.items()
            },
            "long_short": self.long_short_stats.as_dict() if self.long_short_stats else None,
            "long_short_ci": self.long_short_ci.as_dict() if self.long_short_ci else None,
        }

    def passes(self) -> dict[str, bool]:
        """Section-9 economic-significance gates.

        Sharpe above 1.0 is a *target*, not a gate — the evaluation document says
        explicitly that it is unlikely to be reached at this sample size, so it is
        reported as a target flag rather than a pass/fail.
        """
        spread_ok = bool(
            self.long_short_stats
            and np.isfinite(self.long_short_stats.t_stat)
            and abs(self.long_short_stats.t_stat) > 1.96
        )
        direction_ok = bool(
            self.long_short_stats
            and np.isfinite(self.long_short_stats.mean_period_return)
            and self.long_short_stats.mean_period_return < 0
        )
        return {
            "spread_significant": spread_ok,
            "high_risk_underperforms": direction_ok,
            "quantiles_monotonic": bool(self.spread_monotonic),
            "sharpe_above_target": bool(
                self.long_short_stats
                and np.isfinite(self.long_short_stats.sharpe)
                and abs(self.long_short_stats.sharpe) > SHARPE_TARGET
            ),
        }


def assign_cross_sectional_quantiles(
    frame: pd.DataFrame,
    *,
    score_col: str,
    date_col: str,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Add a ``quantile`` column, ranked **within each date**.

    Quantile 0 is the highest-risk group (highest score), so that the table reads
    top-down the way a risk report does and a "monotone" decline is an increasing
    quantile index.

    Dates with fewer distinct scores than ``n_quantiles`` are dropped rather than
    producing empty or duplicated buckets; the count of dropped dates is returned by
    :func:`quantile_returns` as ``insufficient_dates``.

    Args:
        frame: Panel with a score and a date column.
        score_col: Column holding the model score.
        date_col: Column holding the decision date. Required — see the module
            docstring for why this is not optional.
        n_quantiles: Number of buckets.

    Returns:
        A copy of ``frame`` with an integer ``quantile`` column.
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be at least 2, got {n_quantiles}")
    for column in (score_col, date_col):
        if column not in frame.columns:
            raise KeyError(
                f"{column!r} is not in the frame. This function ranks within dates by "
                "construction; a pooled quantile over the whole panel is not offered."
            )

    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col])

    def _rank(group: pd.Series) -> pd.Series:
        if group.notna().sum() < n_quantiles or group.nunique(dropna=True) < 2:
            return pd.Series(np.nan, index=group.index)
        ranks = group.rank(method="first", ascending=False)
        return pd.qcut(ranks, n_quantiles, labels=False)

    working["quantile"] = (
        working.groupby(date_col, sort=False)[score_col].transform(_rank).astype("Float64")
    )
    return working


def quantile_returns(
    frame: pd.DataFrame,
    *,
    score_col: str,
    forward_col: str,
    date_col: str,
    n_quantiles: int = 5,
    return_col: str | None = None,
    bootstrap: bool = False,
    n_boot: int = 500,
    block_days: int = 730,
    seed: int = DEFAULT_SEED,
) -> QuantileBacktestResult:
    """Score-ordered quantile returns, computed date by date then averaged over time.

    Args:
        frame: Panel data.
        score_col: Model score column; higher means higher predicted risk.
        forward_col: Forward return column, e.g. ``fwd_ret_21d``.
        date_col: Decision date column.
        n_quantiles: Number of buckets.
        return_col: Unused placeholder for a future non-return forward quantity;
            kept out of the required arguments on purpose.
        bootstrap: When True, also bootstrap the long-short spread. Off by default
            because it costs ``n_boot`` resamples and the walk-forward loop calls
            this per fold.
        n_boot: Bootstrap resamples.
        block_days: Bootstrap block length; must be at least the label horizon.
        seed: RNG seed.

    Returns:
        A :class:`QuantileBacktestResult`. ``per_date`` is the wide frame of
        per-date mean returns by quantile; ``long_short`` is its top-minus-bottom
        series, which is what the Sharpe and drawdown figures are computed on.
    """
    del return_col  # reserved; the forward quantity is always a return here

    working = assign_cross_sectional_quantiles(
        frame, score_col=score_col, date_col=date_col, n_quantiles=n_quantiles
    )
    usable = working.dropna(subset=["quantile", forward_col])
    n_dates_total = int(working[date_col].nunique())
    n_dates_used = int(usable[date_col].nunique())
    insufficient = n_dates_total - n_dates_used
    if insufficient:
        logger.warning(
            "%d of %d dates had fewer usable rows than %d quantiles and were skipped",
            insufficient,
            n_dates_total,
            n_quantiles,
        )

    if usable.empty:
        return QuantileBacktestResult(
            n_quantiles=n_quantiles,
            n_dates=0,
            n_rows=len(working),
            insufficient_dates=insufficient,
        )

    usable = usable.assign(quantile=usable["quantile"].astype(int))
    per_date = usable.groupby([date_col, "quantile"])[forward_col].mean().unstack().sort_index()
    # Every quantile must exist on a date for the spread to be comparable; a date
    # with a missing bucket is dropped rather than filled, because a filled bucket
    # would silently contribute a zero return to the long-short series.
    per_date = per_date.dropna(how="any")
    if per_date.empty:
        return QuantileBacktestResult(
            n_quantiles=n_quantiles,
            n_dates=0,
            n_rows=len(working),
            insufficient_dates=insufficient,
        )

    mean_by_quantile = per_date.mean(axis=0)
    # Quantile 0 is the highest-scoring (riskiest) group, so the long-short spread is
    # lowest-minus-highest, which for a working risk score is negative.
    spread = per_date.iloc[:, -1] - per_date.iloc[:, 0]

    stats = pd.DataFrame(
        {
            "mean_forward_return": mean_by_quantile,
            "volatility": per_date.std(ddof=1),
            "n_dates": per_date.notna().sum(),
        }
    )

    spread_stats = performance_stats(spread)
    spread_ci: BootstrapCI | None = None
    if bootstrap:
        spread_ci = block_bootstrap_ci(
            spread.index,
            lambda rows: float(spread.to_numpy()[rows].mean()),
            n_boot=n_boot,
            block_days=block_days,
            seed=seed,
        )

    ordered = mean_by_quantile.to_numpy(dtype=float)
    # Monotone means "risk falls as the quantile index rises", i.e. forward return
    # rises. One tail reversal is tolerated per the evaluation document.
    deltas = np.diff(ordered)
    reversals = int((deltas < 0).sum())

    return QuantileBacktestResult(
        n_quantiles=n_quantiles,
        per_date=per_date,
        mean_by_quantile=mean_by_quantile,
        long_short=spread,
        stats_by_quantile=stats,
        long_short_stats=spread_stats,
        long_short_ci=spread_ci,
        spread_monotonic=bool(reversals <= 1),
        n_dates=int(per_date.shape[0]),
        n_rows=len(usable),
        insufficient_dates=insufficient,
    )


def sector_breakdown(
    frame: pd.DataFrame,
    *,
    score_col: str,
    forward_col: str,
    date_col: str,
    sector_col: str = "sector",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Long-short spread per sector, to show no sector is driving the result.

    A pooled result can be produced entirely by one industry — if the score is really
    a proxy for "is a bank", every banking name in the top bucket will look like
    skill. Reporting the spread by sector is the cheapest defence.

    Returns:
        One row per sector with ``n_dates``, ``mean_spread``, ``t_stat`` and a
        ``sufficient`` flag. Sectors with too few dates are reported with a NaN
        statistic and ``sufficient=False`` rather than omitted, so the reader can see
        that coverage is thin rather than that the sector was fine.
    """
    if sector_col not in frame.columns:
        logger.debug("no %r column; sector breakdown skipped", sector_col)
        return pd.DataFrame(columns=["sector", "n_dates", "mean_spread", "t_stat", "sufficient"])

    rows: list[dict[str, Any]] = []
    for sector, subset in frame.groupby(sector_col, dropna=True):
        result = quantile_returns(
            subset,
            score_col=score_col,
            forward_col=forward_col,
            date_col=date_col,
            n_quantiles=n_quantiles,
        )
        if result.long_short.empty:
            rows.append(
                {
                    "sector": str(sector),
                    "n_dates": 0,
                    "mean_spread": NAN,
                    "t_stat": NAN,
                    "sufficient": False,
                }
            )
            continue
        stats = result.long_short_stats
        rows.append(
            {
                "sector": str(sector),
                "n_dates": result.n_dates,
                "mean_spread": float(result.long_short.mean()),
                "t_stat": stats.t_stat if stats else NAN,
                "sufficient": bool(result.n_dates >= 20),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("n_dates", ascending=False)


def benchmark_returns(
    frame: pd.DataFrame,
    *,
    forward_col: str,
    date_col: str,
) -> pd.Series:
    """Equal-weighted portfolio returns, using the same dates as the model paths.

    This is the *rebalanced* benchmark: on every date, the cross-sectional mean of
    the forward return. It is the right comparison against a quantile backtest,
    because the quantile portfolios are also re-formed every date.

    Buy-and-hold is **not** returned here. It cannot be: the panel carries a forward
    return per row, not price levels, so a genuine buy-and-hold portfolio — whose
    weights drift with returns instead of being reset — is not recoverable from it.
    Returning the equal-weighted series twice under two names would have looked like
    a benchmark comparison while being none. Use :func:`buy_and_hold_returns` with the
    price panel for that.

    Args:
        frame: Panel data.
        forward_col: Forward return column.
        date_col: Decision date column.

    Returns:
        A series indexed by date. Empty when there is nothing usable.
    """
    working = frame[[date_col, forward_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col])
    working = working.dropna()
    if working.empty:
        return pd.Series(dtype=float)
    return working.groupby(date_col)[forward_col].mean().sort_index()


def buy_and_hold_returns(
    prices: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
    admit_new_names: bool = True,
) -> pd.Series:
    """Daily returns of a buy-and-hold portfolio built from price levels.

    Weights drift: the portfolio is *not* rebalanced, so a name that rallies grows to
    dominate it. That is the point of the benchmark, and it is why this cannot be
    approximated by averaging forward returns — averaging resets the weights on every
    date, which is the rebalanced benchmark instead.

    Names enter on their first available date. A later entrant is funded by scaling
    the existing book down, using the entrant's equal share of the *combined* book, so
    the portfolio stays fully invested and no name is silently favoured.

    Args:
        prices: Long-format price frame.
        date_col: Date column.
        ticker_col: Ticker column.
        close_col: Close column.
        admit_new_names: When False, only names present on the first date ever receive
            weight. Useful for a constant-universe comparison; entrants' returns are
            then not earned by the portfolio.

    Returns:
        A daily return series indexed by date, empty when the input is unusable.
    """
    for column in (date_col, ticker_col, close_col):
        if column not in prices.columns:
            raise KeyError(f"{column!r} is not in the price frame")

    wide = (
        prices.assign(**{date_col: pd.to_datetime(prices[date_col])})
        .pivot_table(index=date_col, columns=ticker_col, values=close_col, aggfunc="last")
        .sort_index()
    )
    if wide.empty or len(wide) < 2:
        return pd.Series(dtype=float)

    prices_ffill = wide.ffill()
    returns = prices_ffill.pct_change()
    quoted = wide.notna()
    # A name is tradeable once its price has been observed at least once.
    available = quoted.cummax()

    weights = pd.DataFrame(0.0, index=wide.index, columns=wide.columns)
    first_row = available.iloc[0]
    if first_row.any():
        weights.iloc[0, :] = first_row.to_numpy(dtype=float) / float(first_row.sum())

    portfolio = np.zeros(len(wide), dtype=float)
    for position in range(1, len(wide)):
        previous = weights.iloc[position - 1]
        today = returns.iloc[position].fillna(0.0)

        # The portfolio earns today's return on yesterday's weights.
        portfolio[position] = float((previous * today).sum())

        grown = previous * (1.0 + today)
        if admit_new_names:
            entering = available.iloc[position] & ~available.iloc[position - 1]
            n_entering = int(entering.sum())
            if n_entering:
                n_existing = int((previous > 0).sum())
                entrant_share = n_entering / (n_existing + n_entering)
                grown = grown * (1.0 - entrant_share)
                grown[entering.to_numpy()] = entrant_share / n_entering
        else:
            # Names that appear later never receive weight, but their value must not
            # leak into the existing holdings either.
            grown = grown.where(previous > 0, 0.0)

        total = float(grown.sum())
        weights.iloc[position, :] = grown / total if total > 0 else previous

    return pd.Series(portfolio[1:], index=wide.index[1:], name="buy_and_hold")


__all__ = [
    "BENCHMARK_PATHS",
    "SHARPE_TARGET",
    "TRADING_DAYS_PER_YEAR",
    "PerformanceStats",
    "QuantileBacktestResult",
    "assign_cross_sectional_quantiles",
    "benchmark_returns",
    "buy_and_hold_returns",
    "max_drawdown",
    "performance_stats",
    "quantile_returns",
    "sector_breakdown",
    "subsample_every",
]
