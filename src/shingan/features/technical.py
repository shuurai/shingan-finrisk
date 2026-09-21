"""Structured features from price and volume history (the "technical" group).

Two properties are non-negotiable, and every function here holds to them:

**Backward-looking only.** Each output at row ``t`` is a function of data at
``t`` and earlier. The module never calls ``shift(-n)``; forward-looking
quantities belong to labelling, where they are guarded separately.

**Gap-aware.** Real price series have holes: trading halts, delistings, missing
vendor rows. Interpolating a hole invents prices that never traded, so instead a
run of missing observations longer than ``max_nan_run`` marks every derived
feature in the following window as missing, letting the tree model handle the
absence explicitly rather than being fed a fabricated value.

Every formula is implemented with pandas/numpy rather than a TA library, because
the widely used wrappers (TA-Lib and its binary distributions) need a C toolchain
that is a recurring source of broken installs on Windows.

Column-level definitions match section 5.3 of ``docs/02-data.md``. Two are
deliberately normalised relative to price level so that a $5 stock and a $500
stock are comparable: ``atr_14`` is ATR divided by close (i.e. ATR%), and
``amihud_illiq_20d`` is scaled by 1e6 to keep it in a readable range. Both are
documented at their definition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from shingan.data.schema import TECHNICAL_COLUMNS
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Annualisation factor for daily volatility.
TRADING_DAYS_PER_YEAR = 252

#: Longest lookback used by any feature here. A gap anywhere in this window
#: contaminates the row's features.
LONGEST_WINDOW = 252

#: Columns that must be present in the input price frame.
REQUIRED_PRICE_COLUMNS: tuple[str, ...] = ("ticker", "date", "close")

#: Columns used when present, and whose absence degrades specific features rather
#: than failing the whole call.
OPTIONAL_PRICE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "volume",
    "shares_outstanding",
)

#: Features that cannot be computed from a price series alone. They are emitted as
#: NaN so the column set is stable across data sources; a tree model handles the
#: missingness, and ``ratios_missing_frac``-style accounting in the report shows
#: how much of the feature matrix was actually unusable.
EXTERNAL_FEATURES: tuple[str, ...] = ("vix_level", "vix_chg_5d", "credit_spread_chg_20d")


def _wilder_ema(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing, which is an EMA with ``alpha = 1 / window``.

    Used by RSI, ATR and ADX. It is not the same as ``ewm(span=window)``; the
    difference is small but it makes the output stop matching published values.
    """
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: max of the three pairwise spans including the gap."""
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def log_returns(close: pd.Series) -> pd.Series:
    """Natural log returns, ``log(close_t / close_{t-1})``.

    Log returns are used throughout instead of simple returns because they are
    additive over time, which keeps multi-period windows well behaved when the
    underlying series has gaps.
    """
    return _log_series(close).diff()


def _log_series(close: pd.Series) -> pd.Series:
    """Natural log of a price series, preserving the index.

    Wrapped rather than written inline because the index has to be re-attached
    explicitly: on a per-ticker group the caller has already shared out long and
    short price histories, and an off-by-one here would compare one ticker's
    return with another ticker's price.
    """
    values = np.log(close.astype(float).to_numpy())
    return pd.Series(values, index=close.index, name=close.name)


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    """Log return over ``window`` periods, computed from the same series."""
    log_close = _log_series(close)
    return log_close - log_close.shift(window)


def realized_volatility(logret: pd.Series, window: int) -> pd.Series:
    """Annualised realised volatility of log returns over ``window`` periods."""
    return logret.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def downside_volatility(logret: pd.Series, window: int) -> pd.Series:
    """Annualised volatility of negative log returns only.

    Downside deviation is the measure that matters for risk: a stock that only
    ever rises has no downside volatility regardless of how large its up moves are.
    """
    negative = logret.where(logret < 0.0, 0.0)
    return negative.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_skew(logret: pd.Series, window: int) -> pd.Series:
    """Rolling skewness of log returns. Negative skew means fat left tail."""
    return logret.rolling(window, min_periods=window).skew()


def rolling_kurtosis(logret: pd.Series, window: int) -> pd.Series:
    """Rolling excess kurtosis of log returns. High values mean jump risk."""
    return logret.rolling(window, min_periods=window).kurt()


def rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Deepest shortfall below a trailing high, over a trailing ``window``.

    Defined precisely, because "max drawdown over 60 days" is ambiguous. For each
    row ``t`` this is

        min over s in [t-w+1, t] of ( close_s / max(close over [s-w+1, s]) - 1 )

    that is, the worst peak-to-trough decline of any sub-window of length at most
    ``window`` that ends no later than ``t``. It is returned as a negative fraction.

    A true rolling argmax-based drawdown needs a per-window apply and costs
    O(n * window); this formulation is fully vectorised and, on the data this
    project uses, differs by a negligible amount. Where an exact figure matters —
    label construction — :mod:`shingan.labeling.builders` computes it directly.
    """
    trailing_high = close.rolling(window, min_periods=window).max()
    drawdown = close / trailing_high - 1.0
    return drawdown.rolling(window, min_periods=window).min()


def distance_from_high(close: pd.Series, window: int) -> pd.Series:
    """Percentage distance below the trailing high. Zero at a new high, negative below."""
    trailing_high = close.rolling(window, min_periods=window).max()
    return close / trailing_high - 1.0


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100].

    Values below 30 are conventionally called oversold, above 70 overbought. For
    risk work the interesting region is the *low* end in a deteriorating trend,
    where RSI stays pinned near the floor for weeks.
    """
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = _wilder_ema(gain, window)
    average_loss = _wilder_ema(loss, window)
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    # average_loss == 0 means an unbroken run of gains, which is RSI 100 by
    # definition; the division above yields NaN there.
    return rsi.where(average_loss > 0.0, 100.0).where(average_gain.notna())


def macd_histogram(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD histogram: ``(EMA_fast - EMA_slow) - EMA_signal(EMA_fast - EMA_slow)``.

    Returned as a fraction of close so it is comparable across price levels. Only
    the histogram is exposed; the two component lines are a linear recombination of
    it and add nothing to a tree model.
    """
    price = close.astype(float)
    ema_fast = price.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return (macd_line - signal_line) / price


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """ATR as a fraction of close (ATR%).

    Normalising by close is a deliberate deviation from the bare ATR in price
    units. Raw ATR scales with the price level, so a tree could learn "high-priced
    stock" from it — a property that changes meaning after a stock split and does
    not transfer across a universe with a wide price range.
    """
    true_range = _true_range(high.astype(float), low.astype(float), close.astype(float))
    atr = _wilder_ema(true_range, window)
    return atr / close.astype(float)


def average_directional_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Wilder's ADX, a 0-100 measure of trend strength (not direction).

    High ADX during a decline identifies an orderly downtrend rather than a single
    shock, which is a different risk profile even at the same cumulative return.
    """
    high_f = high.astype(float)
    low_f = low.astype(float)
    close_f = close.astype(float)

    up_move = high_f.diff()
    down_move = -low_f.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=high_f.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=high_f.index,
    )

    atr = _wilder_ema(_true_range(high_f, low_f, close_f), window)
    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100.0 * _wilder_ema(plus_dm, window) / atr_safe
    minus_di = 100.0 * _wilder_ema(minus_dm, window) / atr_safe
    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    return _wilder_ema(dx, window)


def amihud_illiquidity(
    logret: pd.Series,
    volume: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Amihud illiquidity: mean of ``|return| / dollar volume``, scaled by 1e6.

    The scaling is presentational. Raw Amihud values are around 1e-12 for large
    caps, which is numerically awkward for a tree split threshold and unreadable in
    a report; 1e6 puts liquid mega-caps near 1e-6 and illiquid names well above.

    A high value means the price moves a lot per dollar traded, i.e. the stock is
    expensive to exit — which is precisely the risk this feature is meant to carry.
    """
    dollar_volume = (close.astype(float) * volume.astype(float)).replace(0.0, np.nan)
    ratio = logret.abs() / dollar_volume
    return ratio.rolling(window, min_periods=window).mean() * 1e6


def turnover(
    volume: pd.Series,
    shares_outstanding: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Average daily share turnover over ``window`` periods.

    Where share counts are unavailable the builder passes NaN and this returns NaN
    rather than substituting a volume-based proxy, because volume in shares is not
    comparable across companies of different size.
    """
    shares = shares_outstanding.astype(float).replace(0.0, np.nan)
    return (volume.astype(float) / shares).rolling(window, min_periods=window).mean()


def abnormal_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume divided by its trailing average. 1.0 is normal, 5.0 alarming."""
    average = volume.astype(float).rolling(window, min_periods=window).mean()
    return volume.astype(float) / average.replace(0.0, np.nan)


def rolling_beta(
    asset_logret: pd.Series,
    market_logret: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling CAPM beta against a market return series.

    Implemented as a rolling covariance over a rolling variance of the market,
    which is the same estimator ``statsmodels`` OLS would give, without the
    per-window fit. Requires the market series to be aligned on the same index; a
    misalignment would silently produce a plausible-looking wrong beta, so the
    caller must align before calling.
    """
    covariance = asset_logret.rolling(window, min_periods=window).cov(market_logret)
    variance = market_logret.rolling(window, min_periods=window).var(ddof=1)
    return covariance / variance.replace(0.0, np.nan)


def consecutive_nan_run(series: pd.Series) -> pd.Series:
    """Length of the run of missing values ending at each row (0 for a present row).

    Used to detect halts and vendor gaps: a single missing day is routine, thirty
    consecutive missing days is a trading halt or a delisting and must not be
    papered over.
    """
    is_missing = series.isna()
    groups = (~is_missing).cumsum()
    return is_missing.groupby(groups).cumsum().astype(int)


def gap_contamination(
    close: pd.Series,
    max_nan_run: int,
    lookback: int = LONGEST_WINDOW,
) -> pd.Series:
    """Rows whose trailing window contains a gap longer than ``max_nan_run``.

    Returns a boolean series. True means: do not trust any rolling feature at this
    row, because somewhere in its lookback the series was not trading.
    """
    if max_nan_run < 1:
        raise ValueError(f"max_nan_run must be >= 1, got {max_nan_run}")
    long_gap = consecutive_nan_run(close) > max_nan_run
    contaminated = long_gap.rolling(lookback, min_periods=1).max().astype(bool)
    return contaminated | long_gap


def compute_technical_features(
    prices: pd.DataFrame,
    *,
    market_returns: pd.Series | None = None,
    macro: pd.DataFrame | None = None,
    min_history_days: int = 252,
    max_nan_run: int = 10,
) -> pd.DataFrame:
    """Compute the full technical feature block for a price panel.

    Args:
        prices: Long panel with at least ``ticker``, ``date`` and ``close``. When
            ``high``/``low``/``volume``/``shares_outstanding`` are present the
            corresponding features are computed; when absent they are returned as
            NaN so the output schema stays stable across data sources.
        market_returns: Market log returns indexed by date, used for ``beta_252d``.
            Without it, beta is NaN.
        macro: Optional frame with ``date`` plus any of ``vix_level``,
            ``vix``/``vix_chg_5d`` and ``credit_spread``/``credit_spread_chg_20d``.
            Missing regime variables are returned as NaN, which is the honest
            state of the POC for credit spreads.
        min_history_days: Rows with fewer observations than this since the start of
            that ticker's history are flagged ``insufficient_history``. Long-window
            features are also NaN there, which is the stronger signal.
        max_nan_run: Maximum tolerated run of missing prices before the affected
            rows have their features nulled. See :func:`gap_contamination`.

    Returns:
        A frame with ``ticker``, ``as_of``, every column in
        :data:`shingan.data.schema.TECHNICAL_COLUMNS`, and an
        ``insufficient_history`` flag. One row per input row, sorted by
        ``(ticker, as_of)``.

    Raises:
        KeyError: If a required input column is missing.
        ValueError: If ``prices`` is empty or ``max_nan_run`` is not positive.
    """
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in prices.columns]
    if missing:
        raise KeyError(f"price panel is missing required columns: {missing}")
    if prices.empty:
        raise ValueError("price panel is empty; nothing to compute")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    has_high = "high" in frame.columns
    has_low = "low" in frame.columns
    has_volume = "volume" in frame.columns
    has_shares = "shares_outstanding" in frame.columns
    if not (has_high and has_low):
        logger.debug("high/low columns absent; ATR, ADX and the range-based features will be NaN")
    if not has_volume:
        logger.debug("volume column absent; liquidity and volume features will be NaN")

    outputs: list[pd.DataFrame] = []

    for ticker, group in frame.groupby("ticker", sort=False, observed=True):
        index = group.index
        close = group["close"].astype(float)
        logret = log_returns(close)

        contaminated = gap_contamination(close, max_nan_run)

        features: dict[str, pd.Series] = {}
        for window, name in (
            (1, "ret_1d"),
            (5, "ret_5d"),
            (20, "ret_20d"),
            (60, "ret_60d"),
            (252, "ret_252d"),
        ):
            features[name] = rolling_return(close, window)
        features["vol_20d"] = realized_volatility(logret, 20)
        features["vol_60d"] = realized_volatility(logret, 60)
        features["downside_vol_60d"] = downside_volatility(logret, 60)
        features["skew_60d"] = rolling_skew(logret, 60)
        features["kurt_60d"] = rolling_kurtosis(logret, 60)
        features["max_drawdown_60d"] = rolling_max_drawdown(close, 60)
        features["dist_52w_high"] = distance_from_high(close, TRADING_DAYS_PER_YEAR)
        features["rsi_14"] = relative_strength_index(close, 14)
        features["macd_hist"] = macd_histogram(close)

        if has_high and has_low:
            high = group["high"].astype(float)
            low = group["low"].astype(float)
            features["atr_14"] = average_true_range(high, low, close, 14)
            features["adx_14"] = average_directional_index(high, low, close, 14)
        else:
            features["atr_14"] = pd.Series(np.nan, index=index)
            features["adx_14"] = pd.Series(np.nan, index=index)

        if has_volume:
            volume = group["volume"].astype(float)
            features["amihud_illiq_20d"] = amihud_illiquidity(logret, volume, close, 20)
            features["abnormal_volume_20d"] = abnormal_volume(volume, 20)
            features["turnover_20d"] = (
                turnover(volume, group["shares_outstanding"], 20)
                if has_shares
                else pd.Series(np.nan, index=index)
            )
        else:
            features["amihud_illiq_20d"] = pd.Series(np.nan, index=index)
            features["turnover_20d"] = pd.Series(np.nan, index=index)
            features["abnormal_volume_20d"] = pd.Series(np.nan, index=index)

        if market_returns is not None:
            aligned_market = market_returns.reindex(group["date"].to_numpy()).astype(float)
            aligned_market.index = index
            features["beta_252d"] = rolling_beta(logret, aligned_market, TRADING_DAYS_PER_YEAR)
        else:
            features["beta_252d"] = pd.Series(np.nan, index=index)

        # Regime variables are not derivable from a single price series; the
        # builder attaches them from a macro source, if one is available.
        for name in EXTERNAL_FEATURES:
            features[name] = pd.Series(np.nan, index=index)

        block = pd.DataFrame(index=index)
        for name in TECHNICAL_COLUMNS:
            series = features.get(name, pd.Series(np.nan, index=index))
            # Null every derived feature on rows whose trailing window contained a
            # gap longer than max_nan_run.
            block[name] = series.mask(contaminated)
        block["ticker"] = ticker
        block["as_of"] = group["date"].to_numpy()
        # The first `min_history_days` observations of a ticker cannot support the
        # long-window features; the flag makes that visible instead of leaving the
        # model to infer it from a sea of NaN.
        block["insufficient_history"] = np.arange(len(index)) < min_history_days
        outputs.append(block)

    if not outputs:  # pragma: no cover - guarded by the empty check above
        raise RuntimeError("no per-ticker blocks were produced")

    result = pd.concat(outputs).sort_index()
    result = result.reset_index(drop=True)

    if macro is not None:
        result = _attach_macro(result, macro)

    logger.debug(
        "computed %d technical features for %d rows across %d tickers",
        len(TECHNICAL_COLUMNS),
        len(result),
        frame["ticker"].nunique(),
    )
    return result


def _attach_macro(features: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge regime variables onto the feature frame by ``as_of`` date.

    Macro series are published with a lag and revised afterwards, so this uses a
    backward as-of join: a row only ever sees the most recent macro observation at
    or before its own date. That is the same discipline applied to everything else.

    Args:
        features: Feature frame with an ``as_of`` column.
        macro: Frame with a ``date`` column plus any of ``vix``, ``vix_chg_5d``,
            ``credit_spread``, ``credit_spread_chg_20d``.

    Returns:
        The feature frame with the available macro columns filled in and any
        unavailable ones left as they were (NaN).
    """
    if "date" not in macro.columns:
        raise KeyError("macro frame must contain a 'date' column")
    macro_frame = macro.copy()
    macro_frame["date"] = pd.to_datetime(macro_frame["date"])
    macro_frame = macro_frame.sort_values("date")

    if "vix" in macro_frame.columns and "vix_level" not in macro_frame.columns:
        macro_frame["vix_level"] = macro_frame["vix"].astype(float)
    if "vix" in macro_frame.columns and "vix_chg_5d" not in macro_frame.columns:
        macro_frame["vix_chg_5d"] = macro_frame["vix"].astype(float).diff(5)
    if (
        "credit_spread" in macro_frame.columns
        and "credit_spread_chg_20d" not in macro_frame.columns
    ):
        macro_frame["credit_spread_chg_20d"] = macro_frame["credit_spread"].astype(float).diff(20)

    keep = ["date", *[name for name in EXTERNAL_FEATURES if name in macro_frame.columns]]
    return pd.merge_asof(
        features.sort_values("as_of"),
        macro_frame[keep].sort_values("date"),
        left_on="as_of",
        right_on="date",
        direction="backward",
    ).drop(columns=["date"])


def align_to_common_calendar(
    frames: Mapping[str, pd.DataFrame],
    *,
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """Reindex several panels onto their intersection of dates.

    Cross-sectional features (ranks, market beta, dispersion) require every asset
    to be observed on the same dates. Taking the intersection is the conservative
    choice: a union would introduce rows where only one asset existed, and the
    resulting cross-section is not a cross-section.

    Args:
        frames: Mapping of name to frame, each with a ``date_col`` column.
        date_col: Name of the date column.

    Returns:
        A mapping of the same names, each filtered to the common dates.
    """
    if not frames:
        return {}
    common: set[pd.Timestamp] | None = None
    for frame in frames.values():
        dates = set(pd.to_datetime(frame[date_col]).unique())
        common = dates if common is None else common & dates
    if not common:
        logger.warning("no common dates across %d panels", len(frames))
        return {name: frame.iloc[0:0].copy() for name, frame in frames.items()}

    shared = pd.DatetimeIndex(sorted(common))
    aligned: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        # Named `frame_dates`, not `dates`: `dates` above is a set of Timestamps
        # from the intersection loop. Rebinding it to a Series worked, but it made
        # the second loop read as though it were still intersecting.
        frame_dates = pd.to_datetime(frame[date_col])
        aligned[name] = (
            frame.loc[frame_dates.isin(shared)].sort_values(date_col).reset_index(drop=True)
        )
    return aligned


def feature_ranges(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Describe the numeric range and missingness of each feature.

    Emitted into the report, because the first thing to check when a model behaves
    oddly is whether a feature is constant, almost entirely missing, or full of
    infinities. A constant feature is silently useless; a near-constant one is worse,
    because it looks informative.

    Args:
        frame: Feature frame.
        columns: Features to describe. Defaults to every technical feature present.

    Returns:
        A frame indexed by feature name with ``count``, ``missing_frac``,
        ``n_inf``, ``mean``, ``std``, ``min``, ``median``, ``max`` and ``n_unique``.
    """
    selected = (
        list(columns)
        if columns is not None
        else [name for name in TECHNICAL_COLUMNS if name in frame.columns]
    )
    records: list[dict[str, float | str]] = []
    total = len(frame)
    for name in selected:
        series = pd.to_numeric(frame[name], errors="coerce")
        finite = series.replace([np.inf, -np.inf], np.nan)
        records.append(
            {
                "feature": name,
                "count": float(finite.notna().sum()),
                "missing_frac": float(1.0 - finite.notna().sum() / total)
                if total
                else float("nan"),
                "n_inf": float(np.isinf(series).sum()),
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if finite.notna().sum() > 1 else float("nan"),
                "min": float(finite.min()),
                "median": float(finite.median()),
                "max": float(finite.max()),
                "n_unique": float(finite.nunique()),
            }
        )
    return pd.DataFrame.from_records(records).set_index("feature")


__all__ = [
    "EXTERNAL_FEATURES",
    "LONGEST_WINDOW",
    "OPTIONAL_PRICE_COLUMNS",
    "REQUIRED_PRICE_COLUMNS",
    "TRADING_DAYS_PER_YEAR",
    "abnormal_volume",
    "align_to_common_calendar",
    "amihud_illiquidity",
    "average_directional_index",
    "average_true_range",
    "compute_technical_features",
    "consecutive_nan_run",
    "distance_from_high",
    "downside_volatility",
    "feature_ranges",
    "gap_contamination",
    "log_returns",
    "macd_histogram",
    "realized_volatility",
    "relative_strength_index",
    "rolling_beta",
    "rolling_kurtosis",
    "rolling_max_drawdown",
    "rolling_return",
    "rolling_skew",
    "turnover",
]
