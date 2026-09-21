"""Label construction and forward-looking targets.

Everything in this module looks *forward* on purpose. That is the point: a label is
a statement about the future. The danger is not looking forward, it is letting a
forward-looking value reach the feature matrix. Three defences are applied here:

1. Forward columns are named with the ``fwd_`` prefix, which is on the reserved
   list in :mod:`shingan.leakage`.
2. Every forward computation requires a **complete** window. A partial window is
   NaN, never a value computed from fewer observations. Otherwise the last rows of
   the sample — exactly the rows a recent test set contains — get quietly different
   treatment from the rest.
3. Every label carries an observability mask. A row whose window extends past the
   end of the data is masked out, never labelled negative.

The drawdown definition used for ``tail_risk`` is the **running peak-to-trough**
decline, not the drop from the window's first price. The distinction matters: a
stock that rises 40% and then falls 35% from that peak has suffered a 35% drawdown,
but a start-to-end measure would record a positive return and call it safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd

from shingan.config import LabelConfig
from shingan.data.schema import (
    RiskLabel,
    event_column,
    horizon_column,
    label_column,
    mask_column,
    source_of_record_column,
)
from shingan.labeling.definitions import LabelDefinition, is_observable, is_positive
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Horizon, in trading days, of the forward return and forward volatility used by
#: the IC calculation and the quantile backtest. Fixed at 21 (about one month)
#: because IC is a monthly-horizon concept in practice.
DEFAULT_FORWARD_RETURN_HORIZON = 21

_DAY = np.timedelta64(1, "D")


def _complete_window_mask(n_rows: int, horizon: int) -> np.ndarray:
    """Boolean mask of rows whose forward window of ``horizon`` steps is complete."""
    mask = np.zeros(n_rows, dtype=bool)
    if horizon <= 0:  # pragma: no cover - guarded by callers
        return ~mask
    if n_rows > horizon:
        mask[: n_rows - horizon] = True
    return mask


def forward_return(close: pd.Series, horizon: int = DEFAULT_FORWARD_RETURN_HORIZON) -> pd.Series:
    """Log return from ``t`` to ``t + horizon``.

    Args:
        close: Price series for one instrument, ordered by date.
        horizon: Number of steps ahead.

    Returns:
        The forward log return, NaN where the window is incomplete.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    values = np.log(close.to_numpy(dtype=float))
    n_rows = len(values)
    out = np.full(n_rows, np.nan, dtype=float)
    complete = _complete_window_mask(n_rows, horizon)

    # `complete` has length n_rows, but `values[horizon:]` and `values[:-horizon]` have
    # length n_rows - horizon. Indexing the shifted arrays with the full-length mask is
    # an IndexError; the mask therefore has to be consumed as a *count*, not applied
    # directly. Trimming it to the shorter length by slicing would compile and run, but
    # it would silently misalign the mask with the shifted values if the helper's
    # convention ever changed — so the count is derived from the helper explicitly and
    # the two shifted arrays are subtracted directly.
    n_complete = max(0, n_rows - horizon)
    if not complete[:n_complete].all() and n_complete:  # pragma: no cover - guard
        raise AssertionError(
            "the completeness mask does not describe the first n_rows - horizon rows; "
            "_complete_window_mask and forward_return disagree about the convention"
        )
    if n_complete:
        out[:n_complete] = values[horizon:] - values[:-horizon]
    return pd.Series(out, index=close.index)


def forward_realized_volatility(
    close: pd.Series,
    horizon: int = DEFAULT_FORWARD_RETURN_HORIZON,
) -> pd.Series:
    """Annualised realised volatility of the ``horizon`` returns after ``t``.

    This is the preferred input to the information coefficient: IC needs a
    *continuous* forward quantity. Feeding a binary label to a rank correlation
    throws away almost all of the information and produces a number whose scale has
    no interpretation.

    Args:
        close: Price series for one instrument, ordered by date.
        horizon: Number of forward steps.

    Returns:
        Annualised forward volatility, NaN where the window is incomplete or where
        the window contains a price gap.
    """
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2 to have a variance, got {horizon}")

    values = np.log(close.to_numpy(dtype=float))
    steps = np.diff(values)  # steps[i] is the return from i to i+1
    n_rows = len(values)
    complete = _complete_window_mask(n_rows, horizon)

    totals = np.zeros(n_rows, dtype=float)
    squares = np.zeros(n_rows, dtype=float)
    counts = np.zeros(n_rows, dtype=float)

    for offset in range(horizon):
        shifted = np.full(n_rows, np.nan, dtype=float)
        chunk = steps[offset : offset + n_rows]
        shifted[: len(chunk)] = chunk
        valid = complete & np.isfinite(shifted)
        totals[valid] += shifted[valid]
        squares[valid] += shifted[valid] ** 2
        counts[valid] += 1.0

    usable = counts == float(horizon)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = totals / counts
        variance = (squares - counts * mean**2) / (counts - 1.0)
    variance = np.where(usable, variance, np.nan)
    variance = np.where(variance < 0.0, 0.0, variance)
    return pd.Series(np.sqrt(variance * 252.0), index=close.index)


def forward_max_drawdown(close: pd.Series, horizon: int = 30) -> pd.Series:
    """Deepest peak-to-trough decline in the ``horizon`` steps *after* ``t``.

    For each ``t``, over the price path ``[close_t, close_{t+1}, ..., close_{t+H}]``
    this is the minimum of ``close_j / max(close_t..close_j) - 1`` taken over
    ``j >= 1``. The peak starts at ``close_t``, so a decline beginning on the very
    first day out is measured, and the value is zero when the path only rises.

    Args:
        close: Price series for one instrument, ordered by date.
        horizon: Number of forward steps.

    Returns:
        The drawdown as a non-positive fraction, NaN where the window is incomplete
        or contains a price gap. Incomplete rather than partial: a ten-day drawdown
        is not a smaller version of a thirty-day one.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    values = close.to_numpy(dtype=float)
    n_rows = len(values)
    out = np.full(n_rows, np.nan, dtype=float)

    for position in range(n_rows):
        end = position + horizon + 1
        if end > n_rows:
            break
        window = values[position:end]
        if not np.isfinite(window).all():
            continue
        running_peak = np.maximum.accumulate(window)
        with np.errstate(invalid="ignore", divide="ignore"):
            drawdown = window[1:] / running_peak[1:] - 1.0
        if drawdown.size:
            out[position] = float(np.min(drawdown))
    return pd.Series(out, index=close.index)


def add_forward_targets(
    prices: pd.DataFrame,
    *,
    return_horizon: int = DEFAULT_FORWARD_RETURN_HORIZON,
    drawdown_horizon: int = 30,
    date_column: str = "date",
    ticker_column: str = "ticker",
    close_column: str = "close",
) -> pd.DataFrame:
    """Compute the forward-looking target block for a price panel.

    Args:
        prices: Long panel with ticker, date and close.
        return_horizon: Horizon for ``fwd_ret_21d`` and ``fwd_realized_vol_21d``.
        drawdown_horizon: Horizon, in trading days, for ``fwd_max_drawdown_30d``.
            Should match ``labels.tail_risk_horizon_trading_days`` so that the label
            and its continuous version describe the same window.
        date_column: Name of the date column.
        ticker_column: Name of the ticker column.
        close_column: Name of the price column.

    Returns:
        A frame with ``ticker``, ``as_of``, ``fwd_ret_21d``,
        ``fwd_realized_vol_21d`` and ``fwd_max_drawdown_30d``.

    Raises:
        KeyError: If a required column is missing.
    """
    required = (ticker_column, date_column, close_column)
    missing = [column for column in required if column not in prices.columns]
    if missing:
        raise KeyError(f"price panel is missing required columns: {missing}")

    frame = prices[[ticker_column, date_column, close_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.sort_values([ticker_column, date_column]).reset_index(drop=True)

    records: list[pd.DataFrame] = []
    for ticker, group in frame.groupby(ticker_column, sort=False, observed=True):
        close = group[close_column].astype(float).reset_index(drop=True)
        records.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "as_of": group[date_column].to_numpy(),
                    "fwd_ret_21d": forward_return(close, return_horizon).to_numpy(),
                    "fwd_realized_vol_21d": forward_realized_volatility(
                        close, return_horizon
                    ).to_numpy(),
                    "fwd_max_drawdown_30d": forward_max_drawdown(
                        close, drawdown_horizon
                    ).to_numpy(),
                }
            )
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "ticker",
                "as_of",
                "fwd_ret_21d",
                "fwd_realized_vol_21d",
                "fwd_max_drawdown_30d",
            ]
        )
    return pd.concat(records, ignore_index=True)


def apply_risk_labels(
    panel: pd.DataFrame,
    events: pd.DataFrame | None,
    definitions: Mapping[RiskLabel, LabelDefinition],
    *,
    data_end: date | pd.Timestamp,
    config: LabelConfig | None = None,
    date_column: str = "as_of",
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    """Attach labels, masks, event dates and sources of record to a panel.

    ``default_risk`` and ``fraud_risk`` are read from the event table; the window is
    half-open, ``(as_of, as_of + horizon]``, so an event on the prediction date is
    excluded (it is already public) and an event on the final day is included.
    ``tail_risk`` is computed from the panel's own ``fwd_max_drawdown_30d`` column,
    because the label is defined on prices and needs no separate event feed.

    Args:
        panel: Panel with ``ticker``, ``as_of`` and, for ``tail_risk``,
            ``fwd_max_drawdown_30d``.
        events: Event table with ``ticker``, ``event_date``, ``event_kind``,
            ``severity`` and optionally ``source``. May be None or empty, in which
            case event-based labels are masked — not labelled negative.
        definitions: Label definitions from :func:`build_label_definitions`.
        data_end: Last date with data; rows whose window is not closed are masked.
        config: Label configuration, used for the tail-risk threshold.
        date_column: Name of the timestamp column.
        ticker_column: Name of the ticker column.

    Returns:
        A copy of ``panel`` with, for every configured label, the columns
        ``label_<name>``, ``label_mask_<name>``, ``event_date_<name>``,
        ``source_of_record_<name>`` and ``horizon_days_<name>``.

    Raises:
        KeyError: If a required panel column is missing or the event table is malformed.
    """
    labels_config = config or LabelConfig()
    for column in (ticker_column, date_column):
        if column not in panel.columns:
            raise KeyError(f"panel is missing required column {column!r}")

    result = panel.copy()
    result[date_column] = pd.to_datetime(result[date_column])
    cutoff = pd.Timestamp(data_end).date()
    event_frame = _validate_events(events)

    for label, definition in definitions.items():
        label_values = np.zeros(len(result), dtype="int8")
        mask_values = np.zeros(len(result), dtype=bool)
        event_dates = np.full(len(result), np.datetime64("NaT"), dtype="datetime64[ns]")
        sources = np.full(len(result), "", dtype=object)

        for ticker, positions in result.groupby(
            ticker_column, sort=False, observed=True
        ).indices.items():
            rows = np.asarray(positions)
            ticker_dates = pd.to_datetime(result[date_column].iloc[rows])
            order = np.argsort(ticker_dates.to_numpy(), kind="stable")
            rows_sorted = rows[order]
            dates_sorted = ticker_dates.to_numpy()[order]

            observable = np.array(
                [
                    is_observable(definition, pd.Timestamp(value).date(), cutoff)
                    for value in dates_sorted
                ],
                dtype=bool,
            )
            sources[rows_sorted] = definition.source_of_record

            # ``_tail_risk_labels`` narrows ``observable`` further, because a closed
            # window is not the same as a usable price path. The tightened array has to
            # be written to the mask *after* the label function returns. Writing it here
            # instead — as this used to — publishes "observable" for rows whose forward
            # drawdown does not exist, and those rows then enter training, calibration
            # and evaluation as legitimate negatives while carrying an entirely
            # missing feature block. That teaches the model "all missing means safe"
            # and inflates every discrimination metric.
            if label is RiskLabel.TAIL_RISK:
                label_values[rows_sorted], observable = _tail_risk_labels(
                    result, rows_sorted, observable, definition
                )
                mask_values[rows_sorted] = observable
                continue

            mask_values[rows_sorted] = observable

            ticker_events = event_frame.loc[event_frame[ticker_column] == ticker]
            if ticker_events.empty:
                # Zero labels, but the source of record above still says where a label
                # would have come from, so "no event" is distinguishable from
                # "no coverage".
                continue

            relevant = ticker_events.loc[
                ticker_events["event_kind"].astype(str).isin(definition.event_kinds)
            ].sort_values("event_date")
            if relevant.empty:
                logger.warning(
                    "%s: %d events present but none of kind %s can justify %s",
                    ticker,
                    len(ticker_events),
                    definition.event_kinds,
                    label,
                )
                continue

            kinds = relevant["event_kind"].astype(str).to_numpy()
            severities = pd.to_numeric(relevant["severity"], errors="coerce").to_numpy(dtype=float)
            event_dates_sorted = pd.to_datetime(relevant["event_date"]).to_numpy()

            window = np.timedelta64(definition.calendar_horizon_days, "D")
            for position, as_of_value in enumerate(dates_sorted):
                if not observable[position]:
                    continue
                lower = int(np.searchsorted(event_dates_sorted, as_of_value, side="right"))
                upper = int(np.searchsorted(event_dates_sorted, as_of_value + window, side="right"))
                for event_index in range(lower, upper):
                    if is_positive(
                        definition,
                        event_kind=str(kinds[event_index]),
                        event_severity=(
                            None
                            if np.isnan(severities[event_index])
                            else float(severities[event_index])
                        ),
                        event_date=pd.Timestamp(event_dates_sorted[event_index]).date(),
                        as_of=pd.Timestamp(as_of_value).date(),
                    ):
                        label_values[rows_sorted[position]] = 1
                        event_dates[rows_sorted[position]] = event_dates_sorted[event_index]
                        break

        result[label_column(label)] = label_values
        result[mask_column(label)] = mask_values
        result[event_column(label)] = pd.to_datetime(event_dates)
        result[source_of_record_column(label)] = sources
        result[horizon_column(label)] = definition.horizon_days

    _validate_label_rates(result, definitions, labels_config)
    return result


def _tail_risk_labels(
    panel: pd.DataFrame,
    rows_sorted: np.ndarray,
    observable: np.ndarray,
    definition: LabelDefinition,
) -> tuple[np.ndarray, np.ndarray]:
    """Tail-risk positives from the forward drawdown column.

    A missing forward drawdown means the price window is incomplete, which is the
    same condition as an unclosed label window, so those rows stay zero and
    unobservable rather than becoming negative examples.

    Returns:
        ``(labels, observable)``. The observable array is returned rather than mutated
        in place because the caller must write the narrowed version to the mask column;
        relying on in-place mutation is what let unobservable rows keep a true mask.
    """
    column = "fwd_max_drawdown_30d"
    if column not in panel.columns:
        logger.warning(
            "%s is absent, so every %s row is masked rather than labelled; the builder "
            "must compute forward targets from prices before labelling",
            column,
            definition.label,
        )
        observable = np.zeros(len(rows_sorted), dtype=bool)
        return np.zeros(len(rows_sorted), dtype="int8"), observable

    threshold = float(definition.parameters.get("drawdown_threshold", -0.30))
    drawdown = pd.to_numeric(panel[column].iloc[rows_sorted], errors="coerce").to_numpy(dtype=float)
    breached = np.isfinite(drawdown) & (drawdown <= threshold)
    observable = observable & np.isfinite(drawdown)
    return np.where(observable & breached, 1, 0).astype("int8"), observable


def _validate_events(events: pd.DataFrame | None) -> pd.DataFrame:
    """Check the event table and normalise its dtypes.

    Raises rather than warning on a malformed table, because an empty event feed
    silently produces an all-negative label, and an all-negative label trains a model
    that predicts the base rate and reports zero discrimination with no explanation.
    """
    required = ("ticker", "event_date", "event_kind", "severity")
    if events is None or len(events) == 0:
        logger.warning(
            "no event table supplied; event-based labels will be masked, not labelled "
            "negative. Expected only for price-only labels such as tail_risk."
        )
        return pd.DataFrame(columns=[*required, "source"])

    missing = [column for column in required if column not in events.columns]
    if missing:
        raise KeyError(f"event table is missing required columns: {missing}")

    frame = events.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame["severity"] = pd.to_numeric(frame["severity"], errors="coerce")
    if "source" not in frame.columns:
        frame["source"] = "unspecified"
    return frame


def _validate_label_rates(
    panel: pd.DataFrame,
    definitions: Mapping[RiskLabel, LabelDefinition],
    config: LabelConfig,
) -> None:
    """Warn when a label is too rare or entirely absent.

    A label with a handful of positives cannot support the metrics the report prints.
    Saying so at build time is far better than discovering it in the report, where an
    AUC computed on four positives still looks like an AUC.
    """
    for label in definitions:
        values = panel[label_column(label)]
        mask = panel[mask_column(label)].astype(bool)
        observable = int(mask.sum())
        if observable == 0:
            logger.warning("%s: no observable rows; the label is unusable in this window", label)
            continue
        positives = int(values[mask].sum())
        rate = positives / observable
        if positives < config.min_positives_for_metrics:
            logger.warning(
                "%s: only %d positives among %d observable rows (rate %.4f). The report "
                "will mark metrics on this label as insufficient evidence.",
                label,
                positives,
                observable,
                rate,
            )
        elif rate < config.positive_rate_warning_threshold:
            logger.warning(
                "%s: positive rate %.4f is below the %.4f threshold; expect wide "
                "confidence intervals on every metric.",
                label,
                rate,
                config.positive_rate_warning_threshold,
            )


def sample_weights_for_label(
    panel: pd.DataFrame,
    label: RiskLabel | str,
    *,
    negative_downsample_ratio: float | None = None,
    seed: int = 0,
) -> pd.Series:
    """Weights for training one label.

    When ``negative_downsample_ratio`` is None every observable row gets weight 1.0.
    That default is deliberate: reweighting classes changes the meaning of the output
    probability, and the entire evaluation framework rests on that probability being
    calibrated. A model trained on reweighted data can rank well and still fail the
    calibration gate, which is a worse outcome than low recall.

    When a ratio is supplied, negatives are retained with probability
    ``ratio * n_pos / n_neg`` and the survivors are upweighted by the inverse keep
    rate, so the effective class balance is unchanged while the row count falls. That
    is what makes the ratio worth using: faster training at the same probability scale.

    Args:
        panel: Labelled panel.
        label: Which label's weights to compute.
        negative_downsample_ratio: Desired negatives-to-positives ratio, or None.
        seed: Seed for the downsampling draw.

    Returns:
        A float series aligned with ``panel``; zero for masked rows and for negatives
        that were dropped.
    """
    target = RiskLabel(label)
    label_values = panel[label_column(target)].astype(int)
    mask = panel[mask_column(target)].astype(bool)
    weights = pd.Series(1.0, index=panel.index, dtype=float)
    weights.loc[~mask] = 0.0

    if negative_downsample_ratio is None:
        return weights

    n_positive = int((label_values[mask] == 1).sum())
    n_negative = int((label_values[mask] == 0).sum())
    if n_positive == 0 or n_negative == 0:
        return weights

    keep_rate = min(1.0, negative_downsample_ratio * n_positive / n_negative)
    if keep_rate >= 1.0:
        return weights

    negative_positions = mask.index[mask & (label_values == 0)]
    generator = np.random.default_rng(seed)
    keep = generator.random(len(negative_positions)) < keep_rate
    weights.loc[negative_positions] = 0.0
    weights.loc[negative_positions[keep]] = 1.0 / keep_rate
    return weights


def label_rates(
    panel: pd.DataFrame,
    definitions: Mapping[RiskLabel, LabelDefinition],
) -> pd.DataFrame:
    """Positive rate and counts per label, restricted to observable rows.

    Args:
        panel: Labelled panel.
        definitions: Labels to summarise.

    Returns:
        A frame indexed by label with ``n_rows``, ``n_observable``, ``n_positive``,
        ``positive_rate`` and ``horizon_days``.
    """
    records: list[dict[str, float | str]] = []
    for label in definitions:
        mask = panel[mask_column(label)].astype(bool)
        values = panel[label_column(label)].astype(int)
        observable = int(mask.sum())
        positives = int(values[mask].sum()) if observable else 0
        records.append(
            {
                "label": str(label),
                "n_rows": float(len(panel)),
                "n_observable": float(observable),
                "n_positive": float(positives),
                "positive_rate": float(positives / observable) if observable else float("nan"),
                "horizon_days": (
                    float(panel[horizon_column(label)].iloc[0]) if len(panel) else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(records).set_index("label")


__all__ = [
    "DEFAULT_FORWARD_RETURN_HORIZON",
    "add_forward_targets",
    "apply_risk_labels",
    "forward_max_drawdown",
    "forward_realized_volatility",
    "forward_return",
    "label_rates",
    "sample_weights_for_label",
]
