"""Look-ahead (leakage) detection.

Leakage in a financial panel model does not announce itself. It does not throw,
it does not produce NaNs, and it does not look wrong: it makes the test metrics
slightly too good, and that is all. The only defence is asserting the invariant
every time data is assembled or a feature matrix is built.

The invariant, from `docs/02-data.md` section 4.1:

    Every feature of a row identified by ``(ticker, as_of)`` must be knowable
    using only information available at ``as_of``. Every label of that row must be
    determined by information that arrives strictly after ``as_of``.

Three classes of check are implemented:

* :func:`assert_feature_matrix_is_clean` — a column-name blacklist. Cheap, exact,
  and catches the most common mistake by far (accidentally passing ``fwd_ret_21d``
  as a feature).
* :func:`assert_asof_monotonic`, :func:`assert_no_future_events`,
  :func:`assert_label_window_observable` — structural checks on the panel.
* :func:`correlation_alarm` — a heuristic. It is a smoke detector, not a proof: a
  feature correlated 0.99 with the label is usually a repackaged label, but
  occasionally it is a genuinely strong signal. Reported, not fatal.

Everything that must never be silently downgraded raises
:class:`LeakageError`. Warnings are used only where a human genuinely has to
adjudicate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from shingan.data.schema import (
    FORBIDDEN_PREFIXES,
    NON_FEATURE_COLUMNS,
    RiskLabel,
    event_column,
    horizon_column,
    mask_column,
)
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Correlation magnitude above which a feature is reported as suspicious. Set high
#: on purpose: the point is to catch a near-perfect relationship, not to nag about
#: every informative feature.
CORRELATION_ALARM_THRESHOLD = 0.95


class LeakageError(RuntimeError):
    """Raised when an assembly step would let future information into a feature set."""


def is_forbidden_column(name: str) -> bool:
    """Whether a column name is reserved for label / forward / event information.

    Args:
        name: Column name to test.

    Returns:
        True if the name starts with a reserved prefix or is an exact reserved name.
    """
    if name in NON_FEATURE_COLUMNS:
        return True
    return any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)


def assert_feature_matrix_is_clean(columns: Iterable[str]) -> None:
    """Fail if any column in a feature matrix is reserved.

    Args:
        columns: The feature column names about to be passed to a model.

    Raises:
        LeakageError: Listing every offending column. All offenders are reported at
            once rather than one per call, so a fix does not need three runs.
    """
    offenders = sorted({name for name in columns if is_forbidden_column(name)})
    if offenders:
        raise LeakageError(
            "feature matrix contains reserved columns that carry label or future "
            f"information: {offenders}. Use shingan.leakage.feature_columns() to "
            "select inputs instead of constructing the list by hand."
        )


def feature_columns(
    frame: pd.DataFrame,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    require_numeric: bool = True,
) -> list[str]:
    """Select model inputs from a processed panel.

    This is the only supported way to build a feature list. It starts from the
    frame's columns, drops everything in
    :data:`shingan.data.schema.NON_FEATURE_COLUMNS`, drops anything with a
    reserved prefix, and then applies the caller's include/exclude filters. The
    result is validated by :func:`assert_feature_matrix_is_clean` before being
    returned.

    Args:
        frame: The processed panel (or any frame with the same conventions).
        include: If given, restrict the result to these columns in this order.
        exclude: Columns to drop, applied after ``include``.
        require_numeric: Keep only numeric dtypes. Default True, since the
            structured track cannot consume object columns.

    Returns:
        The ordered list of usable feature names.

    Raises:
        LeakageError: If the selection somehow retains a reserved column.
        KeyError: If ``include`` names a column that is not in the frame.
    """
    candidates = [name for name in frame.columns if not is_forbidden_column(name)]

    if include is not None:
        missing = [name for name in include if name not in frame.columns]
        if missing:
            raise KeyError(f"requested feature columns are absent from the frame: {missing}")
        order = {name: position for position, name in enumerate(include)}
        candidates = [name for name in candidates if name in order]
        candidates.sort(key=lambda name: order[name])

    if exclude:
        excluded = set(exclude)
        candidates = [name for name in candidates if name not in excluded]

    if require_numeric:
        candidates = [
            name
            for name in candidates
            if pd.api.types.is_numeric_dtype(frame[name])
            and not pd.api.types.is_bool_dtype(frame[name])
        ]

    assert_feature_matrix_is_clean(candidates)
    return candidates


def assert_asof_monotonic(
    frame: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    asof_col: str = "as_of",
) -> None:
    """Fail if ``as_of`` is not strictly increasing within each ticker.

    A duplicated or out-of-order ``as_of`` breaks every ``merge_asof`` downstream
    and makes forward-looking computations silently wrong, because "the next row"
    stops meaning "the next point in time".

    Args:
        frame: Panel to check.
        ticker_col: Name of the ticker column.
        asof_col: Name of the timestamp column.

    Raises:
        LeakageError: On duplicate or non-monotonic timestamps, naming the ticker.
    """
    if frame.empty:
        return
    for ticker, group in frame.groupby(ticker_col, sort=False, observed=True):
        asof = pd.to_datetime(group[asof_col])
        if asof.duplicated().any():
            duplicated = asof[asof.duplicated()].dt.date.unique().tolist()
            raise LeakageError(f"{ticker}: duplicated as_of values {duplicated}")
        if not asof.is_monotonic_increasing:
            raise LeakageError(
                f"{ticker}: as_of is not monotonically increasing; "
                "a merge_asof against this frame would splice the wrong rows"
            )


def assert_no_future_events(
    frame: pd.DataFrame,
    labels: Sequence[RiskLabel | str],
    *,
    asof_col: str = "as_of",
) -> None:
    """Fail if any recorded event happened at or before its row's ``as_of``.

    An event at ``event_date <= as_of`` is not a prediction target, it is a fact
    already in the features. That happens when an event table is joined without a
    forward shift, and it produces a model with excellent in-sample performance
    and no out-of-sample value.

    Rows where the event is unknown (``NaT``) are ignored: absence of an event is
    a legitimate negative, subject to the observability mask.

    Args:
        frame: Panel containing ``event_date_<label>`` columns.
        labels: Labels whose event columns should be checked.
        asof_col: Name of the timestamp column.

    Raises:
        LeakageError: Naming the label, ticker and date of the first violation.
    """
    if frame.empty:
        return
    asof = pd.to_datetime(frame[asof_col])
    for label in labels:
        column = event_column(label)
        if column not in frame.columns:
            continue
        occurred = pd.to_datetime(frame[column], errors="coerce")
        offenders = occurred.notna() & (occurred <= asof)
        if offenders.any():
            first = frame.loc[offenders].iloc[0]
            raise LeakageError(
                f"{column}: event at {occurred[offenders].iloc[0].date()} is not after "
                f"as_of {pd.Timestamp(first[asof_col]).date()} for ticker {first['ticker']}; "
                "the event is already known at the feature timestamp, which is leakage. "
                "Shift the event source forward or use it as a feature instead."
            )


def assert_label_window_observable(
    frame: pd.DataFrame,
    labels: Sequence[RiskLabel | str],
    data_end: date | str | pd.Timestamp,
    *,
    asof_col: str = "as_of",
) -> None:
    """Fail if a row claims a label whose observation window has not finished.

    If ``data_end`` is 2024-12-31 and the ``default_risk`` horizon is 365 days,
    then a row with ``as_of = 2024-06-01`` cannot be labelled: the next seven
    months have not happened yet. Treating such a row as a negative manufactures a
    false negative, and it is the single most common way a backtest overstates a
    risk model.

    A row is acceptable when either the window has completed, or the label is
    explicitly masked off (``label_mask_<label> == False``). The builder masks the
    remainder; this function verifies that it did.

    Args:
        frame: Panel to check.
        labels: Labels to verify.
        data_end: The last date for which data is available.
        asof_col: Name of the timestamp column.

    Raises:
        LeakageError: Naming the label and the count of unmasked, unobservable rows.
    """
    if frame.empty:
        return

    cutoff = pd.Timestamp(data_end)
    asof = pd.to_datetime(frame[asof_col])

    for label in labels:
        mask_col = mask_column(label)
        horizon_col = horizon_column(label)
        if mask_col not in frame.columns:
            continue
        if horizon_col not in frame.columns:
            raise LeakageError(
                f"{horizon_col} is missing, so the observability of {label} cannot be "
                "verified. The column is mandatory: without it there is no record of "
                "which horizon a label claims."
            )
        horizon = int(frame[horizon_col].iloc[0])
        observable = asof + pd.Timedelta(days=horizon) <= cutoff
        unmasked = frame[mask_col].astype(bool)
        offenders = (~observable) & unmasked
        if offenders.any():
            examples = asof[offenders].dt.date.unique().tolist()[:3]
            raise LeakageError(
                f"{label}: {int(offenders.sum())} rows are unmasked but their "
                f"{horizon}-day label window extends past data_end ({cutoff.date()}); "
                f"example as_of values: {examples}. Mask them instead of labelling them "
                "as negatives."
            )


def assert_embargo_gap(
    earlier_end: date | str | pd.Timestamp,
    later_start: date | str | pd.Timestamp,
    margin_days: int,
) -> None:
    """Fail if two adjacent blocks are closer than the required margin.

    Args:
        earlier_end: Last timestamp of the earlier block (usually train).
        later_start: First timestamp of the later block (usually valid or test).
        margin_days: ``purge_days + embargo_calendar_days``.

    Raises:
        LeakageError: If the gap is smaller than the margin.
    """
    gap = (pd.Timestamp(later_start) - pd.Timestamp(earlier_end)).days
    if gap < margin_days:
        raise LeakageError(
            f"blocks are {gap} days apart but the purge+embargo margin is "
            f"{margin_days} days; labels from the earlier block would overlap the "
            "later block"
        )


def assert_temporal_split(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    asof_col: str = "as_of",
    order: Sequence[str] = ("train", "valid", "test"),
    margin_days: int = 0,
) -> None:
    """Fail if the split blocks are not in strict chronological order.

    Args:
        frame: Panel carrying a ``split`` column.
        split_col: Name of the split column.
        asof_col: Name of the timestamp column.
        order: Expected chronological order of split labels. ``purged`` rows are
            ignored, since by construction they straddle the boundaries.
        margin_days: Required separation between consecutive blocks.

    Raises:
        LeakageError: If a block starts before the previous one ends, or the gap is
            smaller than ``margin_days``.
    """
    if frame.empty:
        return
    asof = pd.to_datetime(frame[asof_col])
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name in order:
        selected = asof[frame[split_col] == name]
        if selected.empty:
            continue
        bounds[name] = (selected.min(), selected.max())

    previous_name: str | None = None
    for name in order:
        if name not in bounds:
            logger.debug("split block %r is empty; skipped in ordering check", name)
            continue
        start = bounds[name][0]
        if previous_name is not None:
            previous_end = bounds[previous_name][1]
            if start <= previous_end:
                raise LeakageError(
                    f"split block {name!r} starts at {start.date()} but block "
                    f"{previous_name!r} ends at {previous_end.date()}"
                )
            if margin_days:
                gap = (start - previous_end).days
                if gap < margin_days:
                    raise LeakageError(
                        f"split block {name!r} starts {gap} days after {previous_name!r} "
                        f"but the required margin is {margin_days} days"
                    )
        previous_name = name


def correlation_alarm(
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    threshold: float = CORRELATION_ALARM_THRESHOLD,
) -> list[tuple[str, float]]:
    """Report features almost perfectly correlated with the target.

    This is a heuristic, not a detector. It flags the case where a "feature" is a
    lightly disguised copy of the label (for instance a running count of days since
    an event, or a flag set by the same query that produced the label). A genuine
    strong predictor is possible, so the caller decides; the function only reports.

    Args:
        features: Feature matrix, as a frame or a 2-D array.
        target: Binary or continuous target, aligned with ``features``.
        feature_names: Names to report. Taken from the frame's columns when
            ``features`` is a DataFrame and this is omitted.
        threshold: Absolute correlation above which a feature is reported.

    Returns:
        A list of ``(feature_name, correlation)`` sorted by descending absolute
        correlation, empty when nothing crosses the threshold. Non-numeric or
        all-NaN columns are skipped rather than reported as zero correlation.
    """
    if isinstance(features, pd.DataFrame):
        names = list(feature_names) if feature_names is not None else list(features.columns)
        matrix = features.to_numpy(dtype=float, na_value=np.nan)
    else:
        array = np.asarray(features, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {array.shape}")
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"f{index}" for index in range(array.shape[1])]
        )
        matrix = array

    target_array = np.asarray(target, dtype=float).ravel()
    if matrix.shape[0] != target_array.shape[0]:
        raise ValueError(
            f"features have {matrix.shape[0]} rows but the target has {target_array.shape[0]}"
        )

    alarms: list[tuple[str, float]] = []
    for index, name in enumerate(names):
        column = matrix[:, index]
        valid = np.isfinite(column) & np.isfinite(target_array)
        if valid.sum() < 3 or np.nanstd(column[valid]) == 0 or np.nanstd(target_array[valid]) == 0:
            continue
        correlation = float(np.corrcoef(column[valid], target_array[valid])[0, 1])
        if np.isfinite(correlation) and abs(correlation) >= threshold:
            alarms.append((name, correlation))

    alarms.sort(key=lambda item: abs(item[1]), reverse=True)
    if alarms:
        logger.warning(
            "correlation alarm: %d feature(s) correlate >= %.2f with the target; "
            "confirm these are real predictors and not a repackaged label",
            len(alarms),
            threshold,
        )
    return alarms


def leakage_scan(
    frame: pd.DataFrame,
    *,
    y: pd.Series | np.ndarray,
    feature_names: Sequence[str],
    labels: Sequence[RiskLabel | str],
    data_end: date | pd.Timestamp,
) -> dict[str, Any]:
    """Run the full battery and return a machine-readable summary.

    Intended for the report, so that a reviewer can see the checks actually ran
    rather than trusting that they did. Any structural failure raises
    :class:`LeakageError`; the returned mapping records what passed and what was
    merely flagged.

    Args:
        frame: Processed panel.
        y: Target aligned with ``frame``.
        feature_names: Features that will be used.
        labels: Labels being evaluated.
        data_end: Last date with data.

    Returns:
        Mapping with the feature count, the reserved-column check result, any
        correlation alarms, and the split boundary summary.

    Raises:
        LeakageError: Propagated from any structural check that fails.
    """
    assert_feature_matrix_is_clean(feature_names)
    assert_asof_monotonic(frame)
    assert_no_future_events(frame, labels)
    assert_label_window_observable(frame, labels, data_end)

    alarms = correlation_alarm(frame[list(feature_names)], y)

    asof = pd.to_datetime(frame["as_of"])
    # `dict[str, Any]`, not `dict[str, str]`: this mapping mixes ISO date strings
    # with a row count, and declaring it all-strings made the count entry a type
    # error rather than a moment's thought.
    split_bounds: dict[str, dict[str, Any]] = {}
    for name in frame["split"].astype(str).unique():
        mask = frame["split"].astype(str) == name
        if not mask.any():
            continue
        split_bounds[name] = {
            "start": asof[mask].min().date().isoformat(),
            "end": asof[mask].max().date().isoformat(),
            "rows": int(mask.sum()),
        }

    return {
        "n_features": len(feature_names),
        "reserved_columns_present": sorted(
            name for name in frame.columns if is_forbidden_column(name)
        ),
        "feature_matrix_clean": True,
        "correlation_alarms": [{"feature": name, "correlation": corr} for name, corr in alarms],
        "split_bounds": split_bounds,
    }


def merge_asof_point_in_time(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
    by: str = "ticker",
    date_col: str,
    tolerance_days: int | None = None,
) -> pd.DataFrame:
    """Point-in-time join: take the most recent ``right`` row not after ``left['on']``.

    A thin, opinionated wrapper over :func:`pandas.merge_asof` that enforces the two
    requirements the raw function silently assumes: both frames sorted by the join
    key, and ``right`` free of duplicate ``(by, date_col)`` pairs. Getting either
    wrong produces a plausible-looking frame with the wrong values in it.

    Args:
        left: Frame to attach data to, must contain ``on`` and ``by``.
        right: Frame of observations, must contain ``date_col`` and ``by``.
        on: Name of the timestamp column in ``left``.
        by: Name of the entity key.
        date_col: Name of the timestamp column in ``right``. May equal ``on``.
        tolerance_days: Maximum allowed age of a match. ``None`` accepts any age,
            which is almost never what you want for accounting data.

    Returns:
        A new frame with the ``right`` columns added, suffix-marked on collision.

    Raises:
        ValueError: If a required column is missing, or if ``right`` has duplicate
            ``(by, date_col)`` keys, which would make the join order-dependent.
    """
    for column in (on, by):
        if column not in left.columns:
            raise ValueError(f"left frame is missing required column {column!r}")
    for column in (date_col, by):
        if column not in right.columns:
            raise ValueError(f"right frame is missing required column {column!r}")

    duplicate_keys = right.duplicated(subset=[by, date_col])
    if duplicate_keys.any():
        count = int(duplicate_keys.sum())
        raise ValueError(
            f"right frame has {count} duplicate ({by}, {date_col}) keys; a point-in-time "
            "join would pick whichever came first. De-duplicate by taking the latest "
            "filed revision instead."
        )

    left_sorted = left.assign(**{on: pd.to_datetime(left[on])}).sort_values([on, by])
    right_sorted = right.assign(**{date_col: pd.to_datetime(right[date_col])}).sort_values(
        [date_col, by]
    )

    return pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=on,
        right_on=date_col,
        by=by,
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days) if tolerance_days else None,
    )


def shift_forward_days(series: pd.Series, days: int) -> pd.Series:
    """Move a date column forward, marking rows that leave the sample.

    Used to exclude observations whose label window has not completed. Prefer this
    over ``series + Timedelta`` inside the builder, because it makes the intent
    explicit at the call site.

    Args:
        series: Datetime-like series.
        days: Number of calendar days to add.

    Returns:
        The shifted series; values beyond the last date remain as computed values
        and are masked by the caller, which holds ``data_end``.
    """
    return pd.to_datetime(series) + timedelta(days=days)


def horizon_cutoff(data_end: date | pd.Timestamp, horizon_days: int) -> pd.Timestamp:
    """Last ``as_of`` whose label window fits inside the sample.

    Args:
        data_end: Last date with data.
        horizon_days: Label horizon in calendar days.

    Returns:
        ``data_end - horizon_days`` as a timestamp. Rows after this must be masked.
    """
    return pd.Timestamp(data_end) - timedelta(days=horizon_days)


__all__ = [
    "CORRELATION_ALARM_THRESHOLD",
    "LeakageError",
    "assert_asof_monotonic",
    "assert_embargo_gap",
    "assert_feature_matrix_is_clean",
    "assert_label_window_observable",
    "assert_no_future_events",
    "assert_temporal_split",
    "correlation_alarm",
    "feature_columns",
    "horizon_cutoff",
    "is_forbidden_column",
    "leakage_scan",
    "merge_asof_point_in_time",
    "shift_forward_days",
]
