"""Time-series splitting with purging and embargoing.

Splitting a financial panel is the step where a project most easily convinces itself
it works. This module is the defence, and the reasoning is worth stating once.

A row ``(ticker, as_of)`` carries a label determined by events in
``(as_of, as_of + H]``. Its *information footprint* is therefore a range, not a point.
If the training block ends the day before the test block begins, then:

* a training row from mid-2018 has a ``default_risk`` label (H = 365) determined by
  events up to mid-2019, which is inside the test block;
* a training row from the last day of the training block has a ``fraud_risk`` label
  (H = 730) determined by events almost two years later, entirely inside the test
  block.

Nothing raises. The test metrics are simply too good.

Two mechanisms are applied:

**Purging** removes training rows whose label window overlaps a later block. The margin
is ``purge_days``, which must be at least the longest label horizon in use — the loader
raises it, with a warning, if it is not.

**Embargoing** removes a further slice adjacent to the boundary. Its purpose is
different: even with non-overlapping label windows, neighbouring rows share price
paths, news flow and market regime, so they are not independent. Embargo length is a
prior choice (30 trading days here), not something estimable from the data.

**Right-hand truncation** is the third, less obvious requirement. With
``data_end = 2024-12-31`` and H = 365, no row later than 2023-12-31 can be labelled: its
window has not closed. Those rows are masked, never treated as negatives. Skipping this
step manufactures false negatives at exactly the end of the sample, which is where a
recent test set lives.

The output is a ``split`` column with four values — ``train``, ``valid``, ``test`` and
``purged`` — plus ``excluded`` for rows that fall outside every nominal window. Purged
and excluded rows are kept in the panel so that the arithmetic can be audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from shingan.config import LabelConfig, SplitConfig, TimeWindow
from shingan.data.schema import RiskLabel
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Values the ``split`` column may take.
SPLIT_TRAIN = "train"
SPLIT_VALID = "valid"
SPLIT_TEST = "test"
SPLIT_PURGED = "purged"
SPLIT_EXCLUDED = "excluded"

SPLIT_VALUES: tuple[str, ...] = (
    SPLIT_TRAIN,
    SPLIT_VALID,
    SPLIT_TEST,
    SPLIT_PURGED,
    SPLIT_EXCLUDED,
)


@dataclass(frozen=True, slots=True)
class EffectiveWindows:
    """Nominal windows and the windows that survive purging.

    Both are kept. Reporting only the effective window hides how much data the design
    costs; reporting only the nominal one overstates the sample.
    """

    nominal_train: TimeWindow
    nominal_valid: TimeWindow
    nominal_test: TimeWindow
    effective_train: TimeWindow
    effective_valid: TimeWindow
    effective_test: TimeWindow
    purge_days: int
    embargo_calendar_days: int

    @property
    def margin_days(self) -> int:
        """Total isolation between consecutive blocks."""
        return self.purge_days + self.embargo_calendar_days

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary for run metadata and the report header."""
        return {
            "purge_days": self.purge_days,
            "embargo_calendar_days": self.embargo_calendar_days,
            "margin_days": self.margin_days,
            "nominal": {
                "train": self.nominal_train.render(),
                "valid": self.nominal_valid.render(),
                "test": self.nominal_test.render(),
            },
            "effective": {
                "train": self.effective_train.render(),
                "valid": self.effective_valid.render(),
                "test": self.effective_test.render(),
            },
            "days_lost_to_purge": {
                "train": (self.nominal_train.end - self.effective_train.end).days,
                "valid": (self.nominal_valid.end - self.effective_valid.end).days,
                "test": (self.nominal_test.end - self.effective_test.end).days,
            },
        }


@dataclass(slots=True)
class WalkForwardFold:
    """One expanding-window fold, with its own purge margins applied."""

    name: str
    nominal_train: TimeWindow
    nominal_valid: TimeWindow
    nominal_test: TimeWindow
    effective_train: TimeWindow
    effective_valid: TimeWindow
    effective_test: TimeWindow

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "name": self.name,
            "train": self.effective_train.render(),
            "valid": self.effective_valid.render(),
            "test": self.effective_test.render(),
            "nominal_train": self.nominal_train.render(),
            "nominal_valid": self.nominal_valid.render(),
            "nominal_test": self.nominal_test.render(),
        }


@dataclass(slots=True)
class SplitReport:
    """Everything the report needs to state how the sample was cut."""

    windows: EffectiveWindows
    per_label_horizons: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    folds: list[WalkForwardFold] = field(default_factory=list)
    per_label: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "windows": self.windows.to_dict(),
            "per_label_horizons": self.per_label_horizons,
            "counts": self.counts,
            "per_label": self.per_label,
            "folds": [fold.to_dict() for fold in self.folds],
        }


def resolve_purge_days(
    labels: LabelConfig,
    split: SplitConfig,
    label: RiskLabel | None = None,
) -> int:
    """The purge margin actually used, never smaller than the required horizon.

    Args:
        labels: Label configuration carrying the horizons.
        split: Split configuration carrying the requested ``purge_days``.
        label: When given and ``split.per_label`` is set, use this label's own
            horizon instead of the maximum across all labels.

    Returns:
        The purge margin in calendar days, with a warning when it had to be raised.
    """
    if label is not None and split.per_label:
        required = labels.calendar_horizon_days(label)
    else:
        required = labels.max_calendar_horizon_days()
    if split.purge_days < required:
        logger.warning(
            "purge_days=%d is below the required horizon of %d days; using %d",
            split.purge_days,
            required,
            required,
        )
        return required
    return split.purge_days


def _effective_from_nominal(
    nominal_train: TimeWindow,
    nominal_valid: TimeWindow,
    nominal_test: TimeWindow,
    *,
    purge_days: int,
    embargo_calendar_days: int,
    horizon_days: int,
    data_end: date | pd.Timestamp,
    label: RiskLabel | None = None,
    min_window_days: int = 0,
) -> EffectiveWindows:
    """Apply the purge and truncation rules to a set of nominal windows.

    Shared by the static split and by walk-forward, so that the two cannot drift
    apart. The rules are applied literally:

    * ``train`` keeps ``as_of <= valid.start - margin``
    * ``valid`` keeps ``as_of <= test.start - margin``
    * ``test`` keeps ``as_of <= data_end - horizon``

    Note that only the *ends* move. A block's effective length is therefore decided
    by where the next block starts, which is why a plausible-looking set of nominal
    windows can leave a few months of validation data: the margin, not the nominal
    window, is what binds.

    Args:
        nominal_train: Nominal training window.
        nominal_valid: Nominal validation window.
        nominal_test: Nominal test window.
        purge_days: Overlap, in calendar days, to remove from the end of a block.
        embargo_calendar_days: Additional isolation, in calendar days.
        horizon_days: Longest label horizon, for the terminal truncation of ``test``.
        data_end: Last date with data.
        label: Label being split on, for the error message.
        min_window_days: If positive, every effective block must be at least this
            long. 0 disables the check, which is what walk-forward passes: its blocks
            are short by construction and it skips an unusable fold with a warning,
            whereas the static split is what the report presents and should refuse to
            present a starved one.

    Returns:
        The :class:`EffectiveWindows`.

    Raises:
        ValueError: If purging consumes a window entirely, or if ``min_window_days``
            is positive and an effective window is shorter than it.
    """
    margin = purge_days + embargo_calendar_days
    cutoff = pd.Timestamp(data_end).date()

    train_end = min(nominal_train.end, nominal_valid.start - timedelta(days=margin))
    valid_end = min(nominal_valid.end, nominal_test.start - timedelta(days=margin))
    test_end = min(nominal_test.end, cutoff - timedelta(days=horizon_days))

    for name, start, end in (
        ("train", nominal_train.start, train_end),
        ("valid", nominal_valid.start, valid_end),
        ("test", nominal_test.start, test_end),
    ):
        if end < start:
            raise ValueError(
                f"purging leaves the {name} window empty: nominal start {start} is after "
                f"the purged end {end} (margin {margin} days, horizon {horizon_days} days)"
                + (f" for label {label}" if label else "")
                + ". Lengthen the nominal windows, shorten the horizon, or use "
                "split.per_label to reduce the margin."
            )
        length = (end - start).days
        if min_window_days and length < min_window_days:
            raise ValueError(
                f"purging leaves the {name} window with only {length} days "
                f"({start} .. {end}), below split.min_effective_window_days="
                f"{min_window_days}. The margin is {margin} days "
                f"(purge {purge_days} + embargo {embargo_calendar_days}) and only the "
                f"window's end is moved, so what binds is where the following block "
                f"starts, not the nominal window. Move "
                f"{'valid.start' if name == 'train' else 'test.start' if name == 'valid' else 'test.end'}"
                f" later, shorten the horizon, or use split.per_label to give each "
                f"label its own margin."
            )

    return EffectiveWindows(
        nominal_train=nominal_train,
        nominal_valid=nominal_valid,
        nominal_test=nominal_test,
        effective_train=TimeWindow(start=nominal_train.start, end=train_end),
        effective_valid=TimeWindow(start=nominal_valid.start, end=valid_end),
        effective_test=TimeWindow(start=nominal_test.start, end=test_end),
        purge_days=purge_days,
        embargo_calendar_days=embargo_calendar_days,
    )


def effective_windows(
    split: SplitConfig,
    labels: LabelConfig,
    data_end: date | pd.Timestamp,
    *,
    label: RiskLabel | None = None,
) -> EffectiveWindows:
    """Compute the post-purge windows for the static split.

    Args:
        split: Split configuration with the nominal windows.
        labels: Label configuration with the horizons.
        data_end: Last date with data.
        label: Optional label for a per-label split.

    Returns:
        The :class:`EffectiveWindows`.

    Raises:
        ValueError: If purging consumes a window entirely.
    """
    purge_days = resolve_purge_days(labels, split, label)
    horizon = (
        labels.calendar_horizon_days(label)
        if label is not None
        else labels.max_calendar_horizon_days()
    )
    return _effective_from_nominal(
        split.train,
        split.valid,
        split.test,
        purge_days=purge_days,
        embargo_calendar_days=split.embargo_calendar_days,
        horizon_days=horizon,
        data_end=data_end,
        label=label,
        min_window_days=split.min_effective_window_days,
    )


def assign_split_column(
    panel: pd.DataFrame,
    split: SplitConfig,
    labels: LabelConfig,
    data_end: date | pd.Timestamp,
    *,
    date_column: str = "as_of",
    label: RiskLabel | None = None,
) -> tuple[pd.DataFrame, SplitReport]:
    """Attach the ``split`` column and describe what was cut.

    Args:
        panel: Panel with a date column.
        split: Split configuration.
        labels: Label configuration.
        data_end: Last date with data.
        date_column: Name of the timestamp column.
        label: Optional label, for a per-label split.

    Returns:
        A ``(panel_with_split, report)`` pair.

    Raises:
        KeyError: If ``date_column`` is absent.
        ValueError: If the effective windows are empty.
    """
    if date_column not in panel.columns:
        raise KeyError(f"panel is missing the date column {date_column!r}")

    windows = effective_windows(split, labels, data_end, label=label)
    dates = pd.to_datetime(panel[date_column])

    assignment = pd.Series(SPLIT_EXCLUDED, index=panel.index, dtype=object)

    def block(nominal: TimeWindow, effective: TimeWindow) -> pd.Series:
        """Rows inside the nominal window; effective-suffix rows are marked purged."""
        in_nominal = (dates >= pd.Timestamp(nominal.start)) & (dates <= pd.Timestamp(nominal.end))
        in_effective = (dates >= pd.Timestamp(effective.start)) & (
            dates <= pd.Timestamp(effective.end)
        )
        return pd.Series(
            np.where(in_effective, "keep", np.where(in_nominal, "purge", "none")),
            index=panel.index,
        )

    train_flag = block(windows.nominal_train, windows.effective_train)
    valid_flag = block(windows.nominal_valid, windows.effective_valid)
    test_flag = block(windows.nominal_test, windows.effective_test)

    # Order matters: earlier blocks win, so a row that satisfies two window tests
    # (impossible with non-overlapping nominal windows, but cheap to make total) is
    # assigned to the earlier one.
    for flag, name in (
        (test_flag, SPLIT_TEST),
        (valid_flag, SPLIT_VALID),
        (train_flag, SPLIT_TRAIN),
    ):
        assignment = assignment.where(flag != "keep", name)
    for flag in (train_flag, valid_flag, test_flag):
        assignment = assignment.where(flag != "purge", SPLIT_PURGED)

    result = panel.copy()
    result["split"] = assignment

    counts = {name: int((assignment == name).sum()) for name in SPLIT_VALUES}
    horizons = {str(target): labels.horizon_days(target) for target in labels.targets}
    report = SplitReport(
        windows=windows,
        per_label_horizons=horizons,
        counts=counts,
        per_label=split.per_label,
    )

    logger.info(
        "split assigned: train=%d valid=%d test=%d purged=%d excluded=%d",
        counts[SPLIT_TRAIN],
        counts[SPLIT_VALID],
        counts[SPLIT_TEST],
        counts[SPLIT_PURGED],
        counts[SPLIT_EXCLUDED],
    )
    if counts[SPLIT_TEST] == 0:
        logger.error(
            "the test block is empty after purging; no honest evaluation is possible. "
            "Check the nominal windows in configs/eval/default.yaml against data_end=%s",
            data_end,
        )
    return result, report


def walk_forward_folds(
    split: SplitConfig,
    labels: LabelConfig,
    data_end: date | pd.Timestamp,
    *,
    label: RiskLabel | None = None,
) -> list[WalkForwardFold]:
    """Build expanding-window folds, each with its own purge margins.

    Folds are anchored on ``split.valid.start``: fold *k* uses
    ``[valid.start + k*step, +block)`` as its validation block and the block that
    follows as its test block, with the training set expanding from ``split.train.start``.

    A single static split answers "does this work on 2021-2024". That is not enough: a
    test window can sit entirely inside one market regime, and the conclusion then does
    not extrapolate. Walk-forward folds show whether the performance is stable or
    whether it depends on when you looked.

    Args:
        split: Split configuration.
        labels: Label configuration.
        data_end: Last date with data.
        label: Optional label for per-label margins.

    Returns:
        The list of folds. Empty when the sample is too short to accommodate even one.
    """
    purge_days = resolve_purge_days(labels, split, label)
    margin = purge_days + split.embargo_calendar_days
    horizon = (
        labels.calendar_horizon_days(label)
        if label is not None
        else labels.max_calendar_horizon_days()
    )
    cutoff = pd.Timestamp(data_end).date()
    block_days = int(split.test_block_years * 365.25)
    step_days = int(split.step_years * 365.25)

    folds: list[WalkForwardFold] = []
    index = 0
    while True:
        valid_start = split.valid.start + timedelta(days=index * step_days)
        valid_end = valid_start + timedelta(days=block_days - 1)
        test_start = valid_end + timedelta(days=1)
        test_end = test_start + timedelta(days=block_days - 1)
        if test_start > cutoff:
            break

        nominal_train = TimeWindow(start=split.train.start, end=valid_start - timedelta(days=1))
        nominal_valid = TimeWindow(start=valid_start, end=valid_end)
        nominal_test = TimeWindow(start=test_start, end=min(test_end, cutoff))

        try:
            windows = _effective_from_nominal(
                nominal_train,
                nominal_valid,
                nominal_test,
                purge_days=purge_days,
                embargo_calendar_days=split.embargo_calendar_days,
                horizon_days=horizon,
                data_end=cutoff,
                label=label,
            )
        except ValueError as exc:
            logger.warning("skipping walk-forward fold %d: %s", index + 1, exc)
            index += 1
            continue

        folds.append(
            WalkForwardFold(
                name=f"W{index + 1}",
                nominal_train=nominal_train,
                nominal_valid=nominal_valid,
                nominal_test=nominal_test,
                effective_train=windows.effective_train,
                effective_valid=windows.effective_valid,
                effective_test=windows.effective_test,
            )
        )
        index += 1
        if index > 50:  # pragma: no cover - defensive against a pathological config
            logger.warning("stopping walk-forward generation after 50 folds")
            break

    if not folds:
        logger.warning(
            "no walk-forward fold fits in the sample; only the static split will be "
            "reported. margin=%d days, horizon=%d days",
            margin,
            horizon,
        )
    return folds


def fold_masks(
    panel: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    date_column: str = "as_of",
) -> dict[str, pd.Series]:
    """Boolean row masks for one fold.

    Args:
        panel: Panel with the date column.
        fold: The fold to select.
        date_column: Name of the timestamp column.

    Returns:
        Mapping ``{"train": ..., "valid": ..., "test": ...}`` of boolean series, built
        from the *effective* windows so the masks are already purged and embargoed.
    """
    dates = pd.to_datetime(panel[date_column])
    masks: dict[str, pd.Series] = {}
    for name, window in (
        ("train", fold.effective_train),
        ("valid", fold.effective_valid),
        ("test", fold.effective_test),
    ):
        masks[name] = (dates >= pd.Timestamp(window.start)) & (dates <= pd.Timestamp(window.end))
    return masks


def assert_folds_are_disjoint(folds: list[WalkForwardFold], margin_days: int) -> None:
    """Verify no fold's blocks overlap and every gap meets the margin.

    Called by the test suite. A silent overlap here would make every reported metric
    optimistic, and the failure would be invisible in the output.

    Args:
        folds: Folds to check.
        margin_days: Required separation between consecutive blocks.

    Raises:
        ValueError: On an overlapping pair or an insufficient gap.
    """
    for fold in folds:
        blocks = [
            ("train", fold.effective_train),
            ("valid", fold.effective_valid),
            ("test", fold.effective_test),
        ]
        for (name_a, window_a), (name_b, window_b) in pairwise(blocks):
            if window_b.start <= window_a.end:
                raise ValueError(
                    f"{fold.name}: {name_b} starts {window_b.start} but {name_a} ends "
                    f"{window_a.end}; the blocks overlap"
                )
            gap = (window_b.start - window_a.end).days
            if gap < margin_days:
                raise ValueError(
                    f"{fold.name}: gap between {name_a} and {name_b} is {gap} days, "
                    f"below the required margin of {margin_days}"
                )


@dataclass(slots=True)
class RollingWindow:
    """One time bucket of the panel, for stability analysis.

    Carries its own class counts so that a consumer can decide whether to compute a
    metric at all. A window with two positives produces an AUC that is real, noisy,
    and — the way these tables usually get read — indistinguishable from a stable
    one. Marking the count next to the number is the cheapest available defence.
    """

    label: str
    index: pd.Index
    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int
    n_positives: int = 0
    n_negatives: int = 0

    @property
    def sufficient(self) -> bool:
        """Whether both classes are present often enough for a rank metric to mean
        anything. Mirrors ``metrics.MIN_POSITIVES`` / ``MIN_NEGATIVES``."""
        return self.n_positives >= 2 and self.n_negatives >= 2

    def reason(self) -> str:
        """Why this window is unusable, or ``"ok"``."""
        if self.n_rows == 0:
            return "empty"
        if self.n_positives == 0:
            return "insufficient_positives"
        if self.n_positives < 2:
            return "single_positive"
        if self.n_negatives < 2:
            return "insufficient_negatives"
        return "ok"

    @property
    def positive_rate(self) -> float:
        return self.n_positives / self.n_rows if self.n_rows else float("nan")


def rolling_windows(
    panel: pd.DataFrame,
    *,
    date_column: str = "as_of",
    freq: str = "YE",
    label_column: str | None = None,
    mask_column: str | None = None,
) -> list[RollingWindow]:
    """Split the panel into consecutive time buckets, **by date, never by row**.

    This exists because the single most damaging mistake in the draft evaluation
    script was slicing by array position. Rows are ``(ticker, date)`` pairs, and row
    order has no relation to time: a 200-row window may span three days or six
    months, and if the panel is sorted by ticker — which is common — every row in the
    window belongs to one company, so the "rolling AUC" measures within-firm
    discrimination and says nothing about stability over time.

    Args:
        panel: The assembled panel.
        date_column: Column holding the decision date.
        freq: Any pandas offset alias. ``"YE"`` gives calendar years, ``"QE"``
            quarters, ``"ME"`` months.
        label_column: Optional binary label column, used to count positives.
        mask_column: Optional observability mask. When given, rows whose mask is
            false are excluded before counting — an unobservable row has no label and
            counting it as a negative would deflate the window's positive rate.

    Returns:
        Windows in chronological order. Windows that end up empty are retained with
        ``n_rows == 0`` so the report can show the gap rather than silently skipping a
        period in which nothing was observable.
    """
    if date_column not in panel.columns:
        raise KeyError(
            f"{date_column!r} is not in the panel. Grouping by time requires a date "
            "column; there is no row-position fallback."
        )

    working = panel
    if mask_column is not None and mask_column in panel.columns:
        working = panel.loc[panel[mask_column].astype(bool)]
    dates = pd.to_datetime(working[date_column])
    if dates.empty:
        return []

    windows: list[RollingWindow] = []
    for key, group in working.groupby(dates.dt.to_period(freq.replace("E", "")), sort=True):
        positions = group.index
        n_positives = 0
        n_negatives = 0
        if label_column is not None and label_column in group.columns:
            values = pd.to_numeric(group[label_column], errors="coerce")
            n_positives = int((values >= 0.5).sum())
            n_negatives = int((values < 0.5).sum())
        group_dates = pd.to_datetime(group[date_column])
        windows.append(
            RollingWindow(
                label=str(key),
                index=positions,
                start=group_dates.min(),
                end=group_dates.max(),
                n_rows=len(group),
                n_positives=n_positives,
                n_negatives=n_negatives,
            )
        )
    return windows


def rolling_window_report(windows: list[RollingWindow]) -> dict[str, object]:
    """Summary of which windows can carry a metric and which cannot.

    Returned alongside every rolling table so that a reader sees "3 of 12 windows
    had insufficient positives" rather than a row of suspiciously smooth numbers.
    """
    insufficient = [window for window in windows if not window.sufficient]
    return {
        "n_windows": len(windows),
        "n_usable": len(windows) - len(insufficient),
        "n_insufficient": len(insufficient),
        "insufficient_labels": [f"{window.label}:{window.reason()}" for window in insufficient],
        "total_positives": sum(window.n_positives for window in windows),
        "total_rows": sum(window.n_rows for window in windows),
    }


__all__ = [
    "SPLIT_EXCLUDED",
    "SPLIT_PURGED",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VALID",
    "SPLIT_VALUES",
    "EffectiveWindows",
    "RollingWindow",
    "SplitReport",
    "WalkForwardFold",
    "assert_folds_are_disjoint",
    "assign_split_column",
    "effective_windows",
    "fold_masks",
    "resolve_purge_days",
    "rolling_window_report",
    "rolling_windows",
    "walk_forward_folds",
]
