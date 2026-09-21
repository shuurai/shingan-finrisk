"""Label construction invariants.

The first test in this file is a regression test with a specific history. Stage 2's label
review (``scripts/stage2_label_review.py``) found 225 rows in the real panel carrying
``label_mask_tail_risk = true`` while ``fwd_max_drawdown_30d`` was ``NaN`` — rows counted
as labelled negatives with no price path behind them. 153 of them fell inside
train/valid/test, where they taught the model "this feature block is absent, so the
outcome was safe". The structured test AUC read 0.8080 with them and 0.7686 without.

The cause was ordering: ``apply_risk_labels`` wrote the mask column before calling
``_tail_risk_labels``, so the label function's ``isfinite(drawdown)`` narrowing landed on a
local array that had already been copied out. Nothing in the repository tested it, because
nothing in the repository tested anything.

These tests are deliberately written against ``apply_risk_labels`` rather than the private
helper, because the bug lived in the *sequencing between them*. A test of the helper alone
would have passed while the panel stayed wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shingan.config import LabelConfig
from shingan.labeling.builders import apply_risk_labels, forward_max_drawdown
from shingan.labeling.definitions import build_label_definitions

DATA_END = pd.Timestamp("2024-12-31")


def build_panel(
    drawdowns: list[float | None],
    *,
    include_column: bool = True,
) -> pd.DataFrame:
    """A one-ticker panel whose only interesting column is the forward drawdown.

    Dates step two years apart so that a test can put ``data_end`` between two rows and
    be unambiguous about which side each one falls on, whatever the trading-day to
    calendar-day conversion happens to be.
    """
    years = pd.to_datetime([f"{2011 + 2 * step}-01-31" for step in range(len(drawdowns))])
    frame = pd.DataFrame({"ticker": "TEST", "as_of": years})
    if include_column:
        frame["fwd_max_drawdown_30d"] = [
            np.nan if value is None else value for value in drawdowns
        ]
    return frame


def label(config: LabelConfig | None = None) -> dict:
    return build_label_definitions(config or LabelConfig(targets=["tail_risk"]))


def test_mask_is_false_where_the_forward_drawdown_is_missing() -> None:
    """Regression: an absent price path must not read as an observable label."""
    # Two real drawdowns, one breach, one safe, and two rows with no price window at all.
    panel = build_panel([-0.45, -0.05, None, None])
    result = apply_risk_labels(panel, None, label(), data_end=DATA_END)

    mask = result["label_mask_tail_risk"].to_numpy()
    values = result["label_tail_risk"].to_numpy()

    np.testing.assert_array_equal(mask, [True, True, False, False]), (
        "rows with no forward drawdown must be masked, not labelled negative"
    )
    np.testing.assert_array_equal(values, [1, 0, 0, 0])
    # The count that the base rate is computed from has to agree with the mask.
    assert int(mask.sum()) == 2


def test_breach_threshold_is_inclusive() -> None:
    """``<= threshold`` — a drawdown of exactly -30% is a breach."""
    panel = build_panel([-0.30, -0.2999999])
    result = apply_risk_labels(panel, None, label(), data_end=DATA_END)
    np.testing.assert_array_equal(result["label_tail_risk"].to_numpy(), [1, 0])


def test_absent_drawdown_column_masks_everything() -> None:
    """With no forward-target column, every row is unobservable rather than safe."""
    panel = build_panel([-0.45, -0.05], include_column=False)
    result = apply_risk_labels(panel, None, label(), data_end=DATA_END)
    assert not result["label_mask_tail_risk"].any()
    assert not result["label_tail_risk"].any()


def test_open_window_is_masked() -> None:
    """A row whose horizon runs past ``data_end`` is masked, and stays zero."""
    panel = build_panel([-0.45, -0.05])
    result = apply_risk_labels(panel, None, label(), data_end=pd.Timestamp("2011-06-30"))
    mask = result["label_mask_tail_risk"].to_numpy()
    assert bool(mask[0]), "2011-01-31 + the horizon is closed by 2011-06-30"
    assert not bool(mask[1]), "2013-01-31 + the horizon is not closed by 2011-06-30"
    # A masked row must not carry a positive either, even with a deep drawdown.
    assert result["label_tail_risk"].to_numpy()[1] == 0


@pytest.mark.parametrize(
    ("close", "expected"),
    [
        # Flat then a 50% drop two steps out: the peak starts at t, so it is measured.
        ([100.0, 100.0, 50.0, 50.0], -0.5),
        # Rising throughout: no drawdown, and zero rather than NaN.
        ([100.0, 110.0, 120.0, 130.0], 0.0),
        # Drop only on the final step of the window: still inside the horizon.
        ([100.0, 100.0, 100.0, 80.0], -0.2),
    ],
)
def test_forward_max_drawdown_measures_the_running_peak(
    close: list[float], expected: float
) -> None:
    series = pd.Series(close, dtype=float)
    result = forward_max_drawdown(series, horizon=3)
    assert result.iloc[0] == pytest.approx(expected)


def test_forward_max_drawdown_is_nan_when_the_window_is_incomplete() -> None:
    """Incomplete rather than partial: a short window is not a smaller drawdown."""
    series = pd.Series([100.0, 90.0, 80.0], dtype=float)
    result = forward_max_drawdown(series, horizon=3)
    assert not np.isfinite(result.iloc[0])
    assert not np.isfinite(result.iloc[2])


def test_forward_max_drawdown_rejects_a_degenerate_horizon() -> None:
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        forward_max_drawdown(pd.Series([1.0, 2.0]), horizon=0)
