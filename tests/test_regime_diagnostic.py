"""Tests for the regime diagnostic: the oracle's sign and its denominators.

`scripts/regime_diagnostic.py` answers "is `tail_risk` a company event or a market event?"
with numbers that a reader will act on -- it is the argument for or against spending GPU
time on market features. Two ways it could lie without erroring, so both are pinned here:

* the oracle score is negated so that a deeper benchmark fall ranks higher. A dropped
  minus sign inverts the entire conclusion while every printed number stays finite.
* every share is taken against the block's *positives*. Taken against all rows it would
  be dominated by the ~97% that are negative, and would read as a near-100% market story
  whether or not one existed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from shingan.paths import find_project_root

ROOT = find_project_root()


@pytest.fixture(scope="module")
def diag():
    name = "regime_diagnostic_under_test"
    import sys

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "regime_diagnostic.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def price_frame(closes: list[float], start: str = "2020-01-01", ticker: str = "^GSPC") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"ticker": ticker, "date": dates, "close": closes})


def block_with(rows: list[tuple[int, float]]) -> pd.DataFrame:
    """``(label, benchmark forward drawdown)`` rows, all observable, all in `test`."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * len(rows),
            "as_of": pd.date_range("2020-01-01", periods=len(rows), freq="D"),
            "split": ["test"] * len(rows),
            "label_tail_risk": [row[0] for row in rows],
            "label_mask_tail_risk": [1] * len(rows),
            "bm_fwd_drawdown": [row[1] for row in rows],
        }
    )


def test_the_benchmark_drawdown_uses_the_labels_own_definition(diag) -> None:
    """Reused rather than reimplemented, so the oracle cannot drift from the label."""
    frame = diag.benchmark_forward_drawdown(price_frame([100, 90, 80, 70, 60, 50]), "^GSPC", horizon=2)
    assert frame["bm_fwd_drawdown"].iloc[0] == pytest.approx(-0.20), "100 -> 80 is a 20% fall"
    assert frame["bm_fwd_drawdown"].iloc[1] == pytest.approx(-0.2222, abs=1e-4)
    assert np.isnan(frame["bm_fwd_drawdown"].iloc[-2:]).all(), "an incomplete window is NaN, not partial"


def test_a_missing_benchmark_says_how_to_get_it(diag) -> None:
    with pytest.raises(ValueError, match="market-symbols-only"):
        diag.benchmark_forward_drawdown(price_frame([100, 90], ticker="AAPL"), "^GSPC")


def test_the_benchmark_joins_backward_and_respects_the_tolerance(diag) -> None:
    """A filing date can land on a holiday, but a stale benchmark is worse than a gap."""
    benchmark = diag.benchmark_forward_drawdown(
        price_frame([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]), "^GSPC", horizon=2
    )
    panel = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "as_of": pd.to_datetime(["2020-01-01", "2020-01-20"]),
            "split": ["test", "test"],
            "label_tail_risk": [1, 0],
            "label_mask_tail_risk": [1, 1],
        }
    )
    joined = diag.attach_benchmark(panel, benchmark)
    assert np.isfinite(joined["bm_fwd_drawdown"].iloc[0]), "an exact date must match"
    assert np.isnan(joined["bm_fwd_drawdown"].iloc[1]), "19 days beyond the benchmark ends is not a match"


def test_shares_are_taken_against_positives_not_all_rows(diag) -> None:
    """The denominator is the whole point: most rows are negative and would swamp it."""
    rows = [(1, -0.20)] * 5 + [(1, 0.0)] * 5 + [(0, -0.20)] * 90
    stats = diag.describe_block(block_with(rows))
    assert stats["positives"] == 10
    assert stats["benchmark_in_correction"] == pytest.approx(0.5), "5 of 10 positives, not 95 of 100"


def test_a_deeper_benchmark_fall_ranks_higher(diag) -> None:
    """The sign. Inverted, the diagnostic would argue the exact opposite conclusion."""
    aligned = block_with([(1, -0.40), (1, -0.35), (0, -0.02), (0, 0.0)])
    assert diag.oracle_auc(aligned) == pytest.approx(1.0)
    inverted = block_with([(0, -0.40), (0, -0.35), (1, -0.02), (1, 0.0)])
    assert diag.oracle_auc(inverted) == pytest.approx(0.0)


def test_a_year_with_no_positives_does_not_break_the_report(diag) -> None:
    """Train 2005 genuinely holds zero positives in the real panel."""
    frame = block_with([(1, -0.30), (1, -0.25), (0, -0.01), (0, 0.0)])
    frame.loc[0, "as_of"] = pd.Timestamp("2004-06-01")
    frame.loc[1, "as_of"] = pd.Timestamp("2004-07-01")
    frame.loc[2, "as_of"] = pd.Timestamp("2005-06-01")
    frame.loc[3, "as_of"] = pd.Timestamp("2020-03-01")
    frame.loc[[0, 1, 2], "split"] = "train"
    report = diag.format_report(frame, "^GSPC", 30)
    assert "train block by year" in report
    year_2005 = next(line for line in report.splitlines() if line.strip().startswith("2005"))
    assert "n/a" in year_2005, "a block with no positives has an unknown rate, not a zero one"
    assert "0.0%" not in year_2005, "printing 0.0% would assert the market was calm, which is not measured"
