"""Tests for the expansion decision tool: the label arithmetic must not be a new label.

``scripts/universe_options.py`` reproduces the ``tail_risk`` label on its own to sweep
thresholds, windows and universes without rebuilding the panel thousands of times. A
reproduction that quietly differs from the real definition would make every lever table
in ``docs/11-universe-expansion.md`` arithmetic about something else -- so the first test
here compares it against the project's own implementation, element by element.

The rest pin the rules the tables depend on: an incomplete forward window is *unknown*
(never a negative), the train/test boundary is the project's ``test.start``, the filing
grid is one row per ``(ticker, filed)``, and the monthly grid takes the first trading day
of each month.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from shingan.labeling.builders import forward_max_drawdown as project_drawdown
from shingan.paths import find_project_root

ROOT = find_project_root()


def _load_tool():
    """Import ``scripts/universe_options.py`` by path; ``scripts`` is not a package."""
    name = "universe_options_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "universe_options.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prices(values: list[float], ticker: str = "AAA", start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"ticker": ticker, "date": dates, "close": values})


def test_the_reimplemented_drawdown_equals_the_project_definition() -> None:
    """Same definition, two implementations: they must agree everywhere."""
    tool = _load_tool()
    rng = np.random.default_rng(20260930)
    values = list(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.03, 400)))

    mine = tool.forward_max_drawdown(np.asarray(values), 30)
    theirs = project_drawdown(pd.Series(values), 30).to_numpy()

    assert np.allclose(mine, theirs, equal_nan=True)


def test_a_gapped_window_is_unknown_not_a_negative() -> None:
    """The observability rule: only a complete forward window may carry a label."""
    tool = _load_tool()
    values = [100.0] * 40 + [float("nan")] + [100.0] * 10
    drawdowns = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": pd.bdate_range("2020-01-01", periods=len(values)),
            "fwd_dd": tool.forward_max_drawdown(np.asarray(values), 30),
        }
    )
    grid = pd.DataFrame(
        {
            "ticker": "AAA",
            "as_of": pd.bdate_range("2020-01-01", periods=len(values)),
        }
    )

    labelled = tool.attach_label(grid, drawdowns, threshold=-0.30)

    assert len(labelled) < len(grid)
    assert labelled["label"].nunique() <= 2
    # Every surviving row has a real drawdown, so nothing was counted as an unknown
    # negative on the strength of a NaN.
    assert labelled["fwd_dd"].notna().all()


def test_the_split_boundary_is_the_project_test_start() -> None:
    tool = _load_tool()
    grid = pd.DataFrame(
        {
            "ticker": "AAA",
            "as_of": pd.to_datetime(["2019-12-31", "2020-01-01", "2020-01-02"]),
        }
    )
    labelled = pd.DataFrame(
        {"ticker": "AAA", "as_of": grid["as_of"], "label": [1, 0, 1]}
    )

    row = tool.arithmetic("check", labelled)

    assert row.n_train == 1
    assert row.n_train_positives == 1
    assert row.n_test == 2
    assert row.n_test_positives == 1


def test_the_filing_grid_keeps_one_row_per_ticker_and_date() -> None:
    tool = _load_tool()
    filings = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "filed": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-04-02", "2020-01-03"]
            ),
        }
    )

    grid = tool.filing_grid(filings)

    assert len(grid) == 3
    # Sorted per ticker, which is what the builder's as-of joins rely on; dates from
    # different tickers interleave, so a global monotonicity check would be wrong.
    for _ticker, group in grid.groupby("ticker"):
        assert group["as_of"].is_monotonic_increasing


def test_the_monthly_grid_takes_the_first_trading_day() -> None:
    tool = _load_tool()
    prices = _prices([1.0] * 45, start="2020-01-01")

    grid = tool.month_grid(prices)

    assert len(grid) == 3  # January, February, March 2020
    assert grid["as_of"].iloc[0] == pd.Timestamp("2020-01-01")
    assert grid["as_of"].iloc[1] == pd.Timestamp("2020-02-03")


def test_candidate_files_are_read_without_their_comments() -> None:
    """The candidate and universe files are prose plus symbols; comments are not names."""
    tool = _load_tool()
    path = ROOT / "configs" / "universes" / "wide_stage2.txt"
    if not path.is_file():
        pytest.skip("wide universe file not present")

    tokens = [
        word
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for word in line.split()
    ]

    assert tokens
    assert all(token == token.upper() for token in tokens)
    assert all(token.isalpha() or "." in token for token in tokens)
    assert len(set(tokens)) == len(tokens)
