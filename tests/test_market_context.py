"""The market-context wiring: index symbols must reach the features that need them.

`compute_technical_features` has accepted ``market_returns`` and ``macro`` since it was
written, and for its whole life ``build_panel`` passed neither. The consequence was not an
error but a silence: ``beta_252d``, ``vix_level``, ``vix_chg_5d`` and
``credit_spread_chg_20d`` were NaN in every real run, and the first wide evaluation only
surfaced it as seven features "dropped with no observation in the training block". A model
with no beta and no volatility cannot separate a company falling from a market falling,
which is a plausible reading of its AUC of 0.5088 in 2020 and 0.4534 in the covid block.

These tests pin the contract at the seam that was missing: given configured symbols, the
builder produces the two arguments, and the features stop being empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from shingan.config import DataConfig
from shingan.data.builder import market_context_frames
from shingan.features.technical import compute_technical_features


def price_rows(ticker: str, closes: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ticker, "date": dates, "close": closes})


@pytest.fixture
def panel() -> pd.DataFrame:
    """A company plus both index symbols, long enough for the 252-day beta."""
    dates = pd.bdate_range("2019-01-01", periods=320)
    rng = np.random.default_rng(13)
    benchmark = 3000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    company = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, len(dates))))
    volatility = 20.0 + rng.normal(0.0, 1.0, len(dates))
    return pd.concat(
        [
            price_rows("AAA", company, dates),
            price_rows("^GSPC", benchmark, dates),
            price_rows("^VIX", volatility, dates),
        ],
        ignore_index=True,
    )


def test_nothing_configured_yields_nothing_rather_than_a_guess(panel) -> None:
    """The default state. Empty is not the same as zeros, and callers must handle None."""
    market_returns, macro = market_context_frames(panel, {})
    assert market_returns is None
    assert macro is None


def test_the_benchmark_becomes_log_returns_indexed_by_date(panel) -> None:
    market_returns, _ = market_context_frames(panel, {"benchmark": "^GSPC"})
    assert market_returns is not None
    assert isinstance(market_returns.index, pd.DatetimeIndex)

    levels = panel.loc[panel["ticker"] == "^GSPC"].sort_values("date")["close"].to_numpy()
    expected = np.diff(np.log(levels))
    assert np.allclose(market_returns.to_numpy(), expected), "log returns, not simple ones"


def test_volatility_becomes_the_vix_column_the_macro_layer_reads(panel) -> None:
    """`_attach_macro` derives vix_level and vix_chg_5d from a `vix` column."""
    _, macro = market_context_frames(panel, {"volatility": "^VIX"})
    assert macro is not None
    assert list(macro.columns) == ["date", "vix"]

    levels = panel.loc[panel["ticker"] == "^VIX"].sort_values("date")["close"].to_numpy()
    assert np.allclose(macro["vix"].to_numpy(), levels)


def test_a_role_whose_symbol_was_never_fetched_stays_empty(panel) -> None:
    """A configured symbol with no rows must not become zeros or an exception."""
    market_returns, macro = market_context_frames(panel, {"benchmark": "^RUT", "volatility": "^VIX"})
    assert market_returns is None, "no ^RUT rows, so beta stays NaN rather than betting on a default"
    assert macro is not None, "the volatility role is still usable"


def test_an_unknown_role_fails_at_load_time() -> None:
    """A role nothing reads would leave its feature NaN with no error anywhere."""
    with pytest.raises(ValidationError, match="unknown market_symbols role"):
        DataConfig(market_symbols={"equity": "SPY"})


def test_the_wiring_actually_fills_the_features_it_exists_for(panel) -> None:
    """The end of the seam: symbols in, non-NaN beta and vix out.

    This is the assertion the old code could not make. Without it the change would be
    "the plumbing is connected" rather than "the features are populated", which is exactly
    the distinction the 2026-09-30 evaluation was missing.
    """
    market_returns, macro = market_context_frames(
        panel, {"benchmark": "^GSPC", "volatility": "^VIX"}
    )
    features = compute_technical_features(
        panel, market_returns=market_returns, macro=macro, min_history_days=252
    )
    company = features.loc[features["ticker"] == "AAA"].sort_values("as_of")

    assert company["vix_level"].notna().all(), "vix needs no history, so every row has it"
    assert company["vix_chg_5d"].notna().sum() > 0
    assert company["beta_252d"].notna().sum() > 0, "beta needs the 252-day window to fill"

    without = compute_technical_features(panel, min_history_days=252)
    unplugged = without.loc[without["ticker"] == "AAA"]
    assert unplugged["vix_level"].isna().all(), "the old call shape left these structurally NaN"
    assert unplugged["beta_252d"].isna().all()


def test_macro_columns_replace_their_placeholders_instead_of_being_suffixed(panel) -> None:
    """The collision that made the macro branch unusable even when it was reached.

    The feature frame pre-creates every external feature as NaN, so a naive merge produced
    `vix_level_x` and `vix_level_y` and the canonical name vanished — values present,
    unreachable. The branch had never executed, so nothing reported it.
    """
    market_returns, macro = market_context_frames(
        panel, {"benchmark": "^GSPC", "volatility": "^VIX"}
    )
    features = compute_technical_features(
        panel, market_returns=market_returns, macro=macro, min_history_days=252
    )

    assert "vix_level" in features.columns, "the merged column must keep its own name"
    assert not [c for c in features.columns if c.endswith(("_x", "_y"))], "no suffixed leftovers"
    assert features["vix_level"].notna().any(), "and it must actually carry the macro values"

    # A feature the macro source cannot supply stays a NaN column instead of disappearing,
    # so the frame's schema does not depend on which sources happened to be available.
    assert "credit_spread_chg_20d" in features.columns
    assert features["credit_spread_chg_20d"].isna().all()
