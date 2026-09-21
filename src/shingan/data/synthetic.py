"""Deterministic synthetic panel: the offline backbone of the demo and the tests.

**Why this exists.** Real data drifts. EDGAR backfills filings, vendors revise
prices, and a yfinance endpoint changes shape without notice. A demo that depends on
live data is not a regression test — it is a weather report. This generator produces
the same bytes on every machine, in seconds, on a CPU, with no network, so the whole
pipeline can be run end to end in CI.

**The one design decision that matters.** The generator plants a latent risk state
per company, then makes it observable through two channels with *different
information content*:

* The **structured** channel sees a lossy projection of the latent state: the state
  is scaled down, part of it is discarded (``structure_information_loss``), and
  correlated noise is added. Structured-only performance therefore has a ceiling by
  construction.
* The **text** channel sees the full state, including the component the structured
  side threw away (``text_component_strength``), rendered through financial
  phrasing with distractors.

The consequence is the property the project needs: a fusion model that cannot beat
structured-only is a bug in the fusion, not a fact about text.

**What this does not establish.** The claim "text adds signal" is assumed *into* the
generator; it is not discovered by it. A positive fusion gain on synthetic data
verifies that the pipeline is wired correctly. It is evidence about the *code*, and
exactly zero evidence about real markets. Any real conclusion requires real data —
see `docs/07-roadmap.md` stage 2.

Every row is marked ``is_synthetic = True`` and :mod:`shingan.data.builder` refuses to
mix synthetic and real rows in one panel, so a screenshot cannot be mistaken for a
real result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pandas as pd

from shingan.config import SyntheticConfig
from shingan.features.text import (
    CONSTRAINING_TERMS,
    GOING_CONCERN_PHRASES,
    LITIGIOUS_TERMS,
    NEGATIVE_TERMS,
    RESTATEMENT_PHRASES,
    UNCERTAINTY_TERMS,
)
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Plausible company identifiers. Deliberately fictional: a synthetic panel labelled
#: with real tickers invites a reader to believe the numbers describe those companies.
SYNTHETIC_TICKERS: tuple[str, ...] = (
    "SXAA",
    "SXAB",
    "SXAC",
    "SXAD",
    "SXAE",
    "SXAF",
    "SXAG",
    "SXAH",
    "SXAI",
    "SXAJ",
    "SXAK",
    "SXAL",
)

SYNTHETIC_SECTORS: tuple[str, ...] = (
    "Industrials",
    "Financials",
    "Energy",
    "Consumer Discretionary",
    "Materials",
    "Information Technology",
)

#: Regime multipliers on daily volatility, placed to mimic the shocks in the sample
#: window. They matter because a stress test that has no stressed period to test is
#: not a stress test.
#:
#: ``(start, end, volatility_multiplier)``. The first two entries are ISO date
#: *strings*, not ``date`` objects, because the multipliers are applied by string
#: comparison against the generated calendar; the annotation says so, which the
#: previous ``tuple[str, float, float]`` did not.
_REGIME_SHOCKS: tuple[tuple[str, str, float], ...] = (
    ("2020-02-20", "2020-04-30", 3.4),
    ("2022-01-01", "2022-12-31", 1.7),
    ("2011-08-01", "2011-10-31", 1.6),
)

_DOC_TYPES: tuple[str, ...] = ("10-K", "10-Q", "10-Q", "10-Q")

#: Business days per year, used to convert an annual event incidence rate into a
#: per-day hazard. Deliberately *not* shared with ``eval.backtest``'s annualisation
#: factor: that one converts a daily return series into an annual figure, this one
#: converts a yearly event rate into a daily probability. They coincide at 252 today
#: and mean different things, so coupling them would make one of them wrong.
_BUSINESS_DAYS_PER_YEAR = 252

#: Expected number of events per company per **year**, at the population-average
#: risk state. The per-business-day hazard is derived from these, so the realised
#: base rate does not move when the panel window or ``rows_per_year`` changes.
#:
#: **Calibration constants, not empirical estimates.** They are chosen to land the
#: three labels in the bands ``docs/03-labeling.md`` specifies. The anchors are only
#: order-of-magnitude: credit rating actions touch a low single-digit percentage of
#: issuers a year, restatements around 1%, enforcement actions and adverse audit
#: opinions rarer still. The dataset card must state that these are chosen rather
#: than fitted.
_EVENT_ANNUAL_RATES: dict[str, float] = {
    "rating_downgrade": 0.13,
    "payment_default": 0.008,
    "restatement": 0.010,
    "enforcement_action": 0.004,
    "adverse_audit_opinion": 0.006,
}

#: How hard the latent risk state bends the hazard, in log space. The multiplier is
#: ``exp(_HAZARD_LOG_SPREAD * (exposure - mean_exposure))``, so it is exactly 1.0 at
#: the panel's average risk state and strictly positive everywhere — an additive form
#: cannot make that promise without going negative on the low-risk tail. At the
#: observed spread this gives the riskiest company roughly five times the baseline
#: hazard and the safest about a third.
_HAZARD_LOG_SPREAD = 2.5

#: Magnitude distribution of a rating action, in notches. Most actions move an issuer
#: a single notch, which is precisely why ``default_risk`` (two notches or more) is
#: materially rarer than "a downgrade happened" — a uniform draw over 1..4 notches
#: would make three quarters of all downgrades label-positive and overstate the base
#: rate by roughly a factor of two.
_DOWNGRADE_NOTCH_WEIGHTS: tuple[tuple[int, float], ...] = (
    (1, 0.62),
    (2, 0.24),
    (3, 0.10),
    (4, 0.04),
)

#: The base-rate band each label is calibrated into, as ``(low, high)`` fractions of
#: observable rows. ``default_risk`` and ``fraud_risk`` come out of
#: :data:`_EVENT_ANNUAL_RATES`; ``tail_risk`` is a property of the price process and
#: is set by the volatility in :func:`_synthesise_prices`. Published so the
#: regression test, the dataset card and this docstring cite one source.
TARGET_BASE_RATES: dict[str, tuple[float, float]] = {
    "default_risk": (0.015, 0.08),
    "fraud_risk": (0.005, 0.04),
    "tail_risk": (0.02, 0.15),
}

_NEWS_SOURCES: tuple[str, ...] = ("wire", "broadsheet", "trade_press", "analyst_note")

#: Share of the daily return variance explained by the common market factor. The two
#: loadings in :func:`_synthesise_prices` are ``sqrt(share)`` and ``sqrt(1 - share)``,
#: so a name's returns have exactly ``daily_vol`` of total daily volatility whatever
#: this is set to, and the return correlation between two names is approximately this
#: number. The earlier form weighted a unit-variance idiosyncratic shock and a
#: 0.009-standard-deviation "market" shock by 0.55 and 0.45: only the idiosyncratic
#: term had any scale, so the realised volatility was ``daily_vol * 0.45`` — under a
#: quarter of the intended variance — and the market term contributed nothing.
_MARKET_VARIANCE_SHARE = 0.30

#: Daily volatility, as ``base + per_state * latent``. The latent state rises with
#: risk, so drawdowns concentrate where the latent state is high, which is what makes
#: ``tail_risk`` learnable and gives the stress-period slices something to find. The
#: level is calibrated (with the market share fixed) so ``tail_risk`` lands inside
#: :data:`TARGET_BASE_RATES`; at the default latent state this is about 2% daily.
_DAILY_VOL_BASE = 0.014
_DAILY_VOL_PER_STATE = 0.038

_NARRATIVE_SENTENCES: tuple[str, ...] = (
    "Revenue for the period was in line with the prior year.",
    "The company continues to invest in its core operating footprint.",
    "Management remains focused on cost discipline and working capital.",
    "Demand conditions in the principal end markets were broadly stable.",
    "The board declared a regular quarterly distribution to shareholders.",
)


@dataclass(slots=True)
class SyntheticDataset:
    """The five tables a real build would fetch, generated offline."""

    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    filings: pd.DataFrame
    news: pd.DataFrame
    events: pd.DataFrame
    latent: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        """Row counts and date ranges, for logs and run metadata."""
        return {
            "companies": int(self.prices["ticker"].nunique()) if len(self.prices) else 0,
            "price_rows": int(len(self.prices)),
            "fundamental_rows": int(len(self.fundamentals)),
            "filing_rows": int(len(self.filings)),
            "news_rows": int(len(self.news)),
            "event_rows": int(len(self.events)),
            "events_by_kind": (
                self.events["event_kind"].value_counts().to_dict() if len(self.events) else {}
            ),
            **self.metadata,
        }


def _regime_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Daily volatility multiplier from the configured regime shocks."""
    multiplier = np.ones(len(dates), dtype=float)
    for start, end, factor in _REGIME_SHOCKS:
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        multiplier[mask] *= factor
    return multiplier


def _latent_paths(
    rng: np.random.Generator,
    tickers: list[str],
    dates: pd.DatetimeIndex,
    *,
    noise: float,
) -> pd.DataFrame:
    """Simulate each company's latent risk state over the calendar.

    An AR(1) with mean reversion toward a per-company base level. Persistence is the
    point: risk states cluster, and a generator that redrew the state every period
    would produce a panel on which no time-series model could do better than a
    cross-sectional one — which would not resemble the problem.

    Returns:
        Long frame with ``ticker``, ``date`` and ``latent`` in roughly ``[0, 1]``.
    """
    records: list[pd.DataFrame] = []
    persistence = 0.88
    for ticker in tickers:
        base = float(rng.uniform(0.18, 0.42))
        shocks = rng.normal(0.0, noise * 0.10, size=len(dates))
        path = np.empty(len(dates), dtype=float)
        state = base
        for position in range(len(dates)):
            state = persistence * state + (1.0 - persistence) * base + shocks[position]
            path[position] = state
        records.append(
            pd.DataFrame({"ticker": ticker, "date": dates, "latent": np.clip(path, 0.0, 1.0)})
        )
    return pd.concat(records, ignore_index=True)


def _synthesise_prices(
    rng: np.random.Generator,
    latent: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    inject_edge_cases: bool,
) -> pd.DataFrame:
    """Generate a daily OHLCV panel driven by the latent risk state.

    The mechanism is deliberately simple and directional: higher latent risk means a
    lower drift and a higher volatility, so drawdowns concentrate where the latent
    state is high. That is what makes ``tail_risk`` learnable and gives the
    stress-period slices something to find.

    Returns:
        Long frame with ``ticker``, ``date``, ``open``, ``high``, ``low``, ``close``,
        ``volume`` and ``shares_outstanding``.
    """
    regime = _regime_multiplier(dates)
    market_shocks = rng.normal(0.0, 1.0, size=len(dates)) * regime
    market_loading = float(np.sqrt(_MARKET_VARIANCE_SHARE))
    idiosyncratic_loading = float(np.sqrt(1.0 - _MARKET_VARIANCE_SHARE))
    blocks: list[pd.DataFrame] = []

    for ticker, group in latent.groupby("ticker", sort=False, observed=True):
        state = group.sort_values("date")["latent"].to_numpy()
        idiosyncratic = rng.normal(0.0, 1.0, size=len(dates)) * regime
        daily_vol = _DAILY_VOL_BASE + _DAILY_VOL_PER_STATE * state
        drift = 0.0004 - 0.0016 * state
        returns = drift + daily_vol * (
            market_loading * market_shocks + idiosyncratic_loading * idiosyncratic
        )
        close = 45.0 * np.exp(np.cumsum(returns))

        intraday = np.abs(rng.normal(0.0, daily_vol))
        high = close * (1.0 + intraday)
        low = close * (1.0 - intraday)
        open_price = low + (high - low) * rng.uniform(0.2, 0.8, size=len(dates))

        base_volume = float(rng.uniform(2e6, 3e7))
        volume = base_volume * np.exp(rng.normal(0.0, 0.4, size=len(dates))) * (1.0 + 2.0 * state)
        shares = float(rng.uniform(3e8, 2.5e9))

        blocks.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": np.round(volume),
                    "shares_outstanding": shares,
                }
            )
        )

    prices = pd.concat(blocks, ignore_index=True)

    if inject_edge_cases:
        prices = _inject_price_edge_cases(rng, prices, dates)
    return prices


def _inject_price_edge_cases(
    rng: np.random.Generator,
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Add the awkward rows that every real panel contains.

    Applied to the last three synthetic tickers, whose identifiers are therefore
    meaningful in the generated dataset:

    * ``SXA?`` with a long halt — a run of missing prices, to exercise the gap rules.
    * ``SXA?`` with a truncated history — a company that stopped existing.
    * ``SXA?`` with a very short history — fewer rows than ``min_history_days``.

    Without these, the gap handling, the ``insufficient_history`` flag and the
    completeness checks in the label builder would all be untested code.
    """
    tickers = sorted(prices["ticker"].unique())
    if len(tickers) < 3:  # pragma: no cover - guarded by n_companies >= 2
        return prices

    halted, truncated, short_history = tickers[-3], tickers[-2], tickers[-1]

    # A 25 trading day halt, well beyond the default max_nan_run of 10.
    halt_mask = (prices["ticker"] == halted) & (
        prices["date"].between(dates[200], dates[225])
    )
    price_columns = ["open", "high", "low", "close", "volume"]
    prices.loc[halt_mask, price_columns] = np.nan

    # A company whose series simply stops: delisting or acquisition.
    cutoff = dates[int(len(dates) * 0.72)]
    prices = prices.loc[~((prices["ticker"] == truncated) & (prices["date"] > cutoff))]

    # A recent listing with almost no history.
    recent = dates[int(len(dates) * 0.95)]
    prices = prices.loc[~((prices["ticker"] == short_history) & (prices["date"] < recent))]

    logger.debug(
        "injected edge cases: halt=%s (25 days), delisting=%s after %s, short history=%s",
        halted,
        truncated,
        cutoff.date(),
        short_history,
    )
    return prices.reset_index(drop=True)


def _synthesise_fundamentals(
    rng: np.random.Generator,
    latent: pd.DataFrame,
    config: SyntheticConfig,
) -> pd.DataFrame:
    """Generate quarterly-ish fundamentals with a lossy dependence on the latent state.

    This is where the structured track's information ceiling is set. The latent state
    is scaled by ``(1 - text_component_strength)``, a further
    ``structure_information_loss`` share is discarded, and correlated noise is added.
    A line item is then dropped with probability ``missing_rate`` so that imputation
    and ``ratios_missing_frac`` are exercised.
    """
    rows: list[dict[str, object]] = []
    structure_weight = 1.0 - config.text_component_strength
    surviving = 1.0 - config.structure_information_loss

    for ticker, group in latent.groupby("ticker", sort=False, observed=True):
        ordered = group.sort_values("date")
        # One observation per `rows_per_year`, approximating a filing cadence.
        step = max(1, int(round(252 / config.rows_per_year)))
        sampled = ordered.iloc[::step]

        revenue_base = float(rng.uniform(2e9, 5e10))
        asset_base = revenue_base * float(rng.uniform(0.8, 2.2))
        equity_base = asset_base * float(rng.uniform(0.25, 0.5))
        shares = float(rng.uniform(3e8, 2.5e9))

        for _, row in sampled.iterrows():
            state = float(row["latent"])
            visible = state * structure_weight * surviving
            jitter = rng.normal(0.0, 0.08)

            revenue = revenue_base * (1.0 + 0.12 * np.sin(state * 3.0) + jitter)
            net_income = revenue * (0.09 - 0.16 * visible + jitter * 0.4)
            total_assets = asset_base * (1.0 + 0.05 * jitter)
            total_liabilities = total_assets * (0.55 + 0.30 * visible + 0.02 * jitter)
            total_equity = total_assets - total_liabilities
            current_assets = total_assets * 0.32
            current_liabilities = current_assets * (0.55 + 0.35 * visible)
            ebit = net_income * (1.0 + rng.uniform(0.2, 0.5))
            interest_expense = max(ebit * (0.25 - 0.22 * visible), 1.0)
            operating_cash_flow = net_income * (1.0 + rng.normal(0.0, 0.25))
            capex = revenue * float(rng.uniform(0.02, 0.06))
            goodwill = total_assets * float(rng.uniform(0.02, 0.18))

            record: dict[str, object] = {
                "ticker": ticker,
                "filed": pd.Timestamp(row["date"]),
                "period_end": pd.Timestamp(row["date"]),
                "revenue": float(revenue),
                "net_income": float(net_income),
                "total_assets": float(total_assets),
                "total_liabilities": float(total_liabilities),
                "total_equity": float(total_equity),
                "current_assets": float(current_assets),
                "current_liabilities": float(current_liabilities),
                "inventory": float(current_assets * 0.3),
                "ebit": float(ebit),
                "interest_expense": float(interest_expense),
                "operating_cash_flow": float(operating_cash_flow),
                "capex": float(capex),
                "goodwill": float(goodwill),
                "retained_earnings": float(total_equity * rng.uniform(0.2, 0.7)),
                "short_term_debt": float(total_liabilities * rng.uniform(0.15, 0.5)),
                "total_debt": float(total_liabilities * rng.uniform(0.5, 0.9)),
                "market_cap": float(shares * (10.0 + 120.0 * (1.0 - state))),
            }

            for item in (
                "revenue",
                "net_income",
                "total_assets",
                "total_liabilities",
                "total_equity",
                "current_assets",
                "current_liabilities",
                "ebit",
                "interest_expense",
                "operating_cash_flow",
                "capex",
            ):
                if rng.random() < config.missing_rate:
                    record[item] = np.nan
            rows.append(record)

    return pd.DataFrame.from_records(rows)


def _synthesise_filings(
    rng: np.random.Generator,
    latent: pd.DataFrame,
    config: SyntheticConfig,
) -> pd.DataFrame:
    """Generate filing sections whose wording intensity tracks the latent state.

    The latent state is rendered into text *in full* — it is not passed through the
    lossy projection the fundamentals use. Phrases are drawn from the same lexicons
    :mod:`shingan.features.text` scores, so the count-based features and a language
    model have something real to find; the surrounding boilerplate is distractors.
    """
    rows: list[dict[str, object]] = []
    step = max(1, int(round(252 / config.rows_per_year)))

    for ticker, group in latent.groupby("ticker", sort=False, observed=True):
        ordered = group.sort_values("date")
        sampled = ordered.iloc[::step]
        for position, (_, row) in enumerate(sampled.iterrows()):
            state = float(row["latent"])
            intensity = int(np.clip(state * 14.0, 0, 18))

            risk_sentences = [
                rng.choice(NEGATIVE_TERMS) for _ in range(intensity)
            ] + [rng.choice(UNCERTAINTY_TERMS) for _ in range(max(1, intensity // 2))]
            if state > 0.62:
                risk_sentences.append(rng.choice(RESTATEMENT_PHRASES))
                risk_sentences.append(rng.choice(CONSTRAINING_TERMS))
            if state > 0.75:
                risk_sentences.append(rng.choice(GOING_CONCERN_PHRASES))
            if state > 0.55:
                risk_sentences.append(rng.choice(LITIGIOUS_TERMS))

            narrative = [str(rng.choice(_NARRATIVE_SENTENCES)) for _ in range(6)]
            padding = [
                "The following discussion should be read together with the consolidated "
                "financial statements and the related notes included elsewhere in this report."
            ] * int(np.clip(round((1.0 - state) * 8), 1, 8))

            risk_body = ". ".join(risk_sentences + narrative + padding) + "."
            mdna_body = ". ".join(narrative + risk_sentences[: max(1, intensity // 3)]) + "."

            doc_type = _DOC_TYPES[position % len(_DOC_TYPES)]
            filed = pd.Timestamp(row["date"])
            rows.append(
                {
                    "ticker": ticker,
                    "filed": filed,
                    "doc_type": doc_type,
                    "section": "Item 1A",
                    "text": f"Item 1A. Risk Factors. {risk_body}",
                    "accession": f"{ticker}-{filed.strftime('%Y%m%d')}-1A",
                }
            )
            rows.append(
                {
                    "ticker": ticker,
                    "filed": filed,
                    "doc_type": doc_type,
                    "section": "Item 7",
                    "text": f"Item 7. Management's Discussion and Analysis. {mdna_body}",
                    "accession": f"{ticker}-{filed.strftime('%Y%m%d')}-7",
                }
            )

    return pd.DataFrame.from_records(rows)


def _synthesise_news(
    rng: np.random.Generator,
    latent: pd.DataFrame,
    config: SyntheticConfig,
) -> pd.DataFrame:
    """Generate news items with sentiment and volume driven by the latent state.

    High-risk states produce more articles and more negative ones, so the news
    aggregates are informative. Sentiment carries noise on purpose: the documentation
    is explicit that a provided sentiment score is a feature, never a label.
    """
    rows: list[dict[str, object]] = []
    dates = pd.DatetimeIndex(sorted(latent["date"].unique()))

    for ticker, group in latent.groupby("ticker", sort=False, observed=True):
        ordered = group.sort_values("date").reset_index(drop=True)
        sampled = ordered.iloc[:: max(1, len(ordered) // 60)]
        for _, row in sampled.iterrows():
            state = float(row["latent"])
            n_articles = int(rng.poisson(0.35 + 4.0 * state))
            for index in range(n_articles):
                shift = int(rng.integers(0, 12))
                published = pd.Timestamp(row["date"]) + pd.Timedelta(days=shift)
                if published > dates[-1]:
                    continue
                sentiment = float(np.clip(rng.normal(0.15 - 1.5 * state, 0.35), -1.0, 1.0))
                if sentiment < -0.15:
                    body = (
                        f"Analysts flagged {rng.choice(NEGATIVE_TERMS)} trends and questioned "
                        f"the outlook; shares traded lower on elevated volume."
                    )
                    title = f"{ticker} draws scrutiny after {rng.choice(NEGATIVE_TERMS)} update"
                else:
                    body = (
                        f"Coverage noted stable demand and unchanged guidance; "
                        f"the quarter was described as uneventful."
                    )
                    title = f"{ticker} results broadly in line with expectations"
                rows.append(
                    {
                        "ticker": ticker,
                        "published": published,
                        "source": str(rng.choice(_NEWS_SOURCES)),
                        "title": title,
                        "body": body,
                        "sentiment": sentiment,
                    }
                )

    news = pd.DataFrame.from_records(rows)
    if news.empty:  # pragma: no cover - only with a degenerate configuration
        return pd.DataFrame(columns=["ticker", "published", "source", "title", "body", "sentiment"])
    return news.sort_values(["ticker", "published"]).reset_index(drop=True)


def _draw_downgrade_notches(rng: np.random.Generator) -> float:
    """Draw the magnitude of one rating action, in notches, from the weights."""
    notches = np.array([n for n, _ in _DOWNGRADE_NOTCH_WEIGHTS], dtype=float)
    weights = np.array([w for _, w in _DOWNGRADE_NOTCH_WEIGHTS], dtype=float)
    return float(rng.choice(notches, p=weights / weights.sum()))


def _synthesise_events(
    rng: np.random.Generator,
    latent: pd.DataFrame,
    *,
    base_rate_multiplier: float,
    label_noise: float,
) -> pd.DataFrame:
    """Generate a sparse event table whose hazard rises with the latent state.

    Each label gets its own hazard so the three base rates differ, as they do in
    reality: ``tail_risk`` is comparatively common, ``fraud_risk`` rare.

    The severity field is label-specific, matching how
    :func:`shingan.labeling.definitions.is_positive` reads it — notch count for a
    downgrade, breach depth for a drawdown. Events that would not reach the label
    threshold are still emitted, because a one-notch downgrade is a real event that
    simply does not make the label positive; dropping it would hide the distinction.

    **Calibration.** The hazard is written down as an annual incidence rate per
    company (:data:`_EVENT_ANNUAL_RATES`) and converted to a per-business-day
    probability, so the base rate a label ends up with does not silently change when
    the panel window or ``rows_per_year`` changes. The latent state scales that
    probability through :data:`_HAZARD_LOG_SPREAD`, centred on the *realised* mean of
    the noisy exposure. Centring is the load-bearing part: the earlier form
    ``base * (1.0 + 18.0 * effective)`` multiplied every hazard by roughly seven
    (because ``effective`` averages about 0.33, not zero), which is how a
    "0.16% per day" downgrade hazard produced ~40 downgrades per company over the
    sample and a ``default_risk`` positive rate of 93%.

    These constants are **generator calibration, not empirical estimates**. They are
    chosen to land the three labels in the bands that ``docs/03-labeling.md``
    specifies — single-digit percentages, with ``fraud_risk`` at the low end — and
    the dataset card has to say so. See ``TARGET_BASE_RATES`` for the bands and
    ``tests/test_synthetic.py`` for the regression test that holds them.
    """
    rows: list[dict[str, object]] = []
    date_values = pd.DatetimeIndex(sorted(latent["date"].unique()))

    # Exposure is drawn for every company-day *before* any hazard is applied,
    # because the multiplier is centred on the population mean of the exposure and
    # that mean is not known until every draw has been made.
    exposures: list[tuple[str, np.ndarray, np.ndarray]] = []
    for ticker, group in latent.groupby("ticker", sort=False, observed=True):
        ordered = group.sort_values("date").reset_index(drop=True)
        state = ordered["latent"].to_numpy(dtype=float)
        effective = np.clip(state + rng.normal(0.0, label_noise, size=len(state)), 0.0, 1.0)
        # ``str()`` because the groupby key's inferred type is the whole union of
        # column dtypes, not ``str``, even though this column only ever holds tickers.
        exposures.append((str(ticker), ordered["date"].to_numpy(), effective))

    reference = float(np.concatenate([values for _, _, values in exposures]).mean())

    for ticker, ticker_dates, effective in exposures:
        # Geometric in the exposure, so the multiplier is strictly positive for any
        # spread and averages to 1 across the panel by construction.
        multiplier = np.exp(_HAZARD_LOG_SPREAD * (effective - reference))
        for kind, annual_rate in _EVENT_ANNUAL_RATES.items():
            per_day = 1.0 - (1.0 - annual_rate * base_rate_multiplier) ** (
                1.0 / _BUSINESS_DAYS_PER_YEAR
            )
            # Scaling a per-day probability by a risk multiplier: the natural reading
            # is "this many independent chances", which keeps the result in [0, 1]
            # for every multiplier, unlike a plain product.
            probability = 1.0 - (1.0 - per_day) ** multiplier
            hits = np.flatnonzero(rng.random(len(ticker_dates)) < probability)
            for position in hits:
                if kind == "rating_downgrade":
                    severity = _draw_downgrade_notches(rng)
                elif kind == "payment_default":
                    # Severe by construction: a payment default is a default, and
                    # ``is_positive`` accepts it without a magnitude test.
                    severity = 99.0
                else:
                    severity = 1.0
                rows.append(
                    {
                        "ticker": ticker,
                        "event_date": pd.Timestamp(ticker_dates[position]),
                        "event_kind": kind,
                        "severity": severity,
                        "source": "synthetic",
                    }
                )

    events = pd.DataFrame.from_records(rows)
    if events.empty:  # pragma: no cover - requires a near-zero rate multiplier
        events = pd.DataFrame(
            columns=["ticker", "event_date", "event_kind", "severity", "source"]
        )

    # A single injected restatement, but only when the hazard produced no fraud event
    # at all. The stated purpose was to keep ``fraud_risk`` testable in a tiny
    # configuration; injecting it unconditionally would also move the base rate of a
    # normally-sized run, and a calibration constant that is only correct on one side
    # of a conditional is not a calibration constant.
    #
    # The empty case has to be included in the condition, not excluded from it. An
    # earlier version guarded on ``not events.empty`` first, so the one configuration
    # that most needs the fallback — a rate multiplier small enough to produce nothing
    # — was the one configuration that did not get it, and every event-based label
    # came out all-negative.
    fraud_kinds = {"restatement", "enforcement_action", "adverse_audit_opinion"}
    has_fraud_event = not events.empty and bool(events["event_kind"].isin(fraud_kinds).any())
    if not has_fraud_event:
        logger.warning(
            "no fraud-kind event was generated from the hazard%s; injecting one "
            "restatement so fraud_risk stays testable",
            "" if not events.empty else " (no events at all)",
        )
        events = pd.concat(
            [
                events,
                pd.DataFrame(
                    [
                        {
                            "ticker": sorted(latent["ticker"].unique())[0],
                            "event_date": pd.Timestamp(date_values[int(len(date_values) * 0.6)]),
                            "event_kind": "restatement",
                            "severity": 1.0,
                            "source": "synthetic_injected",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return events.sort_values(["ticker", "event_date"]).reset_index(drop=True)


#: Date column each table is filtered on when coverage is aligned. ``prices`` is
#: absent on purpose — it *defines* the coverage rather than being trimmed to it.
_COVERAGE_DATE_COLUMNS: dict[str, str] = {
    "fundamentals": "filed",
    "filings": "filed",
    "news": "published",
    "events": "event_date",
}


def _apply_coverage(dataset: SyntheticDataset) -> SyntheticDataset:
    """Trim every table to the window each company's prices actually cover.

    ``_inject_price_edge_cases`` deletes price rows to create a delisted company and
    a recent listing. Leaving the other tables at full length made the panel
    internally impossible: the "recent listing" still filed 10-Qs from 2010, and the
    delisted company kept filing for years after its last traded price. The label
    rows built from those filings are unlearnable — every price-derived feature is
    ``NaN`` — and they inflate the measured base rate with positives no model could
    ever have seen. On the default configuration this manufactured 8 of the 16
    ``fraud_risk`` positives, on a company with no price history at those dates.

    A halt is a *gap* inside a covered window, not a change of coverage, so a company
    whose prices are halted keeps its filings: trading stopped, filing did not. Only
    the two companies whose series genuinely start or stop are trimmed.

    Args:
        dataset: The freshly generated dataset.

    Returns:
        The dataset with fundamentals, filings, news and events restricted to each
        company's price coverage. Companies absent from the price table lose all
        their rows in the other tables, which is the correct reading of "no prices".
    """
    observed = dataset.prices.dropna(subset=["close"])
    if observed.empty:  # pragma: no cover - only with a degenerate configuration
        return dataset

    sorted_prices = observed.sort_values(["ticker", "date"])
    grouped = sorted_prices.groupby("ticker", sort=False, observed=True)["date"]
    first = pd.to_datetime(grouped.min())
    last = pd.to_datetime(grouped.max())
    first_by_ticker: dict[str, pd.Timestamp] = {str(k): v for k, v in first.items()}
    last_by_ticker: dict[str, pd.Timestamp] = {str(k): v for k, v in last.items()}

    # Collect the trimmed frames by table name, then pass them to ``replace`` by name.
    # A ``**dict`` unpacking cannot be used here: mypy resolves ``replace`` against
    # ``SyntheticDataset``'s concrete field types, so a generic mapping is untypeable.
    # The cost of naming them is that a future addition to ``_COVERAGE_DATE_COLUMNS``
    # would be computed and then silently dropped, so the guard below turns that into
    # an error.
    frames: dict[str, pd.DataFrame] = {}
    trimmed: dict[str, int] = {}
    for name, column in _COVERAGE_DATE_COLUMNS.items():
        frame: pd.DataFrame = getattr(dataset, name)
        if frame.empty or column not in frame.columns or "ticker" not in frame.columns:
            frames[name] = frame
            continue
        keys = frame["ticker"].astype(str)
        lower = pd.to_datetime(keys.map(first_by_ticker), errors="coerce")
        upper = pd.to_datetime(keys.map(last_by_ticker), errors="coerce")
        stamps = pd.to_datetime(frame[column], errors="coerce")
        # NaT on either side compares False, so an unknown ticker or an unparseable
        # date is dropped rather than silently kept.
        keep = stamps.ge(lower) & stamps.le(upper)
        frames[name] = frame.loc[keep].reset_index(drop=True)
        trimmed[name] = int((~keep).sum())

    applied = {"events", "filings", "fundamentals", "news"}
    unapplied = sorted(set(frames) - applied)
    if unapplied:  # pragma: no cover - guards a future edit to _COVERAGE_DATE_COLUMNS
        raise AssertionError(
            f"_COVERAGE_DATE_COLUMNS contains {unapplied}, which _apply_coverage does not "
            f"pass to replace(); the trimmable tables are {sorted(applied)}. Add them there "
            f"too, or the trim is computed and thrown away."
        )

    if not any(trimmed.values()):
        return dataset

    logger.info(
        "aligned tables to price coverage: %s",
        ", ".join(f"{name} -{count}" for name, count in sorted(trimmed.items()) if count),
    )
    return replace(
        dataset,
        fundamentals=frames["fundamentals"],
        filings=frames["filings"],
        news=frames["news"],
        events=frames["events"],
    )


def generate_synthetic_dataset(config: SyntheticConfig | None = None) -> SyntheticDataset:
    """Generate the full synthetic panel.

    Args:
        config: Generator configuration. Defaults to :class:`SyntheticConfig`.

    Returns:
        A :class:`SyntheticDataset` holding prices, fundamentals, filings, news,
        events and the ground-truth latent state. The latent table is returned so a
        test can assert that the text channel really does carry information the
        structured channel cannot, rather than trusting the design note.

    Raises:
        ValueError: If the configured window contains no business days.
    """
    settings = config or SyntheticConfig()
    # One generator, drawn from in a fixed order, so the output is a pure function of
    # the seed and the configuration.
    rng = np.random.default_rng(13)

    dates = pd.bdate_range(settings.start, settings.end)
    if len(dates) < 60:
        raise ValueError(
            f"the configured window {settings.start}..{settings.end} contains only "
            f"{len(dates)} business days; at least 60 are needed for the long-window features"
        )

    tickers = list(SYNTHETIC_TICKERS[: max(2, min(settings.n_companies, len(SYNTHETIC_TICKERS)))])
    latent = _latent_paths(rng, tickers, dates, noise=settings.label_noise)

    prices = _synthesise_prices(rng, latent, dates, inject_edge_cases=settings.inject_edge_cases)
    fundamentals = _synthesise_fundamentals(rng, latent, settings)
    filings = _synthesise_filings(rng, latent, settings)
    news = _synthesise_news(rng, latent, settings)
    events = _synthesise_events(
        rng,
        latent,
        base_rate_multiplier=settings.base_rate_multiplier,
        label_noise=settings.label_noise,
    )

    dataset = SyntheticDataset(
        prices=prices,
        fundamentals=fundamentals,
        filings=filings,
        news=news,
        events=events,
        latent=latent,
        metadata={
            "generator": "shingan.data.synthetic",
            "is_synthetic": True,
            "start": settings.start.isoformat(),
            "end": settings.end.isoformat(),
            "text_component_strength": settings.text_component_strength,
            "structure_information_loss": settings.structure_information_loss,
            "inject_edge_cases": settings.inject_edge_cases,
        },
    )
    # Applied unconditionally, not only when edge cases are injected: the invariant
    # worth holding is "one coverage per company across all tables", and a conditional
    # invariant is just a convention waiting to be broken.
    dataset = _apply_coverage(dataset)

    logger.info(
        "generated synthetic dataset: %d companies, %d price rows, %d filings, %d news, %d events",
        len(tickers),
        len(dataset.prices),
        len(dataset.filings),
        len(dataset.news),
        len(dataset.events),
    )
    return dataset


__all__ = [
    "SYNTHETIC_SECTORS",
    "SYNTHETIC_TICKERS",
    "TARGET_BASE_RATES",
    "SyntheticDataset",
    "generate_synthetic_dataset",
]
