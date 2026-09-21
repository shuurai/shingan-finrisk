"""Structured features from financial statements (the "ratios" group).

Input is a wide frame of canonical fundamental line items, one row per
``(ticker, filed)``. Canonicalising the SEC's XBRL concept names into a fixed set
of columns is the job of :func:`canonicalise_concepts`; everything downstream
assumes the canonical names, so a new taxonomy tag is handled in one place.

Three conventions worth stating explicitly, because each one is a decision that
changes the numbers:

**Zero denominators become missing, not infinite.** A zero in ``total_equity``
means the concept was not tagged, not that the company has no equity. Division
therefore yields NaN. Negative denominators are *kept*: negative equity is a real
and highly informative state, not a data error.

**Altman Z uses market value of equity when available.** The original 1968 Z-score
uses market capitalisation in X4. When market cap is absent the computation falls
back to book equity and is labelled as such in the logs; the two are not
interchangeable, and a Z built on book equity is systematically lower.

**Missing line items are not imputed.** They stay NaN and are counted by
``ratios_missing_frac``, which is itself a feature. Every imputation strategy in
this domain is a way of inventing a number that the company never reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from shingan.data.schema import RATIO_COLUMNS
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Canonical fundamental line items consumed by :func:`compute_ratios`. Every
#: column is optional individually — the affected ratios simply become NaN — but
#: ``ticker``, ``filed`` and ``period_end`` are required.
CANONICAL_FUNDAMENTALS: tuple[str, ...] = (
    "revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "current_assets",
    "current_liabilities",
    "inventory",
    "ebit",
    "interest_expense",
    "operating_cash_flow",
    "capex",
    "goodwill",
    "retained_earnings",
    "short_term_debt",
    "total_debt",
    "market_cap",
)

REQUIRED_INDEX_COLUMNS: tuple[str, ...] = ("ticker", "filed", "period_end")

#: Ratios that need at least one prior-year observation to be defined.
_YOY_RATIOS: tuple[str, ...] = ("revenue_yoy", "asset_growth")

#: Days used when looking back for the prior-year observation. Kept just under a
#: year so that a filing made a few days early or late still matches.
_YOY_LOOKBACK_DAYS = 365


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division where a zero denominator yields NaN, not infinity.

    Args:
        numerator: Numerator series.
        denominator: Denominator series.

    Returns:
        The quotient, with NaN wherever the denominator is zero or the numerator
        is missing.
    """
    safe_denominator = denominator.replace(0.0, np.nan)
    return numerator / safe_denominator


def altman_z_score(
    *,
    working_capital: pd.Series,
    retained_earnings: pd.Series,
    ebit: pd.Series,
    market_value_equity: pd.Series,
    total_liabilities: pd.Series,
    sales: pd.Series,
    total_assets: pd.Series,
) -> pd.Series:
    """Altman (1968) Z-score for a public manufacturer.

        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    with X1 = working capital / total assets, X2 = retained earnings / total
    assets, X3 = EBIT / total assets, X4 = market value of equity / total
    liabilities, X5 = sales / total assets.

    Interpretation, for the record, is the classic one: below 1.81 is the distress
    zone, above 2.99 is safe, in between is grey. The score was fitted on
    manufacturers in the 1960s and transfers imperfectly to banks and to
    asset-light technology firms — the coefficient on sales in particular punishes
    companies whose revenue is not asset-financed. It is used here as one feature
    among many, never as a standalone signal.

    Args:
        working_capital: current assets minus current liabilities.
        retained_earnings: cumulative retained earnings.
        ebit: earnings before interest and taxes.
        market_value_equity: market capitalisation, or book equity as a fallback.
        total_liabilities: total liabilities.
        sales: revenue.
        total_assets: total assets.

    Returns:
        The Z-score, NaN wherever an input was missing.
    """
    x1 = _safe_div(working_capital, total_assets)
    x2 = _safe_div(retained_earnings, total_assets)
    x3 = _safe_div(ebit, total_assets)
    x4 = _safe_div(market_value_equity, total_liabilities)
    x5 = _safe_div(sales, total_assets)
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def _prior_year_lookup(frame: pd.DataFrame, column: str) -> pd.Series:
    """Value of ``column`` about one year before each row's ``period_end``.

    Implemented per ticker with a binary search over sorted period ends rather than
    a fixed ``shift(n)``, because the cadence is not guaranteed: a company that
    changed its fiscal year, or whose 10-K and 10-Q rows are interleaved, breaks any
    fixed-offset assumption and silently produces a growth rate against the wrong
    period.

    Args:
        frame: Frame containing ``ticker``, ``period_end`` and ``column``.
        column: Column whose prior-year value is wanted.

    Returns:
        A series aligned to ``frame.index`` holding the prior value, NaN where no
        observation exists at least ~1 year back.
    """
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _ticker, group in frame.groupby("ticker", sort=False, observed=True):
        ordered = group.sort_values("period_end")
        period_end = pd.to_datetime(ordered["period_end"]).to_numpy()
        values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=float)
        cutoffs = period_end - np.timedelta64(_YOY_LOOKBACK_DAYS, "D")
        # `side="right"` - 1 gives the last observation at or before the cutoff.
        positions = np.searchsorted(period_end, cutoffs, side="right") - 1
        prior = np.full(len(ordered), np.nan, dtype=float)
        valid = positions >= 0
        prior[valid] = values[positions[valid]]
        result.loc[ordered.index] = prior
    return result


def compute_ratios(
    fundamentals: pd.DataFrame,
    *,
    drop_unusable: bool = True,
) -> pd.DataFrame:
    """Compute the financial-ratio feature block.

    Args:
        fundamentals: Wide frame, one row per ``(ticker, filed)``, containing
            ``ticker``, ``filed``, ``period_end`` and any subset of
            :data:`CANONICAL_FUNDAMENTALS`. Extra columns are ignored.
        drop_unusable: Drop rows where every input line item is missing. Such a row
            carries no information and would only add an all-NaN feature row.

    Returns:
        A frame with ``ticker``, ``filed``, ``period_end``, every column in
        :data:`shingan.data.schema.RATIO_COLUMNS`, and the intermediate
        ``working_capital`` and ``market_value_equity_used`` columns that document
        how the Z-score was built.

    Raises:
        KeyError: If a required index column is missing.
        ValueError: If ``fundamentals`` is empty.
    """
    missing = [column for column in REQUIRED_INDEX_COLUMNS if column not in fundamentals.columns]
    if missing:
        raise KeyError(f"fundamentals frame is missing required columns: {missing}")
    if fundamentals.empty:
        raise ValueError("fundamentals frame is empty; nothing to compute")

    frame = fundamentals.copy()
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])

    present = [column for column in CANONICAL_FUNDAMENTALS if column in frame.columns]
    if drop_unusable:
        if present:
            all_missing = frame[present].isna().all(axis=1)
            dropped = int(all_missing.sum())
            if dropped:
                logger.debug("dropping %d fundamental rows with no usable line items", dropped)
            frame = frame.loc[~all_missing].reset_index(drop=True)
        else:
            logger.warning("no canonical fundamental columns present; every ratio will be missing")
            frame = frame.reset_index(drop=True)

    def column(name: str) -> pd.Series:
        """Fetch a canonical column as float, or an all-NaN series if absent."""
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").astype(float)
        return pd.Series(np.nan, index=frame.index, dtype=float)

    revenue = column("revenue")
    net_income = column("net_income")
    total_assets = column("total_assets")
    total_liabilities = column("total_liabilities")
    total_equity = column("total_equity")
    current_assets = column("current_assets")
    current_liabilities = column("current_liabilities")
    ebit = column("ebit")
    interest_expense = column("interest_expense")
    operating_cash_flow = column("operating_cash_flow")
    capex = column("capex")
    goodwill = column("goodwill")
    retained_earnings = column("retained_earnings")
    short_term_debt = column("short_term_debt")
    total_debt = column("total_debt")
    market_cap = column("market_cap")

    working_capital = current_assets - current_liabilities

    used_market_cap = market_cap.notna()
    if not bool(used_market_cap.all()):
        fallback_count = int((~used_market_cap).sum())
        if fallback_count:
            logger.debug(
                "altman_z: market cap missing for %d rows; falling back to book equity",
                fallback_count,
            )
    market_value_equity_used = market_cap.where(used_market_cap, total_equity)

    out = pd.DataFrame(
        {
            "ticker": frame["ticker"],
            "filed": frame["filed"],
            "period_end": frame["period_end"],
            "debt_to_equity": _safe_div(total_liabilities, total_equity),
            "debt_short_term_ratio": _safe_div(short_term_debt, total_debt),
            "current_ratio": _safe_div(current_assets, current_liabilities),
            "interest_coverage": _safe_div(ebit, interest_expense),
            "net_margin": _safe_div(net_income, revenue),
            "roa": _safe_div(net_income, total_assets),
            "roe": _safe_div(net_income, total_equity),
            "fcf_margin": _safe_div(operating_cash_flow - capex, revenue),
            "altman_z": altman_z_score(
                working_capital=working_capital,
                retained_earnings=retained_earnings,
                ebit=ebit,
                market_value_equity=market_value_equity_used,
                total_liabilities=total_liabilities,
                sales=revenue,
                total_assets=total_assets,
            ),
            # High accruals relative to assets is the classic earnings-quality red
            # flag: profit that is not backed by cash.
            "accruals_ratio": _safe_div(net_income - operating_cash_flow, total_assets),
            "revenue_yoy": _safe_div(revenue, _prior_year_lookup(frame, "revenue")) - 1.0,
            "asset_growth": _safe_div(total_assets, _prior_year_lookup(frame, "total_assets"))
            - 1.0,
            "goodwill_to_assets": _safe_div(goodwill, total_assets),
            "working_capital": working_capital,
            "market_value_equity_used": market_value_equity_used,
        }
    )

    ratio_columns = [name for name in RATIO_COLUMNS if name != "ratios_missing_frac"]
    out["ratios_missing_frac"] = out[ratio_columns].isna().mean(axis=1)

    # Guard against a division that produced a non-finite value from a degenerate
    # input; a tree split on inf is undefined behaviour, not a large number.
    numeric_columns = [name for name in RATIO_COLUMNS if name in out.columns]
    non_finite = np.isinf(out[numeric_columns].to_numpy(dtype=float, na_value=np.nan))
    if non_finite.any():
        logger.warning(
            "replacing %d non-finite ratio values (likely zero denominators) with NaN",
            int(non_finite.sum()),
        )
        out[numeric_columns] = out[numeric_columns].replace([np.inf, -np.inf], np.nan)

    logger.debug("computed %d ratios for %d filings", len(RATIO_COLUMNS), len(out))
    return out


def canonicalise_concepts(
    facts: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    value_column: str = "value",
    concept_column: str = "concept",
) -> pd.DataFrame:
    """Pivot a long XBRL fact table into the canonical wide form.

    SEC XBRL data arrives as ``(cik, concept, period_end, filed, value)`` rows with
    hundreds of taxonomy tags. Different filers tag the same economic quantity with
    different concepts — revenue alone has at least five common tags — so a mapping
    from tag to canonical name is required.

    Within a ``(ticker, filed, period_end)`` group, when several mapped tags provide
    the same canonical item (for instance both ``Revenues`` and
    ``RevenueFromContractWithCustomerExcludingAssessedTax``), the first non-null in
    mapping order wins. That order is the caller's priority list, which keeps the
    choice explicit and reviewable rather than alphabetically arbitrary.

    Args:
        facts: Long fact table with at least ``ticker``, ``period_end``, ``filed``,
            ``concept_column`` and ``value_column``.
        mapping: Tag to canonical-name mapping. Tags absent from the mapping are
            dropped. A canonical name must be in :data:`CANONICAL_FUNDAMENTALS` or
            it is ignored with a warning.
        value_column: Name of the numeric value column.
        concept_column: Name of the concept/tag column.

    Returns:
        A wide frame with ``ticker``, ``filed``, ``period_end`` and the canonical
        columns that were present.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If a mapped canonical name is unknown.
    """
    for column in ("ticker", "period_end", "filed", concept_column, value_column):
        if column not in facts.columns:
            raise KeyError(f"facts frame is missing required column {column!r}")

    unknown = sorted({name for name in mapping.values() if name not in CANONICAL_FUNDAMENTALS})
    if unknown:
        raise ValueError(
            f"mapping targets unknown canonical items: {unknown}. "
            f"Known items: {list(CANONICAL_FUNDAMENTALS)}"
        )

    selected = facts.loc[facts[concept_column].isin(mapping)].copy()
    if selected.empty:
        logger.warning("no facts matched the concept mapping; the wide frame will be empty")
        return pd.DataFrame(columns=["ticker", "filed", "period_end"])

    priority = {concept: position for position, concept in enumerate(mapping)}
    selected["_canonical"] = selected[concept_column].map(mapping)
    selected["_priority"] = selected[concept_column].map(priority)
    selected["_value"] = pd.to_numeric(selected[value_column], errors="coerce")
    selected = selected.sort_values(["ticker", "filed", "period_end", "_priority", "_canonical"])

    wide = (
        selected.dropna(subset=["_value"])
        .drop_duplicates(subset=["ticker", "filed", "period_end", "_canonical"], keep="first")
        .pivot_table(
            index=["ticker", "filed", "period_end"],
            columns="_canonical",
            values="_value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    ordered = [
        "ticker",
        "filed",
        "period_end",
        *[name for name in CANONICAL_FUNDAMENTALS if name in wide.columns],
    ]
    logger.debug("canonicalised %d facts into %d rows", len(selected), len(wide))
    return wide[ordered]


def ratio_columns_present(frame: pd.DataFrame) -> Sequence[str]:
    """The subset of :data:`RATIO_COLUMNS` that exists in ``frame``."""
    return tuple(name for name in RATIO_COLUMNS if name in frame.columns)


__all__ = [
    "CANONICAL_FUNDAMENTALS",
    "REQUIRED_INDEX_COLUMNS",
    "altman_z_score",
    "canonicalise_concepts",
    "compute_ratios",
    "ratio_columns_present",
]
