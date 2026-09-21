"""Price and volume adapters.

**Status: thin adapter, not validated against live endpoints.** It implements request
construction, response mapping, column normalisation and caching. It has not been run
against yfinance or Stooq from this repository. Treat it as an interface to be
verified, not as a working data source — see ``docs/02-data.md`` section 2.3.

Three rules the adapters enforce, because getting them wrong invalidates every
downstream feature:

**Adjusted prices only.** Dividends and splits must be reflected, or the return
series acquires artificial jumps that look exactly like the drawdowns the model is
supposed to predict. The adjustment method is recorded in the returned metadata.

**One trading calendar.** Rows are restricted to the dates present for the
instruments requested; no synthetic rows are invented for weekends or holidays.

**Missing prices stay missing.** A halted or delisted instrument keeps its gaps. The
gap policy lives in :mod:`shingan.features.technical`, not here, so that one place
decides what a long run of missing prices means.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Canonical output columns. ``shares_outstanding`` is optional and NaN when the
#: source does not provide it, which disables ``turnover_20d`` and nothing else.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "shares_outstanding",
)


class PriceUnavailable(RuntimeError):
    """Raised when a price source cannot be reached or returns nothing usable."""


class PriceAdapter(Protocol):
    """Minimal interface every price source implements."""

    #: Short adapter name, recorded as the provenance of every row.
    name: str

    #: What the prices have been adjusted for, e.g. ``"splits_and_dividends"`` or
    #: ``"none"``. Declared on the protocol rather than left to the implementations
    #: because it is copied straight into :class:`AdapterMetadata`, and a missing
    #: attribute there is an ``AttributeError`` after the download has already run.
    adjustment: str

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Return an :data:`OUTPUT_COLUMNS`-shaped frame."""
        ...  # pragma: no cover - protocol definition


@dataclass(slots=True)
class AdapterMetadata:
    """Provenance for a fetched panel, written into run metadata."""

    source: str
    adjustment: str
    n_tickers: int
    n_rows: int
    start: str
    end: str


class YFinanceAdapter:
    """Daily adjusted OHLCV via ``yfinance``.

    The library is imported lazily so that the CPU-only install does not need it. It
    is not a declared dependency: it scrapes an undocumented endpoint frequently
    described as unsuitable for production, and a POC should not silently make it
    load-bearing.
    """

    name = "yfinance"
    adjustment = "auto_adjust"

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Download daily bars for the requested tickers."""
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise PriceUnavailable(
                "yfinance is not installed; install it with 'pip install yfinance' or "
                "use source='synthetic' for an offline run"
            ) from exc

        logger.info("fetching %d tickers from yfinance (%s..%s)", len(tickers), start, end)
        raw = yfinance.download(
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        if raw is None or raw.empty:
            raise PriceUnavailable("yfinance returned no rows")

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    logger.warning("yfinance returned no data for %s", ticker)
                    continue
                block = raw[ticker]
            else:
                block = raw
            frame = block.reset_index().rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            frame["ticker"] = ticker
            frames.append(frame)

        if not frames:
            raise PriceUnavailable("yfinance returned no usable columns for any ticker")
        return normalise_prices(pd.concat(frames, ignore_index=True))


class StooqAdapter:
    """Daily bars via Stooq's CSV endpoint. Used as a cross-check, not a primary.

    Stooq data may not be redistributed, which is why the repository never commits a
    price panel. See ``data/README.md``.
    """

    name = "stooq"
    adjustment = "stooq_adjusted"

    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Download daily bars for the requested tickers."""
        import httpx

        frames: list[pd.DataFrame] = []
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for ticker in tickers:
                symbol = f"{ticker.lower()}.us"
                response = client.get(
                    self.BASE_URL,
                    params={
                        "s": symbol,
                        "i": "d",
                        "d1": start.strftime("%Y%m%d"),
                        "d2": end.strftime("%Y%m%d"),
                    },
                )
                if response.status_code != 200 or "No data" in response.text[:200]:
                    logger.warning("stooq returned no data for %s", ticker)
                    continue
                frame = _csv_to_frame(response.text, ticker)
                if frame is not None:
                    frames.append(frame)

        if not frames:
            raise PriceUnavailable("stooq returned no usable data for any requested ticker")
        return normalise_prices(pd.concat(frames, ignore_index=True))


def _csv_to_frame(payload: str, ticker: str) -> pd.DataFrame | None:
    """Parse a Stooq CSV payload into the canonical column names."""
    from io import StringIO

    frame = pd.read_csv(StringIO(payload))
    if frame.empty:
        return None
    frame = frame.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    frame["ticker"] = ticker
    return frame


def normalise_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw price frame into :data:`OUTPUT_COLUMNS`.

    Args:
        frame: Frame with ``ticker``, ``date`` and at least ``close``.

    Returns:
        A frame with the canonical columns, numeric dtypes, one row per
        ``(ticker, date)``, sorted, with duplicate dates resolved by keeping the last
        observation (a later print supersedes an earlier one).

    Raises:
        KeyError: If ``ticker``, ``date`` or ``close`` is missing.
        ValueError: If the frame contains no rows.
    """
    required = ("ticker", "date", "close")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"price frame is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("price frame is empty")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    for column in OUTPUT_COLUMNS:
        if column in result.columns and column not in {"ticker", "date"}:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    result = (
        result[list(OUTPUT_COLUMNS)]
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )
    result["ticker"] = result["ticker"].astype(str)
    return result


def load_prices(
    tickers: list[str],
    start: date,
    end: date,
    *,
    source: str = "synthetic",
    offline: bool = True,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, AdapterMetadata]:
    """Load a price panel from the configured source.

    Args:
        tickers: Instruments to load.
        start: First date, inclusive.
        end: Last date, inclusive.
        source: ``"synthetic"``, ``"prices_yfinance"`` or ``"prices_stooq"``.
        offline: When True, only ``"synthetic"`` is permitted. Guards the test suite
            and the demo against accidental network access.
        cache_dir: Optional directory for a parquet/CSV snapshot.

    Returns:
        A ``(frame, metadata)`` pair.

    Raises:
        PriceUnavailable: If the source is unavailable, or a network source is
            requested while ``offline`` is True.
        ValueError: If ``source`` is not recognised.
    """
    if source == "synthetic":
        from shingan.config import SyntheticConfig
        from shingan.data.synthetic import generate_synthetic_dataset

        dataset = generate_synthetic_dataset(
            SyntheticConfig(n_companies=max(2, len(tickers)), start=start, end=end)
        )
        prices = dataset.prices
        selected = prices.loc[prices["ticker"].isin(tickers)] if tickers else prices
        metadata = AdapterMetadata(
            source="synthetic",
            adjustment="none",
            n_tickers=int(selected["ticker"].nunique()) if len(selected) else 0,
            n_rows=len(selected),
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return normalise_prices(selected), metadata

    if offline:
        raise PriceUnavailable(
            f"source={source!r} requires network access but offline=True. Set "
            "SHINGAN_OFFLINE=false or data.offline=false to allow it."
        )

    if source == "prices_yfinance":
        adapter: PriceAdapter = YFinanceAdapter()
    elif source == "prices_stooq":
        adapter = StooqAdapter()
    else:
        raise ValueError(
            f"unknown price source {source!r}; expected one of 'synthetic', "
            "'prices_yfinance', 'prices_stooq'"
        )

    frame = adapter.fetch(tickers, start, end)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination = cache_dir / f"prices_{adapter.name}_{start}_{end}.csv"
        frame.to_csv(destination, index=False, encoding="utf-8")
        logger.info("cached %d price rows to %s", len(frame), destination)

    metadata = AdapterMetadata(
        source=adapter.name,
        adjustment=adapter.adjustment,
        n_tickers=int(frame["ticker"].nunique()),
        n_rows=len(frame),
        start=start.isoformat(),
        end=end.isoformat(),
    )
    logger.warning(
        "price adapter %r has not been validated against the live endpoint; verify the "
        "adjustment method and the trading calendar before trusting any result",
        adapter.name,
    )
    return frame, metadata


def price_quality_report(frame: pd.DataFrame, *, max_nan_run: int = 10) -> pd.DataFrame:
    """Summarise coverage and gaps per instrument.

    Emitted into the report because a feature that is NaN for one instrument and
    populated for another changes what the model has learned without changing
    anything visible in the metrics table.

    Args:
        frame: Canonical price frame.
        max_nan_run: Same threshold the feature layer uses.

    Returns:
        One row per ticker with the row count, date range, close missingness, the
        longest run of missing closes, and the number of rows the feature layer would
        discard as contaminated.
    """
    from shingan.features.technical import consecutive_nan_run

    records: list[dict[str, Any]] = []
    for ticker, group in frame.groupby("ticker", sort=False, observed=True):
        ordered = group.sort_values("date")
        runs = consecutive_nan_run(ordered["close"])
        records.append(
            {
                "ticker": ticker,
                "rows": len(ordered),
                "first_date": ordered["date"].min().date().isoformat(),
                "last_date": ordered["date"].max().date().isoformat(),
                "close_missing_frac": float(ordered["close"].isna().mean()),
                "longest_missing_run": int(runs.max()) if len(runs) else 0,
                "contaminated_rows": int((runs > max_nan_run).sum()),
            }
        )
    return pd.DataFrame.from_records(records).set_index("ticker")


__all__ = [
    "OUTPUT_COLUMNS",
    "AdapterMetadata",
    "PriceAdapter",
    "PriceUnavailable",
    "StooqAdapter",
    "YFinanceAdapter",
    "load_prices",
    "normalise_prices",
    "price_quality_report",
]
