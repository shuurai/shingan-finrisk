"""Quantify each "more positives" lever before spending hours of downloads.

The binding constraint on every Track B result is the count of positive rows: the A
split trains on 779 rows holding 4 positives, and a 14B SFT run on that count learned
to answer the marginal (constant 0) rather than to discriminate (docs/09 section 20).
Four levers could raise the count, and they cost very different amounts:

* **threshold** -- move ``labels.tail_risk_drawdown_threshold`` from -30% to -20%.
  Zero downloads, but it redefines the label: every number measured before becomes
  incomparable, and a shallower drawdown is a different (more common, less "tail")
  event.
* **window** -- extend ``data.start`` back from 2010 to 2004-2009. Prices are cheap to
  refetch and the GFC supplies what this universe is missing: simultaneous deep
  drawdowns across the banks, the automakers and the industrials. The grid needs the
  filings of those years too, which is the expensive half.
* **universe** -- add names. More names means more negatives as well as more
  positives, and filings downloads scale with it.
* **grid density** -- one row per filing date is the current rule; monthly decision
  points multiply the rows ~3x at zero download cost, at the price of overlapping
  30-trading-day label windows (correlated rows) and nearly identical text between
  consecutive rows of one name.

This script measures the levers against the cached tables (validating itself against the
shipped panel), fetches prices for a candidate universe into a scratch directory, and --
with ``--filing-grid`` -- enumerates the *actual* 10-K/10-Q dates of the union from
EDGAR's submissions index (one cheap request per filer) so the arithmetic shown is the
arithmetic the panel would have, not an estimate. Documents are never downloaded here:
that spend belongs to ``fetch_sec_docs.py`` and only after this output says which names
and which years earn it.

Usage::

    python scripts/universe_options.py
    python scripts/universe_options.py --candidates configs/universes/candidates.txt \
        --candidate-start 2004-01-01 --filing-grid
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shingan.config import load_config  # noqa: E402
from shingan.data.edgar import SecEdgarClient  # noqa: E402
from shingan.data.schema import label_column, mask_column  # noqa: E402

#: Thresholds the sweep reports. -0.30 is the shipped definition.
THRESHOLDS: tuple[float, ...] = (-0.20, -0.25, -0.30, -0.35)

#: The split the project uses (configs/data/stage2_real.yaml, geometry A).
TEST_START = pd.Timestamp("2020-01-01")

#: SEC's official ticker -> CIK map. 10k+ entries; the fetch that built the current
#: corpus could not reach it (www.sec.gov was 403ing) and carried a hand-written
#: candidate table instead, which cannot cover a 150-name universe.
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

#: Forms whose text the corpus can actually use (Item 1A / Item 7 extraction).
USABLE_FORMS: tuple[str, ...] = ("10-K", "10-Q")


@dataclass
class Arithmetic:
    """Positive counts for one (grid, threshold) combination."""

    name: str
    n_rows: int
    n_positives: int
    n_train: int
    n_train_positives: int
    n_test: int
    n_test_positives: int

    def render(self) -> str:
        rate = 100.0 * self.n_positives / self.n_rows if self.n_rows else 0.0
        return (
            f"{self.name:<34} rows {self.n_rows:>6}  pos {self.n_positives:>4} "
            f"({rate:4.1f}%)  train {self.n_train:>5}/{self.n_train_positives:<3} "
            f"test {self.n_test:>5}/{self.n_test_positives:<3}"
        )


def forward_max_drawdown(values: np.ndarray, horizon: int) -> np.ndarray:
    """Deepest running peak-to-trough decline over the ``horizon`` steps after ``t``.

    The same definition the label uses (``labeling.builders.forward_max_drawdown``);
    reimplemented on ndarray purely so the sweep can call it thousands of times.
    """
    n_rows = len(values)
    out = np.full(n_rows, np.nan, dtype=float)
    for position in range(n_rows):
        end = position + horizon + 1
        if end > n_rows:
            break
        window = values[position:end]
        if not np.isfinite(window).all():
            continue
        peak = np.maximum.accumulate(window)
        with np.errstate(invalid="ignore", divide="ignore"):
            drawdown = window[1:] / peak[1:] - 1.0
        if drawdown.size:
            out[position] = float(np.min(drawdown))
    return out


def drawdown_grid(prices: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    """One row per (ticker, trading date) with the forward drawdown attached."""
    frame = prices[["ticker", "date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for ticker, group in frame.groupby("ticker", sort=False, observed=True):
        values = group["close"].astype(float).to_numpy()
        pieces.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": group["date"].to_numpy(),
                    "fwd_dd": forward_max_drawdown(values, horizon),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def attach_label(
    grid: pd.DataFrame, drawdowns: pd.DataFrame, *, threshold: float
) -> pd.DataFrame:
    """Left-join the forward drawdown onto a decision grid and threshold it.

    ``grid`` carries ``ticker`` and ``as_of``. Rows whose drawdown is NaN have an
    incomplete (or gapped) forward window and are dropped rather than labelled: they
    are unknown, not negative.
    """
    merged = grid.merge(
        drawdowns, left_on=["ticker", "as_of"], right_on=["ticker", "date"], how="left"
    )
    merged = merged[merged["fwd_dd"].notna()].copy()
    merged["label"] = (merged["fwd_dd"] <= threshold).astype(int)
    return merged


def arithmetic(name: str, labelled: pd.DataFrame) -> Arithmetic:
    """Split a labelled grid into the project's train/test blocks."""
    train = labelled[labelled["as_of"] < TEST_START]
    test = labelled[labelled["as_of"] >= TEST_START]
    return Arithmetic(
        name=name,
        n_rows=len(labelled),
        n_positives=int(labelled["label"].sum()),
        n_train=len(train),
        n_train_positives=int(train["label"].sum()),
        n_test=len(test),
        n_test_positives=int(test["label"].sum()),
    )


def filing_grid(filings: pd.DataFrame) -> pd.DataFrame:
    """The shipped decision grid: one row per (ticker, filing date)."""
    grid = filings[["ticker", "filed"]].drop_duplicates().rename(columns={"filed": "as_of"})
    grid["as_of"] = pd.to_datetime(grid["as_of"])
    return grid.sort_values(["ticker", "as_of"]).reset_index(drop=True)


def month_grid(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly decision points: the first trading day of each month per ticker."""
    frame = prices[["ticker", "date"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.to_period("M")
    firsts = frame.sort_values("date").groupby(["ticker", "month"], observed=True).first()
    grid = firsts.reset_index()[["ticker", "date"]].rename(columns={"date": "as_of"})
    return grid.sort_values(["ticker", "as_of"]).reset_index(drop=True)


def validate_against_panel(labelled: pd.DataFrame, panel_path: Path) -> str:
    """Compare this reproduction with the shipped panel's own label column.

    A reproduction that does not match the panel is measuring a different label, so
    the report says which rows disagree rather than quietly presenting them.
    """
    if not panel_path.is_file():
        return "validation skipped: no panel.parquet"
    panel = pd.read_parquet(panel_path)
    if "is_synthetic" in panel.columns and bool(panel["is_synthetic"].any()):
        return "validation skipped: panel is synthetic"
    label_col = label_column("tail_risk")
    if label_col not in panel.columns:
        return f"validation skipped: panel lacks {label_col}"
    truth = panel[["ticker", "as_of", label_col, mask_column("tail_risk")]].copy()
    truth["as_of"] = pd.to_datetime(truth["as_of"])
    truth = truth[truth[mask_column("tail_risk")].astype(bool)]
    mine = labelled.rename(columns={"label": "reproduced"})
    joined = truth.merge(mine, on=["ticker", "as_of"], how="outer", indicator=True)
    both = joined[joined["_merge"] == "both"]
    disagree = both[both[label_col].astype(int) != both["reproduced"]]
    only_panel = joined[joined["_merge"] == "left_only"]
    only_mine = joined[joined["_merge"] == "right_only"]
    return (
        f"validation vs {panel_path.name}: matched {len(both)} rows, "
        f"{len(disagree)} label disagreements, {len(only_panel)} panel-only, "
        f"{len(only_mine)} reproduction-only"
    )


def per_year(labelled: pd.DataFrame) -> pd.DataFrame:
    """Positives and rows per calendar year, for reading where the events sit."""
    frame = labelled.copy()
    frame["year"] = frame["as_of"].dt.year
    grouped = frame.groupby("year").agg(rows=("label", "size"), positives=("label", "sum"))
    return grouped


def fetch_candidate_prices(
    tickers: list[str],
    *,
    start: date,
    end: date,
    sleep_seconds: float,
    scratch: Path | None,
    refresh: bool,
) -> pd.DataFrame:
    """Daily prices for a candidate list, cached in a *scratch* directory.

    Deliberately not the production cache: this run decides which candidates deserve a
    download, and a half-verified candidate that later fails CIK verification must not
    be able to leak a price series into the panel. Writing to scratch also means the
    follow-up questions (filing-grid arithmetic, a second threshold) do not pay for the
    fetch again.
    """
    import time

    from shingan.data.prices import PriceUnavailable, load_prices

    cached_path = scratch / f"candidate_prices_{start}_{end}.parquet" if scratch is not None else None
    cached: pd.DataFrame | None = None
    if cached_path is not None and cached_path.is_file() and not refresh:
        cached = pd.read_parquet(cached_path)
        covered = set(cached["ticker"].astype(str))
        missing = [ticker for ticker in tickers if ticker not in covered]
        if not missing:
            print(f"candidate prices: reusing {cached_path} ({len(cached)} rows)")
            return cached
        print(
            f"candidate prices: {cached_path} covers {len(covered)} names; "
            f"fetching the {len(missing)} missing"
        )
        to_fetch = missing
    else:
        to_fetch = tickers

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for position, ticker in enumerate(to_fetch):
        try:
            frame, metadata = load_prices(
                [ticker], start=start, end=end, source="prices_yfinance", offline=False
            )
            frames.append(frame)
            print(
                f"  {ticker}: {metadata.n_rows} rows "
                f"({frame['date'].min().date()} .. {frame['date'].max().date()})"
            )
        except PriceUnavailable as exc:
            print(f"  ! {ticker}: {exc}")
            failed.append(ticker)
        if position < len(to_fetch) - 1:
            time.sleep(sleep_seconds)
    if failed:
        print(f"warning: no prices for {', '.join(failed)}")
    if not frames and cached is None:
        raise SystemExit("no candidate returned prices; the source is throttling this network")
    combined = pd.concat([*([] if cached is None else [cached]), *frames], ignore_index=True)
    combined = combined.drop_duplicates(["ticker", "date"])
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
        assert cached_path is not None
        combined.to_parquet(cached_path, index=False)
        print(f"candidate prices: cached {len(combined)} rows to {cached_path}")
    return combined


def load_cik_map(cache_path: Path, *, user_agent: str) -> dict[str, str]:
    """Ticker -> zero-padded CIK, from SEC's own map, cached next to the raw tables.

    Only the newest symbol for a CIK is kept when several exist (class shares), which is
    the convention EDGAR's own files use; the alternative -- silently picking a share
    class the universe did not ask for -- is worse than a documented choice.
    """
    if not cache_path.is_file():
        request = urllib.request.Request(
            COMPANY_TICKERS_URL,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
        print(f"cached SEC ticker map to {cache_path} ({len(payload):,} bytes)")
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for entry in raw.values():
        ticker = str(entry["ticker"]).upper()
        cik = f"{int(entry['cik_str']):010d}"
        mapping.setdefault(ticker, cik)
    return mapping


def enumerate_filings(
    client: SecEdgarClient,
    tickers: list[str],
    cik_by_ticker: dict[str, str],
    *,
    since: date,
    limit_per_ticker: int = 400,
) -> tuple[pd.DataFrame, list[str]]:
    """``(ticker, filed)`` for every usable filing from ``since`` on, via submissions.

    Metadata only: this is what the decision grid is made of, and it costs one request
    per filer rather than one per document.
    """
    records: list[dict[str, object]] = []
    unlisted: list[str] = []
    for ticker in tickers:
        cik = cik_by_ticker.get(ticker)
        if cik is None:
            unlisted.append(ticker)
            continue
        try:
            refs = client.list_filings(
                ticker, forms=USABLE_FORMS, since=since, limit=limit_per_ticker, cik=cik
            )
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            print(f"  ! {ticker}: submissions failed ({exc})")
            unlisted.append(ticker)
            continue
        for ref in refs:
            records.append({"ticker": ticker, "filed": pd.Timestamp(ref.filed)})
    frame = pd.DataFrame.from_records(records, columns=["ticker", "filed"])
    return frame, unlisted


def arithmetic_for_configurations(
    name: str,
    prices: pd.DataFrame,
    filings: pd.DataFrame,
    *,
    horizon: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    thresholds: tuple[float, ...] = (-0.25, -0.30),
) -> list[Arithmetic]:
    """Exact filing-grid arithmetic for one ticker set at each threshold.

    Both ends of the window are applied: filings after ``window_end`` are outside the
    panel, and keeping them would put rows in a block the build never produces.
    """
    scoped_filings = filings[
        (filings["filed"] >= window_start) & (filings["filed"] <= window_end)
    ]
    grid = scoped_filings.rename(columns={"filed": "as_of"})
    drawdowns = drawdown_grid(prices, horizon=horizon)
    out: list[Arithmetic] = []
    for threshold in thresholds:
        labelled = attach_label(grid, drawdowns, threshold=threshold)
        out.append(arithmetic(f"{name} @ {threshold:+.0%}", labelled))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        default="configs/data/stage2_real.yaml",
        help="Overlay that names the universe, the window and the label block.",
    )
    parser.add_argument(
        "--candidates",
        action="append",
        default=None,
        help="Text file with one candidate ticker per line (repeatable); fetches their"
            " prices. Without it the script is fully offline.",
    )
    parser.add_argument(
        "--candidate-start",
        default=None,
        help="ISO date to fetch candidate prices from. Defaults to the config window"
            " start; earlier values measure the 'extend the window back' lever.",
    )
    parser.add_argument(
        "--candidate-sleep-seconds",
        type=float,
        default=1.0,
        help="Pause between candidate price requests (yfinance throttles by request).",
    )
    parser.add_argument(
        "--scratch",
        default="data/raw/analysis",
        help="Where candidate prices are cached (never the production cache).",
    )
    parser.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Ignore the scratch price cache and fetch again.",
    )
    parser.add_argument(
        "--filing-grid",
        action="store_true",
        help="Also enumerate the union's real 10-K/10-Q dates from EDGAR and print the"
            " exact configuration arithmetic. Needs network; downloads no documents.",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "configs" / "default.yaml", [args.data_config])
    cache = Path(config.data.cache_dir)
    if not cache.is_absolute():
        cache = PROJECT_ROOT / cache
    prices = pd.read_parquet(cache / "prices.parquet")
    filings = pd.read_parquet(cache / "filings.parquet")
    window_start = pd.Timestamp(str(config.data.start))
    window_end = pd.Timestamp(str(config.data.end))
    horizon = config.labels.tail_risk_horizon_trading_days

    prices = prices[(prices["date"] >= window_start) & (prices["date"] <= window_end)]
    filings = filings[
        (pd.to_datetime(filings["filed"]) >= window_start)
        & (pd.to_datetime(filings["filed"]) <= window_end)
    ]
    universe = [str(item) for item in config.data.universe]
    missing = sorted(set(universe) - set(prices["ticker"].astype(str).unique()))

    print("=" * 100)
    print("inputs")
    print("=" * 100)
    print(f"window        {window_start.date()} .. {window_end.date()}")
    print(f"universe      {len(universe)} names, prices for {prices['ticker'].nunique()}")
    if missing:
        print(f"missing       {', '.join(missing)} (declared in the universe, absent from prices)")
    print(f"horizon       {horizon} trading days; threshold sweep {THRESHOLDS}")
    print(f"threshold     shipped value {config.labels.tail_risk_drawdown_threshold:+.2f}")

    drawdowns = drawdown_grid(prices, horizon=horizon)
    filed = filing_grid(filings)
    monthly = month_grid(prices)
    monthly = monthly[
        (monthly["as_of"] >= window_start) & (monthly["as_of"] <= window_end)
    ]

    print()
    print("=" * 100)
    print("lever 1 -- drawdown threshold (shipped grid: one row per filing date)")
    print("=" * 100)
    results = []
    for threshold in THRESHOLDS:
        labelled = attach_label(filed, drawdowns, threshold=threshold)
        mark = "  <- shipped" if abs(threshold + 0.30) < 1e-9 else ""
        row = arithmetic(f"{threshold:+.0%}{mark}", labelled)
        results.append((threshold, labelled))
        print(row.render())
    shipped = next(item for threshold, item in results if abs(threshold + 0.30) < 1e-9)
    print()
    print(validate_against_panel(shipped, PROJECT_ROOT / "data" / "processed" / "panel.parquet"))

    print()
    print("=" * 100)
    print("lever 2 -- grid density (threshold fixed at the shipped -30%)")
    print("=" * 100)
    for name, grid in (("filing dates (shipped)", filed), ("monthly first trading day", monthly)):
        labelled = attach_label(grid, drawdowns, threshold=-0.30)
        print(arithmetic(name, labelled).render())
    print(
        "monthly rows overlap: the 30-trading-day window of one row is still open when"
        " the next row's opens, so counts and effective sample size differ."
    )

    print()
    print("=" * 100)
    print("positives per year (shipped grid, -30%)")
    print("=" * 100)
    print(per_year(shipped).to_string())

    if args.candidates:
        candidates: list[str] = []
        for path in args.candidates:
            candidates.extend(
                line.strip().upper()
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        candidates = sorted(set(candidates))
        print()
        print("=" * 100)
        print(f"lever 3 -- candidate universe ({len(candidates)} names)")
        print("=" * 100)
        extended_start = (
            date.fromisoformat(args.candidate_start)
            if args.candidate_start
            else date.fromisoformat(str(config.data.start))
        )
        scratch = Path(args.scratch) if args.scratch else None
        if scratch is not None and not scratch.is_absolute():
            scratch = PROJECT_ROOT / scratch
        fetched = fetch_candidate_prices(
            candidates,
            start=extended_start,
            end=date.fromisoformat(str(config.data.end)),
            sleep_seconds=args.candidate_sleep_seconds,
            scratch=scratch,
            refresh=args.refresh_prices,
        )
        print(f"\ncandidate prices: {len(fetched)} rows for {fetched['ticker'].nunique()} names")
        candidate_dd = drawdown_grid(fetched, horizon=horizon)
        candidate_grid = month_grid(fetched)
        if args.candidate_start:
            candidate_grid = candidate_grid[candidate_grid["as_of"] >= pd.Timestamp(extended_start)]
        rows: list[dict[str, object]] = []
        for ticker, group in candidate_grid.groupby("ticker", observed=True):
            labelled = attach_label(group, candidate_dd, threshold=-0.30)
            if labelled.empty:
                continue
            eras = {
                "pre2010": (pd.Timestamp("2010-01-01"), None),
                "2010_2019": (pd.Timestamp("2010-01-01"), pd.Timestamp("2020-01-01")),
                "2020_plus": (TEST_START, None),
            }
            entry: dict[str, object] = {"ticker": ticker, "months": len(labelled)}
            entry["positives"] = int(labelled["label"].sum())
            for name, (lower, upper) in eras.items():
                span = labelled[labelled["as_of"] >= lower]
                if upper is not None:
                    span = span[span["as_of"] < upper]
                entry[f"pos_{name}"] = int(span["label"].sum())
            rows.append(entry)
        table = pd.DataFrame(rows).sort_values("positives", ascending=False)
        print()
        print(f"candidate positive yield, monthly grid, -30%, prices from {extended_start}:")
        print(table.to_string(index=False))
        print(
            f"\ncandidates alone: {len(table)} names, {int(table['months'].sum())} rows, "
            f"{int(table['positives'].sum())} positives"
        )
        for name in ("pre2010", "2010_2019", "2020_plus"):
            column = f"pos_{name}"
            if column in table.columns:
                print(f"  {name:<10} {int(table[column].sum())} positives")

        if args.filing_grid:
            print()
            print("=" * 100)
            print("exact filing grid (EDGAR submissions metadata; no documents downloaded)")
            print("=" * 100)
            user_agent = str(config.data.sec_user_agent)
            cik_by_ticker = load_cik_map(cache / "company_tickers.json", user_agent=user_agent)
            print(f"SEC ticker map: {len(cik_by_ticker):,} symbols")
            client = SecEdgarClient(
                user_agent=user_agent, requests_per_second=8.0, cache_dir=cache / "edgar_json"
            )
            union = sorted({*universe, *candidates})
            filings_meta, unlisted = enumerate_filings(
                client,
                union,
                cik_by_ticker,
                since=extended_start,
            )
            if unlisted:
                print(f"no filings enumerated for: {', '.join(sorted(unlisted))}")
            print(
                f"enumerated {len(filings_meta):,} filings for "
                f"{filings_meta['ticker'].nunique()} filers"
            )
            union_prices = pd.concat([prices, fetched], ignore_index=True).drop_duplicates(
                ["ticker", "date"]
            )
            print()
            print(
                "configuration arithmetic -- one row per filing date, observable rows"
                " only (a row whose forward window is incomplete is unknown, not negative):"
            )
            configurations = (
                ("A shipped: 34 names, 2010 window", set(universe), window_start),
                (f"B window: 34 names, from {extended_start}", set(universe), pd.Timestamp(extended_start)),
                (f"C universe: {len(union)} names, from {extended_start}", set(union), pd.Timestamp(extended_start)),
            )
            for label, ticker_set, start in configurations:
                scoped_prices = union_prices[
                    (union_prices["ticker"].astype(str).isin(ticker_set))
                    & (union_prices["date"] >= start)
                ]
                scoped_filings = filings_meta[
                    (filings_meta["ticker"].isin(ticker_set)) & (filings_meta["filed"] >= start)
                ]
                for row in arithmetic_for_configurations(
                    label,
                    scoped_prices,
                    scoped_filings,
                    horizon=horizon,
                    window_start=start,
                    window_end=window_end,
                ):
                    print(row.render())
            print()
            print(
                "survivorship disclosure -- the price source (yfinance) serves listed names"
                " only, and SEC's current ticker map drops delisted symbols as well: "
                f"{len(unlisted)} of {len(union)} names resolved to no filings at all."
            )


if __name__ == "__main__":
    main()
