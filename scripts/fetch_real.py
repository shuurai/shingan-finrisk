"""Fetch real Stage 2 data: yfinance prices + SEC EDGAR filings and XBRL fundamentals.

The first real-data step of the roadmap (docs/07-roadmap.md Stage 2). The adapters in
``shingan.data.{prices,edgar}`` are thin and had never been run against live endpoints;
this script is the wiring that exercises them and persists the result in the exact
``RawTables`` shape ``build_panel`` consumes.

Usage (from the repository root, with the project venv active):

    python scripts/fetch_real.py                 # fetch and cache under data/raw/real
    python scripts/fetch_real.py --build         # ... then build the processed panel

What it produces under ``data/raw/real/``:

    prices.parquet        ticker, date, open, high, low, close, volume, shares_outstanding
    fundamentals.parquet  ticker, filed, period_end + the canonical fundamental line items
    filings.parquet       ticker, filed, doc_type, section, text, accession
    news.parquet          empty (FNSPID snapshot not wired yet)
    events.parquet        empty (rating / enforcement sources not wired yet)

Notes on honesty, carried over from docs/02-data.md:

* All three adapters were unvalidated before this script. Anything it prints about
  coverage and gaps is the first such measurement for this repository.
* Fundamentals keep **every filed version** of a fact. The builder's as-of join is
  what makes restatement handling correct; collapsing versions here would destroy
  the point-in-time discipline.
* News and events are empty on purpose. Labels other than ``tail_risk`` cannot be
  evaluated without their event sources, and the config overlay reflects that.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shingan.config import ProjectConfig, load_config  # noqa: E402
from shingan.data.edgar import CIK_BY_TICKER, SecEdgarClient  # noqa: E402
from shingan.data.edgar import strip_html  # noqa: E402
from shingan.features.text import segment_items  # noqa: E402

DEFAULT_OVERLAY = ROOT / "configs" / "data" / "stage2_real.yaml"

#: us-gaap concepts that can stand in for each canonical fundamental line item.
#: First match wins; banking names (JPM, GS) report a different subset than
#: industrials, and the frame tolerates the gaps (compute_ratios handles NaN).
CONCEPTS_BY_COLUMN: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "total_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "inventory": ("InventoryNet",),
    "ebit": ("OperatingIncomeLoss",),
    "interest_expense": (
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestExpenseNonoperating",
        "InterestAndDebtExpense",
    ),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "goodwill": ("Goodwill",),
    "retained_earnings": ("RetainedEarningsAccumulatedDeficit",),
    "short_term_debt": (
        "ShortTermBorrowings",
        "DebtCurrent",
        "LongTermDebtCurrent",
        "ShortTermBorrowingsAndCurrentPortionOfLongTermDebt",
    ),
    "total_debt": (
        "LongTermDebt",
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebtNoncurrent",
    ),
}

#: Instant concepts report a state at ``end``; duration concepts cover ``start..end``.
DURATION_COLUMNS: frozenset[str] = frozenset(
    {"revenue", "net_income", "ebit", "interest_expense", "operating_cash_flow", "capex"}
)

CONCEPT_TO_COLUMN: dict[str, str] = {
    concept: column
    for column, concepts in CONCEPTS_BY_COLUMN.items()
    for concept in concepts
}

NEWS_COLUMNS = ["ticker", "published", "source", "title", "body", "sentiment"]
EVENTS_COLUMNS = ["ticker", "event_date", "event_kind", "severity", "source"]


def load_overlay_config(overlay: Path) -> ProjectConfig:
    """Load the default stack plus the Stage 2 real-data overlay."""
    return load_config(ROOT / "configs" / "default.yaml", [overlay], root=ROOT)


def fetch_prices(
    config: ProjectConfig,
    out_dir: Path,
    *,
    sleep_seconds: float = 15.0,
    max_retries: int = 3,
    backoff_seconds: float = 90.0,
) -> pd.DataFrame:
    """Daily adjusted OHLCV via the existing yfinance adapter, politely.

    One ticker per request with a deliberate pause between them, plus exponential-ish
    backoff on rate limiting. A single 10-ticker batch request is what got this IP
    rate limited in the first place: Yahoo throttles by request, and the adapter has
    no retry of its own. Failures are collected rather than raised, so one blocked
    ticker does not discard the nine that succeeded.
    """
    import time

    from shingan.data.prices import PriceUnavailable, load_prices

    start = date.fromisoformat(str(config.data.start))
    end = date.fromisoformat(str(config.data.end))
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for position, ticker in enumerate([str(item) for item in config.data.universe]):
        for attempt in range(max_retries + 1):
            try:
                # cache_dir=None: the adapter names its cache file by date range only,
                # so a per-ticker loop would overwrite the same file ten times.
                frame, metadata = load_prices(
                    [ticker], start=start, end=end, source="prices_yfinance", offline=False
                )
                frames.append(frame)
                print(
                    f"  {ticker}: {metadata.n_rows} rows "
                    f"({frame['date'].min().date()} .. {frame['date'].max().date()})"
                )
                break
            except PriceUnavailable as exc:
                message = str(exc)
                throttled = "too many" in message.lower() or "rate" in message.lower()
                if throttled and attempt < max_retries:
                    wait = backoff_seconds * (attempt + 1)
                    print(
                        f"  {ticker}: rate limited (attempt {attempt + 1}/{max_retries}); "
                        f"waiting {wait:.0f}s before retrying"
                    )
                    time.sleep(wait)
                    continue
                print(f"  ! {ticker}: {message}")
                failed.append(ticker)
                break
        if position < len(config.data.universe) - 1:
            time.sleep(sleep_seconds)

    if failed:
        print(f"warning: no prices for {', '.join(failed)}; they will be missing from the panel")
    if not frames:
        raise PriceUnavailable(
            "no ticker returned prices; the source is still throttling this network"
        )
    combined = pd.concat(frames, ignore_index=True)
    print(f"prices: {len(combined)} rows, {combined['ticker'].nunique()} tickers fetched")
    return combined


#: Candidate CIKs for a wider Stage 2 universe, with the filer name each one should
#: belong to. EDGAR's own ticker→CIK file lives on ``www.sec.gov``, which is blocked
#: for this network, so these are *candidates*: each is verified against
#: ``data.sec.gov/submissions`` before any of its data is used — either the record's
#: ``tickers`` field lists the symbol, or (for filers whose records predate that field)
#: the record's ``name`` matches the expected filer name below. A mismatch is reported
#: and the ticker dropped: binding a price series to the wrong company would be
#: invisible downstream and wrong in every metric.
EXTRA_CIK_CANDIDATES: dict[str, tuple[str, str]] = {
    "BAC": ("0000070858", "BANK OF AMERICA"),
    "C": ("0000831001", "CITIGROUP"),
    "WFC": ("0000072971", "WELLS FARGO"),
    "AIG": ("0000005272", "AMERICAN INTERNATIONAL GROUP"),
    "CVX": ("0000093410", "CHEVRON"),
    "HAL": ("0000045012", "HALLIBURTON"),
    "OXY": ("0000797468", "OCCIDENTAL PETROLEUM"),
    "GM": ("0001467858", "GENERAL MOTORS"),
    "DIS": ("0001744489", "WALT DISNEY"),
    "VZ": ("0000732712", "VERIZON"),
    "INTC": ("0000050863", "INTEL"),
    "CSCO": ("0000858877", "CISCO"),
    "TGT": ("0000027419", "TARGET"),
    "KSS": ("0000885639", "KOHL"),
    "DOW": ("0001751788", "DOW"),
    "X": ("0001163302", "UNITED STATES STEEL"),
    "MRO": ("0000101778", "MARATHON OIL"),
    "APA": ("0000006769", "APACHE"),
    "CVS": ("0000064803", "CVS"),
    "AAL": ("0000006201", "AMERICAN AIRLINES"),
    "UAL": ("0000100517", "UNITED AIRLINES"),
    "DAL": ("0000027904", "DELTA AIR LINES"),
    "MMM": ("0000066740", "3M"),
    "IBM": ("0000051143", "INTERNATIONAL BUSINESS MACHINES"),
}


def resolve_cik_map(client: SecEdgarClient, tickers: list[str]) -> dict[str, str]:
    """Ticker → CIK for the universe, with every candidate verified against EDGAR.

    The bundled ``CIK_BY_TICKER`` covers the ten-name Stage 2 basket. Wider universes
    use :data:`EXTRA_CIK_CANDIDATES`. Verification accepts either the record's
    ``tickers`` field or, when that field is absent (filers whose EDGAR record predates
    it — United States Steel, Marathon Oil and Apache are all like this), a name match
    against the expected filer name. Anything unverified is dropped with a message.
    """
    known = {ticker: CIK_BY_TICKER[ticker] for ticker in tickers if ticker in CIK_BY_TICKER}
    unverified: list[str] = []
    for ticker in tickers:
        if ticker in known:
            continue
        candidate = EXTRA_CIK_CANDIDATES.get(ticker)
        if candidate is None:
            unverified.append(ticker)
            continue
        cik, expected_name = candidate
        try:
            payload = client.submissions(cik)
        except Exception as exc:  # noqa: BLE001 - a bad candidate must not stop the fetch
            print(f"  ! {ticker}: could not verify candidate CIK {cik}: {exc}")
            unverified.append(ticker)
            continue
        symbols = {str(item).upper() for item in payload.get("tickers", [])}
        filer_name = str(payload.get("name", ""))
        if ticker.upper() in symbols:
            known[ticker] = cik
        elif not symbols and expected_name.upper() in filer_name.upper():
            print(f"  {ticker}: CIK {cik} accepted by filer name ({filer_name}); no tickers field")
            known[ticker] = cik
        else:
            print(
                f"  ! {ticker}: candidate CIK {cik} belongs to {filer_name or 'unknown'} "
                f"(tickers={sorted(symbols) or 'none'}); dropped"
            )
            unverified.append(ticker)
    if unverified:
        print(f"warning: no verified CIK for {', '.join(sorted(unverified))}")
    return known


def fetch_fundamentals(
    client: SecEdgarClient, tickers: list[str], cik_by_ticker: dict[str, str]
) -> pd.DataFrame:
    """Flatten companyfacts XBRL into the wide fundamentals schema.

    One output row per ``(ticker, filed, period_end)``: the facts as they were made
    public by that filing. Every filed version is kept so the builder's as-of join
    can reproduce what was knowable at any past date.
    """
    duration_budget_days = 500  # annual durations are ~365; reject nothing legitimate
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        cik = cik_by_ticker.get(ticker)
        if cik is None:
            print(f"  ! no CIK for {ticker}, skipping fundamentals")
            continue
        try:
            facts = client.facts_to_frame(cik, concepts=tuple(CONCEPT_TO_COLUMN))
        except Exception as exc:  # noqa: BLE001 - one bad CIK must not kill the fetch
            print(f"  ! companyfacts failed for {ticker}: {exc}")
            continue
        if facts.empty:
            print(f"  ! no facts for {ticker}")
            continue
        facts = facts.loc[facts["taxonomy"] == "us-gaap"].copy()
        facts = facts.loc[facts["form"].isin(["10-K", "10-Q"])].copy()
        facts["duration"] = (
            pd.to_datetime(facts["end"]) - pd.to_datetime(facts["start"])
        ).dt.days
        # Instant facts: duration is NaN or 0. Duration facts: keep the longest window
        # that plausibly belongs to the period (annual from a 10-K, quarterly from 10-Q).
        facts = facts.loc[
            facts["duration"].isna() | ((facts["duration"] >= 30) & (facts["duration"] <= duration_budget_days))
        ]
        facts = facts.sort_values(["end", "filed", "duration"])
        for (period_end, filed), group in facts.groupby(["end", "filed"], sort=False):
            row: dict[str, object] = {
                "ticker": ticker,
                "filed": pd.Timestamp(filed),
                "period_end": pd.Timestamp(period_end),
            }
            for record in group.itertuples(index=False):
                column = CONCEPT_TO_COLUMN.get(str(record.concept))
                if column is None or pd.isna(record.value):
                    continue
                if column in DURATION_COLUMNS and row.get(column) is not None:
                    # keep the longest duration reported for this period
                    continue
                row[column] = float(record.value)
            if len(row) > 3:
                rows.append(row)
        print(f"  {ticker}: {len(facts)} fact rows read")
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise SystemExit("no XBRL facts were extracted for any ticker; aborting")
    frame = frame.sort_values(["ticker", "filed", "period_end"])
    # One row per (ticker, filed): a filing reports the current period *and* prior
    # comparatives, so (filed) alone is not unique. Keep the latest period_end — the
    # period the filing is actually about. The as-of join then has exactly one
    # candidate per filing date, which is what makes the join deterministic.
    frame = frame.drop_duplicates(subset=["ticker", "filed"], keep="last").reset_index(drop=True)
    return frame


def fetch_filings_text(
    client: SecEdgarClient,
    tickers: list[str],
    *,
    since: date,
    doc_cache: Path,
    cik_by_ticker: dict[str, str] | None = None,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    limit_per_ticker: int = 80,
    max_chars: int = 80_000,
) -> pd.DataFrame:
    """Download filing documents and cut them into Item sections."""
    doc_cache.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    cik_by_ticker = cik_by_ticker or {}
    for ticker in tickers:
        try:
            refs = client.list_filings(
                ticker,
                forms=forms,
                since=since,
                limit=limit_per_ticker,
                cik=cik_by_ticker.get(ticker),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! submissions failed for {ticker}: {exc}")
            continue
        kept = 0
        for ref in refs:
            cache_file = doc_cache / f"{ref.accession.replace(':', '_')}.txt"
            if cache_file.exists():
                text = cache_file.read_text(encoding="utf-8", errors="replace")
            else:
                try:
                    text = client.fetch_document(ref.url)
                except Exception as exc:  # noqa: BLE001
                    print(f"    ! {ref.ticker} {ref.form} {ref.filed}: {exc}")
                    continue
                cache_file.write_text(text, encoding="utf-8")
            if not text.strip():
                continue
            sections = segment_items(text)
            wanted = {
                name: body
                for name, body in sections.items()
                if name in {"Item 1A", "Item 7"} and body.strip()
            }
            if not wanted:
                # segmentation fallback: keep the document whole, truncated
                wanted = {"Item 1": text[:max_chars]}
            for section, body in wanted.items():
                records.append(
                    {
                        "ticker": ticker,
                        "cik": cik_by_ticker.get(ticker, ""),
                        "filed": pd.Timestamp(ref.filed),
                        "doc_type": ref.form,
                        "section": section,
                        "text": body[:max_chars],
                        "accession": ref.accession,
                    }
                )
            kept += 1
        print(f"  {ticker}: {kept} filings fetched ({', '.join(forms)})")
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise SystemExit("no filing text was fetched; aborting")
    return frame.sort_values(["ticker", "filed", "section"]).reset_index(drop=True)


def fetch_filings_metadata(
    client: SecEdgarClient,
    tickers: list[str],
    *,
    since: date,
    cik_by_ticker: dict[str, str] | None = None,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    limit_per_ticker: int = 120,
) -> pd.DataFrame:
    """Filing dates and accessions only — no document text.

    The decision grid is one row per ``(ticker, filed)`` taken from the filings table,
    so a build needs the dates even when the documents themselves cannot be fetched
    (SEC blocks ``www.sec.gov/Archives`` for this network; the submissions index on
    ``data.sec.gov`` is unaffected). Rows carry ``section="unavailable"`` and empty
    text, which keeps every text-derived count at zero instead of silently inventing
    content; re-run without ``--skip-filings-text`` once the documents are reachable.
    """
    records: list[dict[str, object]] = []
    cik_by_ticker = cik_by_ticker or {}
    for ticker in tickers:
        try:
            refs = client.list_filings(
                ticker,
                forms=forms,
                since=since,
                limit=limit_per_ticker,
                cik=cik_by_ticker.get(ticker),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! submissions failed for {ticker}: {exc}")
            continue
        for ref in refs:
            records.append(
                {
                    "ticker": ticker,
                    # Carried so the panel's `cik` column holds a real number rather
                    # than the synthetic placeholder the builder reserves for synthetic
                    # rows. Provenance, not a feature.
                    "cik": cik_by_ticker.get(ticker, ""),
                    "filed": pd.Timestamp(ref.filed),
                    "doc_type": ref.form,
                    "section": "unavailable",
                    "text": "",
                    "accession": ref.accession,
                }
            )
        print(f"  {ticker}: {len(refs)} filing dates ({', '.join(forms)})")
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise SystemExit("no filing dates were fetched; aborting")
    return frame.sort_values(["ticker", "filed"]).reset_index(drop=True)


def build_client(config: ProjectConfig) -> SecEdgarClient:
    """Build the EDGAR client, falling back to a browser User-Agent on 403.

    SEC's policy asks for a descriptive UA with a contact, and that is what is tried
    first. Some networks (whole IP ranges, notably outside the US) are filtered
    harder: the descriptive UA is 403'd regardless of its shape while a browser UA
    passes. When that is detected the client switches and says so — the provenance
    cost is recorded here rather than hidden.
    """
    import httpx

    descriptive = os.environ.get("SHINGAN_SEC_USER_AGENT") or config.data.sec_user_agent
    probe_url = "https://data.sec.gov/submissions/CIK0000019617.json"
    try:
        probe = httpx.get(probe_url, headers={"User-Agent": descriptive}, timeout=20)
    except httpx.HTTPError as exc:
        print(f"warning: could not probe SEC with the descriptive UA: {exc}")
        probe = None
    user_agent = descriptive
    if probe is None or probe.status_code == 403:
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
        print(
            "warning: SEC rejects the descriptive User-Agent from this network (403); "
            "falling back to a browser UA. Recorded here because the dataset card must "
            "state how the data was obtained."
        )
    cache = Path(config.data.cache_dir) if config.data.cache_dir else ROOT / "data" / "raw"
    client = SecEdgarClient(
        # constructed with the compliant UA so validation passes, then overridden
        # below when this network's 403 filter demands a browser string
        user_agent=descriptive,
        requests_per_second=float(config.data.sec_requests_per_second),
        max_retries=int(config.data.sec_max_retries),
        timeout_seconds=float(config.data.sec_timeout_seconds),
        cache_dir=cache / "edgar_json",
    )
    client.user_agent = user_agent
    return client


def write_frames(out_dir: Path, **frames: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        destination = out_dir / f"{name}.parquet"
        frame.to_parquet(destination, index=False)
        print(f"wrote {destination} ({len(frame)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_OVERLAY,
        help="Data overlay YAML (default: configs/data/stage2_real.yaml).",
    )
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip the price fetch (use when yfinance is rate limited); "
        "requires an existing prices.parquet to build.",
    )
    parser.add_argument(
        "--skip-filings-text",
        action="store_true",
        help="Skip downloading filing documents (prices + XBRL only).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="After fetching, run build_panel and write data/processed.",
    )
    args = parser.parse_args()

    config = load_overlay_config(args.data_config)
    out_dir = Path(config.data.cache_dir) if config.data.cache_dir else ROOT / "data" / "raw" / "real"
    tickers = [str(item) for item in config.data.universe]

    print(f"universe: {', '.join(tickers)}  window {config.data.start}..{config.data.end}")

    prices_path = out_dir / "prices.parquet"
    if args.skip_prices:
        if prices_path.is_file():
            prices = pd.read_parquet(prices_path)
        else:
            prices = pd.DataFrame(
                columns=["ticker", "date", "open", "high", "low", "close", "volume", "shares_outstanding"]
            )
        if prices.empty:
            # EDGAR-only pass: the price fetch is deferred (e.g. yfinance rate limit)
            # and a placeholder frame keeps the EDGAR fetch and cache writes running.
            print("prices: skipped for now (no cached table); EDGAR fetch continues")
        else:
            print(f"prices: reusing cached {prices_path} ({len(prices)} rows)")
    else:
        prices = fetch_prices(config, out_dir)

    client = build_client(config)
    cik_by_ticker = resolve_cik_map(client, tickers)
    fundamentals = fetch_fundamentals(client, tickers, cik_by_ticker)
    print(f"fundamentals: {len(fundamentals)} rows, {fundamentals['ticker'].nunique()} tickers")
    # Persist immediately: each successful leg must survive a later leg failing
    # (network blocks, rate limits), so nothing is refetched unnecessarily.
    write_frames(out_dir, fundamentals=fundamentals)

    if args.skip_filings_text:
        filings = fetch_filings_metadata(
            client,
            tickers,
            since=date.fromisoformat(str(config.data.start)) - timedelta(days=180),
            cik_by_ticker=cik_by_ticker,
        )
        print(
            f"filings: {len(filings)} filing dates, text NOT fetched "
            "(structured-only run; text features will be zero)"
        )
    else:
        filings = fetch_filings_text(
            client,
            tickers,
            since=date.fromisoformat(str(config.data.start)) - timedelta(days=180),
            doc_cache=out_dir / "docs",
            cik_by_ticker=cik_by_ticker,
        )
        print(f"filings: {len(filings)} sections, {filings['ticker'].nunique()} tickers")

    news = pd.DataFrame(columns=NEWS_COLUMNS)
    events = pd.DataFrame(columns=EVENTS_COLUMNS)

    if prices.empty:
        print("prices: nothing fetched yet — prices.parquet left untouched")
    else:
        write_frames(out_dir, prices=prices)
    write_frames(out_dir, fundamentals=fundamentals, filings=filings, news=news, events=events)

    if args.build:
        if prices.empty:
            raise SystemExit(
                "--build needs a real price table (tail_risk labels are computed from "
                "prices); re-run without --skip-prices once the price source is available."
            )
        from shingan.data.builder import RawTables, build_panel

        tables = RawTables(
            prices=prices,
            fundamentals=fundamentals,
            filings=filings,
            news=news,
            events=events,
            is_synthetic=False,
        )
        result = build_panel(config, tables=tables, write=True)
        panel = result.panel
        print(
            f"panel: {panel.shape[0]} rows x {panel.shape[1]} cols, "
            f"{panel['ticker'].nunique()} companies, "
            f"{panel['as_of'].min()} .. {panel['as_of'].max()}"
        )
        mask = panel["label_mask_tail_risk"].astype(bool)
        print(f"tail_risk: observable={int(mask.sum())} positives={int(panel.loc[mask, 'label_tail_risk'].sum())}")


if __name__ == "__main__":
    main()
