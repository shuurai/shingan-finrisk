#!/usr/bin/env python3
"""Ingest the FNSPID news snapshot into the project's real news table.

The FNSPID snapshot (HF ``Zihan1004/FNSPID``, CC BY-NC-4.0) ships two news CSVs
totalling ~29 GB. This project needs news for one configured universe over one
decision window, so the ingest streams each CSV in chunks, keeps rows for the
universe inside the window, maps columns onto the ``NEWS_COLUMNS`` contract,
de-duplicates syndicated copies and writes a single parquet under
``data/raw/real/`` — the location ``load_cached_tables`` already reads
optionally, so ``shingan data build`` picks the result up with no further wiring.

Column mapping, verified against the remote files' first bytes on 2026-09-26:

    Date          -> published   (values carry a trailing ' UTC')
    Article_title -> title
    Stock_symbol  -> ticker
    Article       -> body        (nasdaq file only; All_external is headline-only)
    Publisher     -> source

``sentiment`` is absent from both files' headers; the contract column stays
NaN and downstream features must treat it as missing, not as zero.

License: CC BY-NC-4.0. Local research use is fine; the ingested text must not
be redistributed (the dataset card already promises exactly that).

Usage:
    python scripts/ingest_fnspid.py                 # full run
    python scripts/ingest_fnspid.py --limit-chunks 3   # smoke run on the first chunks
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: Columns we read from either FNSPID CSV. Both files carry these names; the
#: nasdaq file additionally has an index column (``Unnamed: 0``) that is ignored
#: by selecting explicitly.
SOURCE_COLUMNS = ("Date", "Article_title", "Stock_symbol", "Publisher", "Article")

#: The contract ``scripts/fetch_real.py`` and ``load_cached_tables`` expect.
OUTPUT_COLUMNS = ["ticker", "published", "source", "title", "body", "sentiment"]

#: News informs decisions up to NEWS_CONTEXT_WINDOW_DAYS back, so the ingest
#: window opens this many days before the panel's start.
LEAD_IN_DAYS = 180


def parse_fnspid_date(series: pd.Series) -> pd.Series:
    """Parse FNSPID timestamps (``2023-12-16 23:00:00 UTC``) to naive datetimes.

    The trailing `` UTC`` literal defeats the fast parser path, so it is stripped
    before parsing. Unparseable values become NaT and are dropped later.
    """
    cleaned = series.astype("string").str.replace(r"\s+UTC$", "", regex=True)
    return pd.to_datetime(cleaned, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def map_fnspid_chunk(chunk: pd.DataFrame, universe: frozenset[str]) -> pd.DataFrame:
    """Map one raw chunk onto the contract columns, filtered to the universe.

    Raises:
        KeyError: If a required source column is missing from the chunk.
    """
    missing = [name for name in SOURCE_COLUMNS if name not in chunk.columns]
    if missing:
        raise KeyError(f"FNSPID chunk is missing columns {missing}; present: {sorted(chunk.columns)}")
    selected = chunk.loc[
        chunk["Stock_symbol"].astype("string").str.upper().str.strip().isin(universe),
        list(SOURCE_COLUMNS),
    ].copy()
    selected["ticker"] = selected["Stock_symbol"].str.upper().str.strip()
    selected["published"] = parse_fnspid_date(selected["Date"])
    selected["source"] = selected["Publisher"].astype("string")
    selected["title"] = selected["Article_title"].astype("string")
    selected["body"] = selected["Article"].astype("string")
    selected["sentiment"] = pd.NA
    return selected[OUTPUT_COLUMNS]


def load_universe_and_window(overlay: Path) -> tuple[frozenset[str], date, date]:
    """Read the universe and decision window from the real-data overlay."""
    from shingan.config import load_config

    cfg = load_config(ROOT / "configs" / "default.yaml", [overlay], root=ROOT)
    universe = frozenset(cfg.data.universe)
    start = date.fromisoformat(str(cfg.data.start)) - timedelta(days=LEAD_IN_DAYS)
    end = date.fromisoformat(str(cfg.data.end))
    return universe, start, end


def ingest(
    src: Path,
    out: Path,
    universe: frozenset[str],
    window_start: date,
    window_end: date,
    *,
    chunksize: int = 200_000,
    limit_chunks: int | None = None,
) -> pd.DataFrame:
    """Stream every CSV under ``src``, filter, de-duplicate and write ``out``."""
    from shingan.data.news import deduplicate_news

    csvs = sorted(src.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSV files under {src}; download the FNSPID snapshot first")

    kept_parts: list[pd.DataFrame] = []
    for source in csvs:
        header = pd.read_csv(source, nrows=0).columns.tolist()
        usecols = [name for name in SOURCE_COLUMNS if name in header]
        total = kept = 0
        for index, chunk in enumerate(pd.read_csv(source, usecols=usecols, chunksize=chunksize, dtype="string")):
            if limit_chunks is not None and index >= limit_chunks:
                break
            total += len(chunk)
            mapped = map_fnspid_chunk(chunk, universe)
            mapped = mapped.loc[mapped["published"].notna()]
            mapped = mapped.loc[
                (mapped["published"].dt.date >= window_start) & (mapped["published"].dt.date <= window_end)
            ]
            kept += len(mapped)
            if len(mapped):
                kept_parts.append(mapped)
        print(f"{source.name}: {kept:,} rows kept of {total:,} read (universe filter)")

    if not kept_parts:
        raise SystemExit("no rows survived the universe/window filter; nothing to write")

    combined = pd.concat(kept_parts, ignore_index=True)
    combined = combined.sort_values(["ticker", "published"]).reset_index(drop=True)
    deduplicated, log = deduplicate_news(combined)
    deduplicated = deduplicated[OUTPUT_COLUMNS]

    out.parent.mkdir(parents=True, exist_ok=True)
    deduplicated.to_parquet(out, index=False)
    if len(log):
        log_path = out.with_name("news_dedup_log.parquet")
        log.to_parquet(log_path, index=False)
        print(f"dedup log: {len(log):,} rows -> {log_path}")

    per_ticker = deduplicated.groupby("ticker", observed=True).size().sort_values(ascending=False)
    print(f"wrote {len(deduplicated):,} news rows to {out}")
    print(f"date range: {deduplicated['published'].min()} .. {deduplicated['published'].max()}")
    print(f"tickers covered: {deduplicated['ticker'].nunique()} of {len(universe)}")
    print("top 10:", ", ".join(f"{t}={n:,}" for t, n in per_ticker.head(10).items()))
    return deduplicated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "raw" / "fnspid" / "Stock_news")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "real" / "news.parquet")
    parser.add_argument("--overlay", type=Path, default=ROOT / "configs" / "data" / "stage2_real.yaml")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--limit-chunks", type=int, default=None, help="Read at most N chunks per file (smoke run).")
    args = parser.parse_args(argv)

    universe, window_start, window_end = load_universe_and_window(args.overlay)
    print(f"universe: {len(universe)} tickers; window: {window_start} .. {window_end}")
    ingest(
        args.src,
        args.out,
        universe,
        window_start,
        window_end,
        chunksize=args.chunksize,
        limit_chunks=args.limit_chunks,
    )


if __name__ == "__main__":
    sys.exit(main())
