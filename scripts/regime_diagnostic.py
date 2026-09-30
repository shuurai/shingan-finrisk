"""Who is the `tail_risk` label actually measuring: the company, or the market?

The wide evaluation's headline is 0.5594 AUC, and the number is dragged down by 2020 --
where 68% of the test positives sit and the model scores 0.5088, no better than a coin.
That invites a specific question before any modelling decision is made on top of it.

`tail_risk` is defined on a company's OWN price: a -30% peak-to-trough decline inside 30
trading days. It is an absolute threshold, not one relative to the market, so any period
where the index itself falls hard mints positives for a large part of the universe at
once. If March 2020 is such a period, then a good share of the test block's positives are
market events, and a model that cannot see the market is being asked to detect a
condition it has no input for.

That is a measurable claim, so it is measured rather than asserted, and the measurement
is deliberately an ORACLE: the benchmark's own realised forward drawdown. No model can
use it (it is the future), so it is an upper bound on how much of the label market
movement explains at all. It is printed next to the point-in-time market features the
model actually receives, because the gap between the two is the interesting part: a high
oracle with a low point-in-time AUC would say the information is in the market but not in
a form this feature set exposes.

Usage:

    python scripts/regime_diagnostic.py \\
        --panel data/processed/panel.parquet \\
        --prices data/raw/real/prices.parquet \\
        --benchmark '^GSPC'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shingan.eval.metrics import roc_auc
from shingan.labeling.builders import forward_max_drawdown

#: Same tolerance the panel builder uses when matching a decision date to a price row.
#: A filing date can fall on an exchange holiday, and an exact match would silently drop
#: those rows.
PRICE_MATCH_TOLERANCE_DAYS = 7

#: A market "correction", used only to describe how stressed the tape was. It is not the
#: label's threshold: the index is far less volatile than a single stock, so applying
#: -30% to it would classify almost nothing and empty the comparison.
CORRECTION = -0.10

LABEL_COLUMN = "label_tail_risk"
MASK_COLUMN = "label_mask_tail_risk"


def benchmark_forward_drawdown(
    prices: pd.DataFrame,
    symbol: str,
    *,
    horizon: int = 30,
    date_column: str = "date",
    ticker_column: str = "ticker",
    close_column: str = "close",
) -> pd.DataFrame:
    """Forward drawdown of one instrument over ``horizon`` steps, indexed by date.

    Reuses the label's own ``forward_max_drawdown`` rather than reimplementing it, so the
    oracle and the label cannot drift apart: if the definition changes, both change.
    """
    frame = prices.loc[prices[ticker_column].astype(str) == symbol, [date_column, close_column]]
    if frame.empty:
        raise ValueError(
            f"no rows for {symbol!r} in the price table; fetch it first "
            f"(scripts/fetch_real.py --market-symbols-only)"
        )
    frame = frame.sort_values(date_column).reset_index(drop=True)
    return pd.DataFrame(
        {
            # Cast to ns explicitly: parquet round-trips through pyarrow can come back as
            # datetime64[ms], and merge_asof refuses to join two resolutions even though
            # the instants are identical.
            "date": pd.to_datetime(frame[date_column]).astype("datetime64[ns]").to_numpy(),
            "bm_fwd_drawdown": forward_max_drawdown(
                frame[close_column].astype(float), horizon
            ).to_numpy(),
        }
    )


def attach_benchmark(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    tolerance_days: int = PRICE_MATCH_TOLERANCE_DAYS,
    date_column: str = "as_of",
) -> pd.DataFrame:
    """Backward as-of join of the benchmark onto the panel's decision dates."""
    left = panel.copy()
    left[date_column] = pd.to_datetime(left[date_column]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        left.sort_values(date_column),
        benchmark.sort_values("date"),
        left_on=date_column,
        right_on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    return merged.drop(columns=["date"])


def _observable(frame: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    block = frame.loc[frame[MASK_COLUMN] == 1]
    if split is not None:
        block = block.loc[block["split"] == split]
    return block


def describe_block(block: pd.DataFrame) -> dict[str, float]:
    """Label and market statistics for one block of observable rows.

    Every share is reported against the positives, because the question is what the
    positives are: a share of all rows would be dominated by the ~97% that are negative.
    """
    if block.empty:
        return {"rows": 0, "positives": 0}
    label = block[LABEL_COLUMN].to_numpy(dtype=int)
    drawdown = block["bm_fwd_drawdown"].to_numpy(dtype=float)
    positives = label == 1
    known = np.isfinite(drawdown)
    stats: dict[str, float] = {
        "rows": int(len(block)),
        "positives": int(positives.sum()),
        "pos_with_drawdown_known": int((positives & known).sum()),
    }
    if not positives.any() or not known.any():
        return stats
    pos_dd = drawdown[positives & known]
    neg_dd = drawdown[(~positives) & known]
    stats["benchmark_breached_label_threshold"] = float((pos_dd <= -0.30).mean())
    stats["benchmark_in_correction"] = float((pos_dd <= CORRECTION).mean())
    stats["median_bm_drawdown_positives"] = float(np.median(pos_dd))
    stats["median_bm_drawdown_negatives"] = float(np.median(neg_dd)) if len(neg_dd) else float("nan")
    return stats


def oracle_auc(block: pd.DataFrame) -> float:
    """AUC of the benchmark's realised forward drawdown as a risk score.

    The score is negated so that a deeper market fall ranks higher, matching the label's
    direction. This is an oracle: it is the outcome, not a prediction.
    """
    usable = block.loc[np.isfinite(block["bm_fwd_drawdown"])]
    if usable.empty or usable[LABEL_COLUMN].nunique() < 2:
        return float("nan")
    return roc_auc(usable[LABEL_COLUMN].to_numpy(), -usable["bm_fwd_drawdown"].to_numpy())


def point_in_time_aucs(block: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Single-feature AUCs for the market features the model is actually handed."""
    out: dict[str, float] = {}
    for column in columns:
        if column not in block.columns:
            out[column] = float("nan")
            continue
        usable = block.loc[np.isfinite(block[column].astype(float))]
        if usable.empty or usable[LABEL_COLUMN].nunique() < 2:
            out[column] = float("nan")
            continue
        out[column] = roc_auc(usable[LABEL_COLUMN].to_numpy(), usable[column].astype(float).to_numpy())
    return out


def year_of(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["as_of"]).dt.year


def format_report(frame: pd.DataFrame, benchmark_symbol: str, horizon: int) -> str:
    """Render the whole diagnostic. Kept separate from the computation so it is testable."""
    lines: list[str] = []
    lines.append(f"regime diagnostic -- is {LABEL_COLUMN} a company event or a market event?")
    lines.append(f"benchmark {benchmark_symbol}, forward drawdown horizon {horizon} trading days")
    lines.append("oracle = benchmark realised forward drawdown (the FUTURE; no model can use it)")
    lines.append("")

    for split in ("train", "valid", "test"):
        block = _observable(frame, split)
        if block.empty:
            continue
        stats = describe_block(block)
        lines.append(f"{split}")
        lines.append(f"  rows {stats['rows']:,}   positives {stats['positives']:,}")
        if not stats["positives"]:
            lines.append("")
            continue
        lines.append(
            f"  benchmark also fell >=30% in the next {horizon} days: "
            f"{stats['benchmark_breached_label_threshold']:.1%} of positives"
        )
        lines.append(
            f"  benchmark in correction (>=10% fall):               "
            f"{stats['benchmark_in_correction']:.1%} of positives"
        )
        lines.append(
            f"  median benchmark drawdown: positives "
            f"{stats['median_bm_drawdown_positives']:+.2%} vs negatives "
            f"{stats['median_bm_drawdown_negatives']:+.2%}"
        )
        aucs = point_in_time_aucs(block, ["vix_level", "vix_chg_5d", "beta_252d", "vol_60d"])
        lines.append(f"  ORACLE   benchmark drawdown alone      AUC {oracle_auc(block):.4f}")
        for column, value in aucs.items():
            lines.append(f"  point-in-time {column:22s} AUC {value:.4f}")
        lines.append("")

    for split in ("train", "valid", "test"):
        block = _observable(frame, split)
        if block.empty or not block[LABEL_COLUMN].sum():
            continue
        block = block.assign(year=year_of(block))
        lines.append(f"{split} block by year -- the label's regime concentration, and what the market did")
        lines.append(
            f"  {'year':>4}  {'rows':>6}  {'pos':>5}  {'bm>=30%':>8}  {'bm>=10%':>8}  "
            f"{'oracle':>8}  {'vol_60d':>8}"
        )
        for year, year_block in block.groupby("year", sort=True):
            stats = describe_block(year_block)
            pos = stats["positives"]
            breach = f"{stats['benchmark_breached_label_threshold']:.1%}" if pos else "n/a"
            corr = f"{stats['benchmark_in_correction']:.1%}" if pos else "n/a"
            pia = point_in_time_aucs(year_block, ["vol_60d"])["vol_60d"]
            lines.append(
                f"  {int(year):>4}  {stats['rows']:>6,}  {pos:>5,}  {breach:>8}  {corr:>8}  "
                f"{oracle_auc(year_block):>8.4f}  {pia:>8.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/panel.parquet"))
    parser.add_argument("--prices", type=Path, default=Path("data/raw/real/prices.parquet"))
    parser.add_argument("--benchmark", default="^GSPC", help="index symbol already present in --prices")
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="trading days; must match labels.tail_risk_horizon_trading_days or the oracle "
        "describes a different window than the label",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    args = parser.parse_args(argv)

    panel = pd.read_parquet(args.panel)
    prices = pd.read_parquet(args.prices, columns=["ticker", "date", "close"])
    benchmark = benchmark_forward_drawdown(prices, args.benchmark, horizon=args.horizon)
    frame = attach_benchmark(panel, benchmark)

    report = format_report(frame, args.benchmark, args.horizon)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
